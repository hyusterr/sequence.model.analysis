import os
import itertools
import argparse  # 用於解析命令行參數
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 匯入你的自定義模組
from dataset import (
    ICLMarkovChainDataset, 
    MarkovChainDataset, 
    HMMDataset, 
    ICLHMMDataset
    # GINCDataset,     # 🌟 依要求先註解掉未檢查的資料集
    # HMMLDADataset
)
from model import Transformer
from trainer import Trainer

def run_experiment():
    # ==========================================
    # 1. 解析命令列參數 (argparse)
    # ==========================================
    parser = argparse.ArgumentParser(description="Parallelize Grid Search across GPUs with Full Architecture Variants.")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU ID (e.g., 0, 1, or 2)")
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5, 1e-4, 5e-4], 
                        help="Specific learning rate(s) to run on this GPU (space-separated)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 GPU {args.gpu} 啟動！負責執行的 Learning Rates: {args.lr}")

    if not os.path.exists("results"): 
        os.makedirs("results")
    
    summary_data = []
    
    # ==========================================
    # 🌟 2. 全面擴充版網格搜尋配置 (Grid Search)
    # ==========================================
    grid = {
        # 資料集清單 (已暫時移除 GINC 與 HMMLDA)
        "data_name": ["ICL-Markov", "Markov", "HMM", "ICL-HMM"],
        "num_symbols": [3, 2],        
        "n_order": [1, 2],            
        "n_layer": [2, 1],         
        
        # 核心架構變體
        "attn_type": ["performer", "standard", "linear"], # 納入我們剛寫好的 performer!
        "attention_only": [True, False],                  # 支援標準與 attention-only 兩種版本
        "pe_type": ["absolute", "rpe", "rope"],           # 支援 APE, RPE, RoPE
        "use_residual": [True, False],                    # 殘差連接開關
        
        "embed_dim": [16],         
        "num_heads": [1, 2],          
        "lr": args.lr   
    }
    
    train_params = {
        "seq_len": 100,
        "epochs": 30,
        "batch_size": 64,
        "eval_interval": 20,
        "num_hidden": 2      # 用於 HMM 與 ICL-HMM 基準狀態數
    }

    # 自動展開所有實驗的排列組合
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 進行矩陣修剪，移除無效或重複的算力浪費組合
    valid_experiments = []
    for p in experiments:
        # 🌟 算力優化護欄：Linear/Performer 遇到 RPE 時會被 model.py 強制校正為 RoPE。
        # 為了避免與 explicit 跑 RoPE 的組合重複，這裡直接跳過，省下 1/3 的無謂算力！
        if p["attn_type"] in ["linear", "performer"] and p["pe_type"] == "rpe":
            continue
        valid_experiments.append(p)

    print(f"GPU {args.gpu} 經過優化過濾後，總共需要檢查 {len(valid_experiments)} 組組合實驗...\n")

    for p in valid_experiments:
        # 建立結構化識別字串，確保檔名完全不重疊
        dataset_str = f"{p['data_name']}_V{p['num_symbols']}_O{p['n_order']}"
        model_str = (f"L{p['n_layer']}_H{p['num_heads']}_D{p['embed_dim']}_"
                     f"{p['attn_type']}_AttnOnly{p['attention_only']}_"
                     f"PE-{p['pe_type']}_Res{p['use_residual']}_LR{p['lr']}")
        model_tag = f"{dataset_str}_{model_str}"
        
        print(f"\n[GPU {args.gpu}] >>> Config: {model_tag}")

        # ==========================================
        # 3. 動態資料集分流
        # ==========================================
        try:
            if p["data_name"] == "ICL-Markov":
                train_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=12800)
                test_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=4000)
                
            elif p["data_name"] == "Markov":
                train_ds = MarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=12800)
                test_ds = MarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=4000)
                
            elif p["data_name"] == "HMM":
                train_ds = HMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=12800)
                test_ds = HMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=4000)
                
            elif p["data_name"] == "ICL-HMM":
                train_ds = ICLHMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=12800)
                test_ds = ICLHMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=4000)
        except Exception as e:
            print(f"❌ 資料集 {p['data_name']} 初始化失敗: {e}，跳過此組合。")
            continue

        # 建立 DataLoader
        test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)
        
        # 預先生成固定的測試集資料流
        fixed_test_data = []
        for x, y, p_true, *info in test_loader:
            fixed_test_data.append((x.clone(), y.clone(), p_true.clone()))

        # ==========================================
        # 4. 初始化全新對標的模型 (包含新版 APE/RoPE/Residual/AttnOnly)
        # ==========================================
        model = Transformer(
            vocab_size=p["num_symbols"],
            d_model=p["embed_dim"],
            nhead=p["num_heads"],
            num_layers=p["n_layer"],
            block_size=train_params["seq_len"],
            pe_type=p["pe_type"],
            attn_type=p["attn_type"],
            attention_only=p["attention_only"],
            use_residual=p["use_residual"]
        )
        
        config_for_trainer = {
            "lr": p["lr"], 
            "epochs": train_params["epochs"],
            "batch_size": train_params["batch_size"]
        }
        trainer = Trainer(model, train_loader, device, config_for_trainer)
        
        # ==========================================
        # 5. 訓練與高頻評估
        # ==========================================
        epoch_pbar = tqdm(range(train_params["epochs"]), desc="    Epochs")
        for epoch in epoch_pbar:
            trainer.train_epoch(epoch + 1, model_tag, fixed_test_data, eval_interval=train_params["eval_interval"])
            if len(trainer.history["test_theory_kl"]) > 0:
                epoch_pbar.set_postfix(Theory_KL=f"{trainer.history['test_theory_kl'][-1]:.4f}")
        
        # 存圖 (會自動產生 Sample 面板與 Theoretical 面板對照圖)
        trainer.save_plots(model_tag, f"results/{model_tag}")
        
        # 提取最後指標
        f_s_ce = trainer.history["test_sample_ce"][-1] if trainer.history["test_sample_ce"] else 0.0
        f_t_ce = trainer.history["test_theory_ce"][-1] if trainer.history["test_theory_ce"] else 0.0
        f_s_kl = trainer.history["test_sample_kl"][-1] if trainer.history["test_sample_kl"] else 0.0
        f_t_kl = trainer.history["test_theory_kl"][-1] if trainer.history["test_theory_kl"] else 0.0

        # 寫入記錄總表 (擴充欄位以進行多維度消融實驗分析)
        summary_data.append({
            "Dataset_Setting": dataset_str,
            "Model_Setting": model_str,
            "Data_Name": p["data_name"],
            "Num_Symbols": p["num_symbols"],
            "N_Order": p["n_order"],
            "Model_Type": p["attn_type"],
            "Attention_Only": p["attention_only"],
            "PE_Type": p["pe_type"],
            "Use_Residual": p["use_residual"],
            "Layers": p["n_layer"],
            "Heads": p["num_heads"],
            "Embed_Dim": p["embed_dim"],
            "LR": p["lr"],               
            "Final_Sample_CE": f"{f_s_ce:.6f}",
            "Final_Theory_CE": f"{f_t_ce:.6f}",
            "Final_Sample_KL": f"{f_s_kl:.6f}",
            "Final_Theory_KL": f"{f_t_kl:.6f}"
        })

        # 分流寫入各自的 GPU CSV 檔案，避免 Race Condition
        csv_filename = f"results/summary_gpu{args.gpu}.csv"
        pd.DataFrame(summary_data).to_csv(csv_filename, index=False)

    print(f"\n[Finished] GPU {args.gpu} 所有矩陣實驗執行完畢！數據已同步至 {csv_filename}")

if __name__ == "__main__":
    run_experiment()
