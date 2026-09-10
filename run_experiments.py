import os
import itertools
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 匯入自定義模組
from dataset import (
    ICLMarkovChainDataset, 
    MarkovChainDataset, 
    HMMDataset, 
    ICLHMMDataset
)
from model import Transformer
from trainer import Trainer

def run_experiment():
    # ==========================================
    # 1. 解析命令列參數 (新增 GPU 動態分流邏輯)
    # ==========================================
    parser = argparse.ArgumentParser(description="Parallelize Grid Search across GPUs for ICL-MC/HMM")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU ID (0 for Linear, 1 for Performer)")
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5, 1e-4, 5e-4], 
                        help="Specific learning rate(s) to run (space-separated). Default includes 3e-5 (paper default) and 5e-4.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 根據 GPU ID 動態分派模型類型
    if args.gpu == 0:
        target_attn = ["linear"]
        print(f"🚀 GPU {args.gpu} 啟動！專屬任務：Linear Transformer。LRs: {args.lr}")
    elif args.gpu == 1:
        target_attn = ["performer"]
        print(f"🚀 GPU {args.gpu} 啟動！專屬任務：Performer。LRs: {args.lr}")
    else:
        target_attn = ["linear", "performer"]
        print(f"🚀 GPU {args.gpu} 啟動！混合任務：Linear & Performer。LRs: {args.lr}")

    if not os.path.exists("results"): 
        os.makedirs("results")
    
    summary_data = []
    
    # ==========================================
    # 2. 專屬 2-layer ICL-MC/HMM 網格搜尋配置
    # ==========================================
    grid = {
        # 1. Data underlying assumption
        "data_name": ["ICL-Markov", "ICL-HMM"],
        
        # 2. Data configuration
        "num_symbols": [2, 3],        
        "n_order": [1, 2],            
        
        # 鎖定 2-layer 結構
        "n_layer": [2],          
        
        # 3. Model: 由 GPU ID 動態決定
        "attn_type": target_attn, 
        
        # 4. Model architecture variants
        "use_residual": [True, False],    # 殘差連接開關
        "use_ln": [True, False],          # LayerNormalization 開關
        "attention_only": [True],         # 依據理論文獻，使用 attention-only 模型
        
        # 5. Positional Encoding
        "pe_type": ["none", "absolute", "rpe", "rope"], 
        
        # 維度與超參數對標文獻設定
        "embed_dim": [16],         
        "num_heads": [1, 2],         
        "lr": args.lr   
    }
    
    train_params = {
        "seq_len": 100,
        "epochs": 30,
        "batch_size": 64,
        "eval_interval": 20,
        "num_hidden": 2      # HMM 隱藏狀態數
    }

    # 自動展開所有實驗排列組合
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 矩陣修剪與算力防呆過濾
    valid_experiments = []
    for p in experiments:
        # Linear/Performer 無法相容 Additive RPE，直接跳過該組合省下算力
        if p["attn_type"] in ["linear", "performer"] and p["pe_type"] == "rpe":
            continue
        valid_experiments.append(p)

    print(f"過濾不相容 PE 後，GPU {args.gpu} 總計分配到 {len(valid_experiments)} 組實驗任務。\n")

    for p in valid_experiments:
        # 結構化識別字串 (加入 LN 標籤)
        dataset_str = f"{p['data_name']}_V{p['num_symbols']}_O{p['n_order']}"
        model_str = (f"L{p['n_layer']}_H{p['num_heads']}_D{p['embed_dim']}_"
                     f"{p['attn_type']}_AttnOnly{p['attention_only']}_"
                     f"PE-{p['pe_type']}_Res{p['use_residual']}_LN{p['use_ln']}_LR{p['lr']}")
        model_tag = f"{dataset_str}_{model_str}"
        
        print(f"\n[GPU {args.gpu}] >>> Config: {model_tag}")

        # ==========================================
        # 3. 動態資料集分流
        # ==========================================
        try:
            if p["data_name"] == "ICL-Markov":
                train_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=12800)
                test_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=4000)
                
            elif p["data_name"] == "ICL-HMM":
                train_ds = ICLHMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=12800)
                test_ds = ICLHMMDataset(train_params["seq_len"], num_hidden=train_params["num_hidden"], num_obs=p["num_symbols"], n_order=p["n_order"], virtual_size=4000)
        except Exception as e:
            print(f"❌ 資料集 {p['data_name']} 初始化失敗: {e}，跳過此組合。")
            continue

        # 建立 DataLoader
        test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)
        
        # 預先生成固定的測試集資料流 (鎖定評估基準)
        fixed_test_data = []
        for x, y, p_true, *info in test_loader:
            fixed_test_data.append((x.clone(), y.clone(), p_true.clone()))

        # ==========================================
        # 4. 模型初始化 (注入 use_ln 參數)
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
            use_residual=p["use_residual"],
            use_ln=p["use_ln"]  # <--- 傳遞 LayerNorm 開關
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
        
        # 儲存圖表
        trainer.save_plots(model_tag, f"results/{model_tag}")
        
        # 提取最終指標
        f_s_ce = trainer.history["test_sample_ce"][-1] if trainer.history["test_sample_ce"] else 0.0
        f_t_ce = trainer.history["test_theory_ce"][-1] if trainer.history["test_theory_ce"] else 0.0
        f_s_kl = trainer.history["test_sample_kl"][-1] if trainer.history["test_sample_kl"] else 0.0
        f_t_kl = trainer.history["test_theory_kl"][-1] if trainer.history["test_theory_kl"] else 0.0

        # 寫入記錄總表
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
            "Use_LN": p["use_ln"],  # <--- 記錄 LN 狀態
            "Layers": p["n_layer"],
            "Heads": p["num_heads"],
            "Embed_Dim": p["embed_dim"],
            "LR": p["lr"],                
            "Final_Sample_CE": f"{f_s_ce:.6f}",
            "Final_Theory_CE": f"{f_t_ce:.6f}",
            "Final_Sample_KL": f"{f_s_kl:.6f}",
            "Final_Theory_KL": f"{f_t_kl:.6f}"
        })

        # 分流寫入 GPU 專屬的 CSV 檔案
        csv_filename = f"results/summary_gpu{args.gpu}.csv"
        pd.DataFrame(summary_data).to_csv(csv_filename, index=False)

    print(f"\n[Finished] GPU {args.gpu} 所有矩陣實驗執行完畢！數據已同步至 {csv_filename}")

if __name__ == "__main__":
    run_experiment()
