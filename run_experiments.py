import os
import itertools
import argparse  # 用於解析命令行參數
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# 匯入你的自定義模組
from dataset import ICLMarkovChainDataset
from model import Transformer
from trainer import Trainer

def run_experiment():
    # ==========================================
    # 🌟 1. 解析命令列參數 (argparse)
    # ==========================================
    parser = argparse.ArgumentParser(description="Parallelize Grid Search across GPUs by LR.")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU ID (e.g., 0, 1, or 2)")
    parser.add_argument("--lr", type=float, nargs="+", default=[3e-5, 1e-4, 5e-4], 
                        help="Specific learning rate(s) to run on this GPU (space-separated)")
    args = parser.parse_args()

    # 設定當前進程使用的 GPU 卡號
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"🚀 GPU {args.gpu} 啟動！負責執行的 Learning Rates: {args.lr}")

    if not os.path.exists("results"): 
        os.makedirs("results")
    
    summary_data = []
    
    # ==========================================
    # 🌟 2. 網格搜尋配置 (動態讀取指定的 args.lr)
    # ==========================================
    grid = {
        "data_name": ["ICL-Markov"],
        "num_symbols": [3, 2],        
        "n_order": [1, 2],            
        "n_layer": [2, 1],         
        "attn": ["attention-only", "standard", "linear"], 
        "embed_dim": [16],         
        "num_heads": [1, 2],          
        "lr": args.lr   # 這張卡只跑外面傳進來的特定 LR
    }
    
    train_params = {
        "seq_len": 100,
        "epochs": 30,
        "batch_size": 64,
        "eval_interval": 20       
    }

    # 自動展開排列組合
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"GPU {args.gpu} 總共需要執行 {len(experiments)} 組組合實驗...\n")

    for p in experiments:
        dataset_str = f"{p['data_name']}_V{p['num_symbols']}_O{p['n_order']}"
        model_str = f"L{p['n_layer']}_H{p['num_heads']}_D{p['embed_dim']}_{p['attn']}_LR{p['lr']}" 
        model_tag = f"{dataset_str}_{model_str}"
        
        print(f"\n[GPU {args.gpu}] >>> Running: {model_tag}")

        # 生成測試集與數據流
        test_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=4000)
        test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
        
        print("    Pre-generating fixed Test Set...")
        fixed_test_data = []
        for x, y, p_true, info in test_loader:
            fixed_test_data.append((x.clone(), y.clone(), p_true.clone()))

        train_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=12800)
        train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)

        # 初始化模型
        model = Transformer(
            vocab_size=p["num_symbols"],
            d_model=p["embed_dim"],
            nhead=p["num_heads"],
            num_layers=p["n_layer"],
            block_size=train_params["seq_len"],
            pe_type='rpe',
            attn_type=("standard" if p["attn"] == "attention-only" else p["attn"]),
            attention_only=(p["attn"] == "attention-only")
        )
        
        # 配置 Trainer 參數
        config_for_trainer = {
            "lr": p["lr"], 
            "epochs": train_params["epochs"],
            "batch_size": train_params["batch_size"]
        }
        trainer = Trainer(model, train_loader, device, config_for_trainer)
        
        # 🌟 5. 訓練流程 (已修正為新指標 test_theory_kl)
        epoch_pbar = tqdm(range(train_params["epochs"]), desc="    Epochs")
        for epoch in epoch_pbar:
            trainer.train_epoch(epoch + 1, model_tag, fixed_test_data, eval_interval=train_params["eval_interval"])
            if len(trainer.history["test_theory_kl"]) > 0:
                epoch_pbar.set_postfix(Theory_KL=f"{trainer.history['test_theory_kl'][-1]:.4f}")
        
        # 存圖與記錄
        trainer.save_plots(model_tag, f"results/{model_tag}")
        
        # 🌟 6. 寫入記錄 (擴充為四項完整指標)
        f_s_ce = trainer.history["test_sample_ce"][-1] if trainer.history["test_sample_ce"] else 0.0
        f_t_ce = trainer.history["test_theory_ce"][-1] if trainer.history["test_theory_ce"] else 0.0
        f_s_kl = trainer.history["test_sample_kl"][-1] if trainer.history["test_sample_kl"] else 0.0
        f_t_kl = trainer.history["test_theory_kl"][-1] if trainer.history["test_theory_kl"] else 0.0

        summary_data.append({
            "Dataset_Setting": dataset_str,
            "Model_Setting": model_str,
            "Data_Name": p["data_name"],
            "Num_Symbols": p["num_symbols"],
            "N_Order": p["n_order"],
            "Model_Type": p["attn"],
            "Layers": p["n_layer"],
            "Heads": p["num_heads"],
            "Embed_Dim": p["embed_dim"],
            "LR": p["lr"],               
            "Final_Sample_CE": f"{f_s_ce:.6f}",
            "Final_Theory_CE": f"{f_t_ce:.6f}",
            "Final_Sample_KL": f"{f_s_kl:.6f}",
            "Final_Theory_KL": f"{f_t_kl:.6f}"
        })

        # 每個組合跑完即時複寫存檔，防止中斷損失數據
        csv_filename = f"results/summary_gpu{args.gpu}.csv"
        pd.DataFrame(summary_data).to_csv(csv_filename, index=False)

    print(f"\n[Finished] GPU {args.gpu} 所有指定實驗執行完畢！結果已儲存至 {csv_filename}")

if __name__ == "__main__":
    run_experiment()
