import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
import torch.nn.functional as F
from model import Transformer

# ==========================================
# 核心 Metrics 修正 (對標論文量級)
# ==========================================

def compute_cross_entropy(logits, targets):
    """計算平均每個 token 的 Cross Entropy (訓練用)"""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1).item()

def compute_kl_divergence(logits, target_probs):
    """計算模型預測與真實 P 之間的 KL Divergence (訓練用)"""
    log_pred_probs = F.log_softmax(logits, dim=-1)
    kl_sum = F.kl_div(log_pred_probs, target_probs, reduction='sum')
    batch_size, seq_len, _ = logits.size()
    return (kl_sum / (batch_size * seq_len)).item()

# ==========================================
# Trainer 類別 (包含 Test Last Token 修正)
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, test_loader, device, batch_size, lr=3e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.batch_size = batch_size
        # self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) # default AdamW with lr=3e-5 or 5e-4 not working for ICL-MC
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.1,       # 🌟 加大權重衰減 (從 0.01 改 0.1)
            betas=(0.9, 0.95)       # 🌟 採用 GPT-2 標準動量
        )
        
        self.history = {
            "examples_seen": [], 
            "step_ce": [],
            "step_kl": [],
            "test_ce": [],
            "test_kl": [],
            "test_at_examples": []
        }
        self.total_steps = 0
        self.examples_seen = 0 

    def train_epoch(self, epoch_idx, model_name):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch_idx}", leave=False)
        
        for x, y, p_true, info in pbar:
            x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
            info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in info.items()}
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0) 
            
            if self.total_steps % 20 == 0:
                with torch.no_grad():
                    # 訓練時依然看整體平均
                    kl = compute_kl_divergence(logits, p_true)
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["step_ce"].append(loss.item())
                    self.history["step_kl"].append(kl)
                    pbar.set_postfix(Examples=self.examples_seen, CE=f"{loss.item():.4f}")

    def evaluate(self):
        """對標 test_error.py：只針對 Last Token 進行評估"""
        self.model.eval()
        all_ce, all_kl = [], []
        
        with torch.no_grad():
            for x, y, p_true, info in self.test_loader:
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                # 🌟 關鍵修正：切片只取最後一個 Token
                last_logits = logits[:, -1, :] 
                last_y = y[:, -1]
                last_p_true = p_true[:, -1, :]
                
                # 計算 Last Token CE
                ce = F.cross_entropy(last_logits, last_y).item()
                
                # 計算 Last Token KL
                log_pred = F.log_softmax(last_logits, dim=-1)
                kl = F.kl_div(log_pred, last_p_true, reduction='batchmean').item()
                
                all_ce.append(ce)
                all_kl.append(kl)
        
        avg_ce = np.mean(all_ce)
        avg_kl = np.mean(all_kl)
        
        self.history["test_ce"].append(avg_ce)
        self.history["test_kl"].append(avg_kl)
        self.history["test_at_examples"].append(self.examples_seen)
        
        return avg_ce, avg_kl

    def save_plots(self, title, filename):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # 繪圖 (左軸: Cross Entropy)
        ax1.plot(self.history["examples_seen"], self.history["step_ce"], alpha=0.2, color='blue')
        ax1.plot(self.history["test_at_examples"], self.history["test_ce"], marker='o', label='Test CE', color='blue')
        ax1.set_xlabel('Number of Examples')
        ax1.set_ylabel('Cross Entropy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 繪圖 (右軸: KL Divergence)
        ax2 = ax1.twinx()
        ax2.plot(self.history["examples_seen"], self.history["step_kl"], alpha=0.2, color='red')
        ax2.plot(self.history["test_at_examples"], self.history["test_kl"], marker='x', label='Test KL (Oracle)', color='red')
        ax2.set_ylabel('KL Divergence', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # ==========================================
        # 🌟 核心修正：強制限制右軸 (KL) 的顯示範圍
        # 把上限壓在 2.5，這樣一開始飆到 25 的雜訊會跑到畫面外面，
        # 但你能極度清晰地看到 KL 是如何從 1.0 附近掉到 0 點幾。
        # ==========================================
        ax2.set_ylim(-0.1, 0.5) 
        
        # 💡 備用方案：如果你還是想在圖上保留一開始飆到 25 的軌跡，
        # 可以把上面那行註解掉，改成下面這行使用「對數尺度 (Log Scale)」
        # ax2.set_yscale('log')

        plt.title(f"{title} - Learning Curve")
        
        # 合併左軸與右軸的圖例，讓畫面更乾淨
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout() # 自動調整邊距避免文字被切掉
        plt.savefig(f"{filename}.png", dpi=300) # 加上 dpi=300 讓輸出圖片更清晰
        plt.close()

    

# ==========================================
# 修正後的實驗啟動流程 (ICL-Markov 對標)
# ==========================================

from dataset import ICLMarkovChainDataset, MarkovChainDataset, HMMDataset, ICLHMMDataset

def run_experiment():
    summary_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("results"): os.makedirs("results")
    
    config = {
        "seq_len": 100, 
        "num_symbols": 3, 
        "n_order": 1,
        "embed_dim": 16,
        "num_heads": 2, 
        "epochs": 20,
        "batch_size": 64, 
        "lr": 5e-4
    }

    dataset_names = ["ICL-Markov"] 
    layer_options = [2, 1] 
    attn_types = ["attention-only", "standard", "linear"]

    for data_name in dataset_names:
        for n_layer in layer_options:
            for attn in attn_types:
                model_tag = f"{data_name}_L{n_layer}_{attn}"
                print(f"\n>>> Running: {model_tag}")

                # 🌟 關鍵修正：將 test_ds 的 virtual_size 設定為 1000
                train_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=12800)
                test_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=1000)
                
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
                # Test DataLoader 的 batch_size 改為 100，這樣剛好跑 10 個 batch 測完 1000 筆資料
                test_loader = DataLoader(test_ds, batch_size=100, shuffle=False) 

                model = Transformer(
                    vocab_size=config["num_symbols"],
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    num_layers=n_layer,
                    block_size=config["seq_len"],
                    pe_type='rpe', # 💡強烈建議使用 'rpe' (Relative PE) 而非 'absolute' 以對標論文
                    attn_type=("standard" if attn == "attention-only" else attn),
                    attention_only=(attn == "attention-only")
                )
                
                trainer = Trainer(model, train_loader, test_loader, device, batch_size=config["batch_size"], lr=config["lr"])
                
                epoch_pbar = tqdm(range(config["epochs"]), desc=f"  Training")
                for epoch in epoch_pbar:
                    trainer.train_epoch(epoch + 1, model_tag)
                    ce, kl = trainer.evaluate()
                    epoch_pbar.set_postfix(Test_CE=f"{ce:.3f}", Test_KL=f"{kl:.4f}")
                
                trainer.save_plots(model_tag, f"results/{model_tag}")
                
                summary_data.append({
                    "Model": model_tag,
                    "Final_CE": f"{ce:.6f}",
                    "Final_KL": f"{kl:.6f}"
                })

    pd.DataFrame(summary_data).to_csv("results/summary.csv", index=False)
    print("\n[Finished] Check the 'results' folder for plots.")

if __name__ == "__main__":
    run_experiment()
