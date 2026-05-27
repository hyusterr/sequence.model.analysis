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
# 核心 Metrics
# ==========================================

def compute_cross_entropy(logits, targets):
    """計算平均每個 token 的 Cross Entropy (用於訓練)"""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1).item()

# ==========================================
# Trainer 類別 (對標論文 test_last_token 邏輯)
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, test_loader, device, batch_size, lr=3e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        self.history = {
            "examples_seen": [], # 橫軸：訓練過的總樣本數 (Examples)
            "step_ce": [],       # 訓練時的全序列平均 Loss
            "step_kl": [],
            "test_ce": [],       # 測試時的 Last Token Loss
            "test_kl": [],       # 測試時的 Last Token KL (Oracle)
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
            logits, loss = self.model(x, y) # 訓練時計算所有 Token 的 Loss
            loss.backward()
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0) 
            
            if self.total_steps % 20 == 0:
                with torch.no_grad():
                    # 訓練階段的監控，依然看全體的平均情況
                    log_pred = F.log_softmax(logits, dim=-1)
                    kl = F.kl_div(log_pred, p_true, reduction='batchmean').item()
                    
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["step_ce"].append(loss.item())
                    self.history["step_kl"].append(kl)
                    pbar.set_postfix(Examples=self.examples_seen, CE=f"{loss.item():.4f}")

    def evaluate(self):
        """對標 test_error.test_last_token 在獨立 Test Set 上進行評估"""
        self.model.eval()
        all_ce, all_kl = [], []
        
        with torch.no_grad():
            for x, y, p_true, info in self.test_loader:
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                # 🌟 關鍵修改：只取最後一個 Token 進行評估！
                # logits 原本是 [Batch, Seq_len, Vocab] -> 變成 [Batch, Vocab]
                last_logits = logits[:, -1, :] 
                last_y = y[:, -1]
                last_p_true = p_true[:, -1, :]
                
                # 1. 計算 Last Token Cross Entropy
                ce = F.cross_entropy(last_logits, last_y).item()
                
                # 2. 計算 Last Token KL Divergence (與 Oracle 比較)
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
        
        ax1.plot(self.history["examples_seen"], self.history["step_ce"], alpha=0.15, color='blue', label='Train CE (All Tokens)')
        ax1.plot(self.history["test_at_examples"], self.history["test_ce"], marker='o', label='Test CE (Last Token)', color='blue', linewidth=2)
        ax1.set_xlabel('Number of Examples')
        ax1.set_ylabel('Cross Entropy (nats)')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history["examples_seen"], self.history["step_kl"], alpha=0.1, color='red', linestyle=':')
        ax2.plot(self.history["test_at_examples"], self.history["test_kl"], marker='x', label='Test KL (Last Token Oracle)', color='red', linewidth=1.5)
        ax2.set_ylabel('KL Divergence')
        
        plt.title(f"Evolution of Induction: {title}")
        fig.tight_layout()
        
        # 讓圖例統整顯示
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()

# ==========================================
# 修正後的實驗啟動流程
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
        "lr": 3e-5
    }

    dataset_names = ["ICL-Markov"] 
    layer_options = [2, 1] 
    attn_types = ["attention-only", "standard", "linear"]

    for data_name in dataset_names:
        for n_layer in layer_options:
            for attn in attn_types:
                model_tag = f"{data_name}_L{n_layer}_{attn}"
                print(f"\n>>> Running: {model_tag}")

                train_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=12800)
                test_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=256)
                
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=64)

                model = Transformer(
                    vocab_size=config["num_symbols"],
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    num_layers=n_layer,
                    block_size=config["seq_len"],
                    # pe_type='absolute',  # edelman et al use RPE
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
