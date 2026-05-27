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
    """計算平均每個 token 的 Cross Entropy"""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1).item()

def compute_kl_divergence(logits, target_probs):
    """
    計算模型預測與真實 P 之間的 KL Divergence。
    確保輸出量級為平均每個 token 的 bits/nats。
    """
    # 轉為 log 空間
    log_pred_probs = F.log_softmax(logits, dim=-1)
    
    # 使用 reduction='sum' 確保我們可以精確控制除數
    # KL(P || Q) = sum(P * log(P/Q))
    kl_sum = F.kl_div(log_pred_probs, target_probs, reduction='sum')
    
    batch_size, seq_len, _ = logits.size()
    # 排除掉第一個無法預測的 token (對標論文實作)
    return (kl_sum / (batch_size * seq_len)).item()

# ==========================================
# Trainer 類別 (對標論文 A.1 訓練邏輯)
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
            # Info 字典自動轉移裝置
            info = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in info.items()}
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0) # 累計處理過的樣本總數
            
            # 頻率監控 (每 20 個 step 記錄一次以提升效能)
            if self.total_steps % 20 == 0:
                with torch.no_grad():
                    kl = self._compute_kl(logits, p_true)
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["step_ce"].append(loss.item())
                    self.history["step_kl"].append(kl)
                    pbar.set_postfix(Examples=self.examples_seen, CE=f"{loss.item():.4f}")

    def _compute_kl(self, logits, target_probs):
        log_pred = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_pred, target_probs, reduction='batchmean').item()

    def evaluate(self):
        """在獨立的 Test Set 上進行評估"""
        self.model.eval()
        all_ce, all_kl = [], []
        
        with torch.no_grad():
            for x, y, p_true, info in self.test_loader:
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                # 計算 CE 與 KL
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1).item()
                kl = self._compute_kl(logits, p_true)
                
                all_ce.append(ce)
                all_kl.append(kl)
        
        avg_ce = np.mean(all_ce)
        avg_kl = np.mean(all_kl)
        
        self.history["test_ce"].append(avg_ce)
        self.history["test_kl"].append(avg_kl)
        self.history["test_at_examples"].append(self.examples_seen)
        
        # 🌟 關鍵：這裡必須有 return！
        return avg_ce, avg_kl

    def save_plots(self, title, filename):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        # 繪圖
        ax1.plot(self.history["examples_seen"], self.history["step_ce"], alpha=0.2, color='blue')
        ax1.plot(self.history["test_at_examples"], self.history["test_ce"], marker='o', label='Test Loss (CE)', color='blue')
        ax1.set_xlabel('Number of Examples')
        ax1.set_ylabel('Cross Entropy')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history["examples_seen"], self.history["step_kl"], alpha=0.2, color='red')
        ax2.plot(self.history["test_at_examples"], self.history["test_kl"], marker='x', label='Test KL (Oracle)', color='red')
        ax2.set_ylabel('KL Divergence')
        
        plt.title(f"{title} - Learning Curve")
        plt.savefig(f"{filename}.png")
        plt.close()

# ==========================================
# 修正後的實驗啟動流程 (ICL-Markov 對標)
# ==========================================

from dataset import ICLMarkovChainDataset, MarkovChainDataset, HMMDataset, ICLHMMDataset

def run_experiment():
    summary_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("results"): os.makedirs("results")
    
    # 論文 A.1 的標準配置
    config = {
        "seq_len": 100, 
        "num_symbols": 3, 
        "n_order": 1,
        "embed_dim": 16, # 關鍵：d=16
        "num_heads": 2, 
        "epochs": 20,
        "batch_size": 64, # 對標論文
        "lr": 3e-5 # 對標一階馬可夫
    }

    # 目前專注於最能觀察相變的 ICL-Markov
    dataset_names = ["ICL-Markov"] 
    layer_options = [2, 1] 
    attn_types = ["attention-only", "standard", "linear"]

    for data_name in dataset_names:
        for n_layer in layer_options:
            for attn in attn_types:
                model_tag = f"{data_name}_L{n_layer}_{attn}"
                print(f"\n>>> Running: {model_tag}")

                # 1. 數據集準備 (確保 Train/Test 分離)
                train_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=12800)
                test_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=256)
                
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
                test_loader = DataLoader(test_ds, batch_size=64)

                # 2. 模型初始化
                model = Transformer(
                    vocab_size=config["num_symbols"],
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    num_layers=n_layer,
                    block_size=config["seq_len"],
                    pe_type='absolute', 
                    attn_type=("standard" if attn == "attention-only" else attn),
                    attention_only=(attn == "attention-only")
                )
                
                # 3. 訓練與評估
                trainer = Trainer(model, train_loader, test_loader, device, batch_size=config["batch_size"], lr=config["lr"])
                
                epoch_pbar = tqdm(range(config["epochs"]), desc=f"  Training")
                for epoch in epoch_pbar:
                    trainer.train_epoch(epoch + 1, model_tag)
                    # 每完成一個 Epoch 就跑一次完整的 Test Evaluation
                    ce, kl = trainer.evaluate()
                    epoch_pbar.set_postfix(Test_CE=f"{ce:.3f}", Test_KL=f"{kl:.4f}")
                
                # 4. 存圖與數據
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
