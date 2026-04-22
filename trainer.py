import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt  # 新增繪圖庫
from tqdm import tqdm
import os

# 假設你之前的模組命名如下
from dataset import MarkovChainDataset, ICLMarkovChainDataset, HMMDataset
from model import Transformer
from metrics import compute_kl_divergence, compute_cross_entropy

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, transition_matrix=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.transition_matrix = transition_matrix
        
        # 用來記錄繪圖數據
        self.history = {"train_loss": [], "test_ce": [], "test_kl": []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}", leave=False)

        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y) # 直接使用 model 內建的 loss 計算
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_ce, all_kl = [], []
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _ = self.model(x)
                
                ce = compute_cross_entropy(logits, y)
                all_ce.append(ce)
                
                if self.transition_matrix is not None:
                    # KL 散度衡量模型預測與真實轉移矩陣 P 的距離
                    kl = compute_kl_divergence(logits, x, self.transition_matrix.to(self.device))
                    all_kl.append(kl)
        
        return np.mean(all_ce), np.mean(all_kl) if all_kl else 0.0

    def save_plots(self, title, filename):
        """ 將訓練結果存成圖檔 """
        epochs = range(1, len(self.history["test_ce"]) + 1)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 繪製 Cross Entropy (左軸)
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross Entropy', color=color)
        ax1.plot(epochs, self.history["test_ce"], color=color, label='Test CE', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)

        # 建立雙軸繪製 KL Divergence (右軸)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('KL Divergence', color=color)
        ax2.plot(epochs, self.history["test_kl"], color=color, label='Test KL', marker='x')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training Metrics: {title}")
        fig.tight_layout()
        plt.savefig(f"{filename}.png", dpi=300) # 儲存高解析度圖片
        plt.close()

# ==========================================
# 實驗啟動流程
# ==========================================

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    
    config = {
        "seq_len": 100, "num_symbols": 3, "n_order": 1,
        "embed_dim": 16, "num_heads": 2, "epochs": 20, "batch_size": 32
    }

    # 數據集與模型清單
    dataset_names = ["Markov", "ICL-Markov", "HMM"]
    attn_types = ["standard", "linear", "performer"] # standard 用來測 normal & attention-only

    for data_name in dataset_names:
        if data_name == "Markov":
            ds = MarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
        elif data_name == "ICL-Markov":
            ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
        else:
            ds = HMMDataset(config["seq_len"], num_hidden=2, num_obs=config["num_symbols"])

        train_loader = DataLoader(ds, batch_size=config["batch_size"])
        p_matrix = getattr(ds, 'P', None) # 只有 Markov 有固定的 P

        for attn in ["attention-only", "standard", "linear", "performer"]:
            print(f">>> Data: {data_name} | Model: {attn}")
            
            # 建立模型實例
            is_attn_only = (attn == "attention-only")
            actual_attn = "standard" if is_attn_only else attn
            
            model = Transformer(
                vocab_size=config["num_symbols"],
                d_model=config["embed_dim"],
                nhead=config["num_heads"],
                num_layers=1, # 實驗先跑一層
                block_size=config["seq_len"],
                attn_type=actual_attn,
                attention_only=is_attn_only
            )
            
            trainer = Trainer(model, train_loader, train_loader, device, transition_matrix=p_matrix)
           
            epoch_pbar = tqdm(range(config["epochs"]), desc="Overall Progress")
            for epoch in epoch_pbar:
                loss = trainer.train_epoch()
                ce, kl = trainer.evaluate()
                trainer.history["test_ce"].append(ce)
                trainer.history["test_kl"].append(kl)

                epoch_pbar.set_postfix(CE=f"{ce:.4f}", KL=f"{kl:.4f}")
            
            # 存圖：檔名格式如 Markov_performer.png
            trainer.save_plots(f"{data_name}_{attn}", f"results/{data_name}_{attn}")

if __name__ == "__main__":
    run_experiment()
