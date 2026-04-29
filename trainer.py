import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd

# 匯入你的自定義模組
from dataset import MarkovChainDataset, ICLMarkovChainDataset, HMMDataset
from model import Transformer
from metrics import compute_kl_divergence, compute_cross_entropy

class Trainer:
    def __init__(self, model, train_loader, device, transition_matrix=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.transition_matrix = transition_matrix
        # 儲存歷史數據
        self.history = {"test_ce": [], "test_kl": []}

    def train_epoch(self, epoch_idx, model_name):
        self.model.train()
        total_loss = 0
        # 內層進度條：顯示每個 Batch 的訓練狀況
        pbar = tqdm(self.train_loader, desc=f"  {model_name} Epoch {epoch_idx}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y) # 使用 model 內建的 loss 計算
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return total_loss / len(self.train_loader)

    # Trainer.evaluate 內部
    def evaluate(self):
        self.model.eval()
        all_kl = []
        with torch.no_grad():
            for batch in self.train_loader:
                # 現在每個 dataset 都回傳 (x, y, target_dist)
                x, y, target_dist = batch
                x, y, target_dist = x.to(self.device), y.to(self.device), target_dist.to(self.device)
            
                logits, _ = self.model(x)
            
                # 直接把正確答案餵進去算 KL
                kl = compute_kl_divergence(logits, target_dist)
                all_kl.append(kl)
        return np.mean(all_kl)


    def save_plots(self, title, filename):
        epochs = range(1, len(self.history["test_ce"]) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Cross Entropy 藍線
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Cross Entropy', color=color)
        ax1.plot(epochs, self.history["test_ce"], color=color, label='CE', marker='o', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)

        # KL Divergence 紅線 (使用雙軸，因為量級不同)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('KL Divergence', color=color)
        # 避免 KL 為 0 時 Y 軸標籤出現負數
        if all(v == 0 for v in self.history["test_kl"]):
            ax2.set_ylim(-0.1, 1.0) 
        ax2.plot(epochs, self.history["test_kl"], color=color, label='KL', marker='x', linestyle='--', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title(f"Training Metrics: {title}")
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()

# ==========================================
# 實驗啟動流程
# ==========================================

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
        "batch_size": 32
    }

    dataset_names = ["Markov", "ICL-Markov", "HMM"]
    layer_options = [1, 2] # 1層與2層比較
    attn_types = ["attention-only", "standard", "linear", "performer"]

    for data_name in dataset_names:
        print(f"\n" + "="*50)
        print(f"Dataset: {data_name}")
        print("="*50)

        # 建立數據集
        if data_name == "Markov":
            ds = MarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
        elif data_name == "ICL-Markov":
            ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
        else:
            ds = HMMDataset(config["seq_len"], num_hidden=2, num_obs=config["num_symbols"])

        train_loader = DataLoader(ds, batch_size=config["batch_size"])
        p_matrix = getattr(ds, 'P', None) # 取得 Markov 的固定轉移矩陣

        for n_layer in layer_options:
            for attn in attn_types:
                model_tag = f"L{n_layer}-{attn}"
                
                # 模型參數切換
                is_attn_only = (attn == "attention-only")
                actual_attn = "standard" if is_attn_only else attn
                
                model = Transformer(
                    vocab_size=config["num_symbols"],
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    num_layers=n_layer,
                    block_size=config["seq_len"],
                    pe_type='absolute', # 預設使用 Absolute PE 以利觀察
                    attn_type=actual_attn,
                    attention_only=is_attn_only
                )
                
                trainer = Trainer(model, train_loader, device, transition_matrix=p_matrix)
                
                # 外層進度條：監控 20 個 Epoch
                epoch_pbar = tqdm(range(config["epochs"]), desc=f"  {model_tag}", leave=True)
                for epoch in epoch_pbar:
                    train_loss = trainer.train_epoch(epoch + 1, model_tag)
                    ce, kl = trainer.evaluate()
                    
                    trainer.history["test_ce"].append(ce)
                    trainer.history["test_kl"].append(kl)
                    
                    epoch_pbar.set_postfix(CE=f"{ce:.3f}", KL=f"{kl:.4f}")
                
                # 存圖：包含數據名、層數與模型類型
                plot_filename = f"{data_name}_L{n_layer}_{attn}"
                trainer.save_plots(plot_filename, f"results/{plot_filename}")

                # 在訓練結束後，抓取最後一個 Epoch 的結果
                final_ce = trainer.history["test_ce"][-1]
                final_kl = trainer.history["test_kl"][-1]
                
                # 存入總表清單
                summary_data.append({
                    "Dataset": data_name,
                    "Layers": n_layer,
                    "Attention_Type": attn,
                    "Final_CrossEntropy": f"{final_ce:.6f}",
                    "Final_KLDivergence": f"{final_kl:.6f}"
                })
    # --- 儲存為 CSV ---
    df = pd.DataFrame(summary_data)
    csv_path = "results/experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n" + "!"*50)
    print(f"所有實驗已完成！總表已存至: {csv_path}")
    print("!"*50)

    return df

if __name__ == "__main__":
    run_experiment()
