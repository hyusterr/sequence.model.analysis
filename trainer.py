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
    def __init__(self, model, train_loader, device, transition_matrix=None, lr=3e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.transition_matrix = transition_matrix
        # 儲存歷史數據
        self.history = {"test_ce": [], "test_kl": []}

    def train_epoch(self, epoch_idx, model_name):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"  {model_name} Epoch {epoch_idx}", leave=False)
        
        for x, y, p_true in pbar: # 確保領取 p_true
            x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            
            # --- 新增：每個 Step 記錄一次 ---
            # 這裡的 loss 就是當前 batch 的 Cross Entropy
            current_ce = loss.item()
            
            # 計算當前 batch 的 KL
            # 注意：這裡呼叫你修正過量級（除以 B*T）後的 compute_kl_divergence
            current_kl = compute_kl_divergence(logits, p_true)
            
            self.history["test_ce"].append(current_ce)
            self.history["test_kl"].append(current_kl)
            # ------------------------------

            total_loss += current_ce
            pbar.set_postfix(CE=f"{current_ce:.4f}", KL=f"{current_kl:.4f}")
            
        return total_loss / len(self.train_loader)

    # Trainer.evaluate 內部
    def evaluate(self):
        self.model.eval()
        all_ce, all_kl = [], []
        with torch.no_grad():
            for x, y, p_true in self.train_loader: # 注意這裡多領一個 p_true
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                ce = compute_cross_entropy(logits, y)
                all_ce.append(ce)
                
                # 直接傳入 p_true 算 KL
                kl = compute_kl_divergence(logits, p_true)
                all_kl.append(kl)
        
        return np.mean(all_ce), np.mean(all_kl)

    def save_plots(self, title, filename, config):
        # 橫軸改為總步數
        total_steps = len(self.history["test_ce"])
        steps = range(1, total_steps + 1)
        
        # 如果你想對標論文的 "Examples Seen (Thousands)"
        # examples_seen = [i * config["batch_size"] / 1000 for i in steps]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Cross Entropy
        ax1.set_xlabel('Steps (Iterations)')
        ax1.set_ylabel('Cross Entropy', color='tab:blue')
        ax1.plot(steps, self.history["test_ce"], color='tab:blue', alpha=0.6, label='CE')
        
        # KL Divergence
        ax2 = ax1.twinx()
        ax2.set_ylabel('KL Divergence', color='tab:red')
        ax2.plot(steps, self.history["test_kl"], color='tab:red', alpha=0.8, label='KL')

        plt.title(f"Step-level Metrics: {title}")
        fig.tight_layout()
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
        "epochs": 100, 
        "batch_size": 64
    }

    dataset_names = ["ICL-Markov"] #, "HMM", "Markov"]
    layer_options = [2, 1] # 1層與2層比較
    attn_types = ["attention-only", "standard", "linear"] #, "performer"]

    for data_name in dataset_names:
        print(f"\n" + "="*50)
        print(f"Dataset: {data_name}")
        print("="*50)

        # 建立數據集
        if data_name == "Markov":
            ds = MarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
        elif data_name == "ICL-Markov":
            train_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"])
            test_ds = test_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], virtual_size=128)
        else:
            ds = HMMDataset(config["seq_len"], num_hidden=2, num_obs=config["num_symbols"])

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"])
        test_loader = DataLoader(test_ds, batch_size=128) # 一次跑完 128 條取平均

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
