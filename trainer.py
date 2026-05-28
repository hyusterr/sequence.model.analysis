import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
import math
import torch.nn.functional as F
from model import Transformer
from dataset import ICLMarkovChainDataset

# ==========================================
# 核心 Metrics
# ==========================================
def compute_cross_entropy(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1).item()

def compute_kl_divergence(logits, target_probs):
    log_pred_probs = F.log_softmax(logits, dim=-1)
    kl_sum = F.kl_div(log_pred_probs, target_probs, reduction='sum')
    batch_size, seq_len, _ = logits.size()
    return (kl_sum / (batch_size * seq_len)).item()

# ==========================================
# Trainer 類別 (完全對標 minGPT 引擎)
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        # 提取 config 參數
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.total_steps = 0
        self.examples_seen = 0
        
        # 🌟 黑魔法 1：精準的 Weight Decay 分流 (放過 Bias, LayerNorm, Embedding)
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            # 包含 bias, 1D 張量(LayerNorm), emb(詞表), wpe/pos(位置) 均不參與衰減
            if name.endswith('bias') or len(param.shape) == 1 or 'emb' in name or 'wpe' in name or 'pos' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        # 採用 GPT 標準的 betas
        self.optimizer = optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95))
        
        # LR Scheduler 參數設定
        self.warmup_steps = 10 # 🌟 黑魔法 2：只有 10 步的極速預熱
        self.max_steps = self.epochs * len(train_loader)
        
        self.history = {
            "examples_seen": [], "step_ce": [], "step_kl": [],
            "test_ce": [], "test_kl": [], "test_at_examples": []
        }

    def _get_lr(self):
        # 🌟 黑魔法 3：Step-based Cosine Decay (最小值保留 10%)
        if self.total_steps < self.warmup_steps:
            return self.lr * float(self.total_steps + 1) / float(max(1, self.warmup_steps))
        if self.total_steps > self.max_steps:
            return self.lr * 0.1
        
        decay_ratio = (self.total_steps - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.lr * 0.1 + self.lr * 0.9 * coeff

    def train_epoch(self, epoch_idx, model_name):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch_idx}", leave=False)
        
        for x, y, p_true, info in pbar:
            # 1. 更新當前 Step 的學習率
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            
            # 🌟 黑魔法 4：保護脆弱的歸納頭不被梯度炸毀
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0)
            
            if self.total_steps % 20 == 0:
                with torch.no_grad():
                    kl = compute_kl_divergence(logits, p_true)
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["step_ce"].append(loss.item())
                    self.history["step_kl"].append(kl)
                    pbar.set_postfix(Examples=self.examples_seen, CE=f"{loss.item():.4f}", LR=f"{lr:.1e}")

    def evaluate(self, fixed_test_data):
        """傳入預先生成好的 Test Data，消除動態抽樣帶來的評估震盪"""
        self.model.eval()
        all_ce, all_kl = [], []
        
        with torch.no_grad():
            for x, y, p_true in fixed_test_data:
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                # 只評估 Last Token
                last_logits = logits[:, -1, :] 
                last_y = y[:, -1]
                last_p_true = p_true[:, -1, :]
                
                ce = F.cross_entropy(last_logits, last_y).item()
                log_pred = F.log_softmax(last_logits, dim=-1)
                kl = F.kl_div(log_pred, last_p_true, reduction='batchmean').item()
                
                all_ce.append(ce)
                all_kl.append(kl)
        
        avg_ce, avg_kl = np.mean(all_ce), np.mean(all_kl)
        self.history["test_ce"].append(avg_ce)
        self.history["test_kl"].append(avg_kl)
        self.history["test_at_examples"].append(self.examples_seen)
        return avg_ce, avg_kl

    def save_plots(self, title, filename):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        ax1.plot(self.history["examples_seen"], self.history["step_ce"], alpha=0.15, color='blue', label='Train CE')
        ax1.plot(self.history["test_at_examples"], self.history["test_ce"], marker='o', label='Test CE (Last Token)', color='blue')
        ax1.set_xlabel('Number of Examples')
        ax1.set_ylabel('Cross Entropy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(self.history["examples_seen"], self.history["step_kl"], alpha=0.15, color='red', linestyle=':')
        ax2.plot(self.history["test_at_examples"], self.history["test_kl"], marker='x', label='Test KL (Oracle)', color='red')
        ax2.set_ylabel('KL Divergence', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 🌟 視角鎖定：看清楚 0 到 0.35 之間的相變
        ax2.set_ylim(-0.02, 0.35) 
        
        plt.title(f"{title} - Learning Curve")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout()
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
        "num_heads": 1,   # 🌟 黑魔法 5：測 Bigram 時強制單頭注意力
        "epochs": 20,
        "batch_size": 64, 
        "lr": 5e-4        # 維持充足的動能
    }

    dataset_names = ["ICL-Markov"] 
    layer_options = [2, 1] 
    attn_types = ["attention-only", "standard", "linear"]

    print(">>> 預先生成並固定測試集 (消滅抽樣震盪)...")
    test_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=4000)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
    fixed_test_data = []
    for x, y, p_true, info in test_loader:
        fixed_test_data.append((x.clone(), y.clone(), p_true.clone()))

    for data_name in dataset_names:
        for n_layer in layer_options:
            for attn in attn_types:
                model_tag = f"{data_name}_L{n_layer}_{attn}"
                print(f"\n>>> 啟動訓練: {model_tag}")

                train_ds = ICLMarkovChainDataset(config["seq_len"], config["num_symbols"], config["n_order"], virtual_size=12800)
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)

                model = Transformer(
                    vocab_size=config["num_symbols"],
                    d_model=config["embed_dim"],
                    nhead=config["num_heads"],
                    num_layers=n_layer,
                    block_size=config["seq_len"],
                    pe_type='rpe', 
                    attn_type=("standard" if attn == "attention-only" else attn),
                    attention_only=(attn == "attention-only")
                )
                
                # 初始化裝備了 minGPT 引擎的 Trainer
                trainer = Trainer(model, train_loader, device, config)
                
                epoch_pbar = tqdm(range(config["epochs"]), desc="  Epoch Progress")
                for epoch in epoch_pbar:
                    trainer.train_epoch(epoch + 1, model_tag)
                    ce, kl = trainer.evaluate(fixed_test_data)
                    # 顯示當前最終的 LR (抓取優化器中實際的數值)
                    current_lr = trainer.optimizer.param_groups[0]['lr']
                    epoch_pbar.set_postfix(Test_CE=f"{ce:.3f}", Test_KL=f"{kl:.4f}", LR=f"{current_lr:.1e}")
                
                trainer.save_plots(model_tag, f"results/{model_tag}")
                
                summary_data.append({
                    "Model": model_tag,
                    "Final_CE": f"{ce:.6f}",
                    "Final_KL": f"{kl:.6f}"
                })

    pd.DataFrame(summary_data).to_csv("results/summary.csv", index=False)
    print("\n[Finished] 訓練完成，請查看 'results' 資料夾中的圖片。")

if __name__ == "__main__":
    run_experiment()
