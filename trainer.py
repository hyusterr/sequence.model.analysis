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

    def train_epoch(self, epoch_idx, model_name, fixed_test_data, eval_interval=20):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch_idx}", leave=False)
        
        for x, y, p_true, info in pbar:
            # 1. 學習率更新
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            
            # (注意：這裡已經依照先前的結論，拿掉 clip_grad_norm_ 以釋放相變動能)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0)
            
            # ==========================================
            # 🌟 高頻率評估：每 eval_interval 步，執行一次 Test
            # ==========================================
            if self.total_steps % eval_interval == 0:
                with torch.no_grad():
                    # 紀錄訓練集 Loss
                    train_kl = compute_kl_divergence(logits, p_true)
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["step_ce"].append(loss.item())
                    self.history["step_kl"].append(train_kl)
                    
                # 評估測試集並紀錄
                test_ce, test_kl = self.evaluate(fixed_test_data)
                
                # ⚠️ 關鍵：evaluate 裡面呼叫了 self.model.eval()，這裡必須切回 train 模式
                self.model.train()
                
                pbar.set_postfix(Test_CE=f"{test_ce:.3f}", Test_KL=f"{test_kl:.4f}")



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

import itertools

def run_experiment():
    summary_data = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("results"): os.makedirs("results")
    
    # ==========================================
    # 🌟 擴充版超參數矩陣 (加入了 lr 進行網格搜尋)
    # ==========================================
    grid = {
        "data_name": ["ICL-Markov"],
        "num_symbols": [3, 2],        
        "n_order": [1, 2],            
        "n_layer": [2, 1],         
        "attn": ["attention-only", "standard", "linear"], 
        "embed_dim": [16],         
        "num_heads": [1, 2],          
        "lr": [3e-5, 1e-4, 5e-4]   # 🌟 將 LR 放進來跑排列組合
    }
    
    # 固定訓練參數 (移除 lr)
    train_params = {
        "seq_len": 100,
        "epochs": 30,
        "batch_size": 64,
        "eval_interval": 20       
    }

    # 自動展開所有的排列組合
    keys, values = zip(*grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 溫馨提醒：排列組合數量會相乘，例如 2(layer) x 3(attn) x 3(lr) = 18 組實驗
    print(f"總共準備執行 {len(experiments)} 組實驗...\n")

    for p in experiments:
        # 1. 🌟 檔名加入 LR，避免不同學習率的圖互相覆蓋
        dataset_str = f"{p['data_name']}_V{p['num_symbols']}_O{p['n_order']}"
        model_str = f"L{p['n_layer']}_H{p['num_heads']}_D{p['embed_dim']}_{p['attn']}_LR{p['lr']}" 
        model_tag = f"{dataset_str}_{model_str}"
        
        print(f"\n>>> Running: {model_tag}")

        # 2. 生成對應參數的 Dataset
        test_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=4000)
        test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)
        
        print("    Pre-generating fixed Test Set...")
        fixed_test_data = []
        for x, y, p_true, info in test_loader:
            fixed_test_data.append((x.clone(), y.clone(), p_true.clone()))

        train_ds = ICLMarkovChainDataset(train_params["seq_len"], p["num_symbols"], p["n_order"], virtual_size=12800)
        train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)

        # 3. 初始化 Model
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
        
        # 4. 初始化 Trainer (🌟 動態讀取 p["lr"])
        config_for_trainer = {
            "lr": p["lr"], 
            "epochs": train_params["epochs"]
        }
        trainer = Trainer(model, train_loader, device, config_for_trainer)
        
        # 5. 訓練流程
        epoch_pbar = tqdm(range(train_params["epochs"]), desc="    Epochs")
        for epoch in epoch_pbar:
            trainer.train_epoch(epoch + 1, model_tag, fixed_test_data, eval_interval=train_params["eval_interval"])
            if len(trainer.history["test_kl"]) > 0:
                epoch_pbar.set_postfix(Test_KL=f"{trainer.history['test_kl'][-1]:.4f}")
        
        # 6. 存圖與寫入記錄
        trainer.save_plots(model_tag, f"results/{model_tag}")
        
        final_ce = trainer.history["test_ce"][-1] if trainer.history["test_ce"] else 0.0
        final_kl = trainer.history["test_kl"][-1] if trainer.history["test_kl"] else 0.0

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
            "LR": p["lr"],               # 🌟 新增 LR 欄位到 CSV
            "Final_CE": f"{final_ce:.6f}",
            "Final_KL": f"{final_kl:.6f}"
        })

        # 即時存檔
        pd.DataFrame(summary_data).to_csv("results/summary.csv", index=False)

    print("\n[Finished] 所有實驗執行完畢！")

if __name__ == "__main__":
    run_experiment()
