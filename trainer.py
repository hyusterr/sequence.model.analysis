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

# ==========================================
# 核心 Metrics 擴充 (完全對準 Last Token)
# ==========================================

def compute_all_metrics(logits, targets, target_probs, eps=1e-12):
    """
    對最後一個位置計算四種指標的排列組合
    logits: [B, T, V]
    targets: [B, T]
    target_probs: [B, T, V]
    """
    # 統一鎖定 Last Token
    last_logits = logits[:, -1, :]          # [B, V]
    last_y = targets[:, -1]                  # [B]
    last_p_true = target_probs[:, -1, :]    # [B, V]
    
    # 模型預測的 Log-probabilities 與 Probabilities
    log_pred = F.log_softmax(last_logits, dim=-1)
    pred_prob = torch.exp(log_pred)
    
    # ------------------------------------------
    # 1. Cross Entropy 家族
    # ------------------------------------------
    # Sample CE: 標準 Cross Entropy (對 One-hot 標籤)
    sample_ce = F.cross_entropy(last_logits, last_y, ignore_index=-1).item()
    
    # Theoretical CE: -sum( P_true * log(Q) )
    theoretical_ce = -(last_p_true * log_pred).sum(dim=-1).mean().item()
    
    # ------------------------------------------
    # 2. KL Divergence 家族
    # ------------------------------------------
    # Theoretical KL: 標準論文觀測指標 sum( P_true * log(P_true / Q) )
    # 使用 'batchmean' 對應批次平均
    theoretical_kl = F.kl_div(log_pred, last_p_true, reduction='batchmean').item()
    
    # Sample KL: sum( Y_onehot * log(Y_onehot / Q) )
    # Y_onehot 在真實出現的位置為 1，其餘為 0。因此簡化為: 1 * log(1 / Q_target) = -log(Q_target)
    # 注意：在資訊理論中，對 One-hot 算 KL，數值等同於 Sample CE 扣掉 One-hot 自身的熵 (0)
    # 為了保持嚴謹度，這裡實作完整的 KL 邏輯：
    y_one_hot = F.one_hot(last_y, num_classes=logits.size(-1)).float()
    sample_kl = (y_one_hot * (torch.log(y_one_hot + eps) - log_pred)).sum(dim=-1).mean().item()
    
    return sample_ce, theoretical_ce, sample_kl, theoretical_kl


# ==========================================
# Trainer 類別
# ==========================================
class Trainer:
    def __init__(self, model, train_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        self.lr = config["lr"]
        self.epochs = config["epochs"]
        self.total_steps = 0
        self.examples_seen = 0
        
        # minGPT Weight Decay 分流
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if name.endswith('bias') or len(param.shape) == 1 or 'emb' in name or 'wpe' in name or 'pos' in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optim_groups = [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        self.optimizer = optim.AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95))
        self.warmup_steps = 10 
        self.max_steps = self.epochs * len(train_loader)
        
        # 擴充歷史儲存架構
        self.history = {
            "examples_seen": [],
            "train_sample_ce": [], "train_theory_ce": [], "train_sample_kl": [], "train_theory_kl": [],
            "test_sample_ce": [], "test_theory_ce": [], "test_sample_kl": [], "test_theory_kl": [],
            "test_at_examples": []
        }

    def _get_lr(self):
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
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
            x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
            
            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            loss.backward()
            self.optimizer.step()
            
            self.total_steps += 1
            self.examples_seen += x.size(0)
            
            if self.total_steps % eval_interval == 0:
                with torch.no_grad():
                    s_ce, t_ce, s_kl, t_kl = compute_all_metrics(logits, y, p_true)
                    self.history["examples_seen"].append(self.examples_seen)
                    self.history["train_sample_ce"].append(s_ce)
                    self.history["train_theory_ce"].append(t_ce)
                    self.history["train_sample_kl"].append(s_kl)
                    self.history["train_theory_kl"].append(t_kl)
                    
                # 評估測試集
                metrics = self.evaluate(fixed_test_data)
                self.model.train()
                
                pbar.set_postfix(
                    T_CE=f"{metrics['t_ce']:.3f}", 
                    T_KL=f"{metrics['t_kl']:.3f}",
                    S_CE=f"{metrics['s_ce']:.3f}"
                )

    def evaluate(self, fixed_test_data):
        self.model.eval()
        batch_metrics = {"s_ce": [], "t_ce": [], "s_kl": [], "t_kl": []}
        
        with torch.no_grad():
            for x, y, p_true in fixed_test_data:
                x, y, p_true = x.to(self.device), y.to(self.device), p_true.to(self.device)
                logits, _ = self.model(x)
                
                s_ce, t_ce, s_kl, t_kl = compute_all_metrics(logits, y, p_true)
                batch_metrics["s_ce"].append(s_ce)
                batch_metrics["t_ce"].append(t_ce)
                batch_metrics["s_kl"].append(s_kl)
                batch_metrics["t_kl"].append(t_kl)
        
        # 紀錄平均值
        avg_metrics = {k: np.mean(v) for k, v in batch_metrics.items()}
        
        self.history["test_sample_ce"].append(avg_metrics["s_ce"])
        self.history["test_theory_ce"].append(avg_metrics["t_ce"])
        self.history["test_sample_kl"].append(avg_metrics["s_kl"])
        self.history["test_theory_kl"].append(avg_metrics["t_kl"])
        self.history["test_at_examples"].append(self.examples_seen)
        
        return avg_metrics

    def save_plots(self, title, filename):
        """同步產出兩張對照圖表：Sample 面板與 Theoretical 面板"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # ------------------------------------------
        # 左圖：Sample Metrics Panel
        # ------------------------------------------
        ax1_twin = ax1.twinx()
        ax1.plot(self.history["examples_seen"], self.history["train_sample_ce"], alpha=0.1, color='blue')
        ax1.plot(self.history["test_at_examples"], self.history["test_sample_ce"], marker='o', color='blue', label='Test Sample CE')
        ax1_twin.plot(self.history["test_at_examples"], self.history["test_sample_kl"], marker='x', linestyle='--', color='red', label='Test Sample KL')
        
        ax1.set_xlabel('Number of Examples')
        ax1.set_ylabel('Cross Entropy (Sample)', color='blue')
        ax1_twin.set_ylabel('KL Divergence (Sample)', color='red')
        ax1.set_title("Sample-level Metrics (vs Realized Token)")
        
        # ------------------------------------------
        # 右圖：Theoretical Metrics Panel (對標論文核心)
        # ------------------------------------------
        ax2_twin = ax2.twinx()
        ax2.plot(self.history["examples_seen"], self.history["train_theory_ce"], alpha=0.1, color='blue')
        ax2.plot(self.history["test_at_examples"], self.history["test_theory_ce"], marker='o', color='blue', label='Test Theory CE')
        ax2_twin.plot(self.history["test_at_examples"], self.history["test_theory_kl"], marker='x', linestyle='--', color='red', label='Test Theory KL')
        
        ax2.set_xlabel('Number of Examples')
        ax2.set_ylabel('Cross Entropy (Theory)', color='blue')
        ax2_twin.set_ylabel('KL Divergence (Theory)', color='red')
        # 視角鎖定
        ax2_twin.set_ylim(-0.02, 0.35)
        ax2.set_title("Theoretical Metrics (vs Transition Matrix)")
        
        plt.suptitle(f"{title} Evolution", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=300)
        plt.close()
