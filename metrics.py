import torch
import torch.nn.functional as F

def compute_cross_entropy(logits, targets):
    """
    計算 Testing set 的 Cross Entropy Loss。
    logits: [batch_size, seq_len, vocab_size]
    targets: [batch_size, seq_len]
    """
    # 將 batch 和時序維度拉平以符合 F.cross_entropy 的輸入要求
    # 忽略 index 為 -1 的目標（如果有 padding 的話）
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss.item()


def compute_kl_divergence(logits, target_probs):
    """
    logits: [batch_size, T-1, V]
    target_probs: [batch_size, T-1, V] (從 Dataset 傳來的正確答案)
    """
    # 轉為 log 空間
    log_pred_probs = F.log_softmax(logits, dim=-1)
    
    # 計算 KL (P || Q) = P * (log P - log Q)
    # 加上 eps 避免 log(0)
    eps = 1e-10
    kl = target_probs * (torch.log(target_probs + eps) - log_pred_probs)
    
    # 回傳平均每個 token 的 KL
    return kl.sum() / (target_probs.size(0) * target_probs.size(1))
