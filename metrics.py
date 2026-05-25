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
    logits: [batch_size, seq_len, vocab_size]
    target_probs: [batch_size, seq_len, vocab_size] 
    """
    # 1. 確保 logits 轉為 log 空間
    log_pred_probs = F.log_softmax(logits, dim=-1)
    
    # 2. 使用 'sum' 再手動除以所有元素的總數 (B * T)
    # 或是維持 'batchmean' 但要再除以序列長度
    kl_sum = F.kl_div(log_pred_probs, target_probs, reduction='sum')
    
    # 計算平均每個 token 的 KL
    batch_size, seq_len, _ = logits.size()
    avg_kl = kl_sum / (batch_size * seq_len)
    
    return avg_kl.item()
