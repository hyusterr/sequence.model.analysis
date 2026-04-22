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

def compute_kl_divergence(logits, inputs, transition_matrix):
    """
    計算模型預測與真實馬可夫轉移矩陣之間的 KL Divergence。
    
    logits: 模型輸出 [batch_size, seq_len, vocab_size]
    inputs: 模型的輸入序列 [batch_size, seq_len]，用來判斷當前狀態
    transition_matrix: 真實的轉移矩陣 P [vocab_size, vocab_size]
                      P[i, j] 代表從狀態 i 轉移到 j 的機率
    """
    batch_size, seq_len, vocab_size = logits.size()
    
    # 1. 取得模型的預測機率分佈 (Softmax)
    # 我們關注的是「已知當前字，預測下一個字」的機率
    pred_probs = F.softmax(logits, dim=-1) # [batch_size, seq_len, vocab_size]
    
    # 2. 取得真實的目標分佈
    # 根據 inputs 中的每一個 token ID，從 transition_matrix 中查表得到真實分佈
    # inputs 是 [batch_size, seq_len]，我們取出對應的轉移向量
    # target_probs shape: [batch_size, seq_len, vocab_size]
    target_probs = transition_matrix[inputs]
    
    # 3. 計算 KL Divergence
    # KL(P || Q) = sum(P * log(P / Q))
    # 在 PyTorch 中，F.kl_div 預期輸入為 log_probs
    log_pred_probs = F.log_softmax(logits, dim=-1)
    
    # 計算每個位置的 KL
    # reduction='batchmean' 是標準做法，但為了精確觀察，我們通常排除第一個 token (起始狀態)
    # 因為第一個 token 沒有「上一個狀態」可以依據馬可夫性質預測
    kl_div = F.kl_div(log_pred_probs[:, :-1, :], target_probs[:, :-1, :], reduction='batchmean')
    
    return kl_div.item()
