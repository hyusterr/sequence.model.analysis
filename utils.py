import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Part A: 理論值計算 (Theoretical Bounds)
# ==========================================

def compute_theoretical_entropy_rate(generator):
    """
    計算 Markov Chain 的理論 Entropy Rate (Loss 下界)。
    適用於: MarkovGenerator, GeneralMarkovGenerator
    公式: H(X) = - sum_i pi_i * (sum_j P_ij log P_ij)
    """
    if not hasattr(generator, 'trans_mat'):
        return None
        
    P = generator.trans_mat.float() # (V, V)
    
    # 1. 計算 Stationary Distribution (pi)
    # P^100 收斂法
    P_limit = torch.matrix_power(P, 100)
    pi = P_limit[0]
    pi = pi / pi.sum()
    
    # 2. 計算 Row Entropies
    eps = 1e-9
    row_entropies = - (P * torch.log(P + eps)).sum(dim=1)
    
    # 3. 加權平均
    entropy_rate = (pi * row_entropies).sum().item()
    return entropy_rate

def compute_oracle_loss(x, y, transition_matrix):
    """
    計算當下 Batch 的 Oracle Loss (基於真實轉移矩陣)。
    這是 Training 時計算 Regret 用的。
    """
    # transition_matrix: (V, V)
    # x: (B, T) - Previous Token
    # y: (B, T) - Target Token
    
    V = transition_matrix.shape[0]
    device = x.device
    
    # 使用 Embedding 技巧快速查表: Input x -> Output Probability Distribution
    oracle_emb = torch.nn.Embedding.from_pretrained(transition_matrix.to(device), freeze=True)
    true_dists = oracle_emb(x) # (B, T, V)
    
    # 取出 target y 的真實機率
    # gather index 必須是 (B, T, 1)
    true_probs = true_dists.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    
    # NLL
    eps = 1e-9
    oracle_loss = -torch.log(true_probs + eps).mean()
    return oracle_loss

# ==========================================
# Part B: Attention 分析 (Post-Training)
# ==========================================

def compute_shannon_entropy(att_weights):
    """
    [IID Metric] 計算 Attention 的集中度。
    Low Entropy = Sharp Attention (專注)
    High Entropy = Uniform Attention (迷茫)
    """
    eps = 1e-9
    # sum over key dimension (last dim)
    entropy = - (att_weights * torch.log(att_weights + eps)).sum(dim=-1)
    return entropy.mean(dim=[0, 1]) # Average over Batch & Time -> (n_heads,)

def compute_markov_alignment_score(att_weights, k=1):
    """
    [Markov Metric] 計算有多少比例的 Attention 落在 t-k 上。
    k=1 代表關注前一個字 (Previous Token)。
    """
    B, nh, T, _ = att_weights.size()
    score_sum = 0
    valid_count = 0
    
    for t in range(k, T):
        # 取出對角線偏移 k 的位置
        val = att_weights[:, :, t, t-k] # (B, nh)
        score_sum += val.sum(dim=0)
        valid_count += B
        
    return score_sum / valid_count # (n_heads,)

def compute_state_attention_entropy(att_weights, hidden_states, n_states):
    """
    [HMM/LDA Metric] 計算 Attention 在「隱藏狀態」上的 Entropy。
    衡量模型是否學會了 "Syntax-aware" 或 "Topic-aware" 的注意力。
    
    att_weights: (B, nh, T, T)
    hidden_states: (B, T) - Ground Truth States/Topics
    n_states: 狀態總數
    """
    B, nh, T, _ = att_weights.size()
    device = att_weights.device
    
    # 1. 聚合 Attention (Aggregate by State)
    # 目標: (B, nh, T, n_states)
    state_att_probs = torch.zeros((B, nh, T, n_states), device=device)
    
    # 將 hidden_states 擴展為 index map
    # target_s: (B, 1, 1, T) -> (B, nh, T, T)
    target_s = hidden_states.view(B, 1, 1, T).expand(B, nh, T, T)
    
    # scatter_add: 把 att_weights 加到對應的 state 籃子裡
    state_att_probs.scatter_add_(dim=-1, index=target_s, src=att_weights)
    
    # 2. 計算 Entropy over States
    eps = 1e-9
    state_entropy = - (state_att_probs * torch.log(state_att_probs + eps)).sum(dim=-1)
    
    return state_entropy.mean(dim=[0, 1]) # (n_heads,)

class AttentionAnalyzer:
    """整合分析器"""
    def __init__(self):
        pass
        
    def analyze(self, att_layers, extra_info=None, config=None):
        """
        全面分析每一層的 Attention 行為。
        extra_info: 包含 's' (states) 或 'z' (topics) 的 dict
        """
        report = {
            'shannon_entropy': [],
            'markov_score_t1': [],
            'syntax_entropy': [], # For HMM
            'topic_entropy': []   # For HMM-LDA
        }
        
        # 取得 Ground Truth (如果有的話)
        gt_states = extra_info.get('s') if extra_info else None
        gt_topics = extra_info.get('z') if extra_info else None
        
        for i, att in enumerate(att_layers):
            # 1. Basic Metrics
            report['shannon_entropy'].append(compute_shannon_entropy(att).cpu().numpy())
            report['markov_score_t1'].append(compute_markov_alignment_score(att, k=1).cpu().numpy())
            
            # 2. HMM Syntax Analysis (if 's' is present)
            if gt_states is not None:
                # 假設 config 裡有 n_states，否則預設 5
                n_states = getattr(config, 'n_states', 5) 
                st_ent = compute_state_attention_entropy(att, gt_states, n_states)
                report['syntax_entropy'].append(st_ent.cpu().numpy())
                
            # 3. HMM-LDA Topic Analysis (if 'z' is present)
            if gt_topics is not None:
                # 假設 config 裡有 n_topics，否則預設 3
                n_topics = getattr(config, 'n_topics', 3)
                tp_ent = compute_state_attention_entropy(att, gt_topics, n_topics)
                report['topic_entropy'].append(tp_ent.cpu().numpy())
                
        return report

    def print_report(self, report):
        print("\n=== Attention Analysis Report ===")
        n_layers = len(report['shannon_entropy'])
        
        for i in range(n_layers):
            print(f"\n[Layer {i}]")
            print(f"  Shannon Entropy : {np.round(report['shannon_entropy'][i], 3)} (Low=Sharp)")
            print(f"  Markov Score(t-1): {np.round(report['markov_score_t1'][i], 3)} (High=Local)")
            
            if len(report['syntax_entropy']) > 0:
                print(f"  Syntax Entropy  : {np.round(report['syntax_entropy'][i], 3)} (Low=Syntax-Aware)")
                
            if len(report['topic_entropy']) > 0:
                print(f"  Topic Entropy   : {np.round(report['topic_entropy'][i], 3)} (Low=Topic-Aware)")
