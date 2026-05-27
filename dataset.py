import torch
from torch.utils.data import Dataset
import torch.distributions as dist


def batched_stationary_distribution(P: torch.Tensor, n_order: int, num_symbols: int, steps: int = 50) -> torch.Tensor:
    """
    批次計算轉移矩陣的平穩分佈。
    P 的形狀應為: [Batch_size, num_states, num_symbols]
    """
    B = P.size(0)
    num_states = num_symbols ** n_order
    
    if n_order == 1:
        T = P
    else:
        # 建立大轉移矩陣 T [B, S^N, S^N]
        T = torch.zeros((B, num_states, num_states), device=P.device)
        for i in range(num_states):
            base_idx = (i % (num_symbols ** (n_order - 1))) * num_symbols
            for j in range(num_symbols):
                T[:, i, base_idx + j] = P[:, i, j]
                
    # 冪次法逼近 (torch.linalg.matrix_power 支援批次運算)
    T_n = torch.linalg.matrix_power(T, steps)
    return T_n[:, 0] # 回傳每個 batch 的平穩分佈


# ==========================================
# Parent: 負責通用的介面與批次切分邏輯
# ==========================================
class SequenceDataset(Dataset):
    def __init__(self, seq_len: int, num_symbols: int, n_order: int = 1, virtual_size: int = 10000):
        super().__init__()
        self.seq_len = seq_len # 實際序列長度 (不包含初始狀態), T=100
        self.num_symbols = num_symbols # 狀態空間大小 (符號數量), K=2 or 3
        self.n_order = n_order # 馬可夫鏈階數, N=1 是一階, N=2 是二階
        self.virtual_size = virtual_size # 虛擬資料集大小 (實際上是無限的, 這裡只是為了 DataLoader 的迭代次數)
        
        self.powers = self.num_symbols ** torch.arange(self.n_order - 1, -1, -1)
        self.dirichlet = dist.Dirichlet(torch.ones(self.num_symbols)) # initial Dirichlet distribution for sampling transition probabilities

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        # 兼容單筆讀取，直接封裝成 batch=1 再解開
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        raise NotImplementedError("子類別必須實作批次向量化 __getitems__")


# ==========================================
# 1. MarkovChainDataset (固定設定)
# ==========================================
class MarkovChainDataset(SequenceDataset):
    '''
    固定設定的馬可夫鏈資料集，P 在初始化時抽樣一次，整個資料集使用同一個 P。
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_states = self.num_symbols ** self.n_order
        self.P = self.dirichlet.sample((num_states,))
        # 透過 batch_size=1 取得平穩分佈
        self.stationary_prob = batched_stationary_distribution(self.P.unsqueeze(0), self.n_order, self.num_symbols).squeeze(0)

    def __getitems__(self, indices):
        B = len(indices)
        info_list = [{}] * B
        
        seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        true_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_symbols))
        
        # 批次初始狀態
        init_idx = torch.multinomial(self.stationary_prob.expand(B, -1), 1).squeeze(-1)
        
        for k in range(self.n_order - 1, -1, -1):
            seq[:, k] = init_idx % self.num_symbols
            init_idx //= self.num_symbols
            true_probs_seq[:, k] = 1.0 / self.num_symbols 
            
        for t in range(self.n_order, self.seq_len + 1):
            idx = (seq[:, t-self.n_order:t] * self.powers).sum(dim=1)
            current_probs = self.P[idx]
            true_probs_seq[:, t] = current_probs 
            seq[:, t] = torch.multinomial(current_probs, 1).squeeze(-1)
            
        return list(zip(seq[:, :-1], seq[:, 1:], true_probs_seq[:, :-1], info_list))


# ==========================================
# 2. ICLMarkovChainDataset (動態設定)
# ==========================================
class ICLMarkovChainDataset(SequenceDataset):
    def __getitems__(self, indices):
        B = len(indices)
        info_list = [{}] * B
        num_states = self.num_symbols ** self.n_order
        
        # 批次動態抽樣 P
        P = self.dirichlet.sample((B, num_states))
        stationary_prob = batched_stationary_distribution(P, self.n_order, self.num_symbols)
        
        seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        true_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_symbols))
        
        init_idx = torch.multinomial(stationary_prob, 1).squeeze(-1)
        for k in range(self.n_order - 1, -1, -1):
            seq[:, k] = init_idx % self.num_symbols
            init_idx //= self.num_symbols
            true_probs_seq[:, k] = 1.0 / self.num_symbols 

        batch_idx = torch.arange(B)
        for t in range(self.n_order, self.seq_len + 1):
            idx = (seq[:, t-self.n_order:t] * self.powers).sum(dim=1)
            current_probs = P[batch_idx, idx]
            true_probs_seq[:, t] = current_probs 
            seq[:, t] = torch.multinomial(current_probs, 1).squeeze(-1)
            
        return list(zip(seq[:, :-1], seq[:, 1:], true_probs_seq[:, :-1], info_list))


# ==========================================
# 3. HMMDataset (固定隱藏狀態與觀測值)
# ==========================================
class HMMDataset(SequenceDataset):
    def __init__(self, seq_len: int, num_hidden: int, num_obs: int, n_order: int = 1, virtual_size: int = 10000):
        super().__init__(seq_len, num_obs, n_order, virtual_size)
        self.num_hidden = num_hidden
        self.num_obs = num_obs
        
        hidden_states_total = num_hidden ** n_order
        self.A = dist.Dirichlet(torch.ones(num_hidden)).sample((hidden_states_total,))
        self.B = dist.Dirichlet(torch.ones(num_obs)).sample((num_hidden,))
        
        self.hidden_powers = self.num_hidden ** torch.arange(self.n_order - 1, -1, -1)
        self.stationary_hidden = batched_stationary_distribution(self.A.unsqueeze(0), self.n_order, self.num_hidden).squeeze(0)

    def __getitems__(self, indices):
        B = len(indices)
        
        # 用來記錄真實走過的隱藏路徑
        z_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        
        # 🌟 1. 新增：用來記錄上帝視角的 Next Token Probability 分佈
        oracle_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_obs))
        
        # 初始狀態抽樣
        init_idx = torch.multinomial(self.stationary_hidden.expand(B, -1), 1).squeeze(-1)
        for k in range(self.n_order - 1, -1, -1):
            z_seq[:, k] = init_idx % self.num_hidden
            init_idx //= self.num_hidden
            
            # 初始階段的 Oracle 機率：平穩分佈 [B, num_hidden] 乘上 發射矩陣 [num_hidden, num_obs]
            oracle_probs_seq[:, k] = torch.matmul(self.stationary_hidden.expand(B, -1), self.B)

        # 遞迴生成 Z 序列與記錄 Oracle 機率
        for t in range(self.n_order, self.seq_len + 1):
            idx = (z_seq[:, t-self.n_order:t] * self.hidden_powers).sum(dim=1)
            
            current_trans_probs = self.A[idx] # Shape: [B, num_hidden]
            
            # 🌟 2. 核心：計算 P(X_t | Z_{<t}) = Trans @ Emission
            # [B, num_hidden] @ [num_hidden, num_obs] -> [B, num_obs]
            oracle_probs_seq[:, t] = torch.matmul(current_trans_probs, self.B)
            
            # 決定下一步真正走到的隱藏狀態 Z_t
            z_seq[:, t] = torch.multinomial(current_trans_probs, 1).squeeze(-1)
        
        # 3. 根據真正發生的 Z_t，生成真實的觀測值 X_t (這部分維持不變，因為物理現實是從確定的 Z 發射 X)
        emission_probs_seq = self.B[z_seq] # 這是 P(X_t | Z_t)
        flat_probs = emission_probs_seq.view(-1, self.num_obs)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
        # 4. 對齊與回傳
        x = x_seq[:, :-1]  # 模型輸入
        y = x_seq[:, 1:]   # 模型預測目標 (Target)
        
        # 🌟 5. 修正對齊：預測 y 的機率，就是時間點 1 到結尾的 Oracle 機率
        p_true = oracle_probs_seq[:, 1:] 
        
        # 你依然可以把原先確定的 Emission 或 Z 路徑封裝進字典，用於 Attention / Probing 分析
        info_list = [
            {
                "z_states": z,                      # Shape: [seq_len]
                "realized_emission_probs": em       # Shape: [seq_len, num_obs]
            } 
            for z, em in zip(z_seq[:, :-1], emission_probs_seq[:, :-1])
        ]
        
        return list(zip(x, y, p_true, info_list))



# ==========================================
# 4. ICLHMMDataset (動態 HMM)
# ==========================================
class ICLHMMDataset(SequenceDataset):
    def __init__(self, seq_len, num_hidden, num_obs, n_order=1, virtual_size=10000):
        # 注意：這裡 num_symbols 在 Dataset 內傳遞的是觀測值空間 num_obs
        super().__init__(seq_len, num_obs, n_order, virtual_size)
        self.num_hidden = num_hidden
        self.hidden_powers = self.num_hidden ** torch.arange(n_order - 1, -1, -1)

    def __getitems__(self, indices):
        B = len(indices)
        
        # 1. 動態抽樣該 Batch 的參數
        A = dist.Dirichlet(torch.ones(self.num_hidden)).sample((B, self.num_hidden**self.n_order))
        B_mat = dist.Dirichlet(torch.ones(self.num_symbols)).sample((B, self.num_hidden))
        stat_h = batched_stationary_distribution(A, self.n_order, self.num_hidden)
        
        z_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        
        # 🌟 新增：Oracle 預測分佈張量 [B, seq_len+1, num_symbols]
        oracle_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_symbols))
        
        # 初始狀態生成
        init_idx = torch.multinomial(stat_h, 1).squeeze(-1)
        for k in range(self.n_order - 1, -1, -1):
            z_seq[:, k] = init_idx % self.num_hidden
            init_idx //= self.num_hidden
            
            # Oracle 初始分佈：stationary @ Emission
            # B_mat 形狀 [B, num_hidden, num_symbols]
            oracle_probs_seq[:, k] = torch.matmul(stat_h, B_mat)

        batch_idx = torch.arange(B)
        
        # 遞迴生成 Z 序列與 Oracle 預測
        for t in range(self.n_order, self.seq_len + 1):
            idx = (z_seq[:, t-self.n_order:t] * self.hidden_powers).sum(dim=1)
            current_A = A[batch_idx, idx] # [B, num_hidden]
            
            # 🌟 Oracle 核心：P(X_t | Z_{<t}) = A @ B
            oracle_probs_seq[:, t] = torch.matmul(current_A, B_mat)
            
            # 實際路徑採樣
            z_seq[:, t] = torch.multinomial(current_A, 1).squeeze(-1)
            
        # 生成實際觀測序列 X
        true_probs_seq = B_mat[batch_idx.unsqueeze(1), z_seq] 
        flat_probs = true_probs_seq.view(-1, self.num_symbols)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
        # 2. 封裝資訊
        # p_true 設為 oracle_probs_seq[:, 1:] 以對齊預測目標 y
        p_true = oracle_probs_seq[:, 1:]
        
        info_list = [
            {
                "z_states": z,
                "A_matrix": a,
                "B_matrix": b,
                "realized_emission_probs": em
            }
            for z, a, b, em in zip(z_seq[:, :-1], A, B_mat, true_probs_seq[:, :-1])
        ]
        
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], p_true, info_list))

# ==========================================
# 5. GINCDataset (Xie et al. 2022)
# ==========================================
class GINCDataset(SequenceDataset):
    def __init__(self, seq_len, num_concepts, num_hidden, num_obs, virtual_size=10000):
        super().__init__(seq_len, num_obs, 1, virtual_size)
        self.num_concepts = num_concepts
        self.num_hidden = num_hidden
        
        self.concept_A = dist.Dirichlet(torch.ones(num_hidden)).sample((num_concepts, num_hidden))
        self.concept_B = dist.Dirichlet(torch.ones(num_obs)).sample((num_concepts, num_hidden))

    def __getitems__(self, indices):
        B = len(indices)
        c_idx = torch.randint(0, self.num_concepts, (B,))
        
        A = self.concept_A[c_idx] 
        B_mat = self.concept_B[c_idx] 
        
        z_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        z_seq[:, 0] = torch.randint(0, self.num_hidden, (B,))
        
        batch_idx = torch.arange(B)
        for t in range(1, self.seq_len + 1):
            z_seq[:, t] = torch.multinomial(A[batch_idx, z_seq[:, t-1]], 1).squeeze(-1)
            
        true_probs_seq = B_mat[batch_idx.unsqueeze(1), z_seq] 
        flat_probs = true_probs_seq.view(-1, self.num_symbols)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], true_probs_seq[:, :-1]))


# ==========================================
# 6. HMMLDADataset (Hsu et al. EMNLP 2026)
# ==========================================
class HMMLDADataset(SequenceDataset):
    def __init__(self, seq_len, num_topics, vocab_size, alpha=0.1, virtual_size=10000):
        super().__init__(seq_len, vocab_size, 1, virtual_size)
        self.num_topics = num_topics
        
        self.topic_transition = dist.Dirichlet(torch.ones(num_topics)).sample((num_topics,))
        self.topic_word_dist = dist.Dirichlet(alpha * torch.ones(vocab_size)).sample((num_topics,))
        self.stationary_topic = batched_stationary_distribution(self.topic_transition.unsqueeze(0), 1, self.num_topics).squeeze(0)

    def __getitems__(self, indices):
        B = len(indices)
        topic_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        
        topic_seq[:, 0] = torch.multinomial(self.stationary_topic.expand(B, -1), 1).squeeze(-1)
        
        for t in range(1, self.seq_len + 1):
            topic_seq[:, t] = torch.multinomial(self.topic_transition[topic_seq[:, t-1]], 1).squeeze(-1)
            
        true_probs_seq = self.topic_word_dist[topic_seq]
        flat_probs = true_probs_seq.view(-1, self.num_symbols)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], true_probs_seq[:, :-1]))
