import torch
from torch.utils.data import Dataset
import torch.distributions as dist

def batched_stationary_distribution(P: torch.Tensor, n_order: int, num_symbols: int, steps: int = 50) -> torch.Tensor:
    B = P.size(0)
    num_states = num_symbols ** n_order
    
    if n_order == 1:
        T = P
    else:
        T = torch.zeros((B, num_states, num_states), device=P.device)
        for i in range(num_states):
            base_idx = (i % (num_symbols ** (n_order - 1))) * num_symbols
            for j in range(num_symbols):
                T[:, i, base_idx + j] = P[:, i, j]
                
    T_n = torch.linalg.matrix_power(T, steps)
    return T_n[:, 0]


class SequenceDataset(Dataset):
    def __init__(self, seq_len: int, num_symbols: int, n_order: int = 1, virtual_size: int = 10000):
        super().__init__()
        self.seq_len = seq_len 
        self.num_symbols = num_symbols 
        self.n_order = n_order 
        self.virtual_size = virtual_size 
        
        self.powers = self.num_symbols ** torch.arange(self.n_order - 1, -1, -1)
        self.dirichlet = dist.Dirichlet(torch.ones(self.num_symbols)) 

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        raise NotImplementedError


# ==========================================
# 1. MarkovChainDataset (🌟 已修正對齊)
# ==========================================
class MarkovChainDataset(SequenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_states = self.num_symbols ** self.n_order
        self.P = self.dirichlet.sample((num_states,))
        self.stationary_prob = batched_stationary_distribution(self.P.unsqueeze(0), self.n_order, self.num_symbols).squeeze(0)

    def __getitems__(self, indices):
        B = len(indices)
        info_list = [{}] * B
        
        seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        true_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_symbols))
        
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
            
        # 🌟 修正：將 true_probs_seq[:, :-1] 改為 [:, 1:] 以完美對齊目標 y
        return list(zip(seq[:, :-1], seq[:, 1:], true_probs_seq[:, 1:], info_list))


# ==========================================
# 2. ICLMarkovChainDataset (🌟 已修正對齊)
# ==========================================
class ICLMarkovChainDataset(SequenceDataset):
    def __getitems__(self, indices):
        B = len(indices)
        info_list = [{}] * B
        num_states = self.num_symbols ** self.n_order
        
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
            
        # 🌟 修正：將 true_probs_seq[:, :-1] 改為 [:, 1:] 
        return list(zip(seq[:, :-1], seq[:, 1:], true_probs_seq[:, 1:], info_list))


# ==========================================
# 3. HMMDataset (原本即正確)
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
        z_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        oracle_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_obs))
        
        init_idx = torch.multinomial(self.stationary_hidden.expand(B, -1), 1).squeeze(-1)
        for k in range(self.n_order - 1, -1, -1):
            z_seq[:, k] = init_idx % self.num_hidden
            init_idx //= self.num_hidden
            oracle_probs_seq[:, k] = torch.matmul(self.stationary_hidden.expand(B, -1), self.B)

        for t in range(self.n_order, self.seq_len + 1):
            idx = (z_seq[:, t-self.n_order:t] * self.hidden_powers).sum(dim=1)
            current_trans_probs = self.A[idx] 
            oracle_probs_seq[:, t] = torch.matmul(current_trans_probs, self.B)
            z_seq[:, t] = torch.multinomial(current_trans_probs, 1).squeeze(-1)
        
        emission_probs_seq = self.B[z_seq] 
        flat_probs = emission_probs_seq.view(-1, self.num_obs)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
        info_list = [
            {"z_states": z, "realized_emission_probs": em} 
            for z, em in zip(z_seq[:, :-1], emission_probs_seq[:, :-1])
        ]
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], oracle_probs_seq[:, 1:], info_list))

# ==========================================
# 4. ICLHMMDataset (🌟 完美修復隱藏狀態維度對齊)
# ==========================================
class ICLHMMDataset(SequenceDataset):
    def __init__(self, seq_len, num_hidden, num_obs, n_order=1, virtual_size=10000):
        super().__init__(seq_len, num_obs, n_order, virtual_size)
        self.num_hidden = num_hidden
        self.hidden_powers = self.num_hidden ** torch.arange(n_order - 1, -1, -1)

    def __getitems__(self, indices):
        B = len(indices)
        
        # 🌟 關鍵修正：HMM 的狀態數由隱藏狀態數 (num_hidden) 決定，而非觀測符號數 (num_symbols)
        num_states = self.num_hidden ** self.n_order  # 修正後為 2^1 = 2
        
        # 1. 動態抽樣該 Batch 的參數
        A = dist.Dirichlet(torch.ones(self.num_hidden)).sample((B, num_states)) # 修正後形狀為 [B, 2, 2] 的完美方陣
        B_mat = dist.Dirichlet(torch.ones(self.num_symbols)).sample((B, self.num_hidden))
        
        # 傳入 self.num_hidden 作為該馬可夫鏈的基礎字母集大小
        stat_h = batched_stationary_distribution(A, self.n_order, self.num_hidden)
        
        z_seq = torch.zeros((B, self.seq_len + 1), dtype=torch.long)
        oracle_probs_seq = torch.zeros((B, self.seq_len + 1, self.num_symbols))
        
        # 初始狀態生成
        init_idx = torch.multinomial(stat_h, 1).squeeze(-1)
        for k in range(self.n_order - 1, -1, -1):
            z_seq[:, k] = init_idx % self.num_hidden
            init_idx //= self.num_hidden
            oracle_probs_seq[:, k] = torch.einsum('bh,bhd->bd', stat_h, B_mat)

        batch_idx = torch.arange(B)
        
        # 遞迴生成 Z 序列與 Oracle 預測
        for t in range(self.n_order, self.seq_len + 1):
            idx = (z_seq[:, t-self.n_order:t] * self.hidden_powers).sum(dim=1)
            current_A = A[batch_idx, idx]
            
            oracle_probs_seq[:, t] = torch.einsum('bh,bhd->bd', current_A, B_mat)
            z_seq[:, t] = torch.multinomial(current_A, 1).squeeze(-1)
            
        # 生成實際觀測序列 X
        true_probs_seq = B_mat[batch_idx.unsqueeze(1), z_seq] 
        flat_probs = true_probs_seq.view(-1, self.num_symbols)
        flat_x = torch.multinomial(flat_probs, 1).squeeze(-1)
        x_seq = flat_x.view(B, self.seq_len + 1)
        
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
# 5. GINCDataset (🌟 已修正對齊)
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
        
        # 🌟 修正：將 true_probs_seq[:, :-1] 改為 [:, 1:] 
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], true_probs_seq[:, 1:]))


# ==========================================
# 6. HMMLDADataset (🌟 已修正對齊)
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
        
        # 🌟 修正：將 true_probs_seq[:, :-1] 改為 [:, 1:] 
        return list(zip(x_seq[:, :-1], x_seq[:, 1:], true_probs_seq[:, 1:]))
