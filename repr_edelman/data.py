import torch
from torch.utils.data import IterableDataset
import torch.distributions as dist

class MarkovChainDataset(IterableDataset):
    def __init__(self, vocab_size=16, seq_len=64, batch_size=32, alpha=1.0, device='cpu'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = device
        self.dirichlet = dist.Dirichlet(torch.ones(vocab_size, device=device) * alpha)

    def get_stationary_distribution(self, P):
        # 使用矩陣快速冪計算穩態分佈
        P_inf = torch.linalg.matrix_power(P, 128)
        return P_inf.mean(dim=-2)

    def generate_batch(self):
        # 1. Sample Transition Matrix P: (B, K, K)
        P = self.dirichlet.sample(torch.Size([self.batch_size, self.vocab_size]))
        
        # 2. Get Initial Distribution and Sample First Token
        pi = self.get_stationary_distribution(P)
        current_token = dist.Categorical(pi).sample()
        
        seq = [current_token]
        true_probs_seq = [] # 儲存每一步的「真實條件機率」
        
        # 3. Generate Sequence
        for _ in range(self.seq_len):
            # 根據當前 token 查表得到「真實的下一個 token 機率分佈」
            # probs shape: (B, K)
            probs = P[torch.arange(self.batch_size), current_token]
            
            # 存下來做為 Ground Truth (Oracle)
            true_probs_seq.append(probs)
            
            # Sample Next Token
            current_token = dist.Categorical(probs).sample()
            seq.append(current_token)
            
        # Stack 起來
        x_seq = torch.stack(seq[:-1], dim=1)      # Input: 0 ~ T-1
        y_seq = torch.stack(seq[1:], dim=1)       # Target: 1 ~ T
        true_probs = torch.stack(true_probs_seq, dim=1) # Oracle Probs: 1 ~ T 的真實分佈
        
        return x_seq, y_seq, true_probs

    def __iter__(self):
        while True:
            yield self.generate_batch()
