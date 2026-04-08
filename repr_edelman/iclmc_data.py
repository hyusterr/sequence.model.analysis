import torch
from torch.utils.data import IterableDataset, Dataset
import torch.distributions as dist

class MarkovChainDataset(IterableDataset):
    """
    無限流 Dataset (Training 用)
    """
    def __init__(self, vocab_size=2, seq_len=100, batch_size=64, n=2, alpha=1.0, device='cuda'):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n = n
        self.device = device
        self.num_states = vocab_size ** max(0, n - 1)
        self.dirichlet = dist.Dirichlet(torch.ones(vocab_size, device=device) * alpha)
        if self.n > 1:
            self.mod_factor = self.vocab_size ** (self.n - 1)
        
    def generate_batch(self):
        # 1. Sample Transition Matrix P
        P = self.dirichlet.sample(torch.Size([self.batch_size, self.num_states]))
        
        # 2. Burn-in
        if self.n > 1:
            current_state = torch.randint(0, self.num_states, (self.batch_size,), device=self.device)
        else:
            current_state = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(20): # Burn-in
                probs = P[torch.arange(self.batch_size), current_state]
                token = dist.Categorical(probs).sample()
                if self.n > 1:
                    current_state = (current_state * self.vocab_size + token) % self.mod_factor
        
        # 3. Generate Sequence
        seq_tokens = []
        true_probs_list = []
        for _ in range(self.seq_len + 1):
            probs = P[torch.arange(self.batch_size), current_state]
            true_probs_list.append(probs)
            token = dist.Categorical(probs).sample()
            seq_tokens.append(token)
            if self.n > 1:
                current_state = (current_state * self.vocab_size + token) % self.mod_factor

        full_seq = torch.stack(seq_tokens, dim=1)
        
        # --- FIX: 加入 .contiguous() ---
        # 這樣可以確保傳出去的 tensor 在記憶體中是連續的
        # Model 裡面的 .view() 就不會報錯了
        x = full_seq[:, :-1].contiguous()
        y = full_seq[:, 1:].contiguous()
        
        # True Probs align with Target (y)
        true_probs_stack = torch.stack(true_probs_list, dim=1)
        true_probs = true_probs_stack[:, 1:, :].contiguous() # 注意這裡也要修正對齊並 contiguous
        
        return x, y, true_probs

    def __iter__(self):
        while True:
            yield self.generate_batch()


class FixedMarkovChainDataset(Dataset):
    """
    固定數據集 (Validation/Testing 用)
    """
    def __init__(self, size=50000, vocab_size=2, seq_len=100, n=2, alpha=1.0, device='cpu'):
        print(f"Generating Fixed Test Set (Size: {size})...")
        self.x_list = []
        self.y_list = []
        self.probs_list = []
        
        temp_batch_size = 1000 
        generator = MarkovChainDataset(vocab_size, seq_len, temp_batch_size, n, alpha, device)
        
        with torch.no_grad():
            for _ in range(size // temp_batch_size):
                x, y, p = generator.generate_batch()
                self.x_list.append(x.cpu())
                self.y_list.append(y.cpu())
                self.probs_list.append(p.cpu())
        
        self.x = torch.cat(self.x_list, dim=0)
        self.y = torch.cat(self.y_list, dim=0)
        self.probs = torch.cat(self.probs_list, dim=0)
        
        print(f"Fixed Test Set Generated. Shape: {self.x.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.probs[idx]
