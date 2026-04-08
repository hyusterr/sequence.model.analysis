import torch
from torch.utils.data import IterableDataset, Dataset
import torch.distributions as dist

class HMMDataset(IterableDataset):
    def __init__(self, num_hidden=4, num_obs=2, seq_len=100, batch_size=64, alpha=1.0, device='cuda'):
        self.num_hidden = num_hidden
        self.num_obs = num_obs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.dirichlet_T = dist.Dirichlet(torch.ones(num_hidden, device=device) * alpha)
        self.dirichlet_E = dist.Dirichlet(torch.ones(num_obs, device=device) * alpha)

    def generate_batch(self):
        # 1. Sample HMM Parameters
        T_matrix = self.dirichlet_T.sample(torch.Size([self.batch_size, self.num_hidden]))
        E_matrix = self.dirichlet_E.sample(torch.Size([self.batch_size, self.num_hidden]))
        
        # 2. Init
        current_hidden = torch.randint(0, self.num_hidden, (self.batch_size,), device=self.device)
        belief_state = torch.ones(self.batch_size, self.num_hidden, device=self.device) / self.num_hidden

        # 3. Burn-in
        with torch.no_grad():
            for _ in range(20):
                # Real Gen
                probs_hidden = T_matrix[torch.arange(self.batch_size), current_hidden]
                current_hidden = dist.Categorical(probs_hidden).sample()
                probs_obs = E_matrix[torch.arange(self.batch_size), current_hidden]
                current_obs = dist.Categorical(probs_obs).sample()
                
                # Oracle Update
                belief_pred = torch.bmm(belief_state.unsqueeze(1), T_matrix).squeeze(1)
                E_obs_likelihood = E_matrix.gather(2, current_obs.view(-1, 1, 1).expand(-1, self.num_hidden, 1)).squeeze(2)
                belief_update = belief_pred * E_obs_likelihood
                belief_state = belief_update / (belief_update.sum(dim=-1, keepdim=True) + 1e-9)

        # 4. Generate Sequence
        seq_obs = []
        oracle_probs_list = []
        god_probs_list = [] # 新增: 上帝視角的機率

        for _ in range(self.seq_len + 1):
            # --- (A) Oracle Prediction (Predict Next Token) ---
            belief_pred = torch.bmm(belief_state.unsqueeze(1), T_matrix).squeeze(1)
            oracle_obs_dist = torch.bmm(belief_pred.unsqueeze(1), E_matrix).squeeze(1)
            oracle_probs_list.append(oracle_obs_dist)

            # --- (B) Real World Generation ---
            # Transition to Next Hidden State (z_t)
            probs_hidden = T_matrix[torch.arange(self.batch_size), current_hidden]
            current_hidden = dist.Categorical(probs_hidden).sample()
            
            # Emission from z_t (Generate x_t)
            # 這就是上帝機率：因為上帝知道 z_t，所以他知道 x_t 是從 E[z_t] 採樣的
            probs_obs = E_matrix[torch.arange(self.batch_size), current_hidden]
            god_probs_list.append(probs_obs) # 存下來
            
            current_obs = dist.Categorical(probs_obs).sample()
            seq_obs.append(current_obs)

            # --- (C) Oracle Update ---
            E_obs_likelihood = E_matrix.gather(2, current_obs.view(-1, 1, 1).expand(-1, self.num_hidden, 1)).squeeze(2)
            belief_update = belief_pred * E_obs_likelihood
            belief_state = belief_update / (belief_update.sum(dim=-1, keepdim=True) + 1e-9)

        # 5. Format
        full_seq = torch.stack(seq_obs, dim=1)
        x = full_seq[:, :-1].contiguous()
        y = full_seq[:, 1:].contiguous()
        
        # Align Probs with Targets (y)
        # oracle_probs_list[t] is prediction for seq_obs[t]
        # We want prediction for seq_obs[1:] (which is y)
        
        oracle_probs_stack = torch.stack(oracle_probs_list, dim=1)
        oracle_probs = oracle_probs_stack[:, 1:, :].contiguous()
        
        god_probs_stack = torch.stack(god_probs_list, dim=1)
        god_probs = god_probs_stack[:, 1:, :].contiguous()

        return x, y, oracle_probs, god_probs

    def __iter__(self):
        while True:
            yield self.generate_batch()

# Fixed HMM Dataset 也同步修改
class FixedHMMDataset(Dataset):
    def __init__(self, size=20000, num_hidden=4, num_obs=2, seq_len=100, alpha=1.0, device='cpu'):
        print(f"Generating Fixed HMM Test Set (Size: {size})...")
        self.x_list = []
        self.y_list = []
        self.oracle_probs_list = []
        self.god_probs_list = [] # 新增
        
        temp_batch_size = 500
        generator = HMMDataset(num_hidden, num_obs, seq_len, temp_batch_size, alpha, device)
        
        with torch.no_grad():
            for _ in range(size // temp_batch_size):
                x, y, op, gp = generator.generate_batch()
                self.x_list.append(x.cpu())
                self.y_list.append(y.cpu())
                self.oracle_probs_list.append(op.cpu())
                self.god_probs_list.append(gp.cpu())
        
        self.x = torch.cat(self.x_list, dim=0)
        self.y = torch.cat(self.y_list, dim=0)
        self.oracle_probs = torch.cat(self.oracle_probs_list, dim=0)
        self.god_probs = torch.cat(self.god_probs_list, dim=0)
        print(f"Fixed HMM Test Set Generated. Shape: {self.x.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.oracle_probs[idx], self.god_probs[idx]
