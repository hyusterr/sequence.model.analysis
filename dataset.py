import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch.distributions as dist
import numpy as np
import logging
import os

import torch
from torch.utils.data import Dataset
from data_utils import get_stationary_distribution, get_index_from_history

# ==========================================
# Parent: 負責通用的介面與切分邏輯
# ==========================================

# Parent Class for all Datasets
class SequenceDataset(Dataset):
    def __init__(self, seq_len: int, num_symbols: int, n_order: int = 1, virtual_size: int = 10000):
        super().__init__()
        self.seq_len = seq_len
        self.num_symbols = num_symbols
        self.n_order = n_order
        self.virtual_size = virtual_size
        
        # 預計算進位權重，供子類別使用
        self.powers = self.num_symbols ** torch.arange(self.n_order - 1, -1, -1)
        self.dirichlet = torch.distributions.Dirichlet(torch.ones(self.num_symbols))

    def __len__(self):
        return self.virtual_size

    def generate_sequence(self) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, idx):
        full_seq = self.generate_sequence()
        x = full_seq[:-1]
        y = full_seq[1:]
        return x, y

# ==========================================
# 1. MarkovChainDataset (ICLR 2025 固定設定)
# ==========================================

# Simplest: Markov Chain
# from Makkuva et al. ICLR 2025
# https://arxiv.org/pdf/2402.04161
 
class MarkovChainDataset(SequenceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_states = self.num_symbols ** self.n_order
        # 初始化時固定 P
        self.P = self.dirichlet.sample((num_states,))
        self.stationary_prob = get_stationary_distribution(self.P, self.n_order, self.num_symbols)

    def generate_sequence(self) -> torch.Tensor:
        seq = torch.zeros(self.seq_len + 1, dtype=torch.long)
        # 起點抽樣
        init_idx = torch.multinomial(self.stationary_prob, 1).item()
        for k in range(self.n_order - 1, -1, -1):
            seq[k] = init_idx % self.num_symbols
            init_idx //= self.num_symbols
        
        # Rollout
        for t in range(self.n_order, self.seq_len + 1):
            idx = get_index_from_history(seq[t-self.n_order:t], self.powers)
            seq[t] = torch.multinomial(self.P[idx], 1).item()
        return seq

# ==========================================
# 2. ICLMarkovChainDataset (Edelman et al. 動態設定)
# ==========================================
# ICL-MC dataset from Edelman et al. 
class ICLMarkovChainDataset(SequenceDataset):
    def generate_sequence(self) -> torch.Tensor:
        num_states = self.num_symbols ** self.n_order
        # 每次生成都重新抽樣 P
        P = self.dirichlet.sample((num_states,))
        stationary_prob = get_stationary_distribution(P, self.n_order, self.num_symbols)
        
        seq = torch.zeros(self.seq_len + 1, dtype=torch.long)
        init_idx = torch.multinomial(stationary_prob, 1).item()
        for k in range(self.n_order - 1, -1, -1):
            seq[k] = init_idx % self.num_symbols
            init_idx //= self.num_symbols

        for t in range(self.n_order, self.seq_len + 1):
            idx = get_index_from_history(seq[t-self.n_order:t], self.powers)
            seq[t] = torch.multinomial(P[idx], 1).item()
        return seq

# ==========================================
# 3. HMMDataset (隱藏狀態與觀測值)
# ==========================================
# Advanced: Hidden Markov Model
class HMMDataset(SequenceDataset):
    def __init__(self, seq_len: int, num_hidden: int, num_obs: int, n_order: int = 1, virtual_size: int = 10000):
        # 這裡的 num_symbols 對 Parent 來說是觀測值的 Vocab Size
        super().__init__(seq_len, num_obs, n_order, virtual_size)
        self.num_hidden = num_hidden
        self.num_obs = num_obs
        
        # 隱藏層轉移權重與發射權重 (採 Fixed 設定)
        hidden_states_total = num_hidden ** n_order
        self.hidden_dirichlet = torch.distributions.Dirichlet(torch.ones(num_hidden))
        self.emission_dirichlet = torch.distributions.Dirichlet(torch.ones(num_obs))
        
        self.A = self.hidden_dirichlet.sample((hidden_states_total,)) # 隱藏轉移 A
        self.B = self.emission_dirichlet.sample((num_hidden,))      # 發射矩陣 B
        
        # 隱藏層平穩分佈的 powers
        self.hidden_powers = self.num_hidden ** torch.arange(self.n_order - 1, -1, -1)
        self.stationary_hidden = get_stationary_distribution(self.A, self.n_order, self.num_hidden)

    def generate_sequence(self) -> torch.Tensor:
        # 1. 先生成隱藏狀態序列 Z
        z_seq = torch.zeros(self.seq_len + 1, dtype=torch.long)
        init_idx = torch.multinomial(self.stationary_hidden, 1).item()
        for k in range(self.n_order - 1, -1, -1):
            z_seq[k] = init_idx % self.num_hidden
            init_idx //= self.hidden_hidden

        for t in range(self.n_order, self.seq_len + 1):
            idx = get_index_from_history(z_seq[t-self.n_order:t], self.hidden_powers)
            z_seq[t] = torch.multinomial(self.A[idx], 1).item()
        
        # 2. 根據隱藏序列 Z，透過發射矩陣 B 生成觀測序列 X
        x_seq = torch.zeros(self.seq_len + 1, dtype=torch.long)
        for t in range(self.seq_len + 1):
            current_z = z_seq[t]
            x_seq[t] = torch.multinomial(self.B[current_z], 1).item()
            
        return x_seq


      







# HMM-LDA (generator)


# HMM-LDA (fitter to a real data)


# GINC dataset from Xie et al. ICLR



