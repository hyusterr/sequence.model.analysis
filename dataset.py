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


      


# ==========================================
# 3. ICLHMMDataset (Edelman et al. 風格的動態 HMM)
# ==========================================
class ICLHMMDataset(SequenceDataset):
    """
    與 HMMDataset 不同在於，每次 generate_sequence 都重新抽樣 A 與 B。
    這用來測試模型是否能 In-context 學會隱藏狀態的轉移邏輯。
    """
    def __init__(self, seq_len, num_hidden, num_obs, n_order=1, virtual_size=10000):
        super().__init__(seq_len, num_obs, n_order, virtual_size)
        self.num_hidden = num_hidden
        self.hidden_powers = num_hidden ** torch.arange(n_order - 1, -1, -1)

    def generate_sequence(self):
        # 1. 每次動態生成參數
        A = dist.Dirichlet(torch.ones(self.num_hidden)).sample((self.num_hidden**self.n_order,))
        B = dist.Dirichlet(torch.ones(self.num_symbols)).sample((self.num_hidden,))
        stat_h = get_stationary_distribution(A, self.n_order, self.num_hidden)

        z_seq = torch.zeros(self.seq_len + 1, dtype=torch.long)
        init_idx = torch.multinomial(stat_h, 1).item()
        # 初始化 Z
        for k in range(self.n_order - 1, -1, -1):
            z_seq[k] = init_idx % self.num_hidden
            init_idx //= self.num_hidden
        # Z 序列轉移
        for t in range(self.n_order, self.seq_len + 1):
            idx = get_index_from_history(z_seq[t-self.n_order:t], self.hidden_powers)
            z_seq[t] = torch.multinomial(A[idx], 1).item()
        
        # 2. 生成觀測序列 X
        return torch.multinomial(B[z_seq], 1).squeeze()

# ==========================================
# 4. GINCDataset (Xie et al. ICLR 2022)
# ==========================================
class GINCDataset(SequenceDataset):
    """
    GINC (Generative In-context) 數據集。
    概念：先抽樣一個隱含的『概念 (Concept)』，該概念決定了整條序列的 HMM 參數。
    """
    def __init__(self, seq_len, num_concepts, num_hidden, num_obs, virtual_size=10000):
        super().__init__(seq_len, num_obs, 1, virtual_size)
        self.num_concepts = num_concepts
        self.num_hidden = num_hidden
        
        # 預先生成數個不同的『概念背景』(固定轉移矩陣的集合)
        self.concept_A = dist.Dirichlet(torch.ones(num_hidden)).sample((num_concepts, num_hidden))
        self.concept_B = dist.Dirichlet(torch.ones(num_obs)).sample((num_concepts, num_hidden))

    def generate_sequence(self):
        # 1. 隨機選取一個概念 θ
        c_idx = torch.randint(0, self.num_concepts, (1,)).item()
        A, B = self.concept_A[c_idx], self.concept_B[c_idx]
        
        # 2. 按照該概念生成 HMM 序列 (此處假設 n_order=1)
        z = torch.randint(0, self.num_hidden, (1,)).item()
        x_seq = []
        for _ in range(self.seq_len + 1):
            x_seq.append(torch.multinomial(B[z], 1).item())
            z = torch.multinomial(A[z], 1).item()
        return torch.tensor(x_seq)

# ==========================================
# 5. HMMLDADataset (Hsu et al. EMNLP 2026)
# ==========================================
class HMMLDADataset(SequenceDataset):
    """
    結合 HMM 的結構與 LDA 的主題混合。
    狀態轉移 (HMM) 決定當前的『主題 (Topic)』，主題決定詞彙分佈。
    """
    def __init__(self, seq_len, num_topics, vocab_size, alpha=0.1, virtual_size=10000):
        super().__init__(seq_len, vocab_size, 1, virtual_size)
        self.num_topics = num_topics
        
        # 轉移矩陣 (Topic to Topic)
        self.topic_transition = dist.Dirichlet(torch.ones(num_topics)).sample((num_topics,))
        # 詞彙分佈 (Topic to Word) - LDA 核心
        self.topic_word_dist = dist.Dirichlet(alpha * torch.ones(vocab_size)).sample((num_topics,))

    def generate_sequence(self):
        topic = torch.randint(0, self.num_topics, (1,)).item()
        x_seq = []
        for _ in range(self.seq_len + 1):
            # 1. 根據當前主題抽樣單詞 (LDA 過程)
            x_seq.append(torch.multinomial(self.topic_word_dist[topic], 1).item())
            # 2. 轉移至下一個主題 (HMM 過程)
            topic = torch.multinomial(self.topic_transition[topic], 1).item()
        return torch.tensor(x_seq)




# HMM-LDA (generator)


# HMM-LDA (fitter to a real data)


# GINC dataset from Xie et al. ICLR



