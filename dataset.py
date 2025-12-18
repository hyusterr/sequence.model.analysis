import torch
from torch.utils.data import IterableDataset
import torch.distributions as dist
import numpy as np

class SyntheticGenerator:
    def __init__(self, vocab_size, block_size):
        self.vocab_size = vocab_size # number of unique tokens
        self.block_size = block_size # length of a sample
    def generate(self):
        raise NotImplementedError

class IIDGenerator(SyntheticGenerator):
    def generate(self):
        return torch.randint(0, self.vocab_size, (self.block_size + 1,))

class MarkovGenerator(SyntheticGenerator):
    """
    通用隨機 Markov Generator。
    不再假設 i -> i+1。而是隨機採樣一個 Transition Matrix。
    
    參數:
        alpha (float): Dirichlet 分佈的濃度參數 (Concentration Parameter)。
            - alpha < 1.0 (例如 0.1): 極度稀疏。每個狀態只會跳去極少數的特定狀態 (Deterministic)。
            - alpha = 1.0: 均勻隨機。每個轉移機率都是隨機的 (Uniform over simplex)。
            - alpha > 1.0: 趨向平均。每個狀態跳去哪裡的機率都差不多 (High Entropy)。
    """
    def __init__(self, vocab_size, block_size, alpha=0.1):
        super().__init__(vocab_size, block_size)
        
        # 1. 使用 Dirichlet 分佈生成隨機的 Transition Matrix
        # 形狀: (vocab_size, vocab_size)
        # 每一列 (Row i) 代表 P(Next | Curr=i)，且總和為 1
        
        # 設定 Dirichlet 的參數向量 (全為 alpha)
        dirichlet_params = torch.full((vocab_size,), alpha)
        
        # 採樣矩陣
        # PyTorch 的 Dirichlet 可以接受 batch_shape，所以我們生成 vocab_size 個分佈
        sampler = dist.Dirichlet(dirichlet_params)
        self.trans_mat = sampler.sample((vocab_size,))
        
        # 除錯用：印出矩陣的 Entropy 平均值，讓你知道這個規則有多難
        avg_entropy = -(self.trans_mat * torch.log(self.trans_mat + 1e-9)).sum(dim=1).mean()
        print(f"Initialized General Markov Matrix with alpha={alpha}")
        print(f"Average Row Entropy: {avg_entropy:.4f} (Theoretical Loss Lower Bound)")

    def generate(self):
        seq = [torch.randint(0, self.vocab_size, (1,)).item()]
        for _ in range(self.block_size):
            prev = seq[-1]
            
            # 根據當前狀態的 Row 進行多項式分佈採樣
            probs = self.trans_mat[prev]
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
            
        return torch.tensor(seq, dtype=torch.long)


class HMMLDAGenerator(SyntheticGenerator):
    """
    Strict implementation of HMM-LDA generative process (Figure 1).
    
    Structure:
    - n_states: HMM 總狀態數。
    - topic_state_idx: 設定哪一個狀態是 s_topic (通常設為最後一個)。
    - n_topics: LDA 的主題數。
    """
    def __init__(self, vocab_size, block_size, n_states=5, n_topics=3, alpha_lda=0.5):
        super().__init__(vocab_size, block_size)
        self.n_states = n_states
        self.n_topics = n_topics
        self.topic_state_idx = n_states - 1 # 設定最後一個狀態為 "Topic State"
        
        # --- 1. HMM Parameters (Syntactic Backbone) ---
        # pi: Transition Matrix (s_{i-1} -> s_i)
        # 讓狀態之間有轉移規則，例如 冠詞 -> 名詞
        self.trans_mat = torch.rand(n_states, n_states)
        self.trans_mat = self.trans_mat / self.trans_mat.sum(dim=1, keepdim=True)
        
        # gamma: Syntactic Emission (s_i -> w_i when s != s_topic)
        # 這些是 "HMM 字典"，通常只包含 Function words
        # 為了模擬真實情況，我們限制這些狀態只能生成 vocab 的前 20 個字
        self.syntactic_emission = torch.zeros(n_states, vocab_size)
        n_func_words = min(20, vocab_size // 2)
        self.syntactic_emission[:, :n_func_words] = torch.rand(n_states, n_func_words)
        self.syntactic_emission = self.syntactic_emission / self.syntactic_emission.sum(dim=1, keepdim=True)
        
        # --- 2. LDA Parameters (Semantic Content) ---
        # beta: Topic Word Distributions (z -> w)
        # 這些是 "LDA 字典"，通常包含 Content words
        self.topic_emission = torch.zeros(n_topics, vocab_size)
        # 讓不同 Topic 偏好不同的字 (從 n_func_words 之後開始選)
        self.topic_emission[:, n_func_words:] = torch.rand(n_topics, vocab_size - n_func_words)
        # 增加區別度：讓 Topic k 偏好特定的區間
        section = (vocab_size - n_func_words) // n_topics
        for k in range(n_topics):
            start = n_func_words + k * section
            end = start + section
            self.topic_emission[k, start:end] += 5.0 # Boost
            
        self.topic_emission = self.topic_emission / self.topic_emission.sum(dim=1, keepdim=True)
        
        # Dirichlet Parameter for document topic distribution
        self.alpha_vec = torch.full((n_topics,), alpha_lda)

    def generate(self):
        # --- Step 1: Document Level ---
        # Draw topic weights theta^d from Dirichlet(alpha)
        theta_d = dist.Dirichlet(self.alpha_vec).sample() # 每一個文章有一個 distribution on topic
        
        # --- Step 2: Sequence Level ---
        seq = []
        hidden_states = [] # s_i sequence (for analysis)
        topics = []        # z_i sequence (for analysis)
        
        # Initial State
        current_state = np.random.choice(self.n_states)
        
        # Loop for words
        for _ in range(self.block_size):
            # b. Draw state s_i from Multinomial(pi^{s_{i-1}})
            # (HMM Transition)
            probs_s = self.trans_mat[current_state]
            current_state = torch.multinomial(probs_s, 1).item()
            hidden_states.append(current_state)
            
            # a. Draw topic z_i from Multinomial(theta^d)
            # (注意：圖中 z_i 是針對每個字抽的)
            z_i = torch.multinomial(theta_d, 1).item()
            topics.append(z_i) # 每個字會抽到一個 topic
            
            # c. Draw word w_i
            if current_state == self.topic_state_idx:
                # Case: s_i = s_topic -> Use LDA Dictionary (beta^{z_i})
                word_probs = self.topic_emission[z_i]
            else:
                # Case: s_i != s_topic -> Use HMM Dictionary (gamma^{s_i})
                word_probs = self.syntactic_emission[current_state]
                
            w_i = torch.multinomial(word_probs, 1).item()
            seq.append(w_i)
            
        # 回傳完整資訊以供分析
        return {
            'x': torch.tensor(seq, dtype=torch.long),
            's': torch.tensor(hidden_states, dtype=torch.long), # HMM States
            'z': torch.tensor(topics, dtype=torch.long),        # Topics
            'theta': theta_d                                    # Doc Topic Dist
        }


class EdelmanMarkovGenerator(SyntheticGenerator):
    """
    Implementation based on 'Evolution of Statistical Induction Heads' (Edelman et al., 2024).
    
    Key Features:
    1. Sequence-Specific Rules: 每條 sequence 隨機生成一個 mapping (token -> next_token)。
    2. Alpha Parameter: 控制 statistical dependency 的強度。
    
    Params:
        alpha (float): Probability of following the bigram rule. 
                       If 1.0, strict deterministic sequence.
                       If 0.0, completely random (IID).
        consistency (bool): 
            If True: Use one Global Rule for ALL sequences (Language Modeling task).
            If False: Sample a NEW Rule for EACH sequence (In-Context Learning task).
    """
    def __init__(self, vocab_size, block_size, alpha=0.95, consistency=False):
        super().__init__(vocab_size, block_size)
        self.alpha = alpha
        self.consistency = consistency
        
        # 如果是 Global Rule (consistency=True)，我們先生成好固定的 mapping
        self.global_mapping = None
        if self.consistency:
            self.global_mapping = torch.randperm(vocab_size)

    def generate(self):
        # 1. 決定這條 sequence 的規則 (Mapping)
        # Mapping[i] = j 代表：看到 token i，下一個應該接 token j
        if self.consistency:
            mapping = self.global_mapping
        else:
            # ICL 模式：每條數據的規則都不一樣，強迫模型看 context
            mapping = torch.randperm(self.vocab_size)
            
        # 2. 生成序列
        # 先隨機選第一個字
        seq = [torch.randint(0, self.vocab_size, (1,)).item()]
        
        for _ in range(self.block_size):
            prev_token = seq[-1]
            
            # 決定是否遵守規則
            # random.random() < alpha -> 遵守
            if np.random.random() < self.alpha:
                # 查表：找出這條規則規定的下一個字
                next_token = mapping[prev_token].item()
            else:
                # 雜訊：隨機生成一個字
                next_token = torch.randint(0, self.vocab_size, (1,)).item()
                
            seq.append(next_token)
            
        return torch.tensor(seq, dtype=torch.long)


class SyntheticDataset(IterableDataset):
    def __init__(self, generator, num_samples=10000):
        self.generator = generator
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            outputs = self.generator.generate()
            
            # --- 1. 標準化輸出格式 ---
            if isinstance(outputs, torch.Tensor):
                # 簡單版 Generator (IID, Simple Markov)
                full_seq = outputs
                extra_info = {} # 沒有 Metadata
            elif isinstance(outputs, dict):
                # 進階版 Generator (HMM, HMM-LDA)
                full_seq = outputs['x']
                # 把 x 以外的東西都當作 extra info
                extra_info = {k: v for k, v in outputs.items() if k != 'x'}
            else:
                raise ValueError("Generator output type not supported")

            # --- 2. 製作 Causal LM 的 Input/Target ---
            # Input: 0 到 T-1
            # Target: 1 到 T
            x = full_seq[:-1]
            y = full_seq[1:]
            
            # --- 3. 處理 Metadata 的對齊 (關鍵!) ---
            # 如果 generator 給了每個 token 的 state (s) 或 topic (z)
            # 我們通常希望這些標籤能跟 Target (y) 對齊
            # 因為我們算 Loss 或 Entropy 是針對 "預測出來的字"
            
            aligned_extra = {}
            for k, v in extra_info.items():
                if k in ['s', 'z']: # 這些是 Sequence Level 的標籤，需要切片
                    # 確保它跟 y 一樣長，代表 "產生 y[t] 的那個 state"
                    aligned_extra[k] = v[1:] 
                else:
                    # 其他像是 theta (Document Level) 不用切
                    aligned_extra[k] = v
            
            # 回傳三個東西：輸入，目標，額外資訊
            yield x, y, aligned_extra

# Factory 更新
def get_dataset(type_name, vocab_size, block_size, **kwargs):
    if type_name == 'iid':
        gen = IIDGenerator(vocab_size, block_size)
        
    elif type_name == 'markov':
        # 這是舊版的全域 Markov
        gen = MarkovGenerator(vocab_size, block_size, **kwargs)
        
    elif type_name == 'edelman_markov':
        # 這是新版，支援 ICL 測試
        alpha = kwargs.get('alpha', 0.95)
        consistency = kwargs.get('consistency', False) # 預設為 False (ICL Mode)
        gen = EdelmanMarkovGenerator(vocab_size, block_size, alpha=alpha, consistency=consistency)
        
    elif type_name == 'hmm':
        # ... (HMM logic)
        gen = HMMGenerator(vocab_size, block_size, **kwargs)

    elif type_name == 'hmm_lda':
        gen = HMMLDAGenerator(vocab_size, block_size, **kwargs)
        
    else:
        raise ValueError(f"Unknown type: {type_name}")
    
    return SyntheticDataset(gen)
