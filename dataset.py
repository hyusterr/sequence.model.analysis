import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributions as dist
import numpy as np
import logging
import os

from utils import stationary_distribution

# ==========================================
# 1. Base Class & Simple Generators
# ==========================================

class SyntheticGenerator:
    def __init__(self, vocab_size, block_size):
        self.vocab_size = vocab_size
        self.block_size = block_size
    def generate(self):
        """Should return either a Tensor (seq) or a Dict {'x': seq, ...}"""
        raise NotImplementedError

class IIDGenerator(SyntheticGenerator):
    def generate(self):
        return torch.randint(0, self.vocab_size, (self.block_size + 1,))

class HMMGenerator(SyntheticGenerator):
    """
    HMM Generator based on Xie et al. (2022) Appendix F (GINC Dataset).
    
    Structure:
    - Latent States z_t in {0, ..., n_states-1} (Concepts)
    - Observed Tokens x_t in {0, ..., vocab_size-1} (Words)
    - Transition Matrix A (n_states x n_states): P(z_t | z_{t-1})
    - Emission Matrix B (n_states x vocab_size): P(x_t | z_t)
    
    Parameters:
        n_states: Number of latent concepts (default 5 in GINC).
        trans_alpha: Dirichlet concentration for Transition Matrix.
                     - Low val (<1): Deterministic transitions (structured).
                     - High val (>1): Uniform transitions.
        emit_alpha: Dirichlet concentration for Emission Matrix.
                     - Low val (<1): Sparse emissions (each concept maps to few words).
                     - High val (>1): Noisy emissions.
        consistency (bool):
            - True: Fixed A and B for the whole dataset (Language Modeling).
            - False: Sample new A and B for EACH sequence (In-Context Learning).
    """
    def __init__(self, vocab_size, block_size, n_states=5, 
                 trans_alpha=1.0, emit_alpha=1.0, consistency=False):
        super().__init__(vocab_size, block_size)
        self.n_states = n_states
        self.trans_alpha = trans_alpha
        self.emit_alpha = emit_alpha
        self.consistency = consistency
        
        # Dirichlet Priors
        self.trans_prior = torch.full((n_states,), trans_alpha)
        self.emit_prior = torch.full((vocab_size,), emit_alpha)
        
        # Global Parameters (if consistency=True)
        self.global_A = None
        self.global_B = None
        
        if self.consistency:
            self._init_global_params()

    def _init_global_params(self):
        # Sample Transition Matrix A
        trans_sampler = dist.Dirichlet(self.trans_prior)
        self.global_A = trans_sampler.sample((self.n_states,))
        
        # Sample Emission Matrix B
        emit_sampler = dist.Dirichlet(self.emit_prior)
        self.global_B = emit_sampler.sample((self.n_states,))

    def generate(self):
        # 1. Determine Parameters for this sequence
        if self.consistency:
            A = self.global_A
            B = self.global_B
        else:
            # ICL Mode: Sample new HMM for this document
            A = dist.Dirichlet(self.trans_prior).sample((self.n_states,))
            B = dist.Dirichlet(self.emit_prior).sample((self.n_states,))
        
        # 2. Compute Stationary Distribution (pi) for x1
        # P^50 trick to find steady state
        A_limit = torch.matrix_power(A, 50)
        pi = A_limit[0]
        
        # 3. Generate Sequence
        seq_x = []
        seq_z = [] # Hidden states
        
        # Initial State
        z_t = torch.multinomial(pi, 1).item()
        seq_z.append(z_t)
        
        # Initial Emission
        x_t = torch.multinomial(B[z_t], 1).item()
        seq_x.append(x_t)
        
        for _ in range(self.block_size):
            # Transition z_{t-1} -> z_t
            probs_z = A[z_t]
            z_t = torch.multinomial(probs_z, 1).item()
            seq_z.append(z_t)
            
            # Emission z_t -> x_t
            probs_x = B[z_t]
            x_t = torch.multinomial(probs_x, 1).item()
            seq_x.append(x_t)
            
        # Return Dict
        # 注意：我們回傳 A 和 B (Oracle Params) 讓你有機會算 Oracle Loss
        return {
            'x': torch.tensor(seq_x, dtype=torch.long),
            's': torch.tensor(seq_z, dtype=torch.long), # Hidden States
            'A': A, # Transition Matrix
            'B': B  # Emission Matrix
        }

class MarkovGenerator(SyntheticGenerator):
    """
    Static Markov Chain: The Transition Matrix is fixed for the whole dataset.
    This tests "Weights Learning" (memorization).
    """
    def __init__(self, vocab_size, block_size, alpha=0.1):
        super().__init__(vocab_size, block_size)
        dirichlet_params = torch.full((vocab_size,), alpha)
        sampler = dist.Dirichlet(dirichlet_params)
        self.trans_mat = sampler.sample((vocab_size,)) # Fixed P
        
    def generate(self):
        seq = [torch.randint(0, self.vocab_size, (1,)).item()]
        for _ in range(self.block_size):
            prev = seq[-1]
            probs = self.trans_mat[prev]
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
        return torch.tensor(seq, dtype=torch.long)

class EdelmanICLMCGenerator(SyntheticGenerator):
    """
    In-Context Learning Markov Chain (ICL-MC) from Edelman et al. (2024).
    Dynamic Markov Chain: A NEW Transition Matrix is sampled for EACH sequence.
    This tests "In-Context Learning" (induction).
    """
    def __init__(self, vocab_size, block_size, alpha=1.0, n_gram=2):
        super().__init__(vocab_size, block_size) # token, sequence_len
        self.alpha = alpha # [1, ..., 1] from paper
        self.dirichlet_params = torch.full((vocab_size,), alpha)
        # make it a vector of [1, 1, ..., 1] of size vocab_size


    def generate(self):
        # 1. Sample P for THIS sequence
        # each sequence has its own Markov Chain, with P ~ Dir(alpha)
        sampler = dist.Dirichlet(self.dirichlet_params)
        P = sampler.sample((self.vocab_size,))
        
        # 2. Compute Stationary Distribution (pi) to sample x1
        # P^50 converges to stationary distribution
        P_limit = torch.matrix_power(P, 50) 
        pi = P_limit[0]
        
        seq = [torch.multinomial(pi, 1).item()]
        
        # 3. Generate sequence
        for _ in range(self.block_size):
            prev = seq[-1]
            probs = P[prev]
            next_token = torch.multinomial(probs, 1).item()
            seq.append(next_token)
            
        return {
            'x': torch.tensor(seq, dtype=torch.long),
            'P': P # Return the oracle P for this specific sequence
        }

# ==========================================
# 2. HMM-LDA Generators (Synthetic & Fitted)
# ==========================================

class HMMLDAGenerator(SyntheticGenerator):
    """
    Standard HMM-LDA Generator with synthetic random parameters.
    """
    def __init__(self, vocab_size, block_size, n_states=5, n_topics=3, alpha_lda=0.5, n_func_words=None):
        super().__init__(vocab_size, block_size)
        self.n_states = n_states
        self.n_topics = n_topics
        self.topic_state_idx = n_states - 1 
        
        # HMM Params
        self.trans_mat = torch.rand(n_states, n_states)
        self.trans_mat = self.trans_mat / self.trans_mat.sum(dim=1, keepdim=True)
        
        self.syntactic_emission = torch.zeros(n_states, vocab_size)
        if not n_func_words:
            n_func_words = min(20, vocab_size // (n_topics + 1))
            self.n_func_words = n_func_words
        else:
            self.n_func_words = n_func_words
        self.syntactic_emission[:, :n_func_words] = torch.rand(n_states, n_func_words)
        self.syntactic_emission = self.syntactic_emission / self.syntactic_emission.sum(dim=1, keepdim=True)
        
        # LDA Params
        self.topic_emission = torch.zeros(n_topics, vocab_size)
        self.topic_emission[:, n_func_words:] = torch.rand(n_topics, vocab_size - n_func_words)
        section = (vocab_size - n_func_words) // n_topics
        for k in range(n_topics):
            start = n_func_words + k * section
            end = start + section
            self.topic_emission[k, start:end] += 5.0
        self.topic_emission = self.topic_emission / self.topic_emission.sum(dim=1, keepdim=True)
        
        self.alpha_vec = torch.full((n_topics,), alpha_lda)

    def generate(self):
        theta_d = dist.Dirichlet(self.alpha_vec).sample()
        seq = []
        hidden_states = []
        topics = []
        current_state = np.random.choice(self.n_states)
        
        for _ in range(self.block_size):
            # Transition
            probs_s = self.trans_mat[current_state]
            current_state = torch.multinomial(probs_s, 1).item()
            hidden_states.append(current_state)
            
            # Topic assignment
            z_i = torch.multinomial(theta_d, 1).item()
            topics.append(z_i)
            
            # Emission
            if current_state == self.topic_state_idx:
                word_probs = self.topic_emission[z_i]
            else:
                word_probs = self.syntactic_emission[current_state]
                
            w_i = torch.multinomial(word_probs, 1).item()
            seq.append(w_i)
            
        return {
            'x': torch.tensor(seq, dtype=torch.long),
            's': torch.tensor(hidden_states, dtype=torch.long),
            'z': torch.tensor(topics, dtype=torch.long),
            'theta': theta_d
        }

# --- Gibbs Sampler Implementation for Fitting ---

def categorical(proportions):
    proportions = np.maximum(proportions, 0)
    total = np.sum(proportions)
    if total == 0: return np.random.randint(0, proportions.size)
    draw = np.random.uniform(0, total)
    for idx, cumsum in enumerate(np.cumsum(proportions)):
        if draw < cumsum: return idx
    return proportions.size - 1

class HMMLDAGibbsSampler:
    """
    Gibbs Sampler for HMM-LDA (Griffiths et al. 2004).
    NOTE: In this sampler, Class 0 is ALWAYS the Topic State.
    """
    def __init__(self, vocab_size, num_topics, num_classes, alpha=0.1, beta=0.01, gamma=0.1, delta=0.1):
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.alpha = alpha; self.beta = beta; self.gamma = gamma; self.delta = delta
        self.documents = []
        self.topic_assignments = []; self.class_assignments = []

    def add_document(self, doc):
        if isinstance(doc, torch.Tensor): doc = doc.cpu().numpy()
        self.documents.append(doc)

    def initialize(self):
        self.topic_assignments = [np.random.randint(0, self.num_topics, size=d.size) for d in self.documents]
        self.class_assignments = [np.random.randint(0, self.num_classes, size=d.size) for d in self.documents]
        self.run_counts()

    def run_counts(self):
        self.num_words_in_doc_assigned_to_topic = [np.zeros(self.num_topics) for _ in self.documents]
        self.num_same_words_assigned_to_topic = np.zeros((self.vocab_size, self.num_topics))
        self.num_words_assigned_to_topic = np.zeros(self.num_topics)
        self.num_same_words_assigned_to_class = np.zeros((self.vocab_size, self.num_classes))
        self.num_words_assigned_to_class = np.zeros(self.num_classes)
        self.num_transitions = np.zeros((self.num_classes, self.num_classes))

        for i, doc in enumerate(self.documents):
            for j, word in enumerate(doc):
                c = self.class_assignments[i][j]
                t = self.topic_assignments[i][j]
                self.num_words_assigned_to_class[c] += 1
                self.num_same_words_assigned_to_class[word, c] += 1
                if j > 0:
                    prev = self.class_assignments[i][j-1]
                    self.num_transitions[prev, c] += 1
                if c == 0:
                    self.num_words_in_doc_assigned_to_topic[i][t] += 1
                    self.num_same_words_assigned_to_topic[word, t] += 1
                    self.num_words_assigned_to_topic[t] += 1

    def train(self, iterations=50):
        print(f"Gibbs Fitting: {len(self.documents)} docs, {iterations} iters...")
        for it in range(iterations):
            for i in range(len(self.documents)):
                for j in range(len(self.documents[i])):
                    self.draw_class(i, j)
                    self.draw_topic(i, j)

    def draw_topic(self, i, j):
        old_topic = self.topic_assignments[i][j]
        old_class = self.class_assignments[i][j]
        word = self.documents[i][j]
        
        props = self.num_words_in_doc_assigned_to_topic[i].copy()
        if old_class == 0: props[old_topic] -= 1
        props += self.alpha
        
        if old_class == 0:
            num = self.num_same_words_assigned_to_topic[word].copy()
            den = self.num_words_assigned_to_topic.copy()
            num[old_topic] -= 1; den[old_topic] -= 1
            props *= (num + self.beta) / (den + self.vocab_size * self.beta)
            
        new_topic = categorical(props)
        self.topic_assignments[i][j] = new_topic
        
        if old_class == 0:
            self.num_words_in_doc_assigned_to_topic[i][old_topic] -= 1
            self.num_words_in_doc_assigned_to_topic[i][new_topic] += 1
            self.num_same_words_assigned_to_topic[word, old_topic] -= 1
            self.num_same_words_assigned_to_topic[word, new_topic] += 1
            self.num_words_assigned_to_topic[old_topic] -= 1
            self.num_words_assigned_to_topic[new_topic] += 1

    def draw_class(self, i, j):
        old_class = self.class_assignments[i][j]
        old_topic = self.topic_assignments[i][j]
        word = self.documents[i][j]
        
        prev = self.class_assignments[i][j-1] if j > 0 else None
        futr = self.class_assignments[i][j+1] if j < len(self.documents[i])-1 else None
        
        if prev is not None:
            t1 = self.num_transitions[prev].copy(); t1[old_class] -= 1
        else: t1 = np.zeros(self.num_classes)
        t1 += self.gamma
        
        if futr is not None:
            t2 = self.num_transitions[:, futr].copy(); t2[old_class] -= 1
        else: t2 = np.zeros(self.num_classes)
        t2 += self.gamma
        
        num = t1 * t2
        den = self.num_words_assigned_to_class.copy()
        if prev is not None: den[prev] += 1
        den += self.num_classes * self.gamma
        
        m_num = self.num_same_words_assigned_to_class[word].copy()
        m_num[0] = self.num_same_words_assigned_to_topic[word, old_topic]
        if old_class != 0: m_num[old_class] -= 1
        else: m_num[0] -= 1
        m_num[1:] += self.delta; m_num[0] += self.beta
        
        m_den = self.num_words_assigned_to_class.copy()
        m_den[0] = self.num_words_assigned_to_topic[old_topic]
        if old_class != 0: m_den[old_class] -= 1
        else: m_den[0] -= 1
        m_den[1:] += self.delta * self.vocab_size; m_den[0] += self.beta * self.vocab_size
        
        props = (m_num / m_den) * num / den
        new_class = categorical(props)
        self.class_assignments[i][j] = new_class
        
        if prev is not None:
            self.num_transitions[prev, old_class] -= 1
            self.num_transitions[prev, new_class] += 1
        if futr is not None:
            self.num_transitions[old_class, futr] -= 1
            self.num_transitions[new_class, futr] += 1
            
        self.num_same_words_assigned_to_class[word, old_class] -= 1
        self.num_same_words_assigned_to_class[word, new_class] += 1
        self.num_words_assigned_to_class[old_class] -= 1
        self.num_words_assigned_to_class[new_class] += 1
        
        if old_class == 0 and new_class != 0:
            self.num_words_in_doc_assigned_to_topic[i][old_topic] -= 1
            self.num_same_words_assigned_to_topic[word, old_topic] -= 1
            self.num_words_assigned_to_topic[old_topic] -= 1
        elif old_class != 0 and new_class == 0:
            self.num_words_in_doc_assigned_to_topic[i][old_topic] += 1
            self.num_same_words_assigned_to_topic[word, old_topic] += 1
            self.num_words_assigned_to_topic[old_topic] += 1

    def get_fitted_params(self):
        # Normalize counts to probs
        trans = self.num_transitions + self.gamma
        trans /= trans.sum(axis=1, keepdims=True)
        top_emit = self.num_same_words_assigned_to_topic.T + self.beta
        top_emit /= top_emit.sum(axis=1, keepdims=True)
        syn_emit = self.num_same_words_assigned_to_class.T + self.delta
        syn_emit /= syn_emit.sum(axis=1, keepdims=True)
        return torch.tensor(trans).float(), torch.tensor(syn_emit).float(), torch.tensor(top_emit).float()

class FittedHMMLDAGenerator(HMMLDAGenerator):
    """
    Fits HMM-LDA on source file, then acts as a generator using learned params.
    """
    def __init__(self, source_file, vocab_size, block_size, n_states=5, n_topics=3):
        # 1. Read & Tokenize
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file {source_file} not found.")
        with open(source_file, 'r', encoding='utf-8') as f:
            raw = f.read().split()
        
        # Naive tokenization mapping to vocab_size
        tokens = [int(t) % vocab_size for t in raw if t.isdigit()] # Expects int tokens or hashes
        if not tokens: # Fallback for real text
             tokens = [hash(t) % vocab_size for t in raw]
             
        docs = [np.array(tokens[i:i+block_size]) for i in range(0, len(tokens)-block_size, block_size)]
        
        # 2. Fit
        sampler = HMMLDAGibbsSampler(vocab_size, n_topics, n_states)
        limit = min(len(docs), 500) # Limit docs for speed
        for d in docs[:limit]: sampler.add_document(d)
        
        sampler.initialize()
        sampler.train(iterations=20)
        
        # 3. Init Parent with Fitted Params
        trans, syn, top = sampler.get_fitted_params()
        super().__init__(vocab_size, block_size, n_states, n_topics)
        
        self.trans_mat = trans
        self.syntactic_emission = syn
        self.topic_emission = top
        self.topic_state_idx = 0 # Gibbs Sampler uses 0 as topic

# ==========================================
# 3. Dataset Wrapper & Factory
# ==========================================

class SyntheticDataset(IterableDataset):
    def __init__(self, generator, num_samples):
        self.generator = generator
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            outputs = self.generator.generate()
            
            # Format handling
            if isinstance(outputs, torch.Tensor):
                full_seq = outputs
                extra_info = {}
            else:
                full_seq = outputs['x']
                extra_info = {k: v for k, v in outputs.items() if k != 'x'}

            # Create Causal Input/Target
            x = full_seq[:-1]
            y = full_seq[1:]
            
            # Align Metadata: Slice sequence-length metadata to match y
            aligned_extra = {}
            for k, v in extra_info.items():
                if k in ['s', 'z'] and len(v) == len(full_seq):
                     aligned_extra[k] = v[1:] 
                else:
                     # e.g., 'theta', 'P' (Transition Matrix) stay global
                     aligned_extra[k] = v
            
            yield x, y, aligned_extra
            
    def __len__(self):
        return self.num_samples

class DataFactory:
    @staticmethod
    def get_loaders(args):
        """
        Parses argparse args and returns (train, val, test) loaders.
        """
        # 1. Select Generator
        if args.data_type == 'iid':
            gen = IIDGenerator(args.vocab_size, args.block_size)
        elif args.data_type == 'markov':
            gen = MarkovGenerator(args.vocab_size, args.block_size)

        elif args.data_type == 'hmm':
            # 這是 Xie et al. (2022) 的 HMM Generator
            # 預設參數可以模擬 GINC (n_states=5)
            # 這裡我們假設 consistency=True (Language Modeling) 或 False (ICL)
            # 你可以加一個 argparse 參數 --consistency 來控制，這裡先預設 False (ICL)
            gen = HMMGenerator(
                vocab_size=args.vocab_size, 
                block_size=args.block_size, 
                n_states=args.n_states,
                trans_alpha=0.1,  # 較低的 alpha 讓轉移比較有結構 (Structure)
                emit_alpha=0.1,   # 較低的 alpha 讓 emission 比較稀疏 (Concept 明確)
                consistency=False # 預設做 In-Context Learning
                )

        elif args.data_type == 'edelman_icl':
            # Note: ICL works best with small vocab (e.g., 3)
            gen = EdelmanICLMCGenerator(args.vocab_size, args.block_size, alpha=1.0)
        elif args.data_type == 'hmm_lda':
            gen = HMMLDAGenerator(args.vocab_size, args.block_size, 
                                  n_states=args.n_states, n_topics=args.n_topics, n_func_words=args.n_func_words)
        elif args.data_type == 'fitted_hmm_lda':
            gen = FittedHMMLDAGenerator(args.source_file, args.vocab_size, args.block_size,
                                        n_states=args.n_states, n_topics=args.n_topics)
        else:
            raise ValueError(f"Unknown data_type: {args.data_type}")

        # 2. Wrap in Dataset
        train_ds = SyntheticDataset(gen, args.train_samples)
        val_ds   = SyntheticDataset(gen, args.val_samples)
        test_ds  = SyntheticDataset(gen, args.test_samples)

        # 3. Return Loaders
        return (
            DataLoader(train_ds, batch_size=args.batch_size),
            DataLoader(val_ds, batch_size=args.batch_size),
            DataLoader(test_ds, batch_size=args.batch_size)
        )
