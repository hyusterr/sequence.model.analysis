import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    block_size: int = 64
    vocab_size: int = 50
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    is_causal: bool = True  # <--- 新增這個開關

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)
        self.is_causal = config.is_causal # 儲存 causal 設定
        
        # Causal mask buffer (只在 is_causal=True 時使用，但先註冊起來比較方便)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_att_weights=False):
        B, T, C = x.size() 

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # --- 核心修改區域 ---
        if self.is_causal:
            # Causal: Apply lower-triangular mask
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        else:
            # Bi-directional: No mask needed (full attention)
            # 如果你有 padding token，這裡可能需要針對 padding 做 mask，
            # 但在 synthetic data block_size 固定且滿的情況下，不需要額外 mask。
            pass 
        # ------------------

        att = F.softmax(att, dim=-1)
        att_weights = att if return_att_weights else None
        
        att = self.dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        return self.c_proj(y), att_weights

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config) # 改名為 Generic SelfAttention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, return_att_weights=True):
        attn_out, att_weights = self.attn(self.ln_1(x), return_att_weights=return_att_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, att_weights

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_att_weights=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) 
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        
        all_att_weights = []
        for block in self.transformer.h:
            x, att_w = block(x, return_att_weights=return_att_weights)
            if return_att_weights:
                all_att_weights.append(att_w)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # 注意：如果是 Bi-directional 且做 Next Token Prediction，
            # 模型會直接看到答案 (Leakage)。
            # 在訓練 Bi-directional 時，通常需要自行實作 MLM (Masked Language Modeling) 的 dataset masking。
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, all_att_weights
