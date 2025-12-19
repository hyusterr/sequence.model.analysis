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
    is_causal: bool = True
    attn_only: bool = False
    use_relative_pos: bool = False

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = nn.Dropout(config.dropout)
        self.is_causal = config.is_causal
        self.use_relative_pos = config.use_relative_pos
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        if self.use_relative_pos:
            self.rel_pos_emb = nn.Parameter(torch.zeros(1, config.n_head, config.block_size, self.head_dim))
            nn.init.normal_(self.rel_pos_emb, mean=0.0, std=0.02)

    def forward(self, x, return_att_weights=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1))
        if self.use_relative_pos:
            range_vec = torch.arange(T, device=x.device)
            dist_idx = (range_vec[:, None] - range_vec[None, :]).clamp(min=0)
            rels = self.rel_pos_emb[:, :, dist_idx, :]
            pos_score = torch.sum(rels * k.unsqueeze(2), dim=-1)
            att = att + pos_score
        att = att * (1.0 / math.sqrt(k.size(-1)))
        if self.is_causal: att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att_weights = att if return_att_weights else None
        y = self.dropout(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), att_weights

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.dropout(self.gelu(self.c_fc(x)))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn_only = config.attn_only
        if not self.attn_only: self.mlp = MLP(config)
    def forward(self, x, return_att_weights=True):
        a, w = self.attn(self.ln_1(x), return_att_weights=return_att_weights)
        x = x + a
        if not self.attn_only: x = x + self.mlp(self.ln_2(x))
        return x, w

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd) if not config.use_relative_pos else None,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): torch.nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding): torch.nn.init.normal_(m.weight, std=0.02)
    def forward(self, idx, targets=None, return_att_weights=False):
        b, t = idx.size()
        tok_emb = self.transformer.wte(idx)
        if self.transformer.wpe is not None:
            pos_emb = self.transformer.wpe(torch.arange(0, t, device=idx.device))
            x = self.transformer.drop(tok_emb + pos_emb)
        else: x = self.transformer.drop(tok_emb)
        
        ws = []
        for block in self.transformer.h:
            x, w = block(x, return_att_weights=return_att_weights)
            if return_att_weights: ws.append(w)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss, ws
