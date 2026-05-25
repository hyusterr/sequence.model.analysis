import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 核心組件：Token Embedding & PE
# ==========================================
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()
        self.wpe = nn.Embedding(block_size, d_model)
    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        return x + self.wpe(pos)

# ==========================================
# 2. Multi-Head Attention (支援 Standard, Linear, Performer)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, block_size, pe_type='none', attn_type='standard', dropout=0.1):
        super().__init__()
        self.d_model, self.nhead, self.attn_type, self.pe_type = d_model, nhead, attn_type, pe_type
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False) # W_Q
        self.k_proj = nn.Linear(d_model, d_model, bias=False) # W_K
        self.v_proj = nn.Linear(d_model, d_model, bias=False) # W_V
        self.c_proj = nn.Linear(d_model, d_model, bias=False) # W_O
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 標準 Mask 與 Edelman RPE 初始化 (同前版本)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        # register causal masking

        if pe_type == 'rpe':
            self.wpe_rel = nn.Embedding(block_size + 1, d_model, padding_idx=block_size)
            pos = torch.arange(block_size).unsqueeze(0); pos = pos.view(-1, 1) - pos.view(1, -1)
            pos = torch.maximum(pos, torch.tensor(-1)); pos[pos == -1] = block_size
            self.register_buffer("pos_matrix", pos)

        # --- Performer 專屬：隨機投影矩陣 (FAVOR+) ---
        if attn_type == 'performer':
            # 投影維度 m 通常設為 head_dim 的幾倍，這裡取 1:1 簡化
            # m = self.head_dim
            # projection_matrix = self._create_orthogonal_projection(m, self.head_dim) 
            # ref: performer_pytorch / original paper
            m = int(self.head_dim * math.log(self.head_dim))
            projection_matrix = self._create_orthogonal_projection(m, self.head_dim)
            self.register_buffer("projection_matrix", projection_matrix)

    def _create_orthogonal_projection(self, m, d):
        # 模仿官方 gaussian_orthogonal_random_matrix 的簡化邏輯
        # 我們需要 m 行 d 列
        nb_full_blocks = m // d
        blocks = []
        
        for _ in range(nb_full_blocks):
            # 生成正交矩陣 (QR 分解)
            q, r = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q) # q 是正交的
            
        remaining_rows = m - nb_full_blocks * d
        if remaining_rows > 0:
            q, r = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q[:remaining_rows])
            
        final_matrix = torch.cat(blocks)
        
        # 官方還會乘以一個隨機的 Norm (multiplier)，讓它符合高斯分佈的長度
        multiplier = torch.randn(m, d).norm(dim=1, keepdim=True)
        return final_matrix * multiplier

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        if self.attn_type == 'standard':
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if self.pe_type == 'rpe':
                rpe_bias = torch.einsum("ijhe,bhei->bhij", self.wpe_rel(self.pos_matrix[:T, :T]).view(T, T, self.nhead, self.head_dim), q)
                att += rpe_bias * (1.0 / math.sqrt(self.head_dim))
            att = F.softmax(att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')), dim=-1)
            y = self.attn_dropout(att) @ v

        elif self.attn_type == 'linear': # Transformers are RNNs (ELU 版)
            q, k = F.elu(q) + 1, F.elu(k) + 1
            k_cumsum = k.cumsum(dim=-2)
            v_cumsum = torch.einsum("bhte,bhtd->bhted", k, v).cumsum(dim=2)
            y = torch.einsum("bhte,bhted->bhtd", q, v_cumsum) / (torch.einsum("bhte,bhte->bht", q, k_cumsum).unsqueeze(-1) + 1e-6)

        elif self.attn_type == 'performer': # FAVOR+ 近似
            # 1. 投影到隨機特徵空間
            # phi(x) = exp(w*x - |x|^2/2) / sqrt(m)
            q_norm = (q ** 2).sum(dim=-1, keepdim=True) / 2
            k_norm = (k ** 2).sum(dim=-1, keepdim=True) / 2
            
            # 使用 einsum 進行隨機投影
            q_proj = torch.einsum("bhte,me->bhtm", q, self.projection_matrix)
            k_proj = torch.einsum("bhte,me->bhtm", k, self.projection_matrix)
            
            # 正向特徵映射 (FAVOR+)
            phi_q = torch.exp(q_proj - q_norm) / math.sqrt(self.projection_matrix.size(0))
            phi_k = torch.exp(k_proj - k_norm) / math.sqrt(self.projection_matrix.size(0))
            
            # 2. 進行因果線性計算 (RNN 遞迴形式)
            k_cumsum = phi_k.cumsum(dim=-2)
            v_cumsum = torch.einsum("bhtm,bhtd->bhtmd", phi_k, v).cumsum(dim=2)
            y = torch.einsum("bhtm,bhtmd->bhtd", phi_q, v_cumsum) / (torch.einsum("bhtm,bhtm->bht", phi_q, k_cumsum).unsqueeze(-1) + 1e-6)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

# ==========================================
# 3. 基礎架構 (FFN, Block, Transformer)
# ==========================================
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(d_model, d_ff), 
                nn.GELU(), 
                nn.Linear(d_ff, d_model), 
                nn.Dropout(dropout)
            )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, block_size, pe_type, attn_type, attention_only=False, use_residual=True):
        super().__init__()
        self.use_residual, self.attention_only = use_residual, attention_only
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, block_size, pe_type, attn_type)
        if not attention_only:
            self.ln2 = nn.LayerNorm(d_model); self.ffn = PositionWiseFFN(d_model, d_ff)

    def forward(self, x):
        a_out = self.attn(self.ln1(x))
        x = (x + a_out) if self.use_residual else a_out
        if not self.attention_only:
            f_out = self.ffn(self.ln2(x))
            x = (x + f_out) if self.use_residual else f_out
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, block_size, 
                 pe_type='none', attn_type='standard', attention_only=False, use_residual=True):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.abs_pe = AbsolutePositionalEncoding(d_model, block_size) if pe_type == 'absolute' else nn.Identity()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead, 4*d_model, block_size, pe_type, attn_type, attention_only, use_residual) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.abs_pe(self.token_emb(idx))
        for block in self.blocks: x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
