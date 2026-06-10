import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 位置編碼家族 (APE & RoPE)
# ==========================================
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        return self.emb(x)

class AbsolutePositionalEncoding(nn.Module):
    """APE: 絕對位置編碼變體"""
    def __init__(self, d_model, block_size):
        super().__init__()
        self.wpe = nn.Embedding(block_size, d_model)
        
    def forward(self, x):
        t = x.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        return x + self.wpe(pos)

class RotaryPositionEmbedding(nn.Module):
    """RoPE: 旋轉位置編碼變體 (完美相容 Linear/Performer 的結合律)"""
    def __init__(self, dim, max_seq_len):
        super().__init__()
        # 依據標準定義計算逆頻率 (Inverse Frequency)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)       # [T, dim]
        
        # 預先快取 Cosine 與 Sine 矩陣，形狀對齊 [B, H, T, D]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def rotate_half(x):
    """將向量後半段與前半段交錯並取負號，用於旋轉矩陣的複數內積變形"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """將 RoPE 旋轉矩陣套用至 Query 與 Key"""
    # q, k 形狀: [B, H, T, D]
    # cos, sin 形狀: [1, 1, T, D]
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated


# ==========================================
# 2. Multi-Head Attention (全面整合 PE 變體)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, block_size, pe_type='none', attn_type='standard', dropout=0.1):
        super().__init__()
        self.d_model, self.nhead, self.attn_type, self.pe_type = d_model, nhead, attn_type, pe_type
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 註冊標準因果遮罩 (Causal Mask)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        # 變體 A: 相對位置編碼 (RPE) 初始化 (僅 Standard 可用)
        if pe_type == 'rpe':
            self.wpe_rel = nn.Embedding(block_size + 1, d_model, padding_idx=block_size)
            pos = torch.arange(block_size).unsqueeze(0)
            pos = pos.view(-1, 1) - pos.view(1, -1)
            pos = torch.maximum(pos, torch.tensor(-1))
            pos[pos == -1] = block_size
            self.register_buffer("pos_matrix", pos)

        # 變體 B: 旋轉位置編碼 (RoPE) 初始化 (所有模型通用)
        if pe_type == 'rope':
            self.rope = RotaryPositionEmbedding(self.head_dim, block_size)

        # Performer FAVOR+ 隨機投影矩陣初始化
        if attn_type == 'performer':
            m = int(self.head_dim * math.log(self.head_dim))
            projection_matrix = self._create_orthogonal_projection(m, self.head_dim)
            self.register_buffer("projection_matrix", projection_matrix)

    def _create_orthogonal_projection(self, m, d):
        nb_full_blocks = m // d
        blocks = []
        for _ in range(nb_full_blocks):
            q, r = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q)
        remaining_rows = m - nb_full_blocks * d
        if remaining_rows > 0:
            q, r = torch.linalg.qr(torch.randn(d, d))
            blocks.append(q[:remaining_rows])
        final_matrix = torch.cat(blocks)
        multiplier = torch.randn(m, d).norm(dim=1, keepdim=True)
        return final_matrix * multiplier

    def forward(self, x):
        B, T, C = x.size()
        
        # 投影並轉換形狀為標準的 [Batch, Head, Time, Dimension]
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # 🌟 核心整合：若選用 RoPE，在進入任何注意力分支前，直接旋轉 Q 與 K 的特徵軸
        if self.pe_type == 'rope':
            cos, sin = self.rope(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ------------------------------------------
        # 分支 1：標準注意力 (Standard Attention)
        # ------------------------------------------
        if self.attn_type == 'standard':
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if self.pe_type == 'rpe':
                rpe_bias = torch.einsum("ijhe,bhie->bhij", self.wpe_rel(self.pos_matrix[:T, :T]).view(T, T, self.nhead, self.head_dim), q)
                att += rpe_bias * (1.0 / math.sqrt(self.head_dim))
            att = F.softmax(att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')), dim=-1)
            y = self.attn_dropout(att) @ v

        # ------------------------------------------
        # 分支 2：標準線性注意力 (Linear Attention)
        # ------------------------------------------
        elif self.attn_type == 'linear':
            q, k = F.elu(q) + 1, F.elu(k) + 1
            k_cumsum = k.cumsum(dim=-2)
            v_cumsum = torch.einsum("bhte,bhtd->bhted", k, v).cumsum(dim=2)
            y = torch.einsum("bhte,bhted->bhtd", q, v_cumsum) / (torch.einsum("bhte,bhte->bht", q, k_cumsum).unsqueeze(-1) + 1e-6)

        # ------------------------------------------
        # 分支 3：隨機特徵線性注意力 (Performer)
        # ------------------------------------------
        elif self.attn_type == 'performer':
            alpha = self.head_dim ** -0.25
            q_scaled = q * alpha
            k_scaled = k * alpha
            
            q_norm = (q_scaled ** 2).sum(dim=-1, keepdim=True) / 2
            k_norm = (k_scaled ** 2).sum(dim=-1, keepdim=True) / 2
            
            q_proj = torch.einsum("bhte,me->bhtm", q_scaled, self.projection_matrix)
            k_proj = torch.einsum("bhte,me->bhtm", k_scaled, self.projection_matrix)
            
            q_max = torch.max(torch.max(q_proj, dim=-2, keepdim=True)[0], dim=-1, keepdim=True)[0]
            k_max = torch.max(torch.max(k_proj, dim=-2, keepdim=True)[0], dim=-1, keepdim=True)[0]
            
            phi_q = torch.exp(q_proj - q_max - q_norm) + 1e-6
            phi_k = torch.exp(k_proj - k_max - k_norm) + 1e-6
            
            k_cumsum = phi_k.cumsum(dim=-2)
            v_cumsum = torch.einsum("bhtm,bhtd->bhtmd", phi_k, v).cumsum(dim=2)
            
            D_inv = 1.0 / (torch.einsum("bhtm,bhtm->bht", phi_q, k_cumsum).unsqueeze(-1) + 1e-6)
            y = torch.einsum("bhtm,bhtmd->bhtd", phi_q, v_cumsum) * D_inv

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ==========================================
# 3. 基礎群組架構 (FFN, Block, Transformer)
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
    def forward(self, x): 
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, block_size, pe_type, attn_type, attention_only=False, use_residual=True):
        super().__init__()
        self.use_residual, self.attention_only = use_residual, attention_only
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, block_size, pe_type, attn_type)
        if not attention_only:
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = PositionWiseFFN(d_model, d_ff)

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
        
        # 🌟 核心修復：升級自動防錯護欄
        # 傳統 RPE 矩陣會徹底摧毀 Linear/Performer 的線性複雜度。若誤配，自動無損升級為支援結合律的 RoPE！
        if attn_type in ['linear', 'performer'] and pe_type == 'rpe':
            print(f"⚠️ [Guardrail] {attn_type} 不支援 Additive RPE。已自動升級為 RoPE 變體，完美保持線性時間複雜度與相對位置感知！")
            pe_type = 'rope'
            
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        
        # 只有當選用 APE ('absolute') 時，才在輸入端疊加絕對編碼矩陣。RPE 與 RoPE 均在 Attention 內部實作。
        self.abs_pe = AbsolutePositionalEncoding(d_model, block_size) if pe_type == 'absolute' else nn.Identity()
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, 4*d_model, block_size, pe_type, attn_type, attention_only, use_residual) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 權重初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        x = self.abs_pe(self.token_emb(idx))
        for block in self.blocks: 
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
