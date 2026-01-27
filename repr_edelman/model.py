import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Minimal Model (Induction Head Logic)
# ==========================================
class MinModel(nn.Module):
    """
    Minimal model defined in the paper.
    It is explicitly constructed to be able to express an Induction Head algorithm:
    Layer 1: Attends to previous token (learned via positional bias v).
    Layer 2: Attends to tokens where 'previous token' matches 'current token' (learned via W).
    """
    def __init__(self, vocab_size, block_size, **kwargs):
        super().__init__()
        self.length = block_size
        self.num_tokens = vocab_size
        
        # Learnable parameters
        self.v = nn.Embedding(self.length, 1) # Layer 1 position bias
        self.W = nn.Linear(self.num_tokens, self.num_tokens, bias=False) # Layer 2 interaction
        
        # Fixed logic
        self.wte = lambda x: F.one_hot(x, self.num_tokens).float()
        
        # Causal mask and relative position indices
        self.register_buffer("bias", torch.tril(torch.ones(self.length, self.length))
                                     .view(1, self.length, self.length) == 0)
        
        pos = torch.arange(self.length, dtype=torch.long)
        pos = pos.view(-1,1) - pos.view(1, -1)
        pos[pos<0] = 0
        self.register_buffer("pos", pos)

        # Init
        torch.nn.init.constant_(self.v.weight, 0)
        torch.nn.init.constant_(self.W.weight, 0.01)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        e = self.wte(idx) # (B, T, vocab_size)

        # --- Layer 1: Positional Look-back ---
        # Attention scores depend ONLY on relative position (distance)
        pos_embd = self.v(self.pos[:T,:T]).squeeze(-1) # (T, T)
        attention1 = pos_embd
        attention1 = attention1.masked_fill(self.bias[:, :T, :T], float('-inf'))
        attention1 = F.softmax(attention1, dim=-1)
        
        # Output is a mixture of token one-hots based on distance
        # Ideally, this becomes "the previous token"
        layer_one = attention1 @ e 

        # --- Layer 2: Content Matching (Induction) ---
        # Query: Current token (e)
        # Key: Transformed previous tokens (W @ layer_one)
        # We want to find j where: Current_Token == Previous_Token_of_j
        key = self.W(layer_one) # (B, T, vocab_size)
        
        # (B, T, V) @ (B, V, T) -> (B, T, T)
        attention2 = e @ key.transpose(1, 2)
        
        attention2 = attention2.masked_fill(self.bias[:, :T, :T], float('-inf'))
        attention2 = F.softmax(attention2, dim=-1) # (B, T, T) (Causal Masked) (tril in original code was redundant with mask)
        
        output = attention2 @ e # Copy the token from the attended position
        logits = output # The output is directly the logits (unnormalized probs)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    # Keeping the specific optimizer config from the paper
    def configure_optimizers(self, learning_rate=1.0):
        optim_groups = []
        optim_groups.append({"params": [self.W.weight], "weight_decay": 0, "lr": learning_rate})
        # The paper often uses a very high LR for v to force a sharp attention transition
        optim_groups.append({"params": [self.v.weight], "weight_decay": 0, "lr": learning_rate}) 
        return torch.optim.SGD(optim_groups)


# ==========================================
# 2. RPE Transformer (Attention Only)
# ==========================================
class RPECausalAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Relative Position Bias: A learnable scalar added to attention scores based on distance
        self.relative_bias = nn.Embedding(max_len, n_head)
        
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))
        
        # Distance matrix for lookup
        pos = torch.arange(max_len, dtype=torch.long)
        dist = pos.view(-1, 1) - pos.view(1, -1)
        dist = dist.clamp(min=0) # We only care about causal distance (0 to T)
        self.register_buffer("dist", dist)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Standard Attention Score
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Add Relative Position Bias
        # r_bias shape: (T, T, n_head) -> permute to (1, n_head, T, T)
        r_bias = self.relative_bias(self.dist[:T, :T]).permute(2, 0, 1).unsqueeze(0)
        att = att + r_bias

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class RPEModel(nn.Module):
    """
    2-Layer Attention-Only Transformer with Relative Position Embeddings.
    Popular for language modeling analysis.
    """
    def __init__(self, vocab_size, d_model=64, n_layer=2, n_head=1, max_len=128, **kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Note: No absolute position embedding here, as we use RPE in attention
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                RPECausalAttention(d_model, n_head, max_len)
            ) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        
        for layer in self.layers:
            x = x + layer(x) # Residual connection
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss


# ==========================================
# 3. Standard Transformer (GPT-style)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_len):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model)
        self.c_proj  = nn.Linear(4 * d_model, d_model)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class StandardTransformer(nn.Module):
    """
    Standard GPT-style Transformer.
    Supports config for 'attention only' or 'with MLP'.
    """
    def __init__(self, vocab_size, d_model=64, n_layer=2, n_head=2, max_len=128, use_mlp=True, **kwargs):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            layers = [
                nn.LayerNorm(d_model),
                CausalSelfAttention(d_model, n_head, max_len)
            ]
            if use_mlp:
                layers.append(nn.LayerNorm(d_model))
                layers.append(MLP(d_model))
            self.layers.append(nn.ModuleList(layers))
            
        self.use_mlp = use_mlp
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        for block in self.layers:
            # Attention block
            x = x + block[1](block[0](x))
            # MLP block (if exists)
            if self.use_mlp:
                x = x + block[3](block[2](x))
                
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
