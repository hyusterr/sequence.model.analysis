"""
Markov property probes for attention-only Transformers (mingpt-based)

What this script does
---------------------
1) Sweeps model & data settings:
   - residual: {on, off}
   - causal: {on, off}
   - n_head: {1, 2}
   - n-gram order n: {1, 2, 3}
   - vocab size K ("k-tokens"): {2, 3}
   - Dirichlet alpha for transition init (default all-ones)
2) Trains a small attention-only model for each setting (via your training_pipeline).
3) Probes Markov-ness on:
   - raw data (sanity check)
   - hidden states (per layer, per head, per dimension)
   - output logits / next-token distribution
4) Outputs figures & CSV summaries into ./runs/<stamp>/ ...

Assumptions
-----------
- You have `training_pipeline.py`, `datasets.py`, and mingpt modules on PYTHONPATH.
- `datasets.ngrams(split, n, length, num_symbols, dirichlet_alpha=None, ...)` accepts optional `dirichlet_alpha`.
  If not, the script will ignore and use dataset default (Dir(1,...,1)).
- `Relative_Transformer` is the model used when model_type contains the word "transformer".
- We will monkey-patch residual off by wrapping each block forward (robust to common mingpt layouts).

Usage
-----
python markov_attn_only_experiments.py
"""
from __future__ import annotations
import os, math, time, json, copy, itertools, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import training_pipeline
import datasets
import test_error
import graphing_pipeline  # optional; we only use our own plotting if unavailable
from mingpt.utils import set_seed

# ---------------------------
# Helpers: time-stamped run dir
# ---------------------------
STAMP = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = os.path.join("runs", STAMP)
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------------------
# Experiment configuration
# ---------------------------
@dataclass
class SweepConfig:
    seeds: List[int] = (0,)
    n_list: List[int] = (1, 2, 3)              # n-gram order
    k_tokens_list: List[int] = (2, 3)          # vocabulary size
    n_head_list: List[int] = (1, 2)
    causal_list: List[bool] = (True, False)
    residual_list: List[bool] = (True, False)
    dirichlet_alpha: Optional[float] = None    # None -> dataset default (Dir(1,...,1))

    # training
    max_iters: int = 1000
    batch_size: int = 64
    block_size: int = 100
    n_layer: int = 2
    n_embd_base: int = 16  # we multiply by n_head below to keep per-head width similar
    learning_rate: float = 5e-4
    num_workers: int = 4

SWEEP = SweepConfig()
# also try 1-layer models in the sweep by overriding per-setting


# ---------------------------
# Residual ON/OFF monkey patch
# ---------------------------

def _patch_residual_off(model: nn.Module):
    """Best-effort patch: turn x <- x + attn(x) into x <- attn(x).
    Works for common mingpt block layouts. If not found, we silently leave as-is."""
    replaced = 0

    def patch_block(block: nn.Module):
        nonlocal replaced
        if hasattr(block, 'attn') and hasattr(block, 'ln_1'):
            # Try common GPT block forward signature
            orig_forward = block.forward
            def new_forward(x, *args, **kwargs):
                # pre-LN
                y = block.attn(block.ln_1(x))
                # normally: x = x + y
                x = y  # drop residual
                if hasattr(block, 'ln_2') and hasattr(block, 'mlp'):
                    # If there is an MLP path in this block, keep or drop residual accordingly.
                    # For attention-only models in your pipeline, MLP is often absent, but we handle both.
                    y2 = block.mlp(block.ln_2(x))
                    x = y2  # drop residual here as well
                return x
            block.forward = new_forward
            replaced += 1

    for m in model.modules():
        # Identify likely blocks; we patch any module that carries attn & ln_1
        if isinstance(m, nn.Module):
            patch_block(m)

    print(f"[patch] residual OFF patched blocks: {replaced}")

# ---------------------------
# Hooks to capture hidden activations
# ---------------------------
class ActivationCache:
    def __init__(self):
        self.cache = {}
        self.handles = []

    def add_hook(self, name: str, module: nn.Module, which: str = 'output'):
        if which == 'output':
            handle = module.register_forward_hook(lambda m, inp, out: self.cache.setdefault(name, []).append(out.detach().cpu()))
        elif which == 'input':
            handle = module.register_forward_pre_hook(lambda m, inp: self.cache.setdefault(name, []).append(inp[0].detach().cpu()))
        else:
            raise ValueError('which must be input or output')
        self.handles.append(handle)

    def clear(self):
        self.cache.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

# Try to attach hooks to per-layer attention outputs (after attention, before residual)

def attach_attn_out_hooks(model: nn.Module) -> ActivationCache:
    ac = ActivationCache()
    # Heuristic: record outputs of each attention module
    idx = 0
    for name, module in model.named_modules():
        if name.endswith('attn') or name.endswith('attn.proj'):
            ac.add_hook(f"attn_out[{idx}]:{name}", module, which='output')
            idx += 1
    for name, module in model.named_modules():
        if name.endswith('ln_f') or name.endswith('head'):
            ac.add_hook(f"{name}", module, which='input')
    return ac

# ---------------------------
# Probing metrics for Markov-ness
# ---------------------------

def contexts_from_tokens(tokens: torch.Tensor, order: int, K: int) -> torch.Tensor:
    B, T = tokens.shape
    powers = torch.tensor([K**p for p in reversed(range(order))], device=tokens.device)
    idx = torch.zeros((B, T), dtype=torch.long, device=tokens.device)
    for t in range(T):
        if t + 1 < order:
            span = tokens[:, :t+1]
            pad = torch.zeros((B, order - (t+1)), dtype=tokens.dtype, device=tokens.device)
            window = torch.cat([pad, span], dim=1)
        else:
            window = tokens[:, t+1-order:t+1]
        if order > 0:
            idx[:, t] = (window * powers).sum(dim=1)
        else:
            idx[:, t] = 0
    return idx

@torch.no_grad()
def empirical_conditional_from_data(tokens: torch.Tensor, K: int, orders: List[int]):
    """Compute empirical P(next | last m tokens) for m in orders from raw discrete data.
    tokens: (N_seq, T) full sequences; we use positions 0..T-2 as contexts and T-1 as next.
    Returns dict m -> probs tensor (num_contexts(m), K).
    """
    device = tokens.device
    N, T = tokens.shape
    results = {}
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    # flatten
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    for m in orders:
        num_ctx = K**m if m>0 else 1
        counts = torch.zeros((num_ctx, K), dtype=torch.long, device=device)
        if m==0:
            # unigram prior for next-token
            for k in range(K):
                counts[0, k] = (y_flat==k).sum()
        else:
            ctx_idx = contexts_from_tokens(x, order=m, K=K).reshape(-1)
            for c in range(num_ctx):
                mask = (ctx_idx==c)
                if mask.any():
                    ys = y_flat[mask]
                    for k in range(K):
                        counts[c, k] = (ys==k).sum()
        probs = (counts.float()+1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True)
        results[m] = probs.cpu()
    return results

@torch.no_grad()
def empirical_conditional_from_model(probs: torch.Tensor, ctx_idx: torch.Tensor, K: int, m: int):
    """Aggregate model next-token probs into P_model(.|last m tokens) by averaging over positions in same context.
    probs: (N, K), ctx_idx: (N,)
    Returns (num_ctx(m), K)
    """
    num_ctx = K**m if m>0 else 1
    out = torch.zeros((num_ctx, K), dtype=torch.float32)
    cnt = torch.zeros((num_ctx,), dtype=torch.float32)
    if m==0:
        out[0] = probs.mean(dim=0)
        cnt[0] = probs.size(0)
    else:
        for c in range(num_ctx):
            mask = (ctx_idx==c)
            if mask.any():
                out[c] = probs[mask].mean(dim=0)
                cnt[c] = mask.float().sum()
    return out, cnt


@torch.no_grad()
def kl_between_histories_same_suffix(model_probs: torch.Tensor,
                                     hist_idx_short: torch.Tensor,
                                     hist_idx_long: torch.Tensor,
                                     K: int) -> float:
    """
    Estimate average pairwise KL divergence between P(.|long history)
    across different long-histories that share the same short suffix.
    Returns mean KL; ~0 indicates Markov (order = len(short)).
    model_probs: (N, K)
    hist_idx_short/long: (N,)
    """
    # to numpy (ints)
    s = hist_idx_short.detach().cpu().numpy().astype(np.int64)
    l = hist_idx_long.detach().cpu().numpy().astype(np.int64)
    N = s.shape[0]

    # buckets[short][long] = list of row indices
    buckets = {}
    for i in range(N):
        sv = int(s[i]); lv = int(l[i])
        d = buckets.setdefault(sv, {})
        d.setdefault(lv, []).append(i)

    kl_list = []
    for sv, longs in buckets.items():
        if len(longs) <= 1:
            continue
        # mean probs for each distinct long-history under this short suffix
        means = []
        for lv, idxs in longs.items():
            idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=model_probs.device)
            P = model_probs[idxs_t].mean(dim=0).clamp_min(1e-8)
            means.append(P)
        # pairwise KL
        for i in range(len(means)):
            for j in range(i+1, len(means)):
                P = means[i]; Q = means[j]
                kl = (P * (P.log() - Q.log())).sum().item()
                kl_list.append(kl)

    return float(np.mean(kl_list)) if kl_list else 0.0

@torch.no_grad()
def gaussian_nll_improvement(hidden: torch.Tensor, ctx_short: torch.Tensor, ctx_long: torch.Tensor) -> float:
    import pandas as pd, numpy as np
    N, D = hidden.shape
    df = pd.DataFrame({'short': ctx_short.cpu().numpy(), 'long': ctx_long.cpu().numpy(), 'row': np.arange(N)})
    improvements = []
    for d in range(D):
        x = hidden[:, d].cpu().numpy()
        nll_short, nll_long = [], []
        for s, sub in df.groupby('short'):
            rows = sub['row'].values
            mu_s = x[rows].mean(); var_s = x[rows].var() + 1e-6
            nll_short.append(0.5*np.log(2*math.pi*var_s) + 0.5*((x[rows]-mu_s)**2/var_s).mean())
            for l, sub2 in sub.groupby('long'):
                rows2 = sub2['row'].values
                mu_l = x[rows2].mean(); var_l = x[rows2].var() + 1e-6
                nll_long.append(0.5*np.log(2*math.pi*var_l) + 0.5*((x[rows2]-mu_l)**2/var_l).mean())
        if len(nll_short) and len(nll_long):
            improvements.append(np.mean(nll_short) - np.mean(nll_long))
    return float(np.mean(improvements)) if improvements else 0.0

# ---------------------------
# Training + Evaluation per setting
# ---------------------------

def build_and_train_one(setting: Dict) -> Tuple[object, List[object], List[float]]:
    conf = training_pipeline.get_default_config()
    conf.model_type = 'Attention-Only Relative positions Transformer'
    conf.n_layer = setting['n_layer']
    conf.n_head = setting['n_head']
    conf.n_embd = SWEEP.n_embd_base * conf.n_head
    conf.vocab_size = setting['K']
    conf.block_size = SWEEP.block_size
    conf.max_iters = SWEEP.max_iters
    conf.batch_size = SWEEP.batch_size
    conf.num_workers = SWEEP.num_workers
    conf.learning_rate = SWEEP.learning_rate
    conf.causal = setting['causal']
    conf.n = setting['n']

    length = conf.block_size + 1
    try:
        conf.dataset = datasets.ngrams('train', conf.n, length, conf.vocab_size,
                                       last_token_only=False,
                                       dirichlet_alpha=SWEEP.dirichlet_alpha)
    except TypeError:
        conf.dataset = datasets.ngrams('train', conf.n, length, conf.vocab_size, last_token_only=False)

    set_seed(setting['seed'])
    model_history, train_loss = training_pipeline.train(conf)

    model = model_history[-1]
    if not setting['residual']:
        _patch_residual_off(model)

    return model, model_history, train_loss

@torch.no_grad()
def evaluate_markovness(model, setting: Dict, sample_N: int = 50000) -> Dict:
    device = next(model.parameters()).device
    K = setting['K']
    n = setting['n']
    length = SWEEP.block_size + 1
    test_ds = datasets.ngrams('test', n, length, K, size=sample_N//length)

    all_tokens = []
    all_probs = []
    total_seqs = 0  # ← 新增：實際累計的序列數（= 所有 batch 的 x.shape[0] 加總）
    T_effective = None


    ac = attach_attn_out_hooks(model)
    model.eval()

    dl = torch.utils.data.DataLoader(test_ds, batch_size=SWEEP.batch_size, shuffle=False, num_workers=0)
    for x, y in dl:
        x = x.to(device)
        logits, loss = model(x)
        probs = logits.softmax(dim=-1)
        all_probs.append(probs[:, :-1, :].reshape(-1, K).cpu())
        all_tokens.append(x[:, :-1].reshape(-1).cpu())

        total_seqs += x.size(0)  # ← 新增
        if T_effective is None:
            T_effective = x.size(1) - 1  # because we used x[:, :-1]


    # stack to shapes ((Nseq* (T-1)), K) and same for tokens
    # 用 cat 而不是 stack（最後一個 batch 可能比較小）
    all_probs = torch.cat(all_probs, dim=0)
    all_tokens = torch.cat(all_tokens, dim=0)
    # all_tokens = torch.stack(all_tokens).reshape(-1)

    # reshape back into (B,T) for context indexing
    # B = len(test_ds)
    # T = SWEEP.block_size
    # 用實際累計的序列數來還原 (B, T)
    # reshape back into (B,T) for context indexing
    B = total_seqs
    T = T_effective if T_effective is not None else SWEEP.block_size
    assert all_tokens.numel() == B * T, f"mismatch: got {all_tokens.numel()} != {B}*{T}"
    toks_bt = all_tokens.view(B, T)


    # Build contexts (short = n-1, long = n)
    m_short = max(0, n-1)
    m_long = max(0, n)
    ctx_short = contexts_from_tokens(toks_bt, order=m_short, K=K).reshape(-1)
    ctx_long = contexts_from_tokens(toks_bt, order=m_long, K=K).reshape(-1)

    # Metric 1: KL same-suffix
    kl_same_suffix = kl_between_histories_same_suffix(all_probs, ctx_short, ctx_long, K)

    # Metric 2: Hidden Gaussian NLL improvement
    hidden_impr = None
    if len(ac.cache):
        key0 = sorted(ac.cache.keys())[0]
        H = torch.cat(ac.cache[key0], dim=0)  # (B,T,C)
        H = H[:, :-1, :].reshape(-1, H.size(-1))
        hidden_impr = gaussian_nll_improvement(H, ctx_short, ctx_long)
    ac.remove()

    # --- Empirical probabilities ---
    # From raw test data (discrete): P_data(next | last m)
    # We need raw sequences including the next token. Rebuild tokens including last step.
    raw_tokens = []
    for x, y in torch.utils.data.DataLoader(test_ds, batch_size=SWEEP.batch_size, shuffle=False, num_workers=0):
        raw_tokens.append(x.cpu())
    raw_tokens = torch.cat(raw_tokens, dim=0)
    data_emp = empirical_conditional_from_data(raw_tokens, K, orders=list(range(0, m_long+1)))

    # From model: average probs by context for m in [0..m_long]
    model_emp = {}
    for m in range(0, m_long+1):
        ctx_m = contexts_from_tokens(toks_bt, order=m, K=K).reshape(-1)
        Pm, cnt = empirical_conditional_from_model(all_probs, ctx_m, K, m)
        model_emp[m] = Pm

    # Save CSVs per setting
    tag = f"n{n}_K{K}_H{setting['n_head']}_causal{int(setting['causal'])}_res{int(setting['residual'])}_seed{setting['seed']}"
    outdir = os.path.join(RUN_DIR, tag)
    os.makedirs(outdir, exist_ok=True)

    # data empirical
    for m, P in data_emp.items():
        df = pd.DataFrame(P.numpy())
        df.columns = [f'tok={k}' for k in range(K)]
        df.insert(0, 'context_idx', np.arange(P.shape[0]))
        df.to_csv(os.path.join(outdir, f"data_P_next_given_last_{m}.csv"), index=False)

    # model empirical
    for m, P in model_emp.items():
        df = pd.DataFrame(P.numpy())
        df.columns = [f'tok={k}' for k in range(K)]
        df.insert(0, 'context_idx', np.arange(P.shape[0]))
        df.to_csv(os.path.join(outdir, f"model_P_next_given_last_{m}.csv"), index=False)

    return {
        'kl_same_suffix': kl_same_suffix,
        'hidden_gauss_nll_impr': hidden_impr,
    }

# ---------------------------
# Main sweep
# ---------------------------

def main():
    rows = []
    curves = {}
    for seed in SWEEP.seeds:
        for n in SWEEP.n_list:
            for K in SWEEP.k_tokens_list:
                for n_head in SWEEP.n_head_list:
                    for causal in SWEEP.causal_list:
                        for residual in SWEEP.residual_list:
                            for n_layer in [1, 2]:  # include 1-layer models as requested
                                setting = dict(seed=seed, n=n, K=K, n_head=n_head, causal=causal, residual=residual, n_layer=n_layer)
                                tag = f"n{n}_K{K}_H{n_head}_L{n_layer}_causal{int(causal)}_res{int(residual)}_seed{seed}"
                                print("=== RUN:", tag)
                                model, model_hist, train_loss = build_and_train_one(setting)
                                curves[tag] = train_loss
                                metrics = evaluate_markovness(model, setting)
                                row = dict(tag=tag, **setting, **metrics)
                                rows.append(row)
                                # Save partial CSV/plots incrementally
                                df = pd.DataFrame(rows)
                                df.to_csv(os.path.join(RUN_DIR, 'summary.csv'), index=False)
                                # Loss curve
                                plt.figure(figsize=(5,3))
                                plt.plot(train_loss)
                                plt.title(tag + " train loss")
                                plt.xlabel("iter")
                                plt.ylabel("loss")
                                plt.tight_layout()
                                plt.savefig(os.path.join(RUN_DIR, f"loss_{tag}.png"), dpi=160)
                                plt.close()

    df = pd.DataFrame(rows)
    print("=== Summary ===")
    print(df)
    df.to_csv(os.path.join(RUN_DIR, 'summary.csv'), index=False)

    if len(df):
        for col in ['kl_same_suffix', 'hidden_gauss_nll_impr']:
            plt.figure(figsize=(8,4))
            ax = plt.gca()
            df.sort_values('tag').plot(x='tag', y=col, kind='bar', ax=ax, legend=False)
            plt.xticks(rotation=80, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(RUN_DIR, f"{col}_by_setting.png"), dpi=160)
            plt.close()

if __name__ == "__main__":
    main()
