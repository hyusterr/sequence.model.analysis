import torch
import numpy as np

def compute_theoretical_entropy_rate(generator):
    if not hasattr(generator, 'trans_mat'): return None
    P = generator.trans_mat.float()
    P_limit = torch.matrix_power(P, 100)
    pi = P_limit[0] / P_limit[0].sum()
    eps = 1e-9
    row_ent = - (P * torch.log(P + eps)).sum(dim=1)
    return (pi * row_ent).sum().item()

def compute_oracle_loss(x, y, transition_matrix):
    device = x.device
    oracle_emb = torch.nn.Embedding.from_pretrained(transition_matrix.to(device), freeze=True)
    true_dists = oracle_emb(x)
    true_probs = true_dists.gather(dim=-1, index=y.unsqueeze(-1)).squeeze(-1)
    return -torch.log(true_probs + 1e-9).mean()

def compute_shannon_entropy(att):
    return -(att * torch.log(att + 1e-9)).sum(dim=-1).mean(dim=[0, 1])

def compute_markov_score(att, k=1):
    B, nh, T, _ = att.size()
    score = 0
    for t in range(k, T): score += att[:, :, t, t-k].sum(dim=0)
    return score / B

def compute_state_entropy(att, states, n_states):
    B, nh, T, _ = att.size()
    probs = torch.zeros((B, nh, T, n_states), device=att.device)
    target_s = states.view(B, 1, 1, T).expand(B, nh, T, T)
    probs.scatter_add_(dim=-1, index=target_s, src=att)
    return -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean(dim=[0, 1])

class AttentionAnalyzer:
    def analyze(self, att_layers, extra_info=None, config=None):
        report = {'shannon': [], 'markov': [], 'syntax': [], 'topic': []}
        for att in att_layers:
            report['shannon'].append(compute_shannon_entropy(att).cpu().numpy())
            report['markov'].append(compute_markov_score(att).cpu().numpy())
            if extra_info and 's' in extra_info:
                report['syntax'].append(compute_state_entropy(att, extra_info['s'], config.n_states).cpu().numpy())
            if extra_info and 'z' in extra_info:
                report['topic'].append(compute_state_entropy(att, extra_info['z'], config.n_topics).cpu().numpy())
        return report
