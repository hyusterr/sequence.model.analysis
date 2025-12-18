import torch
from torch.utils.data import DataLoader
from model import Transformer, TransformerConfig
from dataset import get_dataset
from utils import AttentionAnalyzer, compute_theoretical_entropy_rate, compute_oracle_loss
import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 1. Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Experiment Settings
DATA_TYPE = 'markov' # Options: 'markov', 'general_markov', 'hmm', 'hmm_lda'
VOCAB_SIZE = 50
BLOCK_SIZE = 32

# HMM/LDA Specifics
N_STATES = 5
N_TOPICS = 3

# Model Settings
IS_CAUSAL = True
N_LAYER = 2
N_HEAD = 2
N_EMBD = 64

# Training Settings
BATCH_SIZE = 64
MAX_ITERS = 1000      # 稍微加長一點，比較能看到 Phase Transition
LEARNING_RATE = 1e-3
EVAL_INTERVAL = 50    # 每 50 步分析一次 Attention 行為
SAVE_DIR = "results"  #圖片存檔資料夾

def save_experiment_results(history, save_dir):
    """
    將訓練過程的數據畫成圖並存檔
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Loss & Regret Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['iter'], history['loss'], label='Train Loss', alpha=0.7)
    if 'oracle_loss' in history and history['oracle_loss']:
        # 畫一條水平線代表理論極限
        oracle = history['oracle_loss'][0] # 假設 oracle 是常數
        plt.axhline(y=oracle, color='r', linestyle='--', label=f'Theoretical Bound ({oracle:.3f})')
    
    plt.title(f"Training Loss Curve ({DATA_TYPE})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 2. Attention Entropy Evolution (Layer-wise)
    # 我們想看每一層的 Entropy 隨時間怎麼變化
    # history['layer_entropies'] 結構: [step_idx][layer_idx] -> scalar (avg over heads)
    
    if len(history['layer_entropies']) > 0:
        n_layers = len(history['layer_entropies'][0])
        steps = history['eval_steps']
        
        plt.figure(figsize=(10, 5))
        for l in range(n_layers):
            # 取出這一層在所有時間點的數值
            vals = [snapshot[l] for snapshot in history['layer_entropies']]
            plt.plot(steps, vals, marker='.', label=f'Layer {l}')
            
        plt.title("Shannon Entropy Evolution (Sparsity)")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy (Lower=Sharper)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "entropy_evolution.png"))
        plt.close()

    # 3. Markov Score Evolution (Alignment to t-1)
    if len(history['layer_markov_scores']) > 0:
        n_layers = len(history['layer_markov_scores'][0])
        steps = history['eval_steps']
        
        plt.figure(figsize=(10, 5))
        for l in range(n_layers):
            vals = [snapshot[l] for snapshot in history['layer_markov_scores']]
            plt.plot(steps, vals, marker='.', label=f'Layer {l}')
            
        plt.title("Markov Score Evolution (Prob on t-1)")
        plt.xlabel("Iteration")
        plt.ylabel("Score (Higher=Local Copy)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "markov_score_evolution.png"))
        plt.close()

    # 4. Syntax/Topic Entropy Evolution (如果有 HMM/LDA)
    # 這裡示範 Syntax Entropy
    if len(history['layer_syntax_entropies']) > 0:
        n_layers = len(history['layer_syntax_entropies'][0])
        steps = history['eval_steps']
        
        plt.figure(figsize=(10, 5))
        for l in range(n_layers):
            vals = [snapshot[l] for snapshot in history['layer_syntax_entropies']]
            plt.plot(steps, vals, marker='.', label=f'Layer {l}')
            
        plt.title("Syntax Entropy Evolution (HMM State Awareness)")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "syntax_entropy_evolution.png"))
        plt.close()

    print(f"Figures saved to {save_dir}/")

def main():
    print(f"Running Experiment: {DATA_TYPE} on {device}")
    
    # --- Data Setup ---
    dataset_kwargs = {}
    if DATA_TYPE == 'general_markov':
        dataset_kwargs['alpha'] = 0.1
    elif DATA_TYPE == 'hmm':
        dataset_kwargs['n_states'] = N_STATES
    elif DATA_TYPE == 'hmm_lda':
        dataset_kwargs['n_states'] = N_STATES
        dataset_kwargs['n_topics'] = N_TOPICS
        
    train_dataset = get_dataset(DATA_TYPE, VOCAB_SIZE, BLOCK_SIZE, **dataset_kwargs)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # --- Theoretical Baseline ---
    generator = train_dataset.generator
    theoretical_loss = compute_theoretical_entropy_rate(generator)
    
    # --- Model Setup ---
    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        is_causal=IS_CAUSAL
    )
    # Bind extra config for analysis
    config.n_states = N_STATES
    config.n_topics = N_TOPICS
    
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Logging Structure ---
    history = {
        'iter': [],
        'loss': [],
        'oracle_loss': [], # Store theoretical bound for plotting
        'eval_steps': [],
        'layer_entropies': [],       # List of [L0_val, L1_val...]
        'layer_markov_scores': [],
        'layer_syntax_entropies': []
    }

    # --- Training Loop ---
    model.train()
    iter_loader = iter(train_loader)
    
    # 用來計算 Oracle Loss 的 transition matrix (如果有的話)
    trans_mat = getattr(generator, 'trans_mat', None)
    
    analyzer = AttentionAnalyzer()

    pbar = tqdm.tqdm(range(MAX_ITERS))
    for i in pbar:
        try:
            x, y, extra = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            x, y, extra = next(iter_loader)

        x, y = x.to(device), y.to(device)

        # 1. Forward & Backward
        logits, loss, _ = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 2. Basic Logging
        history['iter'].append(i)
        history['loss'].append(loss.item())
        if theoretical_loss:
            history['oracle_loss'].append(theoretical_loss)
        
        regret_msg = ""
        # 簡單計算 Regret (Optional, 僅顯示用)
        if trans_mat is not None and DATA_TYPE in ['markov', 'general_markov']:
            with torch.no_grad():
                oracle = compute_oracle_loss(x, y, trans_mat)
                regret = loss.item() - oracle.item()
                regret_msg = f" | Regret: {regret:.4f}"

        pbar.set_description(f"Loss: {loss.item():.4f}{regret_msg}")

        # 3. Periodic Evaluation (Analysis)
        if i % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                # 重新 forward 一次拿 attention (或者利用當前的 batch，但記得要 eval mode)
                # 這裡為了方便，直接用當前 batch
                
                # 需要 extra info 裡的 tensor 也在 device 上
                extra_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in extra.items()}
                
                _, _, att_layers = model(x, return_att_weights=True)
                
                # 執行分析
                report = analyzer.analyze(att_layers, extra_info=extra_gpu, config=config)
                
                # 儲存 Snapshot (取各個 Head 的平均值，方便畫成一條線)
                # report['shannon_entropy'] 是一個 list of arrays, len=n_layers, array shape=(n_heads,)
                
                # Helper to mean over heads
                def mean_heads(layer_data):
                    return [np.mean(layer_val) for layer_val in layer_data]

                history['eval_steps'].append(i)
                history['layer_entropies'].append(mean_heads(report['shannon_entropy']))
                history['layer_markov_scores'].append(mean_heads(report['markov_score_t1']))
                
                if report['syntax_entropy']:
                     history['layer_syntax_entropies'].append(mean_heads(report['syntax_entropy']))
            
            model.train() # 切回訓練模式

    # --- End of Training ---
    print("\nTraining Finished. Saving plots...")
    save_experiment_results(history, SAVE_DIR)

if __name__ == '__main__':
    main()
