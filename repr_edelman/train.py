import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

from iclmc_data import MarkovChainDataset, FixedMarkovChainDataset
from iclhmm_data import HMMDataset, FixedHMMDataset
from model import MinModel, RPEModel, StandardTransformer 

FIG_DIR = 'save_fig'
os.makedirs(FIG_DIR, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_kl(logits, target_probs):
    """計算 KL(Target || Model)"""
    log_probs_model = F.log_softmax(logits, dim=-1)
    log_probs_target = torch.log(target_probs + 1e-9)
    kl_pointwise = target_probs * (log_probs_target - log_probs_model)
    return kl_pointwise.sum(dim=-1).mean().item()

@torch.no_grad()
def offline_evaluate(model_history, val_loader, device, save_interval):
    """
    離線評估：跑遍所有存下來的模型歷史
    """
    print(f"Starting Offline Evaluation for {len(model_history)} models...")
    
    history = {
        'iter': [],
        'test_ce': [],
        'test_kl_oracle': [],
        'test_kl_god': [] # 只有 HMM 會有這個值
    }
    
    # 針對每一個存檔點的模型
    for i, model in enumerate(tqdm(model_history, desc="Evaluating History")):
        model = model.to(device)
        model.eval()
        
        ce_losses = []
        kl_oracle_losses = []
        kl_god_losses = []
        
        for batch in val_loader:
            # HMM 回傳 4 個，MC 回傳 3 個，這裡做兼容處理
            if len(batch) == 4:
                x, y, oracle_probs, god_probs = batch
                god_probs = god_probs.to(device)
            else:
                x, y, oracle_probs = batch
                god_probs = None
                
            x, y, oracle_probs = x.to(device), y.to(device), oracle_probs.to(device)
            
            logits, _ = model(x)
            
            # 1. CE Loss
            ce_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
            
            # 2. KL vs Oracle (Gambler)
            kl_oracle_losses.append(compute_kl(logits, oracle_probs))
            
            # 3. KL vs God (Hidden State Emission) - Only for HMM
            if god_probs is not None:
                kl_god_losses.append(compute_kl(logits, god_probs))
        
        # 紀錄平均結果
        step_num = i * save_interval
        history['iter'].append(step_num)
        history['test_ce'].append(np.mean(ce_losses))
        history['test_kl_oracle'].append(np.mean(kl_oracle_losses))
        if god_probs is not None:
            history['test_kl_god'].append(np.mean(kl_god_losses))
            
        # 評估完把模型搬回 CPU 或刪除以省顯存 (雖然這裡 loop 換下一個就會釋放)
        model.to('cpu') 
        
    return history

def train_experiment(config):
    set_seed(config['seed'])
    device = config['device']
    
    # --- Config Logic ---
    calculated_n_head = max(1, config['n'] - 1)
    calculated_n_embd = 16 * calculated_n_head
    print(f"Task: {config['task']} | N={config['n']} | Vocab={config['vocab_size']}")

    # --- Dataset Setup ---
    if config['task'] == 'hmm':
        train_ds = HMMDataset(num_hidden=config['num_hidden'], num_obs=config['vocab_size'], 
                              seq_len=config['seq_len'], batch_size=config['batch_size'], device=device)
        # Validation Set 先生成好，等最後再來跑
        val_ds = FixedHMMDataset(size=20000, num_hidden=config['num_hidden'], num_obs=config['vocab_size'], 
                                 seq_len=config['seq_len'], device=device)
    else: # mc
        train_ds = MarkovChainDataset(vocab_size=config['vocab_size'], seq_len=config['seq_len'], 
                                      batch_size=config['batch_size'], n=config['n'], device=device)
        val_ds = FixedMarkovChainDataset(size=20000, vocab_size=config['vocab_size'], 
                                         seq_len=config['seq_len'], n=config['n'], device=device)
    
    train_iter = iter(train_ds)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'] * 16, shuffle=False)

    # --- Model Setup ---
    if config['model_type'] == 'min_model':
        model = MinModel(config['vocab_size'], config['seq_len'])
        optimizer = model.configure_optimizers(learning_rate=0.5)
    elif config['model_type'] == 'rpe':
        model = RPEModel(config['vocab_size'], d_model=calculated_n_embd, n_layer=2, n_head=calculated_n_head, max_len=config['seq_len'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0)
    else:
        model = StandardTransformer(config['vocab_size'], d_model=calculated_n_embd, n_layer=2, n_head=calculated_n_head, max_len=config['seq_len'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    
    model.to(device)
    
    # --- Training Loop (with History) ---
    model_history = [] # 這裡會吃 RAM，但你說有 1.5TB 所以沒問題
    train_losses = []
    
    pbar = tqdm(range(config['max_iters']), desc="Training")
    for i in pbar:
        # Step
        batch = next(train_iter)
        # 這裡不管 Dataset 回傳幾個，我們只拿前兩個 (x, y) 來訓練
        x, y = batch[0], batch[1] 
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Snapshot (Deepcopy)
        if i % config['save_interval'] == 0:
            # 存到 CPU RAM 以免爆 GPU VRAM
            model_history.append(copy.deepcopy(model).to('cpu'))
            
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # 最後一步也要存
    model_history.append(copy.deepcopy(model).to('cpu'))

    # --- Offline Evaluation ---
    history = offline_evaluate(model_history, val_loader, device, config['save_interval'])
    history['train_loss'] = train_losses # 把 step loss 也存起來 (雖然長度不一樣)

    # --- Plotting ---
    plot_results(history, config)
    return history

def plot_results(history, config):
    plt.figure(figsize=(12, 6))
    
    # Plot CE
    plt.subplot(1, 2, 1)
    # 簡單 downsample train loss 讓它跟 test x-axis 對齊
    plt.plot(history['train_loss'], label='Train CE', color='blue', alpha=0.5, linewidth=1)
    plt.plot(history['iter'], history['test_ce'], label='Test CE', linewidth=2, color='gray')
    plt.title(f"Loss ({config['task'].upper()})")
    plt.legend(); plt.grid(True, alpha=0.3)
    
    # Plot KL
    plt.subplot(1, 2, 2)
    plt.plot(history['iter'], history['test_kl_oracle'], label='KL vs Oracle (Gambler)', color='orange', linewidth=2)
    
    if 'test_kl_god' in history and len(history['test_kl_god']) > 0:
         plt.plot(history['iter'], history['test_kl_god'], label='KL vs God (State)', color='green', linestyle='--')
    
    plt.title("KL divergence")
    # plt.yscale('log')
    plt.legend(); plt.grid(True, alpha=0.3, which='both')
    
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    filename = f"{config['task']}_{config['model_type']}_h{config['num_hidden']}_{timestamp}.png"
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    print(f"Figure saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mc', choices=['mc', 'hmm'])
    parser.add_argument('--num_hidden', type=int, default=4)
    parser.add_argument('--n', type=int, default=2, help='N-gram (for MC)')
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='rpe', choices=['min_model', 'rpe', 'standard'])
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=1, help='Snapshot interval')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()

    config = vars(args)
    config['seq_len'] = 100
    config['batch_size'] = 64
    # config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_experiment(config)
