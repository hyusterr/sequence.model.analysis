import torch
import torch.nn.functional as F
from data import MarkovChainDataset
from model import MinModel, RPEModel, StandardTransformer # 假設你把之前的三個模型存在 model.py
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
CONFIG = {
    'vocab_size': 2,
    'd_model': 16,
    'n_layer': 2,
    'n_head': 1,
    'seq_len': 100,
    'batch_size': 64,
    'max_iters': 3000,
    'eval_interval': 100, # 每幾步做一次 evaluation
    'eval_iters': 20,     # Evaluation 要跑幾個 batch 取平均
    'model_type': 'rpe', # Options: 'min_model', 'rpe', 'standard'
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

@torch.no_grad()
def evaluate(model, dataset, eval_iters):
    """
    計算 Testing Set (Fresh Data) 上的指標
    """
    model.eval()
    losses = {'ce': [], 'kl': []}
    
    # 建立一個臨時的 iterator 來取 eval_iters 個 batch
    data_iter = iter(dataset)
    
    for _ in range(eval_iters):
        x, y, true_probs = next(data_iter)
        x, y, true_probs = x.to(CONFIG['device']), y.to(CONFIG['device']), true_probs.to(CONFIG['device'])
        
        logits, _ = model(x) # Logits: (B, T, K)
        
        # 1. Test Cross Entropy Loss (Model vs Real Token)
        # 用來衡量實際預測準確度
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses['ce'].append(ce_loss.item())
        
        # 2. Test KL Divergence (Model vs Oracle Distribution)
        # 這是檢驗 Induction Head 是否完美的黃金指標
        # KL(True || Model) = sum(True * (log(True) - log(Model)))
        
        log_probs_model = F.log_softmax(logits, dim=-1) # log(Q)
        
        # PyTorch 的 KLDivLoss 預期輸入是 log_probs，Target 是 probs (或是 log_probs 視參數而定)
        # F.kl_div(input, target) -> sum(target * (log(target) - input)) if reduction='batchmean'
        # 注意: 如果 true_probs 含有 0，log(true_probs) 會爆，但 PyTorch F.kl_div 處理 target 為機率時通常比較穩
        # 為了安全，我們直接用公式算： sum(P * (log P - log Q))
        
        # 避免 true_probs 為 0 導致 log 錯誤 (加上極小值)
        log_probs_true = torch.log(true_probs + 1e-9)
        
        kl_pointwise = true_probs * (log_probs_true - log_probs_model)
        kl_div = kl_pointwise.sum(dim=-1).mean() # Average over Batch and Sequence
        
        losses['kl'].append(kl_div.item())
        
    model.train()
    return {k: np.mean(v) for k, v in losses.items()}

def main():
    print(f"Training {CONFIG['model_type']} on {CONFIG['device']}...")
    
    # Init Data
    train_ds = MarkovChainDataset(
        vocab_size=CONFIG['vocab_size'], 
        seq_len=CONFIG['seq_len'], 
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device']
    )
    train_iter = iter(train_ds)
    
    # Init Model
    if CONFIG['model_type'] == 'min_model':
        model = MinModel(CONFIG['vocab_size'], CONFIG['seq_len'])
    elif CONFIG['model_type'] == 'rpe':
        model = RPEModel(CONFIG['vocab_size'], d_model=CONFIG['d_model'], n_layer=CONFIG['n_layer'], n_head=CONFIG['n_head'], max_len=CONFIG['seq_len'])
    else:
        model = StandardTransformer(CONFIG['vocab_size'], d_model=CONFIG['d_model'], n_layer=CONFIG['n_layer'], n_head=CONFIG['n_head'], max_len=CONFIG['seq_len'])
        
    model.to(CONFIG['device'])
    
    # Optimizer
    # MinModel 需要特殊的 LR 設定，如果是其他模型則用 AdamW
    if CONFIG['model_type'] == 'min_model':
        optimizer = model.configure_optimizers(learning_rate=0.5) # MinModel 通常需要大 LR
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Logging Lists
    history = {
        'iter': [],
        'train_loss': [],
        'test_ce': [],
        'test_kl': []
    }

    # Training Loop
    for i in range(CONFIG['max_iters']):
        
        # 1. Training Step
        x, y, _ = next(train_iter) # Training 時不需要 True Probs
        x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
        
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        
        # 2. Evaluation
        if i % CONFIG['eval_interval'] == 0:
            metrics = evaluate(model, train_ds, CONFIG['eval_iters'])
            
            history['iter'].append(i)
            history['train_loss'].append(loss.item())
            history['test_ce'].append(metrics['ce'])
            history['test_kl'].append(metrics['kl'])
            
            print(f"Iter {i:4d} | Train CE: {loss.item():.4f} | Test CE: {metrics['ce']:.4f} | Test KL: {metrics['kl']:.4f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['iter'], history['test_ce'], label='Test Cross Entropy')
    plt.plot(history['iter'], history['test_kl'], label='Test KL Div (to Oracle)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Induction Head Phase Transition ({CONFIG["model_type"]})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
