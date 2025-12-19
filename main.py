import argparse
import torch
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import DataFactory
from model import Transformer, TransformerConfig
from utils import AttentionAnalyzer, compute_theoretical_entropy_rate, compute_oracle_loss

def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_type', type=str, default='hmm_lda', 
                    choices=['iid', 'markov', 'edelman_icl', 'hmm_lda', 'fitted_hmm_lda', 'hmm'], # 加入 hmm
                    help="Type of generative process")
    parser.add_argument('--consistency', action='store_true', 
                    help="If True, use fixed global params (LM). If False, sample params per seq (ICL).")
    parser.add_argument('--n_func_words', type=int, default=None, 
                        help="Number of syntax tokens (indices 0 to K-1)")
    parser.add_argument('--source_file', type=str, default=None, help='Source text for fitted generator')
    parser.add_argument('--vocab_size', type=int, default=50)
    parser.add_argument('--block_size', type=int, default=64)
    parser.add_argument('--n_states', type=int, default=5)
    parser.add_argument('--n_topics', type=int, default=3)
    parser.add_argument('--train_samples', type=int, default=50000)
    parser.add_argument('--val_samples', type=int, default=1000)
    parser.add_argument('--test_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    # Model
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--attn_only', action='store_true')
    parser.add_argument('--use_relative_pos', action='store_true')
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='results')
    return parser.parse_args()

def save_plots(history, args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['iter'], history['loss'], label='Train')
    plt.plot(history['iter'], history['val_loss'], label='Val')
    if history['oracle']: plt.axhline(history['oracle'][0], color='r', linestyle='--', label='Oracle')
    plt.title(f"Loss ({args.data_type})")
    plt.legend(); plt.savefig(os.path.join(args.save_dir, "loss.png")); plt.close()

    # Metrics
    metrics = ['entropies', 'markov', 'syntax', 'topic']
    for m in metrics:
        key = f'layer_{m}'
        if history[key]:
            plt.figure(figsize=(10, 5))
            for l in range(len(history[key][0])):
                vals = [snap[l] for snap in history[key]]
                plt.plot(history['eval_steps'], vals, marker='.', label=f'L{l}')
            plt.title(m.capitalize()); plt.legend(); plt.savefig(os.path.join(args.save_dir, f"{m}.png")); plt.close()

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Config: {vars(args)}")

    train_dl, val_dl, test_dl = DataFactory.get_loaders(args)
    
    config = TransformerConfig(
        vocab_size=args.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        attn_only=args.attn_only, use_relative_pos=args.use_relative_pos
    )
    # Bind extra info for utils
    config.n_states = args.n_states; config.n_topics = args.n_topics
    
    model = Transformer(config).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    analyzer = AttentionAnalyzer()
    generator = train_dl.dataset.generator
    oracle_loss = compute_theoretical_entropy_rate(generator)
    trans_mat = getattr(generator, 'trans_mat', None)

    history = {'iter': [], 'loss': [], 'val_loss': [], 'oracle': [], 'eval_steps': [], 
               'layer_entropies': [], 'layer_markov': [], 'layer_syntax': [], 'layer_topic': []}

    model.train()
    step = 0
    pbar = tqdm.tqdm(train_dl)
    
    for x, y, extra in pbar:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        
        optim.zero_grad(); loss.backward(); optim.step()
        
        step += 1
        if step % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Val Loss
                vlosses = []
                for vx, vy, _ in val_dl:
                    _, vl, _ = model(vx.to(device), targets=vy.to(device))
                    vlosses.append(vl.item())
                    if len(vlosses) > 10: break
                avg_vl = np.mean(vlosses)
                
                # Attention Analysis
                ex_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in extra.items()}
                _, _, atts = model(x, return_att_weights=True)
                rpt = analyzer.analyze(atts, ex_gpu, config)
                
                history['iter'].append(step)
                history['loss'].append(loss.item())
                history['val_loss'].append(avg_vl)
                if oracle_loss: history['oracle'].append(oracle_loss)
                history['eval_steps'].append(step)
                
                mean_h = lambda d: [np.mean(l) for l in d]
                history['layer_entropies'].append(mean_h(rpt['shannon']))
                history['layer_markov'].append(mean_h(rpt['markov']))
                if rpt['syntax']: history['layer_syntax'].append(mean_h(rpt['syntax']))
                if rpt['topic']: history['layer_topic'].append(mean_h(rpt['topic']))
                
            model.train()
            pbar.set_description(f"Loss: {loss.item():.4f} | Val: {avg_vl:.4f}")

    print("Saving..."); save_plots(history, args)

if __name__ == '__main__':
    main()
