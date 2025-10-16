import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import pickle
import numpy as np
import utils

from utils import stationary_distribution, compare_test
from mingpt.utils import set_seed

import datasets
from copy import deepcopy

from mingpt.utils import CfgNode as CN

#2-layered attention only transformer
from mingpt.model import GPT
from mingpt.model_rpe import Relative_Transformer
from mingpt.min_model import min_model
from mingpt.trainer import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_default_config():
    C = CN()
    C.model_type = 'Two Layer Attention Only Transformer'
    C.n_layer = 2 # number of layers
    C.n_head = 1 # number of heads
    C.n_embd = 16 # hidden embedding dimension
    C.vocab_size = 2 # vocabulary size
    C.block_size = 100 # maximum context length

    # dropout hyperparameters
    C.embd_pdrop = 0.1
    C.resid_pdrop = 0.1
    C.attn_pdrop = 0.1

    # causal or not (causal = can only attend to left context)
    C.causal = True
    C.abs_embd = False
    C.max_iters = 2000
    C.seed = 1


    C.learning_rate = None
    C.batch_size = 64
    C.dataset = None
    C.n = 2 # n-gram size if using n-grams dataset
    C.attention_only = True
    C.num_workers = 6
    return C

def train(config=get_default_config()):
    print(f"config seed: {config.seed}")
    set_seed(config.seed)
    config.device = device
    if config.dataset is None:
        print("Using ngrams dataset")
        train_dataset = datasets.ngrams('train', config.n, config.block_size+1, config.vocab_size)
        #  use ngram.class in datasets.py
    else:
        train_dataset = config.dataset

    train_config = Trainer.get_default_config()
    train_config.max_iters = config.max_iters
    train_config.num_workers = config.num_workers
    train_config.batch_size = config.batch_size
    name = config.model_type.lower()
    if config.learning_rate is not None:
        train_config.learning_rate = config.learning_rate
    train_config.device = device
    if "test" in name:
        temp = config.model_type
        config.model_type = None

        model = GPT(config)

        config.model_type = temp
    # if 'mlp' in name:
    #     model = Relative_Transformer(config)
    elif 'transformer' in name:
        model = Relative_Transformer(config)
    elif 'minimal model' in name:
        model = min_model(config)
        

    trainer = Trainer(train_config, model, train_dataset)

    model_history = [deepcopy(model)]
    train_loss = []
    wait = max(train_config.max_iters// 5, 1)
    num_models = 200 # number of models saved (assuming number of iterations is atleast as high)
    if 'minimal model' in name and "wise" in name:
        model.Wq.requires_grad_(False)
        model.v.requires_grad_(False)
    @torch.no_grad
    def batch_end_callback(trainer):
        # trainer.optimizer.param_groups[0]['lr'] = 2e-1
        train_loss.append(trainer.loss.item())
        if trainer.iter_num % max(train_config.max_iters//num_models, 1) == 0:
            model_history.append(deepcopy(model))
            # torch.save(model.state_dict(), os.path.join(path, 'checkpoints', f'model_{trainer.iter_num}'))
        if trainer.iter_num % wait == 0:
            print(f"iter_dt {trainer.iter_dt *1000:.2f} ms; iter {trainer.iter_num}: train loss {trainer.loss.item():f}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    return model_history, train_loss

if __name__ == "__main__":
    config = get_default_config()
    model_history, train_loss = train(config)

    import matplotlib.pyplot as plt
    plt.plot(train_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.title(f"Training Loss Curve for {config.model_type} with seed {config.seed}")
    plt.savefig(f"training_loss_{config.model_type}_seed{config.seed}.png")



