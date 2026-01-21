import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import datasets


@torch.no_grad()
def test_last_token(models, dataset, device, size = 1000):
    batch_size = 64*4
    num_symbols = dataset.num_symbols
    for model in models:
        model.eval()
    loader = DataLoader(dataset, batch_size, num_workers=8, drop_last=False)
    num_samples = len(dataset)
    size = min(size, num_samples)
    num_batches = size//batch_size
    n = dataset.n+1
    size = num_batches * batch_size

    #set up a bunch of tensors for various outputs
    model_true_loss = torch.zeros((len(models)), device=device)
    ngram_losses = torch.zeros((n+1), device = device)
    KL_divs = torch.zeros((n+1, len(models)), device = device)
    for b, (x,(probs, _)) in enumerate(loader):
        if b >= num_batches:
            break
        converted_symbols = dataset.multi_symbol_convert(x[:,1-n:])
        ground_truth = probs.view(batch_size*num_symbols**(n-1), num_symbols)[torch.arange(batch_size)*num_symbols**(n-1)+converted_symbols].to(device)

        # create guess by m-grams
        mgram_guesses = torch.zeros(n+1, x.shape[0], num_symbols,  device = device)
        for m in range(n+1):
            for i in range(x.shape[0]):
                mgram_guesses[m, i] = ngram(x[i], num_symbols, m)
            # loss of m-grams
        for m in range(n+1):
            # temp =  F.kl_div(torch.log(mgram_guesses[m]),ground_truth,reduction="none").sum(dim=(0,1)) / size
            ngram_losses[m] +=  F.kl_div(torch.log(mgram_guesses[m]),ground_truth,reduction="none").sum(dim=(0,1)) / size
            # temp =  F.cross_entropy(mgram_guesses[m], ground_truth, reduction="none").sum(dim=0) / size
        x = x.to(device)
        
        for model_id in range(len(models)):

            model = models[model_id]
            logits, _ = model(x)
            logits = logits[:, -1]
            # logits = logits.permute(0,2,1)
            #loss of model
            # model_true_loss[model_id] += F.cross_entropy(logits, ground_truth, reduction="none").sum(dim=0)/ size

            model_true_loss[model_id] += F.kl_div(F.log_softmax(logits, dim=1), ground_truth, reduction="none").sum(dim=(0,1))/ size
            # model_true_loss[model_id] += F.kl_div(torch.log(.0001+logits), ground_truth, reduction="none").sum(dim=(0,1))/ size
            for m in range(n+1):
                # print(torch.log(mgram_guesses[m]).shape, F.softmax(logits, dim=1).shape)
                KL_divs[m,model_id] += F.kl_div(F.log_softmax(logits, dim=1), mgram_guesses[m], reduction="none").sum(dim=(0,1))/size #CHECK DIM
                # KL_divs[m,model_id] += F.kl_div(torch.log(.0001+logits), mgram_guesses[m], reduction="none").sum(dim=(0,1))/size #CHECK DIM
    return model_true_loss.cpu(), ngram_losses.cpu(), KL_divs.cpu()
