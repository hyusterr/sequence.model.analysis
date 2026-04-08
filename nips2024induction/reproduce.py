import training_pipeline
import datasets
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import graphing_pipeline
import test_error
import utils
import importlib
from mingpt.utils import set_seed
sns.set_context("paper", font_scale=1.5)
matplotlib.rcParams.update({'xtick.labelsize': 12})
matplotlib.rcParams.update({'ytick.labelsize': 12})
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# training_pipeline.device = "cpu"
device = training_pipeline.device

conf = training_pipeline.get_default_config()
histories = []
datem = []
bigram_data = []
unigram_data = []
seeds = [0]
path = "blog"
def get_test_sets(config):
  bigram_dataset = datasets.doubly_stochastic('test', config.block_size+1, num_symbols = config.vocab_size,)
  unigram_dataset = datasets.unigram('test', config.block_size+1, num_symbols=config.vocab_size)
  
  return bigram_dataset, unigram_dataset


n = 2
from torch.profiler import profile, record_function, ProfilerActivity

# from datasets import ngrams
conf = training_pipeline.get_default_config()
histories, datem, bigram_data, unigram_data = [], [], [], []
# conf.model_type = f'one head (attention only, corrected rel pos)'
# conf.model_type = "transformer"
conf.model_type = 'Attention-Only Relative positions Transformer'
print(f"n={n}")
conf.vocab_size = 2
conf.n_head = n - 1
conf.n_embd = 16 * conf.n_head #* 3
conf.n_layer = 2
conf.max_iters = 2000
conf.n = n
conf.block_size = 100
conf.batch_size = 64
conf.num_workers = 6
# conf.learning_rate = 4e-3
conf.learning_rate = 5e-4
conf.dataset = datasets.ngrams('train', n, conf.block_size+1, conf.vocab_size, last_token_only=False)
name = f"{conf.vocab_size}transformer_symb_{n}gram"
test_dataset = datasets.ngrams('test', n, conf.block_size+1, conf.vocab_size, size = int(1e7))
if conf.vocab_size < 4 and conf.n == 2:
    bigram_dataset, unigram_dataset = get_test_sets(conf)
for seed in seeds:
    conf.seed = seed
    set_seed(seed)
    model_history, train_loss = training_pipeline.train(conf) 
    histories.append(model_history)

plt.plot(train_loss)
plt.show()
for model_history in histories:
    data = test_error.test_last_token(model_history, test_dataset, device)
    datem.append(data)
    
    plt.plot(datem[-1][0])
    if conf.vocab_size < 4 and conf.n == 2:
        data = test_error.test_last_token(model_history, bigram_dataset, device)
        bigram_data.append(data)
        data = test_error.test_last_token(model_history, unigram_dataset, device)
        unigram_data.append(data)

axes = graphing_pipeline.test_loss(datem, conf) 
plt.legend()
# plt.savefig(f"{path}/{name}_test_loss.pdf", format='pdf', bbox_inches='tight')
for lab, i in enumerate(datem[-1][1]):
    plt.plot((i,)*len(datem[-1][0]), label = f"{lab}")
plt.legend()
plt.show()

fig, axes = graphing_pipeline.pos_encode_graph(model_history, datem, conf)
# plt.savefig(f"{path}/{name}_pos.pdf", format='pdf', bbox_inches='tight')
plt.show() 
if conf.vocab_size < 4 and conf.n == 2:
    fig, axes = graphing_pipeline.out_of_distribution(datem, unigram_data, bigram_data, conf)
    # plt.savefig(f"{path}/{name}_loss.pdf", format='pdf', bbox_inches='tight')
    plt.show()
axes = graphing_pipeline.similarity(datem, conf)
# plt.savefig(f"{path}/{name}_similarity.pdf", format='pdf', bbox_inches='tight')
plt.show()
