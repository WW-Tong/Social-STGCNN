import numpy as np
import torch
OBS_LEN = 8
PRED_LEN = 12
NUM_WORKERS = 0

MLP_DIM = 64
H_DIM = 64
EMBEDDING_DIM = 16
BOTTLENECK_DIM = 32
NOISE_DIM = 8

DATASET_NAME = 'eth'
NUM_ITERATIONS = 20000
NUM_EPOCHS = 500
G_LR = 1e-3
D_LR = 1e-3
G_STEPS = 1
D_STEPS = 2

MAX_PEDS = 64
BEST_K = 20
PRINT_EVERY = 250
NUM_SAMPLES = 20
NUM_SAMPLES_CHECK = 5000


ATTN_L = 900
ATTN_D = 512
ATTN_D_DOWN = 16
a=torch.randn(2,3,4)
c=torch.ones(2,3,4)

d=a*c
# d.append(a)
# d.append(b)
# d.append(c)
# b = b.repeat(3,1)
# d=torch.cat(d,dim=0)
# b = torch.sum(a,dim=0)/2
# b = b*3
print(a)
print(d.shape)
# print(d)
print(d)
# print(c.sum(dim=0))
# print(a.sum(dim=2).sum(dim=1))


