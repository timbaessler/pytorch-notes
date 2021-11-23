import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Parameters
n = 1
l = 5
h = 2
k = 3
num_layers = 1
# Inputs
x = torch.randn(l, n, k)
h0 = torch.randn(num_layers, n, h)
# Model
rnn = nn.RNN(k, h, num_layers)
# Model Output
out, hn = rnn(x, h0)
# out: (l, n, h), hn: (num_layers, n, h)
