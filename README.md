[![xdoc](https://img.shields.io/badge/Rendered%20with-xdoc-f2eecb?style=flat-square)](https://chrome.google.com/webstore/detail/xdoc/anidddebgkllnnnnjfkmjcaallemhjee)

To display math a Tex browser extension like https://github.com/nschloe/xdoc can be used.



# Recurrent Neural Networks
Consider  sequential data input $`X_t \in \mathbb{R}^{nxk}`$ at time $`t`$ where $`n`$ denotes the batch size,  $`k`$ the number of features. For forecasting sequential data this repository shows the implementation of Recurrent Neural Networks (RNNs) using the deep learning library PyTorch. 

## Vanilla RNN


```python
import torch
import torch.nn as nn

k = 3 # The number of input features
h = 2 # The number of features in the hidden state

rnn = nn.RNN(k, h)

```
where the RNN layer computes
```math
H_t = \text{tanh} \left( X W_{xh}+ H_{t-1} W_{hh} + b_{h} \right)
```
where
* 
* $`W_{xh} \in \mathbb{R}^{kxh}`$
* $`H_{t-1} \in \mathbb{R}^{nxh}`$
* $`W_{hh} \in \mathbb{R}^{hxh}`$
* $`b_{h} \in \mathbb{R}^{1xh}`$

with the two matrix multiplications resulting in two $`nxh`$ matrices. Therefore addition results in $`H_{t} \in \mathbb{R}^{nxh}`$.


where the first matrix multiplication results in a $`(TxH)`$ matrix and the second matrix multiplication in a $`(HxH)`$ matrix. Addition results in a $`(TxH)`$ matrix.
