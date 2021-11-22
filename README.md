[![xdoc](https://img.shields.io/badge/Rendered%20with-xdoc-f2eecb?style=flat-square)](https://chrome.google.com/webstore/detail/xdoc/anidddebgkllnnnnjfkmjcaallemhjee)

To display math a Tex browser extension like https://github.com/nschloe/xdoc can be used.



# Recurrent Neural Networks
Consider a $`(b, t, k)`$ dimensional sequential data input where $`b`$ denotes the batch size, $`t`$ the sequence length, $`k`$ the number of features. For forecasting sequential data this repository shows the implementation of Recurrent Neural Networks (RNNs) using the deep learning library PyTorch. 

## Vanilla RNN


```python
import torch
import torch.nn as nn

rnn = nn.RNN(k, h)

```
where the RNN layer computes
```math
H_t = \text{tanh} \left( X_t W_{xh}+ W_{hh} + b_{h} \right)
```
where
* $`X_t \in \mathbb{R}^{txd}`$ 
* $`W_{xh} \in \mathbb{R}^{kxh}`$



where the first matrix multiplication results in a $`(TxH)`$ matrix and the second matrix multiplication in a $`(HxH)`$ matrix. Addition results in a $`(TxH)`$ matrix.
