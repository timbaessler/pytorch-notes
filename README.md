[![xdoc](https://img.shields.io/badge/Rendered%20with-xdoc-f2eecb?style=flat-square)](https://chrome.google.com/webstore/detail/xdoc/anidddebgkllnnnnjfkmjcaallemhjee)

To display math a Tex browser extension like https://github.com/nschloe/xdoc can be used.



# Recurrent Neural Networks
Consider  sequential data input $`X_t \in \mathbb{R}^{nxk}`$ at time $`t`$ where $`n`$ denotes the batch size,  $`k`$ the number of features. For forecasting sequential data this repository shows the implementation of Recurrent Neural Networks (RNNs) using the deep learning library PyTorch. 

## Vanilla RNN
`torch.nn.RNN`is a Elman RNN layer implementing a full unrolling of sequential data input.
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
* $`W_{xh} \in \mathbb{R}^{k \times h}`$
* $`H_{t-1} \in \mathbb{R}^{n \times h}`$
* $`W_{hh} \in \mathbb{R}^{h \times h}`$
* $`b_{h} \in \mathbb{R}^{1 \times h}`$

with the two matrix multiplications resulting in two $`nxh`$ matrices. Therefore addition results in $`H_{t} \in \mathbb{R}^{n \times h}`$. Then this hidden state is



### RNN Unrolling

Consider the length of the sequential data input has length $`l`$, or $`X \in \mathbb{R}^{l \times n \times k} `$. Since the Elman-Unit uses for each time step $`t`$ the previous hidden state $`H_{t-1}`$, it can be unrolled to $`l`$ hidden layers however using the same weights $`W_{xh}`$ and $`W_{hh}`$. Therefore the network can be seen as a copy for every step.

![alt text](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

> **"RNNs, once unfolded in time, can be seen as very deep feedforward networks in which all the layers share the same weights."** [LeCun et. al 2015]( https://www.nature.com/articles/nature14539.epdf)



### Input and Output
`torch.nn.RNN` completes a full loop of $`X`$ from $`t=0, t=1, ..., t=l`$, after $`l`$ steps returning both the full hidden state matrix $`H \in \mathbb{R}^{l \times h}`$ as well as the last hidden state which also may include the number of layers $`d`$ in case of stacked RNNs $`h_n \in \mathbb{R}^{ \text{num_layers} \times n \times h}`$


