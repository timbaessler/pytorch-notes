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

with the two matrix multiplications resulting in two $`nxh`$ matrices. Therefore addition results in $`H_{t} \in \mathbb{R}^{n \times h}`$.


### RNN Unrolling

Consider the length of the sequential data input has length $`l`$, or $`X \in \mathbb{R}^{l \times n \times k} `$. Since the Elman-Unit uses for each time step $`t`$ the previous hidden state, the Vanilla RNN can be unrolled to a simple neural network with $`l`$ hidden layers.

![alt text](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)



### Input

 

### Output 
The module `torch.nn.RNN` will complete a full loop of $`X`$ from 
