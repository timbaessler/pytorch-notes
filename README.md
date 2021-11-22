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
h_t = \text{tanh} \left( W_{kh} x_t + b_{kh} + W_{hh} h_{t-1} + b_{hh} \right)
```

```math
\begin{bmatrix}
1\\
\end{bmatrix}
```
