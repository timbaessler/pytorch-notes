# Recurrent Neural Networks
Consider a $`(b, t, k)`$ dimensional sequential data input where $`b`$ denotes the batch size, $`t`$ the sequence length, $` k`$ the number of features. For forecasting sequential data this repository shows the implementation of RNNs using the deep learning library PyTorch.

## (Vanilla) Recurrent Neural Network


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
