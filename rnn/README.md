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



### Input and Output of RNN Class
`torch.nn.RNN` completes a full loop of $`X`$ from $`t=0, t=1, ..., t=l`$, after $`l`$ steps returning both the full hidden state matrix $`H \in \mathbb{R}^{l \times h}`$ as well as the last hidden state which also may include the number of layers in case of stacked RNNs $`h_n \in \mathbb{R}^{ num_{layers} \times n \times h }`$.

```python
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
H, hn = rnn(x, h0)
# H: (l, n, h), hn: (num_layers, n, h)
```

### Create PyTorch Model
To create a  model for classification, the last hidden state is passed into a fully connected layer. This can be implemented by the following model:

```python
class RNNModel(nn.Module):
	def __init__(self, k, h, num_layers, num_classes):
		super(RNNModel, self).__init__()
		# Parameters
		self.k = k # The number of input features
		self.h = h # The number of features in the hidden state
		self.num_layers = num_layers # The number of recurrent layers
		# RNN
		self.rnn = nn.RNN(k, h, num_layers)
		# Fully Connected Layer
		self.fc = nn.Linear(h, num_classes)

	def forward(self, x):
		# H0 Initialization
		h0 = torch.autograd.Variable(torch.zeros, self.num_layers, x.size(0), self.h)
		# x: (l, n, k), h0: (num_layers, n, h)
		H, hn = self.rnn(x, h0)
		# out: (l, n, h), hn: (num_layers, n, h)
		out = self.fc(out[-1])
		return out
```


