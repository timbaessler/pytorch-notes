import torch
import torch.nn as nn


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
		out, hn = self.rnn(x, h0)
		# out: (l, n, h), hn: (num_layers, n, h)
		out = self.fc(out[-1])
		return out