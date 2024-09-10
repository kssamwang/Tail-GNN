import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class Generator(nn.Module):
	def __init__(self, in_features, std, ablation):
		super(Generator, self).__init__()
		self.g = nn.Linear(in_features, in_features, bias=True)
		self.std = std
		self.ablation = ablation

	def forward(self, ft):
		if self.training:
			mean = torch.zeros(ft.shape, device='cuda')
			ft = torch.normal(mean, 1.)
		h_s = F.elu(self.g(ft)) 
		return h_s