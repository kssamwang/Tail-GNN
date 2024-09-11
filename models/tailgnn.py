import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.utils import degree
from layers import Relation, Relationv2, Generator

class TransGCN(nn.Module):
	def __init__(self, nfeat, nhid, g_sigma, device, ver, ablation=0):
		super(TransGCN, self).__init__()

		self.device = device
		self.ablation = ablation

		if ver == 1:
			self.r = Relation(nfeat, ablation)
		else:
			self.r = Relationv2(nfeat, nhid, ablation)

		self.g = Generator(nfeat, g_sigma, ablation)
		self.gc = GCNConv(nfeat, nhid, add_self_loops=True)

	def forward(self, x, edge_index, head):
		row, col = edge_index 

		# 计算邻居特征的加权平均
		num_neighbor = scatter_add(torch.ones_like(col, dtype=torch.float), row, dim=0, dim_size=x.size(0))

		neighbor = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

		# 输出层计算
		output = self.r(x, neighbor)

		if head or self.ablation == 2:
			# 归一化操作
			h_k = self.gc(x, edge_index)
		else:
			if self.ablation == 1:
				h_s = self.g(output)
			else:
				h_s = output

			# 基于 self_loops 的计算
			h_k = self.gc(x, edge_index)
			# 融合 h_s 和 h_k
			h_s = torch.mm(h_s, self.gc.lin.weight.T)
			h_k = h_k + h_s

			h_k = h_k / (num_neighbor + 1).unsqueeze(-1)

		return h_k, output

class TransGAT(nn.Module):
	def __init__(self, nfeat, nhid, g_sigma,device, ver, ablation=0, nheads=3, dropout=0.5, concat=True):
		super(TransGAT, self).__init__()
		
		self.device = device
		self.ablation = ablation

		if ver == 1:
			self.r = Relation(nfeat, ablation)
		else:
			self.r = Relationv2(nfeat,nhid, ablation)

		self.g = Generator(nfeat, g_sigma, ablation)
		self.gat = [GATConv(nfeat, nhid, dropout=dropout, heads=1, concat=concat, negative_slope=0.2) for _ in range(nheads)]
		for i, attention in enumerate(self.gat):
			self.add_module('attention_{}'.format(i), attention)

	def forward(self, x, edge_index, head):
		row, col = edge_index 

		# 计算邻居特征的加权平均
		neighbor = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

		output = self.r(x, neighbor)

		if head or self.ablation == 2:
			h_k = torch.cat([att(x, edge_index) for att in self.gat], dim=1)
		else:
			if self.ablation == 1:
				h_s = self.g(output)
			else:
				h_s = output
			x = x + h_s
			h_k = torch.cat([att(x, edge_index) for att in self.gat], dim=1)
		
		return h_k, output


class TransSAGE(nn.Module):
	def __init__(self, nfeat, nhid, g_sigma,device, ver, ablation=0, nheads=3, dropout=0.5, concat=True):
		super(TransSAGE, self).__init__()

		self.device = device
		self.ablation = ablation

		if ver == 1:
			self.r = Relation(nfeat, ablation)
		else:
			self.r = Relationv2(nfeat,nhid, ablation)
		self.g = Generator(nfeat, g_sigma, ablation)
		self.weight = nn.Linear(nfeat, nhid, bias=False)

	def forward(self, x, edge_index, head):
		row, col = edge_index

		neighbor = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

		output = self.r(x, neighbor)

		if head or self.ablation == 2:
			ft_input = self.weight(x)
			ft_neighbor = self.weight(neighbor)
			h_k = torch.cat([ft_input, ft_neighbor], dim=1)

		else:
			if self.ablation == 1:
				h_s = self.g(output)
			else:
				h_s = output
			
			deg = degree(edge_index[0], num_nodes=x.size(0))
			norm = deg.view(-1, 1) + 1

			neighbor = neighbor + h_s / norm
			ft_input = self.weight(x)
			ft_neighbor = self.weight(neighbor)
			h_k = torch.cat([ft_input, ft_neighbor], dim=1)

		return h_k, output 


# latent relation GCN
class TailGNN(nn.Module):
	def __init__(self, nfeat, nclass, params, device, ver=1):
		super(TailGNN, self).__init__()

		self.nhid = params.hidden
		self.dropout = params.dropout
		self.arch = params.arch

		if self.arch == 1:
			self.rel1 = TransGCN(nfeat, self.nhid, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation)
			self.rel2 = TransGCN(self.nhid, nclass, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation)
		
		elif self.arch == 2:
			nheads=3
			self.nhid = 8
			self.rel1 = TransGAT(nfeat, self.nhid, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation, nheads=nheads, dropout=self.dropout, concat=True)
			self.rel2 = TransGAT(self.nhid * nheads, nclass, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation, nheads=1, dropout=self.dropout, concat=False) 

		else:
			self.rel1 = TransSAGE(nfeat, self.nhid, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation)
			self.rel2 = TransSAGE(self.nhid * 2, nclass, g_sigma=params.g_sigma, device=device, \
								ver=ver, ablation=params.ablation)
			self.fc = nn.Linear(nclass * 2, nclass, bias=True)


	def forward(self, x, adj, head):
		
		if self.arch != 3:
			x1, out1 = self.rel1(x, adj, head)
			x1 = F.elu(x1)
			x1 = F.dropout(x1, self.dropout, training=self.training)
			x2, out2 = self.rel2(x1, adj, head)
		
		else:
			x1, out1 = self.rel1(x, adj, head)
			x1 = F.elu(x1)
			x1 = F.normalize(x1)
			x1 = F.dropout(x1, self.dropout, training=self.training)

			x2, out2 = self.rel2(x1, adj, head)
			x2 = F.elu(x2)
			x2 = F.normalize(x2)
			x2 = self.fc(x2)

		return x2, F.log_softmax(x2, dim=1), [out1, out2]

	def embed(self, x, adj): 
		x1, m1 = self.rel1(x, adj, False)
		x1 = F.elu(x1)
		x2, m2 = self.rel2(x1, adj, False)
		return F.log_softmax(x2, dim=1)
