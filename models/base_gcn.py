import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout):
		super(GCN, self).__init__()
		self.gc1 = GCNConv(nfeat, nhid)
		self.gc2 = GCNConv(nhid, nclass)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = F.relu(self.gc1(x, edge_index))
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.gc2(x, edge_index)
		return x

class GAT(nn.Module):
	def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=3):
		super(GAT, self).__init__()
		self.dropout = dropout
		self.attentions = GATConv(nfeat, nhid, heads=nheads, dropout=dropout, concat=True, negative_slope=alpha)
		self.out_att = GATConv(nhid * nheads, nclass, heads=1, dropout=dropout, concat=False, negative_slope=alpha)

	def forward(self, x, edge_index):
		x = F.dropout(x, self.dropout, training=self.training)
		x = self.attentions(x, edge_index)
		x = F.dropout(x, self.dropout, training=self.training)
		x = F.elu(self.out_att(x, edge_index))
		return x

class GraphSAGE(nn.Module):
	def __init__(self, nfeat, nhid1, nhid2, nclass, dropout=0.5):
		super(GraphSAGE, self).__init__()
		self.sage1 = SAGEConv(nfeat, nhid1)
		self.sage2 = SAGEConv(nhid1, nhid2)
		self.fc = nn.Linear(nhid2, nclass, bias=True)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = F.relu(self.sage1(x, edge_index))
		x = F.normalize(x)
		x = F.dropout(x, self.dropout, training=self.training)
		x = F.relu(self.sage2(x, edge_index))
		x = F.normalize(x)
		x = self.fc(x)
		return x