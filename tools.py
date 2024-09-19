import torch 
import torch.nn.functional as F
from torch_geometric.typing import SparseTensor
from utils import *


def normalize_output(out_feat, idx):
	sum_m = 0
	for m in out_feat:
		sum_m += torch.mean(torch.norm(m[idx], dim=1))
	return sum_m 

def to_edge_index(adj,to_sparse=False):
	# 将 adj <class 'scipy.sparse._lil.lil_matrix'> 转换为 coo_matrix
	adj_coo = adj.tocoo()

	# 从 coo_matrix 中提取行和列，构造 edge_index
	row = torch.from_numpy(adj_coo.row).long()
	col = torch.from_numpy(adj_coo.col).long()
	if to_sparse:
		# 获取对应的权重值
		values = torch.from_numpy(adj_coo.data).float()

		# 稀疏张量的大小是 (num_nodes, num_nodes)，即邻接矩阵的大小
		num_nodes = adj.shape[0]

		# 使用 SparseTensor 来创建稀疏张量
		sparse_tensor = SparseTensor(row=row, col=col, value=values, sparse_sizes=(num_nodes, num_nodes))

		return sparse_tensor  # 返回 PyTorch 的 SparseTensor
	else:
		edge_index = torch.stack([row, col], dim=0)
		return edge_index

def to_sparse_edge_index(edge_index, num_nodes=None):
	if num_nodes is None:
		num_nodes = edge_index.max().item() + 1  # 确定节点数量
	
	# 转换为 SparseTensor 格式
	sparse_tensor = SparseTensor.from_edge_index(
		edge_index, edge_attr=None, sparse_sizes=(num_nodes, num_nodes)
	)
	
	return sparse_tensor

def train_disc(disc, embed_model, optimizer_D, criterion, features, adj, tail_adj, h_labels, t_labels, batch):
	disc.train()
	optimizer_D.zero_grad()
	embed_h, _, _ = embed_model(features, adj, True)
	embed_t, _, _ = embed_model(features, tail_adj, False)

	prob_h = disc(embed_h)
	prob_t = disc(embed_t)

	# loss
	errorD = criterion(prob_h[batch], h_labels)
	errorG = criterion(prob_t[batch], t_labels)
	L_d = (errorD + errorG)/2 

	L_d.backward()
	optimizer_D.step()
	# return L_d


def train_embed(args, disc, embed_model, optimizer, criterion, features, adj, tail_adj, labels, t_labels, idx_val, batch):
	embed_model.train()
	optimizer.zero_grad()
	
	embed_h, output_h, support_h  = embed_model(features, adj, True)
	embed_t, output_t, support_t  = embed_model(features, tail_adj, False)

	# loss
	L_cls_h = F.nll_loss(output_h[batch], labels[batch])
	L_cls_t = F.nll_loss(output_t[batch], labels[batch])
	L_cls = (L_cls_h + L_cls_t)/2

	#weight regularizer
	m_h = normalize_output(support_h, batch)
	m_t = normalize_output(support_t, batch)

	prob_h = disc(embed_h)
	prob_t = disc(embed_t)

	errorG = criterion(prob_t[batch], t_labels)
	L_d = errorG
	L_all = L_cls - (args.eta * L_d) + args.mu * m_h 

	L_all.backward()
	optimizer.step()
	acc_train = metrics.accuracy(embed_h[batch], labels[batch])

	# validate:
	embed_model.eval()
	with torch.no_grad():
		_, embed_val, _ = embed_model(features, adj, False)
		loss_val = F.nll_loss(embed_val[idx_val], labels[idx_val])
		acc_val = metrics.accuracy(embed_val[idx_val], labels[idx_val])

	return (L_all, L_cls, L_d), acc_train, loss_val, acc_val


def test(embed_model, features, adj, labels, idx_test):
	embed_model.eval()
	with torch.no_grad():
		_, embed_test, _ = embed_model(features, adj, False)
		loss_test = F.nll_loss(embed_test[idx_test], labels[idx_test])

	acc_test = metrics.accuracy(embed_test[idx_test], labels[idx_test])
	f1_test = metrics.micro_f1(embed_test.cpu(), labels.cpu(), idx_test)

	log =   "Test set results: " + \
			"loss={:.4f} ".format(loss_test.item()) + \
			"accuracy={:.4f} ".format(acc_test) + \
			"f1={:.4f}".format(f1_test)

	print(log) 
