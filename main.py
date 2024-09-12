import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime, time
import os, argparse
from torch_geometric.typing import SparseTensor
from utils import *
from layers import Discriminator
from models import TailGNN
from tools import to_edge_index, to_sparse_edge_index, train_disc, train_embed, test


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default='actor', help='dataset')
	parser.add_argument("--hidden", type=int, default=32, help='hidden layer dimension')
	parser.add_argument("--eta", type=float, default=0.1, help='adversarial constraint')
	parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
	parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
	parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
	parser.add_argument("--k", type=int, default=5, help='num of node neighbor')
	parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
	parser.add_argument("--arch", type=int, default=1, choices=[1,2,3], help='1: gcn, 2: gat, 3: graphsage')
	parser.add_argument("--seed", type=int, default=0, help='Random seed')
	parser.add_argument("--epochs", type=int, default=1000, help='Epochs')
	parser.add_argument("--patience", type=int, default=200, help='Patience')
	parser.add_argument("--id", type=int, default=0, help='gpu ids')
	parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')
	parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
	parser.add_argument("--r_ver", type=int, default=1, choices=[1,2], help='version of Relation Layer')
	args = parser.parse_args()
	print(str(args))
	return args

def load_data(args):
	cuda = torch.cuda.is_available()
	torch.manual_seed(args.seed)
	if cuda:
		torch.cuda.manual_seed(args.seed)
		torch.cuda.set_device(args.id)
	device = 'cuda' if cuda else 'cpu'
	
	save_path = 'saved_model/' + args.dataset
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	
	cur_time = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
	save_path = os.path.join(save_path, cur_time)
	if not os.path.exists(save_path):
		os.mkdir(save_path)
	
	features, adj, labels, idx = data_process.load_dataset(args.dataset, k=args.k)
	adj = to_edge_index(adj)
	features = torch.FloatTensor(features)
	labels = torch.LongTensor(np.argmax(labels, 1))
	
	tail_adj = data_process.link_dropout(adj, idx[0])
	if args.dataset in ['cs-citation', 'amazon']:
		adj = to_sparse_edge_index(adj)
		tail_adj = to_sparse_edge_index(tail_adj)

	idx_train = torch.LongTensor(idx[0])
	idx_val = torch.LongTensor(idx[1])
	idx_test = torch.LongTensor(idx[2])
	if args.dataset == 'email':
		idx_train = torch.LongTensor(np.genfromtxt('dataset/' + args.dataset + '/train.csv') - 1)
		idx_test = torch.LongTensor(np.genfromtxt('dataset/' + args.dataset + '/test.csv') - 1)

	if device == 'cuda':
		features = features.cuda()
		labels = labels.cuda()
		adj = adj.cuda()
		tail_adj = tail_adj.cuda()
	
	h_labels = torch.full((len(idx_train), 1), 1.0, device=device)
	t_labels = torch.full((len(idx_train), 1), 0.0, device=device)

	return {
		'features': features, 'adj': adj, 'tail_adj': tail_adj, 'labels': labels,
		'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test,
		'h_labels': h_labels, 't_labels': t_labels,
		'device': device, 'save_path': save_path
	}

def train_model(args, data):
	criterion = nn.BCELoss()
	device = data['device']
	nfeat = data['features'].shape[1]
	nclass = data['labels'].max().item() + 1
	embed_model = TailGNN(nfeat=nfeat, nclass=nclass, params=args, device=device)
	optimizer = optim.Adam(embed_model.parameters(), lr=args.lr, weight_decay=args.lamda)

	disc = Discriminator(nclass)
	optimizer_D = optim.Adam(disc.parameters(), lr=args.lr, weight_decay=args.lamda)

	if device == 'cuda':
		embed_model.cuda()
		disc.cuda()

	best_acc, best_loss, cur_step = 0.0, 10000.0, 0

	t_total = time.time()
	for epoch in range(args.epochs):
		t = time.time()

		# 这里原作是更新2次optimizer_D，但是好像更新1次更好
		train_disc(disc, embed_model, optimizer_D, criterion, data['features'], data['adj'], data['tail_adj'], data['h_labels'], data['t_labels'], data['idx_train'])
		# train_disc(disc, embed_model, optimizer_D, criterion, data['features'], data['adj'], data['tail_adj'], data['h_labels'], data['t_labels'], data['idx_train'])

		Loss, acc_train, loss_val, acc_val = train_embed(args, disc, embed_model, optimizer, criterion, data['features'], data['adj'], data['tail_adj'], data['labels'], data['t_labels'], data['idx_val'], data['idx_train'])

		epoch_time = time.time() - t
		log = f'Epoch: {epoch + 1:d} loss_train: {Loss[0].item():.4f} loss_val: {loss_val:.4f} acc_train: {acc_train:.4f} acc_val: {acc_val:.4f} epoch_time: {epoch_time:.3f}'
		print(log)

		if acc_val >= best_acc:
			torch.save(embed_model, os.path.join(data['save_path'], 'model.pt'))
			best_acc, best_loss, cur_step = acc_val, min(loss_val.cpu().numpy(), best_loss), 0
			print('Model saved!')
		else:
			cur_step += 1
			if cur_step == args.patience:
				print(f'Early Stopping at epoch {epoch} loss {best_loss:.4f} acc {best_acc:.4f}')
				break
	
	print("Training Finished!")
	print(f"Total time elapsed: {time.time() - t_total:.4f}s")


def test_model(args, data):
	print('Test ...')
	embed_model = torch.load(os.path.join(data['save_path'], 'model.pt'))
	test(embed_model, data['features'], data['adj'], data['labels'], data['idx_test'])


if __name__ == "__main__":
	args = parse_args()
	data = load_data(args)
	train_model(args, data)
	test_model(args, data)