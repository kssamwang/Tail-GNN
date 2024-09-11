import torch
from sklearn.metrics import f1_score

def accuracy(output, labels):
	preds = output.argmax(dim=1)
	correct = (preds == labels).sum().item()
	return correct / labels.size(0)

def micro_f1(output, labels, index):
	label, count = torch.unique(labels, return_counts=True)
	most_freq = label[count.argmax()]
	index = index[labels[index] != most_freq]
	preds = output.argmax(dim=1)
	return f1_score(labels[index].cpu().numpy(), preds[index].cpu().numpy(), average='micro')