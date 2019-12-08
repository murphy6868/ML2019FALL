import os, sys
import numpy as np
import pandas as pd
import spacy
import string
from collections import Counter
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec
class Example_Net(nn.Module):
	def __init__(self, pretrained_embedding, hidden_size, n_layers, bidirectional, dropout, padding_idx):
		super(Example_Net, self).__init__()
		
		pretrained_embedding = torch.FloatTensor(pretrained_embedding)
		self.embedding = nn.Embedding(
			pretrained_embedding.size(0),
			pretrained_embedding.size(1),
			padding_idx=padding_idx)
		# Load pretrained embedding weight
		self.embedding.weight = torch.nn.Parameter(pretrained_embedding)

		self.rnn = nn.LSTM(
			input_size=pretrained_embedding.size(1),
			hidden_size=hidden_size,
			num_layers=n_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			batch_first=True)
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size * (1+bidirectional), 2),
			nn.LogSoftmax(dim=1)
		)
		self.linear = nn.Sequential(
			nn.Linear(pretrained_embedding.size(1)*40, pretrained_embedding.size(1)*40),
		)
	def forward(self, batch):
		batch = self.embedding(batch)
		#s = batch.size()
		#batch = nn.Sequential(nn.Flatten())(batch)
		#batch = self.linear(batch)
		#batch = batch.view(s)
		output, (_, _) = self.rnn(batch)
		#print(output.size())
		output = output.mean(1)
		#print(output.size())
		logit = self.classifier(output)
		return logit

vector_space = 25

nlp = spacy.load('en_core_web_sm')
def clean_doc(doc):
	#return doc.split(" ")
	doc = nlp(doc)
	# split into tokens by white space
	tokens = [token.lemma_ for token in doc if token.is_stop == False]
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	tokens = [w for w in tokens if not w in nlp.Defaults.stop_words]
	# filter out short tokens
	tokens = [word.lower() for word in tokens if len(word) > 1]
	
	return tokens

train_x = pd.read_csv(sys.argv[1])
train_y = pd.read_csv(sys.argv[2])
test_x = pd.read_csv(sys.argv[3])

train_x = train_x['comment'].values.tolist()
test_x = test_x['comment'].values.tolist()
train_y = train_y['label'].values

vocab = Counter()
lencount = [0]*1000
lencount2 = [0]*1000
for i in range(np.shape(train_x)[0]):
	train_x[i] = clean_doc(train_x[i])
	lencount[len(train_x[i])] += 1
	vocab.update(train_x[i])
for i in range(np.shape(test_x)[0]):
	test_x[i] = clean_doc(test_x[i])
	lencount2[len(train_x[i])] += 1
	vocab.update(test_x[i])
for i in range(100):
	print(i, lencount[i])
for i in range(100):
	print(i, lencount2[i])

trainset = train_x+test_x
model_Word2Vec = Word2Vec(trainset, size=vector_space, window=5, min_count=1, workers=4)
model_Word2Vec.train(trainset, total_examples=13240+860, epochs=50)
w2v = []
pretrained = []
for _, key in enumerate(model_Word2Vec.wv.vocab):
	w2v.append((key, model_Word2Vec.wv[key]))
	pretrained.append(model_Word2Vec.wv[key])
special_tokens = ["<PAD>", "<UNK>"]
for token in special_tokens:
	w2v.append((token, [0.0] * vector_space))
	pretrained.append([0.0] * vector_space)
with open("pretrained", "wb") as f:
	pickle.dump(pretrained, f)
with open("w2v", "wb") as f:
	pickle.dump(w2v, f)

class Vocab:
	def __init__(self, w2v):
		self._idx2token = [token for token, _ in w2v]
		self._token2idx = {token: idx for idx,
						   token in enumerate(self._idx2token)}
		self.PAD, self.UNK = self._token2idx["<PAD>"], self._token2idx["<UNK>"]
	def trim_pad(self, tokens, seq_len):
		return tokens[:min(seq_len, len(tokens))] + [self.PAD] * (seq_len - len(tokens))
	def convert_tokens_to_indices(self, tokens):    
		return [
			self._token2idx[token]
			if token in self._token2idx else self.UNK
			for token in tokens]
	def __len__(self):
		return len(self._idx2token)
myVocab = Vocab(w2v)

train_x_idx = []
for i in train_x:
	train_x_idx.append(myVocab.trim_pad(myVocab.convert_tokens_to_indices(i), 40))
train_x_idx = np.array(train_x_idx)
test_x_idx = []
for i in test_x:
	test_x_idx.append(myVocab.trim_pad(myVocab.convert_tokens_to_indices(i), 40))
test_x_idx = np.array(test_x_idx)



class hw5_dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data[idx][0]
        label = self.data[idx][1]
        return feature, label
all_dataset = list(zip(train_x_idx, train_y))
train_dataset = hw5_dataset(all_dataset[:13000])#[:13000]
valid_dataset = hw5_dataset(all_dataset[13000:])#[13000:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_x_idx, batch_size=64, shuffle=False)


model = Example_Net(pretrained_embedding=pretrained, hidden_size=vector_space, n_layers=6, bidirectional=True, dropout=0.5, padding_idx=len(w2v)-2)
use_gpu = torch.cuda.is_available()
if use_gpu:
	print("True")
	model.cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85)
loss_fn = nn.CrossEntropyLoss()
num_epoch = 25
train_loss_trace = []
train_acc_trace = []
valid_loss_trace = []
valid_acc_trace = []
for epoch in range(num_epoch):
	model.train()
	for param_group in optimizer.param_groups:
		print("lr:", param_group['lr'])
	pred = []
	labe = []
	train_loss = []
	train_acc = []
	for idx, (vec, label) in enumerate(train_loader):
		if use_gpu:
			vec = vec.cuda()
			label = label.cuda()
		optimizer.zero_grad()
		
		output = model(vec)
		loss = loss_fn(output, label)
		loss.backward()
		optimizer.step()
		
		predict = torch.max(output, 1)[1]
		acc = np.mean((label == predict).cpu().numpy())
		train_acc.append(acc)
		train_loss.append(loss.item())
	print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))
	scheduler.step()

	model.eval()
	with torch.no_grad():
		valid_loss = []
		valid_acc = []
		for idx, (vec, label) in enumerate(valid_loader):
			if use_gpu:
				vec = vec.cuda()
				label = label.cuda()
			output = model(vec)
			loss = loss_fn(output, label)
			predict = torch.max(output, 1)[1]
			acc = np.mean((label == predict).cpu().numpy())
			predict = predict.tolist()
			label = label.tolist()
			for j in predict:
				pred.append(j)
			for j in label:
				labe.append(j)
			valid_loss.append(loss.item())
			valid_acc.append(acc)
		print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
	train_loss_trace.append(np.mean(train_loss))
	train_acc_trace.append(np.mean(train_acc))
	valid_loss_trace.append(np.mean(valid_loss))
	valid_acc_trace.append(np.mean(valid_acc))
	np.save("train_loss_trace", train_loss_trace)
	np.save("train_acc_trace", train_acc_trace)
	np.save("valid_loss_trace", valid_loss_trace)
	np.save("valid_acc_trace", valid_acc_trace)	
torch.save(model.state_dict(), "hw5.pth")

'''
if use_gpu:
	model.cuda()
model.eval()
with torch.no_grad():
	result = []
	for idx, (vec) in enumerate(test_loader):
		if use_gpu:
			vec = vec.cuda()
		output = model(vec)
		predict = torch.max(output, 1)[1]
		predict = predict.cpu().numpy().tolist()
		for j in predict:
			result.append(j)

	ans = pd.DataFrame({"id" : range(0, len(result))})
	ans['label'] = result
	print(ans)
	ans.to_csv("ans.csv", index = False)
'''
