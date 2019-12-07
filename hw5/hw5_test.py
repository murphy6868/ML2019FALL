import os, sys
import numpy as np
import pandas as pd
import spacy
import string
import torch
import pickle
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from gensim import corpora
from gensim.test.utils import common_texts, get_tmpfile
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

test_x = pd.read_csv(sys.argv[1])

test_x = test_x['comment'].values.tolist()


for i in range(np.shape(test_x)[0]):
	test_x[i] = clean_doc(test_x[i])




with open("pretrained", "rb") as f:
	pretrained = pickle.load(f)
with open("w2v", "rb") as f:
	w2v = pickle.load(f)

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


test_x_idx = []
for i in test_x:
	test_x_idx.append(myVocab.trim_pad(myVocab.convert_tokens_to_indices(i), 60))
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



test_loader = DataLoader(test_x_idx, batch_size=64, shuffle=False)


model = Example_Net(pretrained_embedding=pretrained, hidden_size=vector_space, n_layers=6, bidirectional=True, dropout=0.5, padding_idx=len(w2v)-2)
model.load_state_dict(torch.load('hw5.pth'))
use_gpu = torch.cuda.is_available()
if use_gpu:
	print("True")
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
	ans.to_csv(sys.argv[2], index = False)

'''
wtf = ["Today is hot, but I am happy.", "I am happy, but today is hot."]
for i in range(2):
	wtf[i] = clean_doc(wtf[i])
wtf_idx = []
for i in wtf:
	wtf_idx.append(myVocab.trim_pad(myVocab.convert_tokens_to_indices(i), 60))
wtf_idx = np.array(wtf_idx)
wtf_loader = DataLoader(wtf_idx, batch_size=64, shuffle=False)

for idx, (vec) in enumerate(wtf_loader):
	if use_gpu:
		vec = vec.cuda()
	output = model(vec)
	print(output.cpu().detach().numpy().tolist())
	predict = torch.max(output, 1)[1]
	predict = predict.cpu().numpy().tolist()
	print(predict)
'''