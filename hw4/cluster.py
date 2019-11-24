import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import pandas as pd
import os, sys
#nn.Conv2d(16,16,3,2,1),
#nn.Conv2d(16,16,3,2,1),
#nn.ReLU(True),
#nn.BatchNorm2d(32,affine=False),
#nn.MaxPool2d(2, stride=1),

path_ans = sys.argv[2]
path_trainX = sys.argv[1]

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.two_d_1 = nn.Sequential(nn.Conv2d(3,64,8,4,0), nn.Flatten())
		self.two_d_2 = nn.Sequential(nn.Conv2d(3,64,8,8,0), nn.Flatten())
		self.two_d_3 = nn.Sequential(nn.Conv2d(3,64,4,4,0), nn.Flatten())
		self.flat = nn.Sequential(nn.Flatten(),)
		self.activate = nn.LeakyReLU(negative_slope=0.01, inplace=True)
		#self.activate = nn.ReLU(True)
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.BatchNorm1d(3072,affine=True),

			nn.Linear(3072,2048),
			self.activate,
			nn.BatchNorm1d(2048,affine=True),

			nn.Linear(2048,1024),
			self.activate,
			nn.BatchNorm1d(1024,affine=True),

			nn.Linear(1024,512),
			self.activate,
			nn.BatchNorm1d(512,affine=True),			

			nn.Linear(512,256),
			
		)

		self.decoder = nn.Sequential(
			nn.Linear(256,512),
			#self.activate,
			#nn.BatchNorm1d(512,affine=True),
			nn.Linear(512,1024),
			#self.activate,
			#nn.BatchNorm1d(1024,affine=True),
			nn.Linear(1024,2048),
			#self.activate,
			#nn.BatchNorm1d(2048,affine=True),
			nn.Linear(2048,3072),
			
			#nn.Tanh()
		)
	def forward(self, x):
		s = x.size()
		twod1 = self.two_d_1(x)
		twod2 = self.two_d_2(x)
		twod3 = self.two_d_3(x)
		x = self.flat(x)
		#print(x.size(), twod1.size(), twod2.size(), twod3.size())
		#x = torch.cat((x, twod1), dim=1)
		#print(x.size())
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		decoded = decoded.view(s)

		return encoded, decoded


trainX = np.load(path_trainX)


print(np.shape(trainX))
trainX = np.transpose(trainX, (0,3,1,2)) / 255. * 2 - 1
print(np.shape(trainX))
trainX = torch.Tensor(trainX)
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load('autoencoder.pth'))
autoencoder.eval()
use_gpu = torch.cuda.is_available()
print(use_gpu)
if use_gpu:
	autoencoder.cuda()
	trainX = trainX.cuda()
	train_dataloader = DataLoader(trainX, batch_size=96, shuffle=True)
	test_dataloader = DataLoader(trainX, batch_size=96, shuffle=False)

latents = []
with torch.no_grad():
	for x in test_dataloader:
		latent, reconstruct = autoencoder(x)
		latent = latent.cpu().detach().numpy()
		latent = latent.reshape([len(x), -1])
		latents.append(latent)
print(np.shape(latents))
latents = np.concatenate(latents, axis=0)
print(np.shape(latents))
latents = latents.reshape([9000, -1])
print(np.shape(latents))
latents_mean = np.mean(latents, axis=0)
latents_std = np.std(latents, axis=0)
latents = (latents - latents_mean) / latents_std
#print("std", latents_std)
print("latents number:", len(latents[0]))

'''
from sklearn.decomposition import PCA
print(1,np.shape(latents[0]))
latents = PCA(n_components=32).fit_transform(latents)
#latents = TSNE(n_components=2).fit_transform(latents)
print(2,np.shape(latents[0]))
'''

from sklearn.manifold import TSNE
latents = TSNE(n_components=2, learning_rate = 5).fit_transform(latents)
print("TSNE done")

'''
from sklearn import random_projection
latents = random_projection.GaussianRandomProjection(n_components=8).fit_transform(latents)
print(np.shape(latents[0]))
print("random_projection done")
'''





h = pd.read_csv("human_label.csv")
h = h['label'].values.tolist()

from sklearn.cluster import KMeans
result = KMeans(n_clusters = 2).fit(latents).labels_
result = result.tolist()

correct, total = 0, 0
for idx, v in enumerate(h):
	total += 1
	if h[idx] == result[idx]:
		correct += 1
if correct/total < 0.5:
	result = list(map(lambda x:(x-1)*-1, result))
	print(1 - correct/total)
else:
	print(correct/total)
ans = pd.DataFrame({"id" : range(0, len(result))})
ans['label'] = result
#print(ans)
ans.to_csv(path_ans, index = False)

'''
from sklearn.cluster import SpectralClustering
result = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0,affinity='nearest_neighbors').fit(latents).labels_
result = result.tolist()
correct, total = 0, 0
for idx, v in enumerate(h):
	total += 1
	if h[idx] == result[idx]:
		correct += 1
if correct/total < 0.5:
	result = list(map(lambda x:(x-1)*-1, result))
	print(1 - correct/total)
else:
	print(correct/total)
ans = pd.DataFrame({"id" : range(0, len(result))})
ans['label'] = result
#print(ans)
ans.to_csv("SpectralClustering"+path_ans, index = False)
'''

