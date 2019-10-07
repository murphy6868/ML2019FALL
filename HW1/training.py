import pandas as pd
import numpy as np
import sklearn as sk
import sys

data = pd.read_csv(sys.argv[1])

rm_term = ["WIND_DIREC", "WD_HR"]
for i in range(len(rm_term)):
    data = data[ data["測項"] != rm_term[i] ]
terms = np.unique(data["測項"])
total_features = len(terms)
pm25_pos = list(terms).index("PM2.5")

allfeatures = data.iloc[:,2:].values.tolist()

allfeatures_18_ = [[] for i in range(total_features)]
for i in range(total_features):
    for j in range(data.shape[0]//total_features):
        allfeatures_18_[i] += allfeatures[j*total_features+i]
feature_list = []
        
for i,vi in enumerate(allfeatures_18_):
    for j,vj in enumerate(vi):
        if pd.isna(vj):
            vj = "-1"
        if vj[-1] == 'x' or vj[-1] == '#' or vj[-1] == '*':
            vj = vj[:-1]
        if vj == "NR":
            vj = 0
        vj = float(vj)
        if i == pm25_pos:
            if vj > 100 or vj < 1:
                vj = -1
            if j>0 and j<len(vi)-1:
                if abs(vj-allfeatures_18_[0][j-1]) > 25:
                    vj = -1
        else:
            if vj < 0:
                vj = -1
        allfeatures_18_[i][j] = vj

allfeatures_18_ = np.array(allfeatures_18_)

means = []
for i in allfeatures_18_:
    means.append(np.mean(i))

hours = 4
features = []
labels = []
for i in range(np.shape(allfeatures_18_)[1] - hours):
    valid = 1
    tmp = allfeatures_18_[:, i:i+hours]
    if -1 in tmp:
        valid = 0
    tmp2 = allfeatures_18_[pm25_pos][i+hours]
    if tmp2 == -1:
        valid = 0
    if valid == 1 :
        #tmp = np.append(tmp,1)
        features.append(tmp.reshape(-1))
        labels.append(tmp2)
        
for f in feature_list:
    for i in range(np.shape(f)[1] - hours):
        valid = 1
        tmp = f[:, i:i+hours]
        if -1 in tmp:
            valid = 0
        tmp2 = f[pm25_pos][i+hours]
        if tmp2 == -1:
            valid = 0
        if valid == 1 :
            #tmp = np.append(tmp,1)
            features.append(tmp.reshape(-1))
            labels.append(tmp2)

features = np.array(features)
labels = np.array(labels).reshape(-1, 1)

def train(features_train, labels_train):
    iteration = 1000
    lr = 1e-3
    lam = 0.001
    beta_1 = np.full(features_train[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(features_train[0].shape, 0.99).reshape(-1, 1)
    w = np.full(features_train[0].shape, 0.0).reshape(-1, 1)

    m_t = np.full(features_train[0].shape, 0).reshape(-1, 1)

    v_t = np.full(features_train[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    batch_size = 64
    for i in range(iteration):
        index = np.arange(features_train.shape[0])
        np.random.shuffle(index)
        features_train = features_train[index]
        labels_train = labels_train[index]
        for b in range(features_train.shape[0]//batch_size):
            f_batch = features_train[b*batch_size:(b+1)*batch_size]
            l_batch = labels_train[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            t += 1
            loss = l_batch - np.dot(f_batch, w)
            g_t = np.dot(f_batch.T, loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 

            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
    return w
w = train(features, labels)
def print_weight(w):
    idx = 0
    for i in range(total_features):
        printout = str(terms[i]) + " "*(11-len(terms[i]))
        for j in range(hours):
            printout += str(w[idx]) + " "*(15-len(str(w[idx])))
            idx += 1
        print(printout)
print_weight(w)
np.save("weight_", w)
