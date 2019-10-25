import pandas as pd
import numpy as np
import sklearn as sk
import sys, random, math
print(pd.__version__=="0.25.1")
print(np.__version__=="1.16.5")
print(sk.__version__=="0.21.3")
path_train_csv = sys.argv[1]
path_test_csv = sys.argv[2]
path_X_train = sys.argv[3]
path_Y_train = sys.argv[4]
path_X_test = sys.argv[5]
path_ans = sys.argv[6]
X_train = pd.read_csv(path_X_train)
Y_train = pd.read_csv(path_Y_train, header=None)
Y_train = Y_train.values
def preprocess(X_train):
    
    age_gap = 2
    X_train['age'] = X_train['age'] // age_gap
    for i in range(102//age_gap):
        X_train["age_" + str(i)] = X_train.apply(lambda row: int(row['age'] == i), axis=1)
    #print(X_train)
        
    hours_gap = 2
    X_train['hours_per_week'] = X_train['hours_per_week'] // 2
    for i in range(102//hours_gap):
        X_train["hours_" + str(i)] = X_train.apply(lambda row: int(row['hours_per_week'] == i), axis=1)
    #print(X_train)
    
    
    del X_train['fnlwgt']
    del X_train['age']
    del X_train['hours_per_week']

    X_train = X_train.values
    X_train = X_train.astype(float)
    std = X_train.std(axis = 0)
    ok_index = [np.where(std != 0)[0].tolist()]
    mean = X_train.mean(axis = 0)
    X_train[:,ok_index] = (X_train[:,ok_index] - mean[ok_index]) / std[ok_index]

    return X_train
X_train = preprocess(X_train)

w = np.full(X_train[0].shape, 0.0).reshape(-1, 1)
bias = 0


batch_size = 512
def data_iter(features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j, axis=0), labels.take(j, axis=0)
def logistic(X):
    ret = 1 / (1 + np.exp(X * -1))
    return ret
def cross_entropy(y_hat, y):
    delta = 1e-8
    return - (y * np.log(y_hat + delta) + (1 - y) * np.log(1 - y_hat + delta)).mean()
def evaluate_accuracy():
    global w, bias
    acc_sum, n = 0.0, 0
    for X, y in data_iter(X_train, Y_train):
        y_hat = logistic(np.dot(X, w) + bias)
        acc_sum += ((y_hat >= 0.5) == y).sum()
        n += y.shape[0]
    return acc_sum / (n + 1e-8)
def train():
    iteration = 60
    gap = 1
    global w, bias
    lr = 3
    decay = 0.8
    acc = 0
    for i in range(iteration):
        if i % gap == 0:
            lr *= decay
            print(lr)
        train_l_sum = 0.0
        for X, y in data_iter(X_train, Y_train):
            y_hat = logistic(np.dot(X, w) + bias)
            loss = cross_entropy(y_hat, y)
            train_l_sum += loss.sum(axis=0)
            
            diff = y_hat - y
            
            threshold = 1-acc
            diff[abs(diff) < 0.5-threshold] = 0
            diff[abs(diff) > 0.5+threshold] = 0
            
            w_grad = np.dot(X.T, diff)
            b_grad = (diff).sum()
            
            w -= lr * w_grad / batch_size
            bias -= lr * b_grad / batch_size
        acc = evaluate_accuracy()
        print(i,"/",iteration, train_l_sum, acc, bias)
        np.save("weight_Logistic", w)
        np.save("bias_Logistic", bias)
    return 
train()





