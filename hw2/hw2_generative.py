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


def train(X_train, Y_train):
    type_0 = X_train[np.where(Y_train == 0)[0], :]
    type_1 = X_train[np.where(Y_train == 1)[0], :]

    type_0_mean = type_0.mean(axis = 0)
    type_1_mean = type_1.mean(axis = 0)

    prior_prob_type_0 = len(type_0) / (len(type_0) + len(type_1))
    prior_prob_type_1 = 1 - prior_prob_type_0

    covariance = (np.cov(type_0, rowvar=False)*prior_prob_type_0) + (np.cov(type_1, rowvar=False)*prior_prob_type_1)
    covariance_inv = np.linalg.pinv(covariance)

    w = np.dot((type_0_mean - type_1_mean).T, covariance_inv)
    b = - 0.5 * np.dot(np.dot(type_0_mean.T, covariance_inv), type_0_mean) \
        + 0.5 * np.dot(np.dot(type_1_mean.T, covariance_inv), type_1_mean) \
        + np.log(len(type_0) / len(type_1))
    return w, b
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return res
    return np.clip(res, 1e-8, 1-1e-8)

def predict(X, w, b):
    pred = sigmoid(np.dot(X, w) + b)
    pred = 1 - np.around(pred)
    pred = pred.reshape(-1,1).astype("int")
    return pred

w, b = train(X_train, Y_train)
np.save("weight_Generative", w)
np.save("bias_Generative", b)

#w = np.load("weight_Generative.npy")
#bias = np.load("bias_Generative.npy")
test = pd.read_csv(path_X_test)

test = preprocess(test)

y_hat = predict(test, w, b)
print(y_hat, len(y_hat))


c_id = pd.DataFrame({"id" : range(1, len(y_hat)+1)})
c_id['label'] = y_hat
c_id.to_csv(path_ans, index = False)