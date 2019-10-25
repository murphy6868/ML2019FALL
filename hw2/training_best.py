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


X_train = pd.read_csv(path_train_csv, header=None)
del X_train[14]
X_train.columns = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country"]

Y_train = pd.read_csv(path_Y_train, header=None)
Y_train = Y_train.values

X_test = pd.read_csv(path_test_csv)

all_data = pd.concat([X_train, X_test], axis = 0, ignore_index = True)

def preprocess(X_train):
    
    X_train = X_train.replace([' Asian-Pac-Islander', ' Amer-Indian-Eskimo'], ' Other')
    X_train["capital_gain"] = np.sqrt(X_train["capital_gain"])
    X_train["capital_loss"] = np.sqrt(X_train["capital_loss"])
    X_train["marital_status"] = X_train["marital_status"].replace([' Never-married',' Divorced',' Separated',' Widowed',' Married-spouse-absent'], 'Single')
    X_train["marital_status"] = X_train["marital_status"].replace([' Married-civ-spouse',' Married-AF-spouse'], 'Married')
    age_gap = 2
    X_train['age'] = X_train['age'] // age_gap
    hours_gap = 3
    X_train['hours_per_week'] = X_train['hours_per_week'] // 2
    X_train['age'] = X_train['age'].astype(str)
    X_train['hours_per_week'] = X_train['hours_per_week'].astype(str)
    interaction_terms = ["age","workclass","education","marital_status","occupation","relationship","race","sex","native_country"]
    cat_terms = ["age","hours_per_week","workclass","education","marital_status","occupation","relationship","race","sex","native_country"]
    del X_train["education_num"]
    del X_train["fnlwgt"]
    for i in interaction_terms:
        for j in interaction_terms:
            if i == j: continue
            new_col_name = i + j
            print(new_col_name)
            X_train[new_col_name] = X_train.apply(lambda row: row[i]+row[j], axis=1)
            cat_terms.append(new_col_name)
    print(X_train, X_train.columns)
    X_train = pd.get_dummies(X_train, columns=cat_terms)

    X_train = X_train.values
    X_train = X_train.astype(float)
    std = X_train.std(axis = 0)
    mean = X_train.mean(axis = 0)
    min_ = X_train.min(axis = 0)
    max_ = X_train.max(axis = 0)
    ok_index = [np.where(std != 0)[0].tolist()]

    X_train[:,ok_index] = (X_train[:,ok_index] - mean[ok_index]) / std[ok_index]
    
    return X_train
all_data = preprocess(all_data)
X_train = all_data[:X_train.shape[0]]
X_test = all_data[X_train.shape[0]:]

good = []
b = Y_train.reshape(1,-1)[0]
for i in range(X_train.shape[1]):
    a = X_train[:,i]
    c = np.corrcoef(a,b)[0][1]
    if abs(c) > 0.05:
        good.append(i)
print("ori features:", X_train.shape[1])
X_train = X_train[:,good]
X_test = X_test[:,good]
print("selected features:", X_train.shape[1])
independent = []
bad = 0
for i in range(X_train.shape[1]):
    ok = 1
    if i % 100 == 0:
        print(i, bad)
    for j in range(i+1, X_train.shape[1]):
        c = np.corrcoef(X_train[:,i],X_train[:,j])[0][1]
        if c > 0.95:
            ok = 0
            bad += 1
            break
    if ok == 1:
        independent.append(i)
X_train = X_train[:,independent]
X_test = X_test[:,independent]
print("selected features:", X_train.shape[1])

from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
clf.fit(X_train, Y_train)
dump(clf, 'GradientBoostingClassifier.joblib') 
c = clf.score(X_train, Y_train)
print(c)