import pandas as pd
import numpy==1.16.5 as np
import sklearn as sk
import sys
print(pd.__version__)
print(np.__version__)
print(sk.__version__)

w = np.load("weight.npy")
rm_term = ["WIND_DIREC", "WD_HR"]
test = pd.read_csv(sys.argv[1])
for i in range(len(rm_term)):
    test = test[ test["測項"] != rm_term[i] ]
terms = np.unique(test["測項"])
total_features = len(terms)
means = [23.913953488372094, 1.9066974464204285, 0.4692025763793889, 0.1467111263109895, 5.8170542635658915, 18.43959758321933, 24.25543775649795, 27.712226402188783, 32.383834929320564, 14.718137254901961, 0.29268125854993166, 75.66052211582307, 2.5442430460556316, 2.0589261285909712, 2.0922480620155035, 1.7698016415868674]
hours = 4

for i in range(test.shape[0]//total_features):
    for j in range(total_features):
        for h in range(9):
            if test.iloc[i*total_features+j, 2+h] == '' or pd.isna(test.iloc[i*total_features+j, 2+h]):
                test.iloc[i*total_features+j, 2+h] = means[j]

test_data = test.iloc[:,2:].values
test_data[ test_data == 'NR'] = 0

for i in np.nditer(test_data, flags=["refs_ok"], op_flags=['readwrite']):
    i[...] = float(str(i).rstrip('x*#A'))

features_test = []
for i in range(np.shape(test_data)[0]//total_features):
    a = test_data[i*total_features:i*total_features+total_features,9-hours:9]
    a = a.reshape(-1)
    #a = np.append(a,1)
    features_test.append(a)

features_test = np.array(features_test)
res = np.dot(features_test, w).reshape(-1)
res = [x if x>3 else 3 for x in res]
out = pd.DataFrame(columns = ["id", "value"])

for i in range(len(features_test)):
    out.loc[i] = ["id_" + str(i), res[i]]
out.to_csv(sys.argv[2], index = False)