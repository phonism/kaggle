import numpy as np
import pandas as pd
from cross_validation import *


# feature 1
train = pd.read_csv('../../data/train.csv')
train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
train.fillna(-1, inplace=True)
train['Response'] = train['Response'].astype(int)
train['Split'] = np.random.randint(5, size=train.shape[0])

'''
X = []
Y = []
for i in range(len(train)):
    ans = 0
    for j in train.iloc[i]:
        if j == 0:
            ans += 1
    X.append(ans - 32)
    Y.append(train.iloc[i]['Response'])

train['X'] = X
'''



def print_cv(param, num_rounds, nfold=5):
    global train
    print train.columns.values
    print len(train.columns.values)
    cv = CrossValidation(train, param, num_rounds, nfold)
#    print param
    print cv.cv()

param = {'n_estimators': 500, 'max_features': 'sqrt', 'max_depth': None, 'verbose': 1, 'n_jobs': -1}
print_cv(param, 1500, 3)

'''
# feature 2
train = pd.read_csv('../../data/train.csv')
train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
train.fillna(-1, inplace=True)
train['Response'] = train['Response'].astype(int)
# train['Split'] = np.random.randint(5, size=train.shape[0])

param = {'colsample_bytree': 0.4, 'silent': 1, 'nthread': 5, 'min_child_weight': 80, 'subsample': 0.9, 'eta': 0.02, 'objective': 'count:poisson', 'max_depth': 9}
print_cv(param, 1500, 5)
'''
