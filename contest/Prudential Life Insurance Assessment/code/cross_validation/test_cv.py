import numpy as np
import pandas as pd
from cross_validation import *


def get_params():
    params = {}
    params["objective"] = "reg:linear"     
    params["eta"] = 0.1
    params["min_child_weight"] = 80
    params["subsample"] = 0.8
    params["colsample_bytree"] = 0.30
    params["silent"] = 1
    params["max_depth"] = 9
    params["nthread"] = 10
    return params

param = get_params()
num_rounds = 250

train = pd.read_csv('../../data/train.csv')
train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
train.fillna(-1, inplace=True)
train['Response'] = train['Response'].astype(int)
train['Split'] = np.random.randint(5, size=train.shape[0])



def print_cv(param, num_rounds, nfold=5):
    global train
    cv = CrossValidation(train, param, num_rounds, nfold)
    print param
    print cv.cv()

param = {'colsample_bytree': 0.4, 'silent': 1, 'nthread': 10, 'min_child_weight': 80, 'subsample': 0.9, 'eta': 0.1, 'objective': 'reg:linear', 'max_depth': 9}
# print_cv(param, 500, 5)
