import numpy as np
import pandas as pd
from train import *


train = pd.read_csv('../../data/train.csv')
test = pd.read_csv('../../data/test.csv')
all_data = train.append(test)
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
all_data.fillna(-1, inplace=True)
all_data['Response'] = all_data['Response'].astype(int)
all_data['Split'] = np.random.randint(5, size=all_data.shape[0])
train = all_data[all_data['Response']>0].copy()
test = all_data[all_data['Response']<1].copy()

# param = {'colsample_bytree': 0.4, 'silent': 1, 'nthread': 10, 'min_child_weight': 80, 'subsample': 0.9, 'eta': 0.1, 'objective': 'reg:linear', 'max_depth': 9}
# param = {'colsample_bytree': 0.4, 'silent': 1, 'nthread': 10, 'min_child_weight': 80, 'subsample': 0.9, 'eta': 0.02, 'objective': 'count:poisson', 'max_depth': 9}
param = {'colsample_bytree': 0.4, 'silent': 1, 'nthread': 10, 'min_child_weight': 80, 'subsample': 0.9, 'eta': 0.015, 'objective': 'count:poisson', 'max_depth': 9}
train_model = TrainModel(train, test, param, 1500)
train_model._submit(50)
