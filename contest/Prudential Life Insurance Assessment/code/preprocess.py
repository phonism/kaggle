import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer

# LabelEncoder
train = pd.read_csv('../data/train.csv', index_col=None)
test = pd.read_csv('../data/test.csv', index_col=None)
sample_submission = pd.read_csv('../data/sample_submission.csv')

train_cols = train.columns
test_cols = test.columns
labels = train['Response'].ravel()
train_ids = train['Id'].ravel()
test_ids = test['Id'].ravel()

train.drop('Id', axis=1, inplace=True)
train.drop('Response', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train = np.array(train)
test = np.array(test)

for i in range(1, 2):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:, i]) + list(test[:, i]))
    train[:, i] = lbl.transform(train[:, i])
    test[:, i] = lbl.transform(test[:, i])

train = np.column_stack((train_ids, train, labels))
test = np.column_stack((test_ids, test))
train = pd.DataFrame(train, columns=train_cols)
test = pd.DataFrame(test, columns=test_cols)

train['Id'] = train['Id'].astype(int)
train['Response'] = train['Response'].astype(int)
test['Id'] = test['Id'].astype(int)

train.to_csv('../data/train2.csv', index=None)
test.to_csv('../data/test2.csv', index=None)

# DictVectorizer
train = pd.read_csv('../data/train.csv', index_col=None)
test = pd.read_csv('../data/test.csv', index_col=None)

train_cols = train.columns
test_cols = test.columns
labels = train['Response'].ravel().astype(int)
train_ids = train['Id'].ravel().astype(int)
test_ids = test['Id'].ravel().astype(int)

train.drop('Id', axis=1, inplace=True)
train.drop('Response', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

train = train.T.reset_index(drop=True).to_dict().values()
test = test.T.reset_index(drop=True).to_dict().values()

vec = DictVectorizer(sparse=False)
train = vec.fit_transform(train)
test = vec.transform(test)

train = np.column_stack((train_ids, labels, train))
test = np.column_stack((test_ids, test))
train = pd.DataFrame(train, columns=['Id', 'Response'] + vec.get_feature_names())
test = pd.DataFrame(test, columns=['Id'] + vec.get_feature_names())

train['Id'] = train['Id'].astype(int)
train['Response'] = train['Response'].astype(int)
test['Id'] = test['Id'].astype(int)

train.to_csv('../data/train3.csv', index=None)
test.to_csv('../data/test3.csv', index=None)

# Factors to hazard mean
train = pd.read_csv('../data/train.csv', index_col=None)
test = pd.read_csv('../data/test.csv', index_col=None)

train_cols = train.columns
test_cols = test.columns
labels = train.Response.astype(int)
train_ids = train.Id.astype(int)
test_ids = test.Id.astype(int)

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

for feat in train.select_dtypes(include=['object']).columns:
    m = train.groupby([feat])['Response'].mean()
    train[feat].replace(m, inplace=True)
    test[feat].replace(m, inplace=True)

train = pd.concat((train_ids, train), axis=1)
test = pd.concat((test_ids, test), axis=1)

train.to_csv('../data/train4.csv', index=None)
test.to_csv('../data/test4.csv', index=None)
