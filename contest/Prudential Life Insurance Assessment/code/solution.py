import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import sys
import os
from code.kappa_score import *

train = pd.read_csv('../data/train3.csv')
test = pd.read_csv('../data/test3.csv')
sample_submission = pd.read_csv('../data/sample_submission.csv')

features = train.columns.tolist()
features.remove("Id")
features.remove("Response")
train_features = train[features]
test_features = test[features]
y_train = train["Response"].values


