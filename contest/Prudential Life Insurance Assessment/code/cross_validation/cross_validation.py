import pandas as pd
import numpy as np
import xgboost as xgb
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import KFold
from train import *

class CrossValidation(object):

    def __init__(self, train_set, param, num_rounds, nfold=5):
        self.train_set = train_set
        self.param = param
        self.num_rounds = num_rounds
        self.nfold = nfold


    def cv(self):
        train_set = self.train_set
        kf = KFold(train_set.shape[0], n_folds=self.nfold)
        scores = .0
        for train_loc, test_loc in kf:
            train = train_set.iloc[train_loc]
            test = train_set.iloc[test_loc]
            train_model = TrainModel(train, test, self.param, self.num_rounds)
            preds = train_model._predict()
            scores += train_model._eval_wrapper(preds, test['Response'])
        return scores / self.nfold
