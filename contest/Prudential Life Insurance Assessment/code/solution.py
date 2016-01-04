import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from kappa_score import *
import csv

def output_function(x):
    if x < 1:
        return 1
    elif x > 8:
        return 8
    else:
        x = int(round(x))
        if x == 3:
            return 2
        return x

def cross_validation(train_file_path, test_file_path, param, num_round=1000):
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    features = train.columns.tolist()
    features.remove('Id')
    features.remove('Response')
    train_features = train[features]
    test_reatures = test[features]
    Y_train = train['Response'].values
    
    X_train, X_test, y_train, y_test = train_test_split(train_features, Y_train, test_size=0.33, random_state=random.randint(1, 100))

    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.NaN)
    dtest = xgb.DMatrix(X_test, missing=np.NaN)

    watchlist = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    y_test_bst = bst.predict(dtest)
    y_train_test = [output_function(y) for y in y_test_bst]

    # hack
#    for i, j in zip(y_train_test, y_test):
#        print i, j
    #
    
    return quadratic_weighted_kappa(y_test, y_train_test)


def submit(train_file_path, test_file_path, param):
    train = pd.read_csv(train_file_path)
    test = pd.read_csv(test_file_path)
    sample_submission = pd.read_csv('../data/sample_submission.csv')

    features = train.columns.tolist()
    features.remove('Id')
    features.remove('Response')
    train_features = train[features]
    test_features = test[features]
    y_train = train['Response'].values
    
    # X_train, X_test, y_train, y_test = train_test_split(train_features, y_train, test_size=0.33, random_state=42)

    num_round = 1000

    dtrain = xgb.DMatrix(train_features, label=y_train, missing=np.NaN)
    dtest = xgb.DMatrix(test_features, missing=np.NaN)

    watchlist = [(dtrain,'train')]
    bst = xgb.train(param, dtrain, num_round, watchlist)
    y_test_bst = bst.predict(dtest)
    y_train_test = [output_function(y) for y in y_test_bst]

    ids = test.Id.values.tolist()
    n_ids = len(ids)

    prediction_file = open("xgbresult1.csv", "w")
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["Id","Response"])
    for i in range(0, n_ids):
        prediction_file_object.writerow([ids[i], y_train_test[i]])

    # return quadratic_weighted_kappa(y_test, y_train_test)


def cv():
    print '=======================dataset3====================='
    param1 = {'max_depth': 9, 'eta': 0.02, 'min_child_weight': 2, 'subsample': 0.7, 
            'objective': 'reg:linear', 'eval_metric': 'rmse', 'colsample_bytree': 0.65, 'nthread': 10}
    cv1_1 =  cross_validation('../data/train2.csv', '../data/test2.csv', param1, num_round=1000)
    cv1_2 =  cross_validation('../data/train2.csv', '../data/test2.csv', param1, num_round=1000)
    cv1_3 =  cross_validation('../data/train2.csv', '../data/test2.csv', param1, num_round=1000)
    print '==================================================='
    
    print '=======================dataset3====================='
    param2 = {'max_depth': 9, 'eta': 0.02, 'min_child_weight': 2, 'subsample': 0.7, 
            'objective': 'reg:linear', 'eval_metric': 'rmse', 'colsample_bytree': 0.65, 'nthread': 10}
    cv2_1 =  cross_validation('../data/train3.csv', '../data/test3.csv', param2, num_round=1000)
    cv2_2 =  cross_validation('../data/train3.csv', '../data/test3.csv', param2, num_round=1000)
    cv2_3 =  cross_validation('../data/train3.csv', '../data/test3.csv', param2, num_round=1000)
    print '==================================================='
    
    print '=======================dataset3====================='
    param3 = {'max_depth': 9, 'eta': 0.02, 'min_child_weight': 2, 'subsample': 0.7, 
            'objective': 'reg:linear', 'eval_metric': 'rmse', 'colsample_bytree': 0.65, 'nthread': 10}
    cv3_1 =  cross_validation('../data/train4.csv', '../data/test4.csv', param3, num_round=1000)
    cv3_2 =  cross_validation('../data/train4.csv', '../data/test4.csv', param3, num_round=1000)
    cv3_3 =  cross_validation('../data/train4.csv', '../data/test4.csv', param3, num_round=1000)
    print '==================================================='
    
    print '=======================dataset3====================='
    param4 = {'max_depth': 9, 'eta': 0.02, 'min_child_weight': 2, 'subsample': 0.7, 
            'objective': 'reg:linear', 'eval_metric': 'rmse', 'colsample_bytree': 0.65, 'nthread': 10}
    cv4_1 =  cross_validation('../data/train2.csv', '../data/test2.csv', param4, num_round=1000)
    cv4_2 =  cross_validation('../data/train2.csv', '../data/test2.csv', param4, num_round=1000)
    cv4_3 =  cross_validation('../data/train2.csv', '../data/test2.csv', param4, num_round=1000)
    
    print '==================================================='
    print '==================================================='
    print param1
    print cv1_1, cv1_2, cv1_3
    print '==================================================='
    print '==================================================='
    print param2
    print cv2_1, cv2_2, cv2_3
    print '==================================================='
    print '==================================================='
    print param3
    print cv3_1, cv3_2, cv3_3
    print '==================================================='
    print '==================================================='
    print param4
    print cv4_1, cv4_2, cv4_3
    print '==================================================='
                                                            

def sb():
    # param = {'subsample': 0.8, 'eta': 0.02, 'colsample_bytree': 0.65, 'eval_metric': 'rmse', 'objective': 'count:poisson', 'max_depth': 11, 'min_child_weight': 3, 'nthread': 10}
    param = {'subsample': 0.8, 'eta': 0.02, 'colsample_bytree': 0.65, 'eval_metric': 'rmse', 'objective': 'reg:linear', 'max_depth': 12, 'min_child_weight': 2, 'nthread': 10}
    submit('../data/train2.csv', '../data/test2.csv', param)

def test():
    param = {'subsample': 0.8, 'eta': 0.02, 'colsample_bytree': 0.65, 'eval_metric': 'rmse', 
            'objective': 'count:poisson', 'max_depth': 11, 'min_child_weight': 3, 'nthread': 10}
    cv = cross_validation('../data/train2.csv', '../data/test2.csv', param, num_round=1000)
    print param
    print cv

def main():
    # sb()
    cv()
    # test()

if __name__ == '__main__':
    main()

