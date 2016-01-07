import pandas as pd 
import numpy as np 
import xgboost as xgb
import scipy as sp
from datetime import datetime
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import sklearn.preprocessing as pp
from ml_metrics import quadratic_weighted_kappa

def count_feature(X, tbl_lst = None, min_cnt = 1):
    X_lst = [pd.Series(X[:, i]) for i in range(X.shape[1])]
    if tbl_lst is None:
        tbl_lst = [x.value_counts() for x in X_lst]
        if min_cnt > 1:
            tbl_lst = [s[s >= min_cnt] for s in tbl_lst]
    X = sp.column_stack([x.map(tbl).values for x, tbl in zip(X_lst, tbl_lst)])
    # NA(unseen values) to 0
    return np.nan_to_num(X), tbl_lst

def _eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat) 
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)


path = '../../data/'
train_file = path + 'train2.csv'
test_file = path + 'test2.csv'

train_test = pd.read_csv(train_file, index_col=0)
test = pd.read_csv(test_file, index_col=0)
train = train_test.iloc[range(0, 40000)]
test = train_test.iloc[range(40000, 50000)]

tttt = test['Response']


target = train['Response']
train.drop('Response', inplace=True, axis=1)
test.drop('Response', inplace=True, axis=1)


num_train = train.shape[0]
num_feature = train.shape[1]
print num_feature

# df_trans = pd.Series(range(9), index = target.unique())

# y = target.map(df_trans).values
y = target - 1
# y = target

yMat = pd.get_dummies(y).values

X = np.vstack((train.values, test.values))

nIter = 1
tc = 15 # max_depth
sh = .1 # eta
bf = .8 # subsample

### XGB1: Count feature
nt = 800
mb = 5 # min_child_weight
cs = 45. / 126 # colsample_bytree

X2, ignore = count_feature(X)
dtrain , dtest = xgb.DMatrix(X2[:num_train], label = y), xgb.DMatrix(X2[num_train:])

predAll_train = np.zeros((num_train, 8))
predAll_test = np.zeros((test.shape[0], 8))
scores = []

t0 = datetime.now()
for i in range(nIter):
    seed = i + 123
    param = {'bst:max_depth':9, 'bst:eta':0.02, 'silent':1, 'objective':'multi:softprob','num_class':8,
             'min_child_weight':80, 'subsample':0.9, 'colsample_bytree':cs, 'nthread':12, 'seed':seed}
    plst = param.items()
    bst = xgb.train(plst, dtrain, nt)
    # bst.save_model(path + 'model/model_XGB_CF_' + str(seed) + '.model')
    pred_train = bst.predict(dtrain).reshape((num_train, 8))
    pred_test = bst.predict(dtest).reshape(predAll_test.shape)
    predAll_train += pred_train
    predAll_test += pred_test
    print i, "Time:%s" % (datetime.now() - t0)

pred_XGB = predAll_test / nIter

pred = pp.normalize(pred_XGB, norm = 'l1')
preds_a = []
for i in pred:
    preds_a.append(i.argmax() + 1)

pred1 = np.array(preds_a)
print len(tttt)
print preds_a
print _eval_wrapper(tttt, pred1)
# pred005 = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
# pred005.to_csv(path + 'pred005.csv', float_format='%.8f')
