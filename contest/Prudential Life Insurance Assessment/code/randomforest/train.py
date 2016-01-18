import pandas as pd 
import numpy as np 
import xgboost as xgb
from scipy.optimize import fmin_powell
from sklearn.ensemble import RandomForestRegressor
from ml_metrics import quadratic_weighted_kappa

class TrainModel(object):

    def __init__(self, train, test, param, num_rounds):
        self.train = train
        self.test = test
        self.param = param
        self.num_rounds = num_rounds


    def _eval_wrapper(self, yhat, y):  
        y = np.array(y)
        y = y.astype(int)
        yhat = np.array(yhat) 
        yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
        return quadratic_weighted_kappa(yhat, y)
    
    def _apply_offset(self, data, bin_offset, sv):
        # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
        data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
        score = self._eval_wrapper(data[1], data[2])
        return score

    def _train(self):
        train = self.train
        test = self.test
        features = list(train.drop(['Id', 'Response'], axis=1))
        target = 'Response'
        ne = 1000
        mf = 'sqrt'
        md = 15
        vb = 1
        nj = -1
        mss = 10
        msl = 5
        clf = RandomForestRegressor(n_estimators = ne, 
                                    max_features = mf, 
                                    max_depth = md, 
                                    verbose = vb, 
                                    n_jobs = nj, 
                                    min_samples_leaf = msl,
                                    min_samples_split = mss)
        print 'n_estimators: ' + str(ne) \
            + ', max_features: ' + str(mf) \
            + ', max_depth: ' + str(md) \
            + ', verbose: ' + str(vb) \
            + ', n_jobs: ' + str(nj) \
            + ', min_samples_split: ' + str(mss) \
            + ', min_samples_leaf: ' + str(msl)
        clf.fit(train[features], train[target])
        print clf.feature_importances_
        train_preds = clf.predict(train[features])
        test_preds = clf.predict(test[features])
        print('Train score is:', self._eval_wrapper(train_preds, train['Response'])) 
        train_preds = np.clip(train_preds, -0.99, 8.99)
        test_preds = np.clip(test_preds, -0.99, 8.99)
        num_classes = 8
        offsets = np.ones(num_classes) * -0.5
        offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
        for j in range(num_classes):
            train_offset = lambda x: -self._apply_offset(offset_train_preds, x, j)
            offsets[j] = fmin_powell(train_offset, offsets[j])  
        
        data = np.vstack((test_preds, test_preds, test['Response'].values))
        for j in range(num_classes):
            data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
        
        final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
        return final_test_preds

    def _predict(self):
        train = self.train
        test = self.test
        param = self.param
        num_rounds = self.num_rounds
        num_classes = 8
        all_test_preds = ''
        dtrain = xgb.DMatrix(train.drop(['Id', 'Response'], axis=1), train['Response'].values)
        dtest = xgb.DMatrix(test.drop(['Id', 'Response'], axis=1), label=self.test['Response'].values)
        model = xgb.train(param, dtrain, num_rounds)
        train_preds = model.predict(dtrain, ntree_limit=model.best_iteration)
        print('Train score is:', self._eval_wrapper(train_preds, train['Response'])) 
        test_preds = model.predict(dtest, ntree_limit=model.best_iteration)
        train_preds = np.clip(train_preds, -0.99, 8.99)
        test_preds = np.clip(test_preds, -0.99, 8.99)
        offsets = np.ones(num_classes) * -0.5
        offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
        for j in range(num_classes):
            train_offset = lambda x: -self._apply_offset(offset_train_preds, x, j)
            offsets[j] = fmin_powell(train_offset, offsets[j])  
        
        data = np.vstack((test_preds, test_preds, test['Response'].values))
        for j in range(num_classes):
            data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
        
        final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
        return final_test_preds

    def _submit(self, nt=50):
        train = self.train
        test = self.test
        param = self.param
        num_rounds = self.num_rounds
        num_classes = 8
        all_test_preds = ''
        for i in range(nt):
            seed = np.random.randint(1, 100000) + 778877
            dtrain = xgb.DMatrix(train.drop(['Id', 'Response'], axis=1), train['Response'].values)
            dtest = xgb.DMatrix(test.drop(['Id', 'Response'], axis=1), label=self.test['Response'].values)
            param['seed'] = seed
            model = xgb.train(param, dtrain, num_rounds)
            train_preds = model.predict(dtrain, ntree_limit=model.best_iteration)
            # print('Train score is:', self._eval_wrapper(train_preds, train['Response'])) 
            test_preds = model.predict(dtest, ntree_limit=model.best_iteration)
            train_preds = np.clip(train_preds, -0.99, 8.99)
            test_preds = np.clip(test_preds, -0.99, 8.99)
            offsets = np.ones(num_classes) * -0.5
            offset_train_preds = np.vstack((train_preds, train_preds, train['Response'].values))
            for j in range(num_classes):
                train_offset = lambda x: -self._apply_offset(offset_train_preds, x, j)
                offsets[j] = fmin_powell(train_offset, offsets[j])
            
            data = np.vstack((test_preds, test_preds, test['Response'].values))
            for j in range(num_classes):
                data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
            
            if all_test_preds == '':
                all_test_preds = data[1]
            else:
                all_test_preds += data[1]
            print 'Train ' + str(i + 1) + ' has done!'

        final_test_preds = np.round(np.clip(all_test_preds / nt, 1, 8)).astype(int)
            
        preds_out = pd.DataFrame({"Id": test['Id'].values, "Response": final_test_preds})
        preds_out = preds_out.set_index('Id')
        preds_out.to_csv('xgb_offset_submission.csv')
