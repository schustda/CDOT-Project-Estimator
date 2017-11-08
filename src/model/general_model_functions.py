import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.externals import joblib


class GeneralModelFunctions(object):

    def __init__(self):
        self.colors = {'r2':'r','mse':'g','mape':'b'}

    def _emptydf(self,name):
        columns = ['r2','mse','mape']
        df = pd.DataFrame(columns=columns,index=range(1,len(self.param_values)+1))
        df['index'] = df.index
        df['param_values'] = self.param_values
        df.index.name = name
        return df

    def preprocessing(self,X):
        scale = joblib.load('data/model/scale.pkl')
        pca = joblib.load('data/model/pca.pkl')
        km = joblib.load('data/model/km.pkl')
        X = scale.transform(X)
        X = pca.transform(X)
        clusters = km.predict(X)
        return X, clusters

    def mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def score_model(self,y_true,y_pred=None,need_pred=False,
            model=None,xg_boost=False):
        '''
        Returns r2 score, mse, and mape scores
        '''
        if need_pred:
            if xg_boost:
                X = xgb.DMatrix(X)
            y_pred = model.predict(X)
        return [r2_score(y_true,y_pred),
                mean_squared_error(y_true,y_pred),
                self.mape(y_true,y_pred)]
