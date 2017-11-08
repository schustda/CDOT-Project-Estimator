import numpy as np
import pandas as pd
import xgboost as xgb
from math import log,exp,sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from src.model.general_model_functions import GeneralModelFunctions

class CDOTModel(GeneralModelFunctions):

    def __init__(self,scale = Normalizer(),date_scale = 100,
            n_clusters = 5):
        self.std = scale
        self.date_scale = date_scale
        self.n_clusters = n_clusters

    def _add_prediction(self,model,X):
        pred = model.predict(X)
        return np.hstack((X,pred.reshape(pred.shape[0],1)))

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.

        Creates pickled model files for Ridge, ExtraTreesRegressor,
        and GradientBoostingRegressor
        '''

        X, clusters = self.preprocessing(X)

        #ExtraTreesRegressor
        print ('Fitting ExtraTreesRegressor')
        etr = ExtraTreesRegressor(n_estimators = 25)
        etr.fit(X,y)
        joblib.dump(etr, 'data/model/etr.pkl')
        X = self._add_prediction(etr,X)

        #GradientBoostingRegressor
        print ('Fitting GradientBoostingRegressor model')
        gbr = GradientBoostingRegressor(loss = 'lad')
        gbr.fit(X,y)
        joblib.dump(gbr, 'data/model/gbr.pkl')
        X = self._add_prediction(gbr,X)

        #XGBoost
        print ('Fitting XGBoost model')
        parameters = {'max_depth':7,'silent':1,'eta':0.3,'booster':'gbtree'}
        dmat = xgb.DMatrix(X, label=y)
        self.xgb_model = xgb.train(parameters,dmat)
        joblib.dump(self.xgb_model, 'data/model/xgb_model.pkl')

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y_predict : array-like, shape (n_samples,)
            Target values.
        '''

        etr = joblib.load('data/model/etr.pkl')
        gbr = joblib.load('data/model/gbr.pkl')
        self.xgb_model = joblib.load('data/model/xgb_model.pkl')

        X,clusters = self.preprocessing(X)
        X = self._add_prediction(etr,X)
        X = self._add_prediction(gbr,X)
        return self.xgb_model.predict(xgb.DMatrix(X))


if __name__ == '__main__':
    train = pd.read_csv('data/model/train.csv',index_col='project_number')
    X = train.drop(['bid_days','bid_total','engineers_estimate','start_date'],axis=1)
    y = train.bid_total
    estimate = train.engineers_estimate
    X_train,X_test,y_train,y_test,estimate_train,estimate_test = train_test_split(X,y,estimate,random_state=10)

    model = CDOTModel()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print (model.score_model(y_test,y_pred))
