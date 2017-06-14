import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log,exp,sqrt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score

class CDOTModel(object):

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
        # Transformation
        X = X.values
        y = y.map(lambda x: x**1.4).values

        #Ridge
        print ('Fitting Ridge model...')
        rid = Ridge(fit_intercept = False, solver = 'lsqr')
        rid.fit(X,y)
        joblib.dump(rid, 'data/model/rid.pkl')
        X = np.insert(X,0,rid.predict(X),axis=1)
        print ('Ridge complete.\n')

        #ExtraTreesRegressor
        print ('Fitting ExtraTreesRegressor model...')
        etr = ExtraTreesRegressor(bootstrap = False, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 1000, n_jobs = -1)
        etr.fit(X,y)
        joblib.dump(etr, 'data/model/etr.pkl')
        X = np.insert(X,0,etr.predict(X),axis=1)
        print ('ExtraTreesRegressor complete.\n')

        #GradientBoostingRegressor
        print ('Fitting GradientBoostingRegressor model...')
        gbr = GradientBoostingRegressor(n_estimators = 1000)
        gbr.fit(X,y)
        joblib.dump(gbr, 'data/model/gbr.pkl')

        print ('GradientBoostingRegressor Complete')
        print ('Model successfully fitted \n')

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

        #Load models
        rid = joblib.load('data/model/rid.pkl')
        etr = joblib.load('data/model/etr.pkl')
        gbr = joblib.load('data/model/gbr.pkl')

        #Add etr precitions to feature matrix
        X = X.values
        X = np.insert(X,0,rid.predict(X),axis=1)
        X = np.insert(X,0,etr.predict(X),axis=1)
        y = gbr.predict(X)

        # Reverse transform the target
        vec = np.vectorize(lambda x: x**(1/1.4))
        return vec(y)

    def score(self, X, y):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        y : array-like, shape (n_samples,)
        Target values.

        Returns
        -------
        score : int, r2 score for the current model
        '''

        return r2_score(y, self.predict(X))

def add_unit_prices():
    '''
    Returns
    -------
    unit_prices : array-like, shape (features_with_unit_prices, n_features).
    Matrix with each feature corresponding with the  
    '''
    df = pd.read_csv('data/raw_data/cont_itm.csv',usecols=['UNT_PRIC','ITM_CD'])
    grp = df.groupby('ITM_CD').agg({'UNT_PRIC':'mean'})
    itm_lst = grp.shape[0]
    a = np.zeros((itm_lst,itm_lst))
    np.fill_xdiagonal(a,0.75)
    df2 = pd.DataFrame(a,columns=grp.index,index=grp.index)
    df2['UNT_PRIC'] = grp.UNT_PRIC
    return df2

if __name__ == '__main__':

    train = pd.read_csv('data/train.csv',index_col='project_number')

    # if adding unit prices
    # unit_prices = add_unit_prices()
    # unit_prices.columns.values[-1] = 'bid_total'
    # train = pd.concat([train,unit_prices])

    X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    y = train['bid_total']
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    # model = CDOTModel()
    # model.fit(X_train,y_train)
    # print(model.score(X_test,y_test))

    transformation_lst = [1.12,1.25,1.3,1.4,1.5,1.75,2]
    score_lst = []
    for a in transformation_lst:
        model = CDOTModel()
        model.fit(X_train,y_train,a)
        score_lst.append(model.score(X_test,y_test))

    plt.plot(transformation_lst,score_lst)
    plt.show()
