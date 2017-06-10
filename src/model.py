import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import pickle

class CDOTModel(object):

    def fit(self, X, y):

        #ExtraTreesRegressor
        print ('Fitting ExtraTreesRegressor model... \n')
        etr = ExtraTreesRegressor(bootstrap = False, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 1000, n_jobs = -1)
        etr.fit(X,y)
        joblib.dump(etr, 'data/model/etr.pkl')
        X.loc[:,'etr'] = etr.predict(X)

        #Ridge
        print ('Fitting Ridge model... \n')
        rid = Ridge(fit_intercept = False, solver = 'lsqr')
        rid.fit(X,y)
        joblib.dump(rid, 'data/model/rid.pkl')
        X.loc[:,'rid'] = rid.predict(X)

        #GradientBoostingRegressor
        print ('Fitting GradientBoostingRegressor model... \n')
        gbr = GradientBoostingRegressor(n_estimators = 1000)
        gbr.fit(X,y)
        joblib.dump(gbr, 'data/model/gbr.pkl')

        print ('Models Complete! \n')

    def predict(self, X):
        '''
        Parameters
        ----------
        X:

        Output
        ------
        y_predict:
        '''

        #Load models
        etr = joblib.load('data/model/etr.pkl')
        rid = joblib.load('data/model/rid.pkl')
        gbr = joblib.load('data/model/gbr.pkl')

        #Add etr precitions to feature matrix
        X.loc[:,'etr'] = etr.predict(X)
        X.loc[:,'rid'] = rid.predict(X)

        return gbr.predict(X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))

def add_unit_prices():
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

    model = CDOTModel()
    model.fit(X_train,y_train)
    print (model.score(X_test,y_test))
