import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log,exp,sqrt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.externals import joblib
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import StandardScaler

class CDOTModel(object):

    def __init__(self,model_num,test = None, scale = StandardScaler()):
        self.model_num = model_num
        self.test = test
        self.std = scale

    def save_model(self):
        etr = joblib.load('data/model/'+str(self.model_num)+'-etr.pkl')
        gbr = joblib.load('data/model/'+str(self.model_num)+'-gbr.pkl')
        std = joblib.load('data/model/'+str(self.model_num)+'-std.pkl')
        joblib.dump(etr, 'data/model/best'+str(self.model_num)+'-etr.pkl')
        joblib.dump(gbr, 'data/model/best'+str(self.model_num)+'-gbr.pkl')
        joblib.dump(std, 'data/model/best'+str(self.model_num)+'-std.pkl')

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
        X = self.std.fit_transform(X)
        y = y.values
        joblib.dump(self.std, 'data/model/'+str(self.model_num)+'-std.pkl')

        #ExtraTreesRegressor
        print ('Fitting ExtraTreesRegressor model...')
        # etr = ExtraTreesRegressor(bootstrap = False, max_features = 'sqrt', min_samples_leaf = 1,
        #     min_samples_split = 2, n_estimators = 25, n_jobs = -1, random_state = 10)
        etr = ExtraTreesRegressor(n_estimators = 25,max_features = 2000, max_depth = 65)
        etr.fit(X,y)
        joblib.dump(etr, 'data/model/'+str(self.model_num)+'-etr.pkl')
        print ('ExtraTreesRegressor complete.\n')

        #GradientBoostingRegressor
        print ('Fitting GradientBoostingRegressor model...')
        # gbr = GradientBoostingRegressor(n_estimators = 50, loss = 'ls',
        # max_features = 'sqrt',max_depth = 3, random_state = 10)
        gbr = GradientBoostingRegressor(n_estimators = 100,max_features = 1000,loss = 'huber',alpha = 0.75,subsample=0.5)
        gbr.fit(X,y)
        joblib.dump(gbr, 'data/model/'+str(self.model_num)+'-gbr.pkl')
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
        etr = joblib.load('data/model/'+str(self.model_num)+'-etr.pkl')
        gbr = joblib.load('data/model/'+str(self.model_num)+'-gbr.pkl')
        self.std = joblib.load('data/model/'+str(self.model_num)+'-std.pkl')

        #Add etr precitions to feature matrix
        X = self.std.transform(X)

        etr_pred = etr.predict(X)
        gbr_pred = gbr.predict(X)

        y_predict = (etr_pred + gbr_pred) / 2

        return y_predict

    def r2(self, X, y):
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

    def mse(self, X, y):
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

        return mean_squared_error(y, self.predict(X))/1000000000


def add_unit_prices():
    '''
    Returns
    -------
    unit_prices : array-like, shape (features_with_unit_prices, n_features).
        Matrix with the corresponding feature set to a value of 1, and the
        target being the estimated unit prices for that feature
    '''

    up = pd.read_csv('data/raw_data/cont_itm.csv',usecols=['UNT_PRIC','ITM_CD'])
    grp_up = up.groupby('ITM_CD').agg({'UNT_PRIC':'mean'})
    itm_lst = grp.shape[0]
    empty_matrix = np.zeros((itm_lst,itm_lst))
    np.fill_xdiagonal(empty_matrix,1)
    unit_prices = pd.DataFrame(a,columns=grp.index,index=grp.index)
    unit_prices['UNT_PRIC'] = grp.UNT_PRIC
    return unit_prices

if __name__ == '__main__':


    # if adding unit prices
    # unit_prices = add_unit_prices()
    # unit_prices.columns.values[-1] = 'bid_total'
    # train = pd.concat([train,unit_prices])


    train = pd.read_csv('data/train.csv',index_col='project_number')
    X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    y = train['bid_total']
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    # min_weight_fraction_leaf = [0, 0.1,0.3,0.4]
    subsample = [0.1,0.3,0.4,0.6,0.8]
    max_features = [1000,2000,3000,4000,5000,6000,7000,8000]



    limit = range(1,4)
    transformation_lst = max_features
    hyperparameter = 'max_features'
    avg_r2 = [0] * len(transformation_lst)
    for _ in limit:
        r2_lst = []
        mse_lst = []
        for hyperp in transformation_lst:
            model = CDOTModel(hyperp)
            model.fit(X_train,y_train)
            r2_lst.append(model.r2(X_test,y_test))
        avg_r2 = [r2_lst[i] + avg_r2[i] for i in range(len(r2_lst))]

    plt.close('all')
    # plt.plot(transformation_lst,avg_r2)
    plt.plot(range(len(transformation_lst)),avg_r2)
    plt.xlabel(transformation_lst,fontdict={'fontsize':14,'fontweight':1000})
    plt.savefig('images/model_training/gbr/gbr_r2_'+str(hyperparameter)+'.png')


    # X_train = X_train.applymap(lambda x: x**2)
    # rid = joblib.load('data/model/rid.pkl')







    # # X = np.insert(X,0,etr.predict(X),axis=1)
