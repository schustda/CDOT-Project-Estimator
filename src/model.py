import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.externals import joblib
import pickle

def add_unit_prices():
    df = pd.read_csv('data/raw_data/cont_itm.csv',usecols=['UNT_PRIC','ITM_CD'])
    grp = df.groupby('ITM_CD').agg({'UNT_PRIC':'mean'})
    itm_lst = grp.shape[0]
    a = np.zeros((itm_lst,itm_lst))
    np.fill_diagonal(a,0.75)
    df2 = pd.DataFrame(a,columns=grp.index,index=grp.index)
    df2['UNT_PRIC'] = grp.UNT_PRIC
    return df2


if __name__ == '__main__':
    unit_prices = add_unit_prices()
    unit_prices.columns.values[-1] = 'bid_total'
    train = pd.read_csv('data/train.csv',index_col='project_number')
    train.drop('engineers_estimate',axis=1,inplace=True)
    train = pd.concat([train,unit_prices])
    train.fillna(0,inplace=True)

    X = train.drop('bid_total',axis=1)
    y = train['bid_total']

    # test = pd.read_csv('data/test.csv',index_col='project_number')

    X_train,X_test,y_train,y_test = train_test_split(X,y)

    # model = RandomForestRegressor()
    # model.fit(X_train,y_train)

    model = GradientBoostingRegressor(n_estimators = 1000)
    model.fit(X_train,y_train)

    joblib.dump(model, 'data/model.pkl')
