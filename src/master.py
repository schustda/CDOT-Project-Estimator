from src.data_cleaning import CDOTData
from src.model import CDOTModel
import pandas as pd
from sklearn.model_selection import train_test_split



if __name__ == '__main__':

    # data = CDOTData()
    # data.create_train_test

    train = pd.read_csv('data/train.csv',index_col='project_number')

    X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    y = train['bid_total']

    X_train,X_test,y_train,y_test = train_test_split(X,y)

    model = CDOTModel()
    model.fit(X_train,y_train)
    print (model.score(X_test,y_test))
