from src.data_cleaning import CDOTData
from src.model import CDOTModel
import pandas as pd
from sklearn.model_selection import train_test_split
from src.plot import Plotting
from sklearn.externals import joblib
# from src.mlp_soln import MLP


if __name__ == '__main__':


    # # Collect and split data
    # data = CDOTData(10000000)
    # data.create_train_test()

    # Fit Model
    train = pd.read_csv('data/train.csv',index_col='project_number')
    X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    y = train['bid_total']
    X_train,X_test,y_train,y_test = train_test_split(X,y)

    best_score = 10000

    for _ in range(20):
        model = CDOTModel()
        model.fit(X_train,y_train)
        print ('r2 score: '+ str(model.r2(X_test,y_test)))
        print ('mse: '+ str(model.mse(X_test,y_test)))
        if model.mse(X_test,y_test) < best_score:
            print ('best model! \n')
            m = model
            best_score = model.mse(X_test,y_test)

    joblib.dump(m, 'data/model/gbr.pkl')



    # Score and Plot
    # model = CDOTModel()
    # test = pd.read_csv('data/test.csv',index_col='project_number')
    # plot = Plotting(test,model,'model_training/gbr_test')
    # plot.vs_actual_scatter()
    # plot = Plotting(train,model,'model_training/gbr_train')
    # plot.vs_actual_scatter()

    # X_score = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    # y_score = train['bid_total']





    # train = pd.read_csv('data/train.csv',index_col='project_number')
    # X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
    # y = train['bid_total']
    # X_train,X_test,y_train,y_test = train_test_split(X,y)
    # model = CDOTModel()
