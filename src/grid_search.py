from src.data_cleaning import CDOTData
from src.model import CDOTModel
from src.plot import Plotting
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Define Top Lists
top_5_mse = [0]*5
top_5_params = [None]*5
best_score = 10000

# Import X and y
train = pd.read_csv('data/train.csv',index_col='project_number')
X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
y = train['bid_total']
X_train,X_test,y_train,y_test = train_test_split(X,y)

# Params


# Loop
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
