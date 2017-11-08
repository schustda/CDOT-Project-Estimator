import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv('data/data/cdot_data.csv',index_col='project_number')
    # df.drop(['engineers_estimate','bid_days','start_date'],inplace=True,axis=1)
    train,test = train_test_split(df,random_state=10)
    train.to_csv('data/model/train.csv')
    test.to_csv('data/model/test.csv')
