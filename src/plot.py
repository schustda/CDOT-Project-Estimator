import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle


def plot(filename):
    plt.close('all')
    plt.style.use('seaborn')
    plt.bar(x_axis,engineers_estimate,label='engineers_estimate',alpha = 0.5)
    plt.bar(x_axis,y_predict,label='estimate_predictions',alpha = 0.5)
    plt.bar(x_axis,y_true,label='actual_winning_bid',alpha = 0.5)
    plt.legend()
    plt.savefig('images/plot2-1.png')

def boxplot(data, labels, title, filename):
    plt.close('all')
    plt.boxplot(data,labels = labels)
    plt.title(title)
    plt.savefig('images/'+filename)


if __name__ == '__main__':

    plt.style.use('seaborn')
    model = joblib.load('data/model.pkl')

    train = pd.read_csv('data/train.csv',index_col='project_number')
    test = pd.read_csv('data/test.csv',index_col='project_number')

    x_axis = range(len(train.index.values))
    test_eng = test.engineers_estimate.values
    test_y_true = test.bid_total.values
    test_y_predict = model.predict(test.drop(['engineers_estimate','bid_total'],axis=1).values)

    test_engineer_error = (test_eng-test_y_true)/((test_eng+test_y_true)/2)
    test_model_error = (test_y_predict-test_y_true)/((test_y_predict+test_y_true)/2)

    # plot()
    boxplot(np.concatenate([[test_engineer_error,test_model_error]]).T,('engineers_estimate_error','model_error'),'test error','test_boxplot3.png')




    train_eng = train.engineers_estimate.values
    train_y_true = train.bid_total.values
    train_y_predict = model.predict(train.drop(['engineers_estimate','bid_total'],axis=1).values)

    train_engineer_error = (train_eng-train_y_true)/((train_eng+train_y_true)/2)
    train_model_error = (train_y_predict-train_y_true)/((train_y_predict+train_y_true)/2)

    # plot()
    boxplot(np.concatenate([[train_engineer_error,train_model_error]]).T,('engineers_estimate_error','model_error'),'train error','train_boxplot3.png')
