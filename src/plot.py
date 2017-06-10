import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
from src.data_cleaning import CDOTData

def plot_hist(data):
    plt.close('all')
    plt.style.use('fivethirtyeight')
    plt.hist(data)
    plt.savefig('images/hist.png')

def project_prices(data):
    plt.style.use('seaborn')
    plt.close('all')
    fig, ax = plt.subplots()

    df = data.sort_values('engineers_estimate')

    ax.scatter(range(df.shape[0]),df['engineers_estimate'],s=1,alpha=0.5)
    ax.plot([0,1500],[10000000,10000000],color='black',ls='dashed')
    ax.text(150,11000000,'Model includes projects under $10M',fontdict={'fontsize':10})

    ax.set_ylabel('Estimated Project Cost ($10M)')
    ax.set_xlabel('1,400 CDOT Projects')
    ax.set_title('CDOT Estimates by Individual Project',fontdict={'fontsize':20,'fontweight':1000})
    ax.set_xticklabels('')
    ax.set_ylim(bottom = 0,top = 50000000)
    ax.set_xlim(left = 0,right = 1500)
    plt.savefig('images/bar.png')

# def plot(filename):
#     plt.close('all')
#     plt.style.use('seaborn')
#     plt.bar(x_axis,engineers_estimate,label='engineers_estimate',alpha = 0.5)
#     plt.bar(x_axis,y_predict,label='estimate_predictions',alpha = 0.5)
#     plt.bar(x_axis,y_true,label='actual_winning_bid',alpha = 0.5)
#     plt.legend()
#     plt.savefig('images/plot2-1.png')

def engineer_vs_model(data):
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    ax.boxplot(data,
    labels = ['Engineer Error','Test Model Error'],
    positions = [1,1.3],
    widths = 0.1,
    patch_artist = {'color': '#074F57'},
    manage_xticks = True,
    showfliers = False,
    capprops = {'color': '#074F57','linewidth':2},
    whiskerprops = {'color': '#074F57','linewidth':2},
    flierprops = {'color': '#074F57','linewidth':2},
    boxprops = {'facecolor':'#077187', 'color': '#074F57','linewidth':2},
    medianprops = {'color': '#74A57F','linewidth':3})

    ax.set_ylabel('Percent difference between true and actual (%)',fontdict={'fontsize':14,'fontweight':500})
    ax.set_title("CDOT Estimator vs Machine Learning Model",fontdict={'fontsize':20,'fontweight':1000})
    plt.savefig('images/engineer_vs_model2.png')

def generate_error(df,model):
    engineers_estimate = df.engineers_estimate.values
    y_true = df.bid_total.valueste
    y_predict = model.predict(df.drop(['engineers_estimate','bid_total'],axis=1).values)
    engineers_error = (engineers_estimate-y_true) / ((engineers_estimate+y_true)/2) * 100
    model_error = (y_predict-y_true) / ((y_predict+y_true)/2) * 100

    return np.concatenate([[engineers_error,model_error]]).T

if __name__ == '__main__':
    plt.close('all')
    plt.style.use('seaborn')
    model = joblib.load('data/model.pkl')


    # df = pd.read_csv('data/data.csv',usecols=['engineers_estimate']).sort_values('engineers_estimate')
    # plot_hist(y)
    # plot_bar(df)

    #
    # train = pd.read_csv('data/train.csv',index_col='project_number')
    #
    # x_axis = range(len(train.index.values))


    test = pd.read_csv('data/test.csv',index_col='project_number')
    test_data = generate_error(test,model)
    engineer_vs_model(test_data)

    # train = pd.read_csv('data/train.csv',index_col='project_number')
    # train_data = generate_error(train,model)
    # engineer_vs_model(train_data)

    # plot()
    # boxplot(np.concatenate([[test_engineer_error,test_model_error]]).T,('engineers_estimate_error','model_error'),'test error','test_boxplot3.png')

    #train
    # train_eng = train.engineers_estimate.values
    # train_y_true = train.bid_total.values
    # train_y_predict = model.predict(train.drop(['engineers_estimate','bid_total'],axis=1).values)
    #
    # train_engineer_error = (train_eng-train_y_true)/((train_eng+train_y_true)/2)
    # train_model_error = (train_y_predict-train_y_true)/((train_y_predict+train_y_true)/2)

    # plot()
    # boxplot(np.concatenate([[train_engineer_error,train_model_error]]).T,('engineers_estimate_error','model_error'),'train error','train_boxplot3.png')
