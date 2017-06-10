import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from src.model import CDOTModel



# def engr_v_model_box(data):
#     sns.set_style('darkgrid')
#     ax = sns.boxplot(data = data.values,
#         palette="muted",
#         width = 0.2,
#         fliersize = 0)
#     ax.set_xticklabels(['Engineer Error','Test Model Error'],fontdict={'fontsize':14,'fontweight':1000})
#     ax.set_ylabel('Percent from Actual',fontdict={'fontsize':14,'fontweight':1000})
#     ax.set_title("CDOT Estimator vs Machine Learning Model",fontdict={'fontsize':18,'fontweight':1000})
#     ax.set_ylim([-110,160])
#     plt.show()

def vs_actual_scatter(engr_estimate,model_estimate, actual):

    plt.close('all')


    #Plots
    ax = sns.regplot(x = actual,y=engr_estimate,fit_reg = False,
        label = "Engineer Estimate",scatter_kws={"s": 20, 'alpha' : .7})

    ax = sns.regplot(x = actual,y=model_estimate,fit_reg = False,
        label = "Model Estimate",scatter_kws={"s": 20, 'alpha' : .7})

    plt.plot([0,actual.max()*1.1],[0,actual.max()*1.1],color = 'black',
        linestyle = '--', alpha = 0.4)

    #Labels
    ax.set_ylabel('Predicted Cost ($M)',fontdict={'fontsize':14,'fontweight':1000})
    ax.set_xlabel('Actual Project Cost ($M)',fontdict={'fontsize':14,'fontweight':1000})
    ax.set_title('CDOT Estimator vs. Model', fontdict={'fontsize':18,'fontweight':1000})
    ax.set_xticklabels([0,0,2,4,6,8,10,12])
    ax.set_yticklabels([0,0,2,4,6,8,10,12])

    ax.legend()

    #Export Figure
    plt.savefig('images/model_engr_scatter.png')

def generate_error(df,model):
    engineers_estimate = df.engineers_estimate.values
    y_true = df.bid_total
    y_predict = model.predict(df.drop(['engineers_estimate','bid_total'],axis=1))
    engineers_error = (engineers_estimate-y_true) / ((engineers_estimate+y_true)/2) * 100
    model_error = (y_predict-y_true) / ((y_predict+y_true)/2) * 100
    return pd.concat([engineers_error,model_error],axis=1)

def X_y(df, model):
    X_engr = df.bid_total
    X_model = model.predict(df.drop(['engineers_estimate','bid_total'],axis=1))
    y = df.bid_total
    return X_engr,X_model,y


if __name__ == '__main__':

    sns.set(style = 'white', palette = 'deep')

    model = CDOTModel()
    test = pd.read_csv('data/test.csv',index_col='project_number')

    X_engr,X_model,y = X_y(test,model)


    vs_actual_scatter(X_engr,X_model,y)


    # Generate boxplot data
    boxplot_data = generate_error(test,model)
