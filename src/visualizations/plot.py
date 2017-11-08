import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from src.model.model import CDOTModel


def asphalt_prices(p,y):
    plt.close('all')

    #Plots
    ax = sns.pointplot(x = y,y=p,fit_reg = False,
    scatter_kws={"s": 20, 'alpha' : .7}, label = 'Asphalt Prices')

    #Labels
    ax.set_xlabel('Year',fontdict={'fontsize':14,'fontweight':1000})
    ax.set_ylabel('Cost ($/Ton)',fontdict={'fontsize':14,'fontweight':1000})
    ax.set_title('Hot Mix Asphalt Prices', fontdict={'fontsize':18,'fontweight':1000})
    ax.set_xticklabels([2012,'','','',2013,'','','',2014,'','','',2015,'','','',2016])
    # ax.set_yticklabels([0,0,2,4,6,8,10,12])
    ax.legend()

    #Export Figure
    plt.savefig('images/asphalt_prices.png')


class Plotting(object):

    def __init__(self,data,model_num,plotname = None):

        self.data = data
        self.model_num = model_num
        self.plotname = plotname
        self.model = CDOTModel()

    def engr_v_model_box(self):

        boxplot_data = self.percent_error(self.data,self.model)

        sns.set_style('darkgrid')
        ax = sns.boxplot(data = boxplot_data,
            palette="muted",
            width = 0.2,
            fliersize = 0)
        ax.set_xticklabels(['Engineer Error','Test Model Error'],fontdict={'fontsize':14,'fontweight':1000})
        ax.set_ylabel('Percent from Actual',fontdict={'fontsize':14,'fontweight':1000})
        ax.set_title("CDOT Estimator vs Machine Learning Model "+str(self.plotname),fontdict={'fontsize':18,'fontweight':1000})
        ax.set_ylim([-110,160])
        plt.savefig('images/boxplot'+str(self.plotname)+'.png')

    def vs_actual_scatter(self):
        colors = ['red','yellow','green']
        y_engr,y_model,X = self.X_y(self.data,self.model)
        plt.close('all')

        #Plots
        ax = sns.regplot(x = X,y=y_engr,fit_reg = False, color = 'blue',
            label = "Engineer Estimate",scatter_kws={"s": 20, 'alpha' : .7})
        ax = sns.regplot(x = X,y=y_model,fit_reg = False, color = 'red',
            label = "Model Estimate",scatter_kws={"s": 20, 'alpha' : .7})
        plt.plot([0,X.max()*1.1],[0,X.max()*1.1],color = 'black',
            linestyle = '--', alpha = 0.4)

        #Labels
        ax.set_ylabel('Predicted Cost ($)',fontdict={'fontsize':14,'fontweight':1000})
        ax.set_xlabel('Actual Project Cost ($)',fontdict={'fontsize':14,'fontweight':1000})
        ax.set_title('Estimator vs. Model', fontdict={'fontsize':18,'fontweight':1000})
        ax.legend()

        #Export Figure
        plt.savefig('images/final_model.png',dpi = 200)

    def percent_error(self, df,model):
        engineers_estimate = df.engineers_estimate.values
        y_true = df.bid_total
        X = df.drop(['engineers_estimate','bid_total'],axis=1)
        y_predict = model.predict(X)
        engineers_error = (engineers_estimate-y_true) / ((engineers_estimate+y_true)/2) * 100
        model_error = (y_predict-y_true) / ((y_predict+y_true)/2) * 100
        return pd.concat([engineers_error,model_error],axis=1)

    def X_y(self, df, model):
        X_engr = df.engineers_estimate
        X = df.drop(['engineers_estimate','bid_total',
            'bid_days','start_date'],axis=1)
        y_model = model.predict(X)
        y = df.bid_total
        return X_engr,y_model,y


if __name__ == '__main__':

    sns.set(style = 'white', palette = 'deep')
    test = pd.read_csv('data/model/test.csv',index_col='project_number')
    model = CDOTModel()

    plot = Plotting(test,model)
    plot.vs_actual_scatter()
    # plot.engr_v_model_box()
    # a = plot.engr_v_model_box()



    # Plotting asphalt price history
    # asphalt_prices_history = np.array([83.52,82.65,90.76,102.24,76.07,84.37,85,
    # 80.78,92.28,88.13,100.07,113.42,83.80,94.22,98.61,81.21,84.03])
    # y = np.array(range(len(asphalt_prices_history)))
    # asphalt_prices(asphalt_prices_history, y)
