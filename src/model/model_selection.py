import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from src.model.general_model_functions import GeneralModelFunctions
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

class ModelSelection(GeneralModelFunctions):
    def __init__(self,model_name,testing_param,param_values,param_dict={},final=False,
                        num_iterations=10, threshold = 0.5):
        super().__init__()
        self.model_name = model_name
        self.testing_param = testing_param
        self.param_values = param_values
        self.param_dict = param_dict
        self.summary = self._emptydf(self.model_name)
        self.final = final
        self.num_iterations = num_iterations

    def _plot(self,ax,data,score):
        data[score] = data[score].astype(float)
        color = self.colors[score]

        # Plot Test
        sns.set_color_codes("muted")
        sns.boxplot(x='param_values', y=score, data=data,
                    color=color, ax=ax)
        if self.final == True:
            xticklabels = ''
        else:
            xticklabels = data.param_values

        # Add a legend and informative axis label
        ax.legend(ncol=2, loc="upper right", frameon=True)
        ax.set_xticklabels(xticklabels)
        ax.set(ylabel="",xlabel="")
        ax.set_title(score)

    def _train_models(self,idx):

        model_dict = {
                'linear_regression': LinearRegression(),
                'decision_tree': DecisionTreeRegressor(),
                'random_forest': RandomForestRegressor(),
                'gradient_boost': GradientBoostingRegressor(),
                'extra_trees': ExtraTreesRegressor(),
                'svr': SVR()
                }

        if self.model_name == 'xbg':
            dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
            dtest = xgb.DMatrix(self.X_test)
            model = xgb.train(self.param_dict,self.dtrain)
            y_pred = model.predict(self.dtest)

        else:
            model = model_dict[self.model_name]
            model.set_params(**self.param_dict)
            model.fit(self.X_train,self.y_train)
            y_pred = model.predict(self.X_test)

        self.summary.loc[idx,'r2':'mape'] = self.score_model(self.y_test,y_pred)


    def run(self):
        print ('Testing paramaters for {0}'.format(self.testing_param))
        train = pd.read_csv('data/model/train.csv',index_col='project_number')
        X = np.array(train.iloc[:,0:-4])
        X,_ = self.preprocessing(X)
        y = np.array(train.iloc[:,-4])

        for num in range(1,self.num_iterations):
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y)
            print ('Iteration: {0}'.format(num))
            for idx,value in enumerate(self.param_values):
                print ('Param value: {0}'.format(value))
                self.param_dict[self.testing_param] = value
                self._train_models(idx+1)

            if num == 1:
                self.df = self.summary.copy()
            else:
                self.df = pd.concat([self.df,self.summary])
        self.summary = self.df.copy()

        # Set up figure
        if self.final == True:
            self.figname = '_final'
        else:
            self.figname = self.testing_param
        plt.close('all')
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(1, 3, figsize=(10,4), sharex=True)
        fig.tight_layout(rect=[0, 0.03, 1, .88])
        fig.suptitle("{0} scores for {1}".format(self.figname,self.model_name),fontsize = 20)
        sns.despine(left=True,bottom=True)

        # Plot results
        scores = ['r2','mse','mape']
        axes = [ax[0],ax[1],ax[2]]
        for score_type,axis in zip(scores,axes):
            print ('plotting {0} charts'.format(score_type))
            self._plot(axis,self.summary,score_type)
        plt.savefig('images/model_development/model_selection/{0}/{1}.jpg'
            .format(self.model_name,self.figname))



if __name__ == '__main__':

    # DecisionTreeRegressor
    dt = 'decision_tree'
    dt_parameters = {
            'criterion': ['mse'],
            # 'criterion': ['mse','friedman_mse','mae'],
            # 'splitter': ['best','random'],
            # 'max_depth': [1,2,3,5,10,25,50,75,100,200,None],
            # 'min_samples_split':[2,3,4,5,10,15,20],
            # 'min_samples_leaf': [1,2,3,4,5,10,15,20],
            # 'min_weight_fraction_leaf': [0,.1,.2,.3,.4,.49],
            # 'max_features':[None,2,3,4,5,6,10,15,20,'auto','sqrt','log2'],
            # 'max_leaf_nodes': [None,5,10,15,20,50],
            # 'min_impurity_decrease':[0,.25,.5,.75],
            # 'presort': [True,False]
            }
    dt_dict = {
            'criterion': 'mse',
            'min_samples_split':20,
            'splitter':'random',
            }

    # linear_regression
    lr = 'linear_regression'
    lr_parameters = {
            'fit_intercept':[False]
            # 'fit_intercept':[True,False],
            }
    lr_dict = {
            'fit_intercept':False
            }

    # xgboost
    xgb_name = 'xgboost'
    xgb_parameters = {
                    # 'max_depth': [5]
                    # 'booster':['gbtree','gblinear'],
                    # 'max_depth':[0,1,2,5,10,15,20,25,35,50,100],
                    'max_depth':[5,7,9,11,13,15],
                    # 'eta':[0.01,0.05,0.1,0.2,0.3,0.5,0.8,0.95],
                    # depends on the loss function
                    'gamma':[i/10 for i in range(0,5)],
                    # 'objective':['reg:linear','count:poisson'],
                    # 'objective':['binary:logistic','binary:logitraw'],
                    # 'booster':['gbtree','gblinear','dart'],
                    # 'tree_method':['auto','exact','approx'],
                    # 'process_type': ['default','update'],
                    # 'grow_policy':['depthwise','lossguide']
                    # 'threshold':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    # 'min_child_weight': [0.25,0.5,0.75,1],
                    # 'subsample': [0.25,0.5,0.75,1],
                    'subsample': [0.5,0.6,0.7,0.8,0.9,1],
                    # 'colsample_bytree':[0.5,0.7,0.9,1],
                    # 'lambda': [5,10,20,50,100],
                    # 'eval_metric': ['rmse','mae','logloss']
                }
    xgb_dict = {
        'max_depth':7,
        'silent':1,
        'eta':0.3,
        # 'gamma':0,
        # 'objective':'binary:logistic'
        'booster':'gbtree'
        }

    gb_name = 'gradient_boost'
    gb_parameters = {
                    # 'loss':['ls','lad','huber','quantile'],
                    'criterion':['friedman_mse','mse','mae'],
                    'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                    'max_depth':[1,2,5,10,15,20,25,50,100],
                    'n_estimators':[50,100,200,500,1000],
                    }
    gb_dict = {
                    'loss':'lad'
                    }

    et_name = 'extra_trees'
    et_parameters = {
                    'n_estimators':[50,100,200,500,1000],
                    'criterion':['mse','mae'],
                    'max_features':['auto','sqrt','log2',None],
                    'max_depth':[1,2,5,10,15,20,25,50,100],
                    }
    et_dict = {}

    svr_name = 'svr'
    svr_parameters = {
                    'kernel':['rbf','linear','poly','sigmoid'],

                    }


    param_dict = et_dict
    parameters = et_parameters
    model_name = et_name
    final = False
    for a,b in parameters.items():

        testing_param = a
        param_values = b

        ms = ModelSelection(model_name=model_name,testing_param=testing_param,
                param_values=param_values,param_dict=param_dict,final=final,
                num_iterations=10)
        a = ms.run()
