import numpy as np
import pandas as pd

from time import time
from scipy.stats import randint as sp_randint
from random import randint

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error, make_scorer

# get some data]
train = pd.read_csv('data/train.csv',index_col='project_number')
X = train.drop(['engineers_estimate', 'bid_total'],axis=1)
y = train['bid_total']

# build a regressor
clf = GradientBoostingRegressor()

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# specify parameters and distributions to sample from
# param_dist = {"max_depth": [3, None],
#               "max_features": sp_randint(1, 11),
#               "min_samples_split": sp_randint(2, 11),
#               "min_samples_leaf": sp_randint(1, 11),
#               "bootstrap": [True, False],
#               "criterion": ["mse", "mae"]}

param_dist = {'loss': ['ls','lad','huber','quantile'],
    # 'learning_rate': randint(1,10)/10,
    'n_estimators': sp_randint(100,120),
    # 'max_depth': sp_randint(1,10000),
    'criterion': ['friedman_mse','mse','mae'],
    # 'min_samples_split': sp_randint(2,10000),
    # 'min_samples_leaf': sp_randint(2,10000),
    # 'min_weight_fraction_leaf': randint(0,10)/10,
    # 'subsample':randint(1,10)/10,
    # 'max_features': sp_randint(1,X.shape[1]),
    # 'max_leaf_nodes': sp_randint(1,10000)
    }

# run randomized search
n_iter_search = 20
rs = RandomizedSearchCV(clf, param_distributions=param_dist,
    n_iter=n_iter_search, n_jobs = 4, scoring = make_scorer(mean_squared_error))

start = time()
rs.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(rs.cv_results_)

# # use a full grid over all parameters
# param_grid = {"max_depth": [3, None],
#               "max_features": [1, 3, 10],
#               "min_samples_split": [2, 3, 10],
#               "min_samples_leaf": [1, 3, 10],
#               "bootstrap": [True, False],
#               "criterion": ["mse", "mae"]}
#
# # run grid search
# grid_search = GridSearchCV(clf, param_grid=param_grid)
# start = time()
# grid_search.fit(X, y)
#
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)
