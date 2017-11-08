import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler


if __name__ == '__main__':

    train = pd.read_csv('data/model/train.csv',index_col='project_number')
    train.drop(['bid_days','engineers_estimate','start_date'],axis=1,inplace=True)


    y = np.array(train.iloc[:,-1])
    scale = MaxAbsScaler()
    X = scale.fit_transform(train.iloc[:,0:-1])
    #
    # n = X.shape[0]
    # mat = 1/n * np.dot(X.T,X)
    # eig = np.linalg.eig(mat)
    # ev = eig[0]
    # plt.close('all')
    # plt.plot(range(len(ev)),ev)
    # plt.savefig('images/model_development/preprocessing/pca/maxabs.jpg')

    # 95% of the variance is captured at 537 components
    n_components = 537
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    #
    km = KMeans(n_clusters=2)
    km.fit(X)
    clusters = km.predict(X)
    #
    #
    joblib.dump(pca, 'data/model/pca.pkl')
    joblib.dump(scale, 'data/model/scale.pkl')
    joblib.dump(km, 'data/model/km.pkl')
