import pandas as pd
from sklearn.model_selection import train_test_split


class CDOTData(object):

    def __init__(self):
        self.X, self.y = self._create_X_y()

    def _fill_feature(self, empty_df,dense_df):
        '''
        Parameters
        ----------
        empty_df: pandas dataframe, a dataframe with the index being each
            project number and columns as bid item
        dense_df: pandas dataframe, a dataframe coordinates to use to fill
            the empty matrix

        Output
        ------
        filled_df: pandas dataframe, feature matrix
        '''
        for row in dense_df.iterrows():
            empty_df[row[1][0]][row[0]] = row[1][2]

        return empty_df.fillna(0)


    def _remove_rows(self,feature_data,target_data):
        '''
        Parameters
        ----------
        feature_data: pandas dataframe, the feature matrix
        target_data: pandas dataframe, the target matrix

        Output
        ------
        rows_to_keep: set, the rows that both the target and feature
            set contain
        drop_rows_feature: set, the rows in the feature that are not
            contained in the target set
        drop_rows_target: set, the rows in the target that are not
            contained in the feature set
        '''
        feature_projects = set(feature_data.index)
        target_projects = set(target_data.index)
        rows_to_keep = feature_projects.intersection(target_projects)
        drop_rows_feature = feature_projects - rows_to_keep
        drop_rows_target = target_projects - rows_to_keep
        return rows_to_keep, drop_rows_feature, drop_rows_target

    def _import_target_data(self):
        '''
        imports the target data as well as the CDOT estimate
        '''

        df = pd.read_csv('data/raw_data/bidding_info.csv',
                 usecols=['Proposal Number','Bid Total','Engineers Estimate', 'Awarded'],
                index_col = 'Proposal Number')
        df = df[df.Awarded == 1].drop('Awarded',axis=1)
        df.columns = ['bid_total','engineers_estimate']
        return df.applymap(lambda x: x.replace('$','').replace(',','')).astype(float)

    def _create_X_y(self):
        '''
        Parameters
        ----------
            None

        Output
        ------
        X: pandas dataframe, feature matrix of each project as a row and
            each item number as a feature
        y: pandas dataframe, target matrix, contains the actual target as
            well as CDOT's estimate
        '''

        feature_data = pd.read_csv('data/raw_data/cont_itm.csv',
            usecols=['CONT_ID','ITM_CD','SPC_YR','BID_QTY'],
            index_col = 'CONT_ID')

        target_data = self._import_target_data()

        # Gather the rows and columns for the feature matrix
        feature_list = pd.read_csv('data/raw_data/t_itm.csv',usecols=['ITM_CD']).ITM_CD.unique()
        projects, drop_from_feature,drop_from_target = self._remove_rows(feature_data,target_data)

        # Create X
        feature_data.drop(drop_from_feature,inplace=True)
        empty_df = pd.DataFrame(index = projects,columns = feature_list)
        X = self._fill_feature(empty_df,feature_data)

        #Create y
        y = target_data.drop(drop_from_target)

        return X.sort_index(),y.sort_index()


    def create_train_test(self):
        '''
        Creates train.csv and test.csv within the data folder
        '''
        all_data = self.X.copy()
        for col in self.y.columns:
            all_data[col] = self.y[col]
        all_data.index.name = 'project_number'

        train,test = train_test_split(all_data)

        train.to_csv('data/train.csv')
        test.to_csv('data/test.csv')


if __name__ == '__main__':
    data = CDOTData()
    data.create_train_test()
