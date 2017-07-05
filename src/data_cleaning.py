import pandas as pd
from sklearn.model_selection import train_test_split


class CDOTData(object):

    def __init__(self,proj_min = 0, proj_max = 100000000,full_dataset_created = True, model_num = ''):
        self.proj_min = proj_min
        self.proj_max = proj_max
        self.full_dataset_created = full_dataset_created
        self.model_num = model_num

    def _get_co(self):
        '''
        imports the change order data

        Returns
        -------
        change_orders (co): pandas dataframe, the sum of all change orders for
            a given project
        '''
        co = pd.read_csv('data/raw_data/change_orders.csv',usecols = ['C_O_AMT','CONT_ID'],
            index_col = 'CONT_ID')
        co.index.name = 'proposal_number'
        co = co.applymap(lambda x: float(x.replace('$','').replace(',','')))
        return co.groupby(co.index).sum()

    def _convert_units(self, df, itm_dict, conversions):
        '''
        Parameters
        ----------
        df: pandas dataframe, a dataframe with the index being each
            project number and columns as bid item
        itm_dict:
        conversions:

        Returns
        -------
        filled_df: pandas dataframe, feature matrix
        '''
        for row in df.iterrows():
            if itm_dict[row[1]['ITM_CD']] != row[1]['UNT_T']:
                df.loc[row[0]]['BID_QTY'] = row[1]['BID_QTY'] * conversions[(row[1]['UNT_T'],itm_dict[row[1]['ITM_CD']])]
        return df

    def _fill_feature(self, empty_df,dense_df):
        '''
        Parameters
        ----------
        empty_df: pandas dataframe, a dataframe with the index being each
            project number and columns as bid item
        dense_df: pandas dataframe, a dataframe coordinates to use to fill
            the empty matrix

        Returns
        -------
        filled_df: pandas dataframe, feature matrix
        '''

        for row in dense_df.iterrows():
            empty_df[row[1]['ITM_CD']][row[0]] = row[1]['BID_QTY']
        return empty_df.fillna(0)

    def _remove_rows(self,feature_data,target_data):
        '''
        Parameters
        ----------
        feature_data: pandas dataframe, the feature matrix
        target_data: pandas dataframe, the target matrix

        Returns
        -------
        rows_to_keep: set, the rows that both the target and feature
            set contain
        drop_rows_feature: set, the rows in the feature that are not
            contained in the target set
        drop_rows_target: set, the rows in the target that are not
            contained in the feature set
        '''

        # Which projects are contained in both the bid item and total price datasets
        rows_to_keep = set(feature_data.index).intersection(set(target_data.index))

        # Determine which rows to drop in each data set
        drop_rows_feature = set(feature_data.index) - rows_to_keep
        drop_rows_target = set(target_data.index) - rows_to_keep

        return rows_to_keep, drop_rows_feature, drop_rows_target

    def _import_target_data(self):
        '''
        imports the target data as well as the CDOT estimate data

        Returns
        -------
        target_data: pandas dataframe, the target columns ('bid_total') as well
            as the CDOT Engineer's Estimate ('engineers_estimate')
        '''

        df = pd.read_csv('data/raw_data/bidding_info.csv',
                usecols=['Proposal Number','Bid Total','Engineers Estimate', 'Awarded'],
                index_col = 'Proposal Number')

        # Only use the project that were awarded
        df = df[df.Awarded == 1].drop('Awarded',axis=1)

        # Rename columns
        df.columns = ['bid_total','engineers_estimate']

        # Convert currenty fo float
        df = df.applymap(lambda x: x.replace('$','').replace(',','')).astype(float)

        # Import Change Order data
        co = self._get_co()

        # Join change orders and fill nulls with zeros
        df = df.join(co).fillna(0)

        # Add total change order value to bid total to get the total project cost
        df.bid_total = df.bid_total + df.C_O_AMT
        return df.drop('C_O_AMT',axis=1)

    def _import_feature_data(self):

        print ("Importing quantities")
        df = pd.read_csv('data/raw_data/cont_itm.csv',
            usecols=['CONT_ID','ITM_CD','LAST_CHNG_YR','BID_QTY','UNT_T'],
            low_memory=False)
        df.sort_values(['LAST_CHNG_YR','ITM_CD'],inplace=True)
        itm_dict = df.groupby('ITM_CD').agg({'UNT_T':'last'}).to_dict()['UNT_T']
        conversions = {('CY', 'M3'):1.30795,('HA', 'ACRE'):0.404686,
             ('KG', 'LB'):0.453592,('L', 'GAL'): 3.78541,
             ('LF', 'M'):3.28084,('M', 'LF'):0.3048,
             ('M2', 'SF'):0.092903,('M2', 'SY'):0.836127,
             ('M3', 'CF'):0.0283168,('M3', 'CY'):0.764555,
             ('M3', 'MFBM'):0.00235974,('M3', 'MGAL'):0.2642,
             ('SF', 'M2'):10.7639,('SY', 'M2'):1.19599,
             ('TON', 'T'):1,('T', 'TON'):1,
             ('L S', 'EACH'):1,('MNM', 'MKFT'):1,
             ('LS', 'L S'): 1,('F A', 'HOUR'):1
            }
        print ('Converting different units...')
        feature_data = self._convert_units(df,itm_dict,conversions)
        feature_data.set_index('CONT_ID',inplace=True)
        return feature_data.drop(['LAST_CHNG_YR','UNT_T'],axis=1)

    def _create_X_y(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        X: pandas dataframe, feature matrix of each project as a row and
            each item number as a feature
        y: pandas dataframe, target matrix, contains the actual target as
            well as CDOT's estimate
        '''

        print ('Importing and project data... \n')
        feature_data = self._import_feature_data()
        print ('Importing target data')
        target_data = self._import_target_data()

        # Gather the rows and columns for the feature matrix
        feature_list = pd.read_csv('data/raw_data/t_itm.csv',usecols=['ITM_CD']).ITM_CD.unique()
        projects, drop_from_feature,drop_from_target = self._remove_rows(feature_data,target_data)

        # Create X
        feature_data.drop(drop_from_feature,inplace=True)
        empty_df = pd.DataFrame(index = projects,columns = feature_list)
        X = self._fill_feature(empty_df,feature_data).sort_index()

        #Create y
        y = target_data.drop(drop_from_target).sort_index()

        return X,y

    def create_train_test(self):
        '''
        Creates train.csv and test.csv within the data folder
        '''

        # Create the full dataset if not yet existing
        if not self.full_dataset_created:
            self.X, self.y = self._create_X_y()
            full_dataset = self.X.copy()
            for col in self.y.columns:
                full_dataset[col] = self.y[col]
            full_dataset.index.name = 'project_number'
            full_dataset.fillna(0,inplace=True)
            full_dataset.sort_index(axis=1,inplace=True)
            full_dataset.to_csv('data/full_dataset.csv')

        # If the full dataset DOES exist, load it...
        else:
            full_dataset = pd.read_csv('data/full_dataset.csv',index_col='project_number')

        # Reduce number of projects to the original scope
        data = full_dataset[(full_dataset.engineers_estimate < self.proj_max) & (full_dataset.engineers_estimate > self.proj_min)]

        # Split and save to train.csv and test.csv
        print('Splitting train and test data...')
        train,test = train_test_split(data)
        train.to_csv('data/'+str(self.model_num)+'-train.csv')
        test.to_csv('data/'+str(self.model_num)+'-test.csv')
        print ('Datasets Created \n')
