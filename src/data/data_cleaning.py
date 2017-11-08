import pandas as pd

class CDOTData(object):

    def _get_dates(self):
        '''
        imports the project date data

        Returns
        -------
        dates: pandas dataframe, includes number of bid days and start date
        '''
        dates = pd.read_csv('data/raw_data/project_dates.csv',index_col='Contract ID')
        dates.loc[dates['Time Type']=='WORK DAYS', ['Bid Days']] *= 365/260.89
        dates.drop(['Time Type','Accepted Date','Days Charged'],axis=1,inplace=True)
        dates.columns = ['bid_days','start_date']
        return dates

    def _add_missing_features(self, df):
        feature_set = set(pd.read_csv('data/raw_data/t_itm.csv',usecols=['ITM_CD']).ITM_CD.values)
        missing_features = feature_set - set(df.columns)
        missing_features_df = pd.DataFrame(0,index=df.index,columns=missing_features)
        return pd.concat([df,missing_features_df],axis=1).fillna(0).sort_index(axis=1)

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
        # co = co.applymap(lambda x: float(x.replace('$','').replace(',','')))
        co = co.applymap(lambda x: float(x.replace('$','').replace(',','')))
        return co.groupby(co.index).sum()

    def _convert_units(self, df, itm_dict):
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

        conversions = {
        ('CY', 'M3'):1.30795,
        ('HA', 'ACRE'):0.404686,
        ('KG', 'LB'):0.453592,
        ('L', 'GAL'): 3.78541,
        ('LF', 'M'):3.28084,
        ('M', 'LF'):0.3048,
        ('M2', 'SF'):0.092903,
        ('M2', 'SY'):0.836127,
        ('M3', 'CF'):0.0283168,
        ('M3', 'CY'):0.764555,
        ('M3', 'MFBM'):0.00235974,
        ('M3', 'MGAL'):0.2642,
        ('SF', 'M2'):10.7639,
        ('SY', 'M2'):1.19599,
        ('TON', 'T'):1,
        ('T', 'TON'):1,
        ('L S', 'EACH'):1,
        ('MNM', 'MKFT'):1,
        ('LS', 'L S'): 1,
        ('F A', 'HOUR'):1
        }

        print ('Converting inconsistent units...')

        for row in df.iterrows():
            if itm_dict[row[1]['ITM_CD']] != row[1]['UNT_T']:
                df.loc[row[0]]['BID_QTY'] = row[1]['BID_QTY'] * conversions[(row[1]['UNT_T'],itm_dict[row[1]['ITM_CD']])]
        return df

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

        # Import Change Order data and join
        print('Adding Change Orders...')
        co = self._get_co()
        df = df.join(co).fillna(0)

        # Add total change order value to bid total to get the total project cost
        df.bid_total = df.bid_total + df.C_O_AMT
        return df.drop('C_O_AMT',axis=1)

    def _import_feature_data(self):

        print ("Importing project quantities...")
        df = pd.read_csv('data/raw_data/cont_itm.csv',
            usecols=['CONT_ID','ITM_CD','LAST_CHNG_YR','BID_QTY','UNT_T'],
            low_memory=False)

        # creating the dictionary with each bid item and its most current UOM
        df.sort_values(['LAST_CHNG_YR','ITM_CD'],inplace=True)
        itm_dict = df.groupby('ITM_CD').agg({'UNT_T':'last'}).to_dict()['UNT_T']

        # converts units that are incorrect
        df = self._convert_units(df,itm_dict)
        df.drop(['LAST_CHNG_YR','UNT_T'],axis=1)

        # transform matrix to sparse
        df = df.groupby(['CONT_ID','ITM_CD'])['BID_QTY'].sum().unstack()

        return self._add_missing_features(df)


    def create_dataset(self):
        '''
        Creates train.csv and test.csv within the data folder
        '''
        print ('Importing and project data...')
        feature_data = self._import_feature_data()

        print ('Importing Project Dates...')
        dates = self._get_dates()

        print ('Importing target data...')
        target_data = self._import_target_data()

        df = feature_data.join(target_data,how='inner').join(dates)
        df.index.name = 'project_number'

        df.to_csv('data/data/cdot_data.csv')


if __name__ == '__main__':
    data = CDOTData()
    df = data.create_dataset()
