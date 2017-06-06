import pandas as pd

df = pd.read_csv('data/raw_data/change_orders.csv')

df = df[['C_O_AMT','CONT_ID']]
df.columns = ['co_amt','proposal_number']
df.index = df.proposal_number
df.drop('proposal_number',axis=1,inplace=True)
df = df.applymap(lambda x: float(x.replace('$','').replace(',','')))
df = df.groupby(df.index).sum()
# df.to_csv('data/data/change_orders_by_project.csv')
df_project = pd.read_csv('data/data/project_numbers_and_costs.csv',index_col = 'proposal_number')

df_combined = df_project.join(df).fillna(0)

df_co_percentage = df_combined.co_amt / df_combined.engineers_estimate
