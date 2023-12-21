import pandas as pd

data = pd.read_csv(".././data/updated_ratios_compustat_1999_2022.csv")
df_labels = pd.read_csv('.././data/Annual_Labels2.csv')

# Merge dataframes on specified columns
merged_df = pd.merge(data, df_labels, left_on=['cik', 'fyear'], right_on=['CIK', 'YEARA'], how='left')

# Add a column with 1 for each record that is in df_labels
merged_df['misstate'] = merged_df['CIK'].notnull().astype(int)
final_merged_data = merged_df[list(data.columns)+['misstate']]
final_merged_data = final_merged_data.drop(columns=['Unnamed: 0'])

final_merged_data.to_csv('.././data/merged_compustat_and_labels.csv',index=False)
