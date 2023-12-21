#imports
import pandas as pd
import json

data = pd.read_csv('.././data/Bao_28_items_1990_2023_v3.csv')

with open('.././MLP/features.json') as json_file:
    features = json.load(json_file)

data = data.dropna(subset=['at'])
data = data[data['at'] != 0]

data['Bank'] = (data['sic'] >= 6000) & (data['sic'] <= 6999)
data['Bank'] = data['Bank'].astype(int)

data[features['raw_financial_items_28']+['ppent']] = \
    data[features['raw_financial_items_28']+['ppent']].fillna(0)

df = data.copy()

## Computing the ratios

#1 changes in working capital accruals
df['wc'] = (df['act'] - df['che']) - (df['lct'] - df['dlc'] - df['txp'])
df['ch_wc'] = df['wc'] - df['wc'].shift(1)
df['dch_wc'] = df['ch_wc'] * 2 / (df['at'] - df['at'].shift(1))

#2 changes in RSST_accruals
df['nco'] = (df['at'] - df['act'] - df['ivao']) - (df['lt'] - df['lct'] - df['dltt'])
df['ch_nco'] = df['nco'] - df['nco'].shift(1)

df['fin'] = (df['ivst'] + df['ivao']) - (df['dltt'] + df['dlc'] + df['pstk'])
df['ch_fin'] = df['fin'] - df['fin'].shift(1)

df['ch_rsst'] = (df['ch_wc'] + df['ch_nco'] + df['ch_fin']) * 2 / (df['at'] + df['at'].shift(1))

#3 changes in receivables
df['ch_rec'] = df['rect'] - df['rect'].shift(1)
df['dch_rec'] = df['ch_rec'] * 2 / (df['at'] + df['at'].shift(1))

#4 changes in inventories
df['ch_inv'] = df['invt'] - df['invt'].shift(1)
df['dch_inv'] = df['ch_inv'] * 2 / (df['at'] + df['at'].shift(1))

#5 percentage of soft assets
df['soft_assets'] = (df['at'] - df['ppent'] - df['che']) / df['at']

#6 percentage change in cash sales
df['cs'] = df['sale'] - (df['rect'] - df['rect'].shift(1))
df['ch_cs'] = (df['cs'] - df['cs'].shift(1)) / df['cs'].shift(1)

#7 change in cash margin
df['cmm'] = (df['cogs'] - (df['invt'] - df['invt'].shift(1)) + (df['ap'] - df['ap'].shift(1))) / (df['sale'] - (df['rect'] - df['rect'].shift(1)))
df['ch_cm'] = (df['cmm'] - df['cmm'].shift(1)) / df['cmm'].shift(1)

#8 change in return on assets
df['roa'] = (df['ni'] * 2) / (df['at'] + df['at'].shift(1))

df['ch_roa'] = df['roa'] - df['roa'].shift(1)

#9 actual issuance
df['issue'] = ((df['sstk'] > 0) | (df['dltis'] > 0)).astype(int)

#10 Book-to-market
df['bm'] = df['ceq'] / (df['prcc_f'] * df['csho'])

#11 Depreciation Index (Ratio from Beneish 1999)
df['dpi'] = (df['dp'].shift(1) / (df['dp'].shift(1) + df['ppent'].shift(1))) / (df['dp'] / (df['dp'] + df['ppent']))

#12 Retained earnings over assets
df['reoa'] = df['re'] / df['at']

#13 Earnings before interest and tax (Ratios from Summers and Sweeney, 1998)
df['EBIT'] = (df['ni'] + df['xint'] + df['txt']) / df['at']

#14 changes in free cash flow
df['ch_ib'] = df['ib'] - df['ib'].shift(1)
df['ch_fcf'] = df['ch_ib'] - df['ch_rsst']

selected_columns = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'issue', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf']
data[selected_columns] = df[selected_columns]


data.to_csv('.././data/updated_ratios_compustat_1999_2022.csv')

print("Ratios computed and updated file stored in the data dolder")