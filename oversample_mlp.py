import pandas as pd
import numpy as np
from results.data_processing import DataProcessor
from results.utils import results
from results.models import mlp_grid_search
import json
from collections import defaultdict

data = pd.read_csv('./data/merged_compustat_and_labels.csv')
with open('MLP/features.json') as json_file:
    data_items = json.load(json_file)

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data = data.fillna(0)

data_obj = DataProcessor(data,(1990,2002), (2002,2002), (2003,2019), 5)


test_periods = [(2003,2019)]

train_period = (1990,2002)



## Batch - Tuning

res = defaultdict(lambda: defaultdict(dict))
for test_period in test_periods:
    for data_item in data_items.keys():
        param_grid = {
            'activation': ['logistic', 'tanh', 'relu'],
            'hidden_layer_sizes': [
                (len(data_items[data_item]), 40, 30, 50),
                (len(data_items[data_item]), 40, 50, 60, 40),
                ],
            'learning_rate_init': [0.001, 0.01, 0.1]
            }
        auc = results(data_obj,train_period,test_period,data_items[data_item],mlp_grid_search,param_grid,'over')
        res[test_period][data_item] = auc

with open("oversample_results.txt" "a") as f:
    f.write(res)


res = defaultdict(lambda: defaultdict(dict))
for test_period in test_periods:
    for data_item in data_items.keys():
        param_grid = {
            'activation': ['logistic', 'tanh', 'relu'],
            'hidden_layer_sizes': [
                (len(data_items[data_item]), 40, 50, 40, 40,20),
                (len(data_items[data_item]), 30, 50, 50, 40,45),
                (len(data_items[data_item]), 20,30,40,50,20)
                ],
            'learning_rate_init': [0.001, 0.01, 0.05, 0.09, 0.1, 0.5]
            }
        auc = results(data_obj,train_period,test_period,data_items[data_item],mlp_grid_search,param_grid,'over')
        res[test_period][data_item] = auc

with open("oversample_results.txt" "a") as f:
    f.write(res)
