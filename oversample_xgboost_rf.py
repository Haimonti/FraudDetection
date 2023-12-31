
import pandas as pd
import numpy as np
from results.data_processing import DataProcessor
from results.utils import results
from results.models import  xgb_model, random_forests
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
## XgBoost Tuning

res = defaultdict(lambda: defaultdict(dict))
print("XgBoost Results:))")
for test_period in test_periods:
    for data_item in data_items.keys():
        param_grid = {
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.001, 0.01, 0.05, 0.09, 0.1, 0.5]
                    }
        auc = results(data_obj,train_period,test_period,data_items[data_item],xgb_model,param_grid,'over')
        res[test_period][data_item] = auc


with open("oversample_results.txt" "a") as f:
    f.write(res)



## Random Forest

res = defaultdict(lambda: defaultdict(dict))
print("Random Forest Results:))")
for test_period in test_periods:
    for data_item in data_items.keys():
        param_grid = {
                    'n_estimators': [100, 200, 300,400, 500],  # Number of trees in the forest
                    'max_depth': [ 10, 20],  # Maximum depth of the tree
                    'min_samples_split': [5, 10],  # Minimum number of samples required to split an internal node
                    'min_samples_leaf': [5,6]  # Minimum number of samples required to be at a leaf node
                    }
        auc = results(data_obj,train_period,test_period,data_items[data_item],random_forests,param_grid,'over')
        res[test_period][data_item] = auc


with open("oversample_results.txt" "a") as f:
    f.write(res)