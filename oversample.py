import pandas as pd
import numpy as np
from MLP.utils import datasets,plot_table,train_model
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from tabulate import tabulate
import matplotlib.pyplot as plt
from results.data_processing import DataProcessor
from results.utils import evaluate,null_check,results
from results.models import rus_boost, svm_model, xgb_model, logistic_regression_model,probit_regression_model,MLP,mlp_grid_search, random_forests
import json
from collections import defaultdict

data = pd.read_csv('./data/merged_compustat_and_labels.csv')
with open('MLP/features.json') as json_file:
    data_items = json.load(json_file)

data.replace([np.inf, -np.inf], np.nan, inplace=True)

data = data.fillna(0)

data_obj = DataProcessor(data,(1990,2002), (2002,2002), (2003,2019), 5)

models = {"MLP": MLP,
         "RUS BOOST": rus_boost,
          "Logit":logistic_regression_model,
         "Probit":probit_regression_model,
         "Xg Boost":xgb_model}


### Batch Processing

test_periods = [(2003,2019)]

train_period = (1990,2002)
res = defaultdict(lambda: defaultdict(dict))
for test_period in test_periods:
    for model in models.keys():
        for data_item in data_items.keys():
            auc = results(data_obj,train_period,test_period,data_items[data_item],models[model],'over')
            res[test_period][data_item][model] = auc


with open("oversample_results.txt" "a") as f:
    f.write(res)



## Batch - Tuning

res = defaultdict(lambda: defaultdict(dict))
for test_period in test_periods:
    for data_item in data_items.keys():
        param_grid = {
            'activation': ['logistic', 'tanh', 'relu'],
            'hidden_layer_sizes': [
                (len(data_items[data_item]), 40),
                (len(data_items[data_item]), 50),
                (len(data_items[data_item]), 40, 50),
                (len(data_items[data_item]), 30, 40),
                (len(data_items[data_item]), 40, 30, 50),
                (len(data_items[data_item]), 30, 40, 30),
                (len(data_items[data_item]), 40, 50, 60, 40),
                (len(data_items[data_item]), 50, 50, 50, 50),
                (len(data_items[data_item]), 30, 30, 30, 30)
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
                (len(data_items[data_item]), 20),
                (len(data_items[data_item]), 30),
                (len(data_items[data_item]), 40, 50, 10),
                (len(data_items[data_item]), 30, 40, 10),
                (len(data_items[data_item]), 40, 30, 50, 20),
                (len(data_items[data_item]), 30, 40),
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


## Window Processing

# train_batches,test_batches = data_obj.create_batches()

# for train_period,test_period in zip(train_batches[:-1],test_batches[:-1]):
#     print(train_period,test_period)

# data_items = features_comp
# res = defaultdict(lambda: defaultdict(dict))
# for train_period,test_period in zip(train_batches[:-1],test_batches[:-1]):
#     for model in models.keys():
#         for data_item in data_items.keys():
#             auc = results(data_obj,train_period,test_period,data_items[data_item],models[model],'over')
#             res[str(train_period) + '-' +str(test_period)][data_item][model] = auc

# for key in res.keys():
#     columns = ["Train - Test : "+str(key), 'MLP', 'RUS BOOST', 'Logit', 'Probit']
#     df = pd.DataFrame.from_dict(res[key], orient='index').reset_index()
#     df.columns = columns
#     for col in ['MLP', 'RUS BOOST', 'Logit', 'Probit']:
         
#         try:
#             df[col] = df[col].round(3)
#         except:
#             pass
#     df.set_index("Train - Test : "+str(key), inplace=True)
#     print(tabulate(df, headers='keys', tablefmt='fancy_grid'))



# ## MLP Param Tuning

# train_batches,test_batches = data_obj.create_batches()

# for train_period,test_period in zip(train_batches[:-1],test_batches[:-1]):
#     print(train_period,test_period)

# data_items = features_comp

# from results.models import mlp_grid_search
# model = mlp_grid_search
# res = defaultdict(lambda: defaultdict(dict))


# for train_period,test_period in zip(train_batches[:-1],test_batches[:-1]):

#     for data_item in data_items.keys():
        
#         param_grid = {
#             'activation': ['logistic', 'tanh', 'relu'],
#             'hidden_layer_sizes': [
#                 (len(data_items[data_item]), 40),
#                 (len(data_items[data_item]), 50),
#                 (len(data_items[data_item]), 40, 50),
#                 (len(data_items[data_item]), 30, 40),
#                 (len(data_items[data_item]), 40, 30, 50),
#                 (len(data_items[data_item]), 30, 40, 30),
#                 (len(data_items[data_item]), 40, 50, 60, 40),
#                 (len(data_items[data_item]), 50, 50, 50, 50),
#                 (len(data_items[data_item]), 30, 30, 30, 30)
#                 ],
#             'learning_rate_init': [0.001, 0.01, 0.1]
#             }
#         auc = results(data_obj,train_period,test_period,data_items[data_item],model,param_grid,'over')
#         res[str(train_period) + '-' +str(test_period)][data_item] = auc

# res

