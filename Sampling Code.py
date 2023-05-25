import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics


# read the CSV file
df = pd.read_csv('data_FraudDetection_JAR2020.csv')

# replace nan values with zeros
df.fillna(0, inplace=True)

# write the DataFrame back to a CSV file
df.to_csv('data_FraudDetection_JAR2020_v2.csv', index=False)

# read the new CSV file into a DataFrame
data = pd.read_csv('data_FraudDetection_JAR2020_v2.csv')

# select the columns to use as features
features = ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib', 'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect',
            'sale', 'sstk', 'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets', 'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf']

start_year = 1991
end_year = 2002
test_year = end_year + 2

# split the data into training and testing by year
train_data = data[(data['fyear'] >= start_year) & (data['fyear'] <= end_year)]
test_data = data[data['fyear'] == test_year]

print(f"Training period: ", start_year, end_year)
print(f"Testing period: ", test_year)

# Count positive and negative cases
train_misstate_1 = train_data['misstate'].value_counts()[1]
test_misstate_1 = test_data['misstate'].value_counts()[1]
train_misstate_0 = train_data['misstate'].value_counts()[0]
test_misstate_0 = test_data['misstate'].value_counts()[0]

print(f"Positives cases training: ", train_misstate_1)
print(f"Positives cases test: ", test_misstate_1)
print(f"Negative case training: ", train_misstate_0)
print(f"Negative case test: ", test_misstate_0)

# Select 227 random observations from train_data where 'misstate' is equal to 1
train_sample_1 = train_data[train_data['misstate']
                            == 1].sample(n=train_misstate_1, random_state=42)
# Select 227 random observations from train_data where 'misstate' is equal to 0
train_sample_0 = train_data[train_data['misstate']
                            == 0].sample(n=train_misstate_1, random_state=42)

# Combine the two sample sets
train_sample = pd.concat([train_sample_1, train_sample_0], axis=0)

# Print the number of observations in the train and test sets
print(f"Number of observations in train set: {len(train_sample)}")
print(f"Number of observations in test set: {len(test_data)}")


# Split the training and testing data into features and labels
X_train = train_sample[features]
y_train = train_sample['misstate']

X_test = test_data[features]
y_test = test_data['misstate']

# define empty lists to store the AUC values
auc_list = []
layers_list = []

# loop for different layers
for i in range(50, 301, 50):
    # Create model
    clf = MLPClassifier(hidden_layer_sizes=(42, i),
                        random_state=5,
                        verbose=False,
                        learning_rate_init=0.001)

    # Fit data onto the model
    clf.fit(X_train, y_train)

    # Make prediction on test dataset
    ypred = clf.predict(X_test)

    auc = metrics.roc_auc_score(y_test, ypred)

    # append the AUC and layers to respective lists
    auc_list.append(auc)
    layers_list.append((42, i))

# print the AUC values for each MLP combination
for i in range(len(layers_list)):
    print(f"AUC for MLP with {layers_list[i]} layers: {auc_list[i]}")
