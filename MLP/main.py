# Import Libraries
from utils import datasets, train_model,plot_table,plot_graph
from ensemble_rusboost import ensemble
import json
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # Read data from file - contains 28 financial items, 14 financial ratios, target variable and 
    df = pd.read_csv('../data_FraudDetection_JAR2020.csv')

    # Fill Null values with 0
    data = df.fillna(0)   ## Only Financial ratios have null values

    # Assign Train, Val, and Test periods
    train_period, validation_period, test_period = (1991,1999), (2000,2001), (2003,2008)

    # Load Train, Val and Test data
    train_data, val_data, test_data = datasets(data, train_period,validation_period,test_period)

    # select the features 
    with open('features.json') as json_file:
        data = json.load(json_file)

    raw_financial_items_28 = data['raw_financial_items_28']

    # Split the training and testing data into features and labels
    X_train, y_train = train_data[raw_financial_items_28], train_data['misstate']
    X_test, y_test = test_data[raw_financial_items_28], test_data['misstate'] 

    print("Training data Shape:", X_train.shape)
    print("Testing data Shape:", X_test.shape)

    # Undersampling the data 
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    print("Training data Shape after sampling:", X_train_resampled.shape)

    # Set different values for hyperparameters
    hidden_layer_neurons = [50,70,90,120,150]
    learning_rates = [0.001,0.003,0.005,0.009,0.01]
    actv_funs = ['relu','logistic','tanh']


    # plot_table(50,learning_rates,actv_funs,X_train_resampled,y_train_resampled,X_test,y_test)
    # plot_graph(100,learning_rates,actv_funs,X_train,y_train,X_test,y_test)
    # train_model(X_train_resampled, y_train_resampled, X_test, y_test,inputs = 42,actv_func='relu', hidden_lay_neu=100,learning_rate=0.001)
    print(ensemble(X_train_resampled, y_train_resampled, X_test, y_test))