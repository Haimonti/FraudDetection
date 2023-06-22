from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tabulate import tabulate
import matplotlib.pyplot as plt


def train_model(X_train, y_train, X_test, y_test,inputs = 28,actv_func='relu', hidden_lay_neu=100,learning_rate=0.001):
    """
    Trains a MLPClassifier model on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.
    - inputs (int): The number of inputs.
    - actv_func (str): The activation function to use.
    - hidden_lay_neu (int): The number of neurons in the hidden layer.
    - learning_rate (float): The learning rate for the model.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    - cm_params (tuple): A tuple containing the confusion matrix values (TN, FP, FN, TP).
    
    """
    clf = MLPClassifier(hidden_layer_sizes=(inputs, hidden_lay_neu),
                        random_state=42,
                        verbose=False,
                        learning_rate_init=learning_rate,
                        activation=actv_func)

    # Fit data onto the model
    clf.fit(X_train, y_train)

    # Make prediction on test dataset
    ypred = clf.predict(X_test)
    auc = metrics.roc_auc_score(y_test, ypred)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, ypred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]  # True negatives ,False positives, False negatives, True positives

    return auc,(TN, FP, FN, TP)


def datasets(data, train_period,validation_period,test_period):
    """
    Loads datasets for specific time periods.

    Args:
    - data (DataFrame): The dataset.
    - train_period (tuple): A tuple containing the start and end years of the training period.
    - validation_period (tuple): A tuple containing the start and end years of the validation period.
    - test_period (tuple): A tuple containing the start and end years of the test period.

    Returns:
    - train_data (DataFrame): The dataset for the training period.
    - validation_data (DataFrame): The dataset for the validation period.
    - test_data (DataFrame): The dataset for the test period.
    
    """
    train_data = data[(data['fyear'] >=train_period[0] ) & (data['fyear'] <= train_period[1])]
    validation_data = data[(data['fyear'] >= validation_period[0] ) & (data['fyear'] <= validation_period[1])]
    test_data = data[(data['fyear'] >= test_period[0] ) & (data['fyear'] <= test_period[1])]
    return train_data, validation_data, test_data

def evaluate(i,inp1,inp2,X_train,y_train,X_test,y_test):
    """
    Evaluates the model performance for different parameter values.

    Args:
    - i (int): The number of neurons in the hidden layer.
    - inp1 (list): List of learning rate values.
    - inp2 (list): List of activation function values.
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc_list (list): List of AUC scores for different parameter combinations.
    - cp_params (list): List of tuples containing confusion matrix values (TN, FP, FN, TP) for different parameter combinations.
    - params (list): List of tuples containing the parameter values used for evaluation.
    
    """
    auc_list,cp_params,params = [],[],[]
    for i1 in inp1:
        for i2 in inp2:
            auc,(TN, FP, FN, TP) = train_model(X_train, y_train, X_test, y_test,inputs = 28,actv_func=i2, 
                                               hidden_lay_neu=i,learning_rate=i1)
            auc_list.append(auc)
            cp_params.append((TN, FP, FN, TP))
            params.append((i,i1,i2))
    return auc_list,cp_params,params

def plot_table(neurons,learning_rates,actv_funs,X_train,y_train,X_test,y_test):
    """
    Plots a table showing the evaluation results for different parameter combinations.

    Args:
    - neurons (int): The number of neurons in the hidden layer.
    - learning_rates (list): List of learning rate values.
    - actv_funs (list): List of activation function values.
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - None
    """
    auc_list,cp_params,params = evaluate(neurons,learning_rates,actv_funs,X_train,y_train,X_test,y_test)
    column1 = [x[0] for x in params]
    column2 = [x[1] for x in params]
    column3 = [x[2] for x in params]
    column4 = [x[0] for x in cp_params]
    column5 = [x[1] for x in cp_params]
    column6 = [x[2] for x in cp_params]
    column7 = [x[3] for x in cp_params]
    column8 = auc_list
    # column6 = fn_list

    #list of lists for the rows
    columns = list(zip(column1, column2, column3, column4, column5, column6, column7, column8))

    #headers for each column
    headers = ['Neurons in HL', 'Learning Rate', 'Activation Func','TN', 'FP', 'FN', 'TP', 'AUC-ROC']

    #table using the tabulate function
    table = tabulate(columns, headers, tablefmt="fancy_grid")

    print(table)

def plot_graph(neurons,learning_rates,actv_funs,X_train,y_train,X_test,y_test):
    """
    Plots a graph showing the AUC values for different parameter combinations.

    Args:
    - neurons (int): The number of neurons in the hidden layer.
    - learning_rates (list): List of learning rate values.
    - actv_funs (list): List of activation function values.
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - None
    """

    auc_list,cp_params,params = evaluate(neurons,learning_rates,actv_funs,X_train,y_train,X_test,y_test)
    param2_values = [param[1] for param in params]
    param3_values = [param[2] for param in params]

    # Plotting
    plt.plot(range(len(auc_list)), auc_list, marker='o')

    # Set x-axis ticks and labels
    plt.xticks(range(len(auc_list)), [f"{p2}, {p3}" for p2, p3 in zip(param2_values, param3_values)], rotation=45)

    # Set axis labels and title
    plt.xlabel('Parameters (LR, Act Func)')
    plt.ylabel('AUC Value')
    plt.title('AUC Values with Parameter Variations')

    # Show the plot
    plt.show()