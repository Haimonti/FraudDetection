from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import xgboost as xgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Function to train a Random Under-Sampling Boosting (RUSBoost) classifier
def rus_boost(X_train, y_train, X_test, y_test):
    """
    Trains a RUSBoost classifier on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    # Create a base decision tree classifier with a minimum of 5 samples per leaf
    base_model = DecisionTreeClassifier(min_samples_leaf=5)

    # Create a RUSBoost classifier with 300 estimators and a learning rate of 0.1
    rusboost = AdaBoostClassifier(base_model, n_estimators=300, learning_rate=0.1)

    # Fit the RUSBoost classifier on the training data
    rusboost.fit(X_train, y_train)
 
    # Make predictions on the test data
    y_pred = rusboost.predict(X_test)
    y_scores = rusboost.decision_function(X_test)

    # Calculate the AUC (Area Under the Curve) score to evaluate the model's performance
    auc = roc_auc_score(y_test, y_scores)

    return auc


# Function to train a Support Vector Machine (SVM) classifier
def svm_model(X_train, y_train, X_test, y_test):
    """
    Trains an SVM classifier on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    # Create an SVM classifier with a linear kernel, probability estimates, balanced class weights, and a random state of 42
    svc = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42,max_iter=1000)

    # Fit the SVM classifier on the training data
    svc.fit(X_train, y_train)

    # Get probability scores and calculate the AUC score to evaluate the model's performance
    y_scores = svc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    return auc




# Function to train an XGBoost classifier
def xgb_model(X_train, y_train, X_test, y_test):
    """
    Trains an XGBoost classifier on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, y_train)

    y_scores = xgb_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    return auc


# Function to train a logistic regression model
def logistic_regression_model(X_train, y_train, X_test, y_test):
    """
    Trains a logistic regression model on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    logit_clf = LogisticRegression(solver='liblinear',max_iter=1000) 
    logit_clf.fit(X_train, y_train)

    y_scores = logit_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    return auc


# Function to train a probit regression model
def probit_regression_model(X_train, y_train, X_test, y_test):
    """
    Trains a probit regression model on the training data and evaluates its performance on the test data.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    # Add a constant term to the input features for the probit model
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Create a probit model and fit it to the training data
    probit_model = sm.Probit(y_train, X_train)
    probit_result = probit_model.fit()

    # Make predictions and calculate the AUC score to evaluate the model's performance
    y_scores = probit_result.predict(X_test)
    auc = roc_auc_score(y_test, y_scores)

    return auc


def MLP(X_train, y_train, X_test, y_test,inputs = 28,actv_func='logistic', hidden_lay_neu=(40,50,60,40),learning_rate=0.001):
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
    clf = MLPClassifier(hidden_layer_sizes=(inputs, hidden_lay_neu[0],hidden_lay_neu[1],hidden_lay_neu[2],hidden_lay_neu[3]),
                        max_iter=1000,
                        random_state=42,
                        verbose=False,
                        learning_rate_init=learning_rate,
                        activation=actv_func)

    # Fit data onto the model
    clf.fit(X_train, y_train)
    # Make prediction on test dataset
    ypred = clf.predict(X_test)
    auc = roc_auc_score(y_test, ypred)

    return auc