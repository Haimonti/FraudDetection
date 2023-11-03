from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import xgboost as xgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier



# Function to train a Random Under-Sampling Boosting (RUSBoost) classifier
def rus_boost(X_train, y_train, X_test, y_test,param_grid=None):
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
def svm_model(X_train, y_train, X_test, y_test,param_grid=None):
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
def xgb_model(X_train, y_train, X_test, y_test, param_grid):
    """
    Trains an XGBoost classifier with hyperparameter tuning using grid search on the training data
    and evaluates its performance on the test data.

    Args:
    - X_train: The training input samples.
    - y_train: The target values for the training samples.
    - X_test: The test input samples.
    - y_test: The target values for the test samples.
    - param_grid: A dictionary of hyperparameters for the grid search.

    Returns:
    - auc (float): The AUC (Area Under the Curve) score.
    """
    xgb_clf = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
    
    # Create a GridSearchCV object with the XGBoost classifier and the parameter grid
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc', cv=5)
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train,
                    eval_set = [(X_train,y_train),(X_test,y_test)],
                    verbose=False)

    # Get the best estimator (model) from the grid search
    best_xgb_clf = grid_search.best_estimator_
    

    best_params = grid_search.best_params_
    best_auc = grid_search.best_score_

    # Predict probabilities on the test set
    y_scores = best_xgb_clf.predict_proba(X_test)[:, 1]

    # Print the best hyperparameters, the corresponding AUC score and test AUC Score
    print("Best Hyperparameters:", best_params)
    print("Best AUC Score:", best_auc)
    
    # Calculate AUC
    auc = roc_auc_score(y_test, y_scores)
    print("Test AUC Score:", auc)

    return auc


# Function to train a logistic regression model
def logistic_regression_model(X_train, y_train, X_test, y_test,param_grid=None):
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
def probit_regression_model(X_train, y_train, X_test, y_test,param_grid=None):
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
    clf = MLPClassifier(hidden_layer_sizes=(inputs, *hidden_lay_neu),
                        max_iter=10000,
                        random_state=42,
                        verbose=False,
                        learning_rate_init=learning_rate,
                        activation=actv_func
                        )

    # Fit data onto the model
    clf.fit(X_train, y_train)
    # Make prediction on test dataset
    ypred = clf.predict(X_test)
    auc = roc_auc_score(y_test, ypred)

    return auc


def mlp_grid_search(X_train, y_train, X_test, y_test, param_grid):
    """
    Perform a grid search to find the best hyperparameters for an MLPClassifier model and evaluate its performance.

    Args:
    - X_train : The training input samples.
    - y_train : The target values for the training samples.
    - X_test  : The test input samples.
    - y_test  : The target values for the test samples.
    - param_grid (dict): A dictionary containing hyperparameter values to search.

    Returns:
    - test_auc (float): The Test AUC (Area Under the Curve) score achieved..

    This function uses cross-validated grid search to find the best hyperparameters for an MLPClassifier model
    and evaluates its performance on the test data.
    """
    # Create an MLPClassifier with default values for some hyperparameters
    mlp_model = MLPClassifier(max_iter=4000, random_state=42, verbose=False)

    # Perform grid search using cross-validation
    grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the corresponding AUC score
    best_params = grid_search.best_params_
    best_auc = grid_search.best_score_
    test_auc = grid_search.score(X_test, y_test)
    # Print the best hyperparameters, the corresponding AUC score and test AUC Score
    print("Best Hyperparameters:", best_params)
    print("Best AUC Score:", best_auc)
    print("Test AUC Score:", test_auc)
    # Return the test AUC score
    return test_auc


def random_forests(X_train, y_train, X_test, y_test, param_grid):
    # Define the Random Forest classifier
    rf_model = RandomForestClassifier(random_state=42)

    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }

    # Perform grid search using cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the corresponding AUC score
    best_params = grid_search.best_params_
    best_auc = grid_search.best_score_
    test_auc = grid_search.score(X_test, y_test)

    # Print the best hyperparameters, the corresponding AUC score, and test AUC Score
    print("Best Hyperparameters:", best_params)
    print("Best AUC Score:", best_auc)
    print("Test AUC Score:", test_auc)

    # Return the test AUC score
    return test_auc