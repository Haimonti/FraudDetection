from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import xgboost as xgb
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression



# Function to train an AdaBoostClassifier with random undersampling and evaluate it
def rus_boost(X_train, y_train, X_test, y_test):
    metrics = {}
    # Create a base model (DecisionTreeClassifier with minimum leaf samples)
    base_model = DecisionTreeClassifier(min_samples_leaf=5)
    # Create an AdaBoostClassifier with the base model
    rusboost = AdaBoostClassifier(base_model, n_estimators=300, learning_rate=0.1)
    # Fit the AdaBoost model on the training data
    rusboost.fit(X_train, y_train)
    # Make predictions on the test data
    y_pred = rusboost.predict(X_test)
    y_scores = rusboost.decision_function(X_test)
    # Calculate and store the AUC (Area Under the ROC Curve) score
    metrics['auc'] = roc_auc_score(y_test, y_scores)
    return metrics['auc']



# Function to train an SVM (Support Vector Machine) model and evaluate it
def svm_model(X_train, y_train, X_test, y_test):
    svc = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    svc.fit(X_train, y_train)
    # Predict probabilities and calculate AUC score
    y_scores = svc.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_scores)



# Function to train an XGBoost model and evaluate it
def xgb_model(X_train, y_train, X_test, y_test):
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    # Predict probabilities and calculate AUC score
    y_scores = xgb_clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_scores)



# Function to train a logistic regression model and evaluate it
def logistic_regression_model(X_train, y_train, X_test, y_test):
    logit_clf = LogisticRegression(solver='liblinear') 
    logit_clf.fit(X_train, y_train)
    # Predict probabilities and calculate AUC score
    y_scores = logit_clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_scores)



# Function to train a probit regression model using statsmodels and evaluate it
def probit_regression_model(X_train, y_train, X_test, y_test):
    # Add a constant term to the independent variables for the probit model
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    # Fit a probit model on the training data
    probit_model = sm.Probit(y_train, X_train)
    probit_result = probit_model.fit()
    # Predict probabilities and calculate AUC score
    y_scores = probit_result.predict(X_test)
    return roc_auc_score(y_test, y_scores)
