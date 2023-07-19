from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def ensemble(X_train, y_train, X_test, y_test):
    """
    Trains an AdaBoost ensemble classifier using a DecisionTreeClassifier as the base model and evaluates its performance.

    Args:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """

    metrics = {} 

    base_model = DecisionTreeClassifier(min_samples_leaf=5)  # Define the base model for AdaBoost
    rusboost = AdaBoostClassifier(base_model, n_estimators=300, learning_rate=0.1)  # Create an AdaBoost classifier

    rusboost.fit(X_train, y_train)  # Train the AdaBoost classifier using the training data

    y_pred = rusboost.predict(X_test)  # Predict labels for the test data
    y_scores = rusboost.decision_function(X_test)  # Obtain the decision function scores for the test data

    metrics['auc'] = roc_auc_score(y_test, y_scores)  # Calculate the ROC AUC score and store it in the metrics dictionary

    return metrics  # Return the evaluation metrics dictionary
