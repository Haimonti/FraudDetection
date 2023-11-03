from imblearn.under_sampling import RandomUnderSampler
from results.models import MLP, mlp_grid_search
# Function to evaluate a machine learning model's performance
def evaluate(item, train_data, test_data, model_name,param_grid=None):
    """
    Args:
    - item: A specific item (feature) to be used for evaluation.
    - train_data: Training data containing features and target variable.
    - test_data: Test data containing features and target variable.
    - model_name: A function representing the machine learning model to be evaluated.

    Returns:
    - If test data is empty, returns "Done."
    - Otherwise, returns the results of the machine learning model on resampled data.
    """
    X_train, y_train = train_data[item], train_data['misstate']
    X_test, y_test = test_data[item], test_data['misstate']
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    if X_test.shape[0] == 0:
        return "Done"
    if model_name == MLP:
        return model_name(X_train_resampled, y_train_resampled, X_test, y_test,inputs = len(item),actv_func='logistic', hidden_lay_neu=(40,50,60,40),learning_rate=0.005)
    return model_name(X_train_resampled, y_train_resampled, X_test, y_test,param_grid)

# Function to remove rows with missing values in a specific item (feature)
def null_check(item, train_data, val_data, test_data):
    """
    Args:
    - item: A specific item (feature) to be checked for missing values.
    - train_data: Training data to remove/fill rows with 0 missing values from.
    - val_data: Validation data to remove/fill rows with 0 rows with missing values from.
    - test_data: Test data to remove/fill rows with 0 rows with missing values from.

    Returns:
    - Modified training, validation, and test datasets with rows containing missing values in the specified item either removed or filled in with 0.
    """
    # train_data = train_data.dropna(subset=item)
    # val_data = val_data.dropna(subset=item)
    # test_data = test_data.dropna(subset=item)

    train_data = train_data.fillna(0)
    val_data = val_data.fillna(0)
    test_data = test_data.fillna(0)

    return train_data, val_data, test_data

# Function to process and evaluate model results
def results(obj, train_period, test_period, item, model_name,param_grid=None):
    """
    Args:
    - obj: An object containing data and methods to split data into training, validation, and test sets.
    - train_period: Time period for training data.
    - test_period: Time period for test data.
    - item: A specific item (feature) to be used for evaluation.
    - model_name: A function representing the machine learning model to be evaluated.

    Returns:
    - The results of the machine learning model after processing and evaluation.
    """
    train_data, validation_data, test_data = obj.split_data_periods(train_period, test_period)
    train_data, validation_data, test_data = null_check(item, train_data, validation_data, test_data)
    return evaluate(item, train_data, test_data, model_name,param_grid)

