from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE
from fraudDetec.models import MLP, mlp_grid_search
import pandas as pd
from sklearn.utils import shuffle

# Function to evaluate a machine learning model's performance
def evaluate(item, train_data, test_data, model_name,param_grid=None,sample='under'):
    """
    Args:
    - item: A specific item (feature) to be used for evaluation.
    - train_data: Training data containing features and target variable.
    - test_data: Test data containing features and target variable.
    - model_name: A function representing the machine learning model to be evaluated.
    - param_grid: parameter grid for grid search tuning
    - sample: under for undersampling, over for oversampling, over_under for percentage over sampling and then undersampling
    Returns:
    - If test data is empty, returns "Done."
    - Otherwise, returns the results of the machine learning model on resampled data.
    """
    X_train, y_train = train_data[item], train_data['misstate']
    X_test, y_test = test_data[item], test_data['misstate']
    if sample == 'under':
        rus = RandomUnderSampler(random_state=42)
        X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    elif sample == 'over':
        X_train_resampled, y_train_resampled = BorderlineSMOTE(random_state=42).fit_resample(X_train, y_train)
    elif sample == 'over_under':
        X_train_resampled, y_train_resampled = BorderlineSMOTE(sampling_strategy=0.1, random_state=42).fit_resample(X_train, y_train)
        X_train_resampled, y_train_resampled = RandomUnderSampler(random_state=42).fit_resample(X_train_resampled, y_train_resampled)
    import statsmodels.api as sm

    # Combine X_train_resampled and y_train_resampled into a single DataFrame
    combined_data = pd.concat([pd.DataFrame(X_train_resampled), pd.DataFrame(y_train_resampled, columns=['misstate'])], axis=1)

    # Shuffle the combined DataFrame
    shuffled_data = shuffle(combined_data, random_state=42)

    # Separate the shuffled data back into features (X_train_shuffled) and labels (y_train_shuffled)
    X_train_shuffled = shuffled_data.drop(columns=['misstate'])
    y_train_shuffled = shuffled_data['misstate']

    if X_test.shape[0] == 0:
        return "Done"
    print("Train Shape: ",X_train_shuffled.shape, y_train_shuffled.shape)
    print("Test Shape: ",X_test.shape, y_test.shape)
    if model_name == MLP:
        return model_name(X_train_shuffled, y_train_shuffled, X_test, y_test,inputs = len(item),actv_func='logistic', hidden_lay_neu=(40,50,60,40),learning_rate=0.005)
    return model_name(X_train_shuffled, y_train_shuffled, X_test, y_test,param_grid)

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
def results(obj, train_period, test_period, item, model_name,param_grid=None,sample='under'):
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
    return evaluate(item, train_data, test_data, model_name,param_grid,sample)

