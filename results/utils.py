from imblearn.under_sampling import RandomUnderSampler

# Function to evaluate a machine learning model's performance
def evaluate(item, train_data, test_data, model_name):
    # Extract features (X) and target variable (y) from the training and test data
    X_train, y_train = train_data[item], train_data['misstate']
    X_test, y_test = test_data[item], test_data['misstate'] 

    # Apply random undersampling to balance the class distribution in the training data
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Print the shapes of the resampled training data and the test data
    print("Training data Shape after sampling:", X_train_resampled.shape)
    print("Test data Shape:", X_test.shape)
    if X_test.shape[0] == 0:
        return "Done"
    # Return the results of the machine learning model on the resampled data
    return model_name(X_train_resampled, y_train_resampled, X_test, y_test)

# Function to remove rows with missing values in a specific item (feature)
def null_check(item, train_data, val_data, test_data):
    train_data = train_data.dropna(subset=item)
    val_data = val_data.dropna(subset=item)
    test_data = test_data.dropna(subset=item)
    return train_data, val_data, test_data

# Function to process and evaluate model results
def results(obj, train_period, test_period, item, model_name):
    # Split the data into training, validation, and test sets for the specified periods
    train_data, validation_data, test_data = obj.split_data_periods(train_period, test_period)

    # Remove rows with missing values in the specified item from the datasets
    train_data, validation_data, test_data = null_check(item, train_data, validation_data, test_data)

    # Evaluate the model's performance on the processed data
    return evaluate(item, train_data, test_data, model_name)
