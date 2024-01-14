class DataProcessor:
    def __init__(self, data, train_period, val_period, test_period, window_size):
        """
        Initialize the DataProcessor with data and parameters for data splitting.

        Args:
            data: DataFrame containing the dataset.
            train_period: A tuple representing the start and end years for the training period.
            val_period: A tuple representing the start and end years for the validation period.
            test_period: A tuple representing the start and end years for the test period.
            window_size: Size of the data window used for processing.
        """
        self.data = data
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period
        self.window_size = window_size

    def split_data_periods(self, train_period, test_period):
        """
        Split the data into training, validation, and test sets based on the specified periods.

        Args:
            train_period: A tuple representing the start and end years for the training period.
            test_period: A tuple representing the start and end years for the test period.

        Returns:
            train_data: DataFrame containing the training data.
            validation_data: DataFrame containing the validation data.
            test_data: DataFrame containing the test data.
        """
        train_data = self.data[(self.data['fyear'] >= train_period[0]) & (self.data['fyear'] <= train_period[1])]
        validation_data = self.data[(self.data['fyear'] >= self.val_period[0]) & (self.data['fyear'] <= self.val_period[1])]
        test_data = self.data[(self.data['fyear'] >= test_period[0]) & (self.data['fyear'] <= test_period[1])]
        return train_data, validation_data, test_data

    def create_batches(self):
        """
        Create batches of training and test data using a sliding window approach.

        Returns:
            train_batches: List of tuples representing training data batches.
            test_batches: List of tuples representing test data batches.
        """
        train_batches, test_batches = [], []
        start_year = 1990
        end_year = start_year + self.window_size
        while end_year <= 2023:
            train_batches.append((start_year, end_year - 1))
            test_batches.append((end_year, end_year + self.window_size - 1))

            start_year += 5  # Increment the start year by 5 years
            end_year = start_year + self.window_size

        return train_batches, test_batches

