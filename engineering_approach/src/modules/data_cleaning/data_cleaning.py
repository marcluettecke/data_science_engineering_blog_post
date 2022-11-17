import numpy as np


class DataCleaning:
    """The class to clean the loaded data.

    Args:
        raw_data (pandas df): The actual data to be split into train and test data and to be cleaned.
    """

    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data
        self.training_data = None
        self.test_data = None
        self.X_training = None
        self.y_training = None
        self.X_test = None
        self.y_test = None
        self.theta = None

    def create_training_test_split(self, training_proportion: float):
        """Method to split the data into test and training data according to a pre-specified proportion.

        Args:
            training_proportion (float): The proportion of the data to be used for training the model.
        """
        self.training_data = self.raw_data[:int(
            training_proportion * len(self.raw_data))]
        self.test_data = self.raw_data[int(
            training_proportion * len(self.raw_data)):]

    def prepare_data_log_regression(self):
        """The data needs to be prepared to be appropriately consumed by the model function. Here we create 
        the feature matrix X of shape(# of observations, # of features) and the result vector y of shape
        (# of observations, 1)
        """
        self.training_data.insert(0, 'Ones', 1)
        self.test_data.insert(0, 'Ones', 1)

        cols_training = self.training_data.shape[1]
        cols_test = self.test_data.shape[1]
        self.X_training = self.training_data.iloc[:, 0:cols_training-1]
        self.X_test = self.test_data.iloc[:, 0:cols_test-1]
        self.y_training = self.training_data.iloc[:,
                                                  cols_training-1:cols_training]
        self.y_test = self.test_data.iloc[:, cols_test-1:cols_test]

        # convert to numpy arrays and initalize the parameter array theta
        self.X_training = np.array(self.X_training.values)
        self.X_test = np.array(self.X_test.values)
        self.y_training = np.array(self.y_training.values)
        self.y_test = np.array(self.y_test.values)

    def initialize_parameter(self):
        """Initialize our theta vector that will eventually hold the parameter weights optimized in the model
        """
        self.theta = np.zeros(self.X_training.shape[1])
