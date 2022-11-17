import unittest

from modules.data_cleaning.data_cleaning import DataCleaning
from modules.data_loading.data_loading import DataLoading


class TestDataCleaning(unittest.TestCase):
    """
    Test Class for the Data Cleaning process
    """

    def _setUp(self, data_path, header, column_names) -> None:
        self.loader_instance = DataLoading(data_path)
        self.header = header
        self.column_names = column_names
        self.loader_instance.load_data_to_pd(header, column_names)

    def test_create_training_test_split(self, data_path, header, column_names, expected_shapes):
        """Test method for the split of the loaded data into training and test
        data set.

        Args:
            path (str): Path to the data to be loaded
            header (bool): Input variable to indicate wheter the loaded data includes a header
            column_names (list[str]): Array of the columns included in the data
            expected_shapes (list[tupl[int]]): A list of tuples which indicate the expected dimensions, for example,
            if the training data is expected to be 50 observations with three parameters, then the dimensions will be
            (50,2), same goes for the test data.
        """
        self._setUp(data_path, header, column_names)

        cleaning_instance = DataCleaning(self.loader_instance.data)
        DataCleaning.create_training_test_split(cleaning_instance, 0.8)

        self.assertEqual(cleaning_instance.training_data.shape, (expected_shapes[0]),
                         f'Training data shape should be {expected_shapes[0]}')
        self.assertEqual(cleaning_instance.test_data.shape, (expected_shapes[1]),
                         f'Test data shape should be {expected_shapes[1]}')

    def test_prepare_data_log_regression(self, data_path, header, column_names, expected_shapes):
        """Test method for the preparation of the split training and test data into a feature matrix (X)
        and a result vector (y).

        Args:
            path (str): Path to the data to be loaded
            header (bool): Input variable to indicate wheter the loaded data includes a header
            column_names (list[str]): Array of the columns included in the data
            expected_shapes (list[tupl[int]]): A list of tuples which indicate the expected dimensions, for example,
            if the training data is expected to be 50 observations with three parameters, then the dimensions will be
            (50,2), same goes for the test data - both for the X and v matrices.
        """
        self._setUp(data_path, header, column_names)

        cleaning_instance = DataCleaning(self.loader_instance.data)
        cleaning_instance.create_training_test_split(0.8)
        cleaning_instance.prepare_data_log_regression()

        self.assertEqual(cleaning_instance.X_training.shape,
                         expected_shapes[0], f'Formatted X_training data shape should be {expected_shapes[0]}')
        self.assertEqual(cleaning_instance.X_test.shape,
                         expected_shapes[1], f'Formatted X_test data shape should be {expected_shapes[1]}')
        self.assertEqual(cleaning_instance.y_training.shape,
                         expected_shapes[2], f'Formatted y_training data shape should be {expected_shapes[2]}')
        self.assertEqual(cleaning_instance.y_test.shape,
                         expected_shapes[3], f'Formatted y_test data shape should be {expected_shapes[3]}')
