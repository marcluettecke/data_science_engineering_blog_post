import unittest

from modules.data_loading.data_loading import DataLoading


class TestDataLoading(unittest.TestCase):
    """
    Test Class for the Data Loading process
    """

    def test_load_data_to_pd(self, path, header, column_names):
        """Test method for the actual loading of data. It asserts the loaded
        data shape against expected dimensions.

        Args:
            path (str): Path to the data to be loaded
            header (bool): Input variable to indicate wheter the loaded data includes a header
            column_names (list[str]): Array of the columns included in the data
        """
        loader_instance = DataLoading(path)
        loader_instance.load_data_to_pd(header, column_names)
        loaded_data = loader_instance.data
        self.assertEqual(loaded_data.shape, (100, 3),
                         'Data shape should be (100,3)')
