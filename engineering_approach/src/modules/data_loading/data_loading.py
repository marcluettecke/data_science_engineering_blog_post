import pandas as pd


class DataLoading:
    """Class to load the data from an arbitrary path

    Args:
        path (string): String argument that holds the path (web or local) to load the data - here just for .csv files -
        can be extended to other file types
    """

    def __init__(self, path) -> None:
        self.path = path
        self.data = None

    def load_data_to_pd(self, header: bool, names: list[str]):
        """This method loads the data into a pandas dataframe object.

        Args:
            header (bool): Indicator if the pandas dataframe will load data including a header or not
            names (list[str]): List of column names 
        """
        self.data = pd.read_csv(self.path, header=header, names=names)

    def showcase_pd_data(self, number_of_items: int):
        """Helper function to show the first entries of the data and its actual shape.

        Args:
            number_of_items (int): Number of items to be shown by the preview
        """
        print(f'Data shape is: {self.data.shape}')
        print(self.data.head(number_of_items))
