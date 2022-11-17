import matplotlib.pyplot as plt


class DataExploration:
    """This class helps to explore the data with a scatterplot
    """

    def __init__(self, data) -> None:
        self.data = data

    def show_scatterplot_binary_outcome(self, outcome_variable: str, x_label: str, y_label: str, column_names: list[str], legend_labels: list[str], marker_type: list[str], colors: list[str]):
        """This method illustrates the outcome variable dependent on the two income variables in a colored scatterplot.

        Args:
            outcome_variable (str): Name of the outcome variable column
            x_label (str): Name of the X label (input variable 1)
            y_label (str): Name of the Y label (input variable 2)
            column_names (list[str]): Array of names of the independent variables in the data set
            legend_labels (list[str]): Array of names for the legend to be shown
            marker_type (list[str]): Array of strings for the point marker types (f.ex. ['x', 'o'])
            colors (list[str]): Array of colors for the observations (f.ex. ['red', 'blue'])
        """
        positive = self.data[self.data[outcome_variable].isin([1])]
        negative = self.data[self.data[outcome_variable].isin([0])]
        _, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(positive[column_names[0]], positive[column_names[1]], s=50,
                   c=colors[0], marker=marker_type[0], label=legend_labels[0])
        ax.scatter(negative[column_names[0]], negative[column_names[1]], s=50,
                   c=colors[1], marker=marker_type[1], label=legend_labels[1])
        ax.legend()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()
