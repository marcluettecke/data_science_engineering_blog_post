import numpy as np
from helpers.predict_binary import predict


class ModelEvaluation:
    """Class to evaluate the model based on its prediction on a test data set

    Args:
        X_test (pandas df): Test data frame for the model to apply the optimal parameter weights to
        y_test (pandas df): The actual result, which has not been used in the model description
        model_parameters (numpy array): Vector of optimal model parameter weights found by running the model
    """

    def __init__(self, X_test, y_test, model_parameters) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.model_parameters = model_parameters

    def evaluate_binary_outcome(self):
        """this Method uses the X_test matrix in combination with the model parameters to make predictions on the binary outcome.
        It then compares the result to the actual real values and prints the accuray in % to the console.
        """
        theta_min = np.matrix(self.model_parameters[0])
        predictions = predict(theta_min, self.X_test)
        print(f'predictions: {predictions}')
        print(f'y_test: {self.y_test}')
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0))
                   else 0 for (a, b) in zip(predictions, self.y_test)]
        accuracy = (sum(correct) / len(correct))
        print(f'accuracy = {accuracy * 100}%')
