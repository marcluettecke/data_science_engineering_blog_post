import scipy.optimize as opt


class LogisticModel:
    """Class to run the model with the help of SciPy library
    """

    def __init__(self, X_training, y_training, theta, cost_function, gradient) -> None:
        self.X_training = X_training
        self.y_training = y_training
        self.cost_function = cost_function
        self.gradient = gradient
        self.theta = theta
        self.theta_min = None

    def find_optimal_paramethers(self):
        """This method finds the optimal (minimal cost) parameters of the model and saves them as theta_min
        It utilizes a lot of the pre-defined helper function for cost and gradient functions. Can be used as 
        a dynamic entry point for model evaluation, utilizing different cost or gradient approaches.
        """
        self.theta_min = opt.fmin_tnc(func=self.cost_function, x0=self.theta, fprime=self.gradient,
                                      args=(self.X_training, self.y_training))
