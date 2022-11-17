import os

from helpers.cost_function import cost
from helpers.gradient_function import gradient
from modules.data_cleaning.data_cleaning import DataCleaning
from modules.data_exploration.data_exploration import DataExploration
from modules.data_loading.data_loading import DataLoading
from modules.logistic_model.logistic_model import LogisticModel
from modules.model_evaluation.model_evaluation import ModelEvaluation
from tests.data_cleaning_unittest import TestDataCleaning
from tests.data_loading_unittest import TestDataLoading

loader_instance = DataLoading(
    './data_science_engineering_blog_post/engineering_approach/data/exam_data.txt')
DataLoading.load_data_to_pd(loader_instance, None, ['Exam1', 'Exam2', 'Pass'])
# DataLoading.showcase_pd_data(loader_instance)

cleaning_instance = DataCleaning(loader_instance.data)
DataCleaning.create_training_test_split(cleaning_instance, 0.8)
cleaning_instance.prepare_data_log_regression()
cleaning_instance.initialize_parameter()
training_data = cleaning_instance.training_data

exploration_instance = DataExploration(training_data)
positive = training_data[training_data['Pass'].isin([1])]
negative = training_data[training_data['Pass'].isin([0])]
# exploration_instance.show_scatterplot_binary_outcome(outcome_variable='Pass', x_label='Exam 1 Score', y_label='Exam 2 Score', column_names=[
#                                                      'Exam1', 'Exam2'], legend_labels=['Pass', 'Not Pass'], marker_type=['o', 'x'], colors=['blue', 'red'])

logistic_model_instance = LogisticModel(X_training=cleaning_instance.X_training,
                                        y_training=cleaning_instance.y_training,
                                        theta=cleaning_instance.theta,
                                        cost_function=cost,
                                        gradient=gradient)
logistic_model_instance.find_optimal_paramethers()

evaluation_instance = ModelEvaluation(
    cleaning_instance.X_test, cleaning_instance.y_test, logistic_model_instance.theta_min)
evaluation_instance.evaluate_binary_outcome()

# test_data_loader_instance = TestDataLoading()
# test_data_loader_instance.test_load_data_to_pd(
#     './data_science_engineering_blog_post/engineering_approach/data/exam_data.txt', None, ['Exam1', 'Exam2', 'Pass'])

# test_data_cleaner_instance = TestDataCleaning()
# test_data_cleaner_instance.test_create_training_test_split(
#     './data_science_engineering_blog_post/engineering_approach/data/exam_data.txt', None, ['Exam 1', 'Exam 2', 'Pass'], [(80, 3), (20, 3)])


# test_data_cleaner_instance.test_prepare_data_log_regression(
#     './data_science_engineering_blog_post/engineering_approach/data/exam_data.txt', None, ['Exam 1', 'Exam 2', 'Pass'], [(80, 3), (20, 3), (80, 1), (20, 1)])
