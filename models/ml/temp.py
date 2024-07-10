
# import mlflow
# import psutil
# import time
# import threading

# def log_memory_utilization():
#     while True:
#         memory_utilization = psutil.virtual_memory().percent
#         mlflow.log_metric("memory_utilization", memory_utilization, step=0)
#         time.sleep(10)

# # Start the MLflow run
# with mlflow.start_run():
#     # Start a separate thread to log memory utilization
#     memory_thread = threading.Thread(target=log_memory_utilization)
#     memory_thread.daemon = True  # Allow the thread to exit when the main thread finishes
#     memory_thread.start()







# mlflow_client = mlflow.tracking.MlflowClient()

# class MyModel:
#     def __init__(self, mlflow_client):
#         self.lock = threading.Lock()
#         self.mlflow_client = mlflow_client

#     def train(self):
#         with self.lock:
#             with mlflow.start_run():
#                 # Your training code here

#     def log_memory_utilization(self, logging):
#         '''Logs memory utilization to MLflow'''
#         while True:
#             memory_utilization = psutil.virtual_memory().percent
#             self.mlflow_client.log_metric("memory_utilization", memory_utilization, step=0)
#             time.sleep(10)
#             if memory_utilization > 90:
#                 logging.warning(get_msg(f"Memory usage is {memory_utilization}%. Picking CPU", "WARNING"))
#                 self.mlflow_client.set_tag('cpu', 'True')

# if __name__ == "__main__":
#     model = MyModel(mlflow_client)

#     # Start a separate thread to log memory utilization
#     memory_thread = threading.Thread(target=model.log_memory_utilization, args=(mlflow.log,))
#     memory_thread.daemon = True  # Allow the thread to exit when the main thread finishes
#     memory_thread.start()
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mlflow import log_metric

def log_metrics(metrics, **kwargs):
    for metric, value in metrics.items():
        log_metric(metric, value)

grid_search = GridSearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
    cv=5,
    refit='accuracy',
    scoring_kwargs={'metrics': log_metrics}
)


[10/07, 9:10 pm] Meta AI: To log metrics with MLflow during GridSearchCV training, you can use the `mlflow.log_metrics` function within the `scoring` parameter of GridSearchCV. Here's an example:
```
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mlflow import log_metric

def log_metrics(metrics, **kwargs):
    for metric, value in metrics.items():
        log_metric(metric, value)

grid_search = GridSearchCV(
    estimator=your_estimator,
    param_grid=your_param_grid,
    scoring={'accuracy': 'accuracy', 'f1': 'f1_macro'},
    cv=5,
    refit='accuracy',
    scoring_kwargs={'metrics': log_metrics}
)
```
In this example, the `log_metrics` function is defined to log the metrics using MLflow's `log_metric` function. The `scoring_kwargs` parameter of GridSearchCV is used to pass the `log_metrics` function to the `scoring` parameter.

During GridSearchCV training, the `log_metrics` function will be called for each iteration, logging the metrics to MLflow. You can then view the logged metrics in the MLflow UI or using the `mlflow ui` command.

Note that you need to have the MLflow autologging enabled (using `mlflow.autolog()`) for this to work.

Also, make sure to replace `your_estimator` and `your_param_grid` with your actual estimator and parameter grid.
[10/07, 9:23 pm] Meta AI: The logging of metrics using the `log_metrics` function within GridSearchCV will not be in real-time, as GridSearchCV performs training and evaluation in batches.

However, you can achieve real-time logging by using MLflow's `autolog` feature with the `interval` parameter set to a small value (e.g., 1 second). This will log metrics at regular intervals during training.

Here's an example:
```
mlflow.autolog(interval=1)
```
This will log metrics every 1 second during training. You can adjust the `interval` parameter to control the logging frequency.

Additionally, you can use MLflow's `log_metric` function with the `step` parameter to log metrics at specific training steps. For example:
```
log_metric("accuracy", accuracy, step=10)
```
This will log the accuracy metric at training step 10.

By combining these approaches, you can achieve real-time logging of metrics during GridSearchCV training.

Note that real-time logging may incur additional overhead and impact training performance. Be sure to adjust the logging frequency and interval according to your specific needs.
[10/07, 9:30 pm] Meta AI: I apologize for the confusion. You are correct, I previously defined `log_metrics` (with an 's') for logging multiple metrics, but now I'm defining `log_metric` (without an 's') for logging a single metric.

To log the accuracy metric at a specific training step, you can define `log_metric` within the `fit` method of your estimator. Here's a full example:
```
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from mlflow import log_metric, autolog

class MyEstimator(BaseEstimator):
    def fit(self, X, y):
        # Your training code here
        # ...
        
        # Log accuracy at training step 10
        accuracy = accuracy_score(y, your_predictions)
        log_metric("accuracy", accuracy, step=10)

grid_search = GridSearchCV(
    estimator=MyEstimator(),
    param_grid=your_param_grid,
    cv=5,
    refit='accuracy'
)

# Enable autologging
autolog()

# Perform GridSearchCV
grid_search.fit(your_X, your_y)


Here's the complete example, including the start of the MLflow run, GridSearchCV, and logging of metrics:
```



import logging
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# Configure root logger to display logs of all levels to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def get_msg(message, level):
    return f"{level}: {message}"

# Example function to demonstrate logging with MLflow and to the console
def train_and_log_metrics(self):
    if not hasattr(self, 'n_jobs'):
        logger.warning(get_msg("n_jobs not defined. Defaulting to 1", "WARNING"))
        self.n_jobs = 1
    if self.X_train is None or self.X_test is None:
        logger.error(get_msg("Data has not been loaded or split yet", "ERROR"))
    else:
        grid_search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, n_jobs=self.n_jobs)
        grid_search.fit(self.X_train, self.y_train)

        # Log accuracy at training step 10
        accuracy = accuracy_score(self.y_train, grid_search.predict(self.X_train))
        mlflow.log_metric("accuracy", accuracy, step=10)
        logger.info(f"Step 10 - Accuracy: {accuracy}")  # Log accuracy

        # Log additional metrics and parameters
        precision = precision_score(self.y_train, grid_search.predict(self.X_train))
        mlflow.log_metric("precision", precision)
        logger.info(f"Precision: {precision}")  # Log precision

        recall = recall_score(self.y_train, grid_search.predict(self.X_train))
        mlflow.log_metric("recall", recall)
        logger.info(f"Recall: {recall}")  # Log recall

        param1 = grid_search.best_params_['model__param1']
        mlflow.log_param("param1", param1)
        logger.info(f"Param1: {param1}")  # Log param1

        param2 = grid_search.best_params_['model__param2']
        mlflow.log_param("param2", param2)
        logger.info(f"Param2: {param2}")  # Log param2