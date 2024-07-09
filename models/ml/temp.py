
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
