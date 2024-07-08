#!/usr/bin/env python
"""This module contains helper functions for the models."""
import time
from models.utils.config.msg_config import get_msg
import psutil

def log_memory_utilization(mlflow, logging, step=0, sleep_time=10, target_gpu=95):
    '''
    Logs memory utilization to MLflow
    :param mlflow: MLflow object
    :param logging: logging object
    :param step: int
    :param sleep_time: int
    :param target_gpu: int
    '''
    while True:
        memory_utilization = psutil.virtual_memory().percent
        mlflow.log_metric("memory_utilization", memory_utilization, step=step)
        time.sleep(sleep_time)

        if memory_utilization > target_gpu:
            logging.warning(get_msg(f"Memory usage is {memory_utilization}%. picking cpu",
                                    "WARNING"))
            mlflow.set_tag('cpu', 'True')

            # stop the thread
            break


def update_value(value, new_value):
    '''
    Updates the value of a variable
    :param value: variable
    :param new_value: variable
    :return: variable
    '''
    value = new_value
    return value