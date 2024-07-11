import sys
import os
from pathlib import Path
import argparse
import GPUtil
import mlflow
# sys.path.append("..")
import logging
from models.ml.ml import ML
from models.utils.model_func import load_config
from models.utils.config.msg_config import get_msg
from models.ml.model import ModelTrainer
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeClassifier
# logisitic regression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler



CONFIG_PATH = 'models/utils/config/ml_config.json'
DATA_PATH = '../TripNYC-resources/Data/yellow/2024/yellow_tripdata_2024-01.csv'
# Load configuration
FARE_AMOUNT = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']
config_FA = FARE_AMOUNT # ['mlflow']
config_DATA_PTH = FARE_AMOUNT["DATA_PATH"]
YELLO_TAXI_DATA = config_DATA_PTH["yellow"]["path"]

YELLO_TAXI_DATA_DESC_PATH = config_DATA_PTH["yellow"]["description"]
YELLO_TAXI_DATA_DESC = load_config(YELLO_TAXI_DATA_DESC_PATH)
print(YELLO_TAXI_DATA_DESC)

experiment_tags = config_FA['mlflow'].get('experiment_tags')
model_tags = config_FA['mlflow'].get('model_tags')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def take_args():
    """Parse command line arguments. Use while loop
    to keep asking for the arguments until the user
    provides them.
    :return: args
    """
    try:
        parser = argparse.ArgumentParser(description='Train the model')
        parser.add_argument('--run_name', type=str, help='Name of the run ()')
        parser.add_argument('--run_id', type=str, help='Run ID')
        parser.add_argument('--exp_id', type=str, help='Experiment ID')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment')
        args = parser.parse_args()
        return args
    except Exception as e:
        logging.error(get_msg(f"Error parsing command line arguments: {e}", 'ERROR'))



def main():

    model_namne = model_tags.get('model_name')
    pipeline = experiment_tags.get('model_experiment').get('pipeline')

    PIPELINE = []
    
    # IN_PIPELINE = lambda x: x in [k for k,v in PIPELINE]
    for pipe in pipeline:
        obj = eval(pipeline.get(pipe))
        if pipe == 'scaler':
            if len(PIPELINE) > 0:
                PIPELINE.insert(0, (pipe, obj))
            else:
                PIPELINE.append((pipe, obj))
        else:
            if pipe == 'model':
                # change the model name
                PIPELINE.append((model_namne, obj))
            else:
                PIPELINE.append((pipe, obj))

    trainer = ModelTrainer(
        name=experiment_tags.get('experiment_name'),
        data_path=YELLO_TAXI_DATA,
        description=YELLO_TAXI_DATA_DESC,
        experiment_id=experiment_tags.get('experiment_id'),
        experiment_tags=experiment_tags,
        pipeline=PIPELINE,
        param_grid=experiment_tags.get('model_experiment').get('param_grid'),
        settings=experiment_tags.get('model_experiment').get('settings'),
    )
    # Train the model
    trainer.train()

    # Save trainer
    trainer.save()
  


if __name__ == "__main__":

    logging.info(get_msg("Starting the training process", "INFO"))

    main()