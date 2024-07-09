import sys
import os
import argparse
import mlflow
# sys.path.append("..")
import logging
from models.utils.model_func import load_config
from models.utils.config.msg_config import get_msg
from models.ml.model import TrainModel

CONFIG_PATH = 'models/utils/config/ml_config.json'
DATA_PATH = '../TripNYC-resources/Data/yellow/2024/yellow_tripdata_2024-01.csv'
# Load configuration
config_FA = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']['mlflow']
config_DATA_PTH = load_config(CONFIG_PATH)['DATA_PATH']
PARENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BASE_P = config_DATA_PTH['BASE_PATH']
experiment_tags = config_FA.get('experiment-tags')
EXP_NAME = experiment_tags['experiment_name']

# Initialize MLflow client for logging: GPU memory utilization
mlflow_client_gpu = mlflow.tracking.MlflowClient()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def create_experiment(exp_name=None):
    """Create an experiment in MLflow
    :param EXP_ID: Experiment ID
    """
    logging.info(get_msg(f"Creating experiment: {exp_name}", 'INFO'))
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        return exp.experiment_id
    exp_id = mlflow.create_experiment(exp_name)
    logging.info(get_msg(f"Experiment created with ID: {exp_id}", 'INFO'))
    return exp_id

def take_args():
    """Parse command line arguments. Use while loop
    to keep asking for the arguments until the user
    provides them.
    :return: args
    """
    try:
        parser = argparse.ArgumentParser(description='Train the model')
        parser.add_argument('--run_name', type=str, help='Name of the run ()')
        parser.add_argument('--exp_id', type=str, help='Experiment ID')
        args = parser.parse_args()
        return args
    except Exception as e:
        logging.error(get_msg(f"Error parsing command line arguments: {e}", 'ERROR'))

def main():
    args = take_args()
    EXP_ID = args.exp_id
    config_DATA_yellow = config_DATA_PTH['yellow']
    
    config_DATA_yellow_2024 = config_DATA_yellow['2024']["01"]
    pth = f"{BASE_P}/yellow/2024/{config_DATA_yellow_2024}"

    if not EXP_ID:
        EXP_ID = create_experiment(EXP_NAME)
   
    config_FA.update({**config_FA, **{'exp_id': EXP_ID}, **{"data_path": pth}})

    # Initialize the model trainer
    trainer_FA = TrainModel(**config_FA)
    print(trainer_FA.__dict__.get("data_path"))
    # get mlfow artifact_path


    artifact_path = mlflow.get_artifact_uri()
    logging.info(get_msg(f"Artifact path: {artifact_path}", "INFO"))

    logging.info(get_msg(f"Trainer created", "INFO"))

    try:
        trainer_FA.load_data(trainer_FA.data_path)
        trainer_FA.split_data()
    except FileNotFoundError as e:
        logging.error(get_msg(f"Error loading data: {e}", 'ERROR'))


    try:
        trainer_FA.train()
        trainer_FA.evaluate()
        trainer_FA.save_model('path/to/save/model')
    except AttributeError as e:
        logging.error(get_msg(f"Error training model: {e}", 'ERROR'))
    logging.info(get_msg("Training process completed", "SUCCESS"))


if __name__ == "__main__":

    logging.info(get_msg("Starting the training process", "INFO"))

    main()