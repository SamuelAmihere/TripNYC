import sys
import os
import argparse
import GPUtil
import mlflow
# sys.path.append("..")
import logging
from models.utils.model_func import load_config
from models.utils.config.msg_config import get_msg
from models.ml.model import TrainModel


CONFIG_PATH = 'models/utils/config/ml_config.json'
DATA_PATH = '../TripNYC-resources/Data/yellow/2024/yellow_tripdata_2024-01.csv'
# Load configuration
FARE_AMOUNT = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']
config_FA = FARE_AMOUNT # ['mlflow']
config_DATA_PTH = FARE_AMOUNT["DATA_PATH"]
YELLO_TAXI_DATA = config_DATA_PTH["yellow"]["path"]
experiment_tags = config_FA.get('experiment_tags')

# Initialize MLflow client for logging: GPU memory utilization
mlflow_client_gpu = mlflow.tracking.MlflowClient()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def create_experiment(exp_name=None) -> None:
    """Create an experiment in MLflow
    :param EXP_ID: Experiment ID
    """
    if not exp_name:
        exp_name = config_FA['mlflow']["experiment_tags"].get('experiment_name')
    logging.info(get_msg(f"Creating experiment: {exp_name}", 'INFO'))
    

    #  =============EXPERIMENT================
    # TAGS: Experimament
    experiment_tags = config_FA['mlflow'].get('experiment_tags', {})
    runs = experiment_tags.get('run', {})
    if experiment_tags:
        tracking_uri = experiment_tags.get('tracking_uri')
        if tracking_uri:
            try:
                if 'local' in runs.get('run'):
                    os.makedirs(tracking_uri, exist_ok=True)
                mlflow.set_tracking_uri(tracking_uri)
            except Exception as e:
                logging.error(get_msg(f"Error setting tracking URI: {e}", 'ERROR'))
        # runs:
        if runs:
            run_origin = runs.get('origin')
            if run_origin:
                mlflow.set_tag('run_origin', run_origin)

    
    
    
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        exp_id = exp.experiment_id
    else:
        exp_description = experiment_tags.get('experiment_description')
        if exp_description:
            mlflow.set_tag('experiment_description', exp_description)
        exp_id = mlflow.create_experiment(exp_name)
    
    logging.info(get_msg(f"Experiment created with ID: {exp_id}", 'INFO'))

    
    # TAGS: GPU
    gpus = GPUtil.getGPUs()
    try:
        gpu = gpus[0]
    except IndexError:
        gpu = None       
    if gpu:
        logging.info(get_msg(f"GPU found. Using [{gpu.name}]", "INFO"))
        gpuName = gpu.name
        memTotal = gpu.memoryTotal
        memUsed =  f'{gpu.memoryUtil * memTotal / 100}%'
        processor = {'gpu': gpuName,
                     'memory_total': memTotal,
                     'memory_used': memUsed}
        mlflow.set_tag('gpu.name', f'{gpuName}')
        mlflow.set_tag('gpu.memory_total', f'{memTotal}')
        mlflow.set_tag('gpu.memory_used', f'{memUsed}')
    else:
        processor = {"cpu": True}
        mlflow.set_tag('cpu', 'True')
        # use CPU
        logging.info(get_msg("No GPU found. Using CPU", "WARNING"))
    
    
    config_FA['mlflow'] = {**config_FA['mlflow'],
                           'experiment_id': exp_id,
                           'run_origin': run_origin,
                           'tracking_uri': tracking_uri,
                           'processor': processor
                        }

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

    if not EXP_ID:
        create_experiment()

    config_FA['mlflow']["experiment_tags"]["experiment_id"] = EXP_ID
    config_FA['mlflow']["data-tags"]["data_path"] = YELLO_TAXI_DATA

    # Initialize the model trainer
    trainer_FA = TrainModel(**config_FA)
    
    print(trainer_FA.__dict__.keys())
    return

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