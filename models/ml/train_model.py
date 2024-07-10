import sys
import os
from pathlib import Path
import argparse
import GPUtil
import mlflow
# sys.path.append("..")
import logging
from models.utils.model_func import load_config
from models.utils.config.msg_config import get_msg
from models.ml.model import TrainModel
import sklearn


CONFIG_PATH = 'models/utils/config/ml_config.json'
DATA_PATH = '../TripNYC-resources/Data/yellow/2024/yellow_tripdata_2024-01.csv'
# Load configuration
FARE_AMOUNT = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']
config_FA = FARE_AMOUNT # ['mlflow']
config_DATA_PTH = FARE_AMOUNT["DATA_PATH"]
YELLO_TAXI_DATA = config_DATA_PTH["yellow"]["path"]

YELLO_TAXI_DATA_DESC_PATH = config_DATA_PTH["yellow"]["description"]
YELLO_TAXI_DATA_DESC = load_config(YELLO_TAXI_DATA_DESC_PATH)

experiment_tags = config_FA.get('experiment_tags')


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
             
            tracking_uri = eval(tracking_uri)
            
            
            try:
                if 'local' == runs.get('run_origin', {}):
                    # Get the current file's directory
                    current_dir = Path.cwd()
                    parent_root = current_dir.parent
                    
                    # base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                    models_dir = tracking_uri.split(':///')[-1]
                    tracking_uri = f"{parent_root}/{models_dir}"
                    
                    os.makedirs(models_dir, exist_ok=True)


                tracking_uri = "file:///E:/amihere/programmingLessons/ALX-Backend/FINAL PROJECT/TripNYC/models/ml/models"

                mlflow.set_tracking_uri(f"{tracking_uri}")
                print(f"========tracking_uri: {tracking_uri} set=======")
            except Exception as e:
                logging.error(get_msg(f"Error setting tracking URI: {e}", 'ERROR'))

    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        exp_id = exp.experiment_id
        logging.info(get_msg(f"Resuming experiment: {exp_name} with ID: {exp_id}", 'INFO'))
    else:
        exp_id = mlflow.create_experiment(exp_name)
        # Set the experiment
        mlflow.set_experiment(exp_name)
        if exp_id:
            print(f"========Created new experiment with id : {exp_id}=======")
            
            exp_description = experiment_tags.get('experiment_description')
            if exp_description:
                try:
                    mlflow.set_tag('experiment_description', exp_description)
                except Exception as e:
                    logging.error(get_msg(f"Error setting experiment description: {e}", 'ERROR'))   
            
            # set experiment id
            try:
                mlflow.set_experiment(exp_id)
            except Exception as e:
                logging.error(get_msg(f"Error setting experiment ID: {e}", 'ERROR'))
            logging.info(get_msg(f"Experiment created with ID: {exp_id}", 'INFO'))

            print(f"-------Experiment ID: {exp_id}")
        
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
                try:
                    mlflow.set_tag('gpu.name', f'{gpuName}')
                    mlflow.set_tag('gpu.memory_total', f'{memTotal}')
                    mlflow.set_tag('gpu.memory_used', f'{memUsed}')
                except Exception as e:
                    logging.error(get_msg(f"Error setting GPU tags: {e}", 'ERROR'))
            else:
                processor = {"cpu": True}
                try:
                    mlflow.set_tag('cpu', 'True')
                    # use CPU
                    logging.info(get_msg("No GPU found. Using CPU", "WARNING"))
                except Exception as e:
                    logging.error(get_msg(f"Error setting CPU tags: {e}", 'ERROR'))
            
        else:
            print(f"<<<<< Experiment not created {create_experiment}")
        # runs:
        if runs:
            run_origin = runs.get('origin')
            if run_origin:
                try:
                    mlflow.set_tag('run_origin', run_origin)
                except Exception as e:
                    logging.error(get_msg(f"Error setting run origin: {e}", 'ERROR'))
    
        # Enable autologging
        mlflow.autolog(
            log_model_signatures=True,
        )
        mlflow.sklearn.autolog()

        model_exp = config_FA['mlflow']['experiment_tags']
        config_FA['mlflow']['experiment_tags'] = {
                            **model_exp,
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

    create_experiment()
    if EXP_ID:
        config_FA['mlflow']["experiment_tags"]["experiment_id"] = EXP_ID
    config_FA['mlflow']["data_tags"]["data_path"] = YELLO_TAXI_DATA
    config_FA['mlflow']["data_tags"]["description"] = YELLO_TAXI_DATA_DESC

    print(config_FA.keys())
    # Initialize the model trainer
    trainer_FA = TrainModel(**config_FA)
    
    exp_id = trainer_FA.__dict__.get('mlflow_info').get('experiment_tags').get('experiment_id')
    # print(f"=======Starting RUN===========[{exp_id}]=============")

    artifact_path = mlflow.get_artifact_uri()
    logging.info(get_msg(f"Artifact path: {artifact_path}", "INFO"))

    logging.info(get_msg(f"Trainer created", "INFO"))


    try:
        trainer_FA.load_data()
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