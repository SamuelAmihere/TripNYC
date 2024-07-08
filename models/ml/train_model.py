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

# Initialize MLflow client for logging: GPU memory utilization
mlflow_client_gpu = mlflow.tracking.MlflowClient()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def take_args():
    """Parse command line arguments. Use while loop
    to keep asking for the arguments until the user
    provides them.
    :return: args
    """
    while True:
        try:
            parser = argparse.ArgumentParser(description='Train the model')
            parser.add_argument('--run_name', type=str, help='Name of the run ()')
            args = parser.parse_args()
            if args.run_name:
                return args
        except Exception as e:
            logging.error(get_msg(f"Error parsing command line arguments: {e}", 'ERROR'))


def main():
    # Load configuration
    config_FA = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']
    args = take_args()
    print(args)
    return
    config_FA.update({**config_FA, **{'run_name': args.run_name}})
    

    # Initialize the model trainer
    trainer_FA = TrainModel(**config_FA)

    logging.info(get_msg(f"[Model]: {trainer_FA.model}, [Scalers]: {trainer_FA.scalers}", "INFO"))

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