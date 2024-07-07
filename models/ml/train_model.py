import sys
sys.path.append("..")
import logging
from models.utils.model_func import load_config
from models.utils.config.msg_config import get_msg
CONFIG_PATH = 'models/utils/config/ml_config.json'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Usage
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from models.ml.model import TrainModel

    logging.info(get_msg("Starting the training process", "INFO"))

    # Load configuration
    
    config = load_config(CONFIG_PATH)['FARE_AMOUNT-MODEL1']
    config.update({k: eval(v) if (k == 'scalers' or k=='model') \
                   else v  for k, v in config.items()})

    trainer = TrainModel(**config)
    logging.info(get_msg(f"[Model]: {trainer.model}, [Scalers]: {trainer.scalers}", "INFO"))
    
    try:
        trainer.load_data(trainer.data_path)
        trainer.split_data()
    except FileNotFoundError as e:
        logging.error(get_msg(f"Error loading data: {e}", 'ERROR'))

    try:
        trainer.train()
        trainer.evaluate()
        trainer.save_model('path/to/save/model')
    except AttributeError as e:
        logging.error(get_msg(f"Error training model: {e}", 'ERROR'))
    logging.info(get_msg("Training process completed", "SUCCESS"))