import sys
sys.path.append("..")
import logging
from models.utils.model_func import load_config

CONFIG_PATH = 'models/utils/config/ml_config.json'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Usage
if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from models.ml.model import TrainModel

    logging.info("Starting the training process")

    # Load configuration
    
    config = load_config(CONFIG_PATH)['MODEL1']
    config.update({k: eval(v) if (k == 'scalers' or k=='model') \
                   else v  for k, v in config.items()})

    print(config)
    trainer = TrainModel(**config)
    
    try:
        trainer.load_data(trainer.data_path)
        trainer.split_data()
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e}")

    try:
        trainer.train()
        trainer.evaluate()
        trainer.save_model('path/to/save/model')
    except AttributeError as e:
        logging.error(f"Error training model: {e}")
    logging.info("Training process completed")