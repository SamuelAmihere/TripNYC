#!/usr/bin/env python
"""This module contains the ML class"""
import os
import mlflow
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import uuid
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import logging
from models.utils.config.msg_config import get_msg
# from models.utils.model_func import load_config, process_color_codes

# CONFIG_PATH = 'models/utils/config/config.json'
# config = process_color_codes(CONFIG_PATH)
# ERROR_COLOR = config.get('ERROR_COLOR')
# SUCCESS_COLOR = config.get('SUCCESS_COLOR')
# INFO_COLOR = config.get('INFO_COLOR')
# WARNING_COLOR = config.get('WARNING_COLOR')
# END_COLOR = config.get('END_COLOR')
# DEFAULT_COLOR = config.get('DEFAULT_COLOR')


class TrainingSetUp:
    def __init__(self, **kwargs):
        logging.info(get_msg('Initializing TrainingSetUp', 'INFO'))
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())
        
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = self.y_pred_proba = None
        self.pipeline = None
        logging.info(get_msg('TrainingSetUp initialized', 'SUCCESS'))

    def load_data(self, data_path):
        logging.info(get_msg(f'Loading data from {data_path}', 'INFO'))
        try:
            self.data = pd.read_csv(data_path)
            self.X = self.data.drop('target', axis=1)
            self.y = self.data['target']
            logging.info(get_msg(f"Data loaded. Shape: {self.data.shape}", 'SUCCESS'))
        except FileNotFoundError as e:
            logging.error(get_msg(f"Error loading data: {e}", 'ERROR'))
            raise FileNotFoundError(f"Error loading data: {e}")

    def split_data(self, test_size=0.2, random_state=42):
        logging.info(get_msg("Splitting data into train and test sets", "INFO"))
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        logging.info(get_msg(f"Data split. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}", "SUCCESS"))

    def create_pipeline(self):
        logging.info(get_msg("Creating pipeline", "INFO"))
        if hasattr(self, 'scalers') and hasattr(self, 'model'):
            self.pipeline = Pipeline([
                ('scaler', self.scalers),
                ('model', self.model)
            ])
            logging.info(get_msg("Pipeline created successfully"))
        else:
            logging.error(get_msg("Scalers and model must be defined", "ERROR"))
            raise AttributeError('scalers and model must be defined')

class TrainModel(TrainingSetUp):
    def __init__(self, **kwargs):
        logging.info(get_msg("Initializing TrainModel", "INFO"))
        super().__init__(**kwargs)
        self.mlflow_info = kwargs.get('mlflow', {})
        self.setup_mlflow()
        trainer = self.__dict__.get('id')
        logging.info(get_msg(f"Trainer [{trainer}] initialized", "SUCCESS"))

    def setup_mlflow(self):
        '''
        Sets up MLflow tracking server
        '''
        import GPUtil

        # TAGS: GPU
        gpus = GPUtil.getGPUs()
        try:
            gpu = gpus[0]
        except IndexError:
            gpu = None       
        if gpu:
            logging.info(get_msg(f"GPU found. Using [{gpu.name}]", "INFO"))
            gpuName = gpus.name
            memTotal = gpus.memoryTotal
            memUsed =  f'{gpu.memoryUtil * memTotal / 100}%'
            mlflow.set_tag('gpu.name', f'{gpuName}')
            mlflow.set_tag('gpu.memory_total', f'{memTotal}')
            mlflow.set_tag('gpu.memory_used', f'{memUsed}')
        else:
            mlflow.set_tag('cpu', 'True')
            # use CPU
            logging.info(get_msg("No GPU found. Using CPU", "WARNING"))

        logging.info(get_msg("Setting up MLflow", "INFO"))

        # TAGS: ENV
        env_tags = self.mlflow_info.get('env-tags')
        if env_tags:
            conda_env = env_tags.get('conda_env') # .yml file
            if conda_env:
                mlflow.set_env(conda_env)
            source = env_tags.get('source')
            if source:
                mlflow.set_source(source)
            user = env_tags.get('user')
            if user:
                mlflow.set_user(eval(user))
            os_name = env_tags.get('os')
            if os_name:
                mlflow.set_tag('os', eval(os_name))
            py_version = env_tags.get('python_version')
            if py_version:
                from sys import platform
                mlflow.set_tag('python_version', eval(py_version))
            mlflow_version = env_tags.get('mlflow_version')
            if mlflow_version:
                mlflow.set_tag('mlflow_version', eval(mlflow_version))

        # TAGS: MODEL
        model_tags = self.mlflow_info.get('model-tags')
        if model_tags:
            ml_model = model_tags.get('model')
            if ml_model:
                if ml_model == 'RandomForestClassifier':
                    from sklearn.ensemble import RandomForestClassifier
                if ml_model == 'RandomForestRegressor':
                    from sklearn.ensemble import RandomForestRegressor
                if ml_model == 'RandomForest':
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.ensemble import RandomForestRegressor
                if ml_model == 'SVC':
                    from sklearn.svm import SVC
                if ml_model == 'LogisticRegression':
                    from sklearn.linear_model import LogisticRegression
                if ml_model == 'LinearRegression':
                    from sklearn.linear_model import LinearRegression
                if ml_model == 'KNeighborsClassifier':
                    from sklearn.neighbors import KNeighborsClassifier
                if ml_model == 'KNeighborsRegressor':
                    from sklearn.neighbors import KNeighborsRegressor
                if ml_model == 'DecisionTreeClassifier':
                    from sklearn.tree import DecisionTreeClassifier
                if ml_model == 'DecisionTreeRegressor':
                    from sklearn.tree import DecisionTreeRegressor
                if ml_model == 'GradientBoostingClassifier':
                    from sklearn.ensemble import GradientBoostingClassifier
                if ml_model == 'GradientBoostingRegressor':
                    from sklearn.ensemble import GradientBoostingRegressor
                if ml_model == 'AdaBoostClassifier':
                    from sklearn.ensemble import AdaBoostClassifier
                if ml_model == 'AdaBoostRegressor':
                    from sklearn.ensemble import AdaBoostRegressor
                if ml_model == 'XGBClassifier':
                    from xgboost import XGBClassifier
                if ml_model == 'XGBRegressor':
                    from xgboost import XGBRegressor
                # initialize the specified model
                self.model = eval(ml_model)
                mlflow.set_tag('model_name', ml_model)
                model_source = model_tags.get('source')
                if model_source:
                    mlflow.set_tag('model_source', model_source)
                model_path = model_tags.get('model_path')
                if model_path:
                    mlflow.set_tag('model_path', model_path)
                model_size = model_tags.get('model_size')
                if model_size and model_path:
                    model_path = eval(model_path)
                    mlflow.set_tag('model_size', model_size)
            else:
                logging.error(get_msg("Model not defined", "ERROR"))
                raise AttributeError('Model not defined')
        else:
            logging.error(get_msg("Model tags not defined. \
                                  \nCheck models\utils\config\ml_config.json",
                                  "ERROR"))
            raise AttributeError('Model tags not defined')

        # TAGS: Experimament
        experiment_tags = self.mlflow_info.get('experiment-tags')
        if experiment_tags:
            exp_name = experiment_tags.get('experiment_name', f'mlflow_{self.id}')
            if exp_name:
                mlflow.set_tag('experiment_name', exp_name)
                mlflow.set_experiment(exp_name)
            exp_description = experiment_tags.get('experiment_description')
            if exp_description:
                mlflow.set_tag('experiment_description', exp_description)
            exp_id = experiment_tags.get('experiment_id')
        
        exp_name = self.mlflow_info.get('exp_name', f'mlflow_{self.id}')
        artifact_uri = self.mlflow_info.get('artifact_uri')
        model_path = self.mlflow_info\
            .get('model-tags')\
                .get('model_path')
        
        
        
        model_path = eval(model_path) if model_path else None
        os.makedirs("./mlruns", exist_ok=True)
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI_TRAIN', f'./mlruns/{exp_name}')
        mlflow.set_tracking_uri(tracking_uri)

        if artifact_uri:
            mlflow.set_artifact_uri(artifact_uri)
        logging.info(get_msg(f"MLflow setup complete. Experiment: {exp_name}", "SUCCESS"))

    def train(self):
        '''
        Trains the model
        '''
        import psutil
        import time

        logging.info(get_msg("Starting model training", "INFO"))
        with mlflow.start_run():
            # TAGS: CPU
            while True:
                memUtil = psutil.virtual_memory().percent
                mlflow.log_metric('memory_utilization', memUtil, step=0)
                time.sleep(10)
                if memUtil > 90:
                    logging.warning(get_msg(f"Memory usage is {memUtil}%. picking cpu", "WARNING"))
                    mlflow.set_tag('cpu', 'True')
                    break
            
            # Create pipeline for training
            self.create_pipeline()

            if not all(hasattr(self, attr) for attr in ['param_grid', 'cv', 'scoring']):
                logging.error(get_msg("param_grid, cv, and scoring must be defined", "ERROR"))
                raise AttributeError('param_grid, cv, and scoring must be defined')

            logging.info(get_msg("Starting GridSearchCV", "INFO"))
            grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=self.cv, scoring=self.scoring)
            if not hasattr(self, 'n_jobs'):
                logging.warning(get_msg("n_jobs not defined. Defaulting to 1", "WARNING"))
                self.n_jobs = 1
            if self.X_train is None or self.X_test is None:
                logging.error(get_msg("Data has not been loaded or split yet", "ERROR"))
            else:
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
                logging.info(get_msg(f"GridSearchCV completed. Best score: {grid_search.best_score_}", "SUCCESS"))

                logging.info(get_msg(f"Logging best parameters and score to MLflow", "INFO"))
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_score", grid_search.best_score_)
                mlflow.sklearn.log_model(self.model, "model")

                logging.info(get_msg("Making predictions on test set", "INFO"))
                self.y_pred = self.model.predict(self.X_test)
                self.y_pred_proba = self.model.predict_proba(self.X_test)

                self.log_metrics()
        # check if model is trained
        if hasattr(self, 'model') and self.model is not None:
            logging.error(get_msg("Model has not been trained", "ERROR"))
            raise Exception("Model has not been trained")
        logging.info(get_msg("Model training completed", "SUCCESS"))

    def log_metrics(self):
        logging.info(get_msg("Calculating and logging additional metrics", "INFO"))
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        logging.info(get_msg(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}", "SUCCESS"))

    def evaluate(self):
        logging.info(get_msg("Starting model evaluation", "INFO"))
        # TO DO: Add evaluation logic
        logging.info(get_msg("Model evaluation completed", "SUCCESS"))

    def save_model(self, tolocal=False, toext=False):

        if tolocal:
            ext_path = os.getenv('MODEL_EXT_PATH')
        if toext:
            local_path = os.getenv('MODEL_LOCAL_PATH')
        if (not tolocal) and (not toext):
            path = os.path.join(os.getcwd(), 'models', f'model_{self.id}')
            logging.info(get_msg(f"Defaulting path to {path}", "WARNING"))

        for path in [local_path, ext_path, path]:
            if not path:
                continue
            logging.info(get_msg(f"Saving model to {path}", "INFO"))
            if not hasattr(self, 'model'):
                logging.error(get_msg("Model has not been trained yet", "ERROR"))
                raise Exception("Model has not been trained yet")
            mlflow.sklearn.save_model(self.model, path)
            logging.info(get_msg("Model saved successfully", "SUCCESS"))

    def load_model(self, path):
        logging.info(get_msg(f"Loading model from {path}", "INFO"))
        self.model = mlflow.sklearn.load_model(path)
        logging.info(get_msg("Model loaded successfully", "SUCCESS"))