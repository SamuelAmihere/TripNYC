#!/usr/bin/env python
"""This module contains the ML class"""
import os
import sys
import platform
import threading

import mlflow
from models.utils.config.mlflow_conf import log_memory_utilization

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import pandas as pd
from datetime import datetime
import uuid
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import logging
from models.utils.config.msg_config import get_msg
 
import psutil
import time


class TrainingSetUp:
    def __init__(self, **kwargs):
        logging.info(get_msg('Initializing TrainingSetUp', 'INFO'))
        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.created_at = kwargs.get('created_at', datetime.now())
        self.updated_at = kwargs.get('updated_at', datetime.now())

        # remove id, created_at, and updated_at from kwargs.
        kwargs.pop('id', None)
        kwargs.pop('created_at', None)
        kwargs.pop('updated_at', None)

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = self.y_pred_proba = None
        self.pipeline = None
        self.param_grid = None
        self.settings = None
        self.run_id = None
        self.tracking_uri = None
        
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
        '''
        Creates a pipeline for training
        '''
        if hasattr(self, 'pipeline'):
            logging.info(get_msg("Pipeline ready for creation", "INFO"))

            self.pipeline = make_pipeline([(k, v) for k, v in self.pipeline.items() if v is not None])
            if self.pipeline:
                logging.info(get_msg("Pipeline created successfully", "SUCCESS"))
            else:
                logging.error(get_msg("Pipeline not created", "ERROR"))
                raise AttributeError('Pipeline not created')
    def create_params(self):
        '''
        Creates parameters for training
        '''
        if hasattr(self, 'param_grid'):
            logging.info(get_msg("Parameters ready for creation", "INFO"))
            self.param_grid = {k: v for k, v in self.param_grid.items() if v is not None}
            if self.param_grid:
                logging.info(get_msg("Parameters created successfully", "SUCCESS"))
            else:
                logging.error(get_msg("Parameters not created", "ERROR"))
                raise AttributeError('Parameters not created')

class TrainModel(TrainingSetUp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info(get_msg("Initializing TrainModel", "INFO"))
        self.mlflow_info = kwargs.get('mlflow', {})
        self.data_info = kwargs.get('data', {})
        if self.mlflow_info:
            self.setup_mlflow()
    
        trainer = self.__dict__.get('id')
        logging.info(get_msg(f"Trainer [{trainer}] initialized", "SUCCESS"))

    def setup_mlflow(self):
        '''
        Sets up MLflow tracking server
        '''
        import GPUtil

        # TAGS: DATA
        data_tags = self.data_info.get('data-tags')

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
            os_name = env_tags.get('os')
            if os_name:
                mlflow.set_tag('os', eval(os_name))
            py_version = env_tags.get('python_version')
            if py_version:
                mlflow.set_tag('python_version', eval(py_version))
            mlflow_version = env_tags.get('mlflow_version')
            if mlflow_version:
                mlflow.set_tag('mlflow_version', eval(mlflow_version))
            train_env = env_tags.get('train_env')
            if train_env:
                mlflow.set_tag('train_env', eval(train_env))

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
                artifact_path = model_tags.get(artifact_path)
                if artifact_path:
                    mlflow.set_tag('model_path', eval(model_path))
                    self.model_path = eval(model_path)
                model_size = model_tags.get('model_size')
                if model_size and model_path:
                    model_path = eval(model_path)
                    mlflow.set_tag('model_size', model_size)
                
            else:
                logging.error(get_msg("Model not defined", "ERROR"))
                raise AttributeError('Model not defined')
        else:
            logging.error(get_msg("Model tags not defined. \
                                  \nCheck models/utils/config/ml_config.json",
                                  "ERROR"))
            raise AttributeError('Model tags not defined')

        #  =============EXPERIMENT================
        # TAGS: Experimament
        experiment_tags = self.mlflow_info.get('experiment-tags')
        if experiment_tags:
            # exp_name = experiment_tags.get('experiment_name')
            # if exp_name:
            #     mlflow.set_experiment(exp_name)
            # experiment_id = experiment_tags.get('experiment_id')
            # if experiment_id:
            #     mlflow.set_tag('experiment_id', eval(experiment_id))
            # exp_description = experiment_tags.get('experiment_description')
            # if exp_description:
            #     mlflow.set_tag('experiment_description', exp_description)
            tracking_uri = experiment_tags.get('tracking_uri')
            if tracking_uri:
                if 'local' in mlflow.get_tags():
                    os.makedirs(tracking_uri, exist_ok=True)
                mlflow.set_tracking_uri(tracking_uri)
                self.tracking_uri = tracking_uri

            # runs:
            runs = experiment_tags.get('run')
            if runs:
                run_origin = runs.get('origin')
                if run_origin:
                    mlflow.set_tag('run_origin', run_origin)
                run_name = runs.get('name')
                if run_name and self.get('run_name'):
                    try:
                        mlflow.set_tag('run_name', eval(run_name))
                    except NameError:
                        raise NameError(f"NameError: {run_name.split('.')[-1]} is not defined")
                run_id = runs.get('id')
                if run_id:
                     run_id = eval(run_id)


            # =====model-experiment=====:
            model_experiment = experiment_tags.get('model-experiment')
            if model_experiment:
                # << PIPELINE >>
                PIPELINE = {}
                pipeline = model_experiment.get('pipeline')
                if pipeline:
                    if self.model:
                        PIPELINE['model'] = self.model
                    else:
                        if'model' in pipeline:
                            PIPELINE['model'] = eval(pipeline.get('model'))
                        else:
                            logging.error(get_msg("Model not defined", "ERROR"))
                            raise AttributeError('Model not defined')
                    if 'scaler' in pipeline:
                        PIPELINE['scalers'] = eval(pipeline.get('scaler'))
                    if 'preprocessor' in pipeline:
                        PIPELINE['preprocessor'] = eval(pipeline.get('preprocessor'))
                    if 'feature_selection' in pipeline:
                        PIPELINE['feature_selection'] = eval(pipeline.get('feature_selection'))
                msg = "Setting default values {} for pipeline".format(
                    ', '.join([f"{k}: {v}" for k, v in PIPELINE.items()]))
                logging.info(get_msg(msg, "INFO"))
                self.pipeline = PIPELINE
                #-----------------------------------------

                # << PARAM_GRID >>
                p_grid = {}
                param_grid = model_experiment.get('param_grid')
                if param_grid:
                    model__n_estimators = param_grid.get('model__n_estimators')
                    if model__n_estimators:
                        p_grid['model__n_estimators'] = model__n_estimators
                    model__max_depth = param_grid.get('model__max_depth')
                    if model__max_depth:
                        p_grid['model__max_depth'] = model__max_depth
                msg = "Setting default values {} for param_grid".format(
                    ', '.join([f"{k}: {v}" for k, v in p_grid.items()]))
                logging.info(get_msg(msg, "INFO"))
                self.param_grid = p_grid
                #-----------------------------------------

                # << SETTINGS: Model Training >>
                sett = {}
                settings = model_experiment.get('settings')
                if settings:
                    if 'n_jobs' in settings:
                        sett['n_jobs'] = settings.get('n_jobs')
                    if 'cv' in settings:
                        sett['cv'] = settings.get('cv')
                    if 'scoring' in settings:
                        sett['scoring'] = settings.get('scoring')
                msg = "Setting default values {} for settings".format(
                    ', '.join([f"{k}: {v}" for k, v in sett.items()]))
                #-----------------------------------------
                logging.info(get_msg(msg, "INFO"))
                self.settings = sett
            #============================================

        logging.info(get_msg(f"MLflow setup complete. Experiment: {self.exp_name}", "SUCCESS"))

    def train(self):
        '''
        Trains the model
        '''
            # |-------------------------------------------------------------------------|
            # |------------------|        Start MLflow run              |---------------|
            # |-------------------------------------------------------------------------|
        with mlflow.start_run(experiment_id="{}".format( self.exp_id), 
                              run_name="{}".self.experiment_id) as run:

            # Create pipeline for training
            self.create_pipeline()
            self.create_params()
            self.run_id = run.info.run_id

            if not all(hasattr(self, attr) for attr in ['param_grid', 'cv', 'scoring']):
                logging.error(get_msg("param_grid, cv, and scoring must be defined",
                                    "ERROR"))
                raise AttributeError('param_grid, cv, and scoring must be defined')

            logging.info(get_msg("Starting GridSearchCV", "INFO"))
            grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=self.cv,
                                    scoring=self.scoring)
            if not hasattr(self, 'n_jobs'):
                logging.warning(get_msg("n_jobs not defined. Defaulting to 1", "WARNING"))
                self.n_jobs = 1
            if self.X_train is None or self.X_test is None:
                logging.error(get_msg("Data has not been loaded or split yet", "ERROR"))
            else:
                grid_search.fit(self.X_train, self.y_train)
                self.model = grid_search.best_estimator_
                logging.info(get_msg(f"GridSearchCV completed. Best score: \
                                    {grid_search.best_score_}", "SUCCESS"))

                logging.info(get_msg(f"Logging best parameters and score to MLflow",
                                    "INFO"))
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
        '''
        Logs metrics to MLflow
        '''
        logging.info(get_msg("Calculating and logging additional metrics", "INFO"))
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        logging.info(get_msg(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}", "SUCCESS"))

    def mem_utilization(self, step=0, sleep_time=10, target_gpu=95):
        '''
        Logs memory utilization to MLflow
        '''
        while True:
            memory_utilization = psutil.virtual_memory().percent
            mlflow.log_metric("memory_utilization", memory_utilization, step=step)
            time.sleep(sleep_time)
            if memory_utilization > target_gpu:
                logging.warning(get_msg(f"Memory usage is {memory_utilization}%. Picking CPU", "WARNING"))
                mlflow.set_tag('cpu', 'True')
                break

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