#!/usr/bin/env python
"""This module contains the ML class"""
import os
import sys
import platform

import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.sklearn import log_model
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import TempDir
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH
from mlflow.utils.validation import MAX_PARAMS_TAGS_PER_BATCH
from mlflow.utils.validation import MAX_ENTITY_KEY_LENGTH
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

from models.utils.config.mlflow_conf import log_memory_utilization

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score

import pandas as pd
from datetime import datetime
import uuid
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import logging
from models.utils.config.msg_config import get_msg
 
import psutil
import time
from dotenv import load_dotenv

load_dotenv()


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
        self.model = None

        
        logging.info(get_msg('TrainingSetUp initialized', 'SUCCESS'))

    def load_data(self):
        logging.info(get_msg(f'Loading data ....', 'INFO'))
        
        try:
            path = self.mlflow_info.get("data_tags")
            datapath = path.get("data_path") if path else None
            descpath = path.get("description") if path else None
            if not datapath or not descpath:
                logging.error(get_msg("Error: Check {} and descr: {e}"
                                      .format(
                                          i[1] for i in [(datapath,'[data path] '),
                                                         (descpath,'[data descriptions] ')
                                                         ] if i[0]),
                                                         'ERROR'))
            self.data = pd.read_csv(datapath) if datapath else None
            self.features_target = descpath if datapath else None
            
            if not self.data.empty and self.features_target:
                # Convert categorical variables to numerical values           
                num_feat = self.data.select_dtypes(include=['float64', 'int64']).columns
                cat_feat = self.data.drop(num_feat, axis=1).columns
                # Convert categorical variables to numerical values
                for col in cat_feat:
                    self.data[col] = self.data[col].astype('category')
                self.data = pd.get_dummies(self.data, columns=cat_feat)

                # Sepapare X, and y
                self.X = self.data.drop(self.features_target.get('target'), axis=1)
                self.y = self.data[self.features_target.get('target')]
                logging.info(get_msg(f"Data loaded. Shape: {self.X.shape}, {self.y.shape}", 'SUCCESS'))
            else:
                logging.error(get_msg(f"Error loading data and its features: {e}", 'ERROR'))
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
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            logging.info(get_msg("Pipeline ready for creation", "INFO"))
            # print("======", [(k, v) for k, v in self.pipeline if v is not None])
            self.pipeline = Pipeline([(k, v) for k, v in self.pipeline if v is not None])
            if self.pipeline:
                logging.info(get_msg("Pipeline created successfully", "SUCCESS"))
            else:
                logging.error(get_msg("Pipeline not created", "ERROR"))
                raise AttributeError('Pipeline not created')
        else:
            logging.error(get_msg("Pipeline not defined", "ERROR"))
            raise AttributeError('Pipeline not defined')
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
                # print("======", self.param_grid)
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
        model_tags = self.mlflow_info.get('model_tags')
        if model_tags:
            ml_model = model_tags.get('model')
            if ml_model:
                par = ml_model.split('(')[0]
                print(f"<<<<<ml_model: {ml_model}")
                arg = ml_model.split('(')[1].split(')')[0]
                if par == 'RandomForestClassifier':
                    from sklearn.ensemble import RandomForestClassifier
                if par == 'RandomForestRegressor':
                    from sklearn.ensemble import RandomForestRegressor
                if par == 'RandomForest':
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.ensemble import RandomForestRegressor
                if par == 'SVC':
                    from sklearn.svm import SVC
                if par == 'SVR':
                    from sklearn.svm import SVR
                if par == 'LogisticRegression':
                    from sklearn.linear_model import LogisticRegression
                if par == 'LinearRegression':
                    from sklearn.linear_model import LinearRegression
                if par == 'KNeighborsClassifier':
                    from sklearn.neighbors import KNeighborsClassifier
                if par == 'KNeighborsRegressor':
                    from sklearn.neighbors import KNeighborsRegressor
                if par == 'DecisionTreeClassifier':
                    from sklearn.tree import DecisionTreeClassifier
                if par == 'DecisionTreeRegressor':
                    from sklearn.tree import DecisionTreeRegressor
                if par == 'GradientBoostingClassifier':
                    from sklearn.ensemble import GradientBoostingClassifier
                if par == 'GradientBoostingRegressor':
                    from sklearn.ensemble import GradientBoostingRegressor
                if par == 'AdaBoostClassifier':
                    from sklearn.ensemble import AdaBoostClassifier
                if par == 'AdaBoostRegressor':
                    from sklearn.ensemble import AdaBoostRegressor
                if par == 'XGBClassifier':
                    from xgboost import XGBClassifier
                if par == 'XGBRegressor':
                    from xgboost import XGBRegressor
                # initialize the specified model
                model = f"{par}({arg})"
                self.model = eval(model)
            else:
                logging.error(get_msg("Model not defined", "ERROR"))
                raise AttributeError('Model not defined')

            model_name = model_tags.get('model_name')
            if model_name:
                self.model_name = model_name
        
        else:
            logging.error(get_msg("Model tags not defined. \
                                  \nCheck models/utils/config/ml_config.json",
                                  "ERROR"))
            raise AttributeError('Model tags not defined')

        #  =============EXPERIMENT================
        # TAGS: Experimament
        experiment_tags = self.mlflow_info.get('experiment_tags')
        if experiment_tags:
            # =====model-experiment=====:
            model_experiment = experiment_tags.get('model_experiment')
            if model_experiment:
                # << PIPELINE >>
                PIPELINE = []
                pipeline = model_experiment.get('pipeline')
                if pipeline:
                    IN_PIPELINE = lambda x: x in [i[0] for i in PIPELINE]
                    if self.model.__class__.__name__:
                        PIPELINE.append((self.model_name, self.model))
                    else:
                        if'model' in pipeline:
                            PIPELINE.append(('model', eval(pipeline.get('model'))))
                        else:
                            logging.error(get_msg("Model not defined", "ERROR"))
                            raise AttributeError('Model not defined')
                    if 'scaler' in pipeline:
                        # check if scaler is not in PIPELINE
                        if not IN_PIPELINE('scaler'):
                            if len(PIPELINE) > 0:
                                PIPELINE.insert(0, ('scaler', eval(pipeline.get('scaler'))))
                            else:
                                PIPELINE.append(('scaler', eval(pipeline.get('scaler'))))
                    if 'preprocessor' in pipeline:
                        # check if preprocessor is not in PIPELINE
                        if not IN_PIPELINE('preprocessor') and len(PIPELINE) > 0:
                            PIPELINE.insert(0, ('preprocessor', eval(pipeline.get('preprocessor'))))
                    if 'feature_selection' in pipeline:
                        if not IN_PIPELINE('feature_selection') and len(PIPELINE) > 0:
                            if IN_PIPELINE('preprocessor'):
                                PIPELINE.insert(1, ('feature_selection', eval(pipeline.get('feature_selection'))))
                msg = "Setting default values {} for pipeline".format(
                    ', '.join([f"{k}: {v}" for k, v in PIPELINE]))
                logging.info(get_msg(msg, "INFO"))
                self.pipeline = PIPELINE
                if self.pipeline:
                    logging.info(get_msg("Pipeline setup complete", "SUCCESS"))
                else:
                    logging.error(get_msg("Pipeline not defined", "ERROR"))
                    raise AttributeError('Pipeline not defined')
                #-----------------------------------------

                # << PARAM_GRID >>
                # p_grid = {}
                param_grid = model_experiment.get('param_grid')
                # print(f"======param_grid", param_grid)
                if param_grid:
                    # build param_grid for model


                    msg = "Setting default values {} for param_grid".format(
                        ', '.join([f"{k}: {v}" for k, v in param_grid.items()]))
                    logging.info(get_msg(msg, "INFO"))
                    self.param_grid = param_grid
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
        # runs:
        runs = experiment_tags.get('run')
        if runs:
            run_name = runs.get('name')
            if run_name and self.get('run_name'):
                try:
                    mlflow.set_tag('run_name', eval(run_name))
                except NameError:
                    raise NameError(f"NameError: {run_name.split('.')[-1]} is not defined")
            run_id = runs.get('id')
            if run_id:
                    run_id = eval(run_id)

        logging.info(get_msg(f"MLflow setup complete. Experiment: {experiment_tags.get('exp_name')}", "SUCCESS"))

    def train(self):
        '''
        Trains the model
        '''
            # |-------------------------------------------------------------------------|
            # |------------------|        Start MLflow run              |---------------|
            # |-------------------------------------------------------------------------|
    
 
        self.create_pipeline()
        self.create_params()




        exp_id = self.mlflow_info.get('experiment_tags').get('experiment_id')
        with mlflow.start_run(experiment_id="{}".format(exp_id), 
                              run_name=f"{self.id}", nested=True) as run:

            # Create pipeline for training


            self.run_id = run.info.run_id


            if not all([self.pipeline, self.param_grid, self.settings]):
                raise AttributeError('param_grid, cv, and scoring must be defined')

            logging.info(get_msg("Starting GridSearchCV", "INFO"))           
            grid_search = GridSearchCV(
                self.pipeline,
                self.param_grid,
                **self.settings,
                refit='accuracy'
                )
            if not hasattr(self, 'n_jobs'):
                logging.warning(get_msg("n_jobs not defined. Defaulting to 1", "WARNING"))
                self.n_jobs = 1
            if self.X_train is None or self.X_test is None:
                logging.error(get_msg("Data has not been loaded or split yet", "ERROR"))
            else:
                grid_search.fit(self.X_train, self.y_train)


                # Log accuracy at training step 10
                accuracy = accuracy_score(self.y_train, grid_search.predict(self.X_train))
                log_metric("accuracy", accuracy, step=10)
                print(f"Step 10 - Accuracy: {accuracy}")  # Print accuracy

                # Log additional metrics and parameters
                precision = precision_score(self.y_train, grid_search.predict(self.X_train))
                mlflow.log_metric("precision", precision)
                print(f"Precision: {precision}")  # Print precision

                recall = recall_score(self.y_train, grid_search.predict(self.X_train))
                mlflow.log_metric("recall", recall)
                print(f"Recall: {recall}")  # Print recall

                param1 = grid_search.best_params_['model__param1']
                mlflow.log_param("param1", param1)
                print(f"Param1: {param1}")  # Print param1

                param2 = grid_search.best_params_['model__param2']
                mlflow.log_param("param2", param2)
                print(f"Param2: {param2}")  # Print param2





                # # Log accuracy at training step 10
                # accuracy = accuracy_score(self.y_train, grid_search.predict(self.X_train))
                # log_metric("accuracy", accuracy, step=10)

                # # 
                
                # # Log additional metrics and parameters
                # mlflow.log_metric("precision", precision_score(self.y_train, grid_search.predict(self.X_train)))
                # mlflow.log_metric("recall", recall_score(self.y_train, grid_search.predict(self.X_train)))
                # mlflow.log_param("param1", grid_search.best_params_['model__param1'])
                # mlflow.log_param("param2", grid_search.best_params_['model__param2'])

                # Log model
                log_model(grid_search, "model")
                logging.info(get_msg("GridSearchCV completed", "SUCCESS"))
                self.model = grid_search

        # check if model is trained
        if hasattr(self, 'model') and self.model is not None:
            logging.error(get_msg("Model has not been trained", "ERROR"))
            raise Exception("Model has not been trained")
        logging.info(get_msg("Model training completed", "SUCCESS"))

    def my_log_metrics(self, r):
        '''
        Logs metrics to MLflow
        '''
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print(f"run_id: {r.info.run_id}")
        print(f"artifacts: {artifacts}")
        print(f"params: {r.data.params}")
        print(f"metrics: {r.data.metrics}")
        print(f"tags: {tags}")

    def log_metrics(self):
        '''
        Logs metrics to MLflow
        '''
        for metric, value in self.metrics.items():
            
            log_metric(metric, value)


    def my_scoring(self, **kwargs):
        '''
        Scoring function
        '''
        for metric, value in kwargs.items():
            log_metric(metric, value)


    
    

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