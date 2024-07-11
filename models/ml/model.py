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
from sqlalchemy import Column, ForeignKey, String, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from models.base.base_model import Base, BaseModel
from models.ml.ml import ML
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

from models.utils.model_func import create_experiment, getter, update_value

load_dotenv()

storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class TrainingSetUp:
    def __init__(self, **kwargs):
        logging.info(get_msg('Initializing TrainingSetUp', 'INFO'))
        # remove id, created_at, and updated_at from kwargs.
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = self.y_pred_proba = None

        logging.info(get_msg('TrainingSetUp initialized', 'SUCCESS'))

    def load_data(self):
        logging.info(get_msg(f'Loading data ....', 'INFO'))
        try:
            datapath = self.data_path if self.data_path else None
            descpath = self.description if self.description else None
            if not datapath or not descpath:
                logging.error(get_msg("Error: Check {} and descr: {e}"
                                      .format(
                                          i[1] for i in [(datapath,'[data path] '),
                                                         (descpath,'[data descriptions] ')
                                                         ] if i[0]),
                                                         'ERROR'))
            data = pd.read_csv(datapath) if datapath else None
            features_target = descpath if datapath else None
            
            if not data.empty and features_target:
                # Convert categorical variables to numerical values           
                num_feat = data.select_dtypes(include=['float64', 'int64']).columns
                cat_feat = data.drop(num_feat, axis=1).columns
                # Convert categorical variables to numerical values
                for col in cat_feat:
                    data[col] = data[col].astype('category')
                data = pd.get_dummies(data, columns=cat_feat)

                # Sepapare X, and y
                X = data.drop(features_target.get('target'), axis=1)
                y = data[features_target.get('target')]
                logging.info(get_msg(f"Data loaded. Shape: {X.shape}, {y.shape}", 'SUCCESS'))

                self.split_data(X, y, test_size=0.2, random_state=42)
            else:
                logging.error(get_msg(f"Error loading data and its features: {e}", 'ERROR'))
        except FileNotFoundError as e:
            logging.error(get_msg(f"Error loading data: {e}", 'ERROR'))
            raise FileNotFoundError(f"Error loading data: {e}")

    def split_data(self,X, y, test_size=0.2, random_state=42):
        logging.info(get_msg("Splitting data into train and test sets", "INFO"))
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info(get_msg(f"Data split. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}", "SUCCESS"))
   
class ModelTrainer(BaseModel, Base):
    """This class trains a model"""
    __tablename__ = 'model_trainer'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        data_path = Column(String(255), nullable=False)
        description = Column(String(255), default="")
        experiment_id = Column(String(255), nullable=False)
        experiment_tags = Column(JSONB, nullable=False)
        run_name = Column(String(255), nullable=False)
        run_id = Column(String(255), nullable=False)
        pipeline = Column(JSONB, nullable=False)
        param_grid = Column(JSONB, nullable=False)
        settings = Column(JSONB, nullable=False)

    else:
        name = ""
        data_path = ""
        description = ""
        experiment_id = ""
        experiment_tags = {}
        run_name = ""
        run_id = ""
        model_id = ""
        pipeline = {}
        param_grid = {}
        settings = {}

        @property
        def experiment_tags(self):
            """Gets the experiment tags for the trainer."""
            return getter(ModelTrainer, 'experiment_tags', id=self.id)

        @experiment_tags.setter
        def experiment_tags(self, value: dict):
            """Sets the experiment tags for the trainer."""
            if not value:
                return
            _ = update_value(self, ModelTrainer, 'experiment_tags', value, id=self.id)
            if _:
                self.experiment_tags = _
                logging.info(get_msg(f"Experiment tags set: {self.experiment_tags}", "SUCCESS"))
            else:
                logging.error(get_msg("Experiment tags not set", "ERROR"))

        @property
        def pipeline(self):
            """Gets the pipeline for the trainer."""
            pipe = getter(ModelTrainer, 'pipeline', id=self.id)
            if pipe:
                pipe = [(k, eval(v)) for k, v in pipe.items() if v is not None]
            else:
                pipe = []
            return getter(ModelTrainer, 'pipeline', id=self.id)
        
        @pipeline.setter
        def pipeline(self, value: dict):
            """Sets the pipeline for the trainer."""
            if not value:
                return
            _ = update_value(self, ModelTrainer, 'pipeline', value, id=self.id)
            if _:
                self.pipeline = _
                logging.info(get_msg(f"Pipeline set: {self.pipeline}", "SUCCESS"))
            else:
                logging.error(get_msg("Pipeline not set", "ERROR"))  

        @property
        def param_grid(self):
            """Gets the param_grid for the trainer."""
            return getter(ModelTrainer, 'param_grid', id=self.id)
        
        @param_grid.setter
        def param_grid(self, value: dict):
            """Sets the param_grid for the trainer."""
            if not value:
                return
            _ = update_value(self, ModelTrainer, 'param_grid', value, id=self.id)
            if _:
                self.param_grid = _
                logging.info(get_msg(f"Param_grid set: {self.param_grid}", "SUCCESS"))
            else:
                logging.error(get_msg("Param_grid not set", "ERROR"))

        @property
        def settings(self):
            """Gets the settings for the trainer."""
            return getter(ModelTrainer, 'settings', id=self.id)

        @settings.setter
        def settings(self, value: dict):
            """Sets the settings for the trainer."""
            if not value:
                return
            _ = update_value(self, ModelTrainer, 'settings', value, id=self.id)
            if _:
                self.settings = _
                logging.info(get_msg(f"Settings set: {self.settings}", "SUCCESS"))
            else:
                logging.error(get_msg("Settings not set", "ERROR"))

    def train(self):
        '''
        Trains the model
        '''
        mlflow.autolog()
        
            # |-------------------------------------------------------------------------|
            # |------------------|        Start MLflow run              |---------------|
            # |-------------------------------------------------------------------------|

        setup = TrainingSetUp(**self.__dict__)
        setup.load_data()
        x_train, x_test,  = setup.X_train, setup.X_test
        y_train, y_test = setup.y_train, setup.y_test

        # |-------------------------------------------------------------------------|
        # |------------------|        Start MLflow run              |---------------|
        # |-------------------------------------------------------------------------|
        # Create an experiment in MLflow
        exp_name = self.experiment_tags.get('experiment_name')
        exp_id = create_experiment(exp_name)
        artifact_path = f"mlruns/{exp_id}"
        
        if not self.pipeline and not self.param_grid and not self.settings:
            logging.error(get_msg("Error: Pipeline, param_grid, and settings not set", "ERROR"))
            return
        print(f"=======PIPELINE: {self.pipeline}===========")
        # Start MLflow run
        with mlflow.start_run(experiment_id=exp_id, run_name=self.run_name) as run:
            exp_tags = self.experiment_tags
            if exp_tags:
                self.experiment_tags['experiment_id'] = run.info.experiment_id
                self.experiment_tags.update(
                    {
                        **self.experiment_tags,
                        'run': {'run_id': run.info.run_id,
                             'run_name': run.info.run_name}
                        
                    },
                    )
                
    
            # 

            pipeline = Pipeline(self.pipeline)
            param_grid = self.param_grid
            settings = self.settings
            
            # Train the model
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                **settings,
                refit='neg_mean_squared_error',
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(x_train, y_train)
            # Log model performance
            y_pred = grid_search.predict(x_test)
            log_metric('accuracy', accuracy_score(y_test, y_pred), step=1)
            log_metric('precision', precision_score(y_test, y_pred), step=1)
            log_metric('recall', recall_score(y_test, y_pred), step=1)
            log_metric('mean_squared_error', mean_squared_error(y_test, y_pred), step=1)
            # Log model parameters
            log_param('pipeline', pipeline)
            log_param('param_grid', param_grid)

            # Best parameters
            best_params = grid_search.best_params_
            log_param('best_params', best_params)

            # Save final model
            model_path = f"{artifact_path}/model"
            log_model(grid_search, model_path)

