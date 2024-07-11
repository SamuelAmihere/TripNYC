#!/user/bin/env python3
"""This module contains the ML class"""
import sys
sys.path.append("..")

import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

from sqlalchemy.orm import relationship
from dotenv import load_dotenv

from models.utils.model_func import update_value
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class Prediction(BaseModel, Base):
    """Represents a prediction."""
    __tablename__ = 'prediction'
    if storage_type == 'db':
        predicted_at = Column(DateTime, nullable=False)
        duration = Column(Float, default=0.0)
        result = Column(Float, default=0.0)
        model_id = Column(String(255), ForeignKey('model.id'), nullable=False)
        model = relationship("Model", back_populates="predictions")
    else:
        predicted_at = ""
        duration = 0.0
        result = 0.0
        model = ""
        @property
        def model(self):
            """Gets the model for the prediction."""
            return self.__dict__.get('model', "")


class ML(BaseModel, Base):
    """Represents a model."""
    __tablename__ = 'ml'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        version = Column(String(255), nullable=False)
        description = Column(String(255), default="")
        performance_metrics = Column(JSONB)
        model_path = Column(String(255))
        hyperparameters = Column(JSONB, nullable=False)
    else:
        name = ""
        version = ""
        description = ""
        model = ""
        model_path = ""
        performance_metrics = {}
        hyperparameters = {}

        @property
        def performance_metrics(self):
            """Gets the performance metrics for the model."""
            from models.base import storage
            x = storage.get_by(ML, id=self.id)
            if x:
                return x.to_dict().get('performance_metrics', {})
            return self.__dict__.get('performance_metrics', {})

        @performance_metrics.setter
        def performance_metrics(self, value: dict):
            """Sets the performance metrics for the model"""
            _ = update_value(self, ML, 'performance_metrics', value, True, id=self.id)

        @property
        def hyperparameters(self):
            """Gets the hyperparameters for the model"""
            from models.base import storage
            x = storage.get_by(ML, id=self.id)
            if x:
                return x.to_dict().get('hyperparameters', {})
            return self.__dict__.get('hyperparameters', {})

        @hyperparameters.setter
        def hyperparameters(self, value: dict):
            """Sets the hyperparameters for the model"""
            _ = update_value(self, ML, 'hyperparameters', value, True, id=self.id)

class Model(BaseModel, Base):
    """Represents a machine learning model."""
    __tablename__ = 'model'
    if storage_type == 'db':
        ml_id = Column(String(255), ForeignKey('ml.id'), nullable=False)
        predictions = relationship("Prediction", back_populates="model")

    else:
        ml_id = ""
        predictions = []
        @property
        def predictions(self):
            """Gets all predictions for the model"""
            from models.base import storage
            vals = storage.get_by(Model, id=self.id)
            if vals:
                return vals.to_dict().get('predictions', [])
            return self.__dict__.get('predictions', [])

        @predictions.setter
        def predictions(self, value: list):
            """Sets the predictions for the model"""
            _ = update_value(self, Model, 'predictions', value, id=self.id)
    
    
    def load_context(self, context):
        # Here we fetch and expose the model and any 
        # other requirements for `predict()` calls.
        # `context` will have any artifacts passed to
        # the model.
        pass

    def predict(self, context, model_input):
        # 1. We modify `model_input` at will
        # 2. Call the model and return predictions
        # 3. Return the predictions
        pass

class ModelPerformance(BaseModel, Base):
    """Represents a machine learning model.
    It tracks the performance of the model in
    production.
    """
    __tablename__ = 'model_performance'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        value = Column(Float, default=0.0)
        model = Column(String(255), ForeignKey('model.id'), nullable=False)
    else:
        name = ""
        metric = ""
        value = 0.0
        model = ""

        @property
        def model(self):
            """Gets the model for the performance metric."""
            return self.__dict__.get('model', "")
