#!/user/bin/env python3
"""This module contains the ML class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Integer, Float
from sqlalchemy.dialects.postgresql import JSONB
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
        model = Column(String(255), ForeignKey('ml.id'), nullable=False)
        ml = relationship("ML", back_populates="predictions")

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
        performance_metrics = Column(JSONB, nullable=False)
        predictions = relationship("Prediction", back_populates="ml")

    else:
        name = ""
        version = ""
        description = ""
        model = ""
        predictions = []

        @property
        def predictions(self):
            """Gets all predictions for the model"""
            from models.base import storage
            vals = storage.get_by(ML, id=self.id)
            if vals:
                return vals.to_dict().get('predictions', [])
            return []

        @predictions.setter
        def predictions(self, value: list):
            """Sets the predictions for the model"""
            _ = update_value(self, ML, 'predictions', value, id=self.id)


class ModelPerformance(BaseModel, Base):
    """Represents a machine learning model."""
    __tablename__ = 'model_performance'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        metric = Column(String(255), nullable=False)
        value = Column(Float, default=0.0)
        model = Column(String(255), ForeignKey('ml.id'), nullable=False)
    else:
        name = ""
        metric = ""
        value = 0.0
        model = ""

        @property
        def model(self):
            """Gets the model for the performance metric."""
            return self.__dict__.get('model', "")
