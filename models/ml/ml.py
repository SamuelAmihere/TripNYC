#!/user/bin/env python3
"""This module contains the ML class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, DateTime, ForeignKey, String, Integer, Float
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
        model_id = Column(Integer, ForeignKey('model.id'), nullable=False)

    else:
        predicted_at = ""
        duration = 0.0
        result = 0.0
        model_id = ""

        @property
        def model(self):
            """Gets the model for the prediction."""
            return self.__dict__.get('model', "")