#!/usr/bin/env python3
"""This module contains the Zone class"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from models.base import storage_type
from models.trip.trip import Trip
from models.utils.model_func import add_data


class Borough(BaseModel, Base):
    """Represents a borough."""
    __tablename__ = 'borough'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        state = Column(String(255), default="NY")
        country = Column(String(255), default="USA")

    elif storage_type == 'file':
        name = ""
        state = "NY"
        country = "USA"