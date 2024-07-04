#!/usr/bin/env python3
"""This module contains the Taxi and ForHireVehicle classes"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from models.base import storage_type


class Taxi(BaseModel, Base):
    """Represents a taxi."""
    __tablename__ = 'taxi'
    if storage_type == 'db':
        color = Column(String(50))
    elif storage_type == 'file':
        color = ""