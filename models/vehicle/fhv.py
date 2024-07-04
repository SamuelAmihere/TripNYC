#!/usr/bin/env python3
"""This module contains the Taxi and ForHireVehicle classes"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from models.base import storage_type


class ForHireVehicle(BaseModel, Base):
    """Represents a for-hire vehicle."""
    __tablename__ = 'for_hire_vehicle'
    if storage_type == 'db':
        dispatching_base_number = Column(String(50))

    elif storage_type == 'file':
        dispatching_base_number = ""