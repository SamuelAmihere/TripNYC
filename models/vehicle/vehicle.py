#!/usr/bin/env python3
"""This module contains the Taxi and ForHireVehicle classes"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from dotenv import load_dotenv
from models.utils.model_func import update_value
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class Vehicle(BaseModel, Base):
    """Represents a vehicle."""
    __tablename__ = 'vehicle'
    if storage_type == 'db':
        name = Column(String(50))
        trips = relationship('Trip', back_populates='vehicle')
    elif storage_type == 'file':
        name = ""
        trips = []

        @property
        def trips(self):
            """Gets all zones in the borough"""
            from models.base import storage
            vals = storage.get_by(Vehicle, id=self.id)
            if vals:
                return vals.to_dict().get('zones', [])
            return []

        @trips.setter
        def trips(self, value: list):
            """Sets the zones for the borough"""
            _ = update_value(self, Vehicle, 'zones', value, id=self.id)