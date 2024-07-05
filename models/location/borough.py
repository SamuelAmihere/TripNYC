#!/usr/bin/env python3
"""This module contains the Zone class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from models.utils.model_func import add_data, update_value
from dotenv import load_dotenv
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')

class Borough(BaseModel, Base):
    """Represents a borough."""
    __tablename__ = 'borough'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        state = Column(String(255), default="NY")
        country = Column(String(255), default="USA")
        # Relationships (example)
        zones = relationship("Zone", back_populates="borough")

    elif storage_type == 'file':
        name = ""
        state = "NY"
        country = "USA"
        zones = []

        @property
        def zones(self):
            """Gets all zones in the borough"""
            from models.base import storage
            vals = storage.get_by(Borough, id=self.id)
            if vals:
                return vals.to_dict().get('zones', [])
            return []

        @zones.setter
        def zones(self, value: list):
            """Sets the zones for the borough"""
            _ = update_value(self, Borough, 'zones', value, id=self.id)