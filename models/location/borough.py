#!/usr/bin/env python3
"""This module contains the Zone class"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from models.base import storage_type
from models.location.zone import Zone
from models.trip.trip import Trip
from models.utils.model_func import update_value


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

        # @property
        # def zones(self):
        #     """Gets all zones in the borough"""
        #     return getattr(Borough, 'zones', id=self.id)

        # @zones.setter
        # def zones(self, value: str):
        #     """Sets the zones for the borough"""
        #     self.__dict__['zones'] = update_value(Borough, 'zones', value, id=self.id) + [value]


        @property
        def zones(self):
            """Gets all zones in the borough"""
            from models.base import storage
            vals = storage.get_by(Borough, id=self.id)
            if vals:
                return vals.to_dict().get('zones', [])
            return []

        @zones.setter
        def zones(self, value: str):
            """Sets the zones for the borough"""
            update_value(Borough, 'zones', value, id=self.id)
            # print(f"<={_}=>")

            # if result:
            #     self.__dict__.update(result)
            #     print(f"{self.__dict__['zones']}")
            # else:
            #     self.__dict__.update({'zones': [value]})
            
            
