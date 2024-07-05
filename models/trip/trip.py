#!/usr/bin/env python3
"""This module contains the Trip class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String, Integer, Float, DateTime
from sqlalchemy.orm import relationship
from dotenv import load_dotenv

from models.utils.model_func import update_value
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class Trip(BaseModel, Base):
    """Represents a trip."""
    __tablename__ = 'trip'
    if storage_type == 'db':
        pickup_datetime = Column(DateTime, nullable=False)
        dropoff_datetime = Column(DateTime, nullable=False)
        pickup_location = Column(Integer, ForeignKey('zone.id'), nullable=False)
        dropoff_location = Column(Integer, ForeignKey('zone.id'), nullable=False)
        trip_distance = Column(Float, default=0.0)

        vehicle_id = Column(Integer, ForeignKey('vehicle.id'), nullable=False)
        estimated_duration = Column(Integer, ForeignKey('prediction.id'), nullable=False)

        # Relationships
        vehicle = relationship("Vehicle", back_populates="trips")
        ratings = relationship("Rating", back_populates="trip")
    
    elif storage_type == 'file':
        pickup_datetime = ""
        dropoff_datetime = ""
        pickup_location = ""
        dropoff_location = ""
        trip_distance = 0.0
        vehicle_id = ""
        estimated_duration = ""
        ratings = []

        @property
        def vehicle(self):
            """Gets the vehicle for the trip."""
            return self.__dict__.get('vehicle', "")

        @property
        def pickup_location(self):
            """Gets the pickup location for the trip."""
            return self.__dict__.get('pickup_location', "")

        @property
        def dropoff_location(self):
            """Gets the dropoff location for the trip."""
            return self.__dict__.get('dropoff_location', "")
        
        @property
        def ratings(self):
            """Gets the ratings for the trip."""
            from models.base import storage
            vals = storage.get_by(Trip, id=self.id)
            if vals:
                return vals.to_dict().get('ratings', [])
            return []

        @ratings.setter
        def ratings(self, value: list):
            """Sets the ratings for the trip."""
            _ = update_value(self, Trip, 'ratings', value, id=self.id)