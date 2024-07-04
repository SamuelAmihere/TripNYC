#!/usr/bin/env python3
"""This module contains the Trip class"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.orm import relationship
from models.base import storage_type
from models.utils.model_func import add_data
from datetime import datetime


class Trip(BaseModel, Base):
    """Represents a trip."""
    __tablename__ = 'trip'
    if storage_type == 'db':
        pickup_datetime = Column(DateTime, nullable=False)
        dropoff_datetime = Column(DateTime, nullable=False)
        trip_distance = Column(Float, default=0.0)
        estimated_duration = Column(Float, default=0.0)
        # Relationships (example)
        pickup_location = relationship("Zone", backref="trips_pickup", foreign_keys="Trip.pickup_location")
        dropoff_location = relationship("Zone", backref="trips_dropoff", foreign_keys="Trip.dropoff_location")
    
    elif storage_type == 'file':
        pickup_datetime = ""
        dropoff_datetime = ""
        pickup_location = ""
        dropoff_location = ""
        passenger_count = 1
        trip_distance = 0.0
        fare_amount = 0.0
        tip_amount = 0.0
        total_amount = 0.0