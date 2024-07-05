#!/usr/bin/env python3
"""This module contains the TaxiTrip class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String, Integer, Float, DateTime
from sqlalchemy.orm import relationship
from models.vehicle.vehicle import Taxi
from models.base import classes
from models.utils.model_func import add_data
from dotenv import load_dotenv
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')
Vehicle = classes.get('Vehicle')
Zone = classes.get('Zone')

class TaxiTrip(BaseModel, Base):
    """Represents a taxi trip."""
    __tablename__ = 'taxi_trip'
    if storage_type == 'db':
        vendor_id = Column(String(50))
        passenger_count = Column(Integer, default=1)
        payment_type = Column(String(50))
        fare = Column(Float, default=0.0)
        tip = Column(Float, default=0.0)
        toll = Column(Float, default=0.0)
        total_amount = Column(Float, default=0.0)
        congestion_surcharge = Column(Float, default=0.0)

        trip_id = Column(Integer, ForeignKey('taxi.id'), nullable=False)
        amount_estimator = Column(Integer, ForeignKey('prediction.id'), nullable=False)

        # Relationships

    elif storage_type == 'file':
        taxi_id = ""
        pickup_location_id = ""
        dropoff_location_id = ""
        pickup_datetime = ""
        dropoff_datetime = ""
        passenger_count = 1
        trip_distance = 0.0
        fare_amount = 0.0
        tip_amount = 0.0
        total_amount = 0.0

        @property
        def taxi(self):
            """Gets the taxi for the trip."""
            return self.__dict__.get('taxi', "")

        @property
        def pickup_location(self):
            """Gets the pickup location for the trip."""
            return self.__dict__.get('pickup_location', "")

        @property
        def dropoff_location(self):
            """Gets the dropoff location for the trip."""
            return self.__dict__.get('dropoff_location', "")

