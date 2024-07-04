#!/usr/bin/env python3
"""This module contains the Zone class"""
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String, Integer, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from models.base import storage_type
from models.trip.trip import Trip
from models.utils.model_func import add_data


class Zone(BaseModel, Base):
    """Represents a taxi zone."""
    __tablename__ = 'zone'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        locationID = Column(Integer)
        latitude = Column(Float)
        longitude = Column(Float)
        geometry = Column(Geometry('POLYGON'))
        # foreign key relationships
        borough_id = Column(Integer, ForeignKey('borough.id'))
        # Relationships (example)
        trips_pickup = relationship("Trip", backref="pickup_location", foreign_keys="Trip.pickup_location")
        trips_dropoff = relationship("Trip", backref="dropoff_location", foreign_keys="Trip.dropoff_location")

    elif storage_type == 'file':
        name = ""
        locationID = ""
        latitude = ""
        longitude = ""
        geometry = ""
        state = "NY"
        country = "USA"

        @property
        def borough(self):
            """Gets the borough of the zone."""
            return self.__dict__.get('borough', "")

        @borough.setter
        def borough(self, value):
            """Sets the borough of the zone."""
            self.__dict__['borough'] = value

        @property
        def trips_pickup(self):
            """Gets all pickup trips"""
            return self.__dict__.get('trips_pickup', [])

        @trips_pickup.setter
        def trips_pickup(self, value: Trip):
            """Sets the pickup location for a trip"""
            return self._add_trip('trips_pickup', value)

        @property
        def trips_dropoff(self):
            """Gets the dropoff location for a trip"""
            return self.__dict__.get('trips_dropoff', [])

        @trips_dropoff.setter
        def trips_dropoff(self, value: Trip):
            """Sets the dropoff location for a trip"""
            return self._add_trip('trips_dropoff', value)

        def _add_trip(self, trip_type: str, value: Trip) -> int:
            """Helper method to add a trip to either pickup or dropoff list."""
            if not isinstance(value, Trip):
                return 0
            return add_data(value, trip_type)