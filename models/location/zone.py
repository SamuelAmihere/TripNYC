#!/usr/bin/env python3
"""This module contains the Zone class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String, Integer, Float
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from models.trip.trip import Trip
from models.utils.model_func import add_data
from dotenv import load_dotenv
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')

class Zone(BaseModel, Base):
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
        borough = relationship("Borough", back_populates="zones")
        trips_pickup = relationship("Trip", backref="pickup_location", foreign_keys="Trip.pickup_location")
        trips_dropoff = relationship("Trip", backref="dropoff_location", foreign_keys="Trip.dropoff_location")

    elif storage_type == 'file':
        name = ""
        locationID = ""
        latitude = ""
        longitude = ""
        geometry = ""
        borough_id = ""

        @property
        def borough(self):
            """Gets the borough of the zone."""
            return self.__dict__.get('borough', "")

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

        def _add_trip(self, attr: str, value: Trip) -> int:
            """Helper method to add a trip to either pickup or dropoff list."""
            if not isinstance(value, Trip):
                raise ValueError("Value must be a Trip object.")
            return add_data(value, attr)
        

# class ZoneTrip(BaseModel, Base):
#     """Represents a zone trip."""
#     __tablename__ = 'zone_trip'
#     if storage_type == 'db':
#         zone_id = Column(String, ForeignKey('zone.id'))
#         taxi_trip_id = Column(Integer, ForeignKey('taxi_trip.id'))
#         fhv_trip_id = Column(Integer, ForeignKey('fhv_trip.id'))
#         hvfhs_trip_id = Column(Integer, ForeignKey('hvfhs_trip.id'))
#         # Relationships
#         zone = relationship("Zone", backref="trip")
#         trip = relationship("Trip", backref="zone")

#     elif storage_type == 'file':
#         zone_id = ""
#         trip_id = ""
#         zone = []
#         tripe = []

#         @property
#         def zone(self):
#             """Gets the zone."""
#             return getter(ZoneTrip, 'zone', id=self.id)

#         @zone.setter
#         def zone(self, value: str):
#             """Sets the zone."""
#             self.__dict__['zone'] = update_value(ZoneTrip, 'zone', id=self.id) + [value]

#         @property
#         def trip(self):
#             """Gets the trip."""
#             return getter(ZoneTrip, 'trip', id=self.id)

#         @trip.setter
#         def trip(self, value: str):
#             """Sets the trip."""
#             self.__dict__['trip'] = update_value(ZoneTrip, 'trip', id=self.id) + [value]