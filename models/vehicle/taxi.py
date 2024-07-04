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
        taxi_trips = relationship("TaxiTrip", backref="taxi", cascade="all, delete")

    elif storage_type == 'file':
        color = ""

        @property
        def taxi_trips(self):
            """Gets the list of TaxiTrip instances associated with this Taxi."""
            from models.base import storage
            related_taxi_trips = []
            all_taxi_trips = storage.all("TaxiTrip").values()
            for taxi_trip in all_taxi_trips:
                if taxi_trip.taxi_id == self.id:
                    related_taxi_trips.append(taxi_trip)
            return related_taxi_trips


class ForHireVehicle(BaseModel, Base):
    """Represents a for-hire vehicle."""
    __tablename__ = 'for_hire_vehicle'
    if storage_type == 'db':
        dispatching_base_number = Column(String(50))
        fhv_trips = relationship("FHVTrip", backref="for_hire_vehicle", cascade="all, delete")
        hvfhs_trips = relationship("HVFHSTrip", backref="for_hire_vehicle", cascade="all, delete")

    elif storage_type == 'file':
        dispatching_base_number = ""

        @property
        def fhv_trips(self):
            """Gets the list of FHVTrip instances associated with this ForHireVehicle."""
            from models import storage
            related_fhv_trips = []
            all_fhv_trips = storage.all("FHVTrip").values()
            for fhv_trip in all_fhv_trips:
                if fhv_trip.for_hire_vehicle_id == self.id:
                    related_fhv_trips.append(fhv_trip)
            return related_fhv_trips

        @property
        def hvfhs_trips(self):
            """Gets the list of HVFHSTrip instances associated with this ForHireVehicle."""
            from models import storage
            related_hvfhs_trips = []
            all_hvfhs_trips = storage.all("HVFHSTrip").values()
            for hvfhs_trip in all_hvfhs_trips:
                if hvfhs_trip.for_hire_vehicle_id == self.id:
                    related_hvfhs_trips.append(hvfhs_trip)
            return related_hvfhs_trips