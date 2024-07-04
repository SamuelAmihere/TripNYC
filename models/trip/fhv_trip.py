#!/usr/bin/python3
"""create a unique FileStorage instance for your application"""
from models.trip.trip import Trip
from sqlalchemy.orm import relationship
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String
from models.base import storage_type


class ForHireVehicleTrip(BaseModel, Base):
    """Represents a for-hire vehicle (fhv) trip."""
    __tablename__ = 'for_hire_vehicle_trip'
    if storage_type == 'db':
        for_hire_vehicle_id = Column(String(50))
        trip_id = Column(String(50))
        # Relationships
        trip = relationship("Trip", backref="for_hire_vehicle_trip")

    elif storage_type == 'file':
        for_hire_vehicle_id = ""
        trip_id = ""
        trip = ""

        @property
        def trip(self):
            """Gets the trip"""
            from models.base import storage
            vals = storage.get_by(ForHireVehicleTrip, id=self.id)
            if vals:
                return vals.to_dict().get('trip', "")
            return ""
        
