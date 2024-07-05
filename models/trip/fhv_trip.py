#!/usr/bin/python3
"""create a unique FileStorage instance for your application"""
import os
from models.trip.trip import Trip
from sqlalchemy.orm import relationship
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String
from dotenv import load_dotenv

from models.vehicle.base import DispatchBase
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class FHVTrip(BaseModel, Base):
    """Represents a for-hire vehicle (fhv) trip."""
    __tablename__ = 'for_hire_vehicle_trip'
    if storage_type == 'db':
        affiliated_base_number = Column(String(255), nullable=False)
        dispatching_base_number = Column(String(255), nullable=False)
        trip_id = Column(String(255), ForeignKey('trip.id'))    

    elif storage_type == 'file':
        affiliated_base_number = ""
        dispatching_base_number = ""