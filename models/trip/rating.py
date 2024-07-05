#!/usr/bin/python3
"""Contains the Rating class"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, ForeignKey, String, Integer, Float
from sqlalchemy.orm import relationship
from models.utils.model_func import update_value
from dotenv import load_dotenv
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class Rating(BaseModel, Base):
    """Represents a rating."""
    __tablename__ = 'rating'
    if storage_type == 'db':
        name = Column(String(255), nullable=False)
        rating = Column(Float, default=0.0)
        comment = Column(String(255), default="")
        trip_id = Column(String(255), ForeignKey('trip.id'))

        # Relationships
        trip = relationship("Trip", back_populates="rating")

    elif storage_type == 'file':
        name = ""
        rating = 0.0
        comment = ""
        trip_id = ""

