#!/user/bin/python3
"""This module contains the dispatch base class for fhv trips"""
import os
from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, String
from dotenv import load_dotenv
load_dotenv()


storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')


class DispatchBase(BaseModel, Base):
    """Represents a dispatch base for fhv trips."""
    __tablename__ = 'dispatch_base'
    if storage_type == 'db':
        high_volume_licence_num = Column(String(255), nullable=False)
        license_num = Column(String(255), nullable=False)
        base_name = Column(String(255), nullable=False)
        app_company = Column(String(255), nullable=False)

    elif storage_type == 'file':
        high_volume_licence_num = ""
        license_num = ""
        base_name = ""
        app_company = ""