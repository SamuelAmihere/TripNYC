#! /usr/bin/env python3
"""This module contains the BaseModel class"""
from os import getenv
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from models import base
import uuid
from datetime import datetime
import os
# from dotenv import load_dotenv

# load_dotenv()

storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')

if storage_type == 'db':
    Base = declarative_base()
else:
    Base = object


class BaseModel:
    """This class is the base class for all other
    classes in this project
    """
    if storage_type == 'db':
        id = Column(String(60), primary_key=True, nullable=False,
                    unique=True)
        created_at = Column(DateTime, default=datetime.utcnow,
                            nullable=False)
        updated_at = Column(DateTime, default=datetime.utcnow,
                            nullable=False)

    def __init__(self, *args, **kwargs):
        """Initializes an instance of the BaseModel class"""
        if kwargs:
            if storage_type != 'db':
                kwargs.pop('__class__', None)
            if 'id' not in kwargs:
                kwargs['id'] = str(uuid.uuid4())
            if 'created_at' not in kwargs:
                kwargs['created_at'] = datetime.utcnow()
            elif isinstance(kwargs['created_at'], datetime) == False:
                kwargs['created_at'] = datetime.strptime(kwargs['created_at'],
                                                         '%Y-%m-%dT%H:%M:%S.%f')
            if 'updated_at' not in kwargs:
                kwargs['updated_at'] = datetime.utcnow()
            elif isinstance(kwargs['updated_at'], datetime) == False:
                kwargs['updated_at'] = datetime.strptime(kwargs['updated_at'],
                                                         '%Y-%m-%dT%H:%M:%S.%f')
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            self.id = str(uuid.uuid4())
            self.created_at = datetime.utcnow()

    def __str__(self):
        """Returns a string representation of the instance"""
        return "[{}] ({}) {}".format(self.__class__.__name__, self.id,
                                     self.__dict__)
    
    def save(self):
        """Updates the public instance attribute updated_at
        to the current datetime
        """
        self.updated_at = datetime.now()
        base.storage.new(self)
        base.storage.save()

    def to_dict(self, save_fs=0):
        """Returns a dictionary containing all keys/values
        of __dict__ of the instance
        """
        new_dict = self.__dict__.copy()
        if '_sa_instance_state' in new_dict:
            del new_dict['_sa_instance_state']
        if save_fs == 0:
            if 'password' in new_dict:
                del new_dict['password']
        new_dict['__class__'] = self.__class__.__name__
        new_dict['created_at'] = self.created_at.isoformat()
        new_dict['updated_at'] = self.updated_at.isoformat()
        return new_dict

    def delete(self):
        """Deletes the current instance from the storage"""
        base.storage.delete(self)
        base.storage.save()