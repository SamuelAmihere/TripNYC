#! /usr/bin/env python3
"""This module contains the DBStorage class"""
from os import getenv
from models.base.base_model_loc import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from geoalchemy2 import Geometry
from dotenv import load_dotenv

load_dotenv()


classes= []

class DBStorage:
    """ This class is the storage engine for the project """
    __engine = None
    __session = None

    def __init__(self):
        """ This method initializes a new instance of the DBStorage class """
        self.__engine = create_engine('postgresql+psycopg2://{}:{}@{}/{}'
                                      .format(getenv('TRIPNYC_POSTGRES_USER'),
                                              getenv('TRIPNYC_POSTGRES_PWD'),
                                              getenv('TRIPNYC_POSTGRES_HOST'),
                                              getenv('TRIPNYC_POSTGRES_DB')),
                                      pool_pre_ping=True)

        if getenv('TRIPNYC_TYPE_STORAGE') == 'test':
            Base.metadata.drop_all(self.__engine)

    def all(self, cls=None):
        """ This method returns a dictionary of all instances of a class """
        new_dict = {}
        if cls:
            for obj in self.__session.query(cls):
                key = '{}.{}'.format(obj.__class__.__name__, obj.id)
                new_dict[key] = obj
        else:
            for c in classes:
                for obj in self.__session.query(c):
                    key = '{}.{}'.format(type(obj).__name__, obj.id)
                    new_dict[key] = obj
        return new_dict
    
    def new(self, obj):
        """ This method adds a new instance to the session """
        self.__session.add(obj)
    
    def new_all(self, *args):
        """ This method adds all instances to the session """
        for obj in args:
            self.__session.add(obj)

    def save(self):
        """ This method commits all changes to the database """
        self.__session.commit()

    def execute(self, queries):
        """ This method executes all queries in the list """
        for q in queries:
            self.__session.execute(q)

    def delete(self, obj=None):
        """ This method deletes an instance from the session """
        if obj:
            self.__session.delete(obj)
            self.save()

    def delete_all(self):
        """ This method deletes all instances from the session """
        for c in classes:
            self.__session.query(c).delete()
        self.save()

    def reload(self):
        """ This method creates all tables in the database """
        Base.metadata.create_all(self.__engine)
        self.__session = scoped_session(
            sessionmaker(
                bind=self.__engine,
                expire_on_commit=False))

    def close(self):
        """ This method closes the session """
        self.__session.remove()

    def get(self, cls, id):
        """ This method retrieves an instance from the session
        Args:
            cls (str): The class name
            id (str): The instance id
        Returns: The instance or None
        """
        if id is None or cls is None:
            return None
        objs = self. all(cls)
        return objs.get('{}.{}'.format(cls, id))
    
    def count(self, cls=None):
        """ This method returns the number of instances of a class """
        if cls:
            return self.__session.query(cls).count()
        else:
            count = 0
            for c in classes:
                count += self.__session.query(c).count()
            return count
    
    def get_all(self, cls):
        """ This method returns a list of all instances of a class """
        if cls:
            return list(self.all(cls).values())
        return None
    
    def get_by(self, cls, **kwargs):
        """ This method returns a list of instances of a class that match the keyword arguments """
        if cls:
            return self.__session.query(cls).filter_by(**kwargs).all()
        return None

    def get_one_by(self, cls, **kwargs):
        """ This method returns the first instance of a class that matches the keyword arguments """
        if cls:
            return self.__session.query(cls).filter_by(**kwargs).first()
        return None
    
    def get_or_create(self, cls, **kwargs):
        """ This method returns the first instance of a class that matches the keyword arguments
        or creates a new instance if one does not exist """
        instance = self.get_one_by(cls, **kwargs)
        if instance is None:
            instance = cls(**kwargs)
            self.new(instance)
            self.save()
        return instance
    
    def get_or_create_all(self, cls, *args):
        """ This method returns a list of instances of a class that match the keyword arguments
        or creates new instances if they do not exist """
        instances = []
        for kwargs in args:
            instance = self.get_or_create(cls, **kwargs)
            instances.append(instance)
        return instances
    
    def get_or_create_by(self, cls, **kwargs):
        """ This method returns the first instance of a class that matches the keyword arguments
        or creates a new instance if one does not exist """
        if cls:
            instance = self.get_one_by(cls, **kwargs)
            if instance is None:
                instance = cls(**kwargs)
                self.new(instance)
                self.save()
            return instance
        return None

    def get_or_create_all_by(self, cls, *args):
        """ This method returns a list of instances of a class that match the keyword arguments
        or creates new instances if they do not exist """
        if cls:
            instances = []
            for kwargs in args:
                instance = self.get_or_create_by(cls, **kwargs)
                if instance:
                    instances.append(instance)
            return instances