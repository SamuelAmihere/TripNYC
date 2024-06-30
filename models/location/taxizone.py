#!/usr/bin/env python3
from shapely.wkb import loads as wkb
from sqlalchemy import Column, ForeignKey, Enum, Integer, String, Float, Table
from geoalchemy2 import Geometry
from models.base.base_model import BaseModel, Base

class TaxiZone(BaseModel, Base):
    __tablename__ = 'taxi_zones'

    id = Column(Integer, primary_key=True)
    zone = Column(String(255))
    borough = Column(String(255))
    geometry = Column(Geometry('MULTIPOLYGON'))

    def __repr__(self):
        return f'<TaxiZone {self.zone}>'

    def to_dict(self):
        return {
            'id': self.id,
            'zone': self.zone,
            'borough': self.borough,
            'geometry': wkb.loads(bytes(self.geometry.data)).wkt
        }