#!/usr/bin/python3
"""create a unique FileStorage instance for your application"""
from models.location.borough import Borough
from models.location.zone import Zone
from models.trip.fhv_trip import FHVTrip
from models.trip.taxi_trip import TaxiTrip
from models.trip.trip import Trip
from models.vehicle.base import DispatchBase
from models.vehicle.vehicle import Vehicle
import os
from dotenv import load_dotenv

load_dotenv()

classes = {'Zone': Zone, 'Borough': Borough,
           'Vehicle':Vehicle, 'Trip': Trip,
           'DispatchBase': DispatchBase,
            'FHVTrip': FHVTrip,
            'TaxiTrip': TaxiTrip
           }

storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')
if storage_type == 'db':
    from models.engine.db_storage import DBStorage
    storage = DBStorage()
    storage.reload() # Reload the database
else:
    from models.engine.file_storage import FileStorage
    storage = FileStorage()
    storage.reload()