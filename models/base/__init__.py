#!/usr/bin/python3
"""create a unique FileStorage instance for your application"""
import os
from dotenv import load_dotenv
load_dotenv()

storage_type = os.getenv('TRIPNYC_TYPE_STORAGE')
if storage_type == 'db':
    from models.engine.db_storage import DBStorage
    storage = DBStorage()
    storage.reload() # Reload the database
else:
    from models.engine.file_storage import FileStorage
    storage = FileStorage()
    storage.reload()