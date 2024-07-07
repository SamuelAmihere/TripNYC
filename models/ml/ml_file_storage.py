#!/usr/bin/python3
"""
Contains the FileStorage class
"""

import os
from models.engine.file_storage import FileStorage

class MLFileStorage(FileStorage):
    """serializes instances to a JSON file & deserializes
    back to instances
    """
    # string - path to the JSON file
    base = "models/ml"
    __file_path = f"{base}/ml_file.json"
    __mlflow_metaData = None

    # dictionary - empty but will store all objects by <class name>.id
    __objects = {}

    if os.getenv('TRAIN_ENV') == 'dev': # Local development
        ... 