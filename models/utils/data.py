#!/usr/bin/env python
"""This module contains the data.py script"""

from models.location.zone import Zone
from models.location.borough import Borough
import pandas as pd
from models.base import storage


def create_zone(filename):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("---File not found---")
        return

    for br, zn, lat, lon, locid, geo in \
        zip(df['Borough'], df['Zone'],df['lat'], df['lon'],
            df['LocationID'], df['geometry']):
        
        borough = storage.get_by(Borough, name=br)
        if not borough:
            borough = Borough(name=br)
            if not borough:
                print(f"Error creating borough {br}")
                continue
        
        zone = storage.get_by(Zone, name=zn, locationID=locid)
        if not zone:
            zone = Zone(name=zn, locationID=locid, latitude=lat, longitude=lon, borough_id=borough.id)
            zone.save()
            borough.zones = zone.id
            borough.save()
        else:
            print(f"Zone {zone.id} already exists")