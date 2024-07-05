#!/usr/bin/env python
"""This module contains the data.py script"""

from models.base import classes
# from models.location.borough import Borough
import pandas as pd
from models.base import storage
# from models.vehicle.fhv import ForHireVehicle
# from models.vehicle.taxi import Taxi


Borough = classes['Borough']
Zone = classes['Zone']
Vehicle = classes['Vehicle']

def create_zone(filename):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print("---File not found---")
        return

    boroughs = df['Borough'].unique()
    for br in boroughs:
        # Create borough and its zones
        br_zn = df[df['Borough'] == br]
        
        brgh = storage.get_by(Borough, name=br)
        if not brgh:
            brgh = Borough(name=br)
            if brgh:
                # Create zones
                zone_ids = [] 
                for zn, lat, lon, locid, geo in \
                    zip(br_zn['Zone'],br_zn['lat'], br_zn['lon'],
                        br_zn['LocationID'], br_zn['geometry']):
                    
                    zone = storage.get_by(Zone, name=zn, locationID=brgh)
                    if not zone:
                        zone = Zone(name=zn, locationID=locid, latitude=lat, longitude=lon, borough_id=brgh.id)
                        zone.save()
                        zone_ids.append(zone.id)
                    else:
                        print(f"Zone {zone.id} already exists")
                brgh.zones = zone_ids
                brgh.save()


def create_vehicles():
    """Create vehicles"""
    veh = [{'name': 'yellow'},{'name':'green'},{'name':'fhv'}]
    for v in veh:
        vehicle = storage.get_by(Vehicle, name=v['name'])
        if not vehicle:
            vehicle = Vehicle(name=v['name'])
            vehicle.save()
            print(f"Created {vehicle.name} vehicle")
        else:
            print(f"Vehicle {vehicle.name} already exists")