#!/usr/bin/env python
"""This module contains the data.py script"""

from models.base import classes
# from models.location.borough import Borough
import pandas as pd
from models.base import storage
from shapely import wkt
from shapely.geometry import LineString, Polygon
from geoalchemy2.shape import from_shape
from models.base import classes, storage_type


Borough = classes['Borough']
Zone = classes['Zone']
Vehicle = classes['Vehicle']

# Convert LINESTRING to POLYGON
def linestring_to_polygon(geometry):
    if isinstance(geometry, LineString):
        # Ensure the LINESTRING is closed
        if not geometry.is_closed:
            geometry = LineString(list(geometry.coords) + [geometry.coords[0]])
        return Polygon(geometry)
    return geometry



def create_zone(filename):
    try:
        df = pd.read_csv(filename)

        # print(f"==========Geometry Datatype {df.info()} =======")
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
            brgh.save()
        else:
            brgh = brgh[0]
            print(f"Borough {brgh.name} already exists")
        if brgh:
            # Create zones
            zone_ids = [] 
            for zn, lat, lon, locid, ln, geo in \
                zip(br_zn['Zone'],br_zn['lat'], br_zn['lon'],
                    br_zn['LocationID'], br_zn['geometry_line'], br_zn['coordinates']):
                
                zone = storage.get_by(Zone, name=zn, locationID=locid, latitude=lat, longitude=lon, line=ln, geometry=geo)
                if not zone:
                    zone = Zone(name=zn, locationID=locid, latitude=lat,
                                longitude=lon, borough_id=brgh.id,
                                line=ln,
                                geometry=geo
                                )
                    if zone:
                        # print(f"--------zone: {zone.id} created")
                        zone.geometry = zone.geometry
                        zone.save()
                        # print(f"----------------zone: {zone.id} ")
                        zone_ids.append(zone.id)
                    else:
                        print(f"--------Failed to create zone")
                        
                else:
                    print(f"Zone {zone[0].id if isinstance(zone, list) else zone.id} already exists")
                    # update the zone with all the data
                    zone = zone[0] if isinstance(zone, list) else zone
                    zone.line = ln
                    zone.geometry = geo
                    zone.save()

            print(f"======Created {brgh.name} borough=======")
            
            if storage_type == 'file':
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
            print(f"Created {vehicle[0].name} vehicle")
        else:
            print(f"Vehicle {vehicle[0].name} already exists")