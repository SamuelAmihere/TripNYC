#!/usr/bin/env python3
"""This module loads data for the project"""
from utils.data import create_zone, create_vehicles


src_zones = '../TripNYC-resources/Data/nb_lat_lon_final.csv'

create_zone(src_zones)
create_vehicles()
