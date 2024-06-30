#!/usr/bin/python3
""" Zone API for TripNYC """

from flask import jsonify
from api.v1.views import app_views
from models.location.taxizone import TaxiZone


@app_views.route('/zones', methods=['GET'])
def get_zones():
    return {"status": "OK"}
    # zones = TaxiZone.query.all()
    # return jsonify([zone.to_dict() for zone in zones])

@app_views.route('/zones/<int:zone_id>', methods=['GET'])
def get_zone(zone_id):
    zone = TaxiZone.query.get_or_404(zone_id)
    return jsonify(zone.to_dict())

@app_views.route('/zones/borough/<borough>', methods=['GET'])
def get_zones_by_borough(borough):
    zones = TaxiZone.query.filter_by(borough=borough).all()
    return jsonify([zone.to_dict() for zone in zones])