#!/usr/bin/python3
""" TripNYC Application """
from models.base import storage
from flask import Flask, render_template, make_response, jsonify
from flask_cors import CORS, cross_origin
from api.v1.views import app_views
import os
from flask import Flask, jsonify, request
from geoalchemy2 import Geometry
from shapely import wkb
import psycopg2
from dotenv import load_dotenv

from models.location.borough import Borough
from models.location.zone import Zone
from models.utils.data import create_zone
load_dotenv()


app = Flask(__name__)

host = os.getenv('TRIPNYC_POSTGRES_HOST', '0.0.0.0')
port = os.getenv('TRIPNYC_PORT', '5000')
app.register_blueprint(app_views)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={'/*': {'origins': host+':'+port}})


@app.route('/index',  methods=['GET'])
@app.route('/', methods=['GET'])
def status():
    """ Status of API """

    boroughs = storage.all(Borough)
    zones = storage.all(Zone)
    zones = [zone.to_dict() for zone in zones.values()]
    boroughs = [borough.to_dict() for borough in boroughs.values()]
    
    return jsonify([{"status": "OK"}, boroughs, zones[0:10]])

@app.teardown_appcontext
def teardown_db(exception):
    """
    after each request, this method calls .close() (i.e. .remove()) on
    the current SQLAlchemy Session
    """
    storage.close()

@app.errorhandler(404)
def not_found(error):
    """ 404 Error
    ---
    responses:
      404:
        description: a resource was not found
    """
    print(error)
    return make_response(jsonify({'error': "Not found"}), 404)

if __name__ == '__main__':
    app.debug = True
    app.run(host=host, port=port)