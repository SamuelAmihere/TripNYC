#!/usr/bin/python3
""" TripNYC Application """
import uuid
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
from utils.data import create_zone
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
    cache_id = uuid.uuid4()
    src = '../TripNYC-resources/Data/nb_lat_lon_final.csv'
    # create_zone(src)

    boroughs = storage.all(Borough).values()
    zones = storage.all(Zone).values()

    # zones1 = {i.id: i.name for i in zones}
    
    # # get data by id
    # filter_name = lambda id: zones1.get(id, None)
    
    # boroughs_d = []
    # for borough in boroughs:
    #     borough = borough.to_dict()
    #     br = borough.copy()
    #     borough.update({'zones':
    #                     list(map(lambda x: filter_name(x), br.get('zones')))})
    #     boroughs_d.append(borough)
 
    return render_template('nyc_taxi_zones.html',
                           boroughs=boroughs,
                           zones=zones,
                           cache_id=cache_id,
                           total_trips=1600000)

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