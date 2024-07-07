#!/usr/bin/python3
""" TripNYC Application """
from models.base import storage, storage_type
from flask import Flask, render_template, make_response, jsonify
from utils.svg import svgmap
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
def home():
    """ Status of API """

    src = '../TripNYC-resources/Data/nb_lat_lon_final.csv'
    # create_zone(src)

    boroughs = storage.all(Borough).values()
    zones = storage.all(Zone).values()
    zones1 = {i.id: i.name for i in zones}
    
    # get data by id
    filter_name = lambda id: zones1.get(id, None)
    
    boroughs_d = []
    

    for borough in boroughs:
        borough = borough.to_dict()
        br = borough.copy()
        if storage_type == 'db':
            borough.update({'zones':
                        list(map(lambda x: filter_name(x), zones1))})
            boroughs_d.append(borough)
            
            
        else:
            borough.update({'zones':
                            list(map(lambda x: filter_name(x), br.get('zones')))})
            boroughs_d.append(borough)
    html1 = 'nyc_taxi_zones.html'
    html2 = 'index.html'
    return render_template(html2, bzones=boroughs_d, nyc_map=svgmap.get('nyc'))

@app.route('/taxi_zone_lookup', methods=['GET'])
def taxi_zone_lookup():
    """ get taxi zone by id """
    print('--------------------')
    boroughs = storage.all(Borough).values()
    zones = storage.all(Zone).values()
    
    zones1 = {i.id: i.name for i in zones}
    zones2 = {i.id: [i.to_dict().get('name'), i.to_dict().get('locationID')] for i in zones}
    # print(zones2.values())
    
    # get data by id
    filter_name = lambda id: zones1.get(id, None)
    
    boroughs_d = []
    

    for borough in boroughs:
        borough = borough.to_dict()
        br = borough.copy()
        if storage_type == 'db':
            borough.update({'zones':
                        list(map(lambda x: filter_name(x), zones1))})
            boroughs_d.append(borough)
        else:
            borough.update({'zones':
                            list(map(lambda x: filter_name(x), br.get('zones')))})
            boroughs_d.append(borough)
    zones = []
    for z in zones:
        z = z.to_dict()
        # z.update({'borough': filter_name(z.get('borough'))})
        # print('--------------------')
        # print(z.keys())
        zones.append(z.to_dict())
    
    return jsonify([zones2])

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