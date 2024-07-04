from flask import jsonify
from api.v1.views import app_views
from models.location.zone import TaxiZone
from flask_restx import Api, Resource, fields


api = Api(app_views, version='1.0', title='TripNYC API',
          description='A simple TripNYC API',
          doc='/swagger',)
trips_ns = api.namespace('trips', description='Trip operations')

trip = api.model('Trip', {
    'id': fields.Integer,
    'pickup_datetime': fields.DateTime,
    'dropoff_datetime': fields.DateTime,
    'pickup_location': fields.String,
    'dropoff_location': fields.String,
    'passenger_count': fields.Integer,
    'trip_distance': fields.Float,
    'fare_amount': fields.Float,
    'tip_amount': fields.Float,
    'total_amount': fields.Float
})


@app_views.route('/trips', methods=['GET'])
def get_trips():
    return {"status": "OK"}
    # trips = Trip.query.all()
    # return jsonify([trip.to_dict() for trip in trips])
