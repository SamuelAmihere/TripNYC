#!/usr/bin/env python3


from models.base.base_model import BaseModel, Base
from sqlalchemy import Column, Integer, String, DateTime


class Zone(BaseModel, Base):
    __tablename__ = 'zones'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)

    def __init__(self, *args, **kwargs):
        """Initializes an instance of the BaseModel class"""
        if kwargs:
            kwargs.pop('__class__', None)
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __str__(self):
        """Returns a string representation of the instance"""
        return "[{}] ({}) {}".format(self.__class__.__name__, self.id,
                                     self.__dict__)

m = BaseModel()
m.save()
print(m)

'TAXI'
'VendorID'	'tpep_pickup_datetime'	'tpep_dropoff_datetime'	'passenger_count'	'trip_distance'	'RatecodeID'	'store_and_fwd_flag'	'PULocationID'	'DOLocationID'	'payment_type'	'fare_amount'	'extra'	'mta_tax'	'tip_amount'	'tolls_amount'	'improvement_surcharge'	'total_amount'	'congestion_surcharge'	'Airport_fee'



'FHV'
'dispatching_base_num'	'pickup_datetime'	'dropOff_datetime'	'PUlocationID'	'DOlocationID'	'SR_Flag'	'Affiliated_base_number'

'HVFHS'
'hvfhs_license_num'	'dispatching_base_num'	'originating_base_num'	'request_datetime'	'on_scene_datetime'	'pickup_datetime'	'dropoff_datetime'	'PULocationID'	'DOLocationID'	'trip_miles'	'trip_time'	'base_passenger_fare'	'tolls'	'bcf'	'sales_tax'	'congestion_surcharge'	'airport_fee'	'tips'	'driver_pay'	'shared_request_flag'	'shared_match_flag'	'access_a_ride_flag'	'wav_request_flag'	'wav_match_flag'
