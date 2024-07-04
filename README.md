# TripNYC PostgreSQL Schema

This document outlines the PostgreSQL schema for integrating machine learning (ML) components into the TripNYC taxi service system. The schema includes tables for managing ML models, storing features, logging predictions, tracking model performance, and handling various forecasts and anomalies.

## Visual Schema Representation

![PostgreSQL Schema](schema.png)

## Tables

### 1. Borough
This table stores information about different bouroughs in the city.

**Schema:**
```sql
CREATE TABLE Zone (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    state VARCHAR(255),
    country VARCHAR(255),
);
```

## Tables

### 2. Zone
This table stores information about different zones in the city.

**Schema:**
```sql
CREATE TABLE Zone (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    locationID INTEGER,
    latitude FLOAT,
    longitude FLOAT
    borough_id INTEGER REFERENCES Borough(id)
);
```

### 3. Taxi
This table stores information about taxis.

**Schema:**
```sql
CREATE TABLE Taxi (
    id SERIAL PRIMARY KEY,
    color VARCHAR(50)
);
```

### 4. ForHireVehicle
This table stores information about for-hire vehicles.
**Schema:**
```sql
CREATE TABLE ForHireVehicle (
    id SERIAL PRIMARY KEY,
    dispatching_base_number VARCHAR(50)
);
```

### 5. Trip
This table stores information about trips.

**Schema:**
```sql
CREATE TABLE Trip (
    id SERIAL PRIMARY KEY,
    pickup_datetime TIMESTAMP,
    dropoff_datetime TIMESTAMP,
    pickup_location INTEGER REFERENCES Zone(id),
    dropoff_location INTEGER REFERENCES Zone(id),
    trip_distance FLOAT,
    ml_estimated_duration FLOAT
);
```

### 5. TaxiTrip
This table stores information about taxi trips.

**Schema:**
```sql
CREATE TABLE TaxiTrip (
    id SERIAL PRIMARY KEY,
    vendorID INTEGER,
    passenger_count INTEGER,
    trip_distance FLOAT,
    taxi_id INTEGER REFERENCES Taxi(id),
    trip_id INTEGER REFERENCES Trip(id),
    ratecodeID INTEGER,
    payment_type VARCHAR(50),
    fare FLOAT,
    ml_estimated_fare FLOAT,
    tips FLOAT,
    tolls FLOAT,
    improvement_surcharge FLOAT,
    total_amount FLOAT,
    congestion_surcharge FLOAT,
    airport_fee FLOAT
);
```

### 6. FHVTrip
This table stores information about for-hire vehicle trips.

**Schema:**
```sql
CREATE TABLE FHVTrip (
    id SERIAL PRIMARY KEY,
    for_hire_vehicle_id INTEGER REFERENCES ForHireVehicle(id),
    trip_id INTEGER REFERENCES Trip(id),
    affiliated_base_number VARCHAR(50)
);
```

### 7. Borough
This table stores information about boroughs.
**Schema:**
```sql
CREATE TABLE Borough (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    zone_id INTEGER REFERENCES Zone(id)
);
```

### 8. HVFHSTrip
This table stores information about high-volume for-hire service trips.
**Schema:**
```sql
CREATE TABLE HVFHSTrip (
    id SERIAL PRIMARY KEY,
    for_hire_vehicle_id INTEGER REFERENCES ForHireVehicle(id),
    trip_id INTEGER REFERENCES Trip(id),
    affiliated_base_number VARCHAR(50),
    originating_base_num VARCHAR(50),
    request_datetime TIMESTAMP,
    on_scene_datetime TIMESTAMP,
    fare FLOAT,
    ml_estimated_fare FLOAT,
    tolls FLOAT,
    bcf FLOAT,
    sales_tax FLOAT,
    congestion_surcharge FLOAT,
    airport_fee FLOAT,
    tips FLOAT,
    driver_pay FLOAT
);
```

### 9. Prediction
This table links the demand, fare, and duration forecasts.
**Schema:**
```sql:
CREATE TABLE Prediction (
    id SERIAL PRIMARY KEY,
    model_id INTEGER,
    timestamp TIMESTAMP,
    predicted_duration FLOAT,
    predicted_result FLOAT
);
```

### 10. ABTestResults
This table stores results from A/B tests of different ML models or features.
**Schema:**
```sql:
CREATE TABLE ABTestResults (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES Prediction(id),
    test_name VARCHAR(255),
    variant VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value FLOAT,
    start_time TIMESTAMP,
    end_time TIMESTAMP
);
```

### 11. TripRating
This table stores ratings for trips.
**Schema:**
```sql:
CREATE TABLE TripRating (
    id SERIAL PRIMARY KEY,
    trip_id INTEGER REFERENCES Trip(id),
    rating INTEGER,
    comment TEXT
);
```



# TripNYC ML Components Schema

This section outlines the MongoDB schema for integrating machine learning (ML) components into the TripNYC taxi service system. The schema includes collections for managing ML models, storing features, logging predictions, tracking model performance, and handling various forecasts and anomalies.

## Collections

### 1. MLModelRegistry
This collection stores information about different machine learning models used in the system.

**Fields:**
- `id`: Unique identifier for the model.
- `model_name`: Name of the model.
- `version`: Version of the model.
- `description`: Description of the model.
- `created_at`: Timestamp when the model was created.
- `updated_at`: Timestamp when the model was last updated.
- `performance_metrics`: JSON object containing performance metrics of the model.
- `hyperparameters`: JSON object containing hyperparameters used for the model.
- `model_path`: Path to the model file.

**Example Document:**
```json
{
  "id": 1,
  "model_name": "FarePredictionModel",
  "version": "1.0",
  "description": "Model to predict taxi fare",
  "created_at": "2023-01-01T00:00:00Z",
  "updated_at": "2023-01-02T00:00:00Z",
  "performance_metrics": {
    "accuracy": 0.95,
    "precision": 0.92
  },
  "hyperparameters": {
    "learning_rate": 0.01,
    "epochs": 100
  },
  "model_path": "/models/fare_prediction_model_v1.pkl"
}
```

### 2. FeatureStore
This collection stores pre-computed features for various entities in the system.

**Fields:** Fields:

- id: Unique identifier for the feature.
- feature_name: Name of the feature.
- feature_value: Value of the feature.
- timestamp: Timestamp when the feature was computed.
- entity_id: Unique identifier of the entity the feature is associated with.
- entity_type: Type of the entity (e.g., "driver", "zone", "trip").
- Example Document:

**Example Document:**
```json
{
  "id": 1,
  "feature_name": "average_rating",
  "feature_value": 4.8,
  "timestamp": "2023-01-01T00:00:00Z",
  "entity_id": 123,
  "entity_type": "driver"
}
```

### 3. DataDrift
This collection tracks changes in data distribution over time, which can affect model performance.

**Fields:**:

- id: Unique identifier for the data drift record.
- feature_id: Reference to the feature in the FeatureStore.
- feature_name: Name of the feature.
- drift_score: Score indicating the amount of drift.
- timestamp: Timestamp when the drift was detected.
- Example Document:

**Example Document:**
```json
{
  "id": 1,
  "feature_id": 1,
  "feature_name": "average_rating",
  "drift_score": 0.05,
  "timestamp": "2023-01-01T00:00:00Z"
}
```
### 4. PredictionLog
This collection logs predictions made by ML models for analysis and monitoring.

**Fields:** :

- id: Unique identifier for the prediction log.
- model_id: Reference to the model in the MLModelRegistry.
- input_features: JSON object containing the input features used for the prediction.
- prediction: The predicted value.
- actual_value: The actual value (if available).
- timestamp: Timestamp when the prediction was made.

**Example Document:**
```json
{
  "id": 1,
  "model_id": 1,
  "input_features": {
    "feature1": 0.5,
    "feature2": 1.2
  },
  "prediction": 15.0,
  "actual_value": 14.5,
  "timestamp": "2023-01-01T00:00:00Z"
}
```

### 5. ModelExplanation
This collection stores feature importance for individual predictions, aiding in model explainability.

**Fields:**:
- id: Unique identifier for the model explanation.
- prediction_id: Reference to the prediction in the PredictionLog.
- feature_id: Reference to the feature in the FeatureStore.
- feature_name: Name of the feature.
- feature_importance: Importance score of the feature.

**Example Document:**
```json
{
  "id": 1,
  "prediction_id": 1,
  "feature_id": 1,
  "feature_name": "average_rating",
  "feature_importance": 0.8
}
```
### 6. ModelPerformance
This collection tracks the performance of models over time.

**Fields:**:

- id: Unique identifier for the model performance record.
- model_id: Reference to the model in the MLModelRegistry.
- metric_name: Name of the performance metric.
- metric_value: Value of the performance metric.
- timestamp: Timestamp when the performance was recorded.

**Example Document:**
```json
{
  "id": 1,
  "model_id": 1,
  "metric_name": "accuracy",
  "metric_value": 0.95,
  "timestamp": "2023-01-01T00:00:00Z"
}
```
### 7. ABTestResults
This collection stores results from A/B tests of different ML models or features.

**Fields:**:
- id: Unique identifier for the A/B test result.
- model_id: Reference to the model in the MLModelRegistry.
- test_name: Name of the A/B test.
- variant: Variant of the test (e.g., "A", "B").
- metric_name: Name of the metric being tested.
- metric_value: Value of the metric.
- start_time: Start time of the test.
- end_time: End time of the test.

**Example Document:**
```json
{
  "id": 1,
  "model_id": 1,
  "test_name": "FarePredictionTest",
  "variant": "A",
  "metric_name": "accuracy",
  "metric_value": 0.95,
  "start_time": "2023-01-01T00:00:00Z",
  "end_time": "2023-01-02T00:00:00Z"
}
```
### 8. DemandForecast
This collection stores predicted demand for different zones.

**Fields:**:
- id: Unique identifier for the demand forecast.
- zone_id: Reference to the zone.
- timestamp: Timestamp when the forecast was made.
- predicted_demand: Predicted demand value.

**Example Document:**
```json
{
  "id": 1,
  "zone_id": 1,
  "timestamp": "2023-01-01T00:00:00Z",
  "predicted_demand": 25.3
}
```
### 9. FareForecast
This collection stores predicted fares for different zones.

**Fields:**:
- id: Unique identifier for the fare forecast.
- zone_id: Reference to the zone.
- timestamp: Timestamp when the forecast was made.
- predicted_fare: Predicted fare value.

**Example Document:**
```json
{
  "id": 1,
  "zone_id": 1,
  "timestamp": "2023-01-01T00:00:00Z",
  "predicted_fare": 15.0
}
```
### 10. DurationForecast
This collection stores predicted trip durations for different zones.

**Fields:**:
- id: Unique identifier for the duration forecast.
- zone_id: Reference to the zone.
- timestamp: Timestamp when the forecast was made.
- predicted_duration: Predicted duration value.

**Example Document:**
```json
{
  "id": 1,
  "zone_id": 1,
  "timestamp": "2023-01-01T00:00:00Z",
  "predicted_duration": 18.5
}
```

### 11. TrafficForecast
This collection stores predicted traffic conditions for different zones.

**Fields:**:
- id: Unique identifier for the traffic forecast.
- zone_id: Reference to the zone.
- timestamp: Timestamp when the forecast was made.
- predicted_traffic: Predicted traffic value.

**Example Document:**
```json
{
  "id": 1,
  "zone_id": 1,
  "timestamp": "2023-01-01T00:00:00Z",
  "predicted_traffic": 3
}
```

### 12. AnomalyDetection
This collection stores anomalies detected by ML models in various aspects of the system.

**Fields:**:
- id: Unique identifier for the anomaly detection record.
- entity_id: Unique identifier of the entity where the anomaly was detected.
- entity_type: Type of the entity (e.g., "driver", "zone", "trip").
- anomaly_score: Score indicating the severity of the anomaly.
- timestamp: Timestamp when the anomaly was detected.
- description: Description of the anomaly.

**Example Document:**
```json
{
  "id": 1,
  "entity_id": 123,
  "entity_type": "driver",
  "anomaly_score": 0.9,
  "timestamp": "2023-01-01T00:00:00Z",
  "description": "Unusual trip pattern detected"
}
```
### Summary
This MongoDB schema is designed to support a comprehensive machine learning system for a taxi service. It includes collections for managing ML models, storing features, logging predictions, tracking model performance, and handling various forecasts and anomalies. This schema provides a flexible and scalable foundation for integrating machine learning into your taxi service operations.



This `README.md` file provides a detailed description of the MongoDB schema, including the purpose and structure of each collection, along with example documents. This should help you and your team understand and implement the schema effectively