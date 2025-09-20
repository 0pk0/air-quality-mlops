"""
FastAPI application for Air Quality Prediction System.

This module provides REST API endpoints for:
- Health checks and system status
- Real-time PM2.5 air quality predictions
- Model information and metadata
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
import sys
import logging
from datetime import datetime
import json
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# FastAPI app initialization
app = FastAPI(
    title="Air Quality Prediction API",
    description="ML-powered API for predicting PM2.5 air quality levels across global cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """
    Input model for air quality prediction requests.
    
    All environmental parameters are required for accurate predictions.
    """
    city: str = Field(..., description="City name (London, Tokyo, or Delhi)", example="London")
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=60, example=22.5)
    humidity: float = Field(..., description="Relative humidity percentage", ge=0, le=100, example=65.0)
    wind_speed: float = Field(..., description="Wind speed in m/s", ge=0, le=50, example=8.2)
    pressure: float = Field(1013.25, description="Atmospheric pressure in hPa", ge=800, le=1200, example=1013.25)
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "London",
                "temperature": 22.5,
                "humidity": 65.0,
                "wind_speed": 8.2,
                "pressure": 1013.25
            }
        }

class PredictionOutput(BaseModel):
    """
    Output model for air quality prediction responses.
    """
    city: str = Field(..., description="City name")
    predicted_pm25: float = Field(..., description="Predicted PM2.5 level in µg/m³")
    confidence_level: str = Field(..., description="Prediction confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")
    input_features: Dict[str, Any] = Field(..., description="Processed input features used for prediction")
    model_info: Dict[str, str] = Field(..., description="Model metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "city": "London",
                "predicted_pm25": 18.7,
                "confidence_level": "high",
                "timestamp": "2025-09-20T10:30:00",
                "input_features": {
                    "temperature": 22.5,
                    "humidity": 65.0,
                    "wind_speed": 8.2,
                    "processed_features_count": 15
                },
                "model_info": {
                    "model_type": "random_forest",
                    "version": "1.0.0"
                }
            }
        }

# Global variables for model caching
_model_cache = None
_model_metadata = None

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint providing basic API information.
    """
    return {
        "message": "Air Quality Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancing.
    
    Returns:
        dict: System health status including model availability
    """
    try:
        # Check if models directory exists
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        models_available = os.path.exists(models_dir) and len(glob.glob(os.path.join(models_dir, '*.pkl'))) > 0
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system": {
                "python_version": sys.version.split()[0],
                "api_version": "1.0.0"
            },
            "models": {
                "directory_exists": os.path.exists(models_dir),
                "models_available": models_available,
                "model_count": len(glob.glob(os.path.join(models_dir, '*.pkl'))) if os.path.exists(models_dir) else 0
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        dict: Model metadata and performance metrics
    """
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        if not os.path.exists(models_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Models directory not found"
            )
        
        # Find the most recent model file
        model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
        if not model_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No trained models found"
            )
        
        # Get the most recent model file
        latest_model = max(model_files, key=os.path.getctime)
        model_name = os.path.basename(latest_model)
        
        # Look for corresponding metadata file
        metadata_files = glob.glob(os.path.join(models_dir, '*_metadata.json'))
        model_metadata = {}
        
        if metadata_files:
            latest_metadata = max(metadata_files, key=os.path.getctime)
            try:
                with open(latest_metadata, 'r') as f:
                    model_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        return {
            "model_file": model_name,
            "model_path": latest_model,
            "created": datetime.fromtimestamp(os.path.getctime(latest_model)).isoformat(),
            "size_mb": round(os.path.getsize(latest_model) / (1024 * 1024), 2),
            "metadata": model_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )

def load_model():
    """
    Load the latest trained model into memory for predictions.
    
    Returns:
        tuple: (model_data, metadata) or (None, None) if loading fails
    """
    global _model_cache, _model_metadata
    
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        if not os.path.exists(models_dir):
            logger.error("Models directory not found")
            return None, None
        
        # Find the most recent model file
        model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
        if not model_files:
            logger.error("No trained models found")
            return None, None
        
        latest_model = max(model_files, key=os.path.getctime)
        
        # Load model if not cached or if file is newer
        if _model_cache is None or os.path.getctime(latest_model) > _model_cache.get('load_time', 0):
            import joblib
            
            model_data = joblib.load(latest_model)
            _model_cache = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'feature_columns': model_data['feature_columns'],
                'load_time': os.path.getctime(latest_model),
                'model_path': latest_model
            }
            
            # Load metadata if available
            metadata_files = glob.glob(os.path.join(models_dir, '*_metadata.json'))
            if metadata_files:
                latest_metadata = max(metadata_files, key=os.path.getctime)
                try:
                    with open(latest_metadata, 'r') as f:
                        _model_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
                    _model_metadata = {}
            
            logger.info(f"Model loaded: {os.path.basename(latest_model)}")
        
        return _model_cache, _model_metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

def prepare_single_prediction_features(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Prepare features for a single prediction request.
    
    Args:
        input_data: Validated input data from API request
        
    Returns:
        dict: Processed features ready for model prediction
    """
    import pandas as pd
    from datetime import datetime
    
    # City coordinates mapping (should match training data)
    city_coords = {
        'London': {'lat': 51.5074, 'lon': -0.1278},
        'Tokyo': {'lat': 35.6762, 'lon': 139.6503},
        'Delhi': {'lat': 28.6139, 'lon': 77.2090}
    }
    
    if input_data.city not in city_coords:
        raise ValueError(f"Unsupported city: {input_data.city}. Supported cities: {list(city_coords.keys())}")
    
    coords = city_coords[input_data.city]
    current_time = datetime.now()
    
    # Create a DataFrame with the input features
    df = pd.DataFrame({
        'city': [input_data.city],
        'temperature': [input_data.temperature],
        'pressure': [input_data.pressure],
        'humidity': [input_data.humidity],
        'wind_speed': [input_data.wind_speed],
        'latitude': [coords['lat']],
        'longitude': [coords['lon']],
        'day_of_week': [current_time.weekday()],
        'month': [current_time.month],
        'date': [current_time.strftime('%Y-%m-%d')],
        'timestamp': [current_time],
        # For single predictions, we'll use reasonable defaults for lag features
        'pm2_5': [20.0]  # Default value, will be processed out
    })
    
    # Add city encoding (one-hot)
    df = pd.get_dummies(df, columns=['city'], prefix='city')
    
    # Add lag features with default values for single predictions
    # In production, you might want to store recent predictions for better lag features
    df['pm2_5_lag1'] = 20.0  # Default recent PM2.5 value
    df['pm2_5_ma3'] = 20.0   # Default 3-day moving average
    df['pm2_5_ma7'] = 20.0   # Default 7-day moving average
    
    return df

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict_air_quality(input_data: PredictionInput):
    """
    Predict PM2.5 air quality level for a given city and environmental conditions.
    
    This endpoint uses the trained Random Forest model to predict air quality based on:
    - Temperature, humidity, wind speed, atmospheric pressure
    - City location (coordinates)
    - Temporal features (day of week, month)
    
    Args:
        input_data: Environmental parameters and city information
        
    Returns:
        PredictionOutput: PM2.5 prediction with confidence and metadata
    """
    try:
        # Load the model
        model_cache, metadata = load_model()
        if model_cache is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No trained model available. Please train a model first."
            )
        
        # Prepare features
        df_features = prepare_single_prediction_features(input_data)
        
        # Ensure all required features are present
        model_features = model_cache['feature_columns']
        missing_features = set(model_features) - set(df_features.columns)
        
        # Add missing city dummy variables with zeros
        for feature in missing_features:
            if feature.startswith('city_'):
                df_features[feature] = 0
        
        # Select only the features the model was trained on
        X = df_features[model_features]
        
        # Scale features
        X_scaled = model_cache['scaler'].transform(X)
        
        # Make prediction
        prediction = model_cache['model'].predict(X_scaled)[0]
        
        # Determine confidence level based on input validity and model performance
        confidence = "high"
        if metadata and 'metrics' in metadata:
            test_r2 = metadata['metrics'].get('test_r2', 0)
            if test_r2 < 0.7:
                confidence = "medium"
            elif test_r2 < 0.5:
                confidence = "low"
        
        # Prepare response
        response = PredictionOutput(
            city=input_data.city,
            predicted_pm25=round(float(prediction), 2),
            confidence_level=confidence,
            timestamp=datetime.now().isoformat(),
            input_features={
                "temperature": input_data.temperature,
                "humidity": input_data.humidity,
                "wind_speed": input_data.wind_speed,
                "pressure": input_data.pressure,
                "processed_features_count": len(model_features)
            },
            model_info={
                "model_type": metadata.get('model_type', 'unknown') if metadata else 'unknown',
                "version": "1.0.0",
                "features_used": str(len(model_features))
            }
        )
        
        logger.info(f"Prediction made for {input_data.city}: {prediction:.2f} µg/m³")
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
