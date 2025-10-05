"""
API endpoints for ML model training and management
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from backend.ml_training.continuous_learner import get_learner, start_continuous_training
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False

ml_training_router = APIRouter(prefix="/api/v1/ml-training", tags=["ml_training"])


class ValidatedIncidentInput(BaseModel):
    """Input for storing a validated incident"""
    latitude: float
    longitude: float
    incident_type: str
    hour: int
    day_of_week: int
    neighborhood: str
    severity: str
    prediction_id: Optional[str] = None
    was_predicted: bool = False


class RetrainingRequest(BaseModel):
    """Request to retrain the model"""
    min_samples: int = 100
    force: bool = False


class RiskPredictionRequest(BaseModel):
    """Request for risk prediction"""
    latitude: float
    longitude: float
    hour: int
    day_of_week: int
    neighborhood: str


@ml_training_router.post("/store-incident")
async def store_validated_incident(incident: ValidatedIncidentInput):
    """
    Store a validated incident for future model training

    This endpoint receives incidents that have been validated against
    real-world data (e.g., from Polisen.se) and stores them for
    the next model retraining cycle.
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training system not available")

    try:
        learner = get_learner()
        learner.data_store.add_validated_incident({
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'type': incident.incident_type,
            'hour': incident.hour,
            'day_of_week': incident.day_of_week,
            'neighborhood': incident.neighborhood,
            'severity': incident.severity,
        })

        return {
            'status': 'success',
            'message': 'Incident stored for training',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.post("/retrain")
async def trigger_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger model retraining with accumulated data

    This can be called manually or automatically on a schedule.
    Training happens in the background to avoid blocking.
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training system not available")

    try:
        learner = get_learner()

        # Check if training is already in progress
        if learner.is_training:
            return {
                'status': 'already_training',
                'message': 'Model training is already in progress'
            }

        # Check training data availability
        stats = learner.data_store.get_training_stats()
        if stats['total_samples'] < request.min_samples and not request.force:
            return {
                'status': 'insufficient_data',
                'message': f"Need {request.min_samples} samples, have {stats['total_samples']}",
                'data_stats': stats
            }

        # Start training in background
        background_tasks.add_task(learner.retrain_model, request.min_samples)

        return {
            'status': 'training_started',
            'message': 'Model retraining started in background',
            'data_stats': stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.get("/status")
async def get_training_status():
    """
    Get current training status and statistics

    Returns information about:
    - Whether training is currently in progress
    - Last training time and results
    - Available training data
    - Model performance metrics
    """
    if not TRAINING_AVAILABLE:
        return {
            'available': False,
            'message': 'Training system not available (torch not installed)'
        }

    try:
        learner = get_learner()
        status = learner.get_training_status()

        return {
            'available': True,
            **status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.post("/predict-risk")
async def predict_risk(request: RiskPredictionRequest):
    """
    Get AI risk prediction for a specific location and time

    Uses the trained model to predict crime risk based on:
    - Location (latitude, longitude, neighborhood)
    - Time (hour of day, day of week)
    - Historical patterns learned from real incidents
    """
    if not TRAINING_AVAILABLE:
        return {
            'risk_score': 0.5,
            'confidence': 'low',
            'message': 'Using default prediction (training system not available)'
        }

    try:
        learner = get_learner()
        risk_score = await learner.predict_risk({
            'latitude': request.latitude,
            'longitude': request.longitude,
            'hour': request.hour,
            'day_of_week': request.day_of_week,
            'neighborhood': request.neighborhood,
        })

        # Convert to risk level
        if risk_score >= 0.8:
            risk_level = 'critical'
        elif risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'moderate'
        else:
            risk_level = 'low'

        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'confidence': 'high' if learner.last_training_time else 'medium',
            'last_training': learner.last_training_time.isoformat() if learner.last_training_time else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.post("/batch-store")
async def batch_store_incidents(incidents: List[ValidatedIncidentInput]):
    """
    Store multiple validated incidents at once

    Used when processing a batch of validation results from
    the prediction system.
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training system not available")

    try:
        learner = get_learner()
        stored_count = 0

        for incident in incidents:
            learner.data_store.add_validated_incident({
                'latitude': incident.latitude,
                'longitude': incident.longitude,
                'type': incident.incident_type,
                'hour': incident.hour,
                'day_of_week': incident.day_of_week,
                'neighborhood': incident.neighborhood,
                'severity': incident.severity,
            })
            stored_count += 1

        return {
            'status': 'success',
            'stored_count': stored_count,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.get("/training-history")
async def get_training_history():
    """Get historical training results"""
    if not TRAINING_AVAILABLE:
        return {'available': False}

    try:
        learner = get_learner()
        return {
            'available': True,
            'history': learner.training_history,
            'total_trainings': len(learner.training_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ml_training_router.post("/start-continuous")
async def start_continuous_training_endpoint(
    background_tasks: BackgroundTasks,
    interval_hours: int = 24
):
    """
    Start continuous training loop (runs every N hours)

    This should be called once at startup to enable automatic
    periodic retraining.
    """
    if not TRAINING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training system not available")

    try:
        background_tasks.add_task(start_continuous_training, interval_hours)

        return {
            'status': 'started',
            'message': f'Continuous training started (every {interval_hours} hours)',
            'interval_hours': interval_hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))