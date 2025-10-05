"""
ML Feedback Loop API Endpoints
RESTful API for prediction feedback and model retraining
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
from pydantic import BaseModel
import uuid

from ..ml.feedback_loop import (
    MLFeedbackSystem, PredictionResult, FeedbackData, 
    PredictionType, FeedbackType, AccuracyMetrics
)
from ..auth.jwt_authentication import require_role, UserRole
from ..caching.redis_cache import cache_api_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ml/feedback", tags=["ml-feedback"])

# Global feedback system instance
feedback_system: Optional[MLFeedbackSystem] = None

async def get_feedback_system() -> MLFeedbackSystem:
    """Get or initialize the global feedback system"""
    global feedback_system
    if feedback_system is None:
        feedback_system = MLFeedbackSystem()
        await feedback_system.initialize()
        logger.info("ML feedback system initialized")
    
    return feedback_system

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    model_name: str
    prediction_type: str
    predicted_value: Any
    confidence: float
    input_features: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback_type: str
    actual_value: Any
    comments: Optional[str] = None
    validation_source: Optional[str] = None

class ModelPerformanceResponse(BaseModel):
    model_name: str
    prediction_type: str
    performance_level: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    needs_retraining: bool

@router.post("/predictions")
async def record_prediction(request: PredictionRequest):
    """Record a prediction made by the system"""
    try:
        system = await get_feedback_system()
        
        # Validate prediction type
        try:
            pred_type = PredictionType(request.prediction_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid prediction type: {request.prediction_type}")
        
        # Create prediction result
        prediction = PredictionResult(
            prediction_id=str(uuid.uuid4()),
            model_name=request.model_name,
            prediction_type=pred_type,
            predicted_value=request.predicted_value,
            confidence=request.confidence,
            timestamp=datetime.now(),
            input_features=request.input_features,
            metadata=request.metadata or {}
        )
        
        success = await system.record_prediction(prediction)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record prediction")
        
        return {
            "success": True,
            "prediction_id": prediction.prediction_id,
            "model_name": request.model_name,
            "timestamp": prediction.timestamp.isoformat(),
            "flagged_for_review": prediction.confidence < system.uncertainty_threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction recording error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, current_user: dict = Depends(require_role(UserRole.LAW_ENFORCEMENT))):
    """Submit feedback on a prediction"""
    try:
        system = await get_feedback_system()
        
        # Validate feedback type
        try:
            feedback_type = FeedbackType(request.feedback_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid feedback type: {request.feedback_type}")
        
        # Create feedback data
        feedback = FeedbackData(
            feedback_id=str(uuid.uuid4()),
            prediction_id=request.prediction_id,
            feedback_type=feedback_type,
            actual_value=request.actual_value,
            feedback_timestamp=datetime.now(),
            user_id=current_user.get('user_id'),
            comments=request.comments,
            validation_source=request.validation_source
        )
        
        success = await system.submit_feedback(feedback)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process feedback")
        
        return {
            "success": True,
            "feedback_id": feedback.feedback_id,
            "prediction_id": request.prediction_id,
            "processed_at": feedback.feedback_timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback processing error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/performance")
@cache_api_response(ttl=600)  # Cache for 10 minutes
async def get_model_performance(model_name: Optional[str] = None):
    """Get model performance metrics and reports"""
    try:
        system = await get_feedback_system()
        report = await system.get_model_performance_report(model_name)
        
        return {
            "success": True,
            **report
        }
        
    except Exception as e:
        logger.error(f"Performance report error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance report")

@router.get("/performance/summary")
@cache_api_response(ttl=300)  # Cache for 5 minutes
async def get_performance_summary():
    """Get summary of all model performances"""
    try:
        system = await get_feedback_system()
        report = await system.get_model_performance_report()
        
        # Calculate overall statistics
        models = report.get('models', [])
        if models:
            avg_accuracy = sum(m['accuracy'] for m in models) / len(models)
            models_needing_retraining = sum(1 for m in models if m['needs_retraining'])
            
            summary = {
                "total_models": len(models),
                "average_accuracy": round(avg_accuracy, 3),
                "models_needing_retraining": models_needing_retraining,
                "best_performing_model": max(models, key=lambda x: x['accuracy'])['model_name'],
                "worst_performing_model": min(models, key=lambda x: x['accuracy'])['model_name']
            }
        else:
            summary = {
                "total_models": 0,
                "average_accuracy": 0.0,
                "models_needing_retraining": 0,
                "best_performing_model": None,
                "worst_performing_model": None
            }
        
        return {
            "success": True,
            "summary": summary,
            "generated_at": report['generated_at']
        }
        
    except Exception as e:
        logger.error(f"Performance summary error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance summary")

@router.get("/feedback/summary")
@cache_api_response(ttl=300)  # Cache for 5 minutes
async def get_feedback_summary(days: int = 7):
    """Get summary of recent feedback activity"""
    try:
        system = await get_feedback_system()
        summary = await system.get_feedback_summary(days=days)
        
        return {
            "success": True,
            **summary
        }
        
    except Exception as e:
        logger.error(f"Feedback summary error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback summary")

@router.post("/models/{model_name}/retrain")
async def trigger_model_retraining(model_name: str):
    """Manually trigger model retraining"""
    try:
        system = await get_feedback_system()
        
        # Check if model exists
        if model_name not in system.model_metrics:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Check if sufficient training data exists
        has_data = await system._has_sufficient_training_data(model_name)
        if not has_data:
            raise HTTPException(status_code=400, detail="Insufficient training data for retraining")
        
        # Schedule retraining
        await system._schedule_retraining(model_name)
        
        return {
            "success": True,
            "model_name": model_name,
            "scheduled_at": datetime.now().isoformat(),
            "next_training": system.retraining_schedule.get(model_name, datetime.now()).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retraining trigger error: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

@router.get("/training-data")
async def export_training_data(model_name: Optional[str] = None):
    """Export training data for analysis"""
    try:
        system = await get_feedback_system()
        data = await system.export_training_data(model_name)
        
        return {
            "success": True,
            **data
        }
        
    except Exception as e:
        logger.error(f"Training data export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export training data")

@router.get("/status")
async def get_feedback_system_status():
    """Get current feedback system status"""
    try:
        system = await get_feedback_system()
        status = system.get_system_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")

@router.post("/simulate")
async def simulate_ml_feedback_loop(
    prediction_count: int = 100,
    feedback_rate: float = 0.7,
    accuracy_rate: float = 0.85
):
    """Simulate ML feedback loop for testing and demonstration"""
    try:
        system = await get_feedback_system()
        
        import numpy as np
        import random
        
        # Generate simulated predictions
        model_names = ["hotspot_predictor", "incident_classifier", "threat_assessor"]
        prediction_types = [PredictionType.CRIME_HOTSPOT, PredictionType.INCIDENT_CLASSIFICATION, PredictionType.THREAT_ASSESSMENT]
        
        predictions_created = 0
        feedback_submitted = 0
        
        for i in range(prediction_count):
            # Create random prediction
            model_name = random.choice(model_names)
            pred_type = random.choice(prediction_types)
            
            # Generate realistic prediction data
            if pred_type == PredictionType.CRIME_HOTSPOT:
                predicted_value = {
                    "lat": 59.3293 + np.random.normal(0, 0.01),
                    "lng": 18.0686 + np.random.normal(0, 0.01),
                    "risk_score": np.random.uniform(0.5, 1.0)
                }
            elif pred_type == PredictionType.INCIDENT_CLASSIFICATION:
                predicted_value = random.choice(["assault", "theft", "vandalism", "drug_offense"])
            else:  # THREAT_ASSESSMENT
                predicted_value = random.choice(["low", "medium", "high", "critical"])
            
            prediction = PredictionResult(
                prediction_id=str(uuid.uuid4()),
                model_name=model_name,
                prediction_type=pred_type,
                predicted_value=predicted_value,
                confidence=np.random.uniform(0.5, 0.95),
                timestamp=datetime.now(),
                input_features={"feature_1": np.random.rand(), "feature_2": np.random.rand()},
                metadata={"simulation": True}
            )
            
            await system.record_prediction(prediction)
            predictions_created += 1
            
            # Randomly generate feedback
            if np.random.random() < feedback_rate:
                # Generate actual value (simulate ground truth)
                if np.random.random() < accuracy_rate:
                    # Prediction was correct
                    actual_value = predicted_value
                else:
                    # Prediction was wrong
                    if pred_type == PredictionType.CRIME_HOTSPOT:
                        actual_value = {
                            "lat": 59.3293 + np.random.normal(0, 0.02),
                            "lng": 18.0686 + np.random.normal(0, 0.02),
                            "risk_score": np.random.uniform(0.0, 0.5)
                        }
                    elif pred_type == PredictionType.INCIDENT_CLASSIFICATION:
                        wrong_classes = ["assault", "theft", "vandalism", "drug_offense"]
                        wrong_classes.remove(predicted_value)
                        actual_value = random.choice(wrong_classes)
                    else:
                        wrong_levels = ["low", "medium", "high", "critical"]
                        wrong_levels.remove(predicted_value)
                        actual_value = random.choice(wrong_levels)
                
                feedback = FeedbackData(
                    feedback_id=str(uuid.uuid4()),
                    prediction_id=prediction.prediction_id,
                    feedback_type=random.choice([FeedbackType.GROUND_TRUTH, FeedbackType.USER_CORRECTION]),
                    actual_value=actual_value,
                    feedback_timestamp=datetime.now(),
                    user_id="simulation_user",
                    comments="Simulated feedback",
                    validation_source="simulation"
                )
                
                await system.submit_feedback(feedback)
                feedback_submitted += 1
        
        # Get updated performance report
        performance_report = await system.get_model_performance_report()
        
        return {
            "success": True,
            "simulation_completed": True,
            "predictions_created": predictions_created,
            "feedback_submitted": feedback_submitted,
            "feedback_rate": feedback_rate,
            "expected_accuracy": accuracy_rate,
            "performance_report": performance_report,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to run ML feedback simulation")


# Background task to check for scheduled retraining
async def background_retraining_monitor():
    """Background task to monitor and execute scheduled retraining"""
    while True:
        try:
            system = await get_feedback_system()
            
            # Check for scheduled retraining
            current_time = datetime.now()
            models_to_retrain = [
                model for model, scheduled_time in system.retraining_schedule.items()
                if scheduled_time <= current_time
            ]
            
            for model_name in models_to_retrain:
                logger.info(f"Starting scheduled retraining for {model_name}")
                await system._perform_retraining(model_name)
            
        except Exception as e:
            logger.error(f"Background retraining monitor error: {e}")
        
        # Check every 30 minutes
        await asyncio.sleep(1800)

# Background monitor will be started in main.py startup event
# asyncio.create_task(background_retraining_monitor())