"""
Atlas AI - AI Training API Endpoints
RESTful API endpoints for managing AI model training with real Swedish crime data
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pydantic import BaseModel

from ..auth.jwt_authentication import get_current_user, AuthenticatedUser
from ..ai_integration.data_training_pipeline import get_training_pipeline, execute_automated_training
from ..ai_models.advanced_prediction_engine import get_prediction_engine
from ..data_ingestion.polisen_data_collector import get_polisen_collector
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai-training", tags=["AI Training"])

# Request/Response Models
class TrainingRequest(BaseModel):
    cities: Optional[List[str]] = None
    days_back: int = 30
    retrain_existing: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "cities": ["Stockholm", "Göteborg", "Malmö"],
                "days_back": 30,
                "retrain_existing": True
            }
        }

class TrainingStatus(BaseModel):
    execution_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    total_incidents_processed: int
    models_trained: int
    errors_encountered: int
    processing_time_seconds: float

class ModelPerformance(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    trained_at: datetime

class RiskPredictionRequest(BaseModel):
    latitude: float
    longitude: float
    prediction_horizon: int = 24
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 59.3293,
                "longitude": 18.0686,
                "prediction_horizon": 24
            }
        }

@router.post("/start-training", response_model=Dict[str, str])
async def start_training_pipeline(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Start the AI model training pipeline with real Swedish crime data
    
    Requires law_enforcement or admin role
    """
    if current_user.role not in ["law_enforcement", "admin"]:
        raise HTTPException(
            status_code=403, 
            detail="Insufficient permissions. Requires law enforcement or admin access."
        )
    
    try:
        pipeline = await get_training_pipeline()
        
        if pipeline.is_running:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "conflict",
                    "message": "Training pipeline already running",
                    "execution_id": pipeline.current_execution_id
                }
            )
        
        # Start training in background
        background_tasks.add_task(
            execute_training_pipeline,
            request.cities,
            request.days_back,
            request.retrain_existing
        )
        
        # Track training requests
        training_counter = metrics.counter(
            "ai_training_requests", 
            "AI training requests", 
            ("user_role",)
        )
        training_counter.labels(current_user.role).inc()
        
        logger.info(f"🚀 Training pipeline started by {current_user.username}")
        
        return {
            "status": "started",
            "message": "AI training pipeline started with real Swedish crime data",
            "execution_id": f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting training pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/status", response_model=TrainingStatus)
async def get_training_status(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get current training pipeline status"""
    try:
        pipeline = await get_training_pipeline()
        
        if not pipeline.execution_stats:
            raise HTTPException(
                status_code=404, 
                detail="No training execution found"
            )
        
        stats = pipeline.execution_stats
        
        return TrainingStatus(
            execution_id=stats.execution_id,
            status=stats.status,
            started_at=stats.started_at,
            completed_at=stats.completed_at,
            total_incidents_processed=stats.total_incidents_processed,
            models_trained=stats.models_trained,
            errors_encountered=stats.errors_encountered,
            processing_time_seconds=stats.processing_time_seconds
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/model-performance", response_model=List[ModelPerformance])
async def get_model_performance(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get performance metrics for trained models"""
    try:
        pipeline = await get_training_pipeline()
        
        if not pipeline.execution_stats or not pipeline.execution_stats.training_results:
            return []
        
        performance_metrics = []
        for result in pipeline.execution_stats.training_results:
            performance_metrics.append(ModelPerformance(
                model_name=result.model_name,
                accuracy=result.accuracy,
                precision=result.precision,
                recall=result.recall,
                f1_score=result.f1_score,
                training_data_size=result.training_data_size,
                trained_at=result.trained_at
            ))
        
        return performance_metrics
        
    except Exception as e:
        logger.error(f"❌ Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")

@router.post("/predict-risk")
async def predict_location_risk(
    request: RiskPredictionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Predict crime risk for a specific location using trained AI models
    
    Uses real Swedish crime data for accurate temporal and geographic risk assessment
    """
    try:
        engine = await get_prediction_engine()
        
        # Get risk prediction
        prediction = await engine.predict_risk_for_location(
            latitude=request.latitude,
            longitude=request.longitude,
            prediction_horizon=request.prediction_horizon
        )
        
        # Track prediction requests
        pred_counter = metrics.counter(
            "risk_prediction_requests", 
            "Risk prediction requests", 
            ("risk_level", "user_role")
        )
        pred_counter.labels(prediction.risk_level, current_user.role).inc()
        
        return {
            "location": prediction.location,
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level,
            "confidence": prediction.confidence,
            "predicted_incidents": prediction.predicted_incidents,
            "contributing_factors": prediction.contributing_factors,
            "temporal_pattern": prediction.temporal_pattern,
            "prediction_horizon": prediction.prediction_horizon,
            "generated_at": prediction.generated_at.isoformat(),
            "expires_at": prediction.expires_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error predicting risk: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to predict risk: {str(e)}")

@router.get("/hotspots")
async def get_crime_hotspots(
    latitude: float = Query(..., description="Center latitude"),
    longitude: float = Query(..., description="Center longitude"), 
    radius_km: float = Query(5.0, description="Search radius in kilometers"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Detect crime hotspots in an area using real incident data and ML clustering
    """
    try:
        engine = await get_prediction_engine()
        
        # Detect hotspots
        hotspots = await engine.detect_hotspots(latitude, longitude, radius_km)
        
        hotspot_data = []
        for hotspot in hotspots:
            hotspot_data.append({
                "center_lat": hotspot.center_lat,
                "center_lng": hotspot.center_lng,
                "radius": hotspot.radius,
                "risk_score": hotspot.risk_score,
                "incident_density": hotspot.incident_density,
                "recent_incidents": hotspot.recent_incidents,
                "trend_direction": hotspot.trend_direction,
                "severity_distribution": hotspot.severity_distribution,
                "last_updated": hotspot.last_updated.isoformat()
            })
        
        # Track hotspot requests
        hotspot_counter = metrics.counter(
            "hotspot_detection_requests", 
            "Hotspot detection requests", 
            ("user_role",)
        )
        hotspot_counter.labels(current_user.role).inc()
        
        return {
            "center_location": {"lat": latitude, "lng": longitude},
            "radius_km": radius_km,
            "hotspots_found": len(hotspot_data),
            "hotspots": hotspot_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting hotspots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to detect hotspots: {str(e)}")

@router.post("/collect-training-data")
async def trigger_data_collection(
    background_tasks: BackgroundTasks,
    cities: Optional[List[str]] = Query(None, description="Cities to collect data from"),
    days_back: int = Query(7, description="Days of historical data to collect"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Trigger collection of real crime data from Swedish police API for training
    
    Requires admin role
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403, 
            detail="Insufficient permissions. Requires admin access."
        )
    
    try:
        # Start data collection in background
        background_tasks.add_task(
            collect_training_data_task,
            cities,
            days_back
        )
        
        logger.info(f"📡 Data collection started by {current_user.username}")
        
        return {
            "status": "started",
            "message": "Real crime data collection started from polisen.se",
            "cities": cities or "All major Swedish cities",
            "days_back": days_back
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting data collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start collection: {str(e)}")

@router.get("/data-collection-stats")
async def get_data_collection_stats(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get statistics about collected crime data"""
    try:
        collector = await get_polisen_collector()
        stats = await collector.get_collection_stats()
        
        return {
            "incidents_collected": stats.incidents_collected,
            "incidents_processed": stats.incidents_processed,
            "incidents_stored": stats.incidents_stored,
            "errors": stats.errors,
            "collection_start": stats.collection_start.isoformat(),
            "collection_end": stats.collection_end.isoformat() if stats.collection_end else None,
            "processing_time_seconds": stats.processing_time_seconds
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@router.post("/generate-user-alerts")
async def generate_user_risk_alerts(
    latitude: float = Query(..., description="User latitude"),
    longitude: float = Query(..., description="User longitude"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Generate personalized risk alerts for user's current location
    """
    try:
        engine = await get_prediction_engine()
        
        user_location = {"lat": latitude, "lng": longitude}
        alerts = await engine.generate_user_risk_alerts(current_user.user_id, user_location)
        
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "alert_type": alert.alert_type,
                "location": alert.location,
                "risk_score": alert.risk_score,
                "message": alert.message,
                "recommended_actions": alert.recommended_actions,
                "expiry_time": alert.expiry_time.isoformat(),
                "priority": alert.priority
            })
        
        # Track alert generation
        alert_counter = metrics.counter(
            "user_risk_alerts_generated", 
            "User risk alerts generated", 
            ("alert_count",)
        )
        alert_counter.labels(str(len(alert_data))).inc()
        
        return {
            "user_location": user_location,
            "alerts_generated": len(alert_data),
            "alerts": alert_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate alerts: {str(e)}")

# Background task functions
async def execute_training_pipeline(cities: List[str], days_back: int, retrain_existing: bool):
    """Execute training pipeline in background"""
    try:
        pipeline = await get_training_pipeline()
        results = await pipeline.execute_full_pipeline(
            cities=cities,
            days_back=days_back,
            retrain_existing=retrain_existing
        )
        logger.info(f"✅ Training pipeline completed: {results.execution_id}")
    except Exception as e:
        logger.error(f"❌ Background training failed: {e}")

async def collect_training_data_task(cities: List[str], days_back: int):
    """Collect training data in background"""
    try:
        collector = await get_polisen_collector()
        await collector.initialize()
        
        if cities:
            for city in cities:
                await collector.collect_and_store_batch(
                    location_name=city,
                    days_back=days_back,
                    max_events=1000
                )
        else:
            await collector.collect_major_cities_data(days_back=days_back)
            
        logger.info("✅ Data collection completed")
    except Exception as e:
        logger.error(f"❌ Background data collection failed: {e}")

# Health check endpoint
@router.get("/health")
async def training_health_check():
    """Health check for AI training services"""
    try:
        pipeline = await get_training_pipeline()
        engine = await get_prediction_engine()
        collector = await get_polisen_collector()
        
        return {
            "status": "healthy",
            "services": {
                "training_pipeline": "available",
                "prediction_engine": "available", 
                "data_collector": "available"
            },
            "pipeline_status": pipeline.execution_stats.status if pipeline.execution_stats else "idle",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )