"""
Atlas AI - Production API Endpoints
Unified endpoints for integrated Risk Assessment and AI Engine
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
import logging
import asyncio
import json

from ..risk_assessment.production_risk_engine import ProductionRiskAssessmentEngine, create_production_risk_engine
from ..ai_models.production_ai_engine import ProductionAIEngine, create_production_ai_engine
from ..auth.jwt_authentication import AuthenticationService, UserRole
from ..database.postgis_database import get_database
from .enhanced_models import (
    LocationRiskRequest, AreaRiskRequest, BatchLocationRequest,
    LocationRiskResponse, AreaRiskResponse, BatchLocationResponse,
    ModelTrainingRequest, ModelStatusResponse, SystemPerformanceResponse
)
from .documentation import create_endpoint_documentation, COMMON_RESPONSES
from .api_validation import RequestValidator, ValidationRules
from ..performance.optimization import (
    optimize_location_prediction, optimize_area_analysis, 
    optimize_batch_processing, get_performance_optimizer
)
from ..caching.cache_strategies import cache_invalidate_pattern

logger = logging.getLogger(__name__)

# Global engine instances
risk_engine: Optional[ProductionRiskAssessmentEngine] = None
ai_engine: Optional[ProductionAIEngine] = None
auth_service: Optional[AuthenticationService] = None

security = HTTPBearer()

# Pydantic models for request/response
class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    timestamp: Optional[datetime] = Field(None, description="Optional timestamp for prediction")

class AreaRequest(BaseModel):
    center_latitude: float = Field(..., ge=-90, le=90)
    center_longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0, le=10, description="Radius in kilometers (max 10)")
    grid_size: int = Field(20, ge=5, le=50, description="Grid resolution")
    timestamp: Optional[datetime] = None

class BatchLocationRequest(BaseModel):
    locations: List[LocationRequest] = Field(..., max_items=50, description="Max 50 locations per batch")
    timestamp: Optional[datetime] = None

class PredictionResponse(BaseModel):
    location: Dict[str, float]
    timestamp: str
    risk_assessment: Dict[str, Any]
    ai_prediction: Dict[str, Any]
    combined_score: float
    confidence: float
    processing_time_ms: float

class ModelTrainingRequest(BaseModel):
    force_retrain: bool = Field(False, description="Force retraining even if recently trained")
    training_period_days: int = Field(90, ge=7, le=365, description="Training data period")

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info"""
    global auth_service
    if not auth_service:
        try:
            from ..database.postgis_database import get_database
            db = await get_database()
            auth_service = AuthenticationService(db)
            await auth_service.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize authentication service: {e}")
            raise HTTPException(status_code=500, detail="Authentication service unavailable")
    
    try:
        user_data = auth_service.jwt_manager.verify_token(credentials.credentials)
        if not user_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Dependency for role-based access
def require_role(required_role: UserRole):
    """Create a dependency that requires a specific user role"""
    def role_checker(current_user: dict = Depends(get_current_user)):
        user_role_str = current_user.get('role', 'citizen')
        user_role = UserRole(user_role_str)
        
        # Define role hierarchy (higher number = more permissions)
        role_hierarchy = {
            UserRole.CITIZEN: 1,
            UserRole.LAW_ENFORCEMENT: 2, 
            UserRole.ADMIN: 3,
            UserRole.SYSTEM: 4
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 4)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=403, 
                detail=f"Insufficient permissions. Required: {required_role.value}, Current: {user_role.value}"
            )
        return current_user
    return role_checker

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["Production Atlas AI"])

# Startup and shutdown will be handled by main.py lifespan events
# These functions are available for manual initialization if needed

async def initialize_engines():
    """Initialize production engines"""
    global risk_engine, ai_engine
    
    logger.info("🚀 Initializing production Atlas AI engines...")
    
    try:
        # Initialize engines in parallel
        risk_task = create_production_risk_engine()
        ai_task = create_production_ai_engine()
        
        risk_engine, ai_engine = await asyncio.gather(risk_task, ai_task)
        
        logger.info("✅ Production engines initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize engines: {e}")
        raise

async def cleanup_engines():
    """Cleanup engines on shutdown"""
    global risk_engine, ai_engine
    
    if risk_engine:
        await risk_engine.cleanup()
    if ai_engine:
        await ai_engine.cleanup()
    
    logger.info("🛑 Production engines shut down")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    global risk_engine, ai_engine
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engines": {
            "risk_engine": risk_engine is not None,
            "ai_engine": ai_engine is not None
        }
    }
    
    if risk_engine and ai_engine:
        # Get model status
        try:
            model_status = await ai_engine.get_model_status()
            health_status["models_loaded"] = model_status["total_models_loaded"]
            health_status["neural_networks"] = len(model_status["neural_networks"])
            health_status["traditional_models"] = len(model_status["traditional_models"])
        except Exception as e:
            logger.warning(f"Failed to get model status: {e}")
            health_status["models_loaded"] = "unknown"
    
    return health_status

@router.post(
    "/predict/location", 
    response_model=LocationRiskResponse,
    **create_endpoint_documentation(
        summary="Location Risk Assessment",
        description="""
## Comprehensive Location Risk Assessment

Analyzes crime risk for a specific geographic location using:
- **Historical Crime Data**: Swedish police database integration
- **AI Prediction Models**: Machine learning algorithms trained on local data
- **Real-time Factors**: Current events, time of day, weather conditions
- **Geospatial Analysis**: Proximity to known risk factors

### Use Cases
- **Patrol Planning**: Optimize police resource allocation
- **Citizen Safety**: Personal safety assessment for travel planning
- **Emergency Response**: Quick risk evaluation for incident response
- **Urban Planning**: Crime prevention through environmental design

### Response Details
The API returns a comprehensive risk assessment including:
- Overall risk score (0.0 - 1.0)
- Risk category classification
- Contributing risk factors with weights
- Actionable recommendations
- Confidence score for the assessment

### Rate Limits
- Standard users: 60 requests/minute
- Law enforcement: 300 requests/minute
        """,
        tags=["Risk Assessment"],
        additional_responses={
            403: COMMON_RESPONSES[403],
            503: COMMON_RESPONSES[503]
        }
    )
)
async def predict_location_risk(
    request: LocationRiskRequest,
    current_user: dict = Depends(get_current_user)
):
    global risk_engine, ai_engine
    
    if not risk_engine or not ai_engine:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    # Use performance-optimized prediction with caching
    async def prediction_func(lat: float, lng: float, timestamp: Optional[datetime]):
        # Run both assessments in parallel
        risk_task = risk_engine.assess_risk(lat, lng, timestamp)
        ai_task = ai_engine.predict_crime_probability(lat, lng, timestamp)
        
        risk_result, ai_result = await asyncio.gather(risk_task, ai_task)
        
        # Calculate combined score (weighted average)
        risk_weight = 0.6
        ai_weight = 0.4
        combined_score = (risk_result.risk_level * risk_weight) + (ai_result.crime_probability * ai_weight)
        
        # Calculate combined confidence
        combined_confidence = (risk_result.confidence * risk_weight) + (ai_result.confidence_score * ai_weight)
        
        return {
            "location": {"lat": lat, "lon": lng},
            "timestamp": ai_result.timestamp.isoformat(),
            "risk_assessment": risk_result.to_dict(),
            "ai_prediction": ai_result.to_dict(),
            "combined_score": combined_score,
            "confidence": combined_confidence,
            "cached": False
        }
    
    start_time = datetime.now()
    
    try:
        # Use optimized caching
        result = await optimize_location_prediction(
            prediction_func,
            request.latitude,
            request.longitude,
            request.timestamp
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result["processing_time_ms"] = processing_time
        
        # Convert to response model
        response = LocationRiskResponse(
            status="success",
            message="Risk assessment completed successfully",
            data=result
        )
        
        logger.info(f"Location prediction completed in {processing_time:.1f}ms - Combined score: {result['combined_score']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Location prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post(
    "/predict/area",
    response_model=AreaRiskResponse,
    **create_endpoint_documentation(
        summary="Area Risk Assessment",
        description="""
## Area-Wide Risk Assessment

Analyzes crime risk across a geographic area using grid-based analysis:

### Features
- **Grid Analysis**: Divides area into analysis points for detailed mapping
- **Risk Hotspots**: Identifies high-risk zones within the area
- **Statistical Analysis**: Provides comprehensive area statistics
- **Scalable Resolution**: Configurable grid size for different use cases

### Use Cases
- **District Analysis**: Evaluate entire neighborhoods or districts
- **Event Planning**: Assess security needs for large events
- **Urban Planning**: Identify areas needing infrastructure improvements
- **Resource Allocation**: Optimize patrol routes and station locations

### Performance Notes
- Grid sizes over 30 may result in slower processing
- Areas over 5km radius will show performance warnings
- Maximum radius is 10km for optimal response times
        """,
        tags=["Risk Assessment"],
        additional_responses={
            403: COMMON_RESPONSES[403],
            503: COMMON_RESPONSES[503]
        }
    )
)
async def predict_area_risk(
    request: AreaRiskRequest,
    current_user: dict = Depends(get_current_user)
):
    global risk_engine, ai_engine
    
    if not risk_engine or not ai_engine:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    start_time = datetime.now()
    
    try:
        # Get area risk assessment
        area_risks = await risk_engine.assess_area_risk(
            request.center_latitude,
            request.center_longitude,
            request.radius_km,
            request.grid_size,
            request.timestamp
        )
        
        # Get area statistics
        area_stats = await risk_engine.get_risk_statistics(
            request.center_latitude,
            request.center_longitude,
            request.radius_km
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "area_center": {"lat": request.center_latitude, "lon": request.center_longitude},
            "radius_km": request.radius_km,
            "grid_size": request.grid_size,
            "timestamp": (request.timestamp or datetime.now(timezone.utc)).isoformat(),
            "total_points": len(area_risks),
            "risk_points": [risk.to_dict() for risk in area_risks],
            "statistics": area_stats,
            "processing_time_ms": processing_time
        }
        
        logger.info(f"Area prediction completed in {processing_time:.1f}ms - {len(area_risks)} points")
        
        return response
        
    except Exception as e:
        logger.error(f"Area prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Area prediction failed: {str(e)}")

@router.post(
    "/predict/batch",
    response_model=BatchLocationResponse,
    **create_endpoint_documentation(
        summary="Batch Location Risk Assessment", 
        description="""
## Batch Location Analysis

Efficiently analyze multiple locations in a single request using parallel processing:

### Features
- **Parallel Processing**: Analyze up to 50 locations simultaneously
- **Optimized Performance**: Shared engine initialization across locations
- **Consistent Results**: Same analysis quality as individual requests
- **Bulk Operations**: Ideal for route planning and area surveys

### Use Cases
- **Route Planning**: Analyze safety along travel routes
- **Property Assessment**: Evaluate multiple real estate locations
- **Patrol Optimization**: Assess multiple patrol checkpoints
- **Research Studies**: Bulk analysis for academic research

### Performance Benefits
- Up to 10x faster than individual requests for large batches
- Automatic load balancing across available resources
- Reduced overhead through connection pooling

### Limits
- Maximum 50 locations per batch
- Each location follows same validation rules as individual requests
        """,
        tags=["Risk Assessment"],
        additional_responses={
            403: COMMON_RESPONSES[403],
            503: COMMON_RESPONSES[503]
        }
    )
)
async def predict_batch_locations(
    request: BatchLocationRequest,
    current_user: dict = Depends(get_current_user)
):
    global risk_engine, ai_engine
    
    if not risk_engine or not ai_engine:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    start_time = datetime.now()
    
    try:
        # Prepare tasks for parallel processing
        risk_tasks = []
        ai_tasks = []
        
        for location in request.locations:
            risk_tasks.append(
                risk_engine.assess_risk(
                    location.latitude, 
                    location.longitude, 
                    location.timestamp or request.timestamp
                )
            )
            ai_tasks.append(
                ai_engine.predict_crime_probability(
                    location.latitude,
                    location.longitude,
                    location.timestamp or request.timestamp
                )
            )
        
        # Execute all predictions in parallel
        risk_results, ai_results = await asyncio.gather(
            asyncio.gather(*risk_tasks),
            asyncio.gather(*ai_tasks)
        )
        
        # Combine results
        predictions = []
        for i, (risk_result, ai_result) in enumerate(zip(risk_results, ai_results)):
            combined_score = (risk_result.risk_level * 0.6) + (ai_result.crime_probability * 0.4)
            combined_confidence = (risk_result.confidence * 0.6) + (ai_result.confidence_score * 0.4)
            
            predictions.append({
                "location": {"lat": request.locations[i].latitude, "lon": request.locations[i].longitude},
                "risk_assessment": risk_result.to_dict(),
                "ai_prediction": ai_result.to_dict(),
                "combined_score": combined_score,
                "confidence": combined_confidence
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "total_locations": len(request.locations),
            "predictions": predictions,
            "processing_time_ms": processing_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Batch prediction completed in {processing_time:.1f}ms - {len(predictions)} locations")
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@router.get("/predict/temporal/{lat}/{lon}")
async def predict_temporal_pattern(
    lat: float,
    lon: float,
    hours_ahead: int = Query(24, ge=1, le=168, description="Hours ahead to predict (max 1 week)"),
    current_user: dict = Depends(get_current_user)
):
    """Predict risk/crime probability over time for a location"""
    global risk_engine, ai_engine
    
    # Validate path parameters
    if not (-90 <= lat <= 90):
        raise HTTPException(status_code=422, detail="Latitude must be between -90 and 90")
    if not (-180 <= lon <= 180):
        raise HTTPException(status_code=422, detail="Longitude must be between -180 and 180")
    
    if not risk_engine or not ai_engine:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    start_time = datetime.now()
    
    try:
        # Get temporal patterns from both engines
        risk_trends = await risk_engine.predict_risk_trend(lat, lon, hours_ahead)
        ai_trends = await ai_engine.predict_temporal_pattern(lat, lon, hours_ahead)
        
        # Combine the results
        temporal_predictions = []
        for risk_trend, ai_trend in zip(risk_trends, ai_trends):
            combined_score = (risk_trend.risk_level * 0.6) + (ai_trend.crime_probability * 0.4)
            
            temporal_predictions.append({
                "timestamp": ai_trend.timestamp.isoformat(),
                "hours_from_now": int((ai_trend.timestamp - datetime.now(timezone.utc)).total_seconds() / 3600),
                "risk_level": risk_trend.risk_level,
                "crime_probability": ai_trend.crime_probability,
                "combined_score": combined_score,
                "risk_category": risk_trend.risk_category,
                "confidence": (risk_trend.confidence + ai_trend.confidence_score) / 2
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "location": {"lat": lat, "lon": lon},
            "prediction_horizon_hours": hours_ahead,
            "temporal_predictions": temporal_predictions,
            "processing_time_ms": processing_time
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Temporal prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Temporal prediction failed: {str(e)}")

@router.get("/models/status")
async def get_model_status(
    current_user: dict = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Get detailed model status and performance metrics"""
    global ai_engine
    
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    
    try:
        model_status = await ai_engine.get_model_status()
        return model_status
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.post("/models/retrain")
async def retrain_models(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_role(UserRole.ADMIN))
):
    """Trigger model retraining (admin only)"""
    global ai_engine
    
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    
    # Start retraining in background
    background_tasks.add_task(
        ai_engine.retrain_models, 
        force_retrain=request.force_retrain
    )
    
    return {
        "status": "training_started",
        "message": "Model retraining started in background",
        "force_retrain": request.force_retrain,
        "training_period_days": request.training_period_days,
        "started_at": datetime.now(timezone.utc).isoformat()
    }

@router.get("/models/validate")
async def validate_model_accuracy(
    test_period_days: int = Query(7, ge=1, le=30, description="Test period in days"),
    current_user: dict = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Validate model accuracy against real outcomes"""
    global ai_engine
    
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not initialized")
    
    try:
        validation_results = await ai_engine.validate_prediction_accuracy(test_period_days)
        return validation_results
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model validation failed: {str(e)}")

@router.get("/analytics/performance")
async def get_system_performance(
    current_user: dict = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Get system performance analytics"""
    global risk_engine, ai_engine
    
    if not risk_engine or not ai_engine:
        raise HTTPException(status_code=503, detail="Engines not initialized")
    
    try:
        # Get cached performance metrics
        performance_data = {}
        
        if hasattr(ai_engine, 'cache') and ai_engine.cache:
            # Get cached validation results
            validation_results = await ai_engine.cache.get("model_validation_results")
            if validation_results:
                try:
                    performance_data["model_validation"] = json.loads(validation_results) if isinstance(validation_results, str) else validation_results
                except (json.JSONDecodeError, TypeError):
                    performance_data["model_validation"] = validation_results
            
            # Get cache statistics
            try:
                cache_stats = await ai_engine.cache.get_cache_stats()
                performance_data["cache_performance"] = cache_stats
            except AttributeError:
                # Fallback for different cache implementations
                performance_data["cache_performance"] = {"status": "available"}
        
        # Get model status
        model_status = await ai_engine.get_model_status()
        performance_data["model_status"] = model_status
        
        performance_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Failed to get performance data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance data: {str(e)}")

# Export the router
__all__ = ["router"]