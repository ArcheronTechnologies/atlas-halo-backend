"""
Enhanced Atlas API Server with Redis Caching
Production-ready FastAPI server with comprehensive caching layer
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import Atlas AI components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from caching.redis_cache import AtlasRedisCache, CacheConfig, cache_api_response
from auth.jwt_authentication import AuthenticationService, get_current_user
from analytics.real_feature_engineering import RealFeatureEngineeringSystem, FeatureConfig

logger = logging.getLogger(__name__)

# Pydantic models
class LocationRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(2.0, ge=0.1, le=50)
    time_filter: str = Field("24h", pattern=r"^(1h|6h|12h|24h|48h|72h)$")

class RiskRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    target_time: Optional[datetime] = None

class AreaRiskRequest(BaseModel):
    center_lat: float = Field(..., ge=-90, le=90)
    center_lng: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(2.0, ge=0.1, le=10)
    grid_size: int = Field(20, ge=5, le=50)

class UserLocationUpdate(BaseModel):
    user_id: str
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None

class EnhancedIncidentResponse(BaseModel):
    """Enhanced incident response with cached features"""
    id: str
    incident_type: str
    location: Dict[str, float]
    timestamp: datetime
    severity: int
    enhanced_features: Optional[Dict[str, Any]] = None
    cached: bool = False

# Global systems
redis_cache: Optional[AtlasRedisCache] = None
auth_service: Optional[AuthenticationService] = None
feature_engineering: Optional[RealFeatureEngineeringSystem] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with Redis cache initialization"""
    global redis_cache, auth_service, feature_engineering
    
    logger.info("🚀 Starting Enhanced Atlas API Server with Redis Cache...")
    
    try:
        # Initialize Redis cache
        cache_config = CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            cache_enabled=True,
            session_ttl=3600,
            api_response_ttl=300
        )
        
        redis_cache = AtlasRedisCache(cache_config)
        cache_initialized = await redis_cache.initialize()
        
        if cache_initialized:
            logger.info("✅ Redis cache system initialized successfully")
        else:
            logger.warning("⚠️ Redis cache initialization failed, running without cache")
        
        # Initialize authentication service
        try:
            from ..database.postgis_database import get_database
            db = await get_database()
            auth_service = AuthenticationService(db)
            await auth_service.initialize()
            logger.info("✅ Authentication service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize authentication service: {e}")
            auth_service = None
        
        # Initialize feature engineering
        feature_config = FeatureConfig()
        feature_engineering = RealFeatureEngineeringSystem(feature_config)
        await feature_engineering.__aenter__()
        logger.info("✅ Feature engineering system initialized")
        
        # Start background tasks
        background_task = asyncio.create_task(background_maintenance())
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🔄 Shutting down Enhanced Atlas API Server...")
        
        background_task.cancel()
        
        if feature_engineering:
            await feature_engineering.__aexit__(None, None, None)
        
        if redis_cache:
            await redis_cache.close()
        
        logger.info("✅ Shutdown complete")

async def background_maintenance():
    """Background task for cache maintenance and health checks"""
    while True:
        try:
            if redis_cache and redis_cache.sessions:
                # Clean up expired sessions every 10 minutes
                active_sessions = await redis_cache.sessions.cleanup_expired_sessions()
                logger.debug(f"Active sessions: {active_sessions}")
                
                # Perform health check
                health = await redis_cache.health_check()
                if health['status'] != 'healthy':
                    logger.warning(f"Redis cache health issue: {health}")
            
        except Exception as e:
            logger.error(f"Background maintenance error: {e}")
        
        # Wait 10 minutes before next maintenance cycle
        await asyncio.sleep(10 * 60)

# Create FastAPI app
app = FastAPI(
    title="Atlas AI - Enhanced API",
    description="Production-ready Atlas AI API with Redis caching and advanced features",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get cache instance
async def get_cache() -> AtlasRedisCache:
    """Dependency to get Redis cache instance"""
    if redis_cache is None:
        raise HTTPException(
            status_code=503, 
            detail="Cache system not available"
        )
    return redis_cache

# Health check endpoints
@app.get("/")
async def root():
    """Enhanced health check with cache status"""
    cache_status = "disabled"
    if redis_cache:
        health = await redis_cache.health_check()
        cache_status = health.get('status', 'unknown')
    
    return {
        "status": "ok",
        "service": "Atlas AI Enhanced API",
        "version": "2.1.0",
        "cache_status": cache_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def detailed_health_check(cache: AtlasRedisCache = Depends(get_cache)):
    """Detailed health check including cache metrics"""
    cache_health = await cache.health_check()
    
    return {
        "api_status": "healthy",
        "cache_health": cache_health,
        "feature_engineering": "available" if feature_engineering else "unavailable",
        "authentication": "available" if auth_service else "unavailable",
        "timestamp": datetime.now().isoformat()
    }

# Enhanced API endpoints with caching
@app.post("/api/risk")
@cache_api_response(ttl=180)  # Cache for 3 minutes
async def get_risk_assessment(
    request: RiskRequest,
    cache: AtlasRedisCache = Depends(get_cache)
):
    """Get risk assessment with enhanced features and caching"""
    try:
        # Check cache first for risk assessment
        cache_key_params = {
            "lat": request.lat,
            "lng": request.lng,
            "target_time": request.target_time.isoformat() if request.target_time else None
        }
        
        # Generate enhanced features if available
        enhanced_features = {}
        if feature_engineering:
            try:
                features = await feature_engineering.generate_features(
                    request.lat, 
                    request.lng, 
                    request.target_time or datetime.now()
                )
                enhanced_features = features.to_dict()
                logger.debug("Enhanced features generated for risk assessment")
            except Exception as e:
                logger.warning(f"Feature generation failed: {e}")
        
        # Mock risk calculation (would use real risk engine)
        risk_score = calculate_mock_risk(request.lat, request.lng, enhanced_features)
        
        response = {
            "coordinates": {"lat": request.lat, "lng": request.lng},
            "risk_level": risk_score["risk_level"],
            "risk_category": risk_score["risk_category"],
            "confidence": risk_score["confidence"],
            "enhanced_features": enhanced_features,
            "calculation_time": datetime.now().isoformat(),
            "cached": False
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        raise HTTPException(status_code=500, detail="Risk assessment failed")

@app.post("/api/incidents")
@cache_api_response(ttl=300)  # Cache for 5 minutes
async def get_incidents(
    request: LocationRequest,
    cache: AtlasRedisCache = Depends(get_cache)
):
    """Get crime incidents with enhanced features and caching"""
    try:
        # Mock incident data (would fetch from real data sources)
        incidents = generate_mock_incidents(request.lat, request.lng, request.radius_km)
        
        # Enhance incidents with real features if available
        enhanced_incidents = []
        for incident in incidents:
            enhanced_incident = incident.copy()
            
            if feature_engineering:
                try:
                    features = await feature_engineering.generate_features(
                        incident["location"]["lat"],
                        incident["location"]["lng"],
                        datetime.fromisoformat(incident["timestamp"])
                    )
                    enhanced_incident["enhanced_features"] = features.to_dict()
                except Exception as e:
                    logger.warning(f"Feature enhancement failed for incident: {e}")
            
            enhanced_incidents.append(enhanced_incident)
        
        return {
            "incidents": enhanced_incidents,
            "count": len(enhanced_incidents),
            "area": {
                "center": {"lat": request.lat, "lng": request.lng},
                "radius_km": request.radius_km
            },
            "time_filter": request.time_filter,
            "retrieved_at": datetime.now().isoformat(),
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"Error fetching incidents: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch incidents")

# Session management endpoints
@app.post("/api/auth/login")
async def login(
    credentials: Dict[str, str],
    cache: AtlasRedisCache = Depends(get_cache)
):
    """Enhanced login with Redis session management"""
    try:
        if not auth_service:
            raise HTTPException(status_code=503, detail="Authentication service unavailable")
        
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required")
        
        # Authenticate user
        login_result = await auth_service.login(username, password)
        
        if "access_token" not in login_result:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Store session in Redis cache
        if cache.sessions:
            token_data = auth_service.jwt_manager.decode_token(login_result["access_token"])
            await cache.sessions.store_session(
                token_data["jti"],
                {
                    "user_id": login_result["user"]["user_id"],
                    "username": login_result["user"]["username"],
                    "role": login_result["user"]["role"],
                    "login_time": datetime.now().isoformat()
                }
            )
        
        return {
            **login_result,
            "session_cached": cache.sessions is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/api/auth/logout")
async def logout(
    token_data: Dict[str, str],
    cache: AtlasRedisCache = Depends(get_cache)
):
    """Enhanced logout with session cleanup"""
    try:
        token = token_data.get("access_token")
        if not token:
            raise HTTPException(status_code=400, detail="Access token required")
        
        # Decode token to get JTI
        if auth_service:
            decoded = auth_service.jwt_manager.decode_token(token)
            jti = decoded.get("jti")
            
            # Remove session from cache
            if cache.sessions and jti:
                await cache.sessions.invalidate_session(jti)
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

# Location tracking with caching
@app.post("/api/location/update")
async def update_user_location(
    update: UserLocationUpdate,
    cache: AtlasRedisCache = Depends(get_cache),
    current_user: Dict = Depends(get_current_user)
):
    """Update user location with Redis caching"""
    try:
        # Verify user permission
        if current_user["user_id"] != update.user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Update location in cache
        if cache.locations:
            success = await cache.locations.update_user_location(
                update.user_id,
                update.lat,
                update.lng,
                update.accuracy
            )
            
            if not success:
                logger.warning(f"Failed to cache location for user {update.user_id}")
        
        # Send location-based alerts if needed
        if cache.alerts:
            await send_location_based_alerts(update, cache)
        
        return {
            "status": "success",
            "user_id": update.user_id,
            "coordinates": {"lat": update.lat, "lng": update.lng},
            "cached": cache.locations is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location update error: {e}")
        raise HTTPException(status_code=500, detail="Location update failed")

@app.get("/api/location/{user_id}")
async def get_user_location(
    user_id: str,
    cache: AtlasRedisCache = Depends(get_cache),
    current_user: Dict = Depends(get_current_user)
):
    """Get user's last known location from cache"""
    try:
        # Check permission
        if current_user["user_id"] != user_id and current_user["role"] not in ["admin", "law_enforcement"]:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        # Get location from cache
        location = None
        if cache.locations:
            location = await cache.locations.get_user_location(user_id)
        
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        
        return {
            "user_id": user_id,
            "location": location,
            "from_cache": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get location error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get location")

# WebSocket endpoint with Redis pub/sub
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint with Redis-based real-time alerts"""
    await websocket.accept()
    
    pubsub = None
    try:
        # Subscribe to user alerts via Redis
        if redis_cache and redis_cache.alerts:
            pubsub = await redis_cache.alerts.subscribe_user_alerts(user_id)
            logger.info(f"WebSocket connected for user {user_id} with Redis alerts")
        
        while True:
            try:
                # Handle incoming WebSocket messages
                data = await websocket.receive_json()
                
                if data.get("type") == "location_update":
                    # Update location in cache
                    if redis_cache and redis_cache.locations:
                        await redis_cache.locations.update_user_location(
                            user_id,
                            data["lat"],
                            data["lng"],
                            data.get("accuracy")
                        )
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for user {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    finally:
        # Cleanup Redis subscription
        if redis_cache and redis_cache.alerts and pubsub:
            await redis_cache.alerts.unsubscribe_user_alerts(user_id)

# Cache management endpoints
@app.get("/api/cache/stats")
async def get_cache_stats(
    cache: AtlasRedisCache = Depends(get_cache),
    current_user: Dict = Depends(get_current_user)
):
    """Get cache statistics (admin only)"""
    if current_user["role"] not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        health = await cache.health_check()
        return {
            "cache_health": health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")

@app.post("/api/cache/clear")
async def clear_cache_pattern(
    pattern_data: Dict[str, str],
    cache: AtlasRedisCache = Depends(get_cache),
    current_user: Dict = Depends(get_current_user)
):
    """Clear cache entries matching pattern (admin only)"""
    if current_user["role"] not in ["admin", "system"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        pattern = pattern_data.get("pattern", "atlas:api:*")
        
        if cache.api_responses:
            cleared = await cache.api_responses.invalidate_cache_pattern(pattern)
            return {
                "cleared": cleared,
                "pattern": pattern,
                "timestamp": datetime.now().isoformat()
            }
        
        return {"cleared": 0, "message": "Cache not available"}
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# Helper functions
def calculate_mock_risk(lat: float, lng: float, features: Dict[str, Any]) -> Dict[str, Any]:
    """Mock risk calculation using enhanced features"""
    base_risk = 2.5
    
    # Use enhanced features if available
    if features:
        # Weather impact
        temperature = features.get("temperature", 15.0)
        if temperature < 0 or temperature > 25:
            base_risk += 0.3
        
        # Time impact
        hour = features.get("hour_of_day", 12)
        if 22 <= hour or hour <= 6:  # Late night/early morning
            base_risk += 0.8
        
        # Weekend impact
        if features.get("is_weekend", False):
            base_risk += 0.2
        
        # Population density impact
        pop_density = features.get("population_density", 1000.0)
        if pop_density > 3000:
            base_risk += 0.5
        
        # Economic factors
        unemployment = features.get("unemployment_rate", 5.0)
        base_risk += unemployment / 20.0  # Scale unemployment impact
    
    risk_level = max(1, min(5, round(base_risk, 1)))
    
    return {
        "risk_level": risk_level,
        "risk_category": get_risk_category(risk_level),
        "confidence": 0.8 + (len(features) * 0.02) if features else 0.6
    }

def get_risk_category(risk_level: float) -> str:
    """Get risk category from risk level"""
    if risk_level <= 2:
        return "low"
    elif risk_level <= 3.5:
        return "medium"
    else:
        return "high"

def generate_mock_incidents(lat: float, lng: float, radius_km: float) -> List[Dict[str, Any]]:
    """Generate mock incident data"""
    import random
    
    incidents = []
    incident_count = random.randint(2, 8)
    
    incident_types = ["theft", "vandalism", "assault", "burglary", "drug_offense"]
    
    for i in range(incident_count):
        # Random location within radius
        lat_offset = (random.random() - 0.5) * (radius_km / 111.0) * 2
        lng_offset = (random.random() - 0.5) * (radius_km / 111.0) * 2
        
        incidents.append({
            "id": f"mock_{i}_{int(datetime.now().timestamp())}",
            "incident_type": random.choice(incident_types),
            "location": {
                "lat": lat + lat_offset,
                "lng": lng + lng_offset
            },
            "timestamp": (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
            "severity": random.randint(1, 5),
            "description": f"Mock incident {i+1}"
        })
    
    return incidents

async def send_location_based_alerts(update: UserLocationUpdate, cache: AtlasRedisCache):
    """Send alerts based on user location"""
    try:
        # Mock alert logic - would check for high-risk areas
        risk_threshold = 4.0
        
        # Generate risk assessment for user's location
        if feature_engineering:
            features = await feature_engineering.generate_features(
                update.lat, update.lng
            )
            risk_data = calculate_mock_risk(update.lat, update.lng, features.to_dict())
            
            if risk_data["risk_level"] >= risk_threshold:
                alert_data = {
                    "type": "location_risk",
                    "message": f"You are entering a {risk_data['risk_category']} risk area",
                    "risk_level": risk_data["risk_level"],
                    "location": {"lat": update.lat, "lng": update.lng}
                }
                
                await cache.alerts.send_user_alert(update.user_id, alert_data)
                logger.info(f"Sent risk alert to user {update.user_id}")
    
    except Exception as e:
        logger.error(f"Failed to send location-based alerts: {e}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the enhanced server
    uvicorn.run(
        "main_with_redis:app",
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )