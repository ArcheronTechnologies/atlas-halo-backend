"""
Mobile API Endpoints
Production-ready endpoints for the mobile safety app
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Header
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
import logging

from ..database.mobile_database_manager import get_mobile_database, MobileDatabaseManager
from ..auth.jwt_authentication import verify_token, get_current_user
from ..utils.validation import validate_coordinates, validate_severity_level
from ..websockets.redis_pubsub import get_redis_pubsub

logger = logging.getLogger(__name__)
security = HTTPBearer()


# Helper function for optional authentication
async def get_current_user_optional(authorization: Optional[str] = Header(None)):
    """Get current user if authenticated, otherwise return anonymous user"""
    if not authorization:
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}
    try:
        token = authorization.replace('Bearer ', '')
        payload = verify_token(token)
        return payload
    except:
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}

# MIGRATION (2025-10-03): Standardizing to /api/v1/mobile/*
# Old route: /mobile/* (deprecated, will be removed 2026-04-03)
# New route: /api/v1/mobile/*
router = APIRouter(prefix="/api/v1/mobile", tags=["mobile"])


# Pydantic models
class IncidentCreate(BaseModel):
    """Model for creating a new incident"""
    incident_type: str = Field(..., description="Type of incident (theft, assault, etc.)")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    description: str = Field(..., min_length=10, max_length=1000, description="Incident description")
    severity_level: str = Field(default="moderate", description="Severity level (low, moderate, high, critical)")
    location_address: Optional[str] = Field(None, max_length=200, description="Human-readable address")
    incident_time: Optional[datetime] = Field(None, description="When the incident occurred (defaults to now)")
    media_files: Optional[List[str]] = Field(default=[], description="List of media file URLs")


class IncidentResponse(BaseModel):
    """Model for incident response"""
    id: UUID
    incident_type: str
    latitude: float
    longitude: float
    location_address: Optional[str]
    incident_time: datetime
    reported_time: datetime
    severity_level: str
    description: str
    source: str
    verification_status: str
    resolution_status: str
    data_quality_score: float
    metadata: Dict[str, Any]
    hours_ago: float


class SafetyZoneResponse(BaseModel):
    """Model for safety zone response"""
    id: UUID
    zone_name: str
    center_latitude: float
    center_longitude: float
    radius_meters: int
    current_risk_level: str
    risk_score: float
    incident_count_24h: int
    incident_count_7d: int
    last_incident_time: Optional[datetime]
    area_type: str


class WatchedLocationCreate(BaseModel):
    """Model for creating a watched location"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    name: str = Field(..., min_length=1, max_length=100, description="Location name")
    radius_meters: int = Field(default=500, ge=50, le=5000, description="Alert radius in meters")


class LocationRiskRequest(BaseModel):
    """Model for location risk assessment request"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    radius_meters: int = Field(default=500, ge=50, le=5000, description="Analysis radius in meters")


class DashboardStats(BaseModel):
    """Model for dashboard statistics"""
    total_incidents: int
    critical_count: int
    high_count: int
    moderate_count: int
    low_count: int
    last_hour: int
    verified_count: int
    avg_quality_score: float
    type_breakdown: List[Dict[str, Any]]


@router.get("/incidents", response_model=List[IncidentResponse])
async def get_incidents(
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude for spatial filtering"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude for spatial filtering"),
    radius_km: Optional[float] = Query(None, ge=0.1, le=50, description="Search radius in kilometers"),
    incident_types: Optional[str] = Query(None, description="Comma-separated incident types"),
    severity_levels: Optional[str] = Query(None, description="Comma-separated severity levels"),
    hours_back: int = Query(168, ge=1, le=720, description="Hours to look back (default: 7 days)"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Get incidents with optional spatial and temporal filtering.

    This endpoint provides incident data for the mobile map interface with:
    - Spatial filtering by location and radius
    - Temporal filtering by time range
    - Type and severity filtering
    - Pagination support
    """
    try:
        # Validate coordinates if provided
        if lat is not None or lon is not None or radius_km is not None:
            if not all([lat is not None, lon is not None, radius_km is not None]):
                raise HTTPException(
                    status_code=400,
                    detail="lat, lon, and radius_km must all be provided for spatial filtering"
                )
            validate_coordinates(lat, lon)

        # Parse filter lists
        type_list = incident_types.split(',') if incident_types else None
        severity_list = severity_levels.split(',') if severity_levels else None

        # Validate severity levels
        if severity_list:
            for level in severity_list:
                validate_severity_level(level.strip())

        incidents = await db.get_incidents(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            incident_types=type_list,
            severity_levels=severity_list,
            hours_back=hours_back,
            limit=limit,
            offset=offset
        )

        return incidents

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching incidents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/incidents", response_model=Dict[str, str])
async def create_incident(
    incident_data: IncidentCreate,
    user: Dict[str, Any] = Depends(get_current_user_optional),
    db: MobileDatabaseManager = Depends(get_mobile_database),
    redis_pubsub = Depends(get_redis_pubsub)
):
    """
    Create a new incident report.

    **Authentication Optional** - Anonymous reports allowed

    Allows users to report safety incidents with:
    - Location coordinates and address
    - Incident type and severity classification
    - Detailed description
    - Optional media attachments
    """
    try:
        # Validate coordinates
        validate_coordinates(incident_data.latitude, incident_data.longitude)

        # Validate severity level
        validate_severity_level(incident_data.severity_level)

        incident_id = await db.create_incident(
            incident_type=incident_data.incident_type,
            latitude=incident_data.latitude,
            longitude=incident_data.longitude,
            description=incident_data.description,
            user_id=user['id'],
            severity_level=incident_data.severity_level,
            location_address=incident_data.location_address,
            incident_time=incident_data.incident_time,
            media_files=incident_data.media_files
        )

        # Log user activity
        await db.log_user_activity(
            user_id=user['id'],
            activity_type="incident_reported",
            description=f"Reported {incident_data.incident_type} incident",
            metadata={"incident_id": str(incident_id), "severity": incident_data.severity_level}
        )

        # Publish real-time incident alert
        try:
            incident_alert_data = {
                'id': str(incident_id),
                'incident_type': incident_data.incident_type,
                'severity_level': incident_data.severity_level,
                'latitude': incident_data.latitude,
                'longitude': incident_data.longitude,
                'location_address': incident_data.location_address,
                'description': incident_data.description,
                'reported_time': datetime.now().isoformat(),
                'reporter_id': str(user['id'])
            }
            await redis_pubsub.publish_incident_alert(incident_alert_data)
            logger.info(f"📡 Published real-time alert for incident {incident_id}")
        except Exception as e:
            logger.error(f"❌ Failed to publish incident alert: {e}")

        return {"message": "Incident created successfully", "incident_id": str(incident_id)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/safety-zones", response_model=List[SafetyZoneResponse])
async def get_safety_zones(
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude for spatial filtering"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude for spatial filtering"),
    radius_km: Optional[float] = Query(None, ge=0.1, le=50, description="Search radius in kilometers"),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Get safety zones with risk assessments.

    Returns identified high-risk areas with:
    - Risk scores and levels
    - Incident counts and timing
    - Geographic boundaries
    """
    try:
        # Validate coordinates if provided
        if lat is not None or lon is not None or radius_km is not None:
            if not all([lat is not None, lon is not None, radius_km is not None]):
                raise HTTPException(
                    status_code=400,
                    detail="lat, lon, and radius_km must all be provided for spatial filtering"
                )
            validate_coordinates(lat, lon)

        zones = await db.get_safety_zones(
            lat=lat,
            lon=lon,
            radius_km=radius_km
        )

        return zones

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching safety zones: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/risk-assessment", response_model=Dict[str, float])
async def assess_location_risk(
    location_data: LocationRiskRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Assess risk level for a specific location.

    Provides real-time risk assessment based on:
    - Historical incident data
    - Spatial analysis algorithms
    - Temporal patterns
    """
    try:
        validate_coordinates(location_data.latitude, location_data.longitude)

        risk_score = await db.calculate_location_risk(
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            radius_meters=location_data.radius_meters
        )

        # Determine risk level
        if risk_score >= 8.0:
            risk_level = "critical"
        elif risk_score >= 6.0:
            risk_level = "high"
        elif risk_score >= 4.0:
            risk_level = "moderate"
        elif risk_score >= 2.0:
            risk_level = "low"
        else:
            risk_level = "safe"

        # Log risk assessment activity
        await db.log_user_activity(
            user_id=user['id'],
            activity_type="risk_assessment",
            description=f"Risk assessment for location",
            metadata={
                "latitude": location_data.latitude,
                "longitude": location_data.longitude,
                "risk_score": risk_score,
                "risk_level": risk_level
            }
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "latitude": location_data.latitude,
            "longitude": location_data.longitude
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error assessing location risk: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/watched-locations", response_model=List[Dict[str, Any]])
async def get_watched_locations(
    user: Dict[str, Any] = Depends(get_current_user),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Get user's watched locations for alert monitoring.
    """
    try:
        locations = await db.get_user_watched_locations(user['id'])
        return locations

    except Exception as e:
        logger.error(f"Error fetching watched locations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/watched-locations", response_model=Dict[str, str])
async def add_watched_location(
    location_data: WatchedLocationCreate,
    user: Dict[str, Any] = Depends(get_current_user),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Add a new watched location for monitoring.

    Users can monitor specific locations for safety alerts.
    """
    try:
        validate_coordinates(location_data.latitude, location_data.longitude)

        location_id = await db.add_watched_location(
            user_id=user['id'],
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            name=location_data.name,
            radius_meters=location_data.radius_meters
        )

        # Log activity
        await db.log_user_activity(
            user_id=user['id'],
            activity_type="watched_location_added",
            description=f"Added watched location: {location_data.name}",
            metadata={"location_id": str(location_id)}
        )

        return {"message": "Watched location added successfully", "location_id": str(location_id)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding watched location: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_statistics(
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back for statistics"),
    lat: Optional[float] = Query(None, ge=-90, le=90, description="Latitude for spatial filtering"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="Longitude for spatial filtering"),
    radius_km: Optional[float] = Query(None, ge=0.1, le=50, description="Search radius in kilometers"),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Get dashboard statistics for the mobile app.

    Provides comprehensive incident statistics including:
    - Total incident counts by severity
    - Recent activity trends
    - Incident type breakdown
    - Quality metrics
    """
    try:
        # Validate coordinates if provided
        if lat is not None or lon is not None or radius_km is not None:
            if not all([lat is not None, lon is not None, radius_km is not None]):
                raise HTTPException(
                    status_code=400,
                    detail="lat, lon, and radius_km must all be provided for spatial filtering"
                )
            validate_coordinates(lat, lon)

        stats = await db.get_incident_statistics(
            hours_back=hours_back,
            lat=lat,
            lon=lon,
            radius_km=radius_km
        )

        return stats

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching dashboard statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class PushTokenRegister(BaseModel):
    """Model for registering push notification token"""
    device_id: str = Field(..., min_length=1, max_length=255, description="Anonymous device ID")
    push_token: str = Field(..., min_length=10, description="Expo push token")


@router.post("/push-token/register", response_model=Dict[str, str])
async def register_push_token(
    data: PushTokenRegister,
    db: MobileDatabaseManager = Depends(get_mobile_database),
    user: Dict = Depends(get_current_user_optional)
):
    """
    Register or update push notification token for a device

    **Parameters**:
    - device_id: Anonymous device identifier
    - push_token: Expo push notification token (ExponentPushToken[...])

    **Returns**:
    - Success message
    """
    try:
        from ..services.push_notification_service import get_push_service

        push_service = get_push_service()
        if not push_service:
            raise HTTPException(status_code=500, detail="Push notification service not available")

        # Register token
        user_id = user.get('id') if user.get('id') != '00000000-0000-0000-0000-000000000000' else None
        success = await push_service.register_push_token(
            device_id=data.device_id,
            push_token=data.push_token,
            user_id=user_id
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to register push token")

        logger.info(f"✅ Registered push token for device {data.device_id[:8]}...")
        return {
            "status": "success",
            "message": "Push token registered successfully",
            "device_id": data.device_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering push token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/push-token/deactivate", response_model=Dict[str, str])
async def deactivate_push_token(
    device_id: str = Body(..., embed=True),
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Deactivate push notifications for a device (user opt-out)

    **Parameters**:
    - device_id: Device identifier

    **Returns**:
    - Success message
    """
    try:
        from ..services.push_notification_service import get_push_service

        push_service = get_push_service()
        if not push_service:
            raise HTTPException(status_code=500, detail="Push notification service not available")

        success = await push_service.deactivate_push_token(device_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to deactivate push token")

        return {
            "status": "success",
            "message": "Push notifications deactivated"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating push token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=Dict[str, str])
async def mobile_health_check(db: MobileDatabaseManager = Depends(get_mobile_database)):
    """
    Health check endpoint for mobile API services.
    """
    try:
        # Test database connectivity
        await db.db.execute_query("SELECT 1")

        return {
            "status": "healthy",
            "service": "mobile_api",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Mobile API health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")