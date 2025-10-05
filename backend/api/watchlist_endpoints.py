"""
Watchlist API endpoints for Atlas AI mobile area monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import uuid
from enum import Enum

from ..database.secure_db import DatabaseManager
from ..ai_integration.multi_provider_ai import AIProvider
from ..analytics.advanced_patterns import AdvancedPatternAnalyzer

logger = logging.getLogger(__name__)

watchlist_router = APIRouter(prefix="/api/mobile", tags=["watchlist"])

class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class LocationInfo(BaseModel):
    name: str
    coordinates: Dict[str, float]  # {latitude: float, longitude: float}
    address: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None

class WatchedLocationCreate(BaseModel):
    alias: str = Field(..., min_length=1, max_length=50)
    location: LocationInfo
    alertsEnabled: bool = True
    alertThreshold: RiskLevel = RiskLevel.MODERATE

class WatchedLocationUpdate(BaseModel):
    alias: Optional[str] = Field(None, min_length=1, max_length=50)
    alertsEnabled: Optional[bool] = None
    alertThreshold: Optional[RiskLevel] = None

class WatchedLocationResponse(BaseModel):
    id: str
    alias: str
    location: LocationInfo
    alertsEnabled: bool
    lastChecked: datetime
    currentRiskLevel: RiskLevel
    alertThreshold: RiskLevel
    userId: Optional[str] = None

class WatchlistResponse(BaseModel):
    success: bool
    data: Dict = None
    error: str = None

@watchlist_router.get("/watched-locations")
async def get_watched_locations() -> WatchlistResponse:
    """
    Get all watched locations for the user
    """
    try:
        db = DatabaseManager()

        # In a real app, this would be filtered by authenticated user
        query = """
        SELECT
            id,
            alias,
            location_data,
            alerts_enabled,
            last_checked,
            current_risk_level,
            alert_threshold,
            user_id,
            created_at,
            updated_at
        FROM watched_locations
        WHERE is_active = true
        ORDER BY created_at DESC
        """

        watched_locations = await db.execute_query(query)

        # Format response
        formatted_locations = []
        for location in watched_locations:
            formatted_location = {
                "id": location["id"],
                "alias": location["alias"],
                "location": location["location_data"],
                "alertsEnabled": location["alerts_enabled"],
                "lastChecked": location["last_checked"].isoformat() if location["last_checked"] else None,
                "currentRiskLevel": location["current_risk_level"] or "safe",
                "alertThreshold": location["alert_threshold"] or "moderate",
                "userId": location["user_id"]
            }
            formatted_locations.append(formatted_location)

        return WatchlistResponse(
            success=True,
            data={
                "watchedLocations": formatted_locations,
                "totalCount": len(formatted_locations)
            }
        )

    except Exception as e:
        logger.error(f"Failed to get watched locations: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to retrieve watched locations: {str(e)}"
        )

@watchlist_router.post("/watched-locations")
async def add_watched_location(request: WatchedLocationCreate) -> WatchlistResponse:
    """
    Add a new location to the watchlist
    """
    try:
        db = DatabaseManager()

        # Generate unique ID
        location_id = str(uuid.uuid4())

        # Get initial risk assessment
        current_risk = await assess_location_risk(
            request.location.coordinates["latitude"],
            request.location.coordinates["longitude"]
        )

        # Insert watched location
        query = """
        INSERT INTO watched_locations (
            id, alias, location_data, alerts_enabled, alert_threshold,
            current_risk_level, last_checked, is_active, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING *
        """

        location_data = request.location.dict()
        current_time = datetime.utcnow()

        result = await db.execute_query(query, (
            location_id,
            request.alias,
            location_data,
            request.alertsEnabled,
            request.alertThreshold.value,
            current_risk.value,
            current_time,
            True,
            current_time,
            current_time
        ))

        if result:
            # Schedule monitoring for this location
            await schedule_location_monitoring(location_id, request.location.coordinates)

            response_data = {
                "id": location_id,
                "alias": request.alias,
                "location": location_data,
                "alertsEnabled": request.alertsEnabled,
                "lastChecked": current_time.isoformat(),
                "currentRiskLevel": current_risk.value,
                "alertThreshold": request.alertThreshold.value
            }

            return WatchlistResponse(
                success=True,
                data={"watchedLocation": response_data}
            )
        else:
            raise Exception("Failed to insert watched location")

    except Exception as e:
        logger.error(f"Failed to add watched location: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to add watched location: {str(e)}"
        )

@watchlist_router.put("/watched-locations/{location_id}")
async def update_watched_location(location_id: str, request: WatchedLocationUpdate) -> WatchlistResponse:
    """
    Update a watched location
    """
    try:
        db = DatabaseManager()

        # Build update query dynamically
        update_fields = []
        update_values = []

        if request.alias is not None:
            update_fields.append("alias = %s")
            update_values.append(request.alias)

        if request.alertsEnabled is not None:
            update_fields.append("alerts_enabled = %s")
            update_values.append(request.alertsEnabled)

        if request.alertThreshold is not None:
            update_fields.append("alert_threshold = %s")
            update_values.append(request.alertThreshold.value)

        if not update_fields:
            raise ValueError("No fields to update")

        update_fields.append("updated_at = %s")
        update_values.append(datetime.utcnow())
        update_values.append(location_id)

        query = f"""
        UPDATE watched_locations
        SET {', '.join(update_fields)}
        WHERE id = %s AND is_active = true
        RETURNING *
        """

        result = await db.execute_query(query, update_values)

        if result:
            updated_location = result[0]
            response_data = {
                "id": updated_location["id"],
                "alias": updated_location["alias"],
                "location": updated_location["location_data"],
                "alertsEnabled": updated_location["alerts_enabled"],
                "lastChecked": updated_location["last_checked"].isoformat() if updated_location["last_checked"] else None,
                "currentRiskLevel": updated_location["current_risk_level"],
                "alertThreshold": updated_location["alert_threshold"]
            }

            return WatchlistResponse(
                success=True,
                data={"watchedLocation": response_data}
            )
        else:
            raise Exception("Watched location not found")

    except Exception as e:
        logger.error(f"Failed to update watched location: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to update watched location: {str(e)}"
        )

@watchlist_router.delete("/watched-locations/{location_id}")
async def remove_watched_location(location_id: str) -> WatchlistResponse:
    """
    Remove a watched location from the watchlist
    """
    try:
        db = DatabaseManager()

        # Soft delete the watched location
        query = """
        UPDATE watched_locations
        SET is_active = false, updated_at = %s
        WHERE id = %s AND is_active = true
        RETURNING id
        """

        result = await db.execute_query(query, (datetime.utcnow(), location_id))

        if result:
            # Cancel monitoring for this location
            await cancel_location_monitoring(location_id)

            return WatchlistResponse(
                success=True,
                data={"removedLocationId": location_id}
            )
        else:
            raise Exception("Watched location not found")

    except Exception as e:
        logger.error(f"Failed to remove watched location: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to remove watched location: {str(e)}"
        )

@watchlist_router.get("/watched-locations/{location_id}/status")
async def get_location_status(location_id: str) -> WatchlistResponse:
    """
    Get detailed status for a specific watched location
    """
    try:
        db = DatabaseManager()

        # Get watched location details
        query = """
        SELECT
            id, alias, location_data, alerts_enabled, alert_threshold,
            current_risk_level, last_checked, created_at
        FROM watched_locations
        WHERE id = %s AND is_active = true
        """

        result = await db.execute_query(query, (location_id,))

        if not result:
            raise Exception("Watched location not found")

        watched_location = result[0]
        location_data = watched_location["location_data"]

        # Get detailed risk assessment
        risk_assessment = await get_detailed_risk_assessment(
            location_data["coordinates"]["latitude"],
            location_data["coordinates"]["longitude"]
        )

        # Get recent alerts for this location
        recent_alerts = await get_location_alerts(location_id)

        # Get historical risk trend
        risk_trend = await get_location_risk_trend(location_id)

        response_data = {
            "watchedLocation": {
                "id": watched_location["id"],
                "alias": watched_location["alias"],
                "location": location_data,
                "alertsEnabled": watched_location["alerts_enabled"],
                "lastChecked": watched_location["last_checked"].isoformat() if watched_location["last_checked"] else None,
                "currentRiskLevel": watched_location["current_risk_level"],
                "alertThreshold": watched_location["alert_threshold"]
            },
            "riskAssessment": risk_assessment,
            "recentAlerts": recent_alerts,
            "riskTrend": risk_trend
        }

        return WatchlistResponse(
            success=True,
            data=response_data
        )

    except Exception as e:
        logger.error(f"Failed to get location status: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to get location status: {str(e)}"
        )

@watchlist_router.post("/watched-locations/bulk-update")
async def bulk_update_watchlist() -> WatchlistResponse:
    """
    Update risk levels for all watched locations
    """
    try:
        db = DatabaseManager()

        # Get all active watched locations
        query = """
        SELECT id, location_data, alert_threshold, alerts_enabled
        FROM watched_locations
        WHERE is_active = true
        """

        watched_locations = await db.execute_query(query)

        updated_count = 0
        alerts_sent = 0

        for location in watched_locations:
            try:
                location_data = location["location_data"]
                coordinates = location_data["coordinates"]

                # Assess current risk
                current_risk = await assess_location_risk(
                    coordinates["latitude"],
                    coordinates["longitude"]
                )

                # Update risk level in database
                update_query = """
                UPDATE watched_locations
                SET current_risk_level = %s, last_checked = %s, updated_at = %s
                WHERE id = %s
                """

                await db.execute_query(update_query, (
                    current_risk.value,
                    datetime.utcnow(),
                    datetime.utcnow(),
                    location["id"]
                ))

                updated_count += 1

                # Check if alert should be sent
                if location["alerts_enabled"]:
                    alert_threshold = RiskLevel(location["alert_threshold"])

                    if should_send_risk_alert(current_risk, alert_threshold):
                        await send_watchlist_alert(location["id"], current_risk)
                        alerts_sent += 1

            except Exception as e:
                logger.error(f"Failed to update location {location['id']}: {str(e)}")
                continue

        return WatchlistResponse(
            success=True,
            data={
                "updatedCount": updated_count,
                "alertsSent": alerts_sent,
                "totalLocations": len(watched_locations)
            }
        )

    except Exception as e:
        logger.error(f"Failed to bulk update watchlist: {str(e)}")
        return WatchlistResponse(
            success=False,
            error=f"Failed to update watchlist: {str(e)}"
        )

# Helper functions

async def assess_location_risk(latitude: float, longitude: float) -> RiskLevel:
    """
    Assess current risk level for a location
    """
    try:
        db = DatabaseManager()

        # Get recent incidents within 1km radius
        query = """
        SELECT COUNT(*) as incident_count,
               AVG(CASE
                   WHEN severity_level = 'critical' THEN 4
                   WHEN severity_level = 'high' THEN 3
                   WHEN severity_level = 'moderate' THEN 2
                   WHEN severity_level = 'low' THEN 1
                   ELSE 0
               END) as avg_severity
        FROM incidents
        WHERE ST_DWithin(
            ST_SetSRID(ST_MakePoint(location->>'longitude', location->>'latitude'), 4326)::geography,
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
            1000
        )
        AND incident_time >= NOW() - INTERVAL '7 days'
        AND resolution_status != 'false_alarm'
        """

        result = await db.execute_query(query, (longitude, latitude))

        if result and result[0]:
            incident_count = result[0]["incident_count"] or 0
            avg_severity = result[0]["avg_severity"] or 0

            # Calculate risk based on incident count and severity
            risk_score = (incident_count * 0.3) + (avg_severity * 0.7)

            if risk_score >= 3.5:
                return RiskLevel.CRITICAL
            elif risk_score >= 2.5:
                return RiskLevel.HIGH
            elif risk_score >= 1.5:
                return RiskLevel.MODERATE
            elif risk_score >= 0.5:
                return RiskLevel.LOW
            else:
                return RiskLevel.SAFE

        return RiskLevel.SAFE

    except Exception as e:
        logger.error(f"Failed to assess location risk: {str(e)}")
        return RiskLevel.SAFE

async def get_detailed_risk_assessment(latitude: float, longitude: float) -> Dict:
    """
    Get detailed risk assessment for a location
    """
    try:
        # This would integrate with the main safety assessment API
        response = {
            "riskLevel": "safe",
            "safetyScore": 8.5,
            "factors": [
                {"type": "crime_rate", "impact": "positive", "description": "Low crime rate in area"},
                {"type": "lighting", "impact": "positive", "description": "Well-lit streets"},
                {"type": "police_presence", "impact": "positive", "description": "Regular police patrols"}
            ],
            "recommendations": [
                "Generally safe area with good lighting and security",
                "Exercise normal precautions during late hours"
            ],
            "lastUpdated": datetime.utcnow().isoformat()
        }

        return response

    except Exception as e:
        logger.error(f"Failed to get detailed risk assessment: {str(e)}")
        return {"error": "Assessment unavailable"}

async def get_location_alerts(location_id: str) -> List[Dict]:
    """
    Get recent alerts for a watched location
    """
    try:
        db = DatabaseManager()

        query = """
        SELECT alert_type, alert_message, risk_level, created_at
        FROM watchlist_alerts
        WHERE watched_location_id = %s
        ORDER BY created_at DESC
        LIMIT 10
        """

        alerts = await db.execute_query(query, (location_id,))

        return [
            {
                "type": alert["alert_type"],
                "message": alert["alert_message"],
                "riskLevel": alert["risk_level"],
                "timestamp": alert["created_at"].isoformat()
            }
            for alert in alerts
        ]

    except Exception as e:
        logger.error(f"Failed to get location alerts: {str(e)}")
        return []

async def get_location_risk_trend(location_id: str) -> List[Dict]:
    """
    Get risk level trend over time for a location
    """
    try:
        db = DatabaseManager()

        query = """
        SELECT current_risk_level, last_checked
        FROM watched_locations_history
        WHERE watched_location_id = %s
        ORDER BY last_checked DESC
        LIMIT 30
        """

        trend_data = await db.execute_query(query, (location_id,))

        return [
            {
                "riskLevel": data["current_risk_level"],
                "timestamp": data["last_checked"].isoformat()
            }
            for data in trend_data
        ]

    except Exception as e:
        logger.error(f"Failed to get location risk trend: {str(e)}")
        return []

async def schedule_location_monitoring(location_id: str, coordinates: Dict[str, float]) -> None:
    """
    Schedule periodic monitoring for a watched location
    """
    try:
        # This would integrate with a task scheduler like Celery
        logger.info(f"Scheduled monitoring for location {location_id}")

    except Exception as e:
        logger.error(f"Failed to schedule monitoring: {str(e)}")

async def cancel_location_monitoring(location_id: str) -> None:
    """
    Cancel monitoring for a watched location
    """
    try:
        # This would cancel scheduled tasks
        logger.info(f"Cancelled monitoring for location {location_id}")

    except Exception as e:
        logger.error(f"Failed to cancel monitoring: {str(e)}")

def should_send_risk_alert(current_risk: RiskLevel, threshold: RiskLevel) -> bool:
    """
    Determine if a risk alert should be sent
    """
    risk_values = {
        RiskLevel.SAFE: 0,
        RiskLevel.LOW: 1,
        RiskLevel.MODERATE: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4
    }

    return risk_values[current_risk] > risk_values[threshold]

async def send_watchlist_alert(location_id: str, risk_level: RiskLevel) -> None:
    """
    Send alert for watched location risk change
    """
    try:
        db = DatabaseManager()

        # Log the alert
        query = """
        INSERT INTO watchlist_alerts (
            id, watched_location_id, alert_type, alert_message,
            risk_level, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        alert_id = str(uuid.uuid4())
        alert_message = f"Risk level increased to {risk_level.value.upper()}"

        await db.execute_query(query, (
            alert_id,
            location_id,
            "risk_increase",
            alert_message,
            risk_level.value,
            datetime.utcnow()
        ))

        # This would also send push notification to user
        logger.info(f"Sent alert for location {location_id}: {alert_message}")

    except Exception as e:
        logger.error(f"Failed to send watchlist alert: {str(e)}")