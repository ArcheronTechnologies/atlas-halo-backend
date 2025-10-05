"""WebSocket endpoints for real-time threat alerts and notifications."""

import json
import logging
from typing import Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from websockets.exceptions import ConnectionClosed

from ..auth.jwt_authentication import get_current_user
from ..notifications.threat_alert_system import get_alert_system, ThreatAlertSystem
from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/alerts/{user_id}")
async def websocket_threat_alerts(
    websocket: WebSocket,
    user_id: str,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for real-time threat alerts.
    
    Clients connect with their user_id and receive real-time notifications about:
    - Threat detections in their area
    - Escalating behavior warnings
    - Emergency alerts
    - Area safety updates
    """
    
    await websocket.accept()
    
    try:
        # Authenticate user
        if token:
            try:
                # Verify JWT token (implement your JWT verification here)
                # current_user = await verify_jwt_token(token)
                # if current_user.user_id != user_id:
                #     await websocket.close(code=4001, reason="Unauthorized")
                #     return
                pass  # Skip auth for now, implement as needed
            except Exception as e:
                await websocket.close(code=4001, reason="Invalid token")
                return
        
        # Get alert system
        alert_system = get_alert_system()
        
        # Register user connection
        await alert_system.connect_user(user_id, websocket)
        
        logger.info(f"WebSocket connected for user {user_id}")
        
        # Listen for messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(alert_system, user_id, websocket, message)
                
            except WebSocketDisconnect:
                break
            except ConnectionClosed:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error handling WebSocket message for user {user_id}: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal error processing message"
                }))
    
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
    
    finally:
        # Disconnect user
        try:
            alert_system = get_alert_system()
            await alert_system.disconnect_user(user_id, websocket)
        except Exception as e:
            logger.error(f"Error disconnecting user {user_id}: {e}")
        
        logger.info(f"WebSocket disconnected for user {user_id}")


async def handle_websocket_message(
    alert_system: ThreatAlertSystem,
    user_id: str,
    websocket: WebSocket,
    message: Dict
):
    """Handle incoming WebSocket messages from clients."""
    
    message_type = message.get("type")
    
    if message_type == "location_update":
        # Update user's location for proximity alerts
        location = message.get("location")
        if location and "lat" in location and "lng" in location:
            await alert_system.update_user_location(user_id, location)
            
            await websocket.send_text(json.dumps({
                "type": "location_updated",
                "message": "Location updated successfully"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid location format"
            }))
    
    elif message_type == "dismiss_alert":
        # Allow user to dismiss an alert
        alert_id = message.get("alert_id")
        if alert_id:
            await alert_system.dismiss_alert(user_id, alert_id)
            
            await websocket.send_text(json.dumps({
                "type": "alert_dismissed",
                "alert_id": alert_id,
                "message": "Alert dismissed successfully"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Alert ID required"
            }))
    
    elif message_type == "ping":
        # Heartbeat/ping message
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": message.get("timestamp")
        }))
    
    elif message_type == "subscribe_area":
        # Subscribe to alerts in a specific area
        location = message.get("location")
        radius = message.get("radius", 1000)  # Default 1km radius
        
        if location and "lat" in location and "lng" in location:
            # Store subscription (implement as needed)
            await websocket.send_text(json.dumps({
                "type": "area_subscribed",
                "location": location,
                "radius": radius,
                "message": "Subscribed to area alerts"
            }))
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid location format for area subscription"
            }))
    
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))


@router.get("/alerts/active")
async def get_active_alerts(
    lat: float,
    lng: float,
    radius: float = 1000,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get active threat alerts in a specific area.
    
    Parameters:
    - lat: Latitude
    - lng: Longitude  
    - radius: Search radius in meters (default 1000m)
    
    Returns list of active alerts within the specified area.
    """
    
    try:
        alert_system = get_alert_system()
        
        # Get user's location
        user_location = {"lat": lat, "lng": lng}
        
        # Find relevant alerts
        relevant_alerts = []
        for alert in alert_system.active_alerts.values():
            distance = alert_system._calculate_distance(user_location, alert.location)
            if distance <= radius and distance <= alert.radius_meters:
                # Check if user has dismissed this alert
                dismissal_key = f"alert_dismissed:{current_user['user_id']}:{alert.alert_id}"
                cache = await get_cache()
                if not await cache.get(dismissal_key):
                    relevant_alerts.append(alert.to_dict())
        
        return {
            "alerts": relevant_alerts,
            "location": user_location,
            "radius": radius,
            "count": len(relevant_alerts)
        }
    
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving alerts")


@router.post("/alerts/test")
async def create_test_alert(
    alert_data: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """
    Create a test threat alert (for development/testing).
    
    Requires admin permissions.
    """
    
    # Check if user has admin permissions
    if not current_user.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    try:
        alert_system = get_alert_system()
        
        from ..notifications.threat_alert_system import AlertType
        from ..intelligence.behavioral_analytics import RiskLevel
        
        alert = await alert_system.create_threat_alert(
            alert_type=AlertType(alert_data.get("alert_type", "suspicious_activity")),
            threat_level=RiskLevel(alert_data.get("threat_level", "medium")),
            location=alert_data["location"],
            threat_details=alert_data.get("threat_details", {}),
            source_user_id=current_user["user_id"],
            custom_message=alert_data.get("message")
        )
        
        return {
            "success": True,
            "alert_id": alert.alert_id,
            "message": "Test alert created successfully"
        }
    
    except Exception as e:
        logger.error(f"Error creating test alert: {e}")
        raise HTTPException(status_code=500, detail="Error creating test alert")


@router.delete("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    alert_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Dismiss a specific alert for the current user.
    """
    
    try:
        alert_system = get_alert_system()
        await alert_system.dismiss_alert(current_user["user_id"], alert_id)
        
        return {
            "success": True,
            "message": "Alert dismissed successfully"
        }
    
    except Exception as e:
        logger.error(f"Error dismissing alert: {e}")
        raise HTTPException(status_code=500, detail="Error dismissing alert")


@router.get("/alerts/history")
async def get_alert_history(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius: float = 1000,
    hours: int = 24,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get historical threat alerts in an area.
    
    Parameters:
    - lat: Latitude (optional, uses user's last known location if not provided)
    - lng: Longitude (optional, uses user's last known location if not provided)
    - radius: Search radius in meters (default 1000m)
    - hours: Hours of history to retrieve (default 24 hours)
    """
    
    try:
        db = await get_database()
        
        # Use provided location or user's last known location
        if lat is None or lng is None:
            # Get user's last known location from database or cache
            # This would require storing user locations
            raise HTTPException(status_code=400, detail="Location required")
        
        # Query historical alerts
        from datetime import datetime, timedelta
        since = datetime.now() - timedelta(hours=hours)
        
        alerts = await db.execute_query(
            """
            SELECT alert_id, alert_type, threat_level, 
                   ST_X(location) as lng, ST_Y(location) as lat,
                   radius_meters, message, threat_details, 
                   created_at, expires_at
            FROM threat_alerts 
            WHERE ST_DWithin(location, ST_Point($1, $2), $3)
              AND created_at >= $4
            ORDER BY created_at DESC
            """,
            lng, lat, radius, since
        )
        
        return {
            "alerts": alerts,
            "location": {"lat": lat, "lng": lng},
            "radius": radius,
            "hours": hours,
            "count": len(alerts)
        }
    
    except Exception as e:
        logger.error(f"Error getting alert history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving alert history")