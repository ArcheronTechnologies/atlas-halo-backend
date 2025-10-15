"""
Real-time Notifications Router

Production WebSocket endpoints for real-time notifications with
authentication, connection management, and event handling.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import logging
import json
import asyncio
from datetime import datetime, timezone

from ..core.security import verify_jwt
from ..core.auth import get_current_user
from ..auth.oidc import get_current_oidc_session
from ..notifications import notification_manager
from ..notifications.models import (
    NotificationType, NotificationPriority, NotificationContent,
    NotificationTarget, NotificationChannel, UserNotificationPreferences
)

logger = logging.getLogger(__name__)
router = APIRouter()


# WebSocket connection tracking
active_connections: Dict[str, Dict[str, WebSocket]] = {}


@router.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    """Production WebSocket endpoint for real-time notifications"""
    user_id = None
    connection_id = None
    
    try:
        # Extract and verify authentication
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=4401, reason="Missing authentication token")
            return
        
        try:
            # Verify JWT token and extract user info
            payload = verify_jwt(token)
            user_id = payload.get("sub") or payload.get("user_id")
            if not user_id:
                await websocket.close(code=4401, reason="Invalid token payload")
                return
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            await websocket.close(code=4401, reason="Authentication failed")
            return
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = f"{user_id}_{id(websocket)}"
        
        # Register connection with notification manager
        session_info = {
            "user_id": user_id,
            "connection_id": connection_id,
            "connected_at": datetime.now(timezone.utc),
            "last_ping": datetime.now(timezone.utc),
            "user_agent": websocket.headers.get("user-agent", ""),
            "ip_address": websocket.client.host if websocket.client else "unknown"
        }
        
        notification_manager.add_websocket_connection(user_id, websocket, session_info)
        
        # Track connection locally
        if user_id not in active_connections:
            active_connections[user_id] = {}
        active_connections[user_id][connection_id] = websocket
        
        logger.info(f"WebSocket connected: user={user_id}, connection={connection_id}")
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "user_id": user_id,
            "connection_id": connection_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server_time": datetime.now(timezone.utc).isoformat()
        }))
        
        # Send any pending notifications
        await _send_pending_notifications(websocket, user_id)
        
        # Main message loop
        while True:
            try:
                # Set timeout for receiving messages
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                await _handle_websocket_message(websocket, user_id, message)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text(json.dumps({
                    "type": "ping",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: user={user_id}, connection={connection_id}")
        
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass
    
    finally:
        # Clean up connection
        if user_id and connection_id:
            notification_manager.remove_websocket_connection(user_id, websocket)
            
            # Remove from local tracking
            if user_id in active_connections:
                active_connections[user_id].pop(connection_id, None)
                if not active_connections[user_id]:
                    del active_connections[user_id]


async def _handle_websocket_message(websocket: WebSocket, user_id: str, message: str):
    """Handle incoming WebSocket messages from client"""
    try:
        data = json.loads(message)
        message_type = data.get("type")
        
        if message_type == "ping":
            # Respond to client ping
            await websocket.send_text(json.dumps({
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }))
            
        elif message_type == "mark_read":
            # Mark notification as read
            notification_id = data.get("notification_id")
            if notification_id:
                success = await notification_manager.mark_as_read(notification_id, user_id)
                await websocket.send_text(json.dumps({
                    "type": "mark_read_response",
                    "notification_id": notification_id,
                    "success": success
                }))
        
        elif message_type == "get_notifications":
            # Send recent notifications
            limit = data.get("limit", 20)
            unread_only = data.get("unread_only", False)
            
            notifications = await notification_manager.get_user_notifications(
                user_id, limit=limit, unread_only=unread_only
            )
            
            await websocket.send_text(json.dumps({
                "type": "notifications_list",
                "notifications": [notif.to_websocket_message(user_id) for notif in notifications],
                "count": len(notifications)
            }))
        
        elif message_type == "subscribe_to_types":
            # Subscribe to specific notification types
            types = data.get("types", [])
            # This would update user preferences to subscribe to specific types
            logger.info(f"User {user_id} subscribing to types: {types}")
            
        elif message_type == "heartbeat":
            # Update last activity
            session_info = {"last_ping": datetime.now(timezone.utc)}
            # Would update connection metadata
            
        else:
            logger.warning(f"Unknown WebSocket message type: {message_type}")
            
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON in WebSocket message from user {user_id}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Invalid JSON format"
        }))
    except Exception as e:
        logger.error(f"Error handling WebSocket message from user {user_id}: {e}")
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Error processing message"
        }))


async def _send_pending_notifications(websocket: WebSocket, user_id: str):
    """Send any pending notifications to newly connected user"""
    try:
        # Get recent unread notifications
        notifications = await notification_manager.get_user_notifications(
            user_id, limit=10, unread_only=True
        )
        
        if notifications:
            await websocket.send_text(json.dumps({
                "type": "pending_notifications",
                "notifications": [notif.to_websocket_message(user_id) for notif in notifications],
                "count": len(notifications)
            }))
            
    except Exception as e:
        logger.error(f"Error sending pending notifications to user {user_id}: {e}")


# REST API Endpoints for notification management

@router.get("/notifications", response_model=Dict[str, Any])
async def get_notifications(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    unread_only: bool = Query(False),
    current_user = Depends(get_current_user)
):
    """Get notifications for current user"""
    try:
        notifications = await notification_manager.get_user_notifications(
            current_user.id, limit=limit, offset=offset, unread_only=unread_only
        )
        
        return {
            "notifications": [notif.to_websocket_message(current_user.id) for notif in notifications],
            "count": len(notifications),
            "limit": limit,
            "offset": offset,
            "unread_only": unread_only
        }
        
    except Exception as e:
        logger.error(f"Error getting notifications for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve notifications")


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user = Depends(get_current_user)
):
    """Mark notification as read"""
    try:
        success = await notification_manager.mark_as_read(notification_id, current_user.id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Notification not found or already read")
        
        return {"success": True, "notification_id": notification_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {e}")
        raise HTTPException(status_code=500, detail="Failed to mark notification as read")


@router.get("/notifications/preferences", response_model=Dict[str, Any])
async def get_notification_preferences(current_user = Depends(get_current_user)):
    """Get user notification preferences"""
    try:
        preferences = await notification_manager.get_user_preferences(current_user.id)
        
        if not preferences:
            # Return default preferences
            preferences = UserNotificationPreferences(user_id=current_user.id)
        
        return {
            "user_id": preferences.user_id,
            "enabled_types": [t.value for t in preferences.enabled_types],
            "disabled_types": [t.value for t in preferences.disabled_types],
            "channels": [c.value for c in preferences.channels],
            "email_address": preferences.email_address,
            "phone_number": preferences.phone_number,
            "quiet_hours_start": preferences.quiet_hours_start,
            "quiet_hours_end": preferences.quiet_hours_end,
            "quiet_hours_timezone": preferences.quiet_hours_timezone,
            "min_priority": preferences.min_priority.value,
            "max_notifications_per_hour": preferences.max_notifications_per_hour,
            "digest_mode": preferences.digest_mode
        }
        
    except Exception as e:
        logger.error(f"Error getting preferences for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve preferences")


@router.put("/notifications/preferences")
async def update_notification_preferences(
    preferences_data: Dict[str, Any],
    current_user = Depends(get_current_user)
):
    """Update user notification preferences"""
    try:
        # Convert API data to preferences object
        preferences = UserNotificationPreferences(
            user_id=current_user.id,
            enabled_types=[NotificationType(t) for t in preferences_data.get("enabled_types", [])],
            disabled_types=[NotificationType(t) for t in preferences_data.get("disabled_types", [])],
            channels=[NotificationChannel(c) for c in preferences_data.get("channels", ["websocket"])],
            email_address=preferences_data.get("email_address"),
            phone_number=preferences_data.get("phone_number"),
            quiet_hours_start=preferences_data.get("quiet_hours_start"),
            quiet_hours_end=preferences_data.get("quiet_hours_end"),
            quiet_hours_timezone=preferences_data.get("quiet_hours_timezone", "UTC"),
            min_priority=NotificationPriority(preferences_data.get("min_priority", "normal")),
            max_notifications_per_hour=preferences_data.get("max_notifications_per_hour", 50),
            digest_mode=preferences_data.get("digest_mode", False)
        )
        
        await notification_manager.update_user_preferences(current_user.id, preferences)
        
        return {"success": True, "message": "Preferences updated"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid preference value: {e}")
    except Exception as e:
        logger.error(f"Error updating preferences for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.post("/notifications/send")
async def send_notification(
    notification_data: Dict[str, Any],
    current_user = Depends(get_current_user)  # Admin only in production
):
    """Send a notification (admin endpoint)"""
    try:
        # This would typically require admin permissions
        content = NotificationContent(
            title=notification_data.get("title", ""),
            message=notification_data.get("message", ""),
            data=notification_data.get("data", {}),
            action_url=notification_data.get("action_url"),
            action_text=notification_data.get("action_text")
        )
        
        # Parse targets
        targets = []
        for target_data in notification_data.get("targets", []):
            target = NotificationTarget(
                type=target_data.get("type", "user"),
                value=target_data.get("value", ""),
                channels=[NotificationChannel(c) for c in target_data.get("channels", ["websocket"])]
            )
            targets.append(target)
        
        notification = await notification_manager.send_notification(
            notification_type=NotificationType(notification_data.get("type", "custom")),
            content=content,
            targets=targets,
            priority=NotificationPriority(notification_data.get("priority", "normal")),
            created_by=current_user.id,
            tags=notification_data.get("tags", [])
        )
        
        return {
            "success": True,
            "notification_id": notification.id,
            "message": "Notification sent"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid notification data: {e}")
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")


@router.get("/notifications/stats")
async def get_notification_stats(current_user = Depends(get_current_user)):
    """Get notification statistics"""
    try:
        # Get user notification counts
        total_notifications = await notification_manager.get_user_notifications(current_user.id, limit=1000)
        unread_notifications = await notification_manager.get_user_notifications(current_user.id, limit=1000, unread_only=True)
        
        # Get channel stats (admin only)
        channel_stats = notification_manager.channel_manager.get_all_stats()
        
        return {
            "user_stats": {
                "total_notifications": len(total_notifications),
                "unread_notifications": len(unread_notifications),
                "read_notifications": len(total_notifications) - len(unread_notifications)
            },
            "channel_stats": channel_stats,
            "active_connections": len(active_connections.get(current_user.id, {}))
        }
        
    except Exception as e:
        logger.error(f"Error getting notification stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
