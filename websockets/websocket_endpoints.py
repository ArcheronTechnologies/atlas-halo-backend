"""
WebSocket API Endpoints
Production-ready WebSocket endpoints for real-time communication
"""

import json
import asyncio
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from datetime import datetime

from .websocket_manager import websocket_manager
from .redis_pubsub import get_redis_pubsub
from ..auth.jwt_authentication import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="JWT authentication token")
):
    """
    WebSocket connection endpoint for real-time communication.

    Supports:
    - Anonymous connections for public safety alerts
    - Authenticated connections for personalized notifications
    - Real-time incident alerts and safety zone updates
    - User-specific notifications and alerts
    """
    connection_id = None

    try:
        # Initialize WebSocket manager if not already done
        if not websocket_manager.redis_pubsub:
            await websocket_manager.initialize()

        # Connect WebSocket
        connection_id = await websocket_manager.connect(websocket, token)

        # Start ping-pong handler for this connection
        ping_task = asyncio.create_task(websocket_manager.handle_ping_pong())

        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                await handle_client_message(connection_id, message)

            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket client disconnected: {connection_id}")
                break
            except json.JSONDecodeError:
                await websocket_manager.connections[connection_id].send_message({
                    'type': 'error',
                    'message': 'Invalid JSON format'
                })
            except Exception as e:
                logger.error(f"❌ WebSocket message error: {e}")
                await websocket_manager.connections[connection_id].send_message({
                    'type': 'error',
                    'message': 'Message processing error'
                })

    except Exception as e:
        logger.error(f"❌ WebSocket connection error: {e}")

    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)
        if 'ping_task' in locals():
            ping_task.cancel()


async def handle_client_message(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle incoming messages from WebSocket clients"""

    message_type = message.get('type')

    try:
        if message_type == 'subscribe':
            # Subscribe to specific channels
            channel = message.get('channel')
            if channel:
                await websocket_manager.subscribe_to_channel(connection_id, channel)
                await websocket_manager.connections[connection_id].send_message({
                    'type': 'subscription_confirmed',
                    'channel': channel,
                    'timestamp': datetime.now().isoformat()
                })

        elif message_type == 'unsubscribe':
            # Unsubscribe from channels
            channel = message.get('channel')
            if channel:
                await websocket_manager.unsubscribe_from_channel(connection_id, channel)
                await websocket_manager.connections[connection_id].send_message({
                    'type': 'unsubscription_confirmed',
                    'channel': channel,
                    'timestamp': datetime.now().isoformat()
                })

        elif message_type == 'location_update':
            # Handle user location updates for proximity alerts
            await handle_location_update(connection_id, message)

        elif message_type == 'pong':
            # Handle pong response
            connection = websocket_manager.connections.get(connection_id)
            if connection:
                connection.last_ping = datetime.now()

        elif message_type == 'get_stats':
            # Send connection statistics
            stats = websocket_manager.get_connection_stats()
            await websocket_manager.connections[connection_id].send_message({
                'type': 'stats',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            })

        else:
            await websocket_manager.connections[connection_id].send_message({
                'type': 'error',
                'message': f'Unknown message type: {message_type}'
            })

    except Exception as e:
        logger.error(f"❌ Error handling client message: {e}")
        await websocket_manager.connections[connection_id].send_message({
            'type': 'error',
            'message': 'Message handling error'
        })


async def handle_location_update(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle user location updates for proximity-based alerts"""
    try:
        latitude = message.get('latitude')
        longitude = message.get('longitude')

        if latitude is None or longitude is None:
            await websocket_manager.connections[connection_id].send_message({
                'type': 'error',
                'message': 'Latitude and longitude required for location updates'
            })
            return

        # Subscribe to location-specific incident channels
        # Round coordinates for privacy and channel grouping
        lat_rounded = round(float(latitude), 2)
        lon_rounded = round(float(longitude), 2)
        location_channel = f'incidents:location:{lat_rounded}:{lon_rounded}'

        await websocket_manager.subscribe_to_channel(connection_id, location_channel)

        # Confirm location update
        await websocket_manager.connections[connection_id].send_message({
            'type': 'location_updated',
            'latitude': lat_rounded,
            'longitude': lon_rounded,
            'subscribed_channel': location_channel,
            'timestamp': datetime.now().isoformat()
        })

        logger.debug(f"📍 Location updated for {connection_id}: {lat_rounded}, {lon_rounded}")

    except Exception as e:
        logger.error(f"❌ Error handling location update: {e}")
        await websocket_manager.connections[connection_id].send_message({
            'type': 'error',
            'message': 'Location update error'
        })


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return {
        'stats': websocket_manager.get_connection_stats(),
        'timestamp': datetime.now().isoformat()
    }


@router.post("/broadcast/{channel}")
async def broadcast_message(
    channel: str,
    message: Dict[str, Any],
    redis_pubsub = Depends(get_redis_pubsub)
):
    """
    Broadcast message to WebSocket channel via Redis pub/sub.

    This endpoint allows external services to send real-time notifications
    to connected WebSocket clients.
    """
    try:
        await redis_pubsub.publish(channel, message)
        return {
            'status': 'success',
            'channel': channel,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error broadcasting message: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }