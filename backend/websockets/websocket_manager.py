"""
WebSocket Connection Manager
Production-ready WebSocket handling with Redis pub/sub integration
"""

import json
import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from uuid import uuid4
from fastapi import WebSocket, WebSocketDisconnect
import jwt

from .redis_pubsub import RedisPubSubManager, get_redis_pubsub
from ..auth.jwt_authentication import verify_token

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Individual WebSocket connection with user context"""

    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send message to WebSocket client"""
        try:
            await self.websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"❌ Error sending message to {self.connection_id}: {e}")

    async def send_ping(self) -> None:
        """Send ping to keep connection alive"""
        await self.send_message({
            'type': 'ping',
            'timestamp': datetime.now().isoformat()
        })
        self.last_ping = datetime.now()


class WebSocketManager:
    """Production WebSocket connection manager with Redis integration"""

    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.channel_subscriptions: Dict[str, Set[str]] = {}  # channel -> set of connection_ids
        self.redis_pubsub: Optional[RedisPubSubManager] = None

    async def initialize(self) -> None:
        """Initialize WebSocket manager with Redis"""
        self.redis_pubsub = await get_redis_pubsub()

        # Subscribe to all incident channels
        await self.redis_pubsub.subscribe('incidents:alerts', self._handle_incident_alert)
        await self.redis_pubsub.subscribe('safety_zones:updates', self._handle_safety_zone_update)

        # Subscribe to severity-specific channels
        for severity in ['critical', 'high', 'moderate', 'low']:
            await self.redis_pubsub.subscribe(f'incidents:severity:{severity}', self._handle_incident_alert)

        logger.info("🚀 WebSocket manager initialized with Redis pub/sub")

    async def connect(self, websocket: WebSocket, token: Optional[str] = None) -> str:
        """Accept new WebSocket connection"""
        await websocket.accept()

        connection_id = str(uuid4())
        user_id = None

        # Authenticate user if token provided
        if token:
            try:
                payload = verify_token(token)
                user_id = payload.get('sub')
                logger.info(f"🔐 Authenticated WebSocket connection for user {user_id}")
            except Exception as e:
                logger.warning(f"⚠️ WebSocket authentication failed: {e}")

        # Create connection
        connection = WebSocketConnection(websocket, connection_id, user_id)
        self.connections[connection_id] = connection

        # Track user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)

            # Subscribe to user-specific notifications
            await self.redis_pubsub.subscribe(
                f'users:{user_id}:notifications',
                self._handle_user_notification
            )

        # Send welcome message
        await connection.send_message({
            'type': 'connection_established',
            'connection_id': connection_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"🔌 WebSocket connected: {connection_id} (user: {user_id or 'anonymous'})")
        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """Handle WebSocket disconnection"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        user_id = connection.user_id

        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        # Remove from channel subscriptions
        for channel in connection.subscriptions:
            if channel in self.channel_subscriptions:
                self.channel_subscriptions[channel].discard(connection_id)
                if not self.channel_subscriptions[channel]:
                    del self.channel_subscriptions[channel]

        # Remove connection
        del self.connections[connection_id]

        logger.info(f"🔌 WebSocket disconnected: {connection_id} (user: {user_id or 'anonymous'})")

    async def subscribe_to_channel(self, connection_id: str, channel: str) -> None:
        """Subscribe connection to a specific channel"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        connection.subscriptions.add(channel)

        if channel not in self.channel_subscriptions:
            self.channel_subscriptions[channel] = set()
        self.channel_subscriptions[channel].add(connection_id)

        logger.debug(f"📻 Connection {connection_id} subscribed to {channel}")

    async def unsubscribe_from_channel(self, connection_id: str, channel: str) -> None:
        """Unsubscribe connection from a channel"""
        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]
        connection.subscriptions.discard(channel)

        if channel in self.channel_subscriptions:
            self.channel_subscriptions[channel].discard(connection_id)
            if not self.channel_subscriptions[channel]:
                del self.channel_subscriptions[channel]

        logger.debug(f"📻 Connection {connection_id} unsubscribed from {channel}")

    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]) -> None:
        """Broadcast message to all connections subscribed to a channel"""
        if channel not in self.channel_subscriptions:
            return

        connections_to_notify = self.channel_subscriptions[channel].copy()

        for connection_id in connections_to_notify:
            if connection_id in self.connections:
                try:
                    await self.connections[connection_id].send_message(message)
                except Exception as e:
                    logger.error(f"❌ Error broadcasting to {connection_id}: {e}")
                    # Remove failed connection
                    await self.disconnect(connection_id)

    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> None:
        """Send message to all connections for a specific user"""
        if user_id not in self.user_connections:
            return

        connections_to_notify = self.user_connections[user_id].copy()

        for connection_id in connections_to_notify:
            if connection_id in self.connections:
                try:
                    await self.connections[connection_id].send_message(message)
                except Exception as e:
                    logger.error(f"❌ Error sending to user {user_id}: {e}")
                    await self.disconnect(connection_id)

    async def _handle_incident_alert(self, data: Dict[str, Any]) -> None:
        """Handle incident alert from Redis"""
        message = {
            'type': 'incident_alert',
            'data': data['data'],
            'timestamp': data['timestamp']
        }

        # Broadcast to all connections (real-time public safety)
        for connection_id in list(self.connections.keys()):
            if connection_id in self.connections:
                try:
                    await self.connections[connection_id].send_message(message)
                except Exception as e:
                    logger.error(f"❌ Error broadcasting incident alert: {e}")
                    await self.disconnect(connection_id)

    async def _handle_safety_zone_update(self, data: Dict[str, Any]) -> None:
        """Handle safety zone update from Redis"""
        message = {
            'type': 'safety_zone_update',
            'data': data['data'],
            'timestamp': data['timestamp']
        }

        # Broadcast to interested connections
        await self.broadcast_to_channel('safety_zones', message)

    async def _handle_user_notification(self, data: Dict[str, Any]) -> None:
        """Handle user-specific notification from Redis"""
        notification_data = data['data']
        user_id = notification_data.get('user_id')

        if user_id:
            message = {
                'type': 'user_notification',
                'data': notification_data['notification'],
                'timestamp': data['timestamp']
            }
            await self.send_to_user(user_id, message)

    async def handle_ping_pong(self) -> None:
        """Periodically ping connections to keep them alive"""
        while True:
            try:
                current_time = datetime.now()
                connections_to_check = list(self.connections.keys())

                for connection_id in connections_to_check:
                    if connection_id in self.connections:
                        connection = self.connections[connection_id]

                        # Check if connection is stale (no ping in 60 seconds)
                        if (current_time - connection.last_ping).seconds > 60:
                            try:
                                await connection.send_ping()
                            except Exception:
                                await self.disconnect(connection_id)

                await asyncio.sleep(30)  # Ping every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in ping-pong handler: {e}")
                await asyncio.sleep(30)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            'total_connections': len(self.connections),
            'authenticated_users': len(self.user_connections),
            'active_channels': len(self.channel_subscriptions),
            'connections_by_user': {
                user_id: len(connections)
                for user_id, connections in self.user_connections.items()
            }
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()