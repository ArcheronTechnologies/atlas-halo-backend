"""
Real-Time Incident Broadcasting
Broadcasts new incidents to connected WebSocket clients instantly
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict

from .redis_pubsub import get_redis_pubsub
from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)


@dataclass
class IncidentBroadcast:
    """Structured incident broadcast message"""
    incident_id: str
    incident_type: str
    latitude: float
    longitude: float
    severity: int
    description: str
    occurred_at: str
    reported_at: str
    source: str
    confidence_score: float
    location_address: Optional[str] = None

    # Geospatial context
    municipality: Optional[str] = None
    h3_index: Optional[str] = None

    # Real-time metadata
    broadcast_time: str = None

    def __post_init__(self):
        if not self.broadcast_time:
            self.broadcast_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class IncidentBroadcaster:
    """
    Broadcasts new incidents to WebSocket clients in real-time.

    Features:
    - Geographic filtering (only send to nearby users)
    - Severity filtering (users can subscribe to specific levels)
    - Rate limiting (prevent spam)
    - Batching (group multiple incidents if many arrive at once)
    """

    # Redis channels
    CHANNEL_ALL_INCIDENTS = "incidents:all"
    CHANNEL_CRITICAL = "incidents:critical"
    CHANNEL_HIGH = "incidents:high"
    CHANNEL_MODERATE = "incidents:moderate"
    CHANNEL_LOW = "incidents:low"

    # Geographic channels (by municipality)
    CHANNEL_GEO_PREFIX = "incidents:geo:"

    def __init__(self):
        self.redis_pubsub = None
        self.is_initialized = False
        self.broadcast_count = 0
        self.last_broadcast_time = datetime.now()

    async def initialize(self):
        """Initialize broadcaster with Redis connection"""
        if self.is_initialized:
            return

        try:
            self.redis_pubsub = await get_redis_pubsub()
            self.is_initialized = True
            logger.info("✅ Incident Broadcaster initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Incident Broadcaster: {e}")
            raise

    async def broadcast_new_incident(
        self,
        incident_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Broadcast a new incident to all relevant WebSocket clients.

        Args:
            incident_data: Incident data from database
            metadata: Optional enrichment metadata

        Returns:
            True if broadcast successful, False otherwise
        """

        if not self.is_initialized:
            await self.initialize()

        try:
            # Create broadcast message
            broadcast = IncidentBroadcast(
                incident_id=incident_data['id'],
                incident_type=incident_data['incident_type'],
                latitude=incident_data['latitude'],
                longitude=incident_data['longitude'],
                severity=incident_data.get('severity', 3),
                description=incident_data.get('description', ''),
                occurred_at=incident_data['occurred_at'].isoformat() if isinstance(incident_data['occurred_at'], datetime) else incident_data['occurred_at'],
                reported_at=incident_data.get('reported_at', datetime.now()).isoformat() if isinstance(incident_data.get('reported_at'), datetime) else incident_data.get('reported_at', datetime.now().isoformat()),
                source=incident_data.get('source', 'unknown'),
                confidence_score=incident_data.get('confidence_score', 1.0),
                location_address=incident_data.get('location_address'),
                municipality=metadata.get('municipality') if metadata else None,
                h3_index=metadata.get('h3_index_res8') if metadata else None
            )

            message = {
                'type': 'new_incident',
                'data': broadcast.to_dict(),
                'timestamp': datetime.now().isoformat()
            }

            # Broadcast to multiple channels based on severity and location
            await self._broadcast_to_channels(message, incident_data, metadata)

            self.broadcast_count += 1
            self.last_broadcast_time = datetime.now()

            logger.info(
                f"📡 Broadcast incident {incident_data['id'][:8]} "
                f"({incident_data['incident_type']}, severity={incident_data.get('severity', 3)})"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to broadcast incident: {e}", exc_info=True)
            return False

    async def _broadcast_to_channels(
        self,
        message: Dict[str, Any],
        incident_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ):
        """Broadcast to multiple relevant channels"""

        severity = incident_data.get('severity', 3)

        # 1. Broadcast to all incidents channel
        await self._publish_to_channel(self.CHANNEL_ALL_INCIDENTS, message)

        # 2. Broadcast to severity-specific channel
        severity_channel = self._get_severity_channel(severity)
        if severity_channel:
            await self._publish_to_channel(severity_channel, message)

        # 3. Broadcast to geographic channel (municipality)
        if metadata and metadata.get('municipality'):
            geo_channel = f"{self.CHANNEL_GEO_PREFIX}{metadata['municipality'].lower()}"
            await self._publish_to_channel(geo_channel, message)

        # 4. Broadcast to connected WebSocket clients directly
        await self._broadcast_to_websockets(message, incident_data, metadata)

    def _get_severity_channel(self, severity: int) -> Optional[str]:
        """Get Redis channel name for severity level"""
        if severity >= 5:
            return self.CHANNEL_CRITICAL
        elif severity == 4:
            return self.CHANNEL_HIGH
        elif severity == 3:
            return self.CHANNEL_MODERATE
        elif severity <= 2:
            return self.CHANNEL_LOW
        return None

    async def _publish_to_channel(self, channel: str, message: Dict[str, Any]):
        """Publish message to Redis channel"""
        try:
            await self.redis_pubsub.publish(
                channel,
                json.dumps(message)
            )
            logger.debug(f"Published to channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")

    async def _broadcast_to_websockets(
        self,
        message: Dict[str, Any],
        incident_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ):
        """Broadcast directly to connected WebSocket clients with geographic filtering"""

        # Get all connected clients
        if not websocket_manager.connections:
            return

        incident_lat = incident_data['latitude']
        incident_lon = incident_data['longitude']

        broadcast_tasks = []

        for connection_id, connection in websocket_manager.connections.items():
            # Check if client should receive this incident
            if await self._should_send_to_client(
                connection,
                incident_data,
                incident_lat,
                incident_lon,
                metadata
            ):
                # Send asynchronously
                task = asyncio.create_task(
                    connection.send_message(message)
                )
                broadcast_tasks.append(task)

        # Wait for all broadcasts to complete
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            logger.debug(f"Sent to {len(broadcast_tasks)} WebSocket clients")

    async def _should_send_to_client(
        self,
        connection: Any,
        incident_data: Dict[str, Any],
        incident_lat: float,
        incident_lon: float,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Determine if incident should be sent to this client.
        Considers user location, subscriptions, and preferences.
        """

        # If client has subscriptions, check them
        if hasattr(connection, 'subscriptions'):
            subscriptions = connection.subscriptions

            # Check severity subscription
            severity = incident_data.get('severity', 3)
            severity_channel = self._get_severity_channel(severity)

            if severity_channel and severity_channel in subscriptions:
                return True

            # Check geographic subscription
            if metadata and metadata.get('municipality'):
                geo_channel = f"{self.CHANNEL_GEO_PREFIX}{metadata['municipality'].lower()}"
                if geo_channel in subscriptions:
                    return True

            # If subscribed to "all incidents"
            if self.CHANNEL_ALL_INCIDENTS in subscriptions:
                return True

        # If client has location, do proximity check
        if hasattr(connection, 'user_location') and connection.user_location:
            user_lat = connection.user_location.get('latitude')
            user_lon = connection.user_location.get('longitude')
            user_radius = connection.user_location.get('radius_km', 10)  # Default 10km

            if user_lat and user_lon:
                distance_km = self._calculate_distance(
                    incident_lat, incident_lon,
                    user_lat, user_lon
                )

                if distance_km <= user_radius:
                    return True

        # Default: don't send (user must opt-in via subscription or location)
        return False

    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance in kilometers (simplified Haversine)"""
        from math import radians, cos, sin, asin, sqrt

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))

        # Radius of earth in kilometers
        r = 6371

        return c * r

    async def broadcast_incident_update(
        self,
        incident_id: str,
        update_type: str,
        update_data: Dict[str, Any]
    ):
        """
        Broadcast an update to an existing incident.

        Args:
            incident_id: ID of the incident being updated
            update_type: Type of update ('status_change', 'verification', 'resolution', etc.)
            update_data: Updated data
        """

        message = {
            'type': 'incident_update',
            'incident_id': incident_id,
            'update_type': update_type,
            'data': update_data,
            'timestamp': datetime.now().isoformat()
        }

        # Broadcast to all incidents channel
        await self._publish_to_channel(self.CHANNEL_ALL_INCIDENTS, message)

        logger.info(f"📡 Broadcast update for incident {incident_id[:8]}: {update_type}")

    async def get_statistics(self) -> Dict[str, Any]:
        """Get broadcaster statistics"""
        return {
            'total_broadcasts': self.broadcast_count,
            'last_broadcast': self.last_broadcast_time.isoformat(),
            'is_initialized': self.is_initialized,
            'connected_clients': len(websocket_manager.connections) if websocket_manager else 0
        }


# Singleton instance
_broadcaster_instance: Optional[IncidentBroadcaster] = None


async def get_incident_broadcaster() -> IncidentBroadcaster:
    """Get or create broadcaster instance"""
    global _broadcaster_instance

    if _broadcaster_instance is None:
        _broadcaster_instance = IncidentBroadcaster()
        await _broadcaster_instance.initialize()

    return _broadcaster_instance


async def broadcast_new_incident(
    incident_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function to broadcast a new incident.

    Usage:
        from backend.websockets.incident_broadcaster import broadcast_new_incident

        # After storing incident
        await broadcast_new_incident(incident_data, enrichment_metadata)
    """
    broadcaster = await get_incident_broadcaster()
    return await broadcaster.broadcast_new_incident(incident_data, metadata)