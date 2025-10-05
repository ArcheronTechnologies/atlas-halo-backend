"""
Push Notification Service
Sends push notifications to mobile devices via Expo Push Notification Service
"""

import logging
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)

# Expo Push API endpoint
EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"


class PushNotificationService:
    """
    Handles push notifications to mobile devices
    Uses Expo Push Notification Service
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def register_push_token(
        self,
        device_id: str,
        push_token: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Register or update push token for a device

        Args:
            device_id: Anonymous device ID
            push_token: Expo push token (ExponentPushToken[...])
            user_id: Optional user ID (if authenticated)

        Returns:
            True if successful
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Upsert push token
                await conn.execute('''
                    INSERT INTO push_tokens (device_id, push_token, user_id, created_at, updated_at)
                    VALUES ($1, $2, $3, NOW(), NOW())
                    ON CONFLICT (device_id)
                    DO UPDATE SET
                        push_token = EXCLUDED.push_token,
                        user_id = EXCLUDED.user_id,
                        updated_at = NOW()
                ''', device_id, push_token, user_id)

            logger.info(f"✅ Registered push token for device {device_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to register push token: {e}")
            return False

    async def send_push_notification(
        self,
        push_token: str,
        title: str,
        body: str,
        data: Optional[Dict] = None,
        priority: str = "high",
        sound: str = "default"
    ) -> bool:
        """
        Send push notification to a single device

        Args:
            push_token: Expo push token
            title: Notification title
            body: Notification body
            data: Optional data payload
            priority: 'default' or 'high'
            sound: 'default' or None

        Returns:
            True if sent successfully
        """
        try:
            # Validate push token format
            if not push_token.startswith("ExponentPushToken["):
                logger.warning(f"Invalid push token format: {push_token[:20]}...")
                return False

            # Prepare notification
            notification = {
                "to": push_token,
                "title": title,
                "body": body,
                "priority": priority,
                "sound": sound,
            }

            if data:
                notification["data"] = data

            # Send to Expo API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    EXPO_PUSH_URL,
                    json=[notification],
                    headers={"Content-Type": "application/json"}
                ) as response:
                    result = await response.json()

                    # Check for errors
                    if response.status == 200:
                        ticket = result.get("data", [{}])[0]
                        if ticket.get("status") == "ok":
                            logger.info(f"✅ Push notification sent: {title}")
                            return True
                        else:
                            error = ticket.get("message", "Unknown error")
                            logger.error(f"❌ Push notification error: {error}")
                            return False
                    else:
                        logger.error(f"❌ Expo API error {response.status}: {result}")
                        return False

        except Exception as e:
            logger.error(f"❌ Failed to send push notification: {e}")
            return False

    async def send_high_risk_alert(
        self,
        device_id: str,
        location_name: str,
        risk_score: float,
        prediction_id: str,
        latitude: float,
        longitude: float
    ) -> bool:
        """
        Send high-risk area alert to specific device

        Args:
            device_id: Device to notify
            location_name: Name of high-risk location
            risk_score: Risk score (0-1)
            prediction_id: ID of prediction
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            True if sent successfully
        """
        try:
            # Get push token for device
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT push_token FROM push_tokens WHERE device_id = $1 AND active = true',
                    device_id
                )

            if not row:
                logger.debug(f"No active push token for device {device_id[:8]}...")
                return False

            push_token = row['push_token']
            risk_pct = int(risk_score * 100)

            # Send notification
            return await self.send_push_notification(
                push_token=push_token,
                title="⚠️ High Risk Area",
                body=f"You are near {location_name} ({risk_pct}% risk). Stay alert.",
                data={
                    "type": "risk_alert",
                    "prediction_id": prediction_id,
                    "risk_score": risk_score,
                    "location_name": location_name,
                    "latitude": latitude,
                    "longitude": longitude,
                },
                priority="high",
                sound="default"
            )

        except Exception as e:
            logger.error(f"❌ Failed to send risk alert: {e}")
            return False

    async def send_predictive_alert(
        self,
        device_id: str,
        location_name: str,
        risk_score: float,
        trend: str = "increasing"
    ) -> bool:
        """
        Send predictive risk alert (area becoming high-risk)

        Args:
            device_id: Device to notify
            location_name: Name of location
            risk_score: Current risk score
            trend: Risk trend ('increasing' or 'decreasing')

        Returns:
            True if sent successfully
        """
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    'SELECT push_token FROM push_tokens WHERE device_id = $1 AND active = true',
                    device_id
                )

            if not row:
                return False

            push_token = row['push_token']
            risk_pct = int(risk_score * 100)

            return await self.send_push_notification(
                push_token=push_token,
                title="🔮 Predictive Alert",
                body=f"{location_name} shows {trend} risk trend ({risk_pct}%). Consider alternate route.",
                data={
                    "type": "predictive_alert",
                    "risk_score": risk_score,
                    "trend": trend,
                    "location_name": location_name,
                },
                priority="default",
                sound="default"
            )

        except Exception as e:
            logger.error(f"❌ Failed to send predictive alert: {e}")
            return False

    async def send_incident_notification(
        self,
        device_ids: List[str],
        incident_type: str,
        location_name: str,
        severity: int,
        distance_km: float,
        incident_id: str
    ) -> int:
        """
        Send incident notification to multiple nearby devices

        Args:
            device_ids: List of device IDs to notify
            incident_type: Type of incident
            location_name: Location name
            severity: Severity level (1-5)
            distance_km: Distance in km
            incident_id: Incident ID

        Returns:
            Number of notifications sent successfully
        """
        try:
            # Get push tokens for all devices
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT device_id, push_token
                    FROM push_tokens
                    WHERE device_id = ANY($1) AND active = true
                ''', device_ids)

            if not rows:
                return 0

            # Prepare notification
            severity_emoji = "🚨" if severity >= 4 else "⚠️"
            distance_text = f"{int(distance_km * 1000)}m away" if distance_km < 1 else f"{distance_km:.1f}km away"

            # Send to all devices
            sent_count = 0
            for row in rows:
                success = await self.send_push_notification(
                    push_token=row['push_token'],
                    title=f"{severity_emoji} {incident_type.replace('_', ' ').title()}",
                    body=f"{location_name} - {distance_text}",
                    data={
                        "type": "incident",
                        "incident_id": incident_id,
                        "incident_type": incident_type,
                        "severity": severity,
                        "distance_km": distance_km,
                    },
                    priority="high" if severity >= 4 else "default",
                    sound="default"
                )

                if success:
                    sent_count += 1

            logger.info(f"✅ Sent incident notifications to {sent_count}/{len(rows)} devices")
            return sent_count

        except Exception as e:
            logger.error(f"❌ Failed to send incident notifications: {e}")
            return 0

    async def deactivate_push_token(self, device_id: str) -> bool:
        """
        Deactivate push token (user opt-out or app uninstall)

        Args:
            device_id: Device ID

        Returns:
            True if successful
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    UPDATE push_tokens
                    SET active = false, updated_at = NOW()
                    WHERE device_id = $1
                ''', device_id)

            logger.info(f"✅ Deactivated push token for device {device_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to deactivate push token: {e}")
            return False

    async def get_nearby_devices(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0
    ) -> List[str]:
        """
        Get device IDs near a location (for incident notifications)

        Note: Requires tracking device locations (privacy-sensitive)
        For now, this is a placeholder - actual implementation would need:
        1. Users opt-in to location tracking
        2. Periodic location updates from mobile app
        3. PostGIS queries to find nearby devices

        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Radius in km

        Returns:
            List of device IDs
        """
        # TODO: Implement location tracking with user consent
        # For now, return empty list (no location tracking yet)
        logger.debug("Nearby device lookup not implemented (requires location tracking opt-in)")
        return []


# Singleton instance (will be initialized with db_pool in main.py)
_push_service_instance: Optional[PushNotificationService] = None

def get_push_service() -> Optional[PushNotificationService]:
    """Get singleton push notification service instance"""
    return _push_service_instance

def initialize_push_service(db_pool: asyncpg.Pool):
    """Initialize push notification service with database pool"""
    global _push_service_instance
    _push_service_instance = PushNotificationService(db_pool)
    logger.info("✅ Push notification service initialized")
