"""
Notification Delivery Channels

Implementation of different notification delivery channels including
WebSocket, email, SMS, push notifications, and integrations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import json

from .models import (
    Notification, NotificationChannel, NotificationDelivery, 
    DeliveryStatus, UserNotificationPreferences
)

logger = logging.getLogger(__name__)


class NotificationChannelBase(ABC):
    """Base class for notification delivery channels"""
    
    def __init__(self, channel_type: NotificationChannel):
        self.channel_type = channel_type
        self.enabled = True
        self.delivery_stats = {
            'sent': 0,
            'delivered': 0,
            'failed': 0
        }
    
    @abstractmethod
    async def deliver(self, notification: Notification, target: str, preferences: Optional[UserNotificationPreferences] = None) -> NotificationDelivery:
        """Deliver notification to target via this channel"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if channel is available for delivery"""
        pass
    
    def get_stats(self) -> Dict[str, int]:
        """Get delivery statistics"""
        return self.delivery_stats.copy()


class WebSocketChannel(NotificationChannelBase):
    """WebSocket real-time notification delivery"""
    
    def __init__(self):
        super().__init__(NotificationChannel.WEBSOCKET)
        self.active_connections: Dict[str, Set[Any]] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
    
    def add_connection(self, user_id: str, websocket: Any, session_info: Dict[str, Any] = None):
        """Add WebSocket connection for user"""
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        
        if session_info:
            self.user_sessions[f"{user_id}_{id(websocket)}"] = session_info
        
        logger.debug(f"Added WebSocket connection for user {user_id}")
    
    def remove_connection(self, user_id: str, websocket: Any):
        """Remove WebSocket connection for user"""
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Clean up session info
        session_key = f"{user_id}_{id(websocket)}"
        if session_key in self.user_sessions:
            del self.user_sessions[session_key]
        
        logger.debug(f"Removed WebSocket connection for user {user_id}")
    
    async def deliver(self, notification: Notification, target: str, preferences: Optional[UserNotificationPreferences] = None) -> NotificationDelivery:
        """Deliver notification via WebSocket"""
        delivery = NotificationDelivery(
            channel=self.channel_type,
            target=target,
            status=DeliveryStatus.PENDING,
            attempted_at=datetime.now(timezone.utc)
        )
        
        try:
            if target not in self.active_connections:
                delivery.status = DeliveryStatus.FAILED
                delivery.error_message = "No active WebSocket connections for user"
                self.delivery_stats['failed'] += 1
                return delivery
            
            message = notification.to_websocket_message(target)
            message_json = json.dumps(message)
            
            connections = self.active_connections[target].copy()
            delivered_count = 0
            failed_connections = []
            
            for websocket in connections:
                try:
                    await websocket.send_text(message_json)
                    delivered_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket connection: {e}")
                    failed_connections.append(websocket)
            
            # Clean up failed connections
            for failed_ws in failed_connections:
                self.remove_connection(target, failed_ws)
            
            if delivered_count > 0:
                delivery.status = DeliveryStatus.DELIVERED
                delivery.delivered_at = datetime.now(timezone.utc)
                delivery.metadata = {'delivered_connections': delivered_count}
                self.delivery_stats['delivered'] += 1
                logger.debug(f"WebSocket notification delivered to {delivered_count} connections for user {target}")
            else:
                delivery.status = DeliveryStatus.FAILED
                delivery.error_message = "All WebSocket connections failed"
                self.delivery_stats['failed'] += 1
            
            return delivery
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            self.delivery_stats['failed'] += 1
            logger.error(f"WebSocket delivery error: {e}")
            return delivery
    
    async def is_available(self) -> bool:
        """WebSocket channel is always available"""
        return True
    
    def get_active_users(self) -> List[str]:
        """Get list of users with active connections"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self, user_id: str) -> int:
        """Get number of active connections for user"""
        return len(self.active_connections.get(user_id, set()))
    
    async def broadcast_to_all(self, notification: Notification) -> List[NotificationDelivery]:
        """Broadcast notification to all connected users"""
        deliveries = []
        
        for user_id in self.active_connections.keys():
            delivery = await self.deliver(notification, user_id)
            deliveries.append(delivery)
        
        return deliveries


class EmailChannel(NotificationChannelBase):
    """Email notification delivery"""
    
    def __init__(self, smtp_config: Optional[Dict[str, Any]] = None):
        super().__init__(NotificationChannel.EMAIL)
        self.smtp_config = smtp_config or self._get_smtp_config()
        self.email_client = None
    
    def _get_smtp_config(self) -> Dict[str, Any]:
        """Get SMTP configuration from environment"""
        import os
        return {
            'host': os.getenv('SMTP_HOST', 'localhost'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USERNAME', ''),
            'password': os.getenv('SMTP_PASSWORD', ''),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true',
            'from_email': os.getenv('SMTP_FROM_EMAIL', 'notifications@scip.com'),
            'from_name': os.getenv('SMTP_FROM_NAME', 'SCIP Platform')
        }
    
    async def deliver(self, notification: Notification, target: str, preferences: Optional[UserNotificationPreferences] = None) -> NotificationDelivery:
        """Deliver notification via email"""
        delivery = NotificationDelivery(
            channel=self.channel_type,
            target=target,
            status=DeliveryStatus.PENDING,
            attempted_at=datetime.now(timezone.utc)
        )
        
        try:
            # For now, simulate email delivery
            # In production, would integrate with actual SMTP service
            await self._simulate_email_delivery(notification, target)
            
            delivery.status = DeliveryStatus.SENT
            delivery.metadata = {
                'email_address': target,
                'subject': notification.content.title,
                'smtp_host': self.smtp_config['host']
            }
            self.delivery_stats['sent'] += 1
            
            logger.info(f"Email notification sent to {target}")
            return delivery
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            self.delivery_stats['failed'] += 1
            logger.error(f"Email delivery error: {e}")
            return delivery
    
    async def _simulate_email_delivery(self, notification: Notification, email: str):
        """Simulate email delivery (replace with real SMTP in production)"""
        email_content = notification.to_email_content()
        
        # Log the email that would be sent
        logger.info(f"[EMAIL SIMULATION] To: {email}")
        logger.info(f"[EMAIL SIMULATION] Subject: {email_content['subject']}")
        logger.info(f"[EMAIL SIMULATION] Body: {email_content['text_body']}")
        
        # Simulate delivery delay
        await asyncio.sleep(0.1)
    
    async def is_available(self) -> bool:
        """Check if SMTP server is available"""
        try:
            # In production, would test SMTP connection
            return bool(self.smtp_config.get('host'))
        except Exception:
            return False


class SlackChannel(NotificationChannelBase):
    """Slack notification delivery"""
    
    def __init__(self, bot_token: Optional[str] = None):
        super().__init__(NotificationChannel.SLACK)
        self.bot_token = bot_token or self._get_slack_token()
        self.slack_client = None
    
    def _get_slack_token(self) -> Optional[str]:
        """Get Slack bot token from environment"""
        import os
        return os.getenv('SLACK_BOT_TOKEN')
    
    async def deliver(self, notification: Notification, target: str, preferences: Optional[UserNotificationPreferences] = None) -> NotificationDelivery:
        """Deliver notification via Slack"""
        delivery = NotificationDelivery(
            channel=self.channel_type,
            target=target,
            status=DeliveryStatus.PENDING,
            attempted_at=datetime.now(timezone.utc)
        )
        
        try:
            # Simulate Slack delivery
            await self._simulate_slack_delivery(notification, target)
            
            delivery.status = DeliveryStatus.SENT
            delivery.metadata = {
                'slack_user_id': target,
                'message_type': 'direct_message'
            }
            self.delivery_stats['sent'] += 1
            
            logger.info(f"Slack notification sent to {target}")
            return delivery
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            self.delivery_stats['failed'] += 1
            logger.error(f"Slack delivery error: {e}")
            return delivery
    
    async def _simulate_slack_delivery(self, notification: Notification, user_id: str):
        """Simulate Slack delivery"""
        logger.info(f"[SLACK SIMULATION] To: {user_id}")
        logger.info(f"[SLACK SIMULATION] Message: {notification.content.title} - {notification.content.message}")
        await asyncio.sleep(0.1)
    
    async def is_available(self) -> bool:
        """Check if Slack integration is available"""
        return bool(self.bot_token)


class PushNotificationChannel(NotificationChannelBase):
    """Push notification delivery for mobile apps"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(NotificationChannel.PUSH)
        self.config = config or self._get_push_config()
    
    def _get_push_config(self) -> Dict[str, Any]:
        """Get push notification configuration"""
        import os
        return {
            'firebase_key': os.getenv('FIREBASE_SERVER_KEY', ''),
            'apns_key': os.getenv('APNS_KEY', ''),
            'apns_team_id': os.getenv('APNS_TEAM_ID', ''),
            'apns_bundle_id': os.getenv('APNS_BUNDLE_ID', '')
        }
    
    async def deliver(self, notification: Notification, target: str, preferences: Optional[UserNotificationPreferences] = None) -> NotificationDelivery:
        """Deliver push notification"""
        delivery = NotificationDelivery(
            channel=self.channel_type,
            target=target,
            status=DeliveryStatus.PENDING,
            attempted_at=datetime.now(timezone.utc)
        )
        
        try:
            # Simulate push notification delivery
            await self._simulate_push_delivery(notification, target)
            
            delivery.status = DeliveryStatus.SENT
            delivery.metadata = {
                'device_token': target,
                'platform': 'unknown'  # Would determine iOS/Android
            }
            self.delivery_stats['sent'] += 1
            
            logger.info(f"Push notification sent to {target}")
            return delivery
            
        except Exception as e:
            delivery.status = DeliveryStatus.FAILED
            delivery.error_message = str(e)
            self.delivery_stats['failed'] += 1
            logger.error(f"Push delivery error: {e}")
            return delivery
    
    async def _simulate_push_delivery(self, notification: Notification, device_token: str):
        """Simulate push notification delivery"""
        logger.info(f"[PUSH SIMULATION] To: {device_token}")
        logger.info(f"[PUSH SIMULATION] Title: {notification.content.title}")
        logger.info(f"[PUSH SIMULATION] Body: {notification.content.message}")
        await asyncio.sleep(0.1)
    
    async def is_available(self) -> bool:
        """Check if push notification service is available"""
        return bool(self.config.get('firebase_key') or self.config.get('apns_key'))


class ChannelManager:
    """Manages all notification delivery channels"""
    
    def __init__(self):
        self.channels: Dict[NotificationChannel, NotificationChannelBase] = {}
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize available notification channels"""
        # Always available
        self.channels[NotificationChannel.WEBSOCKET] = WebSocketChannel()
        
        # Optional channels based on configuration
        self.channels[NotificationChannel.EMAIL] = EmailChannel()
        self.channels[NotificationChannel.SLACK] = SlackChannel()
        self.channels[NotificationChannel.PUSH] = PushNotificationChannel()
    
    def get_channel(self, channel_type: NotificationChannel) -> Optional[NotificationChannelBase]:
        """Get channel instance by type"""
        return self.channels.get(channel_type)
    
    async def get_available_channels(self) -> List[NotificationChannel]:
        """Get list of available channels"""
        available = []
        
        for channel_type, channel in self.channels.items():
            if await channel.is_available():
                available.append(channel_type)
        
        return available
    
    async def deliver_to_channel(
        self, 
        channel_type: NotificationChannel,
        notification: Notification, 
        target: str,
        preferences: Optional[UserNotificationPreferences] = None
    ) -> Optional[NotificationDelivery]:
        """Deliver notification via specific channel"""
        channel = self.get_channel(channel_type)
        
        if not channel:
            logger.warning(f"Channel {channel_type} not available")
            return None
        
        if not await channel.is_available():
            logger.warning(f"Channel {channel_type} not available")
            return None
        
        return await channel.deliver(notification, target, preferences)
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all channels"""
        stats = {}
        
        for channel_type, channel in self.channels.items():
            stats[channel_type.value] = channel.get_stats()
        
        return stats
    
    def add_websocket_connection(self, user_id: str, websocket: Any, session_info: Dict[str, Any] = None):
        """Add WebSocket connection"""
        ws_channel = self.get_channel(NotificationChannel.WEBSOCKET)
        if isinstance(ws_channel, WebSocketChannel):
            ws_channel.add_connection(user_id, websocket, session_info)
    
    def remove_websocket_connection(self, user_id: str, websocket: Any):
        """Remove WebSocket connection"""
        ws_channel = self.get_channel(NotificationChannel.WEBSOCKET)
        if isinstance(ws_channel, WebSocketChannel):
            ws_channel.remove_connection(user_id, websocket)
    
    async def broadcast_to_all_websockets(self, notification: Notification) -> List[NotificationDelivery]:
        """Broadcast to all WebSocket connections"""
        ws_channel = self.get_channel(NotificationChannel.WEBSOCKET)
        if isinstance(ws_channel, WebSocketChannel):
            return await ws_channel.broadcast_to_all(notification)
        return []


# Global channel manager instance
channel_manager = ChannelManager()