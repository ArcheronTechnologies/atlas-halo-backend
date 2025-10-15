"""
Notification Manager

Central notification management system with event routing, user targeting,
persistence, delivery guarantees, and template management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json

from .models import (
    Notification, NotificationType, NotificationPriority, NotificationContent,
    NotificationTarget, NotificationChannel, UserNotificationPreferences,
    NotificationTemplate, DeliveryStatus
)
from .channels import channel_manager, ChannelManager
from ..db.mongo import get_mongo_db
from ..cache.redis_cache import cache

logger = logging.getLogger(__name__)


class NotificationStore:
    """Handles notification persistence"""
    
    def __init__(self):
        self.mongo_collection = None
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage backends"""
        try:
            mongo_db = get_mongo_db()
            if mongo_db:
                self.mongo_collection = mongo_db.get_collection("notifications")
        except Exception as e:
            logger.warning(f"MongoDB not available for notifications: {e}")
    
    async def save_notification(self, notification: Notification) -> bool:
        """Save notification to persistent storage"""
        if not notification.persistent:
            return True
        
        try:
            # Save to MongoDB if available
            if self.mongo_collection:
                doc = {
                    "_id": notification.id,
                    "type": notification.type.value,
                    "priority": notification.priority.value,
                    "content": {
                        "title": notification.content.title,
                        "message": notification.content.message,
                        "data": notification.content.data,
                        "action_url": notification.content.action_url,
                        "action_text": notification.content.action_text,
                        "image_url": notification.content.image_url,
                        "icon": notification.content.icon
                    },
                    "targets": [
                        {
                            "type": target.type,
                            "value": target.value,
                            "channels": [c.value for c in target.channels],
                            "preferences": target.preferences
                        }
                        for target in notification.targets
                    ],
                    "created_at": notification.created_at,
                    "created_by": notification.created_by,
                    "expires_at": notification.expires_at,
                    "deliveries": [
                        {
                            "channel": d.channel.value,
                            "target": d.target,
                            "status": d.status.value,
                            "attempted_at": d.attempted_at,
                            "delivered_at": d.delivered_at,
                            "error_message": d.error_message,
                            "metadata": d.metadata
                        }
                        for d in notification.deliveries
                    ],
                    "tags": notification.tags,
                    "conditions": notification.conditions,
                    "read_receipts": {
                        user_id: timestamp.isoformat() 
                        for user_id, timestamp in notification.read_receipts.items()
                    }
                }
                
                await self.mongo_collection.replace_one(
                    {"_id": notification.id}, 
                    doc, 
                    upsert=True
                )
                return True
            
            # Fallback to cache
            await cache.set(
                f"notification:{notification.id}",
                json.dumps(doc, default=str),
                expire=86400  # 24 hours
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to save notification {notification.id}: {e}")
            return False
    
    async def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Retrieve notification from storage"""
        try:
            # Try MongoDB first
            if self.mongo_collection:
                doc = await self.mongo_collection.find_one({"_id": notification_id})
                if doc:
                    return self._document_to_notification(doc)
            
            # Fallback to cache
            cached = await cache.get(f"notification:{notification_id}")
            if cached:
                doc = json.loads(cached)
                return self._document_to_notification(doc)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve notification {notification_id}: {e}")
            return None
    
    async def get_user_notifications(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get notifications for a specific user"""
        try:
            query = {
                "$or": [
                    {"targets.value": user_id},
                    {"targets.type": "all"},
                    {
                        "targets.type": "role",
                        "targets.value": {"$in": await self._get_user_roles(user_id)}
                    }
                ]
            }
            
            if unread_only:
                query[f"read_receipts.{user_id}"] = {"$exists": False}
            
            if self.mongo_collection:
                cursor = self.mongo_collection.find(query).sort("created_at", -1).skip(offset).limit(limit)
                notifications = []
                
                async for doc in cursor:
                    notification = self._document_to_notification(doc)
                    if notification:
                        notifications.append(notification)
                
                return notifications
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get user notifications for {user_id}: {e}")
            return []
    
    async def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as read by user"""
        try:
            # Update in MongoDB
            if self.mongo_collection:
                result = await self.mongo_collection.update_one(
                    {"_id": notification_id},
                    {"$set": {f"read_receipts.{user_id}": datetime.now(timezone.utc).isoformat()}}
                )
                return result.modified_count > 0
            
            # Update in cache
            notification = await self.get_notification(notification_id)
            if notification:
                notification.mark_read_by_user(user_id)
                await self.save_notification(notification)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to mark notification {notification_id} as read: {e}")
            return False
    
    async def cleanup_expired_notifications(self) -> int:
        """Remove expired notifications"""
        try:
            cutoff_time = datetime.now(timezone.utc)
            count = 0
            
            if self.mongo_collection:
                result = await self.mongo_collection.delete_many({
                    "expires_at": {"$lt": cutoff_time}
                })
                count = result.deleted_count
            
            logger.info(f"Cleaned up {count} expired notifications")
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired notifications: {e}")
            return 0
    
    def _document_to_notification(self, doc: Dict[str, Any]) -> Optional[Notification]:
        """Convert storage document to Notification object"""
        try:
            # This would implement full document to object conversion
            # For now, simplified implementation
            return None
        except Exception as e:
            logger.error(f"Failed to convert document to notification: {e}")
            return None
    
    async def _get_user_roles(self, user_id: str) -> List[str]:
        """Get user roles for notification targeting"""
        # This would query the user's roles from the database
        # For now, return empty list
        return []


class NotificationRouter:
    """Routes notifications based on type, priority, and targeting rules"""
    
    def __init__(self):
        self.routing_rules: Dict[NotificationType, Dict[str, Any]] = {}
        self.event_handlers: Dict[NotificationType, List[Callable]] = defaultdict(list)
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default routing rules"""
        self.routing_rules = {
            # Critical alerts should use all channels
            NotificationType.SYSTEM_ERROR: {
                'channels': [NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL, NotificationChannel.SLACK],
                'target_roles': ['admin', 'ops'],
                'escalation_delay': 300  # 5 minutes
            },
            
            NotificationType.GEOPOLITICAL_ALERT: {
                'channels': [NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                'target_roles': ['procurement', 'risk_manager', 'admin'],
                'escalation_delay': 600  # 10 minutes
            },
            
            # RFQ events mainly use WebSocket + Email
            NotificationType.RFQ_CREATED: {
                'channels': [NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                'target_roles': ['sales', 'procurement']
            },
            
            NotificationType.RFQ_RESPONSE_RECEIVED: {
                'channels': [NotificationChannel.WEBSOCKET],
                'target_roles': ['sales']
            },
            
            # Component events
            NotificationType.PRICE_ALERT: {
                'channels': [NotificationChannel.WEBSOCKET],
                'target_roles': ['procurement', 'buyer']
            },
            
            # PO events
            NotificationType.PO_APPROVED: {
                'channels': [NotificationChannel.WEBSOCKET, NotificationChannel.EMAIL],
                'target_roles': ['procurement', 'finance']
            }
        }
    
    def add_routing_rule(self, notification_type: NotificationType, rule: Dict[str, Any]):
        """Add or update routing rule"""
        self.routing_rules[notification_type] = rule
    
    def add_event_handler(self, notification_type: NotificationType, handler: Callable):
        """Add event handler for notification type"""
        self.event_handlers[notification_type].append(handler)
    
    async def route_notification(self, notification: Notification) -> List[NotificationTarget]:
        """Determine routing targets for notification"""
        targets = []
        
        # Use explicit targets if provided
        if notification.targets:
            targets.extend(notification.targets)
        
        # Apply routing rules
        rule = self.routing_rules.get(notification.type)
        if rule:
            # Add role-based targets
            for role in rule.get('target_roles', []):
                target = NotificationTarget(
                    type='role',
                    value=role,
                    channels=rule.get('channels', [NotificationChannel.WEBSOCKET])
                )
                targets.append(target)
        
        # Execute event handlers
        handlers = self.event_handlers.get(notification.type, [])
        for handler in handlers:
            try:
                await handler(notification)
            except Exception as e:
                logger.error(f"Event handler error for {notification.type}: {e}")
        
        return targets
    
    def should_escalate(self, notification: Notification, failed_deliveries: int) -> bool:
        """Determine if notification should be escalated"""
        rule = self.routing_rules.get(notification.type, {})
        escalation_delay = rule.get('escalation_delay', 0)
        
        if not escalation_delay:
            return False
        
        # Check if escalation delay has passed
        time_since_created = (datetime.now(timezone.utc) - notification.created_at).total_seconds()
        return time_since_created > escalation_delay and failed_deliveries > 0


class NotificationManager:
    """Central notification management system"""
    
    def __init__(self):
        self.store = NotificationStore()
        self.router = NotificationRouter()
        self.channel_manager = channel_manager
        self.user_preferences: Dict[str, UserNotificationPreferences] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Rate limiting and batching
        self.rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self.batch_queues: Dict[str, List[Notification]] = defaultdict(list)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.delivery_retry_task: Optional[asyncio.Task] = None
        
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize default notification templates"""
        self.templates = {
            'rfq_created': NotificationTemplate(
                id='rfq_created',
                name='RFQ Created',
                type=NotificationType.RFQ_CREATED,
                title_template='New RFQ: {rfq_number}',
                message_template='A new RFQ {rfq_number} has been created by {customer_name} with {item_count} items.',
                variables=['rfq_number', 'customer_name', 'item_count']
            ),
            
            'price_alert': NotificationTemplate(
                id='price_alert',
                name='Price Alert',
                type=NotificationType.PRICE_ALERT,
                title_template='Price Alert: {component_name}',
                message_template='Price for {component_name} has changed by {change_percent}% from {supplier_name}.',
                default_priority=NotificationPriority.HIGH,
                variables=['component_name', 'change_percent', 'supplier_name']
            ),
            
            'system_error': NotificationTemplate(
                id='system_error',
                name='System Error',
                type=NotificationType.SYSTEM_ERROR,
                title_template='System Error: {service_name}',
                message_template='Error in {service_name}: {error_message}',
                default_priority=NotificationPriority.CRITICAL,
                variables=['service_name', 'error_message']
            )
        }
    
    async def send_notification(
        self,
        notification_type: NotificationType,
        content: NotificationContent,
        targets: Optional[List[NotificationTarget]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        created_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None
    ) -> Notification:
        """Send a notification"""
        
        notification = Notification(
            id="",  # Will be generated
            type=notification_type,
            priority=priority,
            content=content,
            targets=targets or [],
            created_at=datetime.now(timezone.utc),
            created_by=created_by,
            expires_at=expires_at,
            tags=tags or []
        )
        
        return await self.process_notification(notification)
    
    async def send_from_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        targets: Optional[List[NotificationTarget]] = None,
        **kwargs
    ) -> Optional[Notification]:
        """Send notification using template"""
        
        template = self.templates.get(template_id)
        if not template:
            logger.error(f"Template {template_id} not found")
            return None
        
        content = template.render(variables)
        
        return await self.send_notification(
            notification_type=template.type,
            content=content,
            targets=targets,
            priority=kwargs.get('priority', template.default_priority),
            **kwargs
        )
    
    async def process_notification(self, notification: Notification) -> Notification:
        """Process and deliver a notification"""
        try:
            # Route notification to determine targets
            if not notification.targets:
                notification.targets = await self.router.route_notification(notification)
            
            # Save to storage
            await self.store.save_notification(notification)
            
            # Deliver to all targets and channels
            await self._deliver_notification(notification)
            
            logger.info(f"Processed notification {notification.id} of type {notification.type.value}")
            return notification
            
        except Exception as e:
            logger.error(f"Failed to process notification: {e}")
            raise
    
    async def _deliver_notification(self, notification: Notification):
        """Deliver notification to all targets via their preferred channels"""
        delivery_tasks = []
        
        for target in notification.targets:
            # Resolve target to actual user IDs
            user_ids = await self._resolve_target(target)
            
            for user_id in user_ids:
                # Get user preferences
                preferences = await self.get_user_preferences(user_id)
                
                # Check if user should receive this notification
                if preferences and not preferences.should_receive_notification(notification):
                    continue
                
                # Check rate limiting
                if not self._check_rate_limit(user_id, notification.priority):
                    logger.warning(f"Rate limit exceeded for user {user_id}")
                    continue
                
                # Deliver via each channel
                for channel in target.channels:
                    if preferences and channel not in preferences.channels:
                        continue
                    
                    # Create delivery task
                    task = asyncio.create_task(
                        self._deliver_to_channel(notification, channel, user_id, preferences)
                    )
                    delivery_tasks.append(task)
        
        # Wait for all deliveries to complete
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)
    
    async def _deliver_to_channel(
        self, 
        notification: Notification, 
        channel: NotificationChannel, 
        user_id: str,
        preferences: Optional[UserNotificationPreferences]
    ):
        """Deliver notification to user via specific channel"""
        try:
            # Determine target address for channel
            target_address = await self._get_channel_address(user_id, channel, preferences)
            if not target_address:
                logger.warning(f"No address found for user {user_id} on channel {channel}")
                return
            
            # Deliver via channel
            delivery = await self.channel_manager.deliver_to_channel(
                channel, notification, target_address, preferences
            )
            
            if delivery:
                notification.deliveries.append(delivery)
                # Update notification in storage
                await self.store.save_notification(notification)
            
        except Exception as e:
            logger.error(f"Failed to deliver notification {notification.id} to {user_id} via {channel}: {e}")
    
    async def _resolve_target(self, target: NotificationTarget) -> List[str]:
        """Resolve notification target to list of user IDs"""
        user_ids = []
        
        try:
            if target.type == 'user':
                user_ids = [target.value]
            elif target.type == 'role':
                # Query users with this role
                user_ids = await self._get_users_with_role(target.value)
            elif target.type == 'company':
                # Query users in this company
                user_ids = await self._get_company_users(target.value)
            elif target.type == 'all':
                # Get all active users (implement carefully!)
                user_ids = await self._get_all_active_users()
            
            return user_ids
            
        except Exception as e:
            logger.error(f"Failed to resolve target {target.type}:{target.value}: {e}")
            return []
    
    async def _get_channel_address(
        self, 
        user_id: str, 
        channel: NotificationChannel,
        preferences: Optional[UserNotificationPreferences]
    ) -> Optional[str]:
        """Get user's address for specific channel"""
        if channel == NotificationChannel.WEBSOCKET:
            return user_id
        elif channel == NotificationChannel.EMAIL and preferences:
            return preferences.email_address
        elif channel == NotificationChannel.SLACK and preferences:
            return preferences.slack_user_id
        # Add other channel mappings
        
        return None
    
    def _check_rate_limit(self, user_id: str, priority: NotificationPriority) -> bool:
        """Check if user has exceeded rate limits"""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self.rate_limits[user_id] = [
            timestamp for timestamp in self.rate_limits[user_id]
            if timestamp > hour_ago
        ]
        
        # Check limits based on priority
        if priority in [NotificationPriority.CRITICAL, NotificationPriority.URGENT]:
            # No rate limiting for critical notifications
            max_per_hour = 1000
        else:
            max_per_hour = 50  # Default limit
        
        if len(self.rate_limits[user_id]) >= max_per_hour:
            return False
        
        # Add current timestamp
        self.rate_limits[user_id].append(now)
        return True
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserNotificationPreferences]:
        """Get user notification preferences"""
        if user_id in self.user_preferences:
            return self.user_preferences[user_id]
        
        # Load from storage (simplified - would query database)
        # For now, return default preferences
        preferences = UserNotificationPreferences(user_id=user_id)
        self.user_preferences[user_id] = preferences
        return preferences
    
    async def update_user_preferences(self, user_id: str, preferences: UserNotificationPreferences):
        """Update user notification preferences"""
        self.user_preferences[user_id] = preferences
        # Save to storage
        # await self._save_user_preferences(preferences)
    
    async def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as read"""
        return await self.store.mark_as_read(notification_id, user_id)
    
    async def get_user_notifications(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0,
        unread_only: bool = False
    ) -> List[Notification]:
        """Get notifications for user"""
        return await self.store.get_user_notifications(user_id, limit, offset, unread_only)
    
    # WebSocket connection management
    def add_websocket_connection(self, user_id: str, websocket: Any, session_info: Dict[str, Any] = None):
        """Add WebSocket connection for real-time notifications"""
        self.channel_manager.add_websocket_connection(user_id, websocket, session_info)
    
    def remove_websocket_connection(self, user_id: str, websocket: Any):
        """Remove WebSocket connection"""
        self.channel_manager.remove_websocket_connection(user_id, websocket)
    
    async def broadcast_to_all(self, notification: Notification) -> List:
        """Broadcast notification to all connected users"""
        return await self.channel_manager.broadcast_to_all_websockets(notification)
    
    # Background tasks
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.delivery_retry_task = asyncio.create_task(self._retry_failed_deliveries_loop())
    
    async def stop_background_tasks(self):
        """Stop background tasks"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.delivery_retry_task:
            self.delivery_retry_task.cancel()
    
    async def _cleanup_loop(self):
        """Background cleanup of expired notifications"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.store.cleanup_expired_notifications()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _retry_failed_deliveries_loop(self):
        """Background retry of failed deliveries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                # Implementation for retrying failed deliveries
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
    
    # Helper methods (would be implemented with actual database queries)
    async def _get_users_with_role(self, role: str) -> List[str]:
        """Get user IDs with specific role"""
        # Would query database
        return []
    
    async def _get_company_users(self, company_id: str) -> List[str]:
        """Get user IDs in company"""
        # Would query database
        return []
    
    async def _get_all_active_users(self) -> List[str]:
        """Get all active user IDs (use with caution!)"""
        # Would query database with active status
        return []


# Global notification manager instance
notification_manager = NotificationManager()