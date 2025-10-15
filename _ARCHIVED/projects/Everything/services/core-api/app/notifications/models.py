"""
Notification Data Models

Data structures and types for the notification system.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import uuid


class NotificationType(Enum):
    """Types of notifications"""
    # RFQ Events
    RFQ_CREATED = "rfq.created"
    RFQ_UPDATED = "rfq.updated"  
    RFQ_RESPONSE_RECEIVED = "rfq.response_received"
    RFQ_AWARDED = "rfq.awarded"
    RFQ_EXPIRED = "rfq.expired"
    
    # Component Events
    COMPONENT_CREATED = "component.created"
    COMPONENT_UPDATED = "component.updated"
    PRICE_ALERT = "component.price_alert"
    AVAILABILITY_ALERT = "component.availability_alert"
    
    # Purchase Order Events
    PO_CREATED = "po.created"
    PO_APPROVED = "po.approved"
    PO_SHIPPED = "po.shipped"
    PO_DELIVERED = "po.delivered"
    PO_CANCELLED = "po.cancelled"
    
    # Market Intelligence
    MARKET_ALERT = "intelligence.market_alert"
    SUPPLIER_RISK_ALERT = "intelligence.supplier_risk"
    GEOPOLITICAL_ALERT = "intelligence.geopolitical"
    SHORTAGE_ALERT = "intelligence.shortage"
    
    # System Events
    SYSTEM_MAINTENANCE = "system.maintenance"
    SYSTEM_ERROR = "system.error"
    USER_MENTION = "user.mention"
    
    # Custom/Generic
    CUSTOM = "custom"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class DeliveryStatus(Enum):
    """Delivery status tracking"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    EXPIRED = "expired"


class NotificationChannel(Enum):
    """Delivery channels"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class NotificationTarget:
    """Target recipient for notifications"""
    type: str  # user, role, company, all
    value: str  # user_id, role_name, company_id, "*"
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.WEBSOCKET])
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationContent:
    """Notification content structure"""
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    action_url: Optional[str] = None
    action_text: Optional[str] = None
    image_url: Optional[str] = None
    icon: Optional[str] = None


@dataclass
class NotificationDelivery:
    """Delivery attempt tracking"""
    channel: NotificationChannel
    target: str  # user_id, email, phone, etc.
    status: DeliveryStatus
    attempted_at: datetime
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """Core notification object"""
    id: str
    type: NotificationType
    priority: NotificationPriority
    content: NotificationContent
    targets: List[NotificationTarget]
    
    # Metadata
    created_at: datetime
    created_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # Delivery tracking
    deliveries: List[NotificationDelivery] = field(default_factory=list)
    
    # Filtering and routing
    tags: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Persistence
    persistent: bool = True  # Whether to store in database
    read_receipts: Dict[str, datetime] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if notification has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_delivered_to_user(self, user_id: str) -> bool:
        """Check if notification was delivered to specific user"""
        for delivery in self.deliveries:
            if delivery.target == user_id and delivery.status in [DeliveryStatus.DELIVERED, DeliveryStatus.READ]:
                return True
        return False
    
    def mark_read_by_user(self, user_id: str):
        """Mark notification as read by user"""
        self.read_receipts[user_id] = datetime.now(timezone.utc)
        
        # Update delivery status
        for delivery in self.deliveries:
            if delivery.target == user_id and delivery.status == DeliveryStatus.DELIVERED:
                delivery.status = DeliveryStatus.READ
    
    def to_websocket_message(self, user_id: str) -> Dict[str, Any]:
        """Convert to WebSocket message format"""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "title": self.content.title,
            "message": self.content.message,
            "data": self.content.data,
            "action_url": self.content.action_url,
            "action_text": self.content.action_text,
            "image_url": self.content.image_url,
            "icon": self.content.icon,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": self.tags,
            "read": user_id in self.read_receipts
        }
    
    def to_email_content(self) -> Dict[str, Any]:
        """Convert to email format"""
        return {
            "subject": self.content.title,
            "html_body": self._generate_email_html(),
            "text_body": self._generate_email_text(),
            "metadata": {
                "notification_id": self.id,
                "type": self.type.value,
                "priority": self.priority.value
            }
        }
    
    def _generate_email_html(self) -> str:
        """Generate HTML email content"""
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
                <h2 style="color: #333; margin-bottom: 16px;">{self.content.title}</h2>
                <p style="color: #666; line-height: 1.6; margin-bottom: 20px;">
                    {self.content.message}
                </p>
        """
        
        if self.content.action_url and self.content.action_text:
            html += f"""
                <div style="margin: 20px 0;">
                    <a href="{self.content.action_url}" 
                       style="background-color: #007bff; color: white; padding: 12px 24px; 
                              text-decoration: none; border-radius: 4px; display: inline-block;">
                        {self.content.action_text}
                    </a>
                </div>
            """
        
        html += """
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef;">
                    <p style="color: #999; font-size: 12px; margin: 0;">
                        This is an automated notification from SCIP Supply Chain Intelligence Platform.
                    </p>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_email_text(self) -> str:
        """Generate plain text email content"""
        text = f"{self.content.title}\n\n{self.content.message}\n\n"
        
        if self.content.action_url:
            text += f"View details: {self.content.action_url}\n\n"
        
        text += "---\nThis is an automated notification from SCIP Supply Chain Intelligence Platform."
        
        return text


@dataclass 
class UserNotificationPreferences:
    """User notification preferences"""
    user_id: str
    enabled_types: List[NotificationType] = field(default_factory=list)
    disabled_types: List[NotificationType] = field(default_factory=list)
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.WEBSOCKET])
    
    # Channel-specific settings
    email_address: Optional[str] = None
    phone_number: Optional[str] = None
    slack_user_id: Optional[str] = None
    teams_user_id: Optional[str] = None
    
    # Quiet hours
    quiet_hours_start: Optional[str] = None  # "22:00"
    quiet_hours_end: Optional[str] = None    # "08:00"
    quiet_hours_timezone: str = "UTC"
    
    # Priority filtering
    min_priority: NotificationPriority = NotificationPriority.NORMAL
    
    # Frequency limits
    max_notifications_per_hour: int = 50
    digest_mode: bool = False  # Group notifications into digest
    
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def should_receive_notification(self, notification: Notification) -> bool:
        """Check if user should receive this notification based on preferences"""
        
        # Check if type is explicitly disabled
        if notification.type in self.disabled_types:
            return False
        
        # Check if type is in enabled list (if specified)
        if self.enabled_types and notification.type not in self.enabled_types:
            return False
        
        # Check priority threshold
        priority_levels = {
            NotificationPriority.LOW: 1,
            NotificationPriority.NORMAL: 2,
            NotificationPriority.HIGH: 3,
            NotificationPriority.URGENT: 4,
            NotificationPriority.CRITICAL: 5
        }
        
        if priority_levels[notification.priority] < priority_levels[self.min_priority]:
            return False
        
        # Check quiet hours
        if self._is_in_quiet_hours():
            # Allow critical and urgent notifications during quiet hours
            if notification.priority not in [NotificationPriority.CRITICAL, NotificationPriority.URGENT]:
                return False
        
        return True
    
    def _is_in_quiet_hours(self) -> bool:
        """Check if current time is in quiet hours"""
        if not self.quiet_hours_start or not self.quiet_hours_end:
            return False
        
        # This would implement timezone-aware quiet hours checking
        # For now, simplified implementation
        from datetime import datetime
        import pytz
        
        try:
            tz = pytz.timezone(self.quiet_hours_timezone)
            current_time = datetime.now(tz).time()
            
            start_time = datetime.strptime(self.quiet_hours_start, "%H:%M").time()
            end_time = datetime.strptime(self.quiet_hours_end, "%H:%M").time()
            
            if start_time <= end_time:
                # Same day quiet hours
                return start_time <= current_time <= end_time
            else:
                # Overnight quiet hours
                return current_time >= start_time or current_time <= end_time
                
        except Exception:
            return False


@dataclass
class NotificationTemplate:
    """Reusable notification template"""
    id: str
    name: str
    type: NotificationType
    title_template: str
    message_template: str
    default_priority: NotificationPriority = NotificationPriority.NORMAL
    default_channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.WEBSOCKET])
    variables: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def render(self, variables: Dict[str, Any]) -> NotificationContent:
        """Render template with variables"""
        title = self.title_template
        message = self.message_template
        
        # Simple variable substitution
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            title = title.replace(placeholder, str(value))
            message = message.replace(placeholder, str(value))
        
        return NotificationContent(
            title=title,
            message=message,
            data=variables.copy()
        )