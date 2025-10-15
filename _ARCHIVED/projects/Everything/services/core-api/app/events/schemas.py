"""
Event Schema Definitions

This module defines the schemas and contracts for events published through Kafka.
All events follow a standardized format with versioning and validation.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid


class EventType(Enum):
    """Standard event types"""
    # User events
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    
    # Component events
    COMPONENT_CREATED = "component.created"
    COMPONENT_UPDATED = "component.updated"
    COMPONENT_DELETED = "component.deleted"
    COMPONENT_PRICE_CHANGED = "component.price_changed"
    COMPONENT_AVAILABILITY_CHANGED = "component.availability_changed"
    
    # RFQ events
    RFQ_CREATED = "rfq.created"
    RFQ_UPDATED = "rfq.updated"
    RFQ_SUBMITTED = "rfq.submitted"
    RFQ_RESPONDED = "rfq.responded"
    RFQ_AWARDED = "rfq.awarded"
    RFQ_CLOSED = "rfq.closed"
    
    # Purchase Order events
    PO_CREATED = "po.created"
    PO_UPDATED = "po.updated"
    PO_APPROVED = "po.approved"
    PO_SENT = "po.sent"
    PO_ACKNOWLEDGED = "po.acknowledged"
    PO_DELIVERED = "po.delivered"
    PO_CANCELLED = "po.cancelled"
    
    # Supply Chain Intelligence events
    MARKET_ALERT = "intelligence.market_alert"
    GEOPOLITICAL_ALERT = "intelligence.geopolitical_alert"
    SUPPLIER_ALERT = "intelligence.supplier_alert"
    SHORTAGE_ALERT = "intelligence.shortage_alert"
    PRICE_TREND_ALERT = "intelligence.price_trend_alert"
    
    # System events
    SYSTEM_HEALTH_CHECK = "system.health_check"
    SYSTEM_ERROR = "system.error"
    AUDIT_LOG = "audit.log"


@dataclass
class EventHeader:
    """Standard event header"""
    event_id: str
    event_type: str
    event_version: str
    source_service: str
    timestamp: datetime
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class BaseEvent:
    """Base event structure"""
    header: EventHeader
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'header': asdict(self.header),
            'payload': self.payload,
            'metadata': self.metadata or {}
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        event_dict = self.to_dict()
        # Handle datetime serialization
        event_dict['header']['timestamp'] = self.header.timestamp.isoformat()
        return json.dumps(event_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEvent':
        """Create event from JSON string"""
        data = json.loads(json_str)
        header_data = data['header']
        header_data['timestamp'] = datetime.fromisoformat(header_data['timestamp'])
        
        return cls(
            header=EventHeader(**header_data),
            payload=data['payload'],
            metadata=data.get('metadata')
        )


# User Events

@dataclass
class UserCreatedPayload:
    """User created event payload"""
    user_id: str
    email: str
    name: str
    roles: List[str]
    created_at: datetime


@dataclass
class UserUpdatedPayload:
    """User updated event payload"""
    user_id: str
    email: str
    name: str
    roles: List[str]
    updated_at: datetime
    changed_fields: List[str]


@dataclass
class UserLoginPayload:
    """User login event payload"""
    user_id: str
    email: str
    login_method: str  # password, oidc, api_key
    ip_address: Optional[str]
    user_agent: Optional[str]
    session_id: Optional[str]
    login_at: datetime


# Component Events

@dataclass
class ComponentCreatedPayload:
    """Component created event payload"""
    component_id: str
    manufacturer_part_number: str
    manufacturer_id: str
    category: str
    description: str
    specifications: Dict[str, Any]
    created_by: str
    created_at: datetime


@dataclass
class ComponentUpdatedPayload:
    """Component updated event payload"""
    component_id: str
    manufacturer_part_number: str
    manufacturer_id: str
    category: str
    description: str
    specifications: Dict[str, Any]
    updated_by: str
    updated_at: datetime
    changed_fields: List[str]
    previous_values: Dict[str, Any]


@dataclass
class ComponentPriceChangedPayload:
    """Component price changed event payload"""
    component_id: str
    supplier_id: str
    previous_price: float
    new_price: float
    currency: str
    quantity_break: int
    price_change_percent: float
    effective_date: datetime
    source: str  # manual, supplier_feed, market_data


# RFQ Events

@dataclass
class RFQCreatedPayload:
    """RFQ created event payload"""
    rfq_id: str
    rfq_number: str
    customer_id: str
    items: List[Dict[str, Any]]
    total_items: int
    due_date: Optional[datetime]
    created_by: str
    created_at: datetime
    source: str  # manual, email, api


@dataclass
class RFQSubmittedPayload:
    """RFQ submitted event payload"""
    rfq_id: str
    rfq_number: str
    customer_id: str
    submitted_to: List[str]  # supplier IDs
    items: List[Dict[str, Any]]
    submitted_by: str
    submitted_at: datetime


@dataclass
class RFQRespondedPayload:
    """RFQ responded event payload"""
    rfq_id: str
    supplier_id: str
    response_id: str
    quoted_items: List[Dict[str, Any]]
    total_quote_value: float
    currency: str
    quote_validity_days: int
    responded_at: datetime


# Purchase Order Events

@dataclass
class POCreatedPayload:
    """Purchase order created event payload"""
    po_id: str
    po_number: str
    supplier_id: str
    items: List[Dict[str, Any]]
    total_value: float
    currency: str
    delivery_date: Optional[datetime]
    created_by: str
    created_at: datetime


@dataclass
class POApprovedPayload:
    """Purchase order approved event payload"""
    po_id: str
    po_number: str
    supplier_id: str
    approved_by: str
    approved_at: datetime
    approval_notes: Optional[str]


# Intelligence Events

@dataclass
class MarketAlertPayload:
    """Market alert event payload"""
    alert_id: str
    alert_type: str  # price_increase, shortage, surplus, trend_change
    component_ids: List[str]
    regions: List[str]
    severity: str  # low, medium, high, critical
    confidence: float  # 0.0 to 1.0
    description: str
    data_sources: List[str]
    estimated_impact: Dict[str, Any]
    recommendation: str
    valid_until: Optional[datetime]
    created_at: datetime


@dataclass
class GeopoliticalAlertPayload:
    """Geopolitical alert event payload"""
    alert_id: str
    event_type: str  # sanction, trade_war, policy_change, conflict
    affected_countries: List[str]
    affected_regions: List[str]
    risk_level: str  # low, medium, high, critical
    description: str
    potential_impact: Dict[str, Any]
    recommended_actions: List[str]
    source_urls: List[str]
    created_at: datetime


@dataclass
class SupplierAlertPayload:
    """Supplier alert event payload"""
    alert_id: str
    supplier_id: str
    alert_type: str  # financial_risk, delivery_delay, quality_issue, capacity_change
    severity: str
    description: str
    affected_components: List[str]
    risk_score: float
    recommended_actions: List[str]
    data_sources: List[str]
    created_at: datetime


# System Events

@dataclass
class SystemHealthCheckPayload:
    """System health check event payload"""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    response_time: float
    memory_usage: float
    cpu_usage: float
    active_connections: int
    error_rate: float
    timestamp: datetime


@dataclass
class AuditLogPayload:
    """Audit log event payload"""
    action: str
    resource_type: str
    resource_id: str
    user_id: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    success: bool
    error_message: Optional[str]
    timestamp: datetime


class EventFactory:
    """Factory for creating standardized events"""
    
    @staticmethod
    def create_event(
        event_type: EventType,
        payload: Any,
        source_service: str = "scip-api",
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaseEvent:
        """Create a standardized event"""
        
        # Convert payload to dict if it's a dataclass
        if hasattr(payload, '__dataclass_fields__'):
            payload_dict = asdict(payload)
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            payload_dict = {'data': payload}
        
        # Create header
        header = EventHeader(
            event_id=str(uuid.uuid4()),
            event_type=event_type.value,
            event_version="1.0",
            source_service=source_service,
            timestamp=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id
        )
        
        return BaseEvent(
            header=header,
            payload=payload_dict,
            metadata=metadata
        )
    
    @staticmethod
    def create_user_created_event(user_id: str, email: str, name: str, roles: List[str], **kwargs) -> BaseEvent:
        """Create user created event"""
        payload = UserCreatedPayload(
            user_id=user_id,
            email=email,
            name=name,
            roles=roles,
            created_at=datetime.now(timezone.utc)
        )
        return EventFactory.create_event(EventType.USER_CREATED, payload, **kwargs)
    
    @staticmethod
    def create_component_created_event(
        component_id: str, 
        manufacturer_part_number: str,
        manufacturer_id: str,
        category: str,
        description: str,
        specifications: Dict[str, Any],
        created_by: str,
        **kwargs
    ) -> BaseEvent:
        """Create component created event"""
        payload = ComponentCreatedPayload(
            component_id=component_id,
            manufacturer_part_number=manufacturer_part_number,
            manufacturer_id=manufacturer_id,
            category=category,
            description=description,
            specifications=specifications,
            created_by=created_by,
            created_at=datetime.now(timezone.utc)
        )
        return EventFactory.create_event(EventType.COMPONENT_CREATED, payload, **kwargs)
    
    @staticmethod
    def create_rfq_created_event(
        rfq_id: str,
        rfq_number: str,
        customer_id: str,
        items: List[Dict[str, Any]],
        created_by: str,
        due_date: Optional[datetime] = None,
        source: str = "manual",
        **kwargs
    ) -> BaseEvent:
        """Create RFQ created event"""
        payload = RFQCreatedPayload(
            rfq_id=rfq_id,
            rfq_number=rfq_number,
            customer_id=customer_id,
            items=items,
            total_items=len(items),
            due_date=due_date,
            created_by=created_by,
            created_at=datetime.now(timezone.utc),
            source=source
        )
        return EventFactory.create_event(EventType.RFQ_CREATED, payload, **kwargs)
    
    @staticmethod
    def create_market_alert_event(
        alert_type: str,
        component_ids: List[str],
        regions: List[str],
        severity: str,
        confidence: float,
        description: str,
        **kwargs
    ) -> BaseEvent:
        """Create market alert event"""
        payload = MarketAlertPayload(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            component_ids=component_ids,
            regions=regions,
            severity=severity,
            confidence=confidence,
            description=description,
            data_sources=kwargs.get('data_sources', []),
            estimated_impact=kwargs.get('estimated_impact', {}),
            recommendation=kwargs.get('recommendation', ''),
            valid_until=kwargs.get('valid_until'),
            created_at=datetime.now(timezone.utc)
        )
        return EventFactory.create_event(EventType.MARKET_ALERT, payload, **kwargs)
    
    @staticmethod
    def create_audit_log_event(
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> BaseEvent:
        """Create audit log event"""
        payload = AuditLogPayload(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=kwargs.get('user_agent'),
            before_state=before_state,
            after_state=after_state,
            success=success,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc)
        )
        return EventFactory.create_event(EventType.AUDIT_LOG, payload, **kwargs)


# Schema validation
PAYLOAD_SCHEMAS = {
    EventType.USER_CREATED: UserCreatedPayload,
    EventType.USER_UPDATED: UserUpdatedPayload,
    EventType.USER_LOGIN: UserLoginPayload,
    EventType.COMPONENT_CREATED: ComponentCreatedPayload,
    EventType.COMPONENT_UPDATED: ComponentUpdatedPayload,
    EventType.COMPONENT_PRICE_CHANGED: ComponentPriceChangedPayload,
    EventType.RFQ_CREATED: RFQCreatedPayload,
    EventType.RFQ_SUBMITTED: RFQSubmittedPayload,
    EventType.RFQ_RESPONDED: RFQRespondedPayload,
    EventType.PO_CREATED: POCreatedPayload,
    EventType.PO_APPROVED: POApprovedPayload,
    EventType.MARKET_ALERT: MarketAlertPayload,
    EventType.GEOPOLITICAL_ALERT: GeopoliticalAlertPayload,
    EventType.SUPPLIER_ALERT: SupplierAlertPayload,
    EventType.SYSTEM_HEALTH_CHECK: SystemHealthCheckPayload,
    EventType.AUDIT_LOG: AuditLogPayload
}


def validate_event_payload(event_type: EventType, payload: Dict[str, Any]) -> bool:
    """Validate event payload against schema"""
    if event_type not in PAYLOAD_SCHEMAS:
        return False
    
    schema_class = PAYLOAD_SCHEMAS[event_type]
    
    try:
        # Try to create the payload object
        schema_class(**payload)
        return True
    except (TypeError, ValueError):
        return False