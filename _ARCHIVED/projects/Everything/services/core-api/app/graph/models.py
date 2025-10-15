"""
Neo4j Graph Data Models

Node and relationship models for supply chain graph database.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Graph node types"""
    COMPONENT = "Component"
    COMPANY = "Company"
    USER = "User"
    RFQ = "RFQ"
    PURCHASE_ORDER = "PurchaseOrder"
    PRICE = "Price"
    ALERT = "Alert"
    MARKET_EVENT = "MarketEvent"


class RelationType(Enum):
    """Graph relationship types"""
    MANUFACTURES = "MANUFACTURES"
    SUPPLIES = "SUPPLIES"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    CREATED_BY = "CREATED_BY"
    SUBMITTED_TO = "SUBMITTED_TO"
    CONTAINS = "CONTAINS"
    AWARDED_TO = "AWARDED_TO"
    AFFECTS = "AFFECTS"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    LOCATED_IN = "LOCATED_IN"


@dataclass
class BaseNode:
    """Base graph node"""
    id: str
    created_at: datetime
    node_type: Optional[NodeType] = field(default=None)
    updated_at: Optional[datetime] = field(default=None)
    properties: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to Cypher-compatible properties"""
        props = {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
        }
        
        if self.updated_at:
            props['updated_at'] = self.updated_at.isoformat()
        
        if self.properties:
            props.update(self.properties)
        
        return props


@dataclass 
class ComponentNode:
    """Component graph node"""
    id: str
    created_at: datetime
    manufacturer_part_number: str
    manufacturer_id: str
    category: str
    description: str
    specifications: Dict[str, Any]
    node_type: Optional[NodeType] = field(default=None)
    lifecycle_status: Optional[str] = None
    updated_at: Optional[datetime] = field(default=None)
    properties: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.node_type = NodeType.COMPONENT
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to Cypher-compatible properties"""
        props = {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'manufacturer_part_number': self.manufacturer_part_number,
            'manufacturer_id': self.manufacturer_id,
            'category': self.category,
            'description': self.description,
            'specifications': self.specifications,
            'lifecycle_status': self.lifecycle_status
        }
        
        if self.updated_at:
            props['updated_at'] = self.updated_at.isoformat()
        
        if self.properties:
            props.update(self.properties)
        
        return props


@dataclass
class CompanyNode(BaseNode):
    """Company graph node"""
    name: str
    company_type: str  # manufacturer, supplier, customer, distributor
    industry: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    website: Optional[str] = None
    employees_count: Optional[int] = None
    annual_revenue: Optional[float] = None
    risk_score: Optional[float] = None
    certifications: Optional[List[str]] = None
    
    def __post_init__(self):
        self.node_type = NodeType.COMPANY
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'name': self.name,
            'type': self.company_type,
            'industry': self.industry,
            'country': self.country,
            'region': self.region,
            'website': self.website,
            'employees_count': self.employees_count,
            'annual_revenue': self.annual_revenue,
            'risk_score': self.risk_score,
            'certifications': self.certifications or []
        })
        return props


@dataclass
class UserNode(BaseNode):
    """User graph node"""
    email: str
    name: str
    roles: List[str]
    company_id: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = NodeType.USER
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'email': self.email,
            'name': self.name,
            'roles': self.roles,
            'company_id': self.company_id
        })
        return props


@dataclass
class RFQNode(BaseNode):
    """RFQ graph node"""
    rfq_number: str
    customer_id: str
    status: str
    total_items: int
    due_date: Optional[datetime] = None
    
    def __post_init__(self):
        self.node_type = NodeType.RFQ
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'rfq_number': self.rfq_number,
            'customer_id': self.customer_id,
            'status': self.status,
            'total_items': self.total_items,
            'due_date': self.due_date.isoformat() if self.due_date else None
        })
        return props


@dataclass
class PriceNode(BaseNode):
    """Price graph node"""
    component_id: str
    supplier_id: str
    price: float
    currency: str
    quantity_break: int
    lead_time_days: Optional[int] = None
    effective_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    
    def __post_init__(self):
        self.node_type = NodeType.PRICE
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'component_id': self.component_id,
            'supplier_id': self.supplier_id,
            'price': self.price,
            'currency': self.currency,
            'quantity_break': self.quantity_break,
            'lead_time_days': self.lead_time_days,
            'effective_date': self.effective_date.isoformat() if self.effective_date else None,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None
        })
        return props


@dataclass
class BaseRelationship:
    """Base graph relationship"""
    from_node_id: str
    to_node_id: str
    relationship_type: RelationType
    created_at: datetime
    properties: Optional[Dict[str, Any]] = None
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        """Convert to Cypher-compatible properties"""
        props = {
            'created_at': self.created_at.isoformat(),
        }
        
        if self.properties:
            props.update(self.properties)
        
        return props


@dataclass
class ManufacturesRelationship(BaseRelationship):
    """Company manufactures component relationship"""
    active: bool = True
    primary_manufacturer: bool = False
    
    def __post_init__(self):
        self.relationship_type = RelationType.MANUFACTURES
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'active': self.active,
            'primary_manufacturer': self.primary_manufacturer
        })
        return props


@dataclass
class SuppliesRelationship(BaseRelationship):
    """Company supplies component relationship"""
    price: Optional[float] = None
    currency: Optional[str] = None
    lead_time_days: Optional[int] = None
    minimum_order_quantity: Optional[int] = None
    availability_status: Optional[str] = None
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        self.relationship_type = RelationType.SUPPLIES
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'price': self.price,
            'currency': self.currency,
            'lead_time_days': self.lead_time_days,
            'minimum_order_quantity': self.minimum_order_quantity,
            'availability_status': self.availability_status,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        })
        return props


@dataclass
class AlternativeToRelationship(BaseRelationship):
    """Component alternative relationship"""
    confidence_score: float  # 0.0 to 1.0
    compatibility_level: str  # exact, functional, form_fit_function
    verified: bool = False
    verified_by: Optional[str] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        self.relationship_type = RelationType.ALTERNATIVE_TO
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'confidence_score': self.confidence_score,
            'compatibility_level': self.compatibility_level,
            'verified': self.verified,
            'verified_by': self.verified_by,
            'notes': self.notes
        })
        return props


@dataclass
class CompetesWithRelationship(BaseRelationship):
    """Company competition relationship"""
    market_overlap: float  # 0.0 to 1.0
    competitive_strength: str  # weak, moderate, strong
    market_segments: List[str]
    
    def __post_init__(self):
        self.relationship_type = RelationType.COMPETES_WITH
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'market_overlap': self.market_overlap,
            'competitive_strength': self.competitive_strength,
            'market_segments': self.market_segments
        })
        return props


@dataclass
class PartnersWithRelationship(BaseRelationship):
    """Company partnership relationship"""
    partnership_type: str  # strategic, supplier, distributor, technology
    partnership_status: str  # active, inactive, pending
    contract_value: Optional[float] = None
    contract_duration: Optional[int] = None  # months
    
    def __post_init__(self):
        self.relationship_type = RelationType.PARTNERS_WITH
    
    def to_cypher_properties(self) -> Dict[str, Any]:
        props = super().to_cypher_properties()
        props.update({
            'partnership_type': self.partnership_type,
            'partnership_status': self.partnership_status,
            'contract_value': self.contract_value,
            'contract_duration': self.contract_duration
        })
        return props