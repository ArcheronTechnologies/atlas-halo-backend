"""
Atlas API Contracts
Unified API specifications for external product integration
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ServiceScope(Enum):
    """Atlas service access scopes"""
    INTELLIGENCE_READ = "intelligence:read"
    INTELLIGENCE_ANALYZE = "intelligence:analyze"
    ANALYTICS_PREDICT = "analytics:predict"
    ANALYTICS_DETECT = "analytics:detect"
    ENTITIES_RESOLVE = "entities:resolve"
    ENTITIES_MERGE = "entities:merge"
    KNOWLEDGE_QUERY = "knowledge:query"
    KNOWLEDGE_INFER = "knowledge:infer"
    FUSION_INGEST = "fusion:ingest"
    FUSION_TRANSFORM = "fusion:transform"


class DataClassification(Enum):
    """Data classification levels for Atlas services"""
    PUBLIC = "public"
    OFFICIAL = "official"
    RESTRICTED = "restricted"
    SECRET = "secret"


# Base Request/Response Models

class AtlasRequest(BaseModel):
    """Base Atlas API request"""
    request_id: str = Field(description="Unique request identifier")
    client_id: str = Field(description="Client application identifier")
    purpose: str = Field(description="Processing purpose")
    classification: DataClassification = Field(description="Data classification level")
    scopes: List[ServiceScope] = Field(description="Required service scopes")


class AtlasResponse(BaseModel):
    """Base Atlas API response"""
    request_id: str
    result: Dict[str, Any]
    metadata: Dict[str, Any]
    compliance: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# Intelligence Engine Contracts

class IntelligenceAnalysisRequest(AtlasRequest):
    """Request for intelligence analysis"""
    data_sources: List[str] = Field(description="Data sources to analyze")
    analysis_type: str = Field(description="Type of analysis requested")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Analysis context")
    time_range: Optional[Dict[str, str]] = Field(default=None, description="Time range filter")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")


class IntelligenceAnalysisResponse(AtlasResponse):
    """Response from intelligence analysis"""
    insights: List[Dict[str, Any]]
    confidence: float
    sources_used: List[str]
    processing_time: float


# Analytics Engine Contracts

class PredictionRequest(AtlasRequest):
    """Request for predictive analytics"""
    model_type: str = Field(description="Prediction model type")
    input_features: Dict[str, Any] = Field(description="Input features for prediction")
    prediction_horizon: str = Field(description="Time horizon for prediction")
    include_confidence_intervals: bool = Field(default=True)


class PredictionResponse(AtlasResponse):
    """Response from predictive analytics"""
    predictions: List[Dict[str, Any]]
    confidence_intervals: Optional[Dict[str, Any]]
    model_version: str
    feature_importance: Dict[str, float]


class AnomalyDetectionRequest(AtlasRequest):
    """Request for anomaly detection"""
    data_stream: List[Dict[str, Any]] = Field(description="Data stream to analyze")
    detection_method: str = Field(default="statistical", description="Detection method")
    sensitivity: float = Field(default=0.8, description="Detection sensitivity")


class AnomalyDetectionResponse(AtlasResponse):
    """Response from anomaly detection"""
    anomalies: List[Dict[str, Any]]
    anomaly_scores: List[float]
    detection_threshold: float
    baseline_statistics: Dict[str, float]


# Entity Resolution Contracts

class EntityResolutionRequest(AtlasRequest):
    """Request for entity resolution"""
    entities: List[Dict[str, Any]] = Field(description="Entities to resolve")
    resolution_strategy: str = Field(default="fuzzy", description="Resolution strategy")
    confidence_threshold: float = Field(default=0.85)
    include_relationships: bool = Field(default=True)


class EntityResolutionResponse(AtlasResponse):
    """Response from entity resolution"""
    resolved_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    resolution_confidence: Dict[str, float]
    duplicate_clusters: List[List[str]]


# Knowledge Graph Contracts

class KnowledgeQueryRequest(AtlasRequest):
    """Request for knowledge graph query"""
    query: str = Field(description="Knowledge graph query")
    query_type: str = Field(default="cypher", description="Query language type")
    limit: int = Field(default=100, description="Result limit")
    include_paths: bool = Field(default=False, description="Include relationship paths")


class KnowledgeQueryResponse(AtlasResponse):
    """Response from knowledge graph query"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    paths: Optional[List[Dict[str, Any]]]
    query_statistics: Dict[str, Any]


# Data Fusion Contracts

class DataIngestionRequest(AtlasRequest):
    """Request for data ingestion"""
    data_source: str = Field(description="Source system identifier")
    data_format: str = Field(description="Data format specification")
    data_payload: Union[Dict, List] = Field(description="Data to ingest")
    transformation_rules: Optional[Dict[str, Any]] = Field(default=None)
    validation_schema: Optional[Dict[str, Any]] = Field(default=None)


class DataIngestionResponse(AtlasResponse):
    """Response from data ingestion"""
    ingestion_id: str
    records_processed: int
    records_accepted: int
    records_rejected: int
    validation_errors: List[Dict[str, Any]]
    transformation_summary: Dict[str, Any]


# Client-Specific API Extensions

# Everything Business Intelligence APIs

class BusinessIntelligenceRequest(AtlasRequest):
    """Business intelligence specific request"""
    market_segment: Optional[str] = None
    analysis_period: str = "30d"
    competitive_analysis: bool = False
    risk_assessment: bool = False


class MarketAnalysisRequest(BusinessIntelligenceRequest):
    """Market analysis request for Everything platform"""
    market_segment: str = Field(description="Market segment to analyze")
    competitors: List[str] = Field(default_factory=list, description="Competitor list")
    metrics: List[str] = Field(default_factory=list, description="Metrics to analyze")


class MarketAnalysisResponse(AtlasResponse):
    """Market analysis response"""
    market_insights: Dict[str, Any]
    competitive_landscape: Dict[str, Any]
    trend_analysis: List[Dict[str, Any]]
    recommendations: List[str]
    risk_factors: List[Dict[str, Any]]


# Sourcing Supply Chain APIs

class SupplyChainRequest(AtlasRequest):
    """Supply chain specific request"""
    supplier_ids: List[str] = Field(default_factory=list)
    risk_categories: List[str] = Field(default_factory=list)
    assessment_horizon: str = "90d"


class SupplyRiskAssessmentRequest(SupplyChainRequest):
    """Supply risk assessment for Sourcing platform"""
    suppliers: List[str] = Field(description="Supplier identifiers")
    risk_factors: List[str] = Field(description="Risk factors to evaluate")


class SupplyRiskAssessmentResponse(AtlasResponse):
    """Supply risk assessment response"""
    risk_scores: Dict[str, float]
    risk_breakdown: Dict[str, Dict[str, float]]
    mitigation_strategies: List[Dict[str, Any]]
    monitoring_recommendations: List[str]
    supplier_rankings: List[Dict[str, Any]]


# API Gateway Contracts

class AuthenticationRequest(BaseModel):
    """API authentication request"""
    client_id: str
    client_secret: str
    scopes: List[ServiceScope]
    grant_type: str = "client_credentials"


class AuthenticationResponse(BaseModel):
    """API authentication response"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    scope: str
    issued_at: datetime = Field(default_factory=datetime.now)


# Error Response Models

class AtlasError(BaseModel):
    """Standard Atlas error response"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    help_url: Optional[str] = None
    request_id: Optional[str] = None


class ValidationError(AtlasError):
    """Validation error response"""
    field_errors: List[Dict[str, str]]


class AuthorizationError(AtlasError):
    """Authorization error response"""
    required_scopes: List[str]
    current_scopes: List[str]


# Service Health and Status

class ServiceHealthResponse(BaseModel):
    """Service health check response"""
    service: str
    status: str  # healthy, degraded, unhealthy
    version: str
    uptime: int  # seconds
    last_updated: datetime
    dependencies: Dict[str, str]  # dependency -> status


class ServiceMetricsResponse(BaseModel):
    """Service metrics response"""
    requests_per_second: float
    average_response_time: float
    error_rate: float
    active_connections: int
    queue_depth: int
    resource_utilization: Dict[str, float]


# API Rate Limiting

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


# Webhook Contracts

class WebhookEvent(BaseModel):
    """Webhook event notification"""
    event_id: str
    event_type: str
    timestamp: datetime
    client_id: str
    data: Dict[str, Any]
    signature: str  # HMAC signature for verification


# Configuration Models

class ClientConfiguration(BaseModel):
    """Client configuration settings"""
    client_id: str
    client_name: str
    allowed_scopes: List[ServiceScope]
    rate_limits: Dict[str, int]
    webhook_urls: List[str] = Field(default_factory=list)
    data_retention_days: int = 365
    compliance_requirements: List[str] = Field(default_factory=list)


# Atlas Service Registry

ATLAS_SERVICES = {
    "intelligence": {
        "base_url": "/api/atlas/v1/intelligence",
        "endpoints": {
            "analyze": "/analyze",
            "reason": "/reason", 
            "synthesize": "/synthesize"
        },
        "auth_required": True,
        "rate_limit": 100  # requests per minute
    },
    "analytics": {
        "base_url": "/api/atlas/v1/analytics",
        "endpoints": {
            "predict": "/predict",
            "detect": "/detect",
            "analyze": "/analyze",
            "forecast": "/forecast"
        },
        "auth_required": True,
        "rate_limit": 200
    },
    "entities": {
        "base_url": "/api/atlas/v1/entities", 
        "endpoints": {
            "resolve": "/resolve",
            "merge": "/merge",
            "relate": "/relate",
            "search": "/search"
        },
        "auth_required": True,
        "rate_limit": 300
    },
    "knowledge": {
        "base_url": "/api/atlas/v1/knowledge",
        "endpoints": {
            "query": "/query",
            "infer": "/infer", 
            "traverse": "/traverse",
            "update": "/update"
        },
        "auth_required": True,
        "rate_limit": 150
    },
    "fusion": {
        "base_url": "/api/atlas/v1/fusion",
        "endpoints": {
            "ingest": "/ingest",
            "transform": "/transform", 
            "merge": "/merge",
            "validate": "/validate"
        },
        "auth_required": True,
        "rate_limit": 50  # Lower limit for data ingestion
    }
}


# Client SDK Configuration

EVERYTHING_CLIENT_CONFIG = ClientConfiguration(
    client_id="everything_business_intelligence",
    client_name="Everything Business Intelligence Platform",
    allowed_scopes=[
        ServiceScope.ANALYTICS_PREDICT,
        ServiceScope.ENTITIES_RESOLVE,
        ServiceScope.KNOWLEDGE_QUERY,
        ServiceScope.FUSION_INGEST
    ],
    rate_limits={
        "analytics": 200,
        "entities": 300, 
        "knowledge": 150,
        "fusion": 50
    },
    compliance_requirements=["GDPR"]
)

SOURCING_CLIENT_CONFIG = ClientConfiguration(
    client_id="sourcing_supply_chain",
    client_name="Sourcing Supply Chain Intelligence",
    allowed_scopes=[
        ServiceScope.ANALYTICS_PREDICT,
        ServiceScope.ENTITIES_RESOLVE, 
        ServiceScope.KNOWLEDGE_TRAVERSE,
        ServiceScope.FUSION_INGEST
    ],
    rate_limits={
        "analytics": 100,
        "entities": 200,
        "knowledge": 100, 
        "fusion": 30
    },
    compliance_requirements=["GDPR"]
)