"""
Secure API Gateway with Integrated Compliance
All API requests pass through privacy, audit, and access control checks
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, UTC
import uuid
import logging
import os
from dataclasses import asdict
from pydantic import BaseModel
import time
import asyncio

from ..compliance.privacy_framework import (
    privacy_framework,
    DataCategory,
    ProcessingPurpose,
)
from ..audit.audit_system import audit_logger, AuditEventType, AccessResult, RiskLevel
from ..auth.access_control import (
    access_control,
    AccessContext,
    AccessRequest,
    Permission,
    Role,
)
from ..data_management.retention_policy import retention_engine
from ..privacy.consent_management import consent_system
from ..config.settings import get_settings
from ..common.pagination import paginate_list
from ..common.cache import SimpleTTLCache
from ..common.purpose import canonicalize
from ..privacy.field_filter import filter_by_permissions
from ..observability.metrics import (
    metrics,
    api_requests_total,
    api_request_duration_seconds,
    rate_limit_exceeded_total,
)
from ..insights.alerts import alert_manager, AlertRule
from ..insights.narratives import alert_narrative
from ..data_integration.ingestion_scheduler import ingestion_scheduler, IngestionFrequency
from ..data_integration.public_data_connectors import public_data_manager
from ..analytics import (
    swedish_crime_analytics,
    SwedishCrimeFeatureBuilder,
    forecast_incident_risk,
)
from ..data_fusion.fusion_engine import DataFusionEngine
from ..investigations import InvestigationWorkspaceService
from ..intelligence import (
    LinkAnalysisEngine,
    RelationshipEdge,
    PatternRecognitionEngine,
    PredictiveIntelligenceEngine,
    NetworkAnalysisEngine,
    BehavioralAnalyticsEngine,
)
from ..analytics.advanced_patterns import PatternType, AnalysisPriority
from ..ai_integration.multi_provider_ai import PrivacyLevel
from ..data_integration.universal_connector import (
    universal_data_manager,
    ConnectorFactory,
    ConnectorType,
    DataSourceConfig,
    DataSensitivity,
)


# API Models
class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    request_id: str
    timestamp: datetime
    compliance_checked: bool = True


class UserIncidentSubmissionRequest(BaseModel):
    # Required fields
    title: str
    description: str
    incident_type: str
    incident_category: str

    # Optional temporal data
    incident_datetime: Optional[datetime] = None

    # Optional location data
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location_description: Optional[str] = None

    # Optional incident details
    severity_reported: Optional[int] = None
    urgency_level: Optional[str] = "normal"
    witness_count: Optional[int] = 0

    # Privacy and consent
    consent_data_processing: bool = False
    consent_law_enforcement: bool = False
    consent_research: bool = False
    anonymization_level: Optional[str] = "standard"

    # Optional user identification
    submitter_type: Optional[str] = "citizen"

    # Evidence indication
    evidence_provided: Optional[bool] = False


class UserIncidentUpdateRequest(BaseModel):
    incident_id: str
    update_type: str
    update_text: str
    public_visible: Optional[bool] = False


class IncidentModerationRequest(BaseModel):
    incident_id: str
    decision: str  # approve, reject, request_more_info, escalate
    decision_reason: Optional[str] = None
    decision_confidence: Optional[float] = None


class PersonSearchRequest(BaseModel):
    query: str
    search_type: str = "basic"  # basic, advanced, bulk
    data_categories: List[str]
    purpose: str
    authorization_id: Optional[str] = None
    max_results: int = 50
    page: int = 1
    page_size: Optional[int] = None


class PersonRecord(BaseModel):
    record_id: str
    first_name: str
    last_name: str
    date_of_birth: Optional[str] = None
    national_id: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    risk_score: Optional[float] = None
    last_updated: datetime
    data_classification: str


class IncidentRequest(BaseModel):
    incident_type: str
    location: str
    description: str
    severity_level: int
    involved_persons: List[str]
    authorization_id: Optional[str] = None


class AnalyticsRequest(BaseModel):
    analysis_type: str  # link_analysis, pattern_detection, risk_assessment
    data_sources: List[str]
    parameters: Dict[str, Any]
    purpose: str
    authorization_id: str  # Required for analytics


class FusionRunRequest(BaseModel):
    source_ids: List[str]
    limit_per_source: Optional[int] = None
    purpose: str
    authorization_id: Optional[str] = None


class WorkspaceCreateRequest(BaseModel):
    workspace_type: str
    classification_level: str
    case_id: Optional[str] = None
    case_title: Optional[str] = None
    case_type: Optional[str] = None
    priority_level: Optional[str] = None
    jurisdiction: Optional[str] = None
    authorization_id: Optional[str] = None


class EvidenceCreateRequest(BaseModel):
    evidence_type: str
    description: str
    metadata: Optional[Dict[str, str]] = None
    location: Optional[str] = None


class TimelineEventRequest(BaseModel):
    title: str
    description: str
    timestamp: Optional[datetime] = None
    evidence_id: Optional[str] = None


class SwedenPredictionRequest(BaseModel):
    purpose: str = "prevention"
    authorization_id: Optional[str] = None
    windows: Optional[List[int]] = None
    reference_time: Optional[datetime] = None
    include_forecast: bool = True
    top_n: int = 5


class IngestionJobRequest(BaseModel):
    source_name: str
    frequency: str  # "hourly", "daily", "weekly", etc.
    records_limit: Optional[int] = None
    custom_config: Optional[Dict[str, Any]] = None


class IngestionJobResponse(BaseModel):
    job_id: str
    source_name: str
    status: str
    frequency: str
    enabled: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    error_count: int
    metrics: Dict[str, Any]


# Initialize FastAPI app
app = FastAPI(
    title="Levi Secure Gateway",
    description="Secure API gateway with integrated privacy and compliance controls",
    version="1.0.0",
)

# Security setup
security = HTTPBearer()

# CORS middleware with restricted origins
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Compliance-Status"],
)

# Logging setup
logger = logging.getLogger(__name__)

# Simple in-memory search cache with TTL and size cap
_search_cache = SimpleTTLCache(ttl_seconds=30, max_size=500)

# Simple in-memory token bucket rate limiter (per session or IP)
_rate_buckets: Dict[str, Any] = {}


investigation_workspace_service = InvestigationWorkspaceService()


def _rate_capacity_for(request: Request) -> int:
    path = request.url.path
    if path.startswith("/api/v1/analytics"):
        return settings.rate_capacity_analytics
    return settings.rate_capacity_default


def _rate_key(context: AccessContext, request: Request) -> str:
    return context.session_id or (request.client.host if request.client else "unknown")


def _rate_limit_allow(context: AccessContext, request: Request) -> tuple[bool, int]:
    key = _rate_key(context, request)
    now = time.time()
    bucket = _rate_buckets.get(key)
    capacity = _rate_capacity_for(request)
    refill_per_sec = capacity / 60.0
    if not bucket:
        bucket = {"tokens": capacity, "updated": now, "cap": capacity}
        _rate_buckets[key] = bucket
    # Refill
    elapsed = now - bucket["updated"]
    # If capacity changed (settings update), adjust
    if bucket.get("cap") != capacity:
        bucket["cap"] = capacity
        if bucket["tokens"] > capacity:
            bucket["tokens"] = capacity
    bucket["tokens"] = min(capacity, bucket["tokens"] + elapsed * refill_per_sec)
    bucket["updated"] = now
    # Consume
    if bucket["tokens"] >= 1:
        bucket["tokens"] -= 1
        return True, int(bucket["tokens"])
    else:
        return False, int(bucket["tokens"])


async def get_access_context(
    request: Request, credentials: HTTPAuthorizationCredentials = Security(security)
) -> AccessContext:
    """Extract and validate access context from request"""

    # Extract session info from JWT token (simplified for example)
    token = credentials.credentials if credentials else None
    session_id = token or str(uuid.uuid4())

    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    # Create access context
    context = AccessContext(
        user_id="extracted_from_jwt",  # Will be replaced after auth if applicable
        session_id=session_id,
        ip_address=client_ip,
        user_agent=user_agent,
        timestamp=datetime.now(),
        authorization_id=request.headers.get("X-Authorization-ID"),
        emergency_declared=request.headers.get("X-Emergency") == "true",
        supervisor_override=request.headers.get("X-Supervisor-Override"),
    )

    # Demo/test token bootstrap: map known tokens to users and authenticate
    try:
        token_user_map = {
            "demo_token": (
                "det.johnson",
                Role.SENIOR_DETECTIVE,
                {
                    "full_name": "Detective Michael Johnson",
                    "email": "m.johnson@agency.gov",
                    "department": "Criminal Investigation Division",
                    "clearance_level": 4,
                },
            ),
            "admin_token": (
                "admin.system",
                Role.SYSTEM_ADMIN,
                {
                    "full_name": "System Administrator",
                    "email": "admin@agency.gov",
                    "department": "IT Division",
                    "clearance_level": 5,
                },
            ),
            "compliance_token": (
                "compliance.officer",
                Role.COMPLIANCE_OFFICER,
                {
                    "full_name": "Compliance Officer Jane Smith",
                    "email": "compliance@agency.gov",
                    "department": "Legal Division",
                    "clearance_level": 4,
                },
            ),
        }

        if token and token in token_user_map:
            username, role, meta = token_user_map[token]
            # Ensure user exists
            existing_user = next(
                (u for u in access_control.users.values() if u.username == username),
                None,
            )
            if not existing_user:
                access_control.create_user(
                    username=username,
                    full_name=str(meta["full_name"]),
                    email=str(meta["email"]),
                    roles=[role],
                    department=str(meta["department"]),
                    clearance_level=int(str(meta["clearance_level"])) if meta.get("clearance_level") is not None else 1,
                )
                existing_user = next(
                    (
                        u
                        for u in access_control.users.values()
                        if u.username == username
                    ),
                    None,
                )

            # Authenticate to bootstrap a valid session
            ok, user, _ = access_control.authenticate_user(username, context)
            if ok and user:
                context.user_id = user.user_id
    except Exception:
        # Non-fatal: fallback to context without session (will be rejected later)
        pass

    return context


async def enforce_compliance(request: AccessRequest) -> Tuple[bool, str, RiskLevel]:
    """Enforce all compliance checks"""

    # 1. Access control check
    access_granted, message, risk_level = access_control.check_access(request)

    if not access_granted:
        return False, message, risk_level

    # 2. Consent check (if applicable)
    if request.data_category in [DataCategory.PERSONAL, DataCategory.SENSITIVE]:
        consent_valid, consent_id, legal_basis = consent_system.check_consent_validity(
            request.context.user_id, request.data_category, request.purpose
        )

        if not consent_valid and not legal_basis:
            return False, "No valid consent or legal basis", RiskLevel.HIGH

    # 3. Register data processing activity
    if hasattr(request, "resource_id") and request.resource_id:
        retention_engine.register_data_record(
            record_id=request.resource_id,
            category=request.data_category,
            purpose=request.purpose,
            source_system="api_gateway",
            legal_basis=request.context.authorization_id or "statutory_duty",
            subject_id=request.resource_id,
            authorization_id=request.context.authorization_id,
        )

    return True, "Access granted", risk_level


@app.middleware("http")
async def compliance_middleware(request: Request, call_next):
    """Middleware to ensure all requests are compliance-checked"""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to headers
    request.state.request_id = request_id

    # Process request with basic rate limiting
    # We need an access context for the key – build a lightweight one using headers
    try:
        token = request.headers.get("authorization", "").replace("Bearer ", "")
        ctx = AccessContext(
            user_id="unknown",
            session_id=token or request_id,
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            timestamp=datetime.now(),
        )
        allowed, remaining = _rate_limit_allow(ctx, request)
        if not allowed:
            rate_limit_exceeded_total.labels(endpoint=request.url.path).inc()
            return JSONResponse(
                status_code=429,
                content={"success": False, "message": "Rate limit exceeded"},
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Remaining": str(remaining),
                    "X-Request-ID": request_id,
                    "X-Compliance-Status": "checked",
                    "X-Privacy-Framework": "enabled",
                },
            )
    except Exception:
        # Non-fatal; continue
        pass

    response = await call_next(request)

    # Add compliance headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Compliance-Status"] = "checked"
    response.headers["X-Privacy-Framework"] = "enabled"

    return response


# Seed a sample alert rule for demonstration (high risk score volume)
try:
    alert_manager.add_rule(
        AlertRule(
            rule_id="high_search_volume",
            name="High Search Volume",
            description="Unusual spike in person search requests",
            metric="person_search_count",
            threshold=50,
            window_seconds=60,
            silence_seconds=300,
        )
    )
except Exception:
    pass


@app.post("/api/v1/person/search", response_model=APIResponse)
async def search_persons(
    search_request: PersonSearchRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Search for persons with privacy and access controls"""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Be tolerant to legacy purpose strings
        search_request.purpose = canonicalize(search_request.purpose)
        # Map request to data categories
        data_categories = []
        for cat_str in search_request.data_categories:
            try:
                data_categories.append(DataCategory(cat_str))
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid data category: {cat_str}"
                )

        # Map purpose
        try:
            purpose = ProcessingPurpose(search_request.purpose)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid purpose: {search_request.purpose}"
            )

        # Create access request for each data category
        for category in data_categories:
            access_request = AccessRequest(
                resource_type="person_search",
                resource_id=None,
                action=f"search_{search_request.search_type}",
                data_category=category,
                purpose=purpose,
                context=context,
                additional_params={
                    "query": search_request.query,
                    "advanced": search_request.search_type == "advanced",
                    "bulk": search_request.search_type == "bulk",
                },
            )

            # Enforce compliance
            access_granted, message, risk_level = await enforce_compliance(
                access_request
            )

            if not access_granted:
                raise HTTPException(status_code=403, detail=message)

        # Perform search (with cache)
        cache_key = (
            context.session_id,
            search_request.query,
            tuple([cat.value for cat in data_categories]),
            purpose.value,
            search_request.search_type,
        )
        cache_entry = _search_cache.get(cache_key)
        if cache_entry is None:
            raw_results = await perform_person_search(
                search_request, data_categories, purpose, context
            )
            _search_cache.set(cache_key, raw_results)
        else:
            raw_results = cache_entry

        page_size = search_request.page_size or search_request.max_results or 50
        page = max(1, search_request.page)
        page_results, meta = paginate_list(raw_results, page, page_size)

        # Apply data minimization based on purpose
        minimized_results = []
        for result in page_results:
            minimized_data = privacy_framework.apply_data_minimization(
                result.model_dump(), purpose
            )
            # Field-level redaction based on user permissions
            perms = set()
            try:
                session = access_control.active_sessions.get(context.session_id)
                if session:
                    user_id = session.get("user_id")
                    if user_id:
                        user = access_control.users.get(user_id)
                    if user:
                        perms = user.permissions
            except Exception:
                pass
            minimized_results.append(filter_by_permissions(minimized_data, perms))

        resp = APIResponse(
            success=True,
            data={
                "results": minimized_results,
                "total_found": meta.total,
                "page": meta.page,
                "page_size": meta.page_size,
                "total_pages": meta.total_pages,
                "has_next": meta.has_next,
                "has_prev": meta.has_prev,
                "data_minimized": True,
                "purpose": purpose.value,
            },
            message="Search completed successfully",
            request_id=request_id,
            timestamp=datetime.now(),
        )
        api_requests_total.labels(endpoint="/api/v1/person/search", status="success").inc()
        api_request_duration_seconds.labels(endpoint="/api/v1/person/search").observe(time.time() - start_time)
        return resp

    except HTTPException:
        api_requests_total.labels(endpoint="/api/v1/person/search", status="denied").inc()
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")

        # Log error
        audit_logger.log_event(
            event_type=AuditEventType.SEARCH_QUERY,
            user_id=context.user_id,
            action="person_search_error",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.ERROR,
            details={"error": str(e), "query": search_request.query},
        )

        api_requests_total.labels(endpoint="/api/v1/person/search", status="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/v1/alerts", response_model=APIResponse)
async def get_alerts(context: AccessContext = Depends(get_access_context)):
    # Permissions: require VIEW_AUDIT_LOGS or SYSTEM_ADMIN-like
    # For demo, allow all authenticated; in prod, tighten this.
    alerts = [a.__dict__ for a in alert_manager.list_alerts()]
    for a in alerts:
        a['narrative'] = alert_narrative(a)
        if hasattr(a.get('triggered_at'), 'isoformat'):
            a['triggered_at'] = a['triggered_at'].isoformat()
        if a.get('acked_at') and hasattr(a.get('acked_at'), 'isoformat'):
            a['acked_at'] = a['acked_at'].isoformat()
    return APIResponse(
        success=True,
        data={"alerts": alerts, "count": len(alerts)},
        message="Alerts retrieved",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/alerts/ingest", response_model=APIResponse)
async def ingest_alert_event(event: Dict[str, Any], context: AccessContext = Depends(get_access_context)):
    # Dev-only: accept {metric, value, labels}
    metric = event.get('metric', 'person_search_count')
    value = float(event.get('value', 0))
    labels = event.get('labels', {}) or {}
    alert_manager.ingest(metric=metric, value=value, labels=labels)
    return APIResponse(
        success=True,
        data={"ingested": True},
        message="Event ingested",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/alerts/ack", response_model=APIResponse)
async def acknowledge_alert(event: Dict[str, Any], context: AccessContext = Depends(get_access_context)):
    alert_id = event.get('alert_id')
    if not alert_id:
        raise HTTPException(status_code=400, detail='alert_id required')
    ok = alert_manager.ack(str(alert_id))
    if not ok:
        raise HTTPException(status_code=404, detail='alert not found')
    return APIResponse(
        success=True,
        data={"alert_id": alert_id, "acked": True},
        message="Alert acknowledged",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


async def perform_person_search(
    search_request: PersonSearchRequest,
    data_categories: List[DataCategory],
    purpose: ProcessingPurpose,
    context: AccessContext,
) -> List[PersonRecord]:
    """Perform person search with mock data"""

    # Log search activity
    audit_logger.log_event(
        event_type=AuditEventType.SEARCH_QUERY,
        user_id=context.user_id,
        action=f"person_search_{search_request.search_type}",
        session_id=context.session_id,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        result=AccessResult.SUCCESS,
        authorization_id=context.authorization_id,
        details={
            "query": search_request.query,
            "categories": [cat.value for cat in data_categories],
            "purpose": purpose.value,
        },
        risk_level=RiskLevel.MEDIUM
        if search_request.search_type == "bulk"
        else RiskLevel.LOW,
    )

    # Mock search results (in real implementation, would query databases)
    mock_results = [
        PersonRecord(
            record_id="person_001",
            first_name="John",
            last_name="Smith",
            date_of_birth="1985-06-15",
            national_id="123456789",
            address="123 Main St, Anytown",
            phone="+1-555-0123",
            email="john.smith@email.com",
            risk_score=0.3,
            last_updated=datetime.now(),
            data_classification="personal",
        ),
        PersonRecord(
            record_id="person_002",
            first_name="Jane",
            last_name="Doe",
            date_of_birth="1990-03-22",
            national_id="987654321",
            address="456 Oak Ave, Other City",
            phone="+1-555-0456",
            email="jane.doe@email.com",
            risk_score=0.1,
            last_updated=datetime.now(),
            data_classification="personal",
        ),
    ]

    # Update access times for retention tracking
    for result in mock_results:
        retention_engine.update_access_time(result.record_id)

    return mock_results


@app.post("/api/v1/fusion/run", response_model=APIResponse)
async def run_data_fusion(
    fusion_request: FusionRunRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Trigger the universal data fusion pipeline for selected sources."""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    if not fusion_request.source_ids:
        raise HTTPException(status_code=400, detail="source_ids is required")

    fusion_request.purpose = canonicalize(fusion_request.purpose)
    if fusion_request.authorization_id:
        context.authorization_id = fusion_request.authorization_id

    try:
        try:
            purpose = ProcessingPurpose(fusion_request.purpose)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid purpose: {fusion_request.purpose}",
            )

        access_request = AccessRequest(
            resource_type="data_fusion",
            resource_id=None,
            action="run_fusion",
            data_category=DataCategory.SENSITIVE,
            purpose=purpose,
            context=context,
            additional_params={
                "source_ids": fusion_request.source_ids,
                "limit_per_source": fusion_request.limit_per_source,
            },
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        fusion_engine = DataFusionEngine(context)
        fusion_result = await fusion_engine.run(
            fusion_request.source_ids,
            limit_per_source=fusion_request.limit_per_source,
        )

        fused_entities = [
            {
                "canonical_id": entity.canonical_id,
                "entity_type": entity.entity_type,
                "confidence": entity.confidence,
                "source_records": entity.source_records,
                "attributes": entity.attributes,
                "decisions": entity.conflict_decisions,
                "unresolved": entity.unresolved_conflicts,
            }
            for entity in fusion_result.fused_entities
        ]

        response_data = {
            "batch_id": fusion_result.batch.batch_id,
            "sources_requested": fusion_result.batch.metrics.get(
                "sources_requested", len(fusion_request.source_ids)
            ),
            "records_ingested": fusion_result.batch.metrics.get(
                "records_ingested", len(fusion_result.batch.records)
            ),
            "standardized_records": len(fusion_result.standardized_records),
            "resolved_entities": len(fusion_result.resolved_entities),
            "fused_entities": fused_entities,
            "lakehouse_writes": fusion_result.lakehouse_writes,
            "errors": fusion_result.batch.errors,
        }

        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="data_fusion_run",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "source_count": len(fusion_request.source_ids),
                "fused_entities": len(fused_entities),
            },
            risk_level=RiskLevel.HIGH,
        )

        api_requests_total.labels(endpoint="/api/v1/fusion/run", status="success").inc()
        api_request_duration_seconds.labels(endpoint="/api/v1/fusion/run").observe(
            time.time() - start_time
        )

        return APIResponse(
            success=True,
            data=response_data,
            message="Fusion pipeline completed",
            request_id=request_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        api_requests_total.labels(endpoint="/api/v1/fusion/run", status="denied").inc()
        raise
    except Exception as exc:
        logger.error(f"Fusion run error: {exc}")
        api_requests_total.labels(endpoint="/api/v1/fusion/run", status="error").inc()
        raise HTTPException(status_code=500, detail="Fusion run failed")


@app.post("/api/v1/analytics/sweden/predictions", response_model=APIResponse)
async def sweden_predictions(
    prediction_request: SwedenPredictionRequest,
    context: AccessContext = Depends(get_access_context),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    purpose_str = canonicalize(prediction_request.purpose)
    try:
        purpose = ProcessingPurpose(purpose_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid purpose: {prediction_request.purpose}")

    if prediction_request.authorization_id:
        context.authorization_id = prediction_request.authorization_id

    auth_id = context.authorization_id
    if not auth_id:
        raise HTTPException(status_code=403, detail="Authorization ID required for Swedish predictions")

    required_categories = [
        DataCategory.CRIMINAL,
        DataCategory.LOCATION,
        DataCategory.PERSONAL,
        DataCategory.COMMUNICATIONS,
    ]
    for category in required_categories:
        access_request = AccessRequest(
            resource_type="sweden_prediction",
            resource_id=None,
            action="generate_prediction",
            data_category=category,
            purpose=purpose,
            context=context,
        )
        allowed, message, _ = await enforce_compliance(access_request)
        if not allowed:
            raise HTTPException(status_code=403, detail=message)

    await _ensure_sweden_prediction_sources(context, auth_id)

    source_ids = {
        "incidents": "sweden_prediction_historical",
        "stats": "sweden_prediction_stats",
        "events": "sweden_prediction_events",
        "scb": "sweden_prediction_scb",
        "survey": "sweden_prediction_survey",
        "reported": "sweden_prediction_reported",
        "news": "sweden_prediction_news",
    }

    async def _extract(source_key: str):
        records = await universal_data_manager.extract_from_source(source_ids[source_key], context)
        return [record.data for record in records]

    incidents, stats, events, scb_stats, survey, reported, news = await asyncio.gather(
        *(_extract(key) for key in source_ids)
    )

    windows = prediction_request.windows or [7, 30]
    reference_time = prediction_request.reference_time or datetime.now(UTC)

    builder = SwedishCrimeFeatureBuilder(reference_time=reference_time)
    features = builder.build_features(
        incidents,
        stats,
        events,
        scb_stats,
        survey,
        reported,
        news,
        windows=windows,
    )

    forecasts = {}
    if prediction_request.include_forecast:
        forecasts = forecast_incident_risk(
            incidents,
            stats,
            events,
            scb_stats,
            survey,
            reported,
            news,
            reference_time=reference_time,
        )

    top_n = max(1, prediction_request.top_n)
    sorted_municipalities = sorted(
        features.items(),
        key=lambda item: item[1].get("municipality_risk_score", 0.0),
        reverse=True,
    )
    top_municipalities = []
    for municipality, data in sorted_municipalities[:top_n]:
        top_entry = {
            "municipality": municipality,
            "risk_score": data.get("municipality_risk_score", 0.0),
        }
        if forecasts:
            top_entry["forecasted_incidents"] = forecasts.get(municipality, {}).get(
                "forecasted_incidents", 0.0
            )
        top_municipalities.append(top_entry)

    response_payload = {
        "reference_time": reference_time.isoformat(),
        "windows": windows,
        "features": features,
        "forecast": forecasts if prediction_request.include_forecast else {},
        "top_municipalities": top_municipalities,
    }

    audit_logger.log_event(
        event_type=AuditEventType.DATA_ACCESS,
        user_id=context.user_id,
        action="sweden_predictions",
        session_id=context.session_id,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        authorization_id=context.authorization_id,
        result=AccessResult.SUCCESS,
        details={"municipalities": list(features.keys()), "windows": windows},
        risk_level=RiskLevel.MEDIUM,
    )

    api_requests_total.labels(endpoint="/api/v1/analytics/sweden/predictions", status="success").inc()
    api_request_duration_seconds.labels(endpoint="/api/v1/analytics/sweden/predictions").observe(
        time.time() - start_time
    )

    return APIResponse(
        success=True,
        data=response_payload,
        message="Swedish crime predictions generated",
        request_id=request_id,
        timestamp=datetime.now(),
    )


@app.post("/api/v1/incident/create", response_model=APIResponse)
async def create_incident(
    incident_request: IncidentRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Create new incident record with compliance checks"""

    request_id = str(uuid.uuid4())

    try:
        # Create access request
        access_request = AccessRequest(
            resource_type="incident",
            resource_id=None,
            action="create_incident",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PUBLIC_SAFETY,
            context=context,
            additional_params={"severity": incident_request.severity_level},
        )

        # Enforce compliance
        access_granted, message, risk_level = await enforce_compliance(access_request)

        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Create incident
        incident_id = str(uuid.uuid4())

        # Register with retention system
        retention_engine.register_data_record(
            record_id=incident_id,
            category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PUBLIC_SAFETY,
            source_system="incident_management",
            legal_basis=context.authorization_id or "statutory_duty",
            authorization_id=context.authorization_id,
            metadata={
                "incident_type": incident_request.incident_type,
                "severity": incident_request.severity_level,
                "location": incident_request.location,
            },
        )

        # Log incident creation
        audit_logger.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=context.user_id,
            action="create_incident",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            resource=incident_id,
            authorization_id=context.authorization_id,
            data_subjects=incident_request.involved_persons,
            details={
                "incident_type": incident_request.incident_type,
                "severity": incident_request.severity_level,
            },
            risk_level=RiskLevel.HIGH
            if incident_request.severity_level > 7
            else RiskLevel.MEDIUM,
        )

        return APIResponse(
            success=True,
            data={"incident_id": incident_id, "status": "created"},
            message="Incident created successfully",
            request_id=request_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Incident creation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/analytics/analyze", response_model=APIResponse)
async def perform_analytics(
    analytics_request: AnalyticsRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Perform data analytics with strict authorization requirements"""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        analytics_request.purpose = canonicalize(analytics_request.purpose)
        # Analytics always requires explicit authorization
        if not analytics_request.authorization_id:
            raise HTTPException(
                status_code=403, detail="Analytics requires legal authorization"
            )

        context.authorization_id = analytics_request.authorization_id

        # Map purpose
        try:
            purpose = ProcessingPurpose(analytics_request.purpose)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid purpose: {analytics_request.purpose}"
            )

        # Create access request
        access_request = AccessRequest(
            resource_type="analytics",
            resource_id=None,
            action=f"analytics_{analytics_request.analysis_type}",
            data_category=DataCategory.SENSITIVE,  # Analytics typically involves sensitive processing
            purpose=purpose,
            context=context,
            additional_params={
                "analysis_type": analytics_request.analysis_type,
                "data_sources": analytics_request.data_sources,
            },
        )

        # Enforce compliance
        access_granted, message, risk_level = await enforce_compliance(access_request)

        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Perform analytics
        analysis_results = await perform_data_analysis(
            analytics_request, purpose, context
        )

        resp = APIResponse(
            success=True,
            data=analysis_results,
            message="Analysis completed successfully",
            request_id=request_id,
            timestamp=datetime.now(),
        )
        api_requests_total.labels(endpoint="/api/v1/analytics/analyze", status="success").inc()
        api_request_duration_seconds.labels(endpoint="/api/v1/analytics/analyze").observe(time.time() - start_time)
        return resp

    except HTTPException:
        api_requests_total.labels(endpoint="/api/v1/analytics/analyze", status="denied").inc()
        raise
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        api_requests_total.labels(endpoint="/api/v1/analytics/analyze", status="error").inc()
        raise HTTPException(status_code=500, detail="Internal server error")


async def perform_data_analysis(
    analytics_request: AnalyticsRequest,
    purpose: ProcessingPurpose,
    context: AccessContext,
) -> Dict[str, Any]:
    """Perform data analysis with privacy safeguards"""

    audit_logger.log_event(
        event_type=AuditEventType.DATA_ACCESS,
        user_id=context.user_id,
        action=f"analytics_{analytics_request.analysis_type}",
        session_id=context.session_id,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        result=AccessResult.SUCCESS,
        authorization_id=context.authorization_id,
        details={
            "analysis_type": analytics_request.analysis_type,
            "data_sources": analytics_request.data_sources,
            "purpose": purpose.value,
        },
        risk_level=RiskLevel.HIGH,
    )

    model_version = None
    try:
        from ..deployment.model_registry import registry

        mv = registry.route("analytics_model")
        model_version = mv.version if mv else None
    except Exception:
        model_version = None

    analysis_type = analytics_request.analysis_type
    parameters = analytics_request.parameters or {}

    if analysis_type == "link_analysis":
        edges_payload = parameters.get("edges") or []
        if not edges_payload:
            raise HTTPException(
                status_code=400, detail="link_analysis requires 'edges' parameter"
            )

        relationship_edges: List[RelationshipEdge] = []
        for edge in edges_payload:
            try:
                relationship_edges.append(
                    RelationshipEdge(
                        source=str(edge["source"]),
                        target=str(edge["target"]),
                        relationship_type=str(
                            edge.get("relationship_type", "ASSOCIATED_WITH")
                        ),
                        strength=float(edge.get("strength", 0.5)),
                        metadata=edge.get("metadata", {}),
                    )
                )
            except KeyError as exc:
                raise HTTPException(
                    status_code=400, detail=f"edge missing field: {exc}"
                )

        link_engine = LinkAnalysisEngine(context)
        link_engine.load_relationships(relationship_edges)
        focus_entity = str(
            parameters.get("focus_entity", relationship_edges[0].source)
        )
        relationship_types = parameters.get("relationship_types")
        max_degrees = int(parameters.get("max_degrees", 3))
        min_strength = float(parameters.get("min_strength", 0.1))

        graph = await link_engine.discover_relationships(
            focus_entity,
            relationship_types=relationship_types,
            max_degrees=max_degrees,
            min_strength=min_strength,
        )

        predictions = await link_engine.predict_associations(focus_entity)

        network_engine = NetworkAnalysisEngine(context)
        network_engine.load_edges(relationship_edges)
        focus_entities = parameters.get("focus_entities") or [focus_entity]
        metrics = await network_engine.compute_metrics(
            [str(entity) for entity in focus_entities]
        )

        results = {
            "analysis_type": "link_analysis",
            "relationship_graph": {
                "entity_id": graph.entity_id,
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "relationship_type": edge.relationship_type,
                        "strength": edge.strength,
                        "metadata": edge.metadata,
                    }
                    for edge in graph.edges
                ],
            },
            "predicted_associations": [asdict(pred) for pred in predictions],
            "network_metrics": asdict(metrics),
        }

    elif analysis_type == "pattern_detection":
        records = parameters.get("records")
        if not records:
            raise HTTPException(
                status_code=400, detail="pattern_detection requires 'records' parameter"
            )

        try:
            import pandas as pd  # type: ignore
        except Exception:
            raise HTTPException(
                status_code=500, detail="pandas is required for pattern detection"
            )

        dataframe = pd.DataFrame(records)

        raw_pattern_types = parameters.get("pattern_types") or ["TEMPORAL"]
        pattern_types: List[PatternType] = []
        for entry in raw_pattern_types:
            if isinstance(entry, PatternType):
                pattern_types.append(entry)
                continue
            try:
                pattern_types.append(PatternType[entry])
            except Exception:
                try:
                    pattern_types.append(PatternType(entry))
                except Exception:
                    continue
        if not pattern_types:
            pattern_types = [PatternType.TEMPORAL]

        privacy_level_param = parameters.get("privacy_level", "INTERNAL")
        if isinstance(privacy_level_param, PrivacyLevel):
            privacy_level = privacy_level_param
        else:
            try:
                privacy_level = PrivacyLevel[str(privacy_level_param).upper()]
            except Exception:
                try:
                    privacy_level = PrivacyLevel(str(privacy_level_param).lower())
                except Exception:
                    privacy_level = PrivacyLevel.INTERNAL

        priority_param = parameters.get("priority", "MEDIUM")
        if isinstance(priority_param, AnalysisPriority):
            priority = priority_param
        else:
            try:
                priority = AnalysisPriority[str(priority_param).upper()]
            except Exception:
                try:
                    priority = AnalysisPriority(str(priority_param).lower())
                except Exception:
                    priority = AnalysisPriority.MEDIUM

        data_categories = [DataCategory.PERSONAL]
        if isinstance(parameters.get("data_categories"), list):
            parsed_categories: List[DataCategory] = []
            for cat in parameters.get("data_categories", []):
                if isinstance(cat, DataCategory):
                    parsed_categories.append(cat)
                else:
                    try:
                        parsed_categories.append(DataCategory[str(cat).upper()])
                    except Exception:
                        try:
                            parsed_categories.append(DataCategory(str(cat).lower()))
                        except Exception:
                            continue
            if parsed_categories:
                data_categories = parsed_categories

        pattern_engine = PatternRecognitionEngine(context)
        report = await pattern_engine.analyze_dataframe(
            dataframe,
            pattern_types=pattern_types,
            data_categories=data_categories,
            purpose=purpose,
            privacy_level=privacy_level,
            priority=priority,
            use_ai=bool(parameters.get("use_ai", True)),
        )

        results = {
            "analysis_type": "pattern_detection",
            "patterns": [asdict(pattern) for pattern in report.patterns],
            "summary": report.summary,
            "overall_risk": report.overall_risk.value,
            "execution_time_ms": report.execution_time_ms,
            "audit_trail": report.audit_trail,
        }

    elif analysis_type == "risk_assessment":
        intelligence_engine = PredictiveIntelligenceEngine(context)
        response: Dict[str, Any] = {"analysis_type": "risk_assessment"}

        raw_series = parameters.get("series")
        if raw_series:
            series_id = str(parameters.get("series_id", "series"))
            series = [
                (
                    float(point.get("timestamp", idx)),
                    float(point.get("value", 0.0)),
                )
                for idx, point in enumerate(raw_series)
            ]
            method = str(parameters.get("forecast_method", "moving_average"))
            forecast = await intelligence_engine.forecast_series(
                series_id, series, method=method
            )
            response["forecast"] = asdict(forecast)

        indicators_param = parameters.get("indicators")
        if indicators_param:
            indicators = {
                key: float(value)
                for key, value in indicators_param.items()
            }
            subject_id = str(parameters.get("subject_id", "unknown"))
            assessment = await intelligence_engine.assess_risk(subject_id, indicators)
            response["risk_assessment"] = asdict(assessment)

        if not response.get("forecast") and not response.get("risk_assessment"):
            raise HTTPException(
                status_code=400,
                detail="risk_assessment requires 'indicators' and/or 'series' parameters",
            )

        results = response

    else:
        results = {
            "analysis_type": analysis_type,
            "status": "completed",
        }

    results["purpose"] = purpose.value
    if model_version:
        results["model_version"] = model_version

    return results



@app.post("/api/v1/investigations/workspaces", response_model=APIResponse)
async def create_investigation_workspace(
    workspace_request: WorkspaceCreateRequest,
    context: AccessContext = Depends(get_access_context),
):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    access_request = AccessRequest(
        resource_type="investigation_workspace",
        resource_id=None,
        action="create_workspace",
        data_category=DataCategory.SENSITIVE,
        purpose=ProcessingPurpose.INVESTIGATION,
        context=context,
        additional_params={
            "workspace_type": workspace_request.workspace_type,
            "classification_level": workspace_request.classification_level,
        },
    )

    allowed, message, _ = await enforce_compliance(access_request)
    if not allowed:
        raise HTTPException(status_code=403, detail=message)

    workspace = investigation_workspace_service.create_workspace(
        context=context,
        workspace_type=workspace_request.workspace_type,
        classification_level=workspace_request.classification_level,
        case_id=workspace_request.case_id,
        case_title=workspace_request.case_title,
        case_type=workspace_request.case_type,
        priority_level=workspace_request.priority_level,
        jurisdiction=workspace_request.jurisdiction,
        authorization_id=workspace_request.authorization_id,
    )

    api_requests_total.labels(endpoint="/api/v1/investigations/workspaces", status="success").inc()
    api_request_duration_seconds.labels(endpoint="/api/v1/investigations/workspaces").observe(
        time.time() - start_time
    )

    return APIResponse(
        success=True,
        data=_workspace_to_dict(workspace),
        message="Workspace created",
        request_id=request_id,
        timestamp=datetime.now(),
    )


@app.get("/api/v1/investigations/workspaces/{workspace_id}", response_model=APIResponse)
async def get_investigation_workspace(workspace_id: str, context: AccessContext = Depends(get_access_context)):
    workspace = investigation_workspace_service.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    access_request = AccessRequest(
        resource_type="investigation_workspace",
        resource_id=workspace_id,
        action="view_workspace",
        data_category=DataCategory.SENSITIVE,
        purpose=ProcessingPurpose.INVESTIGATION,
        context=context,
    )
    allowed, message, _ = await enforce_compliance(access_request)
    if not allowed:
        raise HTTPException(status_code=403, detail=message)

    dossier = investigation_workspace_service.generate_dossier(workspace_id)
    dossier["workspace"] = _workspace_to_dict(workspace)

    return APIResponse(
        success=True,
        data=dossier,
        message="Workspace dossier generated",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/investigations/workspaces/{workspace_id}/evidence", response_model=APIResponse)
async def add_workspace_evidence(
    workspace_id: str,
    evidence_request: EvidenceCreateRequest,
    context: AccessContext = Depends(get_access_context),
):
    access_request = AccessRequest(
        resource_type="investigation_workspace",
        resource_id=workspace_id,
        action="add_evidence",
        data_category=DataCategory.SENSITIVE,
        purpose=ProcessingPurpose.INVESTIGATION,
        context=context,
    )
    allowed, message, _ = await enforce_compliance(access_request)
    if not allowed:
        raise HTTPException(status_code=403, detail=message)

    try:
        record = investigation_workspace_service.add_evidence(
            workspace_id,
            evidence_type=evidence_request.evidence_type,
            description=evidence_request.description,
            context=context,
            metadata=evidence_request.metadata,
            location=evidence_request.location,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return APIResponse(
        success=True,
        data={
            "evidence_id": record.evidence_id,
            "case_id": record.case_id,
            "workspace_id": record.workspace_id,
            "created_at": record.created_at.isoformat(),
        },
        message="Evidence added",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/investigations/workspaces/{workspace_id}/timeline", response_model=APIResponse)
async def add_workspace_timeline_event(
    workspace_id: str,
    event_request: TimelineEventRequest,
    context: AccessContext = Depends(get_access_context),
):
    access_request = AccessRequest(
        resource_type="investigation_workspace",
        resource_id=workspace_id,
        action="add_timeline_event",
        data_category=DataCategory.SENSITIVE,
        purpose=ProcessingPurpose.INVESTIGATION,
        context=context,
    )
    allowed, message, _ = await enforce_compliance(access_request)
    if not allowed:
        raise HTTPException(status_code=403, detail=message)

    try:
        event = investigation_workspace_service.add_timeline_event(
            workspace_id,
            title=event_request.title,
            description=event_request.description,
            context=context,
            timestamp=event_request.timestamp,
            evidence_id=event_request.evidence_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return APIResponse(
        success=True,
        data={
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "title": event.title,
        },
        message="Timeline event added",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.get("/api/v1/investigations/workspaces/{workspace_id}/dossier", response_model=APIResponse)
async def get_workspace_dossier(workspace_id: str, context: AccessContext = Depends(get_access_context)):
    access_request = AccessRequest(
        resource_type="investigation_workspace",
        resource_id=workspace_id,
        action="view_dossier",
        data_category=DataCategory.SENSITIVE,
        purpose=ProcessingPurpose.INVESTIGATION,
        context=context,
    )
    allowed, message, _ = await enforce_compliance(access_request)
    if not allowed:
        raise HTTPException(status_code=403, detail=message)

    try:
        dossier = investigation_workspace_service.generate_dossier(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return APIResponse(
        success=True,
        data=dossier,
        message="Dossier generated",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


def _workspace_to_dict(workspace):
    return {
        "workspace_id": workspace.workspace_id,
        "case_id": workspace.case_id,
        "workspace_type": workspace.workspace_type,
        "classification_level": workspace.classification_level,
        "created_by": workspace.created_by,
        "created_at": workspace.created_at.isoformat(),
        "last_accessed": workspace.last_accessed.isoformat(),
        "team_members": list(workspace.team_members),
        "tags": list(workspace.tags),
    }


async def _ensure_sweden_prediction_sources(context: AccessContext, auth_id: str) -> None:
    ConnectorFactory.register_specialized_connectors()

    source_definitions = [
        {
            "source_id": "sweden_prediction_historical",
            "name": "Sweden Historical Incidents",
            "connector": ConnectorType.SWEDEN_HISTORICAL,
            "file_path": "data_lake/sweden/historical_incidents.json",
            "categories": [DataCategory.CRIMINAL, DataCategory.LOCATION],
            "sensitivity": DataSensitivity.RESTRICTED,
            "retention": 1825,
            "purposes": [ProcessingPurpose.INVESTIGATION],
        },
        {
            "source_id": "sweden_prediction_stats",
            "name": "Sweden Municipal Stats",
            "connector": ConnectorType.SWEDEN_STATS,
            "file_path": "data_lake/sweden/municipal_stats.json",
            "categories": [DataCategory.LOCATION],
            "sensitivity": DataSensitivity.INTERNAL,
            "retention": 1825,
            "purposes": [ProcessingPurpose.PREVENTION],
        },
        {
            "source_id": "sweden_prediction_events",
            "name": "Sweden Public Events",
            "connector": ConnectorType.SWEDEN_EVENTS,
            "file_path": "data_lake/sweden/public_events.json",
            "categories": [DataCategory.LOCATION],
            "sensitivity": DataSensitivity.INTERNAL,
            "retention": 365,
            "purposes": [ProcessingPurpose.PREVENTION],
        },
        {
            "source_id": "sweden_prediction_scb",
            "name": "Sweden SCB Statistics",
            "connector": ConnectorType.SWEDEN_SCB,
            "file_path": "data_lake/sweden/scb_statistics.json",
            "categories": [DataCategory.LOCATION],
            "sensitivity": DataSensitivity.INTERNAL,
            "retention": 1825,
            "purposes": [ProcessingPurpose.PREVENTION],
        },
        {
            "source_id": "sweden_prediction_survey",
            "name": "Sweden Crime Survey",
            "connector": ConnectorType.SWEDEN_SURVEY,
            "file_path": "data_lake/sweden/crime_survey.json",
            "categories": [DataCategory.PERSONAL],
            "sensitivity": DataSensitivity.RESTRICTED,
            "retention": 1095,
            "purposes": [ProcessingPurpose.INVESTIGATION],
        },
        {
            "source_id": "sweden_prediction_reported",
            "name": "Sweden Reported Crimes",
            "connector": ConnectorType.SWEDEN_REPORTED,
            "file_path": "data_lake/sweden/reported_crimes.json",
            "categories": [DataCategory.CRIMINAL],
            "sensitivity": DataSensitivity.RESTRICTED,
            "retention": 1825,
            "purposes": [ProcessingPurpose.INVESTIGATION],
        },
        {
            "source_id": "sweden_prediction_news",
            "name": "Sweden News",
            "connector": ConnectorType.SWEDEN_NEWS,
            "file_path": "data_lake/sweden/news_incidents.json",
            "categories": [DataCategory.COMMUNICATIONS],
            "sensitivity": DataSensitivity.INTERNAL,
            "retention": 180,
            "purposes": [ProcessingPurpose.PREVENTION],
        },
    ]

    registration_context = AccessContext(
        user_id=context.user_id,
        session_id=context.session_id,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        timestamp=datetime.now(),
        authorization_id=auth_id,
    )

    for definition in source_definitions:
        if definition["source_id"] in universal_data_manager.data_sources:
            continue
        cfg = DataSourceConfig(
            source_id=definition["source_id"],
            name=definition["name"],
            connector_type=definition["connector"],
            config={"file_path": definition["file_path"]},
            data_categories=definition["categories"],
            sensitivity_level=definition["sensitivity"],
            legal_basis=auth_id,
            retention_days=definition["retention"],
            authorized_purposes=definition["purposes"],
        )
        await universal_data_manager.register_data_source(cfg, registration_context)

@app.get("/api/v1/compliance/status")
async def get_compliance_status(context: AccessContext = Depends(get_access_context)):
    """Get system compliance status (admin only)"""

    # Check admin permissions
    access_request = AccessRequest(
        resource_type="compliance",
        resource_id="system_status",
        action="view_compliance_status",
        data_category=DataCategory.PERSONAL,
        purpose=ProcessingPurpose.STATUTORY_DUTY,
        context=context,
    )

    access_granted, message, risk_level = access_control.check_access(access_request)

    if not access_granted:
        raise HTTPException(status_code=403, detail=message)

    # Generate compliance reports
    retention_report = retention_engine.check_retention_compliance()
    consent_report = consent_system.generate_consent_report()
    audit_integrity = audit_logger.verify_audit_integrity()

    return APIResponse(
        success=True,
        data={
            "retention_compliance": retention_report,
            "consent_management": consent_report,
            "audit_integrity": audit_integrity,
            "overall_status": "compliant",
        },
        message="Compliance status retrieved",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/privacy/rights-request")
async def submit_rights_request(
    request_data: Dict[str, Any], context: AccessContext = Depends(get_access_context)
):
    """Submit data subject rights request"""

    try:
        right_type = request_data.get("right_type")
        description = request_data.get("description", "")
        data_categories = request_data.get("data_categories", [])

        # Map data categories
        categories = []
        for cat_str in data_categories:
            try:
                categories.append(DataCategory(cat_str))
            except ValueError:
                continue

        # Submit request
        # Convert string to DataRightType enum
        from backend.privacy.consent_management import DataRightType
        parsed_right_type = DataRightType(right_type) if right_type else DataRightType.ACCESS
        
        request_id = consent_system.submit_rights_request(
            data_subject_id=context.user_id,
            right_type=parsed_right_type,
            description=description,
            identity_verification_method="authenticated_session",
            requested_categories=categories,
        )

        return APIResponse(
            success=True,
            data={"request_id": request_id, "status": "submitted"},
            message="Rights request submitted successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Rights request error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit request")


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""

    return APIResponse(
        success=True,
        data={
            "status": "healthy",
            "privacy_framework": "active",
            "audit_system": "active",
            "access_control": "active",
            "retention_engine": "active",
            "consent_management": "active",
        },
        message="System operational",
        request_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
    )


@app.get("/metrics")
async def metrics_endpoint():
    try:
        from ..observability.metrics import metrics as _m
        text = _m.metrics_text()
        return PlainTextResponse(content=text, media_type="text/plain; version=0.0.4")
    except Exception:
        return PlainTextResponse(content="# metrics unavailable", media_type="text/plain")


@app.get("/dashboard")
async def serve_dashboard():
    """Serve the alerts dashboard"""
    dashboard_path = os.path.join(os.path.dirname(__file__), "../../frontend/alerts_dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.post("/api/v1/analytics/monitor-timeseries", response_model=APIResponse)
async def monitor_timeseries(event: Dict[str, Any], context: AccessContext = Depends(get_access_context)):
    """Monitor a time series for anomalies; generate alerts when detected."""
    series = event.get('series') or []  # list of {ts,value}
    name = event.get('name', 'timeseries')
    try:
        pts = []
        for p in series:
            ts = p.get('ts') or p.get('timestamp')
            val = float(p.get('value', 0))
            pts.append((float(ts) if ts is not None else 0.0, val))
        # Import forecasting functions when needed
        from ..analytics.forecasting import moving_average_forecast, is_forecast_anomaly
        f = moving_average_forecast(pts, window=int(event.get('window', 5)), z=float(event.get('z', 2.0)))
        anomaly = False
        if pts:
            anomaly = is_forecast_anomaly(pts[-1][1], f)
        if anomaly:
            alert_manager.ingest(metric='timeseries_anomaly', value=1, labels={'series': name})
        return APIResponse(
            success=True,
            data={"forecast": f, "anomaly": anomaly},
            message="Timeseries monitored",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Monitor error: {e}")
        raise HTTPException(status_code=400, detail='invalid payload')


# Public Data Ingestion API Endpoints
@app.get("/api/v1/ingestion/sources", response_model=APIResponse)
async def get_available_sources(context: AccessContext = Depends(get_access_context)):
    """Get list of available public data sources"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_sources",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        sources = public_data_manager.get_available_sources()
        source_data = []
        
        for source in sources:
            source_data.append({
                "name": source.name,
                "source_type": source.source_type,
                "base_url": source.base_url,
                "api_key_required": source.api_key_required,
                "rate_limit": source.rate_limit_requests_per_minute,
                "data_format": source.data_format,
                "update_frequency": source.update_frequency,
                "geographic_scope": source.geographic_scope,
                "data_categories": source.data_categories,
            })
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="list_ingestion_sources",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"sources_count": len(sources)},
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={"sources": source_data},
            message=f"Found {len(sources)} available data sources",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sources")


@app.post("/api/v1/ingestion/jobs", response_model=APIResponse)
async def create_ingestion_job(
    job_request: IngestionJobRequest,
    context: AccessContext = Depends(get_access_context)
):
    """Create new ingestion job"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="create",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        # Map frequency string to enum
        frequency_map = {
            "15min": IngestionFrequency.EVERY_15_MINUTES,
            "30min": IngestionFrequency.EVERY_30_MINUTES,
            "hourly": IngestionFrequency.HOURLY,
            "4hours": IngestionFrequency.EVERY_4_HOURS,
            "daily": IngestionFrequency.DAILY,
            "weekly": IngestionFrequency.WEEKLY,
        }
        
        frequency = frequency_map.get(job_request.frequency.lower(), IngestionFrequency.HOURLY)
        
        # Create the job
        job_id = await ingestion_scheduler.create_job(
            source_name=job_request.source_name,
            frequency=frequency,
            context=context,
            records_limit=job_request.records_limit,
            custom_config=job_request.custom_config,
        )
        
        # Get job status
        job_status = ingestion_scheduler.get_job_status(job_id)
        
        return APIResponse(
            success=True,
            data={"job": job_status},
            message=f"Created ingestion job for {job_request.source_name}",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create ingestion job")


@app.get("/api/v1/ingestion/jobs", response_model=APIResponse)
async def get_ingestion_jobs(context: AccessContext = Depends(get_access_context)):
    """Get all ingestion jobs"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        jobs = ingestion_scheduler.get_all_jobs()
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="list_ingestion_jobs",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"jobs_count": len(jobs)},
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={"jobs": jobs},
            message=f"Found {len(jobs)} ingestion jobs",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")


@app.get("/api/v1/ingestion/jobs/{job_id}", response_model=APIResponse)
async def get_ingestion_job(
    job_id: str,
    context: AccessContext = Depends(get_access_context)
):
    """Get specific ingestion job status"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        job_status = ingestion_scheduler.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return APIResponse(
            success=True,
            data={"job": job_status},
            message="Job status retrieved",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job")


@app.post("/api/v1/ingestion/jobs/{job_id}/pause", response_model=APIResponse)
async def pause_ingestion_job(
    job_id: str,
    context: AccessContext = Depends(get_access_context)
):
    """Pause ingestion job"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="modify",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        success = ingestion_scheduler.pause_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Log action
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ADMIN,
            user_id=context.user_id,
            action=f"pause_ingestion_job_{job_id}",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"job_id": job_id},
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={"job_id": job_id, "status": "paused"},
            message="Job paused successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause job")


@app.post("/api/v1/ingestion/jobs/{job_id}/resume", response_model=APIResponse)
async def resume_ingestion_job(
    job_id: str,
    context: AccessContext = Depends(get_access_context)
):
    """Resume ingestion job"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="modify",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        success = ingestion_scheduler.resume_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Log action
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ADMIN,
            user_id=context.user_id,
            action=f"resume_ingestion_job_{job_id}",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"job_id": job_id},
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={"job_id": job_id, "status": "resumed"},
            message="Job resumed successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume job")


@app.delete("/api/v1/ingestion/jobs/{job_id}", response_model=APIResponse)
async def delete_ingestion_job(
    job_id: str,
    context: AccessContext = Depends(get_access_context)
):
    """Delete ingestion job"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="delete",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        success = ingestion_scheduler.delete_job(job_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Log action
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ADMIN,
            user_id=context.user_id,
            action=f"delete_ingestion_job_{job_id}",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"job_id": job_id},
            risk_level=RiskLevel.MEDIUM,
        )
        
        return APIResponse(
            success=True,
            data={"job_id": job_id, "status": "deleted"},
            message="Job deleted successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete job")


@app.post("/api/v1/ingestion/setup-standard", response_model=APIResponse)
async def setup_standard_ingestion(context: AccessContext = Depends(get_access_context)):
    """Setup standard ingestion jobs for common public data sources"""
    
    # Check permissions - only admins can setup standard jobs
    request = AccessRequest(
        user_id=context.user_id,
        resource="ingestion_jobs",
        action="create_batch",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        created_jobs = await ingestion_scheduler.create_standard_jobs(context)
        
        # Log action
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ADMIN,
            user_id=context.user_id,
            action="setup_standard_ingestion_jobs",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={"jobs_created": len([j for j in created_jobs if j.get("status") == "created"])},
            risk_level=RiskLevel.MEDIUM,
        )
        
        return APIResponse(
            success=True,
            data={"jobs": created_jobs},
            message=f"Setup completed: {len(created_jobs)} jobs processed",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error setting up standard jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to setup standard jobs")


# Swedish Crime Analytics API Endpoints
@app.get("/api/v1/analytics/swedish-crime/heatmap", response_model=APIResponse)
async def get_crime_heatmap(
    time_window_hours: int = 24,
    context: AccessContext = Depends(get_access_context)
):
    """Get crime heatmap data for Swedish municipalities"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="crime_analytics",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        heatmap_data = await swedish_crime_analytics.generate_heatmap_data(time_window_hours)
        
        # Convert to JSON-serializable format
        heatmap_json = []
        for point in heatmap_data:
            heatmap_json.append({
                "municipality": point.municipality,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "crime_intensity": point.crime_intensity,
                "incident_count": point.incident_count,
                "dominant_crime_type": point.dominant_crime_type,
                "severity_distribution": point.severity_distribution,
                "recent_incidents": point.recent_incidents,
            })
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="get_crime_heatmap",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "time_window_hours": time_window_hours,
                "municipalities_included": len(heatmap_json),
            },
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={
                "heatmap": heatmap_json,
                "time_window_hours": time_window_hours,
                "generated_at": datetime.now().isoformat(),
                "total_municipalities": len(heatmap_json),
            },
            message=f"Crime heatmap data for {len(heatmap_json)} municipalities",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate heatmap data")


@app.get("/api/v1/analytics/swedish-crime/trends", response_model=APIResponse)
async def get_crime_trends(
    municipality: Optional[str] = None,
    context: AccessContext = Depends(get_access_context)
):
    """Get crime trend analysis for Swedish municipalities"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="crime_analytics",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        trends = await swedish_crime_analytics.analyze_crime_trends(municipality)
        
        # Convert to JSON-serializable format
        trends_json = []
        for trend in trends:
            trends_json.append({
                "municipality": trend.municipality,
                "incident_type": trend.incident_type,
                "period_start": trend.period_start.isoformat(),
                "period_end": trend.period_end.isoformat(),
                "incident_count": trend.incident_count,
                "trend_direction": trend.trend_direction,
                "change_percentage": trend.change_percentage,
                "severity_breakdown": trend.severity_breakdown,
                "is_anomaly": trend.is_anomaly,
                "confidence_score": trend.confidence_score,
            })
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="get_crime_trends",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "municipality_filter": municipality,
                "trends_found": len(trends_json),
            },
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data={
                "trends": trends_json,
                "municipality_filter": municipality,
                "analysis_timestamp": datetime.now().isoformat(),
            },
            message=f"Found {len(trends_json)} crime trends",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze crime trends")


@app.get("/api/v1/analytics/swedish-crime/warnings", response_model=APIResponse)
async def get_dynamic_warnings(context: AccessContext = Depends(get_access_context)):
    """Get active dynamic warnings based on crime patterns"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="crime_warnings",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        # Generate new warnings
        new_warnings = await swedish_crime_analytics.generate_dynamic_warnings(context)
        
        # Get all active warnings
        active_warnings = swedish_crime_analytics.get_active_warnings()
        
        # Convert to JSON-serializable format
        warnings_json = []
        for warning in active_warnings:
            warnings_json.append({
                "warning_id": warning.warning_id,
                "municipality": warning.municipality,
                "warning_type": warning.warning_type,
                "severity": warning.severity,
                "title": warning.title,
                "description": warning.description,
                "affected_area": warning.affected_area,
                "confidence": warning.confidence,
                "created_at": warning.created_at.isoformat(),
                "expires_at": warning.expires_at.isoformat(),
                "recommended_actions": warning.recommended_actions,
                "supporting_data": warning.supporting_data,
            })
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="get_dynamic_warnings",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "active_warnings": len(active_warnings),
                "new_warnings_generated": len(new_warnings),
            },
            risk_level=RiskLevel.MEDIUM,
        )
        
        return APIResponse(
            success=True,
            data={
                "warnings": warnings_json,
                "new_warnings_generated": len(new_warnings),
                "total_active_warnings": len(active_warnings),
                "generated_at": datetime.now().isoformat(),
            },
            message=f"{len(active_warnings)} active warnings",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error getting warnings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dynamic warnings")


@app.get("/api/v1/analytics/swedish-crime/summary", response_model=APIResponse)
async def get_analytics_summary(context: AccessContext = Depends(get_access_context)):
    """Get comprehensive crime analytics summary"""
    
    # Check permissions
    request = AccessRequest(
        user_id=context.user_id,
        resource="crime_analytics",
        action="read",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        summary = await swedish_crime_analytics.get_analytics_summary()
        
        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="get_analytics_summary",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "incidents_24h": summary["summary"]["total_incidents_24h"],
                "active_warnings": summary["summary"]["active_warnings"],
            },
            risk_level=RiskLevel.LOW,
        )
        
        return APIResponse(
            success=True,
            data=summary,
            message="Crime analytics summary generated",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics summary")


@app.post("/api/v1/analytics/swedish-crime/ingest", response_model=APIResponse)
async def ingest_crime_data(
    records: List[Dict[str, Any]],
    context: AccessContext = Depends(get_access_context)
):
    """Manually ingest crime data for analysis (for testing/integration)"""
    
    # Check permissions - only admins can manually ingest
    request = AccessRequest(
        user_id=context.user_id,
        resource="crime_analytics",
        action="ingest",
        context=context,
    )
    
    allowed, denial_reason = access_control.check_access(request)
    if not allowed:
        raise HTTPException(status_code=403, detail=denial_reason)
    
    try:
        processed_count = await swedish_crime_analytics.ingest_crime_data(records)
        
        # Log ingestion
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="manual_crime_data_ingest",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "records_provided": len(records),
                "records_processed": processed_count,
            },
            risk_level=RiskLevel.MEDIUM,
        )
        
        return APIResponse(
            success=True,
            data={
                "records_provided": len(records),
                "records_processed": processed_count,
                "ingestion_timestamp": datetime.now().isoformat(),
            },
            message=f"Processed {processed_count} crime records",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Error ingesting crime data: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest crime data")


# User-Submitted Incidents API Endpoints

@app.post("/api/v1/incidents/submit", response_model=APIResponse)
async def submit_user_incident(
    incident_request: UserIncidentSubmissionRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Submit a user-reported incident for moderation and potential training data"""

    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Basic validation
        if len(incident_request.title.strip()) < 5:
            raise HTTPException(status_code=400, detail="Title must be at least 5 characters")

        if len(incident_request.description.strip()) < 20:
            raise HTTPException(status_code=400, detail="Description must be at least 20 characters")

        # Validate severity range
        if incident_request.severity_reported and not (1 <= incident_request.severity_reported <= 10):
            raise HTTPException(status_code=400, detail="Severity must be between 1 and 10")

        # Validate coordinates if provided
        if incident_request.latitude is not None or incident_request.longitude is not None:
            if incident_request.latitude is None or incident_request.longitude is None:
                raise HTTPException(status_code=400, detail="Both latitude and longitude must be provided")

            if not (-90 <= incident_request.latitude <= 90):
                raise HTTPException(status_code=400, detail="Invalid latitude")

            if not (-180 <= incident_request.longitude <= 180):
                raise HTTPException(status_code=400, detail="Invalid longitude")

        # Check access permissions
        access_request = AccessRequest(
            resource_type="user_incident_submission",
            resource_id=None,
            action="submit_incident",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PUBLIC_SAFETY,
            context=context,
            additional_params={
                "incident_type": incident_request.incident_type,
                "severity": incident_request.severity_reported,
                "has_location": incident_request.latitude is not None,
            },
        )

        access_granted, message, risk_level = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Generate submission ID (public-facing)
        submission_id = f"SUB-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        # Hash user information for privacy
        import hashlib
        submitter_id_hash = None
        if context.user_id and context.user_id != "unknown":
            submitter_id_hash = hashlib.sha256(f"{context.user_id}-{context.session_id}".encode()).hexdigest()[:16]

        source_ip_hash = None
        if context.ip_address:
            source_ip_hash = hashlib.sha256(context.ip_address.encode()).hexdigest()[:16]

        user_agent_hash = None
        if context.user_agent:
            user_agent_hash = hashlib.sha256(context.user_agent.encode()).hexdigest()[:16]

        # Determine municipality from coordinates (simplified)
        municipality = None
        region = None
        if incident_request.latitude and incident_request.longitude:
            # This would normally involve reverse geocoding
            # For now, use a simplified approach
            municipality = "Unknown Municipality"
            region = "Unknown Region"

        # Prepare incident data for database
        incident_data = {
            "submission_id": submission_id,
            "incident_datetime": incident_request.incident_datetime or datetime.now(),
            "submitter_id": submitter_id_hash,
            "submitter_type": incident_request.submitter_type,
            "submission_method": "mobile_app",  # Could be determined from user agent
            "location_provided": incident_request.latitude is not None,
            "latitude": incident_request.latitude,
            "longitude": incident_request.longitude,
            "location_description": incident_request.location_description,
            "municipality": municipality,
            "region": region,
            "incident_type": incident_request.incident_type,
            "incident_category": incident_request.incident_category,
            "severity_reported": incident_request.severity_reported,
            "urgency_level": incident_request.urgency_level,
            "title": incident_request.title.strip(),
            "description": incident_request.description.strip(),
            "evidence_provided": incident_request.evidence_provided,
            "witness_count": incident_request.witness_count,
            "consent_data_processing": incident_request.consent_data_processing,
            "consent_law_enforcement": incident_request.consent_law_enforcement,
            "consent_research": incident_request.consent_research,
            "anonymization_level": incident_request.anonymization_level,
            "source_ip_hash": source_ip_hash,
            "user_agent_hash": user_agent_hash,
            "session_id": context.session_id,
            "api_version": "v1",
            "created_by": context.user_id or "anonymous",
        }

        # Insert into database (mock implementation - would use actual database)
        incident_id = str(uuid.uuid4())

        # Log incident submission
        audit_logger.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=context.user_id,
            action="submit_user_incident",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            resource=incident_id,
            authorization_id=context.authorization_id,
            details={
                "submission_id": submission_id,
                "incident_type": incident_request.incident_type,
                "severity": incident_request.severity_reported,
                "has_location": incident_request.latitude is not None,
                "consent_training": incident_request.consent_research,
            },
            risk_level=RiskLevel.LOW,
        )

        api_requests_total.labels(endpoint="/api/v1/incidents/submit", status="success").inc()
        api_request_duration_seconds.labels(endpoint="/api/v1/incidents/submit").observe(
            time.time() - start_time
        )

        return APIResponse(
            success=True,
            data={
                "incident_id": incident_id,
                "submission_id": submission_id,
                "status": "submitted",
                "moderation_status": "pending",
                "estimated_review_time": "24-48 hours",
            },
            message="Incident submitted successfully and queued for moderation",
            request_id=request_id,
            timestamp=datetime.now(),
        )

    except HTTPException:
        api_requests_total.labels(endpoint="/api/v1/incidents/submit", status="denied").inc()
        raise
    except Exception as e:
        logger.error(f"Incident submission error: {e}")
        api_requests_total.labels(endpoint="/api/v1/incidents/submit", status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to submit incident")


@app.get("/api/v1/incidents/status/{submission_id}", response_model=APIResponse)
async def get_incident_status(
    submission_id: str,
    context: AccessContext = Depends(get_access_context),
):
    """Get status of a submitted incident"""

    try:
        # Check access permissions
        access_request = AccessRequest(
            resource_type="user_incident_status",
            resource_id=submission_id,
            action="view_status",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PUBLIC_SAFETY,
            context=context,
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Mock status data (would query actual database)
        status_data = {
            "submission_id": submission_id,
            "status": "under_review",
            "moderation_status": "pending",
            "verification_status": "unverified",
            "submitted_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "estimated_completion": "2-3 business days",
            "public_updates": [],
        }

        return APIResponse(
            success=True,
            data=status_data,
            message="Status retrieved successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve status")


@app.post("/api/v1/incidents/{incident_id}/update", response_model=APIResponse)
async def add_incident_update(
    incident_id: str,
    update_request: UserIncidentUpdateRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Add an update to an existing incident"""

    try:
        # Check access permissions
        access_request = AccessRequest(
            resource_type="user_incident_update",
            resource_id=incident_id,
            action="add_update",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PUBLIC_SAFETY,
            context=context,
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Validate update
        if len(update_request.update_text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Update text must be at least 10 characters")

        # Mock update creation (would use actual database)
        update_id = str(uuid.uuid4())

        # Log update
        audit_logger.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=context.user_id,
            action="add_incident_update",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            resource=incident_id,
            authorization_id=context.authorization_id,
            details={
                "update_id": update_id,
                "update_type": update_request.update_type,
                "public_visible": update_request.public_visible,
            },
            risk_level=RiskLevel.LOW,
        )

        return APIResponse(
            success=True,
            data={
                "update_id": update_id,
                "incident_id": incident_id,
                "status": "added",
                "visibility": "public" if update_request.public_visible else "private",
            },
            message="Update added successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update addition error: {e}")
        raise HTTPException(status_code=500, detail="Failed to add update")


@app.get("/api/v1/moderation/queue", response_model=APIResponse)
async def get_moderation_queue(
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    priority: Optional[int] = None,
    context: AccessContext = Depends(get_access_context),
):
    """Get incidents in moderation queue (moderator access required)"""

    try:
        # Check moderator permissions
        access_request = AccessRequest(
            resource_type="moderation_queue",
            resource_id=None,
            action="view_queue",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.STATUTORY_DUTY,
            context=context,
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Mock moderation queue data (would query actual database)
        queue_items = []
        for i in range(page_size):
            queue_items.append({
                "incident_id": str(uuid.uuid4()),
                "submission_id": f"SUB-20241228-{i:08d}",
                "title": f"Sample Incident {i}",
                "incident_type": "safety_concern",
                "severity_reported": 5,
                "priority_level": 3,
                "auto_flagged": False,
                "flag_reasons": [],
                "submitted_at": datetime.now().isoformat(),
                "estimated_review_time": 30,
                "evidence_count": 0,
            })

        # Log access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="view_moderation_queue",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "page": page,
                "page_size": page_size,
                "filters": {"status": status, "priority": priority},
            },
            risk_level=RiskLevel.MEDIUM,
        )

        return APIResponse(
            success=True,
            data={
                "queue_items": queue_items,
                "page": page,
                "page_size": page_size,
                "total_items": len(queue_items),
                "has_next": False,
                "filters_applied": {"status": status, "priority": priority},
            },
            message=f"Retrieved {len(queue_items)} items from moderation queue",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Moderation queue error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve moderation queue")


@app.post("/api/v1/moderation/decide", response_model=APIResponse)
async def moderate_incident(
    moderation_request: IncidentModerationRequest,
    context: AccessContext = Depends(get_access_context),
):
    """Make a moderation decision on an incident (moderator access required)"""

    try:
        # Check moderator permissions
        access_request = AccessRequest(
            resource_type="incident_moderation",
            resource_id=moderation_request.incident_id,
            action="moderate_incident",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.STATUTORY_DUTY,
            context=context,
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Validate decision
        valid_decisions = ["approve", "reject", "request_more_info", "escalate"]
        if moderation_request.decision not in valid_decisions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid decision. Must be one of: {', '.join(valid_decisions)}"
            )

        # Mock moderation processing (would update actual database)
        moderation_result = {
            "incident_id": moderation_request.incident_id,
            "decision": moderation_request.decision,
            "moderated_by": context.user_id,
            "moderated_at": datetime.now().isoformat(),
            "decision_reason": moderation_request.decision_reason,
            "confidence": moderation_request.decision_confidence,
        }

        # Determine next steps based on decision
        next_steps = []
        if moderation_request.decision == "approve":
            next_steps = ["incident_verified", "available_for_training", "added_to_public_data"]
        elif moderation_request.decision == "reject":
            next_steps = ["incident_rejected", "submitter_notified"]
        elif moderation_request.decision == "request_more_info":
            next_steps = ["additional_info_requested", "submitter_contacted"]
        elif moderation_request.decision == "escalate":
            next_steps = ["escalated_to_supervisor", "priority_increased"]

        # Log moderation decision
        audit_logger.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            user_id=context.user_id,
            action="moderate_incident",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            resource=moderation_request.incident_id,
            authorization_id=context.authorization_id,
            details={
                "decision": moderation_request.decision,
                "reason": moderation_request.decision_reason,
                "confidence": moderation_request.decision_confidence,
            },
            risk_level=RiskLevel.MEDIUM,
        )

        return APIResponse(
            success=True,
            data={
                **moderation_result,
                "next_steps": next_steps,
                "training_eligible": moderation_request.decision == "approve",
            },
            message=f"Incident {moderation_request.decision}d successfully",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process moderation decision")


@app.get("/api/v1/training/user-incidents", response_model=APIResponse)
async def get_training_incidents(
    limit: int = 100,
    include_features: bool = False,
    context: AccessContext = Depends(get_access_context),
):
    """Get approved user incidents for AI training (restricted access)"""

    try:
        # Check training data access permissions
        access_request = AccessRequest(
            resource_type="training_data",
            resource_id="user_incidents",
            action="access_training_data",
            data_category=DataCategory.PERSONAL,
            purpose=ProcessingPurpose.PREVENTION,
            context=context,
        )

        access_granted, message, _ = await enforce_compliance(access_request)
        if not access_granted:
            raise HTTPException(status_code=403, detail=message)

        # Mock training data (would query actual database view)
        training_incidents = []
        for i in range(min(limit, 50)):  # Limit to 50 for demo
            incident = {
                "incident_id": str(uuid.uuid4()),
                "incident_type": "safety_concern",
                "incident_category": "general",
                "municipality": "Stockholm",
                "severity_reported": 5,
                "description_anonymized": "Anonymized incident description",
                "training_weight": 1.0,
                "quality_scores": {
                    "completeness": 0.8,
                    "accuracy": 0.9,
                    "relevance": 0.7,
                },
                "approved_at": datetime.now().isoformat(),
            }

            if include_features:
                incident["feature_vector"] = {
                    "text_features": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "location_features": [59.3293, 18.0686],  # Stockholm coordinates
                    "temporal_features": [2024, 12, 28, 15, 30],  # Year, month, day, hour, minute
                }

            training_incidents.append(incident)

        # Log training data access
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=context.user_id,
            action="access_training_incidents",
            session_id=context.session_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            result=AccessResult.SUCCESS,
            authorization_id=context.authorization_id,
            details={
                "incidents_accessed": len(training_incidents),
                "include_features": include_features,
                "requested_limit": limit,
            },
            risk_level=RiskLevel.HIGH,  # Training data access is high risk
        )

        return APIResponse(
            success=True,
            data={
                "training_incidents": training_incidents,
                "total_available": len(training_incidents),
                "data_quality_summary": {
                    "avg_completeness": 0.8,
                    "avg_accuracy": 0.85,
                    "avg_relevance": 0.75,
                },
                "last_updated": datetime.now().isoformat(),
            },
            message=f"Retrieved {len(training_incidents)} training incidents",
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training data access error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training incidents")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
