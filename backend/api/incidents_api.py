"""
Comprehensive Incidents API
Complete CRUD operations and advanced querying for incidents
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status, Header
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from ..database.postgis_database import get_database
from ..auth.jwt_authentication import verify_token, get_current_user

logger = logging.getLogger(__name__)
security = HTTPBearer()

router = APIRouter(prefix="/api/v1/incidents", tags=["incidents"])


# Helper function for optional authentication
async def get_current_user_optional(authorization: Optional[str] = Header(None)):
    """Get current user if authenticated, otherwise return anonymous user"""
    if not authorization:
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}
    try:
        token = authorization.replace('Bearer ', '')
        payload = verify_token(token)
        return payload
    except:
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}


# Helper function to parse metadata
def parse_metadata(metadata):
    """Parse metadata field which might be dict or JSON string"""
    if not metadata:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except:
            return {}
    return {}


# Enums
class IncidentType(str, Enum):
    THEFT = "theft"
    ASSAULT = "assault"
    VANDALISM = "vandalism"
    BURGLARY = "burglary"
    VEHICLE_CRIME = "vehicle_crime"
    DRUG_RELATED = "drug_related"
    VIOLENCE = "violence"
    DISTURBANCE = "disturbance"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    OTHER = "other"


class SeverityLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    VERIFIED = "verified"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    FALSE_REPORT = "false_report"


# Request Models
class IncidentCreate(BaseModel):
    """Create a new incident"""
    incident_type: IncidentType
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    description: str = Field(..., min_length=10, max_length=2000)
    severity: int = Field(default=3, ge=1, le=5)
    occurred_at: Optional[datetime] = None
    location_name: Optional[str] = Field(None, max_length=200)
    is_anonymous: bool = False
    media_ids: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('occurred_at', pre=True, always=True)
    def set_occurred_at(cls, v):
        return v or datetime.now()


class IncidentUpdate(BaseModel):
    """Update an existing incident"""
    description: Optional[str] = Field(None, min_length=10, max_length=2000)
    severity: Optional[int] = Field(None, ge=1, le=5)
    incident_type: Optional[IncidentType] = None
    status: Optional[IncidentStatus] = None
    verification_status: Optional[VerificationStatus] = None


class IncidentComment(BaseModel):
    """Add a comment to an incident"""
    comment: str = Field(..., min_length=1, max_length=500)
    is_official: bool = False


# Response Models
class IncidentResponse(BaseModel):
    """Complete incident response"""
    id: str
    incident_type: str
    latitude: float
    longitude: float
    description: str
    severity: int
    status: str
    verification_status: str
    occurred_at: datetime
    created_at: datetime
    updated_at: datetime
    source: str
    source_id: Optional[str]
    location_name: Optional[str]
    confidence_score: float
    user_id: Optional[str]
    is_anonymous: bool
    media_count: int
    comment_count: int
    metadata: Dict[str, Any]
    hours_ago: float  # Computed field for mobile app


class IncidentListResponse(BaseModel):
    """Paginated list of incidents"""
    total: Optional[int] = None  # None for large datasets to avoid slow COUNT
    page: int
    page_size: int
    total_pages: Optional[int] = None  # None when total is unavailable
    incidents: List[IncidentResponse]


class IncidentStats(BaseModel):
    """Statistics for incidents"""
    total_count: int
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    last_24h: int
    last_7d: int
    last_30d: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("", response_model=IncidentListResponse)
async def list_incidents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    incident_type: Optional[IncidentType] = Query(None, description="Filter by incident type"),
    severity_min: Optional[int] = Query(None, ge=1, le=5, description="Minimum severity"),
    severity_max: Optional[int] = Query(None, ge=1, le=5, description="Maximum severity"),
    status: Optional[IncidentStatus] = Query(None, description="Filter by status"),
    from_date: Optional[datetime] = Query(None, description="Start date"),
    to_date: Optional[datetime] = Query(None, description="End date"),
    rolling_days: Optional[int] = Query(1, ge=1, le=365, description="Filter incidents from last N days - defaults to 1 (last 24 hours)"),
    latitude: Optional[float] = Query(None, ge=-90, le=90, description="Center latitude for proximity search"),
    longitude: Optional[float] = Query(None, ge=-180, le=180, description="Center longitude for proximity search"),
    radius_km: Optional[float] = Query(None, ge=0.1, le=100, description="Search radius in km"),
    verified_only: bool = Query(False, description="Only show verified incidents"),
    db = Depends(get_database)
):
    """
    Get a paginated list of incidents with filtering options.

    **Filters:**
    - Type, severity, status
    - Date range (or rolling days)
    - Geographic proximity (lat/lon + radius)
    - Verification status

    **Pagination:**
    - Default: 50 items per page
    - Maximum: 200 items per page

    **Note:** Use `rolling_days=7` for map display (last 7 days only)
    """

    try:
        # Build query
        conditions = []
        params = []
        param_count = 1

        # Basic filters
        if incident_type:
            conditions.append(f"incident_type = ${param_count}")
            params.append(incident_type.value)
            param_count += 1

        if severity_min:
            conditions.append(f"severity >= ${param_count}")
            params.append(severity_min)
            param_count += 1

        if severity_max:
            conditions.append(f"severity <= ${param_count}")
            params.append(severity_max)
            param_count += 1

        if status:
            conditions.append(f"metadata->>'status' = ${param_count}")
            params.append(status.value)
            param_count += 1

        if verified_only:
            conditions.append("metadata->>'verification_status' = 'verified'")

        # Date range - rolling_days takes precedence over from_date/to_date
        if rolling_days:
            cutoff_date = datetime.now() - timedelta(days=rolling_days)
            conditions.append(f"occurred_at >= ${param_count}")
            params.append(cutoff_date)
            param_count += 1
        else:
            if from_date:
                conditions.append(f"occurred_at >= ${param_count}")
                params.append(from_date)
                param_count += 1

            if to_date:
                conditions.append(f"occurred_at <= ${param_count}")
                params.append(to_date)
                param_count += 1

        # Geographic proximity
        if latitude is not None and longitude is not None and radius_km:
            conditions.append(f"""
                ST_DWithin(
                    location::geography,
                    ST_MakePoint(${param_count}, ${param_count + 1})::geography,
                    ${param_count + 2}
                )
            """)
            params.extend([longitude, latitude, radius_km * 1000])  # Convert km to meters
            param_count += 3

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # For performance, skip COUNT for large datasets
        # Use approximate row count from pg_class for rolling_days queries
        total = None
        total_pages = None

        if not rolling_days or rolling_days <= 7:
            # Only do exact count for small time windows
            count_query = f"""
            SELECT COUNT(*) as total
            FROM crime_incidents
            WHERE {where_clause}
            """

            count_result = await db.execute_query_single(count_query, *params)
            total = count_result['total'] if count_result else 0
            total_pages = (total + page_size - 1) // page_size

        # Calculate pagination
        offset = (page - 1) * page_size

        # Get incidents
        query = f"""
        SELECT
            id,
            incident_type,
            latitude,
            longitude,
            description,
            severity,
            occurred_at,
            created_at,
            updated_at,
            source,
            source_id,
            confidence_score,
            metadata
        FROM crime_incidents
        WHERE {where_clause}
        ORDER BY occurred_at DESC
        LIMIT ${param_count}
        OFFSET ${param_count + 1}
        """

        params.extend([page_size, offset])

        results = await db.execute_query(query, *params)

        # Format incidents
        incidents = []
        now = datetime.now()
        for row in results:
            metadata = parse_metadata(row.get('metadata'))
            # Calculate hours_ago
            occurred_at = row['occurred_at']
            if occurred_at.tzinfo is not None:
                # If occurred_at has timezone, make now timezone-aware
                from datetime import timezone
                now_aware = now.replace(tzinfo=timezone.utc)
                hours_ago = (now_aware - occurred_at).total_seconds() / 3600
            else:
                hours_ago = (now - occurred_at).total_seconds() / 3600

            incidents.append(IncidentResponse(
                id=str(row['id']),
                incident_type=row['incident_type'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                description=row['description'],
                severity=row['severity'],
                status=metadata.get('status', 'reported'),
                verification_status=metadata.get('verification_status', 'pending'),
                occurred_at=row['occurred_at'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                source=row['source'],
                source_id=row.get('source_id'),
                location_name=metadata.get('location_name'),
                confidence_score=float(row['confidence_score']) if row['confidence_score'] else 0.0,
                user_id=metadata.get('user_id'),
                is_anonymous=metadata.get('is_anonymous', False),
                media_count=len(metadata.get('media_ids', [])),
                comment_count=0,
                metadata=metadata,
                hours_ago=hours_ago
            ))

        return IncidentListResponse(
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            incidents=incidents
        )

    except Exception as e:
        logger.error(f"Error listing incidents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve incidents: {str(e)}"
        )


@router.get("/{incident_id}", response_model=IncidentResponse)
async def get_incident(
    incident_id: UUID = Path(..., description="Incident UUID"),
    db = Depends(get_database)
):
    """
    Get a single incident by ID.

    Returns complete incident details including:
    - Basic information
    - Location data
    - Status and verification
    - Media attachments
    - Comments
    """

    try:
        query = """
        SELECT
            id,
            incident_type,
            latitude,
            longitude,
            description,
            severity,
            occurred_at,
            created_at,
            updated_at,
            source,
            source_id,
            confidence_score,
            metadata
        FROM crime_incidents
        WHERE id = $1
        """

        result = await db.execute_query_single(query, incident_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found"
            )

        metadata = parse_metadata(result.get('metadata'))

        return IncidentResponse(
            id=str(result['id']),
            incident_type=result['incident_type'],
            latitude=float(result['latitude']),
            longitude=float(result['longitude']),
            description=result['description'],
            severity=result['severity'],
            status=metadata.get('status', 'reported'),
            verification_status=metadata.get('verification_status', 'pending'),
            occurred_at=result['occurred_at'],
            created_at=result['created_at'],
            updated_at=result['updated_at'],
            source=result['source'],
            source_id=result.get('source_id'),
            location_name=metadata.get('location_name'),
            confidence_score=float(result['confidence_score']) if result['confidence_score'] else 0.0,
            user_id=metadata.get('user_id'),
            is_anonymous=metadata.get('is_anonymous', False),
            media_count=len(metadata.get('media_ids', [])),
            comment_count=0,  # TODO: Add when incident_reports table is implemented
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve incident: {str(e)}"
        )


@router.post("", response_model=Dict[str, str], status_code=status.HTTP_201_CREATED)
async def create_incident(
    incident: IncidentCreate,
    current_user: Dict = Depends(get_current_user_optional),
    db = Depends(get_database)
):
    """
    Create a new incident report.

    **Authentication Optional** - Anonymous reports allowed

    Citizens can report incidents with:
    - Location (GPS coordinates)
    - Description and type
    - Severity level
    - Optional media attachments
    - Anonymous reporting option
    """

    try:
        # Prepare incident data
        # Convert timezone-aware datetime to naive (database expects naive UTC)
        occurred_at = incident.occurred_at
        if occurred_at and occurred_at.tzinfo is not None:
            occurred_at = occurred_at.replace(tzinfo=None)

        # Build metadata - merge user-provided metadata with system metadata
        base_metadata = {
            'user_id': str(current_user['id']) if not incident.is_anonymous else None,
            'is_anonymous': incident.is_anonymous,
            'location_name': incident.location_name,
            'status': 'reported',
            'verification_status': 'pending',
            'media_ids': incident.media_ids or [],
            'reported_via': 'mobile_app'
        }

        # Merge with user-provided metadata (sensor bundle, etc.)
        if incident.metadata:
            base_metadata.update(incident.metadata)

        incident_data = {
            'incident_type': incident.incident_type.value,
            'latitude': incident.latitude,
            'longitude': incident.longitude,
            'description': incident.description,
            'severity': incident.severity,
            'occurred_at': occurred_at,
            'source': 'citizen_report',
            'confidence_score': 0.7,  # User reports start at 0.7
            'metadata': base_metadata
        }

        # Store incident
        incident_id = await db.store_incident(incident_data)

        if not incident_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to create incident"
            )

        logger.info(f"✅ Incident created: {incident_id} by user {current_user['id']}")

        return {
            "incident_id": str(incident_id),
            "message": "Incident reported successfully",
            "status": "reported"
        }

    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create incident: {str(e)}"
        )


@router.put("/{incident_id}", response_model=Dict[str, str])
async def update_incident(
    incident_id: UUID,
    update: IncidentUpdate,
    current_user: Dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Update an existing incident.

    **Authentication Required**

    Users can update their own incidents.
    Admins can update any incident.
    """

    try:
        # Check if incident exists
        check_query = "SELECT metadata FROM crime_incidents WHERE id = $1"
        existing = await db.execute_query_single(check_query, incident_id)

        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id} not found"
            )

        metadata = existing.get('metadata', {}) or {}

        # Check permissions (user can only update their own, unless admin)
        if current_user.get('user_type') != 'admin':
            if metadata.get('user_id') != str(current_user['id']):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only update your own incidents"
                )

        # Build update query
        updates = []
        params = []
        param_count = 1

        if update.description:
            updates.append(f"description = ${param_count}")
            params.append(update.description)
            param_count += 1

        if update.severity:
            updates.append(f"severity = ${param_count}")
            params.append(update.severity)
            param_count += 1

        if update.incident_type:
            updates.append(f"incident_type = ${param_count}")
            params.append(update.incident_type.value)
            param_count += 1

        # Update metadata fields
        if update.status:
            metadata['status'] = update.status.value

        if update.verification_status:
            metadata['verification_status'] = update.verification_status.value

        if updates or update.status or update.verification_status:
            updates.append(f"metadata = ${param_count}")
            params.append(metadata)
            param_count += 1

            updates.append("updated_at = NOW()")

            update_query = f"""
            UPDATE crime_incidents
            SET {', '.join(updates)}
            WHERE id = ${param_count}
            """
            params.append(incident_id)

            await db.execute_query(update_query, *params)

            logger.info(f"✅ Incident updated: {incident_id} by user {current_user['id']}")

            return {
                "incident_id": str(incident_id),
                "message": "Incident updated successfully"
            }
        else:
            return {
                "incident_id": str(incident_id),
                "message": "No changes made"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update incident: {str(e)}"
        )


@router.delete("/{incident_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_incident(
    incident_id: UUID,
    current_user: Dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete an incident.

    **Admin Only**

    Soft delete - marks incident as dismissed rather than removing from database.
    """

    # Check admin permission
    if current_user.get('user_type') != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete incidents"
        )

    try:
        # Soft delete - update status to dismissed
        query = """
        UPDATE crime_incidents
        SET metadata = jsonb_set(
            COALESCE(metadata, '{}'::jsonb),
            '{status}',
            '"dismissed"'
        ),
        updated_at = NOW()
        WHERE id = $1
        """

        await db.execute_query(query, incident_id)

        logger.info(f"✅ Incident deleted: {incident_id} by admin {current_user['id']}")

        return None

    except Exception as e:
        logger.error(f"Error deleting incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete incident: {str(e)}"
        )


@router.get("/nearby/search", response_model=List[IncidentResponse])
async def search_nearby_incidents(
    latitude: float = Query(..., ge=-90, le=90, description="Center latitude"),
    longitude: float = Query(..., ge=-180, le=180, description="Center longitude"),
    radius_km: float = Query(1.0, ge=0.1, le=50, description="Search radius in kilometers"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    incident_type: Optional[IncidentType] = Query(None, description="Filter by type"),
    severity_min: int = Query(1, ge=1, le=5, description="Minimum severity"),
    hours_ago: Optional[int] = Query(None, ge=1, le=168, description="Only incidents within last N hours"),
    db = Depends(get_database)
):
    """
    Search for incidents near a location.

    **Geographic Proximity Search**

    Returns incidents within specified radius, ordered by distance.

    - Default radius: 1 km
    - Maximum radius: 50 km
    - Results ordered by distance (nearest first)
    """

    try:
        # Build query with distance calculation
        conditions = [f"severity >= {severity_min}"]
        params = [longitude, latitude, radius_km * 1000, limit]

        if incident_type:
            conditions.append(f"incident_type = '{incident_type.value}'")

        if hours_ago:
            cutoff_time = datetime.now() - timedelta(hours=hours_ago)
            conditions.append(f"occurred_at >= '{cutoff_time.isoformat()}'")

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            id,
            incident_type,
            latitude,
            longitude,
            description,
            severity,
            occurred_at,
            created_at,
            updated_at,
            source,
            source_id,
            confidence_score,
            metadata,
            ST_Distance(
                location::geography,
                ST_MakePoint($1, $2)::geography
            ) as distance_meters
        FROM crime_incidents
        WHERE ST_DWithin(
            location::geography,
            ST_MakePoint($1, $2)::geography,
            $3
        )
        AND {where_clause}
        ORDER BY distance_meters ASC
        LIMIT $4
        """

        results = await db.execute_query(query, *params)

        # Format incidents
        incidents = []
        for row in results:
            metadata = parse_metadata(row.get('metadata'))
            incidents.append(IncidentResponse(
                id=str(row['id']),
                incident_type=row['incident_type'],
                latitude=float(row['latitude']),
                longitude=float(row['longitude']),
                description=row['description'],
                severity=row['severity'],
                status=metadata.get('status', 'reported'),
                verification_status=metadata.get('verification_status', 'pending'),
                occurred_at=row['occurred_at'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                source=row['source'],
                source_id=row.get('source_id'),
                location_name=metadata.get('location_name'),
                confidence_score=float(row['confidence_score']) if row['confidence_score'] else 0.0,
                user_id=metadata.get('user_id'),
                is_anonymous=metadata.get('is_anonymous', False),
                media_count=len(metadata.get('media_ids', [])),
                comment_count=0,
                metadata={**metadata, 'distance_meters': row.get('distance_meters')}
            ))

        return incidents

    except Exception as e:
        logger.error(f"Error searching nearby incidents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search incidents: {str(e)}"
        )


@router.get("/stats/summary", response_model=IncidentStats)
async def get_incident_statistics(
    from_date: Optional[datetime] = Query(None, description="Start date for stats"),
    to_date: Optional[datetime] = Query(None, description="End date for stats"),
    db = Depends(get_database)
):
    """
    Get incident statistics and breakdowns.

    Returns:
    - Total count
    - Breakdown by severity, type, and status
    - Time-based counts (24h, 7d, 30d)
    """

    try:
        # Date filtering
        date_filter = ""
        if from_date and to_date:
            date_filter = f"AND occurred_at BETWEEN '{from_date.isoformat()}' AND '{to_date.isoformat()}'"

        # Total count
        total_query = f"SELECT COUNT(*) as total FROM crime_incidents WHERE 1=1 {date_filter}"
        total_result = await db.execute_query_single(total_query)
        total_count = total_result['total'] if total_result else 0

        # By severity
        severity_query = f"""
        SELECT severity, COUNT(*) as count
        FROM crime_incidents
        WHERE 1=1 {date_filter}
        GROUP BY severity
        ORDER BY severity
        """
        severity_results = await db.execute_query(severity_query)
        by_severity = {str(row['severity']): row['count'] for row in severity_results}

        # By type
        type_query = f"""
        SELECT incident_type, COUNT(*) as count
        FROM crime_incidents
        WHERE 1=1 {date_filter}
        GROUP BY incident_type
        ORDER BY count DESC
        """
        type_results = await db.execute_query(type_query)
        by_type = {row['incident_type']: row['count'] for row in type_results}

        # By status
        status_query = f"""
        SELECT metadata->>'status' as status, COUNT(*) as count
        FROM crime_incidents
        WHERE metadata->>'status' IS NOT NULL {date_filter}
        GROUP BY metadata->>'status'
        """
        status_results = await db.execute_query(status_query)
        by_status = {row['status']: row['count'] for row in status_results}

        # Time-based counts
        now = datetime.now()
        last_24h_query = f"SELECT COUNT(*) as count FROM crime_incidents WHERE occurred_at >= '{(now - timedelta(hours=24)).isoformat()}'"
        last_7d_query = f"SELECT COUNT(*) as count FROM crime_incidents WHERE occurred_at >= '{(now - timedelta(days=7)).isoformat()}'"
        last_30d_query = f"SELECT COUNT(*) as count FROM crime_incidents WHERE occurred_at >= '{(now - timedelta(days=30)).isoformat()}'"

        last_24h_result = await db.execute_query_single(last_24h_query)
        last_7d_result = await db.execute_query_single(last_7d_query)
        last_30d_result = await db.execute_query_single(last_30d_query)

        return IncidentStats(
            total_count=total_count,
            by_severity=by_severity,
            by_type=by_type,
            by_status=by_status,
            last_24h=last_24h_result['count'] if last_24h_result else 0,
            last_7d=last_7d_result['count'] if last_7d_result else 0,
            last_30d=last_30d_result['count'] if last_30d_result else 0
        )

    except Exception as e:
        logger.error(f"Error getting incident statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )