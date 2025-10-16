"""
Incident Report Clustering API
Automatically combines duplicate reports from multiple anonymous users
Uses spatial-temporal-semantic matching to identify same incidents
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import hashlib
from datetime import datetime
import uuid
from ..database.postgis_database import get_database

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

class SubmitReportRequest(BaseModel):
    incident_type: str
    latitude: float
    longitude: float
    occurred_at: str  # ISO datetime
    device_id: str  # Will be hashed for privacy
    description: Optional[str] = None
    media_ids: Optional[List[str]] = None
    severity: Optional[int] = 3

class ReportResponse(BaseModel):
    success: bool
    incident_id: str
    is_new_incident: bool
    matched_existing: bool
    report_count: int
    unique_reporters: int
    cluster_confidence: float
    is_verified: bool  # True if 2+ unique reporters

class IncidentClusterInfo(BaseModel):
    incident_id: str
    report_count: int
    unique_reporters: int
    cluster_confidence: float
    is_verified: bool
    reports: List[dict]

@router.post("/submit", response_model=ReportResponse)
async def submit_incident_report(
    request: SubmitReportRequest,
    db = Depends(get_database)
):
    """
    Submit an incident report - automatically clusters with existing reports

    Privacy Features:
    - Device ID hashed with SHA-256 before storage
    - No personal information collected
    - One report per device per incident cluster

    Clustering Logic:
    - Finds incidents within 2km and 24 hours with same type
    - Combines reports with similarity score > 0.6
    - Auto-verifies incidents with 2+ unique reporters
    """
    # Hash device ID for privacy
    device_fingerprint = hashlib.sha256(request.device_id.encode()).hexdigest()

    try:
        # Find matching cluster using smart spatial-temporal matching
        result = await db.execute_query(
            """
            SELECT find_matching_cluster_cross_source(%s, %s, %s, %s, %s) as cluster_id
            """,
            request.latitude,
            request.longitude,
            datetime.fromisoformat(request.occurred_at.replace('Z', '+00:00')),
            request.incident_type,
            device_fingerprint
        )

        cluster_id = result[0]['cluster_id'] if result else None
        is_new_incident = (cluster_id is None)
        matched_existing = not is_new_incident

        if is_new_incident:
            # Create new incident
            incident_id = str(uuid.uuid4())
            cluster_id = str(uuid.uuid4())

            await db.execute_non_query(
                """
                INSERT INTO crime_incidents (
                    id, incident_type, severity, description,
                    latitude, longitude, location,
                    occurred_at, reported_at, created_at, updated_at,
                    source, confidence_score, status,
                    cluster_id, is_cluster_primary, cluster_confidence,
                    report_count, unique_reporters, is_verified
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                """,
                incident_id, request.incident_type, request.severity, request.description,
                request.latitude, request.longitude, request.longitude, request.latitude,
                datetime.fromisoformat(request.occurred_at.replace('Z', '+00:00')),
                datetime.now(), datetime.now(), datetime.now(),
                'user_report', 0.5, 'active',
                cluster_id, True, 0.5,
                1, 1, False
            )
        else:
            # Find primary incident in cluster
            result = await db.execute_query(
                """
                SELECT id FROM crime_incidents
                WHERE cluster_id = %s AND is_cluster_primary = true
                LIMIT 1
                """,
                str(cluster_id)
            )

            if not result:
                raise HTTPException(status_code=500, detail="Cluster found but no primary incident")

            incident_id = result[0]['id']

        # Add individual report record
        try:
            await db.execute_non_query(
                """
                INSERT INTO incident_reports (
                    incident_id, device_fingerprint,
                    latitude, longitude, location,
                    report_time, media_ids, description, confidence_score
                )
                VALUES (
                    %s, %s,
                    %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                    %s, %s, %s, %s
                )
                """,
                incident_id, device_fingerprint,
                request.latitude, request.longitude, request.longitude, request.latitude,
                datetime.now(), request.media_ids or [], request.description, 1.0
            )
        except Exception as e:
            # User already reported this incident
            if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
                raise HTTPException(
                    status_code=409,
                    detail="You have already reported this incident"
                )
            raise

        # Get updated cluster info
        cluster_info = await db.execute_query_single(
            """
            SELECT
                report_count,
                unique_reporters,
                cluster_confidence,
                is_verified
            FROM crime_incidents
            WHERE id = %s
            """,
            incident_id
        )

        return ReportResponse(
            success=True,
            incident_id=incident_id,
            is_new_incident=is_new_incident,
            matched_existing=matched_existing,
            report_count=cluster_info['report_count'],
            unique_reporters=cluster_info['unique_reporters'],
            cluster_confidence=cluster_info['cluster_confidence'],
            is_verified=cluster_info['is_verified']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit report: {str(e)}")

@router.get("/cluster/{incident_id}", response_model=IncidentClusterInfo)
async def get_cluster_info(incident_id: str, db = Depends(get_database)):
    """
    Get clustering information for an incident
    Shows how many people reported it (anonymously)
    """
    # Get incident cluster info
    incident = await db.execute_query_single(
        """
        SELECT
            id,
            report_count,
            unique_reporters,
            cluster_confidence,
            is_verified
        FROM crime_incidents
        WHERE id = %s
        """,
        incident_id
    )

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Get individual reports (anonymized)
    reports = await db.execute_query(
        """
        SELECT
            report_time,
            latitude,
            longitude,
            description,
            confidence_score,
            LEFT(device_fingerprint, 8) as reporter_id
        FROM incident_reports
        WHERE incident_id = %s
        ORDER BY report_time DESC
        """,
        incident_id
    )

    return IncidentClusterInfo(
        incident_id=incident['id'],
        report_count=incident['report_count'],
        unique_reporters=incident['unique_reporters'],
        cluster_confidence=incident['cluster_confidence'],
        is_verified=incident['is_verified'],
        reports=[
            {
                'report_time': r['report_time'].isoformat(),
                'location': {'lat': r['latitude'], 'lon': r['longitude']},
                'description': r['description'],
                'confidence': r['confidence_score'],
                'reporter_id': r['reporter_id']
            }
            for r in reports
        ]
    )

@router.get("/stats")
async def get_clustering_stats(db = Depends(get_database)):
    """
    Get overall clustering statistics
    """
    stats = await db.execute_query_single(
        """
        SELECT
            COUNT(*) FILTER (WHERE is_verified = true) as verified_incidents,
            COUNT(*) FILTER (WHERE is_verified = false) as unverified_incidents,
            AVG(unique_reporters) as avg_reporters_per_incident,
            MAX(unique_reporters) as max_reporters_single_incident,
            SUM(report_count) as total_reports
        FROM crime_incidents
        WHERE source = 'user_report'
        """
    )

    return {
        'verified_incidents': stats['verified_incidents'] or 0,
        'unverified_incidents': stats['unverified_incidents'] or 0,
        'avg_reporters_per_incident': float(stats['avg_reporters_per_incident'] or 0),
        'max_reporters_single_incident': stats['max_reporters_single_incident'] or 0,
        'total_reports': stats['total_reports'] or 0
    }

@router.get("/cross-source-stats")
async def get_cross_source_validation_stats(db = Depends(get_database)):
    """
    Get statistics on cross-source validation
    Shows how many official Polisen.se incidents have been validated by user reports
    """
    stats = await db.execute_query_single(
        "SELECT * FROM get_cross_source_stats()"
    )

    if not stats:
        return {
            'total_incidents': 0,
            'official_only': 0,
            'user_only': 0,
            'cross_validated': 0,
            'validation_rate_percent': 0.0,
            'description': "No data available"
        }

    return {
        'total_incidents': stats['total_incidents'],
        'official_only': stats['official_only'],
        'user_only': stats['user_only'],
        'cross_validated': stats['cross_validated'],
        'validation_rate_percent': float(stats['validation_rate']) if stats['validation_rate'] else 0,
        'description': f"{stats['cross_validated']} official incidents validated by citizen reports ({stats['validation_rate']}%)"
    }
