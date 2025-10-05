"""
Media Capture API for Atlas AI
Handles incident reporting with photo/video/audio capture
Integrates with sensor fusion engine for multimodal analysis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid
import os
from pydantic import BaseModel

# Import Celery tasks for async AI processing
from backend.tasks.ai_tasks import analyze_photo_task, analyze_video_task, analyze_audio_task

# Import S3 storage (falls back to local if not enabled)
from backend.media_processing.s3_storage import S3MediaStorage

# Import database for incident queries
from backend.database.postgis_database import PostGISDatabase

logger = logging.getLogger(__name__)

# Initialize S3 storage (auto-detects S3_ENABLED from env)
s3_storage = S3MediaStorage()

# Initialize database connection
db = PostGISDatabase()

media_capture_router = APIRouter(prefix="/api/v1/media", tags=["media_capture"])


class IncidentReport(BaseModel):
    """Schema for incident report submission"""
    incident_type: str
    description: str
    location: Dict[str, float]  # {latitude, longitude}
    severity: Optional[int] = 3
    tags: Optional[List[str]] = []
    user_id: Optional[str] = None
    authorization_id: Optional[str] = "PUBLIC_SAFETY_GENERAL"


class MediaUploadResponse(BaseModel):
    """Response for media upload"""
    media_id: str
    media_type: str
    file_path: str
    upload_timestamp: str
    processing_status: str


@media_capture_router.post("/upload/photo", response_model=MediaUploadResponse)
async def upload_photo(
    file: UploadFile = File(...),
    incident_id: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
):
    """
    Upload a photo for incident reporting
    Supports multipart/form-data for mobile app integration
    """
    try:
        # Generate unique media ID
        media_id = f"photo_{uuid.uuid4().hex[:12]}"

        # Create upload directory if it doesn't exist
        upload_dir = "uploads/images"
        os.makedirs(upload_dir, exist_ok=True)

        # Save file
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        file_path = os.path.join(upload_dir, f"{media_id}.{file_ext}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Photo uploaded: {media_id}, size: {len(content)} bytes")

        # Queue for async AI analysis (threat detection, scene understanding)
        try:
            task = analyze_photo_task.delay(media_id, file_path)
            logger.info(f"Queued photo analysis task: {task.id}")
            processing_status = "queued_for_analysis"
        except Exception as e:
            logger.error(f"Failed to queue photo analysis: {e}")
            processing_status = "uploaded_analysis_failed"

        return MediaUploadResponse(
            media_id=media_id,
            media_type="photo",
            file_path=file_path,
            upload_timestamp=datetime.utcnow().isoformat(),
            processing_status=processing_status
        )

    except Exception as e:
        logger.error(f"Error uploading photo: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@media_capture_router.post("/upload/audio", response_model=MediaUploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    incident_id: Optional[str] = Form(None),
    duration_seconds: Optional[float] = Form(None)
):
    """
    Upload audio for incident reporting
    Can be used for threat sound detection (gunshots, alarms, etc.)
    """
    try:
        media_id = f"audio_{uuid.uuid4().hex[:12]}"

        upload_dir = "uploads/audio"
        os.makedirs(upload_dir, exist_ok=True)

        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'm4a'
        file_path = os.path.join(upload_dir, f"{media_id}.{file_ext}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Audio uploaded: {media_id}, size: {len(content)} bytes, duration: {duration_seconds}s")

        # Queue for async audio analysis (threat sound detection, speech-to-text)
        try:
            task = analyze_audio_task.delay(media_id, file_path)
            logger.info(f"Queued audio analysis task: {task.id} (will be sent to sensor fusion)")
            processing_status = "queued_for_analysis"
        except Exception as e:
            logger.error(f"Failed to queue audio analysis: {e}")
            processing_status = "uploaded_analysis_failed"

        return MediaUploadResponse(
            media_id=media_id,
            media_type="audio",
            file_path=file_path,
            upload_timestamp=datetime.utcnow().isoformat(),
            processing_status=processing_status
        )

    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@media_capture_router.post("/upload/video", response_model=MediaUploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    incident_id: Optional[str] = Form(None),
    duration_seconds: Optional[float] = Form(None)
):
    """
    Upload video for incident reporting
    Supports threat detection and activity recognition
    """
    try:
        media_id = f"video_{uuid.uuid4().hex[:12]}"

        upload_dir = "uploads/video"
        os.makedirs(upload_dir, exist_ok=True)

        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'mp4'
        file_path = os.path.join(upload_dir, f"{media_id}.{file_ext}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Video uploaded: {media_id}, size: {len(content)} bytes, duration: {duration_seconds}s")

        # Queue for async video analysis (keyframe extraction, threat detection, object recognition)
        try:
            task = analyze_video_task.delay(media_id, file_path)
            logger.info(f"Queued video analysis task: {task.id} (frames will be sent to sensor fusion)")
            processing_status = "queued_for_analysis"
        except Exception as e:
            logger.error(f"Failed to queue video analysis: {e}")
            processing_status = "uploaded_analysis_failed"

        return MediaUploadResponse(
            media_id=media_id,
            media_type="video",
            file_path=file_path,
            upload_timestamp=datetime.utcnow().isoformat(),
            processing_status=processing_status
        )

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@media_capture_router.post("/incidents/submit")
async def submit_incident(report: IncidentReport):
    """
    Submit an incident report (with or without media)
    This is the main endpoint for citizen reporting
    """
    try:
        incident_id = f"incident_{uuid.uuid4().hex[:12]}"

        # Create incident record
        incident_data = {
            "incident_id": incident_id,
            "incident_type": report.incident_type,
            "description": report.description,
            "location": report.location,
            "severity": report.severity,
            "tags": report.tags,
            "user_id": report.user_id,
            "authorization_id": report.authorization_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "submitted",
            "verification_status": "pending"
        }

        # TODO: Store in database
        # TODO: Trigger safety analysis
        # TODO: Check if location is in high-risk area
        # TODO: Send notifications to nearby users if severity is high

        logger.info(f"Incident submitted: {incident_id}, type: {report.incident_type}, severity: {report.severity}")

        return {
            "incident_id": incident_id,
            "status": "submitted",
            "message": "Thank you for your report. We're processing your submission.",
            "estimated_verification_time_minutes": 15
        }

    except Exception as e:
        logger.error(f"Error submitting incident: {e}")
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@media_capture_router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Get incident details by ID"""
    try:
        async with db.pool.acquire() as conn:
            incident = await conn.fetchrow('''
                SELECT
                    id,
                    incident_type,
                    description,
                    latitude,
                    longitude,
                    severity,
                    status,
                    is_verified,
                    created_at,
                    occurred_at
                FROM crime_incidents
                WHERE id = $1
            ''', incident_id)

            if not incident:
                raise HTTPException(status_code=404, detail="Incident not found")

            # Get media count
            media_count = await conn.fetchval('''
                SELECT COUNT(*) FROM media_files WHERE incident_id = $1
            ''', incident_id)

            return {
                "incident_id": incident['id'],
                "incident_type": incident['incident_type'],
                "description": incident['description'],
                "location": {
                    "latitude": incident['latitude'],
                    "longitude": incident['longitude']
                },
                "severity": incident['severity'],
                "status": incident['status'],
                "verification_status": "confirmed" if incident['is_verified'] else "pending",
                "media_count": media_count or 0,
                "created_at": incident['created_at'].isoformat() if incident['created_at'] else None,
                "occurred_at": incident['occurred_at'].isoformat() if incident['occurred_at'] else None
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@media_capture_router.get("/incidents/nearby")
async def get_nearby_incidents(
    latitude: float,
    longitude: float,
    radius_km: float = 5.0,
    limit: int = 50
):
    """
    Get incidents near a location
    Used by mobile app to show nearby safety issues
    """
    try:
        radius_meters = radius_km * 1000

        async with db.pool.acquire() as conn:
            incidents = await conn.fetch('''
                SELECT
                    id,
                    incident_type,
                    description,
                    latitude,
                    longitude,
                    severity,
                    status,
                    is_verified,
                    occurred_at,
                    created_at,
                    ST_Distance(
                        location,
                        ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography
                    ) as distance_meters
                FROM crime_incidents
                WHERE ST_DWithin(
                    location,
                    ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
                    $3
                )
                ORDER BY distance_meters ASC
                LIMIT $4
            ''', latitude, longitude, radius_meters, limit)

            incident_list = []
            for incident in incidents:
                incident_list.append({
                    "incident_id": incident['id'],
                    "incident_type": incident['incident_type'],
                    "description": incident['description'],
                    "location": {
                        "latitude": incident['latitude'],
                        "longitude": incident['longitude']
                    },
                    "severity": incident['severity'],
                    "status": incident['status'],
                    "is_verified": incident['is_verified'],
                    "occurred_at": incident['occurred_at'].isoformat() if incident['occurred_at'] else None,
                    "distance_km": round(incident['distance_meters'] / 1000, 2) if incident['distance_meters'] else None
                })

            return {
                "incidents": incident_list,
                "count": len(incident_list),
                "search_params": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius_km": radius_km
                }
            }

    except Exception as e:
        logger.error(f"Error fetching nearby incidents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@media_capture_router.get("/stats")
async def get_capture_stats():
    """Get media capture statistics"""
    try:
        # Count uploaded files
        upload_dir = "uploads"

        stats = {
            "photos": len(os.listdir(os.path.join(upload_dir, "images"))) if os.path.exists(os.path.join(upload_dir, "images")) else 0,
            "audio": len(os.listdir(os.path.join(upload_dir, "audio"))) if os.path.exists(os.path.join(upload_dir, "audio")) else 0,
            "video": len(os.listdir(os.path.join(upload_dir, "video"))) if os.path.exists(os.path.join(upload_dir, "video")) else 0
        }

        return stats

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))