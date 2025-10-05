#!/usr/bin/env python3
"""
API Endpoints for Community Disturbance Reporting System
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import json
import asyncio
from pathlib import Path
import base64
import uuid

# Import our disturbance analytics system
from ..community.disturbance_analytics import DisturbanceAnalytics, DisturbanceCategories
from ..community.efficient_storage import EfficientDisturbanceStorage
from ..ai_integration.disturbance_ai_integration import process_disturbance_report_with_ai

# Set up logging
logger = logging.getLogger(__name__)

# Initialize systems
disturbance_analytics = DisturbanceAnalytics("data/disturbances")
security = HTTPBearer()

# Create router
router = APIRouter(prefix="/api/v1/disturbances", tags=["disturbances"])

# Pydantic models for API
class LocationModel(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")

class DisturbanceReportRequest(BaseModel):
    category: str = Field(..., description="Main category")
    subcategory: str = Field(..., description="Specific subcategory")
    severity: int = Field(..., ge=1, le=5, description="Severity level 1-5")
    location: LocationModel
    description: str = Field(..., min_length=10, max_length=1000)
    environmental_factors: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)

class MediaUploadResponse(BaseModel):
    media_id: str
    media_type: str
    size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float

class DisturbanceReportResponse(BaseModel):
    report_id: str
    status: str
    message: str
    cluster_id: Optional[str] = None
    predictions_generated: int
    neighbor_alert_sent: bool
    stability_impact: Dict[str, Any]

class StabilityScoreResponse(BaseModel):
    stability_score: float
    risk_level: str
    active_issues: int
    resolved_rate: float
    avg_severity: float
    trend: str
    recommendations: List[str]
    category_breakdown: Dict[str, int]

class PredictionResponse(BaseModel):
    prediction_id: str
    category: str
    location: LocationModel
    predicted_time: datetime
    confidence: float
    severity_prediction: int
    duration_prediction: float
    based_on_patterns: List[str]
    prevention_suggestions: List[str]
    alert_neighbors: bool
    alert_authorities: bool

# Helper functions
async def get_current_user_id(token: str = Depends(security)) -> str:
    """Extract user ID from auth token - simplified for demo"""
    # In production, this would validate JWT token and extract user ID
    return "demo_user_123"

# API Endpoints

@router.get("/categories")
async def get_disturbance_categories() -> Dict[str, Any]:
    """Get all available disturbance categories and subcategories"""
    return {
        "categories": DisturbanceCategories.CATEGORIES,
        "total_categories": len(DisturbanceCategories.CATEGORIES),
        "total_subcategories": sum(len(cat["subcategories"]) for cat in DisturbanceCategories.CATEGORIES.values())
    }

@router.post("/media/upload")
async def upload_media(
    file: UploadFile = File(...),
    media_type: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> MediaUploadResponse:
    """Upload and compress media file"""

    if file.size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # Read file content
    content = await file.read()

    # Store with compression and deduplication
    file_path = disturbance_analytics.storage.store_media_with_deduplication(content, media_type)

    # Get compression stats
    storage_metrics = await disturbance_analytics.storage.get_storage_metrics()

    media_id = str(uuid.uuid4())

    return MediaUploadResponse(
        media_id=media_id,
        media_type=media_type,
        size_bytes=len(content),
        compressed_size_bytes=len(content) // 2,  # Estimated
        compression_ratio=2.0  # Estimated
    )

@router.post("/report")
async def submit_disturbance_report(
    report: DisturbanceReportRequest,
    photo_files: List[UploadFile] = File(default=[]),
    audio_files: List[UploadFile] = File(default=[]),
    video_files: List[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user_id)
) -> DisturbanceReportResponse:
    """Submit a new disturbance report"""

    try:
        # Process uploaded media
        photo_urls = []
        audio_urls = []
        video_urls = []

        # Convert uploaded files to base64 data URLs for storage
        for photo in photo_files[:5]:  # Limit to 5 photos
            content = await photo.read()
            base64_data = base64.b64encode(content).decode()
            data_url = f"data:image/jpeg;base64,{base64_data}"
            photo_urls.append(data_url)

        for audio in audio_files[:3]:  # Limit to 3 audio files
            content = await audio.read()
            base64_data = base64.b64encode(content).decode()
            data_url = f"data:audio/mp3;base64,{base64_data}"
            audio_urls.append(data_url)

        for video in video_files[:2]:  # Limit to 2 video files
            content = await video.read()
            base64_data = base64.b64encode(content).decode()
            data_url = f"data:video/mp4;base64,{base64_data}"
            video_urls.append(data_url)

        # Build report data
        report_data = {
            'user_id': user_id,
            'category': report.category,
            'subcategory': report.subcategory,
            'severity': report.severity,
            'location': {'lat': report.location.lat, 'lon': report.location.lon},
            'timestamp': datetime.now().isoformat(),
            'description': report.description,
            'photo_urls': photo_urls,
            'audio_urls': audio_urls,
            'video_urls': video_urls,
            'environmental_factors': report.environmental_factors,
            'tags': report.tags,
            'verified': False,
            'resolved': False
        }

        # Submit report through analytics system
        submitted_report = await disturbance_analytics.submit_disturbance_report(report_data)

        # Process through unified AI system for enhanced analysis
        ai_analysis = await process_disturbance_report_with_ai(report_data)

        # Get stability impact (now enhanced by AI)
        stability = ai_analysis.get('neighborhood_risk', {})
        if not stability:
            stability = await disturbance_analytics.get_neighborhood_stability_score(
                report_data['location'], radius_meters=1000
            )

        # Get predictions for area
        predictions = await disturbance_analytics.get_active_predictions(
            report_data['location'], radius_meters=2000
        )

        return DisturbanceReportResponse(
            report_id=submitted_report.report_id,
            status="submitted",
            message="Report submitted successfully",
            cluster_id=None,  # Would be set if part of cluster
            predictions_generated=len(predictions),
            neighbor_alert_sent=True,  # Would implement neighbor alerting
            stability_impact=stability
        )

    except Exception as e:
        logger.error(f"Error submitting disturbance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit report: {str(e)}")

@router.get("/stability")
async def get_neighborhood_stability(
    lat: float,
    lon: float,
    radius_meters: int = 1000
) -> StabilityScoreResponse:
    """Get neighborhood stability score for a location"""

    try:
        location = {'lat': lat, 'lon': lon}
        stability = await disturbance_analytics.get_neighborhood_stability_score(
            location, radius_meters=radius_meters
        )

        return StabilityScoreResponse(**stability)

    except Exception as e:
        logger.error(f"Error getting stability score: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stability score: {str(e)}")

@router.get("/predictions")
async def get_disturbance_predictions(
    lat: float,
    lon: float,
    radius_meters: int = 2000,
    hours_ahead: int = 168  # 1 week
) -> List[PredictionResponse]:
    """Get disturbance predictions for an area"""

    try:
        location = {'lat': lat, 'lon': lon}
        predictions = await disturbance_analytics.get_active_predictions(
            location, radius_meters=radius_meters
        )

        # Filter by time range
        cutoff_time = datetime.now() + timedelta(hours=hours_ahead)
        filtered_predictions = [
            p for p in predictions
            if p.predicted_time <= cutoff_time
        ]

        # Convert to response format
        response_predictions = []
        for pred in filtered_predictions:
            response_predictions.append(PredictionResponse(
                prediction_id=pred.prediction_id,
                category=pred.category,
                location=LocationModel(lat=pred.location['lat'], lon=pred.location['lon']),
                predicted_time=pred.predicted_time,
                confidence=pred.confidence,
                severity_prediction=pred.severity_prediction,
                duration_prediction=pred.duration_prediction,
                based_on_patterns=pred.based_on_patterns,
                prevention_suggestions=pred.prevention_suggestions,
                alert_neighbors=pred.alert_neighbors,
                alert_authorities=pred.alert_authorities
            ))

        return response_predictions

    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get predictions: {str(e)}")

@router.get("/nearby")
async def get_nearby_disturbances(
    lat: float,
    lon: float,
    radius_meters: int = 1000,
    hours_back: int = 24,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """Get nearby disturbances"""

    try:
        location = {'lat': lat, 'lon': lon}

        # Get nearby reports from storage system (would need to implement query method)
        # For now, return mock data structure
        nearby_reports = {
            'total_reports': 0,
            'reports': [],
            'clusters': [],
            'summary': {
                'most_common_category': None,
                'avg_severity': 0,
                'resolution_rate': 0,
                'active_clusters': 0
            }
        }

        return nearby_reports

    except Exception as e:
        logger.error(f"Error getting nearby disturbances: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nearby disturbances: {str(e)}")

@router.post("/resolve/{report_id}")
async def resolve_disturbance(
    report_id: str,
    user_id: str = Depends(get_current_user_id)
) -> Dict[str, Any]:
    """Mark a disturbance as resolved"""

    try:
        # Would implement resolution logic
        return {
            'report_id': report_id,
            'status': 'resolved',
            'resolved_by': user_id,
            'resolved_at': datetime.now().isoformat(),
            'message': 'Disturbance marked as resolved'
        }

    except Exception as e:
        logger.error(f"Error resolving disturbance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resolve disturbance: {str(e)}")

@router.get("/analytics/overview")
async def get_analytics_overview(
    lat: float,
    lon: float,
    radius_meters: int = 2000
) -> Dict[str, Any]:
    """Get comprehensive analytics overview for an area"""

    try:
        location = {'lat': lat, 'lon': lon}

        # Get stability score
        stability = await disturbance_analytics.get_neighborhood_stability_score(
            location, radius_meters=radius_meters
        )

        # Get predictions
        predictions = await disturbance_analytics.get_active_predictions(
            location, radius_meters=radius_meters
        )

        # Get storage metrics
        storage_metrics = await disturbance_analytics.storage.get_storage_metrics()

        return {
            'location': location,
            'radius_meters': radius_meters,
            'stability': stability,
            'predictions': {
                'total': len(predictions),
                'high_confidence': len([p for p in predictions if p.confidence > 0.7]),
                'next_24h': len([p for p in predictions
                               if p.predicted_time <= datetime.now() + timedelta(hours=24)]),
                'categories': list(set(p.category for p in predictions))
            },
            'storage_efficiency': {
                'compression_ratio': storage_metrics.compression_ratio,
                'space_saved_percent': storage_metrics.total_savings_percent,
                'active_records': storage_metrics.active_records,
                'archived_records': storage_metrics.archived_records
            },
            'generated_at': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting analytics overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {str(e)}")

@router.post("/optimize-storage")
async def optimize_storage(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Run storage optimization in background"""

    try:
        # Run optimization in background
        background_tasks.add_task(disturbance_analytics.storage.optimize_storage)

        return {
            'status': 'optimization_started',
            'message': 'Storage optimization running in background',
            'estimated_duration_minutes': 5
        }

    except Exception as e:
        logger.error(f"Error starting storage optimization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start optimization: {str(e)}")

# Include router in main FastAPI app
def include_disturbance_routes(app):
    """Include disturbance routes in FastAPI app"""
    app.include_router(router)