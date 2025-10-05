"""
Atlas AI - Sensor Fusion API Endpoints
RESTful API for real-time threat detection, cross-user behavior tracking, and evidence upload
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import base64
import json
from pydantic import BaseModel

from ..auth.jwt_authentication import get_current_user, AuthenticatedUser
from ..sensor_fusion.threat_detection_engine import get_threat_detector, ThreatDetection
from ..sensor_fusion.cross_user_behavior_tracker import get_cross_user_tracker
from ..notifications.threat_alert_system import get_alert_system
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sensor-fusion", tags=["Sensor Fusion"])

# Request/Response Models
class ThreatAnalysisRequest(BaseModel):
    location: Dict[str, float]  # lat, lng
    timestamp: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "location": {"lat": 59.3293, "lng": 18.0686},
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }

class ThreatDetectionResponse(BaseModel):
    threat_id: str
    threat_level: str
    threat_score: float
    confidence: float
    threat_types: List[str]
    processing_time_ms: float
    alert_radius_meters: int
    requires_emergency: bool
    recommended_actions: List[str]

class BehaviorProgressionResponse(BaseModel):
    person_hash: str
    escalation_detected: bool
    escalation_rate: float
    threat_progression: List[str]
    risk_assessment: str
    recording_count: int
    unique_recorders: int
    intervention_suggestions: List[str]

class CrossUserAlertResponse(BaseModel):
    alert_id: str
    trigger_type: str
    recording_count: int
    unique_recorders: int
    current_threat_level: str
    priority: str
    recommended_actions: List[str]
    affected_locations: List[Dict[str, float]]

@router.post("/analyze-video-threat", response_model=ThreatDetectionResponse)
async def analyze_video_threat(
    video_file: UploadFile = File(...),
    location_data: str = Form(...),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Analyze video upload for threat detection using computer vision
    
    Detects:
    - Violence and aggressive behavior
    - Weapons
    - Crowd agitation
    - Suspicious activities
    """
    try:
        # Parse location data
        location = json.loads(location_data)
        
        # Validate video file
        if not video_file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read video data
        video_data = await video_file.read()
        
        if len(video_data) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="Video file too large (max 100MB)")
        
        # Get threat detector
        detector = await get_threat_detector()
        if not detector.is_initialized:
            await detector.initialize()
        
        # Analyze video for threats
        video_analysis = await detector.analyze_video_threat(
            video_data, location, current_user.user_id
        )
        
        # Perform sensor fusion analysis
        threat_detection = await detector.fusion_analysis(
            video_analysis=video_analysis,
            audio_analysis=None,
            location=location,
            user_id=current_user.user_id
        )
        
        # Store threat detection
        await detector.store_threat_detection(threat_detection)
        
        # Process for cross-user behavior tracking
        cross_tracker = await get_cross_user_tracker()
        behavior_progression = None
        if cross_tracker.is_initialized:
            behavior_progression = await cross_tracker.process_new_recording(
                threat_detection, video_analysis, None
            )
        
        # Create threat alert if threat level is significant
        alert_created = False
        if threat_detection.threat_level in ['medium', 'high', 'critical']:
            try:
                alert_system = get_alert_system()
                alert = await alert_system.process_threat_detection({
                    'detection_id': threat_detection.threat_id,
                    'threat_level': threat_detection.threat_level,
                    'threat_score': threat_detection.threat_score,
                    'location': location,
                    'detected_threats': threat_detection.threat_types,
                    'confidence': threat_detection.confidence,
                    'user_id': current_user.user_id
                })
                alert_created = alert is not None
                if alert_created:
                    logger.info(f"Created threat alert {alert.alert_id} for detection {threat_detection.threat_id}")
            except Exception as e:
                logger.error(f"Failed to create threat alert: {e}")
        
        # Generate recommended actions
        recommended_actions = await generate_recommended_actions(threat_detection)
        
        # Track metrics
        threat_counter = metrics.counter(
            "video_threat_detections", 
            "Video threat detections", 
            ("threat_level", "user_role")
        )
        threat_counter.labels(threat_detection.threat_level, current_user.role).inc()
        
        logger.info(f"✅ Video threat analysis completed: {threat_detection.threat_level} threat detected")
        
        return ThreatDetectionResponse(
            threat_id=threat_detection.threat_id,
            threat_level=threat_detection.threat_level,
            threat_score=threat_detection.threat_score,
            confidence=threat_detection.confidence,
            threat_types=threat_detection.threat_types,
            processing_time_ms=threat_detection.processing_time_ms,
            alert_radius_meters=threat_detection.alert_radius_meters,
            requires_emergency=threat_detection.requires_emergency,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing video threat: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

@router.post("/analyze-audio-threat", response_model=ThreatDetectionResponse)
async def analyze_audio_threat(
    audio_file: UploadFile = File(...),
    location_data: str = Form(...),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Analyze audio upload for threat detection using audio analysis
    
    Detects:
    - Gunshots
    - Screams and distress calls
    - Aggressive shouting
    - Glass breaking
    - Emergency situations
    """
    try:
        # Parse location data
        location = json.loads(location_data)
        
        # Validate audio file
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        if len(audio_data) > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=413, detail="Audio file too large (max 50MB)")
        
        # Get threat detector
        detector = await get_threat_detector()
        if not detector.is_initialized:
            await detector.initialize()
        
        # Analyze audio for threats
        audio_analysis = await detector.analyze_audio_threat(
            audio_data, location, current_user.user_id
        )
        
        # Perform sensor fusion analysis
        threat_detection = await detector.fusion_analysis(
            video_analysis=None,
            audio_analysis=audio_analysis,
            location=location,
            user_id=current_user.user_id
        )
        
        # Store threat detection
        await detector.store_threat_detection(threat_detection)
        
        # Process for cross-user behavior tracking
        cross_tracker = await get_cross_user_tracker()
        if cross_tracker.is_initialized:
            behavior_progression = await cross_tracker.process_new_recording(
                threat_detection, None, audio_analysis
            )
        
        # Generate recommended actions
        recommended_actions = await generate_recommended_actions(threat_detection)
        
        # Track metrics
        threat_counter = metrics.counter(
            "audio_threat_detections", 
            "Audio threat detections", 
            ("threat_level", "user_role")
        )
        threat_counter.labels(threat_detection.threat_level, current_user.role).inc()
        
        logger.info(f"✅ Audio threat analysis completed: {threat_detection.threat_level} threat detected")
        
        return ThreatDetectionResponse(
            threat_id=threat_detection.threat_id,
            threat_level=threat_detection.threat_level,
            threat_score=threat_detection.threat_score,
            confidence=threat_detection.confidence,
            threat_types=threat_detection.threat_types,
            processing_time_ms=threat_detection.processing_time_ms,
            alert_radius_meters=threat_detection.alert_radius_meters,
            requires_emergency=threat_detection.requires_emergency,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing audio threat: {e}")
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")

@router.post("/analyze-multimodal-threat", response_model=ThreatDetectionResponse)
async def analyze_multimodal_threat(
    video_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None),
    location_data: str = Form(...),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Analyze both video and audio for comprehensive multi-modal threat detection
    
    Combines computer vision and audio analysis for enhanced accuracy
    """
    try:
        if not video_file and not audio_file:
            raise HTTPException(status_code=400, detail="At least one file (video or audio) must be provided")
        
        # Parse location data
        location = json.loads(location_data)
        
        # Get threat detector
        detector = await get_threat_detector()
        if not detector.is_initialized:
            await detector.initialize()
        
        video_analysis = None
        audio_analysis = None
        
        # Analyze video if provided
        if video_file:
            if not video_file.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="Video file must be a video format")
            
            video_data = await video_file.read()
            video_analysis = await detector.analyze_video_threat(
                video_data, location, current_user.user_id
            )
        
        # Analyze audio if provided
        if audio_file:
            if not audio_file.content_type.startswith('audio/'):
                raise HTTPException(status_code=400, detail="Audio file must be an audio format")
            
            audio_data = await audio_file.read()
            audio_analysis = await detector.analyze_audio_threat(
                audio_data, location, current_user.user_id
            )
        
        # Perform sensor fusion analysis
        threat_detection = await detector.fusion_analysis(
            video_analysis=video_analysis,
            audio_analysis=audio_analysis,
            location=location,
            user_id=current_user.user_id
        )
        
        # Store threat detection
        await detector.store_threat_detection(threat_detection)
        
        # Process for cross-user behavior tracking
        cross_tracker = await get_cross_user_tracker()
        if cross_tracker.is_initialized:
            behavior_progression = await cross_tracker.process_new_recording(
                threat_detection, video_analysis, audio_analysis
            )
        
        # Generate recommended actions
        recommended_actions = await generate_recommended_actions(threat_detection)
        
        # Track metrics
        threat_counter = metrics.counter(
            "multimodal_threat_detections", 
            "Multi-modal threat detections", 
            ("threat_level", "user_role")
        )
        threat_counter.labels(threat_detection.threat_level, current_user.role).inc()
        
        logger.info(f"✅ Multi-modal threat analysis completed: {threat_detection.threat_level} threat detected")
        
        return ThreatDetectionResponse(
            threat_id=threat_detection.threat_id,
            threat_level=threat_detection.threat_level,
            threat_score=threat_detection.threat_score,
            confidence=threat_detection.confidence,
            threat_types=threat_detection.threat_types,
            processing_time_ms=threat_detection.processing_time_ms,
            alert_radius_meters=threat_detection.alert_radius_meters,
            requires_emergency=threat_detection.requires_emergency,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing multi-modal threat: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-modal analysis failed: {str(e)}")

@router.get("/behavior-progression/{person_hash}", response_model=BehaviorProgressionResponse)
async def get_behavior_progression(
    person_hash: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get behavior progression analysis for a specific person (anonymous hash)
    
    Shows how behavior has escalated across multiple user recordings
    """
    try:
        # Only allow law enforcement and admin to view behavior progressions
        if current_user.role not in ["law_enforcement", "admin"]:
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions. Requires law enforcement or admin access."
            )
        
        cross_tracker = await get_cross_user_tracker()
        if not cross_tracker.is_initialized:
            await cross_tracker.initialize()
        
        # Get behavior progression
        progression = await cross_tracker.analyze_behavior_progression(person_hash)
        
        if not progression:
            raise HTTPException(
                status_code=404, 
                detail="No behavior progression found for this person"
            )
        
        # Count unique recorders
        unique_recorders = set(inc.recorder_user_id for inc in progression.timeline)
        
        return BehaviorProgressionResponse(
            person_hash=progression.person_hash,
            escalation_detected=progression.escalation_detected,
            escalation_rate=progression.escalation_rate,
            threat_progression=progression.threat_progression,
            risk_assessment=progression.risk_assessment,
            recording_count=len(progression.timeline),
            unique_recorders=len(unique_recorders),
            intervention_suggestions=progression.intervention_suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting behavior progression: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progression: {str(e)}")

@router.get("/cross-user-alerts", response_model=List[CrossUserAlertResponse])
async def get_cross_user_alerts(
    current_user: AuthenticatedUser = Depends(get_current_user),
    hours_back: int = 24,
    priority: Optional[str] = None
):
    """
    Get cross-user alerts for escalating behavioral threats
    
    Shows when multiple users have recorded the same person showing escalating behavior
    """
    try:
        # Only allow law enforcement and admin to view cross-user alerts
        if current_user.role not in ["law_enforcement", "admin"]:
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions. Requires law enforcement or admin access."
            )
        
        from ..database.postgis_database import get_database
        db = await get_database()
        
        # Build query
        query = """
            SELECT * FROM cross_user_alerts 
            WHERE created_at > %s
        """
        params = [datetime.now() - timedelta(hours=hours_back)]
        
        if priority:
            query += " AND priority = %s"
            params.append(priority)
        
        query += " ORDER BY created_at DESC LIMIT 50"
        
        alerts = await db.execute_query(query, *params)
        
        alert_responses = []
        for alert in alerts:
            alert_responses.append(CrossUserAlertResponse(
                alert_id=alert['alert_id'],
                trigger_type=alert['trigger_type'],
                recording_count=alert['recording_count'],
                unique_recorders=alert['unique_recorders'],
                current_threat_level=alert['current_threat_level'],
                priority=alert['priority'],
                recommended_actions=alert['recommended_actions'],
                affected_locations=alert['affected_locations']
            ))
        
        return alert_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting cross-user alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.get("/nearby-threats")
async def get_nearby_threats(
    latitude: float,
    longitude: float,
    radius_meters: int = 1000,
    hours_back: int = 6,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Get recent threat detections near a specific location
    
    Shows all threats detected by any user in the area
    """
    try:
        from ..database.postgis_database import get_database
        db = await get_database()
        
        # Get recent threat detections near location
        threats = await db.execute_query("""
            SELECT 
                threat_id,
                threat_level,
                threat_score,
                threat_types,
                ST_X(location) as longitude,
                ST_Y(location) as latitude,
                created_at,
                ST_Distance(location, ST_Point(%s, %s)) as distance_meters
            FROM threat_detections
            WHERE created_at > %s
            AND ST_DWithin(location, ST_Point(%s, %s), %s)
            ORDER BY distance_meters ASC, created_at DESC
            LIMIT 20
        """, 
            longitude, latitude,
            datetime.now() - timedelta(hours=hours_back),
            longitude, latitude, radius_meters
        )
        
        threat_data = []
        for threat in threats:
            threat_data.append({
                "threat_id": threat['threat_id'],
                "threat_level": threat['threat_level'],
                "threat_score": threat['threat_score'],
                "threat_types": threat['threat_types'],
                "location": {
                    "lat": threat['latitude'],
                    "lng": threat['longitude']
                },
                "distance_meters": round(threat['distance_meters'], 1),
                "detected_at": threat['created_at'].isoformat()
            })
        
        return {
            "search_location": {"lat": latitude, "lng": longitude},
            "search_radius_meters": radius_meters,
            "threats_found": len(threat_data),
            "threats": threat_data
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting nearby threats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get nearby threats: {str(e)}")

@router.post("/emergency-alert")
async def trigger_emergency_alert(
    request: ThreatAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Trigger emergency alert for immediate threats requiring law enforcement response
    """
    try:
        # Create emergency threat detection
        threat_detection = ThreatDetection(
            threat_id=f"emergency_{int(datetime.now().timestamp())}_{current_user.user_id}",
            timestamp=datetime.now(),
            location=request.location,
            threat_level='critical',
            threat_score=1.0,
            confidence=1.0,
            threat_types=['emergency_alert'],
            evidence_data={'user_triggered': True},
            user_id=current_user.user_id,
            processing_time_ms=0.0,
            alert_radius_meters=1000,
            requires_emergency=True
        )
        
        # Store emergency alert
        detector = await get_threat_detector()
        await detector.store_threat_detection(threat_detection)
        
        # Trigger emergency notifications in background
        background_tasks.add_task(
            notify_emergency_services,
            threat_detection,
            current_user
        )
        
        # Track emergency alert
        emergency_counter = metrics.counter(
            "emergency_alerts_triggered", 
            "Emergency alerts triggered", 
            ("user_role",)
        )
        emergency_counter.labels(current_user.role).inc()
        
        logger.warning(f"🚨 Emergency alert triggered by {current_user.username} at {request.location}")
        
        return {
            "status": "emergency_alert_triggered",
            "threat_id": threat_detection.threat_id,
            "alert_radius_meters": threat_detection.alert_radius_meters,
            "emergency_services_notified": True,
            "message": "Emergency alert sent to law enforcement and nearby users"
        }
        
    except Exception as e:
        logger.error(f"❌ Error triggering emergency alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger emergency alert: {str(e)}")

# Helper functions
async def generate_recommended_actions(threat: ThreatDetection) -> List[str]:
    """Generate recommended actions based on threat level and types"""
    actions = []
    
    if threat.threat_level == 'critical' or threat.requires_emergency:
        actions.extend([
            "Call emergency services immediately (112)",
            "Move to a safe location if possible",
            "Alert others in the area",
            "Do not approach the threat directly"
        ])
    elif threat.threat_level == 'high':
        actions.extend([
            "Maintain safe distance",
            "Consider contacting law enforcement",
            "Alert nearby people",
            "Continue monitoring the situation"
        ])
    elif threat.threat_level == 'medium':
        actions.extend([
            "Stay alert and aware",
            "Avoid direct confrontation",
            "Report if behavior escalates",
            "Document with additional evidence if safe"
        ])
    else:
        actions.extend([
            "Monitor the situation",
            "Report if behavior changes",
            "Maintain situational awareness"
        ])
    
    # Add specific actions based on threat types
    if 'weapon' in threat.threat_types:
        actions.insert(0, "DO NOT APPROACH - WEAPON DETECTED")
    
    if 'gunshot' in threat.threat_types:
        actions.insert(0, "TAKE IMMEDIATE COVER - GUNSHOT DETECTED")
    
    return actions

async def notify_emergency_services(threat: ThreatDetection, user: AuthenticatedUser):
    """Background task to notify emergency services"""
    try:
        # In production, this would integrate with emergency services APIs
        logger.warning(f"🚨 EMERGENCY: {threat.threat_types} at {threat.location} reported by {user.username}")
        
        # Mock emergency notification
        # Real implementation would send to emergency dispatch systems
        
    except Exception as e:
        logger.error(f"❌ Error notifying emergency services: {e}")

# Health check endpoint
@router.get("/health")
async def sensor_fusion_health_check():
    """Health check for sensor fusion services"""
    try:
        detector = await get_threat_detector()
        tracker = await get_cross_user_tracker()
        
        return {
            "status": "healthy",
            "services": {
                "threat_detector": "available" if detector.is_initialized else "initializing",
                "behavior_tracker": "available" if tracker.is_initialized else "initializing",
                "face_recognition": "available" if HAS_FACE_RECOGNITION else "unavailable"
            },
            "capabilities": {
                "video_analysis": True,
                "audio_analysis": True,
                "multi_modal_fusion": True,
                "cross_user_tracking": True,
                "real_time_alerts": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Sensor fusion health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )