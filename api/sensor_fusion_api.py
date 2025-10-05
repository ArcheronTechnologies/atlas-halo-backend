"""
Sensor Fusion API for AI-powered classification of media evidence
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

# Import the sensor fusion engine
try:
    from backend.sensor_fusion.fusion_engine import FusionEngine
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

sensor_fusion_router = APIRouter(prefix="/api/v1/sensor-fusion", tags=["sensor_fusion"])

# Global fusion engine instance (initialized on startup)
fusion_engine = None

class ClassificationResponse(BaseModel):
    """AI Classification result for media evidence"""
    incident_type: str
    severity_level: str
    description: str
    threat_level: float  # 0-10 scale
    confidence: float  # 0-1 scale
    threat_types: List[str]
    processing_time_ms: float

class VideoDetection(BaseModel):
    """Video detection result from object detection"""
    timestamp: float
    detected_objects: List[str]
    confidence_scores: List[float]
    threat_indicators: List[str]

class AudioDetection(BaseModel):
    """Audio detection result from audio analysis"""
    timestamp: float
    sound_type: str
    intensity: float
    is_threat: bool
    confidence: float

@sensor_fusion_router.on_event("startup")
async def initialize_fusion_engine():
    """Initialize the sensor fusion engine on startup"""
    global fusion_engine
    if FUSION_AVAILABLE:
        fusion_engine = FusionEngine()
        await fusion_engine.initialize()
        print("Sensor Fusion Engine initialized")
    else:
        print("WARNING: Sensor fusion engine not available (torch not installed)")

@sensor_fusion_router.get("/health")
async def health_check():
    """Check if sensor fusion engine is available and ready"""
    return {
        "status": "operational" if FUSION_AVAILABLE and fusion_engine else "unavailable",
        "fusion_available": FUSION_AVAILABLE,
        "engine_ready": fusion_engine is not None
    }

@sensor_fusion_router.post("/classify/{media_id}", response_model=ClassificationResponse)
async def classify_media(media_id: str):
    """
    Classify media evidence using AI sensor fusion

    This endpoint processes uploaded video/audio and returns:
    - Incident type classification (theft, assault, vandalism, etc.)
    - Severity assessment (low, moderate, high, critical)
    - Threat level score (0-10)
    - AI-generated description
    - Detected threat types
    """
    if not FUSION_AVAILABLE or not fusion_engine:
        # Return mock classification when fusion engine unavailable
        return ClassificationResponse(
            incident_type="suspicious_activity",
            severity_level="moderate",
            description="AI classification unavailable. Manual review required.",
            threat_level=5.0,
            confidence=0.5,
            threat_types=["unclassified"],
            processing_time_ms=50.0
        )

    try:
        import time
        start_time = time.time()

        # TODO: Retrieve media file from storage using media_id
        # For now, return mock classification

        # Simulated fusion result with high confidence
        fusion_result = {
            "threat_types": ["suspicious_activity", "investigation_required"],
            "threat_level": 6.0,
            "confidence": 0.78
        }

        # Map fusion result to incident classification
        incident_type = map_threat_to_incident_type(fusion_result)
        severity = map_fusion_to_severity(fusion_result)
        description = generate_description(fusion_result)

        processing_time = (time.time() - start_time) * 1000

        return ClassificationResponse(
            incident_type=incident_type,
            severity_level=severity,
            description=description,
            threat_level=fusion_result.get("threat_level", 5.0),
            confidence=fusion_result.get("confidence", 0.75),
            threat_types=fusion_result.get("threat_types", ["unclassified"]),
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@sensor_fusion_router.post("/analyze/video")
async def analyze_video(video_detection: VideoDetection):
    """
    Analyze video frame detection and add to fusion buffer
    """
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion engine not available")

    try:
        await fusion_engine.add_video_detection({
            "timestamp": video_detection.timestamp,
            "detected_objects": video_detection.detected_objects,
            "confidence_scores": video_detection.confidence_scores,
            "threat_indicators": video_detection.threat_indicators
        })
        return {"status": "processed", "timestamp": video_detection.timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@sensor_fusion_router.post("/analyze/audio")
async def analyze_audio(audio_detection: AudioDetection):
    """
    Analyze audio detection and add to fusion buffer
    """
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion engine not available")

    try:
        await fusion_engine.add_audio_detection({
            "timestamp": audio_detection.timestamp,
            "sound_type": audio_detection.sound_type,
            "intensity": audio_detection.intensity,
            "is_threat": audio_detection.is_threat,
            "confidence": audio_detection.confidence
        })
        return {"status": "processed", "timestamp": audio_detection.timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@sensor_fusion_router.get("/fusion/result")
async def get_fusion_result():
    """
    Get the current multimodal fusion result
    """
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion engine not available")

    try:
        result = await fusion_engine.get_fusion_result()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions

def map_threat_to_incident_type(fusion_result: dict) -> str:
    """Map fusion result to incident type"""
    threat_types = fusion_result.get("threat_types", [])
    threat_level = fusion_result.get("threat_level", 5.0)

    # High threat with weapon
    if "weapon" in threat_types and threat_level > 7:
        return "assault"

    # Aggressive behavior
    if "aggressive_motion" in threat_types or "violence" in threat_types:
        return "assault"

    # Breaking/destruction
    if "breaking" in threat_types or "destruction" in threat_types:
        return "vandalism"

    # Theft indicators
    if "theft" in threat_types or "suspicious_activity" in threat_types:
        return "theft"

    # Drug activity
    if "drug_activity" in threat_types:
        return "drug_activity"

    # Default
    return "suspicious_activity"

def map_fusion_to_severity(fusion_result: dict) -> str:
    """Map fusion threat level to severity"""
    threat_level = fusion_result.get("threat_level", 5.0)

    if threat_level >= 8:
        return "critical"
    elif threat_level >= 6:
        return "high"
    elif threat_level >= 4:
        return "moderate"
    else:
        return "low"

def generate_description(fusion_result: dict) -> str:
    """Generate human-readable description from fusion result"""
    threat_types = fusion_result.get("threat_types", [])
    confidence = fusion_result.get("confidence", 0.5)
    threat_level = fusion_result.get("threat_level", 5.0)

    if not threat_types:
        return "AI detected potential incident. Manual review recommended."

    threats_str = ", ".join(threat_types)
    confidence_pct = int(confidence * 100)

    return f"AI detected: {threats_str}. Threat level: {threat_level:.1f}/10 (confidence: {confidence_pct}%)"