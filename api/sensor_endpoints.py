"""
Sensor Fusion API Endpoints
RESTful API for hardware-agnostic sensor data ingestion and analysis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
import json
import base64

from ..sensors.fusion_system import (
    SensorFusionSystem, SensorData, SensorType, DetectionEvent, DetectionType
)
from ..auth.jwt_authentication import require_role, UserRole, AuthenticatedUser
from ..database.postgis_database import get_database
from ..caching.redis_cache import cache_api_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sensors", tags=["sensors"])

# Global sensor fusion system instance
sensor_fusion: Optional[SensorFusionSystem] = None

async def get_fusion_system() -> SensorFusionSystem:
    """Get or initialize the global sensor fusion system"""
    global sensor_fusion
    if sensor_fusion is None:
        def hotspot_update_callback(detections: List[DetectionEvent]):
            # Update hotspot database with new detections
            asyncio.create_task(update_hotspot_database(detections))
        
        sensor_fusion = SensorFusionSystem(hotspot_update_callback=hotspot_update_callback)
        await sensor_fusion.initialize()
        logger.info("Sensor fusion system initialized")
    
    return sensor_fusion

async def update_hotspot_database(detections: List[DetectionEvent]):
    """Update hotspot database with new sensor detections"""
    try:
        db = await get_database()
        
        for detection in detections:
            # Insert detection into database for hotspot analysis
            query = """
            INSERT INTO sensor_detections 
            (event_id, sensor_id, detection_type, confidence, timestamp, latitude, longitude, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (event_id) DO NOTHING
            """
            
            metadata = {
                'description': detection.description,
                'bounding_box': detection.bounding_box,
                'audio_features': detection.audio_features
            }
            
            await db.execute_query(query, [
                detection.event_id,
                detection.sensor_id,
                detection.detection_type.value,
                detection.confidence,
                detection.timestamp,
                detection.location['lat'],
                detection.location['lng'],
                json.dumps(metadata)
            ])
            
        logger.info(f"Updated database with {len(detections)} sensor detections")
        
    except Exception as e:
        logger.error(f"Failed to update hotspot database: {e}")


@router.post("/register")
async def register_sensor(
    sensor_id: str = Form(...),
    sensor_type: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT)),
    metadata: Optional[str] = Form(None)
):
    """Register a new sensor with the fusion system"""
    try:
        fusion = await get_fusion_system()
        
        # Validate sensor type
        try:
            sensor_type_enum = SensorType(sensor_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid sensor type: {sensor_type}")
        
        location = {"lat": latitude, "lng": longitude}
        sensor_metadata = json.loads(metadata) if metadata else {}
        
        success = await fusion.register_sensor(
            sensor_id=sensor_id,
            sensor_type=sensor_type_enum,
            location=location,
            metadata=sensor_metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to register sensor")
        
        return {
            "success": True,
            "message": f"Sensor {sensor_id} registered successfully",
            "sensor_id": sensor_id,
            "location": location
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sensor registration error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/data/video")
async def ingest_video_data(
    sensor_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    video_file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Ingest video data from sensors for analysis"""
    try:
        fusion = await get_fusion_system()
        
        # Read video data
        video_data = await video_file.read()
        
        # Parse timestamp
        if timestamp:
            timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp_dt = datetime.now()
        
        # Create sensor data object
        sensor_data = SensorData(
            sensor_id=sensor_id,
            sensor_type=SensorType.SECURITY_CAMERA,  # Default, should be from registration
            timestamp=timestamp_dt,
            location={"lat": latitude, "lng": longitude},
            data_type="video",
            raw_data=video_data,
            metadata=json.loads(metadata) if metadata else {}
        )
        
        # Process the data
        detections = await fusion.process_sensor_data(sensor_data)
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "timestamp": timestamp_dt.isoformat(),
            "detections": [
                {
                    "event_id": d.event_id,
                    "detection_type": d.detection_type.value,
                    "confidence": d.confidence,
                    "description": d.description,
                    "location": d.location,
                    "bounding_box": d.bounding_box
                } for d in detections
            ],
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Video data ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process video data")


@router.post("/data/audio")
async def ingest_audio_data(
    sensor_id: str = Form(...),
    timestamp: Optional[str] = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    audio_file: UploadFile = File(...),
    sample_rate: int = Form(44100),
    metadata: Optional[str] = Form(None),
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Ingest audio data from sensors for analysis"""
    try:
        fusion = await get_fusion_system()
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Parse timestamp
        if timestamp:
            timestamp_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            timestamp_dt = datetime.now()
        
        # Create sensor data object
        audio_metadata = json.loads(metadata) if metadata else {}
        audio_metadata['sample_rate'] = sample_rate
        
        sensor_data = SensorData(
            sensor_id=sensor_id,
            sensor_type=SensorType.AUDIO_MICROPHONE,  # Default, should be from registration
            timestamp=timestamp_dt,
            location={"lat": latitude, "lng": longitude},
            data_type="audio",
            raw_data=audio_data,
            metadata=audio_metadata
        )
        
        # Process the data
        detections = await fusion.process_sensor_data(sensor_data)
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "timestamp": timestamp_dt.isoformat(),
            "detections": [
                {
                    "event_id": d.event_id,
                    "detection_type": d.detection_type.value,
                    "confidence": d.confidence,
                    "description": d.description,
                    "location": d.location,
                    "audio_features": d.audio_features
                } for d in detections
            ],
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audio data ingestion error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio data")


@router.get("/hotspots")
@cache_api_response(ttl=300)  # Cache for 5 minutes
async def get_sensor_hotspots(
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Get current hotspot data derived from sensor analysis"""
    try:
        fusion = await get_fusion_system()
        hotspot_data = await fusion.get_hotspot_data()
        
        return {
            "success": True,
            "hotspots": hotspot_data["hotspots"],
            "total_detections": hotspot_data["total_detections"],
            "analysis_period": hotspot_data["analysis_period"],
            "generated_at": hotspot_data["generated_at"]
        }
        
    except Exception as e:
        logger.error(f"Hotspot data retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve hotspot data")


@router.get("/detections/recent")
@cache_api_response(ttl=60)  # Cache for 1 minute
async def get_recent_detections(
    hours: int = 24,
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Get recent detection events from sensors"""
    try:
        fusion = await get_fusion_system()
        detections = await fusion.get_recent_detections(hours=hours)
        
        return {
            "success": True,
            "detections": [
                {
                    "event_id": d.event_id,
                    "sensor_id": d.sensor_id,
                    "detection_type": d.detection_type.value,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat(),
                    "location": d.location,
                    "description": d.description
                } for d in detections
            ],
            "period_hours": hours,
            "total_detections": len(detections)
        }
        
    except Exception as e:
        logger.error(f"Recent detections retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent detections")


@router.get("/training-data")
async def export_training_data(
    current_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Export collected training data for ML model improvement"""
    try:
        fusion = await get_fusion_system()
        training_data = await fusion.export_training_data()
        
        return {
            "success": True,
            **training_data
        }
        
    except Exception as e:
        logger.error(f"Training data export error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export training data")


@router.get("/status")
async def get_system_status(
    current_user: AuthenticatedUser = Depends(require_role(UserRole.LAW_ENFORCEMENT))
):
    """Get current sensor fusion system status"""
    try:
        fusion = await get_fusion_system()
        status = fusion.get_system_status()
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Status retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@router.post("/simulate")
async def simulate_sensor_data(
    sensor_count: int = 5,
    detection_probability: float = 0.1,
    current_user: AuthenticatedUser = Depends(require_role(UserRole.ADMIN))
):
    """Simulate sensor data for testing and demonstration"""
    try:
        fusion = await get_fusion_system()
        
        # Register mock sensors
        sensors_registered = 0
        detections_generated = 0
        
        import numpy as np
        
        for i in range(sensor_count):
            sensor_id = f"sim_sensor_{i}"
            sensor_type = SensorType.SECURITY_CAMERA
            location = {
                "lat": 59.3293 + np.random.normal(0, 0.01),
                "lng": 18.0686 + np.random.normal(0, 0.01)
            }
            
            success = await fusion.register_sensor(sensor_id, sensor_type, location)
            if success:
                sensors_registered += 1
            
            # Generate some mock data
            if np.random.random() < detection_probability:
                sensor_data = SensorData(
                    sensor_id=sensor_id,
                    sensor_type=sensor_type,
                    timestamp=datetime.now(),
                    location=location,
                    data_type="video",
                    raw_data=b"mock_simulation_data",
                    metadata={"simulation": True}
                )
                
                detections = await fusion.process_sensor_data(sensor_data)
                detections_generated += len(detections)
        
        return {
            "success": True,
            "sensors_registered": sensors_registered,
            "detections_generated": detections_generated,
            "simulation_completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to run simulation")


# Database schema creation
async def create_sensor_tables():
    """Create database tables for sensor data storage"""
    try:
        db = await get_database()
        
        # Create sensor_detections table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sensor_detections (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR(255) UNIQUE NOT NULL,
            sensor_id VARCHAR(255) NOT NULL,
            detection_type VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            latitude FLOAT NOT NULL,
            longitude FLOAT NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            
            -- Spatial index for location queries
            location_point GEOMETRY(POINT, 4326) GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)) STORED
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_sensor_detections_location ON sensor_detections USING GIST(location_point);
        CREATE INDEX IF NOT EXISTS idx_sensor_detections_timestamp ON sensor_detections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_sensor_detections_sensor_id ON sensor_detections(sensor_id);
        CREATE INDEX IF NOT EXISTS idx_sensor_detections_type ON sensor_detections(detection_type);
        """
        
        await db.execute_query(create_table_query)
        logger.info("Sensor detection tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create sensor tables: {e}")
        raise


# Initialize tables - should be called during application startup
# asyncio.create_task(create_sensor_tables())  # Commented out - will be called from main.py

async def initialize_sensor_system():
    """Initialize sensor system including database tables"""
    await create_sensor_tables()
    logger.info("Sensor system initialized")