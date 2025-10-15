from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import logging

from app.database import db_manager

logger = logging.getLogger(__name__)

config_router = APIRouter()

class ConfigUpdate(BaseModel):
    key: str
    value: Any

class ConfidenceThresholds(BaseModel):
    person: float = 0.5
    car: float = 0.5
    truck: float = 0.5
    motorcycle: float = 0.5
    weapon: float = 0.3

@config_router.get("/")
async def get_all_config():
    try:
        confidence_thresholds = await db_manager.get_config("confidence_thresholds", {})
        alerts_enabled = await db_manager.get_config("alerts_enabled", True)
        audio_alerts = await db_manager.get_config("audio_alerts", True)
        
        return {
            "confidence_thresholds": confidence_thresholds,
            "alerts_enabled": alerts_enabled,
            "audio_alerts": audio_alerts
        }
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.get("/{key}")
async def get_config_value(key: str):
    try:
        value = await db_manager.get_config(key)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
        
        return {"key": key, "value": value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config value: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/")
async def update_config(config: ConfigUpdate):
    try:
        success = await db_manager.set_config(config.key, config.value)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update config")
        
        return {"message": f"Config '{config.key}' updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.put("/confidence-thresholds")
async def update_confidence_thresholds(thresholds: ConfidenceThresholds, request: Request):
    try:
        threshold_dict = thresholds.dict()
        
        success = await db_manager.set_config("confidence_thresholds", threshold_dict)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update confidence thresholds")
        
        inference_engine = request.app.state.inference_engine
        if inference_engine:
            await inference_engine.update_confidence_thresholds(threshold_dict)
        
        return {
            "message": "Confidence thresholds updated successfully",
            "thresholds": threshold_dict
        }
        
    except Exception as e:
        logger.error(f"Error updating confidence thresholds: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/alerts/enable")
async def enable_alerts():
    try:
        success = await db_manager.set_config("alerts_enabled", True)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to enable alerts")
        
        return {"message": "Alerts enabled successfully"}
        
    except Exception as e:
        logger.error(f"Error enabling alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/alerts/disable")
async def disable_alerts():
    try:
        success = await db_manager.set_config("alerts_enabled", False)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to disable alerts")
        
        return {"message": "Alerts disabled successfully"}
        
    except Exception as e:
        logger.error(f"Error disabling alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/audio-alerts/enable")
async def enable_audio_alerts():
    try:
        success = await db_manager.set_config("audio_alerts", True)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to enable audio alerts")
        
        return {"message": "Audio alerts enabled successfully"}
        
    except Exception as e:
        logger.error(f"Error enabling audio alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@config_router.post("/audio-alerts/disable")
async def disable_audio_alerts():
    try:
        success = await db_manager.set_config("audio_alerts", False)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to disable audio alerts")
        
        return {"message": "Audio alerts disabled successfully"}
        
    except Exception as e:
        logger.error(f"Error disabling audio alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))