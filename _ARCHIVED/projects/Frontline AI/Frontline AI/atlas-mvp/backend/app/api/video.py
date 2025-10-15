from fastapi import APIRouter, HTTPException, Request
from typing import Dict, List
import logging

from app.database import db_manager

logger = logging.getLogger(__name__)

video_router = APIRouter()

@video_router.post("/start")
async def start_stream(request: Request):
    print("=== VIDEO START API CALLED - PRINT TO STDOUT ===")
    logger.error("=== VIDEO START API CALLED - ERROR LEVEL ===")
    try:
        logger.error("Step 1: Video start API called")
        
        logger.error("Step 2: Getting inference engine from app state")
        inference_engine = request.app.state.inference_engine
        logger.error(f"Step 3: Got inference engine: {inference_engine is not None}")
        
        if not inference_engine:
            logger.error("Step 3a: No inference engine - raising HTTPException")
            raise HTTPException(status_code=500, detail="Inference engine not available")
        
        logger.error("Step 4: About to call start_processing")
        logger.error("Step 4a: Stopping any existing processing first")
        await inference_engine.stop_processing()
        logger.error("Step 4b: Now calling start_processing")
        await inference_engine.start_processing()
        logger.error("Step 5: start_processing completed")
        
        logger.error("Step 6: Setting start_streaming to True")
        inference_engine.start_streaming = True
        logger.error("Step 7: Set start_streaming to True")
        
        logger.error("Step 8: Setting broadcast callbacks")
        inference_engine.set_broadcast_callbacks({
            'broadcast_detection': request.app.state.broadcast_detection,
            'broadcast_frame': request.app.state.broadcast_frame,
            'broadcast_alert': request.app.state.broadcast_alert
        })
        logger.error("Step 9: Broadcast callbacks set")
        
        logger.error("Step 10: Returning success response")
        return {"status": "streaming_started", "message": "Video stream started successfully"}
        
    except HTTPException as he:
        logger.error(f"HTTPException in start_stream: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting stream: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.post("/stop")
async def stop_stream(request: Request):
    try:
        inference_engine = request.app.state.inference_engine
        if not inference_engine:
            raise HTTPException(status_code=500, detail="Inference engine not available")
        
        inference_engine.start_streaming = False
        await inference_engine.stop_processing()
        
        return {"status": "streaming_stopped", "message": "Video stream stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.get("/status")
async def get_stream_status(request: Request):
    try:
        inference_engine = request.app.state.inference_engine
        if not inference_engine:
            raise HTTPException(status_code=500, detail="Inference engine not available")
        
        stats = await inference_engine.get_stats()
        return {
            "status": "active" if stats['streaming'] else "inactive",
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.get("/detections")
async def get_recent_detections(limit: int = 100):
    try:
        detections = await db_manager.get_recent_detections(limit)
        return {"detections": detections}
        
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.get("/stats")
async def get_detection_stats(hours: int = 24):
    try:
        stats = await db_manager.get_detection_stats(hours)
        return {"stats": stats, "hours": hours}
        
    except Exception as e:
        logger.error(f"Error getting detection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))