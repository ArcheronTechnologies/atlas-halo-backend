import asyncio
import json
import logging
import base64
from contextlib import asynccontextmanager
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.database import db_manager
from app.inference.engine import InferenceEngine
from app.api.video import video_router
from app.api.training import training_router
from app.api.config import config_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Atlas MVP...")
    
    db_manager.create_tables()
    
    await db_manager.set_config("confidence_thresholds", {
        "person": 0.5,
        "car": 0.5,
        "truck": 0.5,
        "motorcycle": 0.5,
        "weapon": 0.3
    })
    await db_manager.set_config("alerts_enabled", True)
    await db_manager.set_config("audio_alerts", True)
    
    app.state.inference_engine = InferenceEngine()
    await app.state.inference_engine.initialize()
    
    yield
    
    logger.info("Shutting down Atlas MVP...")
    if hasattr(app.state, 'inference_engine'):
        await app.state.inference_engine.cleanup()

app = FastAPI(
    title="Atlas MVP",
    description="Edge AI Perception System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router, prefix="/api/video", tags=["video"])
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(config_router, prefix="/api/config", tags=["config"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def send_binary_message(self, data: bytes, websocket: WebSocket):
        try:
            await websocket.send_bytes(data)
        except Exception as e:
            logger.error(f"Error sending binary message: {e}")
    
    async def broadcast(self, message: str):
        if not self.active_connections:
            logger.warning("No active WebSocket connections to broadcast to")
            return
        
        logger.debug(f"Broadcasting to {len(self.active_connections)} connections, message size: {len(message)}")
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_binary(self, data: bytes, message_type: str):
        if not self.active_connections:
            logger.warning("No active WebSocket connections to broadcast to")
            return
        
        logger.debug(f"Broadcasting binary to {len(self.active_connections)} connections, data size: {len(data)} bytes")
        disconnected = []
        
        # Create a header to identify message type
        header = json.dumps({"type": message_type}).encode('utf-8')
        header_length = len(header).to_bytes(4, 'big')
        full_message = header_length + header + data
        
        for connection in self.active_connections:
            try:
                await connection.send_bytes(full_message)
            except Exception as e:
                logger.error(f"Error broadcasting binary to connection: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Atlas MVP API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message_data.get("type") == "start_stream":
                app.state.inference_engine.start_streaming = True
            elif message_data.get("type") == "stop_stream":
                app.state.inference_engine.start_streaming = False
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def broadcast_detection(detection_data: Dict):
    logger.info(f"Broadcasting detection: {detection_data.get('class_name', 'unknown')}")
    message = json.dumps({
        "type": "detection",
        "data": detection_data
    })
    await manager.broadcast(message)

async def broadcast_frame(frame_data: str):
    logger.info(f"broadcast_frame called with data length: {len(frame_data) if frame_data else 0} chars")
    
    if frame_data and frame_data.startswith('data:image/jpeg;base64,'):
        # Extract base64 data and decode to binary
        try:
            base64_data = frame_data.split(',', 1)[1]
            binary_data = base64.b64decode(base64_data)
            logger.info(f"Broadcasting binary frame data: {len(binary_data)} bytes")
            await manager.broadcast_binary(binary_data, "frame")
            return
        except Exception as e:
            logger.error(f"Error processing frame for binary broadcast: {e}")
    
    # Fallback to text message for other data
    logger.info(f"Broadcasting frame via text message")
    message = json.dumps({
        "type": "frame",
        "data": frame_data
    })
    await manager.broadcast(message)

async def broadcast_alert(alert_data: Dict):
    logger.info(f"Broadcasting alert: {alert_data.get('message', 'unknown')}")
    message = json.dumps({
        "type": "alert", 
        "data": alert_data
    })
    await manager.broadcast(message)

app.state.broadcast_detection = broadcast_detection
app.state.broadcast_frame = broadcast_frame
app.state.broadcast_alert = broadcast_alert

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )