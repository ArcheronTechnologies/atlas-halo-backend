"""
Atlas AI Main Application
FastAPI application with comprehensive API endpoints and OpenAPI documentation
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
import os
from pathlib import Path

# Security imports
from backend.security.rate_limiting import RateLimitMiddleware, rate_limiter
from backend.security.input_validation import add_security_headers

# Import routers
from backend.api.mobile_endpoints import router as mobile_router
from backend.api.auth_endpoints import router as auth_router
from backend.api.incidents_api import router as incidents_router
from backend.api.media_api import router as media_router
from backend.api.admin_endpoints import router as admin_router
from backend.websockets.websocket_endpoints import router as websocket_router
from backend.api.sensor_fusion_api import sensor_fusion_router
from backend.api.ml_training_api import ml_training_router
from backend.api.comments_api import comments_router
from backend.api.predictions_endpoints import router as predictions_router
from backend.api.ai_analysis_api import router as ai_analysis_router
from backend.api.clustering_api import router as clustering_router
from backend.api.map_api import router as map_router
from backend.api.ml_monitoring_api import router as ml_monitoring_router
from backend.api.deprecated_routes import deprecated_router

# Import database
from backend.database.postgis_database import get_database

# Import background services
from backend.services.data_ingestion_service import start_ingestion_service, stop_ingestion_service, IngestionConfig
from backend.services.prediction_scheduler import start_prediction_scheduler, stop_prediction_scheduler

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application metadata
APP_VERSION = "1.0.0"
APP_TITLE = "Atlas AI Public Safety Intelligence Platform"
APP_DESCRIPTION = """
**Atlas AI** is an advanced public safety intelligence platform that provides:

## 🌟 Key Features

- **🔍 Real-Time Threat Detection**: AI-powered analysis of video, audio, and sensor data
- **🗺️ Geospatial Intelligence**: Interactive crime hotspot mapping and predictive analytics
- **📱 Mobile Integration**: Citizen reporting with video/audio capture and sensor fusion
- **🤖 Machine Learning**: Continuous model improvement through feedback loops
- **🔒 Enterprise Security**: End-to-end encryption and role-based access control

## 🚀 API Capabilities

### Core Functions
- **Incident Management**: Create, track, and manage public safety incidents
- **User Authentication**: JWT-based authentication with role-based access control
- **Real-Time Analytics**: Live crime data analysis and pattern recognition

### AI-Powered Features
- **Sensor Fusion**: Multi-modal analysis of video, audio, and sensor data
- **Machine Learning Pipeline**: Automated model training and performance monitoring
- **Predictive Analytics**: Crime forecasting and resource optimization

### Swedish Crime Intelligence
- **Hotspot Analysis**: Specialized analytics for Swedish municipalities
- **Investigation Tools**: Advanced dashboards for law enforcement
- **Multi-Language Support**: Swedish and English interface support

## 🔒 Security & Compliance

- **End-to-End Encryption**: AES-256 encryption for all sensitive data
- **GDPR Compliance**: Privacy-first design with data anonymization
- **Role-Based Access**: Granular permissions for different user types
- **Audit Logging**: Comprehensive activity tracking for compliance

## 📊 Real-Time Capabilities

- **WebSocket Support**: Live updates for incidents and alerts
- **Streaming Analytics**: Real-time processing of sensor data
- **Push Notifications**: Instant alerts for critical events
- **Live Dashboards**: Real-time visualization of public safety metrics

## 🌐 Integration Ready

- **RESTful APIs**: Standard HTTP methods with comprehensive error handling
- **OpenAPI 3.0**: Complete API documentation with interactive examples
- **Webhook Support**: Real-time notifications to external systems
- **Mobile SDKs**: React Native integration for mobile applications

---

**Built with**: FastAPI, PostgreSQL, Redis, React Native, Docker, Kubernetes
**Version**: {version}
**Environment**: {environment}
""".format(
    version=APP_VERSION,
    environment=os.getenv('ATLAS_AI_ENV', 'development')
)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - MINIMAL for Railway deployment
    logger.info(f"🚀 Starting Atlas AI {APP_VERSION}")
    logger.info("✅ Minimal startup mode - services disabled for initial deployment")
    logger.info("🎉 Atlas AI ready to accept requests")

    yield

    # Shutdown
    logger.info("🔄 Atlas AI shutting down gracefully...")

    # Stop background services
    try:
        await stop_ingestion_service()
        logger.info("✅ Data ingestion service stopped")
    except Exception as e:
        logger.error(f"Error stopping ingestion service: {e}")

    try:
        await stop_prediction_scheduler()
        logger.info("✅ Prediction scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping prediction scheduler: {e}")

    logger.info("✅ Shutdown completed")

# Create FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Atlas AI Support",
        "email": "support@atlas-ai.com",
        "url": "https://atlas-ai.com/support"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "https://api.atlas-ai.com",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.atlas-ai.com",
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
)

# Basic middleware setup

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, global_limit=2000, global_window=3600)

# Security headers middleware
@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    return add_security_headers(response)

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Simple health check"""
    try:
        db = await get_database()
        await db.execute_query("SELECT 1")
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")

# Root endpoint
@app.get("/", tags=["info"])
async def root():
    """API information"""
    return {
        "service": "Atlas AI Public Safety Intelligence Platform",
        "version": APP_VERSION,
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

# Include API routers
app.include_router(mobile_router)
app.include_router(auth_router)
app.include_router(incidents_router)  # Comprehensive incidents API
app.include_router(media_router)  # Media upload and management
app.include_router(admin_router)  # Admin endpoints for data collection & system management
app.include_router(websocket_router)
app.include_router(sensor_fusion_router)
app.include_router(ml_training_router)
app.include_router(predictions_router)  # ML predictions API
app.include_router(ai_analysis_router)  # AI-powered photo/video/audio analysis
app.include_router(clustering_router)  # Anonymous incident report clustering
app.include_router(map_router)  # High-performance map API with H3 spatial indexing
app.include_router(ml_monitoring_router)  # ML training and model performance monitoring
app.include_router(comments_router)  # Incident comments and discussion system

# Include deprecated routes (for backward compatibility - removal date: 2026-04-03)
app.include_router(deprecated_router)

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )