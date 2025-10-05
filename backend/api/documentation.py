"""
Atlas AI API Documentation Enhancement
Comprehensive OpenAPI documentation utilities and response models
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResponseStatus(str, Enum):
    """Standard response status values."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PROCESSING = "processing"


class ErrorCode(str, Enum):
    """Standardized error codes."""
    VALIDATION_ERROR = "VAL_001"
    AUTHENTICATION_ERROR = "AUTH_001"
    AUTHORIZATION_ERROR = "AUTH_002"
    RATE_LIMIT_ERROR = "RATE_001"
    SERVICE_UNAVAILABLE = "SVC_001"
    INTERNAL_ERROR = "INT_001"
    NOT_FOUND = "NOT_001"
    CONFLICT = "CON_001"


class APIResponse(BaseModel):
    """Base response model for all API endpoints."""
    status: ResponseStatus = Field(..., description="Response status")
    message: str = Field(..., description="Human-readable message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Operation completed successfully",
                "timestamp": "2025-09-26T18:00:00Z",
                "request_id": "req_abc123"
            }
        }


class ErrorResponse(APIResponse):
    """Error response model."""
    error_code: ErrorCode = Field(..., description="Standardized error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "message": "Invalid input parameters",
                "error_code": "VAL_001",
                "error_details": {"field": "latitude", "value": "invalid"},
                "timestamp": "2025-09-26T18:00:00Z",
                "request_id": "req_abc123"
            }
        }


class DataResponse(APIResponse):
    """Response model with data payload."""
    data: Any = Field(..., description="Response data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success", 
                "message": "Data retrieved successfully",
                "data": {"key": "value"},
                "metadata": {"total_count": 1, "page": 1},
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


class PaginatedResponse(DataResponse):
    """Paginated response model."""
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Data retrieved successfully", 
                "data": [{"id": 1}, {"id": 2}],
                "pagination": {
                    "page": 1,
                    "per_page": 20,
                    "total_pages": 5,
                    "total_items": 100,
                    "has_next": True,
                    "has_previous": False
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


class LocationModel(BaseModel):
    """Geographic location model."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate (-180 to 180)")
    accuracy: Optional[float] = Field(None, ge=0, description="GPS accuracy in meters")
    altitude: Optional[float] = Field(None, description="Altitude in meters")
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 59.3293,
                "longitude": 18.0686,
                "accuracy": 5.0,
                "altitude": 25.0
            }
        }


class ThreatLevel(str, Enum):
    """Threat level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Risk category classification."""
    VIOLENT_CRIME = "violent_crime"
    PROPERTY_CRIME = "property_crime"
    DRUG_RELATED = "drug_related"
    TRAFFIC_INCIDENT = "traffic_incident"
    PUBLIC_DISORDER = "public_disorder"
    TERRORISM = "terrorism"
    OTHER = "other"


class UserRole(str, Enum):
    """User role enumeration."""
    CITIZEN = "citizen"
    LAW_ENFORCEMENT = "law_enforcement"
    ADMIN = "admin"
    SYSTEM = "system"


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# Enhanced documentation for OpenAPI
OPENAPI_TAGS = [
    {
        "name": "Authentication",
        "description": """
**User Authentication & Authorization**

Secure JWT-based authentication system with role-based access control.
        
Features:
- JWT token authentication
- Role-based permissions (Citizen, Law Enforcement, Admin, System)
- Secure password hashing
- Token refresh mechanism
        """,
    },
    {
        "name": "Risk Assessment", 
        "description": """
**Crime Risk Assessment Engine**

AI-powered risk assessment for locations using multiple data sources.

Features:
- Real-time location risk scoring
- Historical crime data analysis
- Area-wide risk mapping
- Temporal risk patterns
- Swedish crime database integration
        """,
    },
    {
        "name": "Threat Detection",
        "description": """
**Real-Time Threat Detection**

Advanced sensor fusion for multi-modal threat detection.

Features:
- Video analysis for violence/weapons detection
- Audio analysis for distress signals
- Sensor data fusion (accelerometer, GPS, heart rate)
- Real-time threat classification
- Emergency response automation
        """,
    },
    {
        "name": "Sensor Fusion",
        "description": """
**Multi-Modal Sensor Integration**

Comprehensive sensor data processing and analysis.

Features:
- Video/audio analysis
- IoT sensor integration
- Real-time data streaming
- Cross-user behavior tracking
- Evidence collection and storage
        """,
    },
    {
        "name": "AI Training",
        "description": """
**Machine Learning Pipeline**

Continuous AI model training and improvement.

Features:
- Real Swedish crime data training
- Model performance validation
- Feedback loop integration
- Automated retraining
- A/B testing for model variants
        """,
    },
    {
        "name": "Analytics",
        "description": """
**Crime Analytics & Reporting**

Advanced analytics for crime patterns and trends.

Features:
- Crime hotspot analysis
- Temporal trend analysis
- Predictive analytics
- Custom dashboards
- Export capabilities
        """,
    },
    {
        "name": "System Health",
        "description": """
**System Monitoring & Health**

Comprehensive system health monitoring and metrics.

Features:
- Real-time health checks
- Performance metrics
- Resource monitoring
- Alert management
- Service status reporting
        """,
    },
    {
        "name": "User Management",
        "description": """
**User Account Management**

Complete user lifecycle management system.

Features:
- User registration and profiles
- Role management
- Permission controls
- Activity tracking
- Account security
        """,
    },
]


# Common HTTP status code responses for OpenAPI documentation
COMMON_RESPONSES = {
    200: {
        "description": "Successful operation",
        "model": DataResponse,
    },
    400: {
        "description": "Invalid request parameters",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Invalid input parameters",
                    "error_code": "VAL_001",
                    "error_details": {"field": "latitude", "issue": "Must be between -90 and 90"},
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    401: {
        "description": "Authentication required",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Authentication required",
                    "error_code": "AUTH_001",
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    403: {
        "description": "Insufficient permissions",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Insufficient permissions for this operation",
                    "error_code": "AUTH_002",
                    "error_details": {"required_role": "law_enforcement", "current_role": "citizen"},
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    404: {
        "description": "Resource not found",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Requested resource not found",
                    "error_code": "NOT_001",
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    422: {
        "description": "Validation error",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Request validation failed",
                    "error_code": "VAL_001",
                    "error_details": {
                        "validation_errors": [
                            {"field": "latitude", "message": "Must be a number between -90 and 90"}
                        ]
                    },
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    429: {
        "description": "Rate limit exceeded",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Rate limit exceeded",
                    "error_code": "RATE_001",
                    "error_details": {
                        "limit": 60,
                        "window": "minute",
                        "reset_time": "2025-09-26T18:01:00Z"
                    },
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    500: {
        "description": "Internal server error",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Internal server error occurred",
                    "error_code": "INT_001",
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
    503: {
        "description": "Service unavailable",
        "model": ErrorResponse,
        "content": {
            "application/json": {
                "example": {
                    "status": "error",
                    "message": "Service temporarily unavailable",
                    "error_code": "SVC_001",
                    "error_details": {"service": "ai_engine", "status": "initializing"},
                    "timestamp": "2025-09-26T18:00:00Z"
                }
            }
        }
    },
}


def create_endpoint_documentation(
    summary: str,
    description: str,
    tags: List[str],
    response_model: BaseModel = DataResponse,
    additional_responses: Optional[Dict[int, Dict]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive endpoint documentation.
    
    Args:
        summary: Brief endpoint description
        description: Detailed endpoint description with examples
        tags: OpenAPI tags for grouping
        response_model: Pydantic model for successful response
        additional_responses: Additional HTTP status responses
    
    Returns:
        Dictionary with OpenAPI documentation
    """
    responses = {
        200: {"model": response_model},
        400: COMMON_RESPONSES[400],
        401: COMMON_RESPONSES[401], 
        422: COMMON_RESPONSES[422],
        429: COMMON_RESPONSES[429],
        500: COMMON_RESPONSES[500],
    }
    
    if additional_responses:
        responses.update(additional_responses)
    
    return {
        "summary": summary,
        "description": description,
        "tags": tags,
        "responses": responses,
    }


# Security schemes for OpenAPI documentation
SECURITY_SCHEMES = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token authentication. Format: `Bearer <token>`"
    },
    "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key authentication for service-to-service calls"
    }
}


# Example requests for different endpoint types
EXAMPLE_REQUESTS = {
    "location_risk": {
        "summary": "Stockholm City Center",
        "description": "Risk assessment for Stockholm city center",
        "value": {
            "latitude": 59.3293,
            "longitude": 18.0686,
            "timestamp": "2025-09-26T18:00:00Z"
        }
    },
    "area_risk": {
        "summary": "Downtown Area Analysis",
        "description": "Risk assessment for downtown Stockholm area",
        "value": {
            "center_latitude": 59.3293,
            "center_longitude": 18.0686,
            "radius_km": 2.0,
            "grid_size": 25,
            "timestamp": "2025-09-26T18:00:00Z"
        }
    },
    "batch_locations": {
        "summary": "Multiple Location Analysis",
        "description": "Batch risk assessment for multiple locations",
        "value": {
            "locations": [
                {"latitude": 59.3293, "longitude": 18.0686},
                {"latitude": 59.3320, "longitude": 18.0649},
                {"latitude": 59.3275, "longitude": 18.0710}
            ],
            "timestamp": "2025-09-26T18:00:00Z"
        }
    },
    "threat_detection": {
        "summary": "Video Threat Analysis",
        "description": "Analyze uploaded video for potential threats",
        "value": {
            "location": {"lat": 59.3293, "lng": 18.0686},
            "timestamp": "2025-09-26T18:00:00Z"
        }
    }
}


def get_openapi_customization() -> Dict[str, Any]:
    """Get custom OpenAPI schema modifications."""
    return {
        "info": {
            "title": "Atlas AI Public Safety Intelligence Platform",
            "version": "1.0.0",
            "description": """
# Atlas AI API Documentation

**Atlas AI** is a comprehensive public safety intelligence platform that provides AI-powered crime prevention and threat detection capabilities.

## 🚀 Quick Start

1. **Authentication**: Obtain a JWT token via `/auth/login` endpoint
2. **Risk Assessment**: Use `/api/v1/predict/location` for location-based risk analysis  
3. **Threat Detection**: Upload media via `/api/sensor-fusion/analyze-video-threat`
4. **Real-time Updates**: Connect to WebSocket endpoints for live alerts

## 🔐 Authentication

All API endpoints require authentication using JWT tokens:

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
     https://api.atlas-ai.com/api/v1/predict/location
```

## 📊 Rate Limits

- **Standard Users**: 60 requests/minute, 1000 requests/hour
- **Premium Users**: 300 requests/minute, 10000 requests/hour
- **Enterprise**: Custom limits available

## 🌍 Supported Regions

Currently optimized for:
- **Sweden**: Full crime database integration
- **Nordic Countries**: Regional crime pattern analysis
- **EU**: General threat detection capabilities

## 📋 Response Format

All responses follow a consistent structure:

```json
{
  "status": "success|error|warning",
  "message": "Human-readable description",
  "data": {...},
  "timestamp": "2025-09-26T18:00:00Z",
  "request_id": "req_abc123"
}
```

## 🔍 Error Handling

Errors include standardized codes and detailed context:

```json
{
  "status": "error",
  "message": "Validation failed",
  "error_code": "VAL_001", 
  "error_details": {
    "field": "latitude",
    "issue": "Must be between -90 and 90"
  }
}
```

## 📈 Monitoring

Monitor your API usage via:
- Rate limit headers in responses
- `/health` endpoint for system status
- `/metrics` endpoint for detailed performance data

## 🆘 Support

- **Documentation**: https://docs.atlas-ai.com
- **Support**: support@atlas-ai.com
- **Status**: https://status.atlas-ai.com
            """,
            "termsOfService": "https://atlas-ai.com/terms",
            "contact": {
                "name": "Atlas AI Support",
                "url": "https://atlas-ai.com/support",
                "email": "support@atlas-ai.com"
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
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
        ],
        "tags": OPENAPI_TAGS,
        "components": {
            "securitySchemes": SECURITY_SCHEMES
        }
    }