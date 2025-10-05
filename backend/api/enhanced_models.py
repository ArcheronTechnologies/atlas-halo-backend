"""
Atlas AI Enhanced API Models
Comprehensive Pydantic models with detailed documentation for OpenAPI
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from .documentation import (
    LocationModel, ThreatLevel, RiskCategory, 
    APIResponse, DataResponse, ErrorResponse
)


class PredictionConfidence(str, Enum):
    """Prediction confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ProcessingStatus(str, Enum):
    """Processing status for async operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Enhanced Location Request Models
class LocationRiskRequest(BaseModel):
    """Request model for location-based risk assessment."""
    latitude: float = Field(
        ..., 
        ge=-90, 
        le=90, 
        description="Latitude coordinate (-90 to 90 degrees)",
        example=59.3293
    )
    longitude: float = Field(
        ..., 
        ge=-180, 
        le=180, 
        description="Longitude coordinate (-180 to 180 degrees)",
        example=18.0686
    )
    timestamp: Optional[datetime] = Field(
        None, 
        description="Optional timestamp for historical/future prediction. Defaults to current time.",
        example="2025-09-26T18:00:00Z"
    )
    include_details: bool = Field(
        True,
        description="Include detailed breakdown of risk factors"
    )
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v and v > datetime.utcnow().replace(year=datetime.utcnow().year + 1):
            raise ValueError('Timestamp cannot be more than 1 year in the future')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 59.3293,
                "longitude": 18.0686,
                "timestamp": "2025-09-26T18:00:00Z",
                "include_details": True
            }
        }


class AreaRiskRequest(BaseModel):
    """Request model for area-wide risk assessment."""
    center_latitude: float = Field(
        ..., 
        ge=-90, 
        le=90,
        description="Center latitude of the area to analyze",
        example=59.3293
    )
    center_longitude: float = Field(
        ..., 
        ge=-180, 
        le=180,
        description="Center longitude of the area to analyze", 
        example=18.0686
    )
    radius_km: float = Field(
        ..., 
        gt=0, 
        le=10,
        description="Radius in kilometers (maximum 10km for performance)",
        example=2.5
    )
    grid_size: int = Field(
        20, 
        ge=5, 
        le=50,
        description="Grid resolution for analysis (5-50 points)",
        example=25
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Analysis timestamp. Defaults to current time."
    )
    risk_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum risk threshold to include in results (0.0-1.0)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "center_latitude": 59.3293,
                "center_longitude": 18.0686,
                "radius_km": 2.5,
                "grid_size": 25,
                "timestamp": "2025-09-26T18:00:00Z",
                "risk_threshold": 0.3
            }
        }


class BatchLocationRequest(BaseModel):
    """Request model for batch location analysis."""
    locations: List[LocationRiskRequest] = Field(
        ..., 
        max_items=50,
        description="List of locations to analyze (maximum 50 per batch)"
    )
    parallel_processing: bool = Field(
        True,
        description="Enable parallel processing for faster results"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "locations": [
                    {"latitude": 59.3293, "longitude": 18.0686},
                    {"latitude": 59.3320, "longitude": 18.0649},
                    {"latitude": 59.3275, "longitude": 18.0710}
                ],
                "parallel_processing": True
            }
        }


# Enhanced Response Models
class RiskFactorDetails(BaseModel):
    """Detailed risk factor breakdown."""
    factor_name: str = Field(..., description="Name of the risk factor")
    weight: float = Field(..., ge=0, le=1, description="Factor weight (0.0-1.0)")
    value: float = Field(..., description="Factor value")
    description: str = Field(..., description="Human-readable description")
    source: str = Field(..., description="Data source for this factor")


class LocationRiskResponse(DataResponse):
    """Response model for location risk assessment."""
    data: Dict[str, Any] = Field(..., description="Risk assessment data")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Risk assessment completed successfully",
                "data": {
                    "location": {"latitude": 59.3293, "longitude": 18.0686},
                    "risk_level": 0.65,
                    "risk_category": "medium",
                    "confidence": 0.87,
                    "threat_types": ["property_crime", "public_disorder"],
                    "risk_factors": [
                        {
                            "factor_name": "historical_incidents",
                            "weight": 0.4,
                            "value": 0.72,
                            "description": "Historical crime incidents in area",
                            "source": "swedish_police_database"
                        }
                    ],
                    "recommendations": [
                        "Increased police presence recommended during evening hours",
                        "Consider additional lighting in the area"
                    ],
                    "processing_time_ms": 145.2
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


class AreaRiskResponse(DataResponse):
    """Response model for area risk assessment."""
    data: Dict[str, Any] = Field(..., description="Area risk assessment data")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Area risk assessment completed",
                "data": {
                    "area_center": {"latitude": 59.3293, "longitude": 18.0686},
                    "radius_km": 2.5,
                    "grid_size": 25,
                    "total_points": 625,
                    "high_risk_points": 89,
                    "risk_statistics": {
                        "average_risk": 0.42,
                        "max_risk": 0.91,
                        "min_risk": 0.05,
                        "risk_distribution": {
                            "low": 312,
                            "medium": 224,
                            "high": 89
                        }
                    },
                    "hotspots": [
                        {
                            "center": {"latitude": 59.3301, "longitude": 18.0695},
                            "risk_level": 0.91,
                            "radius_meters": 150
                        }
                    ],
                    "processing_time_ms": 2341.7
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


class BatchLocationResponse(DataResponse):
    """Response model for batch location analysis."""
    data: Dict[str, Any] = Field(..., description="Batch analysis results")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Batch analysis completed",
                "data": {
                    "total_locations": 3,
                    "successful_analyses": 3,
                    "failed_analyses": 0,
                    "results": [
                        {
                            "location": {"latitude": 59.3293, "longitude": 18.0686},
                            "risk_level": 0.65,
                            "risk_category": "medium",
                            "confidence": 0.87
                        }
                    ],
                    "processing_time_ms": 423.1,
                    "parallel_processing": True
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


# Model Training and Status Models
class ModelTrainingRequest(BaseModel):
    """Request model for AI model training."""
    force_retrain: bool = Field(
        False,
        description="Force retraining even if models were recently trained"
    )
    training_period_days: int = Field(
        90,
        ge=7,
        le=365,
        description="Number of days of training data to use"
    )
    model_types: Optional[List[str]] = Field(
        None,
        description="Specific model types to train (default: all models)"
    )
    validation_split: float = Field(
        0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of data to use for validation"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "force_retrain": False,
                "training_period_days": 90,
                "model_types": ["threat_detection", "risk_assessment"],
                "validation_split": 0.2
            }
        }


class ModelStatusResponse(DataResponse):
    """Response model for model status information."""
    data: Dict[str, Any] = Field(..., description="Model status data")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Model status retrieved successfully",
                "data": {
                    "total_models": 5,
                    "models_loaded": 5,
                    "last_training": "2025-09-20T14:30:00Z",
                    "training_status": "completed",
                    "model_details": [
                        {
                            "name": "threat_detection_v2",
                            "type": "neural_network", 
                            "status": "active",
                            "accuracy": 0.94,
                            "last_updated": "2025-09-20T14:30:00Z",
                            "training_samples": 125000
                        }
                    ],
                    "performance_metrics": {
                        "average_accuracy": 0.91,
                        "average_inference_time_ms": 45.2,
                        "total_predictions_today": 15234
                    }
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


class ModelValidationRequest(BaseModel):
    """Request model for model validation."""
    test_period_days: int = Field(
        7,
        ge=1,
        le=30,
        description="Number of days to use for validation testing"
    )
    validation_type: str = Field(
        "accuracy",
        description="Type of validation to perform",
        regex="^(accuracy|precision|recall|f1|comprehensive)$"
    )
    include_confusion_matrix: bool = Field(
        True,
        description="Include detailed confusion matrix in results"
    )


class SystemPerformanceResponse(DataResponse):
    """Response model for system performance analytics."""
    data: Dict[str, Any] = Field(..., description="Performance analytics data")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Performance data retrieved successfully",
                "data": {
                    "system_health": "healthy",
                    "response_times": {
                        "average_ms": 156.3,
                        "p95_ms": 285.7,
                        "p99_ms": 421.9
                    },
                    "request_statistics": {
                        "total_requests_24h": 15234,
                        "successful_requests": 15121,
                        "failed_requests": 113,
                        "success_rate": 0.9926
                    },
                    "resource_usage": {
                        "cpu_usage_percent": 23.4,
                        "memory_usage_percent": 67.2,
                        "disk_usage_percent": 45.1
                    },
                    "cache_performance": {
                        "hit_rate": 0.87,
                        "miss_rate": 0.13,
                        "eviction_rate": 0.02
                    }
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }


# Threat Detection Models
class ThreatAnalysisRequest(BaseModel):
    """Request model for threat analysis."""
    location: LocationModel = Field(..., description="Location where threat analysis is requested")
    threat_types: Optional[List[str]] = Field(
        None,
        description="Specific threat types to analyze (default: all types)"
    )
    sensitivity_level: str = Field(
        "medium",
        description="Analysis sensitivity level",
        regex="^(low|medium|high|maximum)$"
    )
    include_recommendations: bool = Field(
        True,
        description="Include actionable recommendations in response"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "location": {
                    "latitude": 59.3293,
                    "longitude": 18.0686,
                    "accuracy": 5.0
                },
                "threat_types": ["violence", "weapons", "public_disorder"],
                "sensitivity_level": "high",
                "include_recommendations": True
            }
        }


class ThreatDetectionResponse(DataResponse):
    """Response model for threat detection results."""
    data: Dict[str, Any] = Field(..., description="Threat detection data")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "message": "Threat analysis completed",
                "data": {
                    "threat_id": "threat_abc123",
                    "threat_level": "medium",
                    "threat_score": 0.67,
                    "confidence": 0.89,
                    "threat_types": ["public_disorder", "escalation_risk"],
                    "detected_objects": ["crowd", "agitated_individuals"],
                    "alert_radius_meters": 200,
                    "requires_emergency": False,
                    "recommended_actions": [
                        "Monitor situation closely",
                        "Position additional units nearby",
                        "Prepare for potential escalation"
                    ],
                    "processing_time_ms": 234.5
                },
                "timestamp": "2025-09-26T18:00:00Z"
            }
        }