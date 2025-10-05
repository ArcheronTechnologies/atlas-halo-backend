"""
Risk prediction API endpoints for Atlas AI mobile geofencing system
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import asyncio
from enum import Enum

from ..database.secure_db import DatabaseManager
from ..ai_integration.multi_provider_ai import AIProvider
from ..analytics.advanced_patterns import AdvancedPatternAnalyzer

logger = logging.getLogger(__name__)

risk_prediction_router = APIRouter(prefix="/api/mobile", tags=["risk-prediction"])

class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class RiskPredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius: int = Field(default=1000, ge=100, le=5000)
    timeHorizon: int = Field(default=60, ge=15, le=360, description="Prediction horizon in minutes")

class RiskZonePrediction(BaseModel):
    zoneId: str
    currentRisk: RiskLevel
    predictedRisk: RiskLevel
    timeToChange: int  # minutes
    confidence: float = Field(..., ge=0, le=1)
    factors: List[str]
    recommendation: str

class RiskPredictionResponse(BaseModel):
    success: bool
    data: Dict[str, any] = None
    error: str = None

@risk_prediction_router.post("/risk-predictions")
async def get_risk_predictions(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Get predictive risk analysis for areas around a location
    """
    try:
        db = DatabaseManager()
        ai_provider = AIProvider()
        pattern_analyzer = AdvancedPatternAnalyzer()

        # Get current time and prediction window
        current_time = datetime.utcnow()
        prediction_end = current_time + timedelta(minutes=request.timeHorizon)

        # Fetch historical incident data for the area
        historical_data = await get_historical_incident_data(
            db, request.latitude, request.longitude, request.radius
        )

        # Get current conditions that affect risk
        current_conditions = await get_current_conditions(
            request.latitude, request.longitude
        )

        # Analyze patterns and make predictions
        predictions = await analyze_risk_patterns(
            pattern_analyzer,
            historical_data,
            current_conditions,
            request.timeHorizon
        )

        # Generate zone-specific predictions
        zone_predictions = await generate_zone_predictions(
            predictions,
            request.latitude,
            request.longitude,
            request.radius
        )

        response_data = {
            "predictions": zone_predictions,
            "analysisTimestamp": current_time.isoformat(),
            "predictionHorizon": request.timeHorizon,
            "confidenceLevel": calculate_overall_confidence(zone_predictions),
            "methodology": "advanced_pattern_analysis_with_ai"
        }

        return RiskPredictionResponse(
            success=True,
            data=response_data
        )

    except Exception as e:
        logger.error(f"Risk prediction failed: {str(e)}")
        return RiskPredictionResponse(
            success=False,
            error=f"Risk prediction analysis failed: {str(e)}"
        )

async def get_historical_incident_data(
    db: DatabaseManager,
    latitude: float,
    longitude: float,
    radius: int
) -> List[Dict]:
    """
    Fetch historical incident data for risk prediction analysis
    """
    query = """
    SELECT
        incident_type,
        location,
        incident_time,
        severity_level,
        resolution_status,
        weather_conditions,
        crowd_density_estimate,
        EXTRACT(DOW FROM incident_time) as day_of_week,
        EXTRACT(HOUR FROM incident_time) as hour_of_day
    FROM incidents
    WHERE ST_DWithin(
        ST_SetSRID(ST_MakePoint(location->>'longitude', location->>'latitude'), 4326)::geography,
        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
        %s
    )
    AND incident_time >= NOW() - INTERVAL '90 days'
    ORDER BY incident_time DESC
    """

    return await db.execute_query(query, (longitude, latitude, radius))

async def get_current_conditions(latitude: float, longitude: float) -> Dict:
    """
    Get current conditions that affect risk prediction
    """
    current_time = datetime.utcnow()

    conditions = {
        "timestamp": current_time.isoformat(),
        "hour_of_day": current_time.hour,
        "day_of_week": current_time.weekday(),
        "is_weekend": current_time.weekday() >= 5,
        "is_night": current_time.hour >= 22 or current_time.hour <= 5,
        "is_rush_hour": current_time.hour in [7, 8, 9, 17, 18, 19]
    }

    # Add weather data (would integrate with weather API)
    conditions.update({
        "weather": "clear",  # Would be from actual weather API
        "temperature": 20,
        "precipitation": 0,
        "visibility": "good"
    })

    # Add event data (would integrate with events API)
    conditions.update({
        "major_events_nearby": False,
        "school_hours": current_time.hour >= 8 and current_time.hour <= 15,
        "business_hours": current_time.hour >= 9 and current_time.hour <= 17
    })

    return conditions

async def analyze_risk_patterns(
    analyzer: AdvancedPatternAnalyzer,
    historical_data: List[Dict],
    current_conditions: Dict,
    time_horizon: int
) -> Dict:
    """
    Analyze historical patterns to predict future risk
    """
    try:
        # Temporal pattern analysis
        temporal_patterns = await analyzer.analyze_temporal_patterns(
            historical_data,
            time_horizon_minutes=time_horizon
        )

        # Spatial clustering analysis
        spatial_patterns = await analyzer.analyze_spatial_clusters(
            historical_data,
            current_conditions["timestamp"]
        )

        # Risk escalation patterns
        escalation_patterns = await analyzer.analyze_escalation_patterns(
            historical_data,
            current_conditions
        )

        # Weather correlation analysis
        weather_patterns = await analyzer.analyze_weather_correlations(
            historical_data,
            current_conditions
        )

        return {
            "temporal": temporal_patterns,
            "spatial": spatial_patterns,
            "escalation": escalation_patterns,
            "weather": weather_patterns,
            "base_risk": calculate_base_risk_level(historical_data),
            "trend_direction": calculate_trend_direction(historical_data)
        }

    except Exception as e:
        logger.error(f"Pattern analysis failed: {str(e)}")
        # Return conservative fallback predictions
        return {
            "temporal": {"risk_multiplier": 1.0, "confidence": 0.3},
            "spatial": {"hotspot_factor": 1.0, "confidence": 0.3},
            "escalation": {"escalation_probability": 0.1, "confidence": 0.3},
            "weather": {"weather_factor": 1.0, "confidence": 0.3},
            "base_risk": RiskLevel.MODERATE,
            "trend_direction": "stable"
        }

async def generate_zone_predictions(
    patterns: Dict,
    center_lat: float,
    center_lon: float,
    radius: int
) -> List[Dict]:
    """
    Generate specific zone predictions based on pattern analysis
    """
    predictions = []

    # Create grid of prediction zones
    zone_size = min(radius // 4, 500)  # Create 4x4 grid or max 500m zones

    for i in range(-1, 2):  # 3x3 grid around center
        for j in range(-1, 2):
            # Calculate zone center
            zone_lat = center_lat + (i * zone_size * 0.00001)
            zone_lon = center_lon + (j * zone_size * 0.00001)

            # Calculate current and predicted risk
            current_risk = calculate_zone_current_risk(
                patterns, zone_lat, zone_lon, i, j
            )

            predicted_risk, time_to_change, confidence = calculate_zone_predicted_risk(
                patterns, current_risk, zone_lat, zone_lon
            )

            # Only include zones with risk changes
            if predicted_risk != current_risk and confidence > 0.4:
                zone_id = f"zone_{center_lat:.4f}_{center_lon:.4f}_{i}_{j}"

                prediction = {
                    "zoneId": zone_id,
                    "currentRisk": current_risk,
                    "predictedRisk": predicted_risk,
                    "timeToChange": time_to_change,
                    "confidence": confidence,
                    "factors": get_risk_factors(patterns, predicted_risk > current_risk),
                    "recommendation": get_risk_recommendation(
                        current_risk, predicted_risk, time_to_change
                    ),
                    "coordinates": {
                        "latitude": zone_lat,
                        "longitude": zone_lon
                    }
                }

                predictions.append(prediction)

    # Sort by urgency (time to change and risk level)
    predictions.sort(key=lambda x: (
        get_risk_level_value(x["predictedRisk"]),
        -x["timeToChange"]
    ), reverse=True)

    return predictions

def calculate_zone_current_risk(
    patterns: Dict,
    lat: float,
    lon: float,
    grid_i: int,
    grid_j: int
) -> RiskLevel:
    """
    Calculate current risk level for a specific zone
    """
    base_risk = patterns["base_risk"]

    # Apply spatial factors
    spatial_factor = patterns["spatial"].get("hotspot_factor", 1.0)

    # Apply temporal factors
    temporal_factor = patterns["temporal"].get("risk_multiplier", 1.0)

    # Calculate combined risk score
    risk_score = get_risk_level_value(base_risk) * spatial_factor * temporal_factor

    # Convert back to risk level
    return value_to_risk_level(risk_score)

def calculate_zone_predicted_risk(
    patterns: Dict,
    current_risk: RiskLevel,
    lat: float,
    lon: float
) -> tuple:
    """
    Calculate predicted risk level, time to change, and confidence
    """
    current_value = get_risk_level_value(current_risk)

    # Apply escalation probability
    escalation_prob = patterns["escalation"].get("escalation_probability", 0.1)

    # Apply weather factors
    weather_factor = patterns["weather"].get("weather_factor", 1.0)

    # Calculate trend impact
    trend = patterns.get("trend_direction", "stable")
    trend_factor = {
        "improving": 0.8,
        "stable": 1.0,
        "worsening": 1.3
    }.get(trend, 1.0)

    # Predict risk change
    predicted_value = current_value * trend_factor * weather_factor

    if escalation_prob > 0.6:  # High escalation probability
        predicted_value += 1
    elif escalation_prob > 0.3:  # Moderate escalation probability
        predicted_value += 0.5

    predicted_risk = value_to_risk_level(predicted_value)

    # Calculate time to change based on historical patterns
    if predicted_risk != current_risk:
        # Time estimates based on escalation speed
        if escalation_prob > 0.7:
            time_to_change = 15  # 15 minutes for rapid escalation
        elif escalation_prob > 0.4:
            time_to_change = 30  # 30 minutes for moderate escalation
        else:
            time_to_change = 60  # 1 hour for slow changes
    else:
        time_to_change = 120  # No significant change expected

    # Calculate confidence based on data quality
    confidence = min(
        patterns["temporal"].get("confidence", 0.5),
        patterns["spatial"].get("confidence", 0.5),
        patterns["escalation"].get("confidence", 0.5)
    ) * 0.8  # Conservative confidence adjustment

    return predicted_risk, time_to_change, confidence

def get_risk_factors(patterns: Dict, is_increasing: bool) -> List[str]:
    """
    Get human-readable risk factors
    """
    factors = []

    if is_increasing:
        if patterns["temporal"].get("risk_multiplier", 1.0) > 1.2:
            factors.append("Historical incident pattern for this time")

        if patterns["escalation"].get("escalation_probability", 0.1) > 0.5:
            factors.append("Similar incidents nearby recently")

        if patterns["weather"].get("weather_factor", 1.0) > 1.1:
            factors.append("Weather conditions increase risk")

        if patterns.get("trend_direction") == "worsening":
            factors.append("Overall crime trend increasing in area")
    else:
        factors.append("No significant risk factors identified")

    return factors

def get_risk_recommendation(
    current_risk: RiskLevel,
    predicted_risk: RiskLevel,
    time_to_change: int
) -> str:
    """
    Get recommendation based on risk prediction
    """
    current_val = get_risk_level_value(current_risk)
    predicted_val = get_risk_level_value(predicted_risk)

    if predicted_val > current_val:
        if predicted_val >= 3:  # HIGH or CRITICAL
            return f"Consider leaving the area within {time_to_change} minutes as risk may increase significantly."
        elif predicted_val >= 2:  # MODERATE
            return f"Stay alert - risk level may increase to {predicted_risk.upper()} in {time_to_change} minutes."
        else:
            return f"Minor risk increase expected in {time_to_change} minutes. Stay aware of surroundings."
    else:
        return "Risk level expected to remain stable or improve."

def calculate_base_risk_level(historical_data: List[Dict]) -> RiskLevel:
    """
    Calculate base risk level from historical data
    """
    if not historical_data:
        return RiskLevel.SAFE

    recent_incidents = len([
        incident for incident in historical_data
        if (datetime.utcnow() - datetime.fromisoformat(
            incident['incident_time'].replace('Z', '+00:00')
        )).days <= 7
    ])

    if recent_incidents >= 10:
        return RiskLevel.HIGH
    elif recent_incidents >= 5:
        return RiskLevel.MODERATE
    elif recent_incidents >= 2:
        return RiskLevel.LOW
    else:
        return RiskLevel.SAFE

def calculate_trend_direction(historical_data: List[Dict]) -> str:
    """
    Calculate if crime trend is improving, worsening, or stable
    """
    if len(historical_data) < 4:
        return "stable"

    # Compare recent 2 weeks vs previous 2 weeks
    current_time = datetime.utcnow()

    recent_count = len([
        incident for incident in historical_data
        if (current_time - datetime.fromisoformat(
            incident['incident_time'].replace('Z', '+00:00')
        )).days <= 14
    ])

    previous_count = len([
        incident for incident in historical_data
        if 14 < (current_time - datetime.fromisoformat(
            incident['incident_time'].replace('Z', '+00:00')
        )).days <= 28
    ])

    if previous_count == 0:
        return "stable"

    change_ratio = recent_count / previous_count

    if change_ratio > 1.3:
        return "worsening"
    elif change_ratio < 0.7:
        return "improving"
    else:
        return "stable"

def calculate_overall_confidence(predictions: List[Dict]) -> float:
    """
    Calculate overall confidence in predictions
    """
    if not predictions:
        return 0.0

    return sum(p["confidence"] for p in predictions) / len(predictions)

def get_risk_level_value(risk_level: RiskLevel) -> float:
    """
    Convert risk level to numeric value for calculations
    """
    return {
        RiskLevel.SAFE: 0,
        RiskLevel.LOW: 1,
        RiskLevel.MODERATE: 2,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 4
    }.get(risk_level, 0)

def value_to_risk_level(value: float) -> RiskLevel:
    """
    Convert numeric value back to risk level
    """
    if value >= 3.5:
        return RiskLevel.CRITICAL
    elif value >= 2.5:
        return RiskLevel.HIGH
    elif value >= 1.5:
        return RiskLevel.MODERATE
    elif value >= 0.5:
        return RiskLevel.LOW
    else:
        return RiskLevel.SAFE