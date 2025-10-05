"""
Data Quality Validation Pipeline
Validates and enriches incidents before storage
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

from ..analytics import h3_utils

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'coordinates', 'timestamp', 'content', 'type'
    message: str
    field: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of quality validation"""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    issues: List[QualityIssue]
    enrichment: Dict[str, Any]


class DataQualityPipeline:
    """
    Validates and enriches incident data before database storage.

    Quality Checks:
    1. Coordinate validation (within Sweden, reasonable precision)
    2. Timestamp validation (not future, not too old)
    3. Content validation (description length, type mapping)
    4. Completeness scoring

    Enrichment:
    1. H3 geospatial indexing
    2. Municipality detection
    3. Historical context
    """

    # Sweden bounding box
    SWEDEN_BOUNDS = {
        'lat_min': 55.0,  # Southernmost point (Skåne)
        'lat_max': 69.5,  # Northernmost point (Lapland)
        'lon_min': 10.5,  # Westernmost point
        'lon_max': 24.5   # Easternmost point (Finnish border)
    }

    # Known Swedish municipalities (major ones)
    SWEDISH_MUNICIPALITIES = {
        'Stockholm': {'lat': 59.3293, 'lon': 18.0686, 'radius_km': 30},
        'Göteborg': {'lat': 57.7089, 'lon': 11.9746, 'radius_km': 25},
        'Malmö': {'lat': 55.6050, 'lon': 13.0038, 'radius_km': 20},
        'Uppsala': {'lat': 59.8586, 'lon': 17.6389, 'radius_km': 15},
        'Västerås': {'lat': 59.6162, 'lon': 16.5528, 'radius_km': 15},
        'Örebro': {'lat': 59.2741, 'lon': 15.2066, 'radius_km': 15},
        'Linköping': {'lat': 58.4108, 'lon': 15.6214, 'radius_km': 15},
        'Helsingborg': {'lat': 56.0465, 'lon': 12.6945, 'radius_km': 12},
        'Jönköping': {'lat': 57.7826, 'lon': 14.1618, 'radius_km': 12},
        'Norrköping': {'lat': 58.5942, 'lon': 16.1826, 'radius_km': 15},
    }

    def __init__(self, min_quality_threshold: float = 0.5):
        self.min_quality_threshold = min_quality_threshold

    async def validate_incident(
        self,
        incident_data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate incident and calculate quality score.

        Returns:
            ValidationResult with validation status, score, and issues
        """

        issues: List[QualityIssue] = []
        quality_score = 1.0
        enrichment = {}

        # 1. Validate coordinates
        coord_issues, coord_score = self._validate_coordinates(incident_data)
        issues.extend(coord_issues)
        quality_score *= coord_score

        # 2. Validate timestamp
        time_issues, time_score = self._validate_timestamp(incident_data)
        issues.extend(time_issues)
        quality_score *= time_score

        # 3. Validate content
        content_issues, content_score = self._validate_content(incident_data)
        issues.extend(content_issues)
        quality_score *= content_score

        # 4. Validate incident type
        type_issues, type_score = self._validate_incident_type(incident_data)
        issues.extend(type_issues)
        quality_score *= type_score

        # 5. Perform enrichment (only if coordinates are valid)
        if coord_score > 0.5:
            enrichment = await self._enrich_incident(incident_data)

        # Determine if valid (meets minimum threshold)
        is_valid = quality_score >= self.min_quality_threshold

        return ValidationResult(
            is_valid=is_valid,
            quality_score=max(0.0, min(1.0, quality_score)),
            issues=issues,
            enrichment=enrichment
        )

    def _validate_coordinates(
        self,
        incident_data: Dict[str, Any]
    ) -> Tuple[List[QualityIssue], float]:
        """Validate geographic coordinates"""

        issues = []
        score = 1.0

        lat = incident_data.get('latitude')
        lon = incident_data.get('longitude')

        # Check if coordinates exist
        if lat is None or lon is None:
            issues.append(QualityIssue(
                severity='critical',
                category='coordinates',
                message='Missing latitude or longitude',
                field='latitude/longitude'
            ))
            return issues, 0.0

        # Check if coordinates are numbers
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            issues.append(QualityIssue(
                severity='critical',
                category='coordinates',
                message='Invalid coordinate format',
                field='latitude/longitude'
            ))
            return issues, 0.0

        # Check if within Sweden bounds
        if not self._is_within_sweden(lat, lon):
            issues.append(QualityIssue(
                severity='critical',
                category='coordinates',
                message=f'Coordinates ({lat:.4f}, {lon:.4f}) outside Sweden',
                field='latitude/longitude'
            ))
            score *= 0.3  # Heavy penalty but not disqualifying (could be border)

        # Check coordinate precision (should have at least 4 decimal places)
        lat_str = str(lat)
        lon_str = str(lon)

        if '.' in lat_str and len(lat_str.split('.')[1]) < 4:
            issues.append(QualityIssue(
                severity='warning',
                category='coordinates',
                message='Low latitude precision (< 4 decimal places)',
                field='latitude'
            ))
            score *= 0.9

        if '.' in lon_str and len(lon_str.split('.')[1]) < 4:
            issues.append(QualityIssue(
                severity='warning',
                category='coordinates',
                message='Low longitude precision (< 4 decimal places)',
                field='longitude'
            ))
            score *= 0.9

        # Check for obviously wrong coordinates (0, 0)
        if abs(lat) < 0.001 and abs(lon) < 0.001:
            issues.append(QualityIssue(
                severity='critical',
                category='coordinates',
                message='Coordinates at (0, 0) - likely placeholder',
                field='latitude/longitude'
            ))
            return issues, 0.1

        return issues, score

    def _validate_timestamp(
        self,
        incident_data: Dict[str, Any]
    ) -> Tuple[List[QualityIssue], float]:
        """Validate occurred_at timestamp"""

        issues = []
        score = 1.0

        occurred_at = incident_data.get('occurred_at')

        if not occurred_at:
            issues.append(QualityIssue(
                severity='warning',
                category='timestamp',
                message='Missing occurred_at timestamp',
                field='occurred_at'
            ))
            return issues, 0.7  # Still usable, will use reported time

        # Convert to datetime if string
        if isinstance(occurred_at, str):
            try:
                occurred_at = datetime.fromisoformat(occurred_at)
            except ValueError:
                issues.append(QualityIssue(
                    severity='critical',
                    category='timestamp',
                    message='Invalid timestamp format',
                    field='occurred_at'
                ))
                return issues, 0.3

        # Check if timestamp is in the future
        now = datetime.now()
        if occurred_at > now + timedelta(hours=1):  # Allow 1 hour clock skew
            issues.append(QualityIssue(
                severity='critical',
                category='timestamp',
                message=f'Future timestamp: {occurred_at}',
                field='occurred_at'
            ))
            score *= 0.2

        # Check if timestamp is too old
        age_days = (now - occurred_at).days

        if age_days > 365:
            issues.append(QualityIssue(
                severity='warning',
                category='timestamp',
                message=f'Very old incident ({age_days} days)',
                field='occurred_at'
            ))
            score *= 0.8
        elif age_days > 90:
            issues.append(QualityIssue(
                severity='info',
                category='timestamp',
                message=f'Old incident ({age_days} days)',
                field='occurred_at'
            ))
            score *= 0.95

        return issues, score

    def _validate_content(
        self,
        incident_data: Dict[str, Any]
    ) -> Tuple[List[QualityIssue], float]:
        """Validate description and content"""

        issues = []
        score = 1.0

        description = incident_data.get('description', '')

        # Check description exists
        if not description or len(description.strip()) < 10:
            issues.append(QualityIssue(
                severity='warning',
                category='content',
                message='Missing or very short description',
                field='description'
            ))
            score *= 0.8

        # Check description length (too long might be spam)
        elif len(description) > 2000:
            issues.append(QualityIssue(
                severity='warning',
                category='content',
                message='Unusually long description',
                field='description'
            ))
            score *= 0.9

        # Check for placeholder text
        placeholder_terms = ['test', 'placeholder', 'TODO', 'FIXME', 'xxx']
        description_lower = description.lower()

        for term in placeholder_terms:
            if term in description_lower:
                issues.append(QualityIssue(
                    severity='warning',
                    category='content',
                    message=f'Description contains placeholder term: {term}',
                    field='description'
                ))
                score *= 0.7
                break

        return issues, score

    def _validate_incident_type(
        self,
        incident_data: Dict[str, Any]
    ) -> Tuple[List[QualityIssue], float]:
        """Validate incident type"""

        issues = []
        score = 1.0

        incident_type = incident_data.get('incident_type', '').lower()

        # Check if type exists
        if not incident_type or incident_type in ['', 'unknown', 'other', 'none']:
            issues.append(QualityIssue(
                severity='warning',
                category='type',
                message='Unknown or missing incident type',
                field='incident_type'
            ))
            score *= 0.85

        return issues, score

    def _is_within_sweden(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Sweden's bounding box"""
        return (
            self.SWEDEN_BOUNDS['lat_min'] <= lat <= self.SWEDEN_BOUNDS['lat_max'] and
            self.SWEDEN_BOUNDS['lon_min'] <= lon <= self.SWEDEN_BOUNDS['lon_max']
        )

    async def _enrich_incident(
        self,
        incident_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add enrichment data to incident"""

        enrichment = {}

        lat = float(incident_data.get('latitude', 0))
        lon = float(incident_data.get('longitude', 0))

        # 1. Add H3 geospatial indexes at multiple resolutions
        try:
            enrichment['h3_index_res8'] = h3_utils.latlon_to_h3(lat, lon, 8)
            enrichment['h3_index_res9'] = h3_utils.latlon_to_h3(lat, lon, 9)
            enrichment['h3_index_res10'] = h3_utils.latlon_to_h3(lat, lon, 10)
        except Exception as e:
            logger.warning(f"Failed to generate H3 index: {e}")

        # 2. Detect municipality
        municipality = self._detect_municipality(lat, lon)
        if municipality:
            enrichment['municipality'] = municipality

        # 3. Add geographic context
        enrichment['country'] = 'Sweden'
        enrichment['coordinate_precision'] = self._calculate_precision(lat, lon)

        # 4. Add temporal context
        occurred_at = incident_data.get('occurred_at')
        if occurred_at:
            if isinstance(occurred_at, str):
                occurred_at = datetime.fromisoformat(occurred_at)

            enrichment['hour_of_day'] = occurred_at.hour
            enrichment['day_of_week'] = occurred_at.strftime('%A')
            enrichment['month'] = occurred_at.month
            enrichment['is_weekend'] = occurred_at.weekday() >= 5
            enrichment['is_nighttime'] = occurred_at.hour < 6 or occurred_at.hour >= 22

        return enrichment

    def _detect_municipality(self, lat: float, lon: float) -> Optional[str]:
        """Detect which municipality the coordinates fall within"""

        for municipality, location in self.SWEDISH_MUNICIPALITIES.items():
            distance_km = self._calculate_distance(
                lat, lon,
                location['lat'], location['lon']
            )

            if distance_km <= location['radius_km']:
                return municipality

        return None

    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate approximate distance in kilometers (simplified)"""

        # Simple Euclidean approximation (good enough for short distances)
        lat_diff = lat2 - lat1
        lon_diff = lon2 - lon1

        # Convert to km (rough approximation at Sweden's latitude)
        lat_km = lat_diff * 111
        lon_km = lon_diff * 111 * 0.6  # Longitude shrinks at higher latitudes

        distance = (lat_km ** 2 + lon_km ** 2) ** 0.5

        return distance

    def _calculate_precision(self, lat: float, lon: float) -> str:
        """Calculate coordinate precision level"""

        lat_str = str(lat)
        lon_str = str(lon)

        lat_decimals = len(lat_str.split('.')[1]) if '.' in lat_str else 0
        lon_decimals = len(lon_str.split('.')[1]) if '.' in lon_str else 0

        min_decimals = min(lat_decimals, lon_decimals)

        if min_decimals >= 6:
            return 'high'  # ~10cm precision
        elif min_decimals >= 4:
            return 'medium'  # ~10m precision
        else:
            return 'low'  # ~1km precision


# Convenience function
async def validate_and_enrich_incident(
    incident_data: Dict[str, Any],
    min_quality_threshold: float = 0.5
) -> ValidationResult:
    """
    Validate and enrich an incident.

    Args:
        incident_data: Raw incident data
        min_quality_threshold: Minimum quality score to be considered valid

    Returns:
        ValidationResult with validation status and enrichment
    """

    pipeline = DataQualityPipeline(min_quality_threshold=min_quality_threshold)
    return await pipeline.validate_incident(incident_data)