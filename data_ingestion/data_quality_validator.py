#!/usr/bin/env python3
"""
Enhanced Data Quality Validator for Atlas AI
Validates and scores crime data quality from multiple sources
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class QualityScore(Enum):
    """Quality score levels"""
    EXCELLENT = 5
    GOOD = 4
    FAIR = 3
    POOR = 2
    UNUSABLE = 1


@dataclass
class QualityAssessment:
    """Quality assessment result for a single incident"""
    score: float
    level: QualityScore
    issues: List[str]
    strengths: List[str]
    confidence: float
    duplicate_risk: float
    completeness_score: float
    reliability_score: float


class EnhancedDataQualityValidator:
    """
    Enhanced data quality validator with comprehensive checks
    
    Features:
    - Multi-dimensional quality scoring
    - Duplicate detection with fuzzy matching
    - Content validation and classification
    - Temporal and spatial consistency checks
    - Source reliability scoring
    """
    
    def __init__(self):
        self.seen_incidents = set()
        self.location_cache = {}
        self.source_reliability = {
            'polisen_api': 0.95,
            'polisen_rss': 0.85,
            'municipal': 0.80,
            'social_media': 0.60,
            'user_report': 0.50
        }
        
        # Swedish crime type patterns for validation
        self.crime_patterns = {
            'theft': [
                r'stöld', r'stulit', r'inbrott', r'rån', r'rånade',
                r'snatteri', r'tillgrepp', r'biltillgrepp'
            ],
            'violence': [
                r'misshandel', r'våld', r'slagsmål', r'hot', r'våldtäkt',
                r'rån', r'knivrån', r'pistolhot'
            ],
            'drugs': [
                r'narkotika', r'droger', r'cannabis', r'amfetamin',
                r'kokain', r'heroin', r'smuggling'
            ],
            'traffic': [
                r'trafikolycka', r'kollision', r'påkörning',
                r'rattfylleri', r'fortkörning'
            ],
            'vandalism': [
                r'skadegörelse', r'vandalism', r'klotter',
                r'sabotage', r'åverkan'
            ]
        }
    
    def validate_incident(self, incident: Dict[str, Any]) -> QualityAssessment:
        """
        Comprehensive quality validation of a single incident
        
        Args:
            incident: Crime incident data
            
        Returns:
            QualityAssessment with detailed quality metrics
        """
        
        issues = []
        strengths = []
        
        # 1. Completeness validation
        completeness_score = self._assess_completeness(incident, issues, strengths)
        
        # 2. Content quality validation
        content_score = self._assess_content_quality(incident, issues, strengths)
        
        # 3. Temporal validation
        temporal_score = self._assess_temporal_quality(incident, issues, strengths)
        
        # 4. Spatial validation
        spatial_score = self._assess_spatial_quality(incident, issues, strengths)
        
        # 5. Source reliability
        reliability_score = self._assess_source_reliability(incident, issues, strengths)
        
        # 6. Duplicate detection
        duplicate_risk = self._assess_duplicate_risk(incident, issues, strengths)
        
        # 7. Classification accuracy
        classification_score = self._assess_classification_accuracy(incident, issues, strengths)
        
        # Calculate weighted overall score
        weights = {
            'completeness': 0.25,
            'content': 0.20,
            'temporal': 0.15,
            'spatial': 0.15,
            'reliability': 0.10,
            'duplicate': 0.10,
            'classification': 0.05
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            content_score * weights['content'] +
            temporal_score * weights['temporal'] +
            spatial_score * weights['spatial'] +
            reliability_score * weights['reliability'] +
            (1 - duplicate_risk) * weights['duplicate'] +
            classification_score * weights['classification']
        )
        
        # Determine quality level
        if overall_score >= 4.5:
            level = QualityScore.EXCELLENT
        elif overall_score >= 3.5:
            level = QualityScore.GOOD
        elif overall_score >= 2.5:
            level = QualityScore.FAIR
        elif overall_score >= 1.5:
            level = QualityScore.POOR
        else:
            level = QualityScore.UNUSABLE
        
        # Calculate confidence based on data availability
        confidence = min(1.0, (completeness_score + reliability_score) / 2.0)
        
        return QualityAssessment(
            score=overall_score,
            level=level,
            issues=issues,
            strengths=strengths,
            confidence=confidence,
            duplicate_risk=duplicate_risk,
            completeness_score=completeness_score,
            reliability_score=reliability_score
        )
    
    def _assess_completeness(self, incident: Dict[str, Any], 
                           issues: List[str], strengths: List[str]) -> float:
        """Assess data completeness"""
        
        score = 0.0
        total_fields = 0
        
        # Required fields with weights
        required_fields = {
            'title': 2.0,
            'description': 2.0,
            'datetime': 2.0,
            'location': 1.5,
            'crime_type': 1.5,
            'source': 1.0
        }
        
        for field, weight in required_fields.items():
            total_fields += weight
            
            if field in incident and incident[field]:
                if field == 'description' and len(str(incident[field])) < 10:
                    score += weight * 0.3  # Very short description
                    issues.append(f"Very short {field}")
                elif field == 'title' and len(str(incident[field])) < 5:
                    score += weight * 0.3  # Very short title
                    issues.append(f"Very short {field}")
                else:
                    score += weight
                    if field in ['title', 'description', 'datetime']:
                        strengths.append(f"Complete {field}")
            else:
                issues.append(f"Missing {field}")
        
        # Optional fields that boost quality
        optional_fields = {
            'severity_score': 0.5,
            'coordinates': 0.5,
            'url': 0.3
        }
        
        for field, weight in optional_fields.items():
            if field in incident and incident[field]:
                score += weight
                strengths.append(f"Has {field}")
                total_fields += weight
        
        return min(5.0, (score / total_fields) * 5.0)
    
    def _assess_content_quality(self, incident: Dict[str, Any],
                              issues: List[str], strengths: List[str]) -> float:
        """Assess content quality and coherence"""
        
        score = 5.0
        
        title = str(incident.get('title', '')).lower()
        description = str(incident.get('description', '')).lower()
        combined_text = f"{title} {description}"
        
        # Check for obvious spam or invalid content
        spam_indicators = [
            r'test', r'testing', r'exempel', r'sample',
            r'http://bit\.ly', r'click here', r'gratis',
            r'🎉', r'💰', r'🔥'  # Suspicious emojis
        ]
        
        for pattern in spam_indicators:
            if re.search(pattern, combined_text):
                score -= 1.0
                issues.append("Potential spam content")
                break
        
        # Check for Swedish language consistency (police reports should be in Swedish)
        swedish_indicators = [
            r'polisen', r'händelse', r'anmälan', r'misstänkt',
            r'personer', r'område', r'tid', r'plats'
        ]
        
        swedish_count = sum(1 for pattern in swedish_indicators 
                           if re.search(pattern, combined_text))
        
        if swedish_count >= 2:
            strengths.append("Swedish language content")
            score += 0.5
        elif len(combined_text) > 50 and swedish_count == 0:
            issues.append("Non-Swedish content detected")
            score -= 0.5
        
        # Check content length and informativeness
        if len(combined_text) < 20:
            issues.append("Very sparse content")
            score -= 1.0
        elif len(combined_text) > 100:
            strengths.append("Detailed content")
            score += 0.5
        
        # Check for structured information
        if re.search(r'\d{2}:\d{2}', combined_text):  # Time mentioned
            strengths.append("Time information in text")
            score += 0.2
        
        if re.search(r'[A-ZÅÄÖ][a-zåäö]+(?:gatan|vägen|torget)', combined_text):  # Street names
            strengths.append("Street address in text")
            score += 0.3
        
        return max(0.0, min(5.0, score))
    
    def _assess_temporal_quality(self, incident: Dict[str, Any],
                               issues: List[str], strengths: List[str]) -> float:
        """Assess temporal data quality"""
        
        score = 5.0
        
        if 'datetime' not in incident:
            issues.append("No datetime provided")
            return 1.0
        
        try:
            if isinstance(incident['datetime'], str):
                # Try parsing various datetime formats
                dt = self._parse_datetime(incident['datetime'])
            else:
                dt = incident['datetime']
            
            if not dt:
                issues.append("Invalid datetime format")
                return 1.0
            
            # Check if datetime is reasonable
            now = datetime.utcnow()
            
            if dt > now:
                issues.append("Future datetime")
                score -= 2.0
            elif dt < now - timedelta(days=365):
                issues.append("Very old incident (>1 year)")
                score -= 1.0
            elif dt < now - timedelta(days=7):
                # Older incidents are still valuable
                strengths.append("Historical data")
            else:
                strengths.append("Recent incident")
                score += 0.5
            
            # Check for precision
            if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                issues.append("Low temporal precision (date only)")
                score -= 0.5
            else:
                strengths.append("High temporal precision")
                score += 0.3
                
        except Exception as e:
            issues.append(f"Datetime parsing error: {e}")
            score = 1.0
        
        return max(0.0, min(5.0, score))
    
    def _assess_spatial_quality(self, incident: Dict[str, Any],
                              issues: List[str], strengths: List[str]) -> float:
        """Assess spatial data quality"""
        
        score = 3.0  # Base score
        
        location_info = incident.get('location', {})
        
        if not location_info:
            issues.append("No location information")
            return 1.0
        
        # Check for coordinates
        if 'coordinates' in location_info and location_info['coordinates']:
            coords = location_info['coordinates']
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                lat, lon = coords
                # Check if coordinates are in Sweden
                if 55.0 <= lat <= 69.0 and 10.0 <= lon <= 25.0:
                    strengths.append("Valid Swedish coordinates")
                    score += 1.5
                else:
                    issues.append("Coordinates outside Sweden")
                    score -= 1.0
            else:
                issues.append("Invalid coordinate format")
                score -= 0.5
        
        # Check for address/location names
        if 'address' in location_info and location_info['address']:
            strengths.append("Has address information")
            score += 1.0
        
        if 'city' in location_info and location_info['city']:
            strengths.append("Has city information")
            score += 0.5
        
        if 'region' in location_info and location_info['region']:
            strengths.append("Has region information")
            score += 0.3
        
        # Validate Swedish location names
        location_text = ' '.join(str(v) for v in location_info.values() if v).lower()
        
        # Common Swedish place name suffixes
        swedish_suffixes = [r'berg', r'borg', r'stad', r'köping', r'by', r'ö', r'ås', r'rud']
        
        if any(re.search(suffix, location_text) for suffix in swedish_suffixes):
            strengths.append("Swedish place names")
            score += 0.2
        
        return max(0.0, min(5.0, score))
    
    def _assess_source_reliability(self, incident: Dict[str, Any],
                                 issues: List[str], strengths: List[str]) -> float:
        """Assess source reliability"""
        
        source = incident.get('source', 'unknown').lower()
        
        reliability = self.source_reliability.get(source, 0.5)
        
        if reliability >= 0.9:
            strengths.append("High-reliability source")
        elif reliability >= 0.7:
            strengths.append("Medium-reliability source")
        else:
            issues.append("Low-reliability source")
        
        # Additional checks
        if 'url' in incident and incident['url']:
            if 'polisen.se' in incident['url']:
                reliability = max(reliability, 0.9)
                strengths.append("Official police source")
        
        return reliability * 5.0
    
    def _assess_duplicate_risk(self, incident: Dict[str, Any],
                             issues: List[str], strengths: List[str]) -> float:
        """Assess risk of being a duplicate"""
        
        # Create incident fingerprint
        fingerprint = self._create_incident_fingerprint(incident)
        
        if fingerprint in self.seen_incidents:
            issues.append("Potential exact duplicate")
            return 0.9
        
        self.seen_incidents.add(fingerprint)
        
        # Check for near-duplicates based on fuzzy matching
        duplicate_risk = self._check_fuzzy_duplicates(incident)
        
        if duplicate_risk > 0.7:
            issues.append("High duplicate risk")
        elif duplicate_risk > 0.4:
            issues.append("Moderate duplicate risk")
        else:
            strengths.append("Unique incident")
        
        return duplicate_risk
    
    def _assess_classification_accuracy(self, incident: Dict[str, Any],
                                      issues: List[str], strengths: List[str]) -> float:
        """Assess accuracy of crime type classification"""
        
        crime_type = incident.get('crime_type', '').lower()
        title = str(incident.get('title', '')).lower()
        description = str(incident.get('description', '')).lower()
        
        text = f"{title} {description}"
        
        if not crime_type or crime_type == 'other':
            # Try to detect if we can classify it
            detected_types = []
            
            for category, patterns in self.crime_patterns.items():
                if any(re.search(pattern, text) for pattern in patterns):
                    detected_types.append(category)
            
            if detected_types:
                issues.append("Misclassified - detectable crime type")
                return 2.0
            else:
                return 3.0  # Neutral - genuinely unclear
        
        # Verify classification matches content
        if crime_type in self.crime_patterns:
            patterns = self.crime_patterns[crime_type]
            if any(re.search(pattern, text) for pattern in patterns):
                strengths.append("Accurate classification")
                return 5.0
            else:
                issues.append("Classification doesn't match content")
                return 2.0
        
        return 3.0  # Neutral for unknown crime types
    
    def _create_incident_fingerprint(self, incident: Dict[str, Any]) -> str:
        """Create unique fingerprint for duplicate detection"""
        
        key_fields = [
            str(incident.get('title', '')).strip().lower(),
            str(incident.get('datetime', '')),
            str(incident.get('location', {}).get('address', '')).strip().lower()
        ]
        
        fingerprint_data = '|'.join(key_fields)
        return hashlib.md5(fingerprint_data.encode('utf-8')).hexdigest()
    
    def _check_fuzzy_duplicates(self, incident: Dict[str, Any]) -> float:
        """Check for fuzzy duplicate matches"""
        
        # Simple fuzzy duplicate detection based on title and location similarity
        # In production, this could use more sophisticated algorithms
        
        title = str(incident.get('title', '')).lower()
        location = str(incident.get('location', {}).get('address', '')).lower()
        
        # For now, return low risk unless we detect obvious patterns
        duplicate_indicators = [
            len(title) < 10,  # Very short titles are often duplicates
            'test' in title,
            not title.strip()
        ]
        
        risk = sum(0.3 for indicator in duplicate_indicators if indicator)
        return min(0.9, risk)
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse datetime string with multiple format support"""
        
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def batch_validate(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a batch of incidents and return summary statistics"""
        
        assessments = []
        quality_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unusable': 0}
        common_issues = {}
        
        for incident in incidents:
            assessment = self.validate_incident(incident)
            assessments.append(assessment)
            
            # Update distribution
            quality_distribution[assessment.level.name.lower()] += 1
            
            # Count issues
            for issue in assessment.issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
        
        if assessments:
            avg_score = sum(a.score for a in assessments) / len(assessments)
            avg_confidence = sum(a.confidence for a in assessments) / len(assessments)
        else:
            avg_score = 0.0
            avg_confidence = 0.0
        
        return {
            'total_incidents': len(incidents),
            'average_quality_score': avg_score,
            'average_confidence': avg_confidence,
            'quality_distribution': quality_distribution,
            'common_issues': dict(sorted(common_issues.items(), 
                                       key=lambda x: x[1], reverse=True)[:10]),
            'assessments': assessments
        }


# Usage example
if __name__ == "__main__":
    validator = EnhancedDataQualityValidator()
    
    # Test incident
    test_incident = {
        'title': 'Stöld från bil, Malmö',
        'description': 'Anmälan kom in om stöld från bil på Storgatan i Malmö. Okänd person har brutit sig in i fordon.',
        'datetime': '2024-09-24T14:30:00',
        'location': {
            'city': 'Malmö',
            'address': 'Storgatan',
            'coordinates': [55.6059, 13.0007]
        },
        'crime_type': 'theft',
        'source': 'polisen_api',
        'url': 'https://polisen.se/aktuellt/handelser/2024/september/24/stold-fran-bil-malmo/'
    }
    
    assessment = validator.validate_incident(test_incident)
    
    print(f"Quality Score: {assessment.score:.2f} ({assessment.level.name})")
    print(f"Confidence: {assessment.confidence:.2f}")
    print(f"Issues: {', '.join(assessment.issues)}")
    print(f"Strengths: {', '.join(assessment.strengths)}")