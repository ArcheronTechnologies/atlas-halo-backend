"""
Atlas Intelligence Client for Halo
Connects Halo backend to centralized Atlas Intelligence ML services
"""

import logging
import httpx
import os
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record successful call"""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed call"""
        self.failures += 1
        self.last_failure_time = datetime.now()

        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker OPEN after {self.failures} failures")

    def can_attempt(self) -> bool:
        """Check if call should be attempted"""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if timeout period has passed
            if self.last_failure_time and datetime.now() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                logger.info("Circuit breaker HALF-OPEN - attempting recovery")
                return True
            return False

        # half-open state
        return True

    def is_open(self) -> bool:
        """Check if circuit is open"""
        return self.state == "open" and not self.can_attempt()


class ResponseCache:
    """Simple in-memory cache for Atlas Intelligence responses"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: Dict[str, tuple[datetime, Dict]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Optional[Dict]:
        """Get cached response if not expired"""
        if key in self.cache:
            timestamp, value = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            else:
                # Expired - remove
                del self.cache[key]
        return None

    def set(self, key: str, value: Dict):
        """Store response in cache"""
        self.cache[key] = (datetime.now(), value)

    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()


class AtlasIntelligenceClient:
    """
    Client for Atlas Intelligence APIs

    Replaces local ML models with centralized Atlas Intelligence services:
    - Threat classification
    - Visual detection (YOLOv8)
    - Audio analysis (SAIT)
    - Media analysis

    Environment Variables:
        ATLAS_INTELLIGENCE_URL: Base URL (default: http://localhost:8001)
        ATLAS_INTELLIGENCE_TIMEOUT: Request timeout in seconds (default: 30)
    """

    def __init__(self):
        self.base_url = os.getenv(
            "ATLAS_INTELLIGENCE_URL",
            "http://localhost:8001"  # Local dev default
        )
        self.timeout = float(os.getenv("ATLAS_INTELLIGENCE_TIMEOUT", "30"))

        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )

        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout_seconds=60
        )

        # Response cache for performance
        self.cache = ResponseCache(ttl_seconds=300)  # 5 minute cache

        logger.info(f"✅ Atlas Intelligence client initialized: {self.base_url}")

    async def health_check(self) -> Dict:
        """Check Atlas Intelligence service health"""
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Atlas Intelligence health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def classify_incident(
        self,
        incident_type: str,
        description: str,
        location: Dict[str, float],
        timestamp: datetime,
        user_id: str,
        severity: Optional[int] = None
    ) -> Dict:
        """
        Classify incident using Atlas Intelligence

        Maps to: POST /api/v1/halo/classify-incident

        Returns:
            {
                "incident_id": "uuid",
                "incident_type": "weapons_threat",
                "severity": 5,
                "confidence": 0.85,
                "threat_level": "critical",
                "recommended_actions": [...],
                "emergency_services_needed": true
            }
        """
        try:
            response = await self.client.post(
                "/api/v1/halo/classify-incident",
                json={
                    "incident_type": incident_type,
                    "description": description,
                    "location": location,
                    "timestamp": timestamp.isoformat(),
                    "user_id": user_id,
                    "severity": severity
                }
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"Incident classified: {result.get('incident_type')} (confidence: {result.get('confidence')})")
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Atlas classify incident failed: {e.response.status_code} - {e.response.text}")
            return self._fallback_classify(description)
        except Exception as e:
            logger.error(f"Atlas classify incident error: {e}")
            return self._fallback_classify(description)

    async def analyze_media(
        self,
        file_bytes: bytes,
        media_type: str,  # 'photo', 'video', 'audio'
        filename: str,
        analysis_depth: str = "quick"  # 'quick' or 'detailed'
    ) -> Dict:
        """
        Analyze media using Atlas Intelligence

        Maps to: POST /api/v1/analyze/media

        Returns:
            {
                "success": true,
                "media_type": "photo",
                "objects_detected": [...],
                "threats_detected": [...],
                "threat_level": "medium",
                "confidence": 0.92,
                "processing_time_ms": 450
            }
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            logger.warning("Circuit breaker OPEN - using fallback")
            return self._fallback_media_analysis(media_type)

        # Generate cache key (hash of file bytes + media type)
        import hashlib
        cache_key = f"media_{media_type}_{hashlib.md5(file_bytes).hexdigest()}"

        # Check cache (only for quick analysis to avoid stale results)
        if analysis_depth == "quick":
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache HIT for {media_type} analysis")
                return cached

        try:
            # Determine content type
            content_type = self._get_content_type(filename, media_type)

            # Upload file
            files = {
                'file': (filename, file_bytes, content_type)
            }
            data = {
                'media_type': media_type,
                'analysis_depth': analysis_depth
            }

            response = await self.client.post(
                "/api/v1/analyze/media",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()

            # Record success
            self.circuit_breaker.record_success()

            # Cache result (only for quick analysis)
            if analysis_depth == "quick":
                self.cache.set(cache_key, result)

            logger.info(f"Media analyzed: {media_type} - {len(result.get('objects_detected', []))} objects detected")
            return result

        except httpx.HTTPStatusError as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Atlas analyze media failed: {e.response.status_code} - {e.response.text}")
            return self._fallback_media_analysis(media_type)
        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Atlas analyze media error: {e}")
            return self._fallback_media_analysis(media_type)

    async def classify_threat(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Classify threat from text description

        Maps to: POST /api/v1/classify/threat

        Returns:
            {
                "classification": {
                    "threat_category": "weapons",
                    "severity": 5,
                    "confidence": 0.85
                },
                "product_mappings": {
                    "halo_incident_type": "weapon_possession"
                },
                "recommendations": [...]
            }
        """
        try:
            response = await self.client.post(
                "/api/v1/classify/threat",
                json={
                    "type": "text",
                    "data": text,
                    "context": context or {}
                }
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Atlas classify threat error: {e}")
            return self._fallback_classify(text)

    async def get_nearby_intelligence(
        self,
        lat: float,
        lon: float,
        radius_km: float = 5.0,
        threat_types: Optional[List[str]] = None
    ) -> Dict:
        """
        Get nearby threat intelligence

        Maps to: GET /api/v1/halo/intelligence/nearby
        """
        try:
            params = {
                "lat": lat,
                "lon": lon,
                "radius_km": radius_km
            }
            if threat_types:
                params["threat_types"] = ",".join(threat_types)

            response = await self.client.get(
                "/api/v1/halo/intelligence/nearby",
                params=params
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Atlas nearby intelligence error: {e}")
            return {
                "threats": [],
                "patterns": [],
                "hotspots": [],
                "recommendations": []
            }

    async def analyze_halo_threat(
        self,
        incident_id: str,
        description: str,
        media_urls: Optional[List[str]] = None,
        location: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Full threat analysis for Halo incident

        Maps to: POST /api/v1/halo/analyze
        """
        try:
            response = await self.client.post(
                "/api/v1/halo/analyze",
                json={
                    "incident_id": incident_id,
                    "description": description,
                    "media_urls": media_urls or [],
                    "location": location or {}
                }
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Atlas analyze threat error: {e}")
            return {
                "threat_level": "unknown",
                "confidence": 0.0,
                "analysis": {},
                "recommendations": []
            }

    def _get_content_type(self, filename: str, media_type: str) -> str:
        """Get content type from filename"""
        ext = filename.lower().split('.')[-1]

        if media_type == "photo":
            return {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png'
            }.get(ext, 'image/jpeg')
        elif media_type == "video":
            return {
                'mp4': 'video/mp4',
                'mov': 'video/quicktime',
                'avi': 'video/x-msvideo'
            }.get(ext, 'video/mp4')
        elif media_type == "audio":
            return {
                'wav': 'audio/wav',
                'mp3': 'audio/mpeg',
                'm4a': 'audio/mp4'
            }.get(ext, 'audio/wav')

        return 'application/octet-stream'

    def _fallback_classify(self, description: str) -> Dict:
        """Fallback classification when Atlas is unavailable"""
        logger.warning("Using fallback classification (Atlas unavailable)")
        return {
            "incident_type": "other",
            "severity": 1,
            "confidence": 0.0,
            "threat_level": "unknown",
            "recommended_actions": ["Manual review required - AI service unavailable"],
            "emergency_services_needed": False,
            "fallback": True
        }

    def _fallback_media_analysis(self, media_type: str) -> Dict:
        """Fallback media analysis when Atlas is unavailable"""
        logger.warning(f"Using fallback {media_type} analysis (Atlas unavailable)")
        return {
            "success": False,
            "media_type": media_type,
            "objects_detected": [],
            "threats_detected": [],
            "threat_level": "unknown",
            "confidence": 0.0,
            "processing_time_ms": 0,
            "fallback": True,
            "error": "Atlas Intelligence service unavailable"
        }

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
_atlas_client: Optional[AtlasIntelligenceClient] = None


def get_atlas_client() -> AtlasIntelligenceClient:
    """Get or create Atlas Intelligence client singleton"""
    global _atlas_client

    if _atlas_client is None:
        _atlas_client = AtlasIntelligenceClient()

    return _atlas_client


async def close_atlas_client():
    """Close Atlas Intelligence client"""
    global _atlas_client

    if _atlas_client is not None:
        await _atlas_client.close()
        _atlas_client = None
