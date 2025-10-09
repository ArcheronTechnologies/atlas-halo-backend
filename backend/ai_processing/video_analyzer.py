"""Video analysis AI service"""
import logging
from typing import Dict, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """AI-powered video analysis using Atlas Intelligence"""

    def __init__(self):
        self.atlas_client = None
        logger.info("VideoAnalyzer initialized")

    async def _get_atlas_client(self):
        """Lazy load Atlas client to avoid circular imports"""
        if self.atlas_client is None:
            from backend.services.atlas_client import get_atlas_client
            self.atlas_client = get_atlas_client()
        return self.atlas_client

    async def analyze(self, video_data: bytes, filename: str = "video.mp4") -> Dict:
        """
        Analyze video for threats using Atlas Intelligence

        Args:
            video_data: Raw video bytes
            filename: Original filename (for content-type detection)

        Returns:
            {
                "success": bool,
                "media_type": "video",
                "objects_detected": [{"class": str, "confidence": float, "bbox": []}],
                "threats_detected": [{"type": str, "severity": str, "confidence": float}],
                "threat_level": str,
                "confidence": float,
                "processing_time_ms": int,
                "fallback": bool (optional)
            }
        """
        try:
            atlas_client = await self._get_atlas_client()

            # Analyze using Atlas Intelligence
            result = await atlas_client.analyze_media(
                file_bytes=video_data,
                media_type="video",
                filename=filename,
                analysis_depth="detailed"
            )

            logger.info(f"Video analysis complete: {len(result.get('objects_detected', []))} objects, "
                       f"{len(result.get('threats_detected', []))} threats")

            return result

        except Exception as e:
            logger.error(f"Video analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "media_type": "video",
                "objects_detected": [],
                "threats_detected": [],
                "threat_level": "unknown",
                "confidence": 0.0,
                "processing_time_ms": 0,
                "error": str(e)
            }

_video_analyzer = None

def get_video_analyzer() -> VideoAnalyzer:
    global _video_analyzer
    if _video_analyzer is None:
        _video_analyzer = VideoAnalyzer()
    return _video_analyzer
