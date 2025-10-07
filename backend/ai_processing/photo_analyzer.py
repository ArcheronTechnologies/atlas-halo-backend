"""Photo analysis AI service - Atlas Intelligence Integration"""
import logging
from typing import Dict, Optional
from backend.services.atlas_client import get_atlas_client

logger = logging.getLogger(__name__)

class PhotoAnalyzer:
    """AI-powered photo analysis (powered by Atlas Intelligence YOLOv8m)"""

    def __init__(self):
        self.atlas = get_atlas_client()
        self.model_loaded = True  # Atlas handles model loading
        logger.info("PhotoAnalyzer initialized (Atlas Intelligence + YOLOv8m)")

    async def analyze(
        self,
        image_data: bytes,
        filename: str = "photo.jpg",
        analysis_depth: str = "quick"
    ) -> Dict:
        """
        Analyze photo for threats and objects using Atlas Intelligence

        Returns:
            {
                "success": true,
                "objects_detected": [
                    {"class": "person", "confidence": 0.95, "bbox": [...]},
                    ...
                ],
                "threats_detected": [...],
                "threat_level": "medium",
                "confidence": 0.92,
                "processing_time_ms": 450
            }
        """
        try:
            logger.info(f"Analyzing photo via Atlas Intelligence ({len(image_data)} bytes)")

            result = await self.atlas.analyze_media(
                file_bytes=image_data,
                media_type="photo",
                filename=filename,
                analysis_depth=analysis_depth
            )

            logger.info(
                f"Photo analyzed: {len(result.get('objects_detected', []))} objects, "
                f"{len(result.get('threats_detected', []))} threats"
            )

            return result

        except Exception as e:
            logger.error(f"Photo analysis failed: {e}")
            return {
                "success": False,
                "objects_detected": [],
                "threats_detected": [],
                "confidence": 0.0,
                "analysis_time_ms": 0,
                "error": str(e)
            }

# Singleton instance
_photo_analyzer = None

def get_photo_analyzer() -> PhotoAnalyzer:
    """Get or create photo analyzer singleton"""
    global _photo_analyzer
    if _photo_analyzer is None:
        _photo_analyzer = PhotoAnalyzer()
    return _photo_analyzer
