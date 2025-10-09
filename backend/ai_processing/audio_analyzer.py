"""Audio analysis AI service"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """AI-powered audio analysis using Atlas Intelligence"""

    def __init__(self):
        self.atlas_client = None
        logger.info("AudioAnalyzer initialized")

    async def _get_atlas_client(self):
        """Lazy load Atlas client to avoid circular imports"""
        if self.atlas_client is None:
            from backend.services.atlas_client import get_atlas_client
            self.atlas_client = get_atlas_client()
        return self.atlas_client

    async def analyze(self, audio_data: bytes, filename: str = "audio.wav") -> Dict:
        """
        Analyze audio for threats using Atlas Intelligence

        Args:
            audio_data: Raw audio bytes
            filename: Original filename (for content-type detection)

        Returns:
            {
                "success": bool,
                "media_type": "audio",
                "threats_detected": [{"type": str, "severity": str, "confidence": float}],
                "transcript": str,
                "threat_level": str,
                "confidence": float,
                "audio_features": {...},
                "processing_time_ms": int,
                "fallback": bool (optional)
            }
        """
        try:
            atlas_client = await self._get_atlas_client()

            # Analyze using Atlas Intelligence
            result = await atlas_client.analyze_media(
                file_bytes=audio_data,
                media_type="audio",
                filename=filename,
                analysis_depth="detailed"
            )

            logger.info(f"Audio analysis complete: {len(result.get('threats_detected', []))} threats, "
                       f"transcript length: {len(result.get('transcript', ''))}")

            return result

        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            return {
                "success": False,
                "media_type": "audio",
                "threats_detected": [],
                "transcript": "",
                "threat_level": "unknown",
                "confidence": 0.0,
                "processing_time_ms": 0,
                "error": str(e)
            }

_audio_analyzer = None

def get_audio_analyzer() -> AudioAnalyzer:
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer
