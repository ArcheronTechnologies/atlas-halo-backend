"""Celery tasks for AI processing"""
import logging

logger = logging.getLogger(__name__)

# Celery tasks stub - will log warning if Celery not available
try:
    from celery import shared_task

    @shared_task
    def analyze_photo_task(photo_data: bytes):
        """Async photo analysis task"""
        logger.info("Photo analysis task started")
        return {"success": True}

    @shared_task
    def analyze_video_task(video_data: bytes):
        """Async video analysis task"""
        logger.info("Video analysis task started")
        return {"success": True}

    @shared_task
    def analyze_audio_task(audio_data: bytes):
        """Async audio analysis task"""
        logger.info("Audio analysis task started")
        return {"success": True}

except ImportError:
    logger.warning("Celery not available - AI tasks will run synchronously")

    def analyze_photo_task(photo_data: bytes):
        logger.info("Photo analysis (sync)")
        return {"success": True}

    def analyze_video_task(video_data: bytes):
        logger.info("Video analysis (sync)")
        return {"success": True}

    def analyze_audio_task(audio_data: bytes):
        logger.info("Audio analysis (sync)")
        return {"success": True}
