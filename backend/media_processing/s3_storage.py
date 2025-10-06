"""S3 storage service for media files"""
import logging
from typing import Optional, Dict
import os

logger = logging.getLogger(__name__)

class S3StorageService:
    """S3 storage service (stub)"""

    def __init__(self):
        self.bucket = os.getenv("S3_BUCKET", "atlas-ai-media")
        logger.info(f"S3Storage initialized (bucket: {self.bucket})")

    async def upload(self, file_data: bytes, filename: str) -> Dict:
        """Upload file to S3"""
        logger.info(f"Upload: {filename}")
        return {
            "success": True,
            "url": f"https://{self.bucket}.s3.amazonaws.com/{filename}",
            "key": filename
        }

    async def delete(self, key: str) -> bool:
        """Delete file from S3"""
        logger.info(f"Delete: {key}")
        return True

_s3_storage = None

def get_s3_storage() -> S3StorageService:
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3StorageService()
    return _s3_storage
