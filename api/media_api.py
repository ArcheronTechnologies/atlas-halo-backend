"""
Media Upload and Management API
Handle photo, video, and audio uploads for incidents
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Header, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPBearer
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import logging
import os
import hashlib
import io
from pathlib import Path

from ..auth.jwt_authentication import get_current_user
from ..database.postgis_database import get_database

# Celery task imports (async AI processing)
try:
    from ..tasks.ai_tasks import analyze_photo_task, analyze_video_task, analyze_audio_task
    from ..tasks.media_tasks import generate_thumbnail_task
    CELERY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Celery tasks loaded - async AI processing enabled")
except ImportError:
    CELERY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Celery not available - AI processing will be synchronous")

security = HTTPBearer()

router = APIRouter(prefix="/api/v1/media", tags=["media"])

# Configuration
MEDIA_STORAGE_PATH = Path(os.getenv('MEDIA_STORAGE_PATH', './media_storage'))
MEDIA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/heic'}
ALLOWED_VIDEO_TYPES = {'video/mp4', 'video/quicktime', 'video/x-m4v'}
ALLOWED_AUDIO_TYPES = {'audio/mpeg', 'audio/mp4', 'audio/x-m4a'}


# Response Models
class MediaUploadResponse(BaseModel):
    """Response after successful media upload"""
    media_id: str
    media_type: str
    original_filename: str
    file_size: int
    upload_url: str
    thumbnail_url: Optional[str] = None
    uploaded_at: datetime


class MediaInfo(BaseModel):
    """Media file information"""
    media_id: str
    media_type: str
    original_filename: str
    file_size: int
    mime_type: str
    uploaded_at: datetime
    incident_id: Optional[str]
    user_id: str


# Helper Functions
def generate_media_id(file_data: bytes) -> str:
    """Generate unique media ID from file hash"""
    return hashlib.sha256(file_data).hexdigest()[:32]


def get_media_type(content_type: str) -> str:
    """Determine media type from MIME type"""
    if content_type in ALLOWED_IMAGE_TYPES:
        return 'image'
    elif content_type in ALLOWED_VIDEO_TYPES:
        return 'video'
    elif content_type in ALLOWED_AUDIO_TYPES:
        return 'audio'
    else:
        return 'unknown'


def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return Path(filename).suffix.lower()


async def save_media_file(
    file_data: bytes,
    media_id: str,
    extension: str,
    media_type: str
) -> Path:
    """Save media file to storage with organized directory structure"""

    # Organize by date: YYYY/MM/DD/
    now = datetime.now()
    date_path = now.strftime('%Y/%m/%d')

    # Create directory structure
    storage_dir = MEDIA_STORAGE_PATH / media_type / date_path
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    file_path = storage_dir / f"{media_id}{extension}"

    with open(file_path, 'wb') as f:
        f.write(file_data)

    return file_path


def compress_image(image_data: bytes) -> bytes:
    """
    Compress image using PIL
    Returns compressed image data
    """
    try:
        from PIL import Image

        # Open image
        image = Image.open(io.BytesIO(image_data))

        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        # Resize if too large (max 2048px on longest side)
        max_size = 2048
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Compress to JPEG
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85, optimize=True)

        return output.getvalue()

    except ImportError:
        logger.warning("PIL not available, returning original image data")
        return image_data
    except Exception as e:
        logger.error(f"Image compression failed: {e}")
        return image_data


def create_thumbnail(image_data: bytes, size: tuple = (300, 300)) -> Optional[bytes]:
    """Create thumbnail from image"""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_data))

        # Convert RGBA to RGB
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)

        output = io.BytesIO()
        image.save(output, format='JPEG', quality=75)

        return output.getvalue()

    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return None


# =============================================================================
# ENDPOINTS
# =============================================================================

async def get_current_user_optional(authorization: Optional[str] = Header(None)):
    """Get current user if authenticated, otherwise return anonymous user"""
    if not authorization:
        # Use a fixed UUID for anonymous users (all zeros)
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}
    try:
        # Extract token and get user
        from ..auth.jwt_authentication import verify_token
        token = authorization.replace('Bearer ', '')
        payload = verify_token(token)
        # Ensure user ID is a valid UUID string
        if 'id' in payload and isinstance(payload['id'], str) and len(payload['id']) > 0:
            return payload
        else:
            # Fallback to anonymous if user ID is invalid
            return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}
    except:
        # Use a fixed UUID for anonymous users (all zeros)
        return {'id': '00000000-0000-0000-0000-000000000000', 'username': 'anonymous'}


@router.post("/upload", response_model=MediaUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile = File(..., description="Media file (image, video, or audio)"),
    incident_id: Optional[str] = Form(None, description="Associated incident ID"),
    authorization: Optional[str] = Header(None),
    db = Depends(get_database)
):
    """
    Upload media file (photo, video, or audio).

    **Authentication Optional** - Anonymous uploads allowed

    **Supported Formats:**
    - Images: JPEG, PNG, HEIC
    - Videos: MP4, MOV, M4V
    - Audio: MP3, M4A

    **Size Limit:** 50MB per file

    **Features:**
    - Automatic image compression
    - Thumbnail generation
    - File deduplication
    - Organized storage (YYYY/MM/DD/)
    """

    try:
        # Get current user (or anonymous)
        current_user = await get_current_user_optional(authorization)

        # Validate content type
        content_type = file.content_type
        media_type = get_media_type(content_type)

        if media_type == 'unknown':
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {content_type}"
            )

        # Read file data
        file_data = await file.read()
        original_size = len(file_data)

        # Check file size
        if original_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {original_size / (1024*1024):.2f}MB (max: 50MB)"
            )

        # Generate media ID
        media_id = generate_media_id(file_data)

        # Check if file already exists (deduplication)
        existing_query = "SELECT media_id, file_path FROM media_files WHERE media_id = $1"
        existing = await db.execute_query_single(existing_query, media_id)

        if existing:
            logger.info(f"🔄 Duplicate media detected: {media_id}")

            return MediaUploadResponse(
                media_id=media_id,
                media_type=media_type,
                original_filename=file.filename,
                file_size=original_size,
                upload_url=f"/api/v1/media/{media_id}",
                thumbnail_url=f"/api/v1/media/{media_id}/thumbnail" if media_type == 'image' else None,
                uploaded_at=datetime.now()
            )

        # Process based on media type
        processed_data = file_data
        thumbnail_data = None
        compressed_size = original_size

        if media_type == 'image':
            # Compress image
            processed_data = compress_image(file_data)
            compressed_size = len(processed_data)

            # Create thumbnail
            thumbnail_data = create_thumbnail(processed_data)

            logger.info(f"📸 Image compressed: {original_size/(1024*1024):.2f}MB → {compressed_size/(1024*1024):.2f}MB")

        # Save file
        extension = get_file_extension(file.filename)
        file_path = await save_media_file(processed_data, media_id, extension, media_type)

        # Save thumbnail if created
        thumbnail_path = None
        if thumbnail_data:
            thumbnail_path = await save_media_file(
                thumbnail_data,
                f"{media_id}_thumb",
                '.jpg',
                f"{media_type}/thumbnails"
            )

        # Store metadata in database
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

        metadata_query = """
        INSERT INTO media_files (
            media_id, media_type, file_path, thumbnail_path,
            original_filename, original_size, compressed_size,
            compression_ratio, mime_type, incident_id, uploaded_by, uploaded_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
        """

        await db.execute_query(
            metadata_query,
            media_id,
            media_type,
            str(file_path.relative_to(MEDIA_STORAGE_PATH)),
            str(thumbnail_path.relative_to(MEDIA_STORAGE_PATH)) if thumbnail_path else None,
            file.filename,
            original_size,
            compressed_size,
            compression_ratio,
            content_type,
            incident_id,
            current_user['id']
        )

        logger.info(f"✅ Media uploaded: {media_id} by user {current_user['id']}")

        # Queue async AI analysis if Celery is available
        if CELERY_AVAILABLE:
            absolute_file_path = str(file_path.absolute())

            if media_type == 'image':
                task = analyze_photo_task.delay(media_id, absolute_file_path)
                logger.info(f"🤖 Queued photo analysis task: {task.id} for media {media_id}")

            elif media_type == 'video':
                task = analyze_video_task.delay(media_id, absolute_file_path)
                logger.info(f"🤖 Queued video analysis task: {task.id} for media {media_id}")

            elif media_type == 'audio':
                task = analyze_audio_task.delay(media_id, absolute_file_path)
                logger.info(f"🤖 Queued audio analysis task: {task.id} for media {media_id}")
        else:
            logger.debug("Celery not available - skipping async AI analysis")

        return MediaUploadResponse(
            media_id=media_id,
            media_type=media_type,
            original_filename=file.filename,
            file_size=compressed_size,
            upload_url=f"/api/v1/media/{media_id}",
            thumbnail_url=f"/api/v1/media/{media_id}/thumbnail" if thumbnail_path else None,
            uploaded_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/{media_id}", response_class=FileResponse)
async def get_media(
    media_id: str,
    db = Depends(get_database)
):
    """
    Retrieve media file by ID.

    Returns the actual file with appropriate content type.
    """

    try:
        # Get file info from database
        query = "SELECT file_path, mime_type, original_filename FROM media_files WHERE media_id = $1"
        result = await db.execute_query_single(query, media_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media {media_id} not found"
            )

        # Construct full path
        file_path = MEDIA_STORAGE_PATH / result['file_path']

        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media file not found on disk"
            )

        return FileResponse(
            path=str(file_path),
            media_type=result['mime_type'],
            filename=result['original_filename']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving media {media_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve media: {str(e)}"
        )


@router.get("/{media_id}/thumbnail", response_class=FileResponse)
async def get_media_thumbnail(
    media_id: str,
    db = Depends(get_database)
):
    """
    Retrieve thumbnail for media file.

    Only available for images.
    """

    try:
        # Get thumbnail info from database
        query = "SELECT thumbnail_path, media_type FROM media_files WHERE media_id = $1"
        result = await db.execute_query_single(query, media_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media {media_id} not found"
            )

        if result['media_type'] != 'image' or not result['thumbnail_path']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thumbnail not available for this media"
            )

        # Construct full path
        thumbnail_path = MEDIA_STORAGE_PATH / result['thumbnail_path']

        if not thumbnail_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thumbnail not found on disk"
            )

        return FileResponse(
            path=str(thumbnail_path),
            media_type='image/jpeg'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving thumbnail {media_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve thumbnail: {str(e)}"
        )


@router.get("/{media_id}/info", response_model=MediaInfo)
async def get_media_info(
    media_id: str,
    db = Depends(get_database)
):
    """
    Get metadata about media file without downloading it.

    Returns information like file size, type, upload date, etc.
    """

    try:
        query = """
        SELECT
            media_id, media_type, original_filename,
            compressed_size as file_size, mime_type,
            uploaded_at, incident_id, uploaded_by as user_id
        FROM media_files
        WHERE media_id = $1
        """

        result = await db.execute_query_single(query, media_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media {media_id} not found"
            )

        return MediaInfo(
            media_id=result['media_id'],
            media_type=result['media_type'],
            original_filename=result['original_filename'],
            file_size=result['file_size'],
            mime_type=result['mime_type'],
            uploaded_at=result['uploaded_at'],
            incident_id=str(result['incident_id']) if result['incident_id'] else None,
            user_id=str(result['user_id'])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting media info {media_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get media info: {str(e)}"
        )


@router.delete("/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_media(
    media_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete media file.

    **Authentication Required**

    Users can only delete their own media.
    Admins can delete any media.
    """

    try:
        # Get media info
        query = "SELECT file_path, thumbnail_path, uploaded_by FROM media_files WHERE media_id = $1"
        result = await db.execute_query_single(query, media_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Media {media_id} not found"
            )

        # Check permissions
        if current_user.get('user_type') != 'admin':
            if str(result['uploaded_by']) != str(current_user['id']):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You can only delete your own media"
                )

        # Delete files from disk
        file_path = MEDIA_STORAGE_PATH / result['file_path']
        if file_path.exists():
            file_path.unlink()

        if result['thumbnail_path']:
            thumbnail_path = MEDIA_STORAGE_PATH / result['thumbnail_path']
            if thumbnail_path.exists():
                thumbnail_path.unlink()

        # Delete from database
        delete_query = "DELETE FROM media_files WHERE media_id = $1"
        await db.execute_query(delete_query, media_id)

        logger.info(f"✅ Media deleted: {media_id} by user {current_user['id']}")

        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting media {media_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete media: {str(e)}"
        )