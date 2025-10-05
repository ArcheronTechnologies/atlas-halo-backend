"""
Incident Comments API
Allows users to discuss and provide additional information on incidents
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import logging
from backend.database.postgis_database import PostGISDatabase

logger = logging.getLogger(__name__)
comments_router = APIRouter(prefix="/api/v1/incidents", tags=["comments"])

db = PostGISDatabase()


class CommentCreate(BaseModel):
    comment_text: str
    user_id: Optional[str] = None


class CommentResponse(BaseModel):
    id: str
    incident_id: str
    user_id: Optional[str]
    comment_text: str
    is_verified: bool
    created_at: str
    upvotes: int
    downvotes: int


@comments_router.post("/{incident_id}/comments", response_model=CommentResponse)
async def create_comment(incident_id: str, comment: CommentCreate):
    """Create a new comment on an incident"""
    try:
        async with db.pool.acquire() as conn:
            # Verify incident exists
            incident_exists = await conn.fetchval(
                'SELECT EXISTS(SELECT 1 FROM crime_incidents WHERE id = $1)',
                incident_id
            )

            if not incident_exists:
                raise HTTPException(status_code=404, detail="Incident not found")

            # Insert comment
            result = await conn.fetchrow('''
                INSERT INTO incident_comments (incident_id, user_id, comment_text)
                VALUES ($1, $2, $3)
                RETURNING id, incident_id, user_id, comment_text, is_verified,
                          created_at, upvotes, downvotes
            ''', incident_id, comment.user_id, comment.comment_text)

            return {
                "id": str(result['id']),
                "incident_id": result['incident_id'],
                "user_id": result['user_id'],
                "comment_text": result['comment_text'],
                "is_verified": result['is_verified'],
                "created_at": result['created_at'].isoformat(),
                "upvotes": result['upvotes'],
                "downvotes": result['downvotes']
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@comments_router.get("/{incident_id}/comments", response_model=List[CommentResponse])
async def get_comments(incident_id: str, limit: int = 50, offset: int = 0):
    """Get all comments for an incident"""
    try:
        async with db.pool.acquire() as conn:
            comments = await conn.fetch('''
                SELECT
                    id, incident_id, user_id, comment_text, is_verified,
                    created_at, upvotes, downvotes
                FROM incident_comments
                WHERE incident_id = $1 AND is_deleted = false
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            ''', incident_id, limit, offset)

            return [
                {
                    "id": str(comment['id']),
                    "incident_id": comment['incident_id'],
                    "user_id": comment['user_id'],
                    "comment_text": comment['comment_text'],
                    "is_verified": comment['is_verified'],
                    "created_at": comment['created_at'].isoformat(),
                    "upvotes": comment['upvotes'],
                    "downvotes": comment['downvotes']
                }
                for comment in comments
            ]

    except Exception as e:
        logger.error(f"Error fetching comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@comments_router.get("/{incident_id}/comments/count")
async def get_comment_count(incident_id: str):
    """Get total comment count for an incident"""
    try:
        async with db.pool.acquire() as conn:
            count = await conn.fetchval('''
                SELECT COUNT(*) FROM incident_comments
                WHERE incident_id = $1 AND is_deleted = false
            ''', incident_id)

            return {"incident_id": incident_id, "comment_count": count}

    except Exception as e:
        logger.error(f"Error counting comments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@comments_router.delete("/comments/{comment_id}")
async def delete_comment(comment_id: str, user_id: Optional[str] = None):
    """Soft delete a comment (mark as deleted)"""
    try:
        async with db.pool.acquire() as conn:
            # If user_id provided, verify ownership
            if user_id:
                comment = await conn.fetchrow(
                    'SELECT user_id FROM incident_comments WHERE id = $1',
                    comment_id
                )

                if not comment:
                    raise HTTPException(status_code=404, detail="Comment not found")

                if comment['user_id'] != user_id:
                    raise HTTPException(status_code=403, detail="Not authorized to delete this comment")

            # Soft delete
            result = await conn.execute('''
                UPDATE incident_comments
                SET is_deleted = true, updated_at = NOW()
                WHERE id = $1
            ''', comment_id)

            if result == "UPDATE 0":
                raise HTTPException(status_code=404, detail="Comment not found")

            return {"message": "Comment deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@comments_router.post("/comments/{comment_id}/upvote")
async def upvote_comment(comment_id: str):
    """Upvote a comment"""
    try:
        async with db.pool.acquire() as conn:
            result = await conn.fetchrow('''
                UPDATE incident_comments
                SET upvotes = upvotes + 1
                WHERE id = $1
                RETURNING upvotes
            ''', comment_id)

            if not result:
                raise HTTPException(status_code=404, detail="Comment not found")

            return {"comment_id": comment_id, "upvotes": result['upvotes']}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error upvoting comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@comments_router.post("/comments/{comment_id}/downvote")
async def downvote_comment(comment_id: str):
    """Downvote a comment"""
    try:
        async with db.pool.acquire() as conn:
            result = await conn.fetchrow('''
                UPDATE incident_comments
                SET downvotes = downvotes + 1
                WHERE id = $1
                RETURNING downvotes
            ''', comment_id)

            if not result:
                raise HTTPException(status_code=404, detail="Comment not found")

            return {"comment_id": comment_id, "downvotes": result['downvotes']}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downvoting comment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
