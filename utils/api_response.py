"""
Standardized API Response Format
Ensures consistent response structure across all endpoints
"""

from typing import Any, Optional, List, Dict
from pydantic import BaseModel
from fastapi.responses import JSONResponse


class StandardAPIResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool = True
    data: Any = None
    message: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def create_success_response(
    data: Any = None,
    message: str = None,
    metadata: Dict[str, Any] = None,
    status_code: int = 200
) -> JSONResponse:
    """Create a standardized success response"""
    response_data = {
        "success": True,
        "data": data,
    }
    if message:
        response_data["message"] = message
    if metadata:
        response_data["metadata"] = metadata

    return JSONResponse(
        content=response_data,
        status_code=status_code
    )


def create_error_response(
    error: str,
    message: str = None,
    data: Any = None,
    status_code: int = 400
) -> JSONResponse:
    """Create a standardized error response"""
    response_data = {
        "success": False,
        "error": error,
    }
    if message:
        response_data["message"] = message
    if data:
        response_data["data"] = data

    return JSONResponse(
        content=response_data,
        status_code=status_code
    )


def format_incident_response(incidents: List[Dict], total: int = None, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    """Format incidents in standardized response structure"""
    return {
        "incidents": incidents,
        "total": total if total is not None else len(incidents),
        "page": page,
        "page_size": page_size,
        "total_pages": (total // page_size) + (1 if total % page_size > 0 else 0) if total else 1
    }


def format_prediction_response(predictions: List[Dict], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format predictions in standardized response structure"""
    return {
        "predictions": predictions,
        "count": len(predictions),
        "metadata": metadata or {}
    }