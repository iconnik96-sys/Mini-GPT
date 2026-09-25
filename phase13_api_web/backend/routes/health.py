"""
Health check endpoint.
"""

from fastapi import APIRouter
from phase13_api_web.backend.schemas import HealthResponse
from phase13_api_web.backend.config import settings

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check service health and availability."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )
