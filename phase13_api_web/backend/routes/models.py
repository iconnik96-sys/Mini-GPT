"""
Model catalog and metadata endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from phase13_api_web.backend.schemas import ModelListResponse, ModelMetadataResponse
from phase13_api_web.backend.dependencies import ModelManager, get_model_manager

router = APIRouter(prefix="/api/models", tags=["Models"])


@router.get("", response_model=ModelListResponse)
def list_models(
    manager: ModelManager = Depends(get_model_manager)
) -> ModelListResponse:
    """List metadata for all supported models without loading model weights."""
    models = manager.list_models()
    return ModelListResponse(models=models)


@router.get("/{model_id}", response_model=ModelMetadataResponse)
def get_model(
    model_id: str,
    manager: ModelManager = Depends(get_model_manager)
) -> ModelMetadataResponse:
    """Retrieve detailed metadata for a specific model ID."""
    try:
        return manager.get_model_info(model_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' was not found in the model registry."
        )
