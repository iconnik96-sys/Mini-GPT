"""
Text generation endpoints: synchronous one-shot and Server-Sent Events (SSE) streaming.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from phase13_api_web.backend.schemas import GenerationRequest, GenerationResponse
from phase13_api_web.backend.dependencies import ModelManager, get_model_manager

router = APIRouter(prefix="/api/generate", tags=["Generation"])


@router.post("", response_model=GenerationResponse)
def generate_text_endpoint(
    req: GenerationRequest,
    manager: ModelManager = Depends(get_model_manager)
) -> GenerationResponse:
    """Generate text synchronously from the requested model."""
    try:
        return manager.generate(req)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/stream")
def generate_stream_endpoint(
    req: GenerationRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Generate text progressively using Server-Sent Events (SSE).
    Emits data payloads for each token and a final payload with completion metrics.
    """
    try:
        return StreamingResponse(
            manager.stream_generate(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
