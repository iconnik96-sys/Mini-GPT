"""
FastAPI dependency injection utilities for Phase 13 backend.
"""

from phase13_api_web.backend.model_manager import ModelManager, get_model_manager

# Re-export provider
__all__ = ["ModelManager", "get_model_manager"]
