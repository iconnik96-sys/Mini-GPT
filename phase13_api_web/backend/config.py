"""
Configuration settings for Phase 13 FastAPI backend.
"""

import os
import sys
import importlib.util
from typing import List
from pydantic import BaseModel

# Ensure project root is at front of sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if sys.path[0] != PROJECT_ROOT:
    if PROJECT_ROOT in sys.path:
        sys.path.remove(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

# Safeguard against root config.py shadowing when tests run from subdirectories
try:
    root_config_path = os.path.join(PROJECT_ROOT, "config.py")
    if os.path.exists(root_config_path):
        spec = importlib.util.spec_from_file_location("root_config", root_config_path)
        if spec and spec.loader:
            root_cfg_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(root_cfg_mod)
            MiniGPTConfig = getattr(root_cfg_mod, "MiniGPTConfig", None)
except Exception:
    pass


class Settings(BaseModel):
    app_name: str = "MiniGPT Studio API"
    app_version: str = "1.0.0"
    app_description: str = (
        "REST API & Server-Sent Events backend for local MiniGPT scratch models "
        "and fine-tuned DistilGPT-2 models."
    )
    host: str = os.getenv("MINIGPT_HOST", "127.0.0.1")
    port: int = int(os.getenv("MINIGPT_PORT", "8000"))
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    default_device: str = os.getenv("MINIGPT_DEVICE", "cpu")
    log_level: str = os.getenv("MINIGPT_LOG_LEVEL", "info")


settings = Settings()
