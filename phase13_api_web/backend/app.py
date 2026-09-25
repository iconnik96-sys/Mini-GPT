"""
Main FastAPI Application for MiniGPT Studio.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from phase13_api_web.backend.config import settings
from phase13_api_web.backend.routes import health, models, generation

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits local Vite (5173), Next (3000), or curl/PowerShell
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(health.router)
app.include_router(models.router)
app.include_router(generation.router)


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/health",
        "models": "/api/models",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("phase13_api_web.backend.app:app", host=settings.host, port=settings.port, reload=True)
