"""
FastAPI application.
Access all media processing features via REST API.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mediaforge.core.config import get_settings
from mediaforge.api.routes import image_router, video_router, audio_router, ai_router, health_router


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="MediaForge API",
        description="Comprehensive media processing REST API. Video, photo, audio, and AI-powered operations.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, tags=["Health"])
    app.include_router(image_router, prefix="/api/v1/image", tags=["Image"])
    app.include_router(video_router, prefix="/api/v1/video", tags=["Video"])
    app.include_router(audio_router, prefix="/api/v1/audio", tags=["Audio"])
    app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])

    return app
