from mediaforge.api.routes.image import router as image_router
from mediaforge.api.routes.video import router as video_router
from mediaforge.api.routes.audio import router as audio_router
from mediaforge.api.routes.ai import router as ai_router
from mediaforge.api.routes.health import router as health_router

__all__ = ["image_router", "video_router", "audio_router", "ai_router", "health_router"]
