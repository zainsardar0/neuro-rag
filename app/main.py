from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import app_logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    app_logger.info("=" * 50)
    app_logger.info(f"Starting {settings.app_name}")
    app_logger.info(f"Environment: {settings.app_env}")
    app_logger.info("=" * 50)
    yield
    # Shutdown
    app_logger.info(f"{settings.app_name} shutting down...")


app = FastAPI(
    title="NeuroRAG — AI Research Assistant",
    description="A production-grade RAG system for research documents",
    version="1.0.0",
    lifespan=lifespan
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }