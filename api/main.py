"""FastAPI application entry point."""

import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
import logging

# Windows-specific event loop policy fix for async socket issues
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from api.config import settings
from api.routes import health
from api.routes.v1 import (
    memories, query, actions, auth,
    analytics, monitoring, batch, history, export, webhooks, temporal,
    fast_query
)
from core.db.session import close_db, init_db
from core.storage.connections import get_pool_manager, warmup_connections, shutdown_connections

# Configure structured logging
log_level = getattr(logging, settings.api_log_level.upper(), logging.INFO)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(min_level=log_level),
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting LoopMind_Memory HSG v4 API")
    
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))

    # Warm up all database connections
    pool_manager = get_pool_manager()
    warmup_results = await warmup_connections()
    logger.info("Connection pools warmed up", results=warmup_results)
    
    # Start background health checks
    await pool_manager.start_health_checks(interval_seconds=30)

    yield

    # Shutdown
    logger.info("Shutting down LoopMind_Memory HSG v4 API")
    await shutdown_connections()
    await close_db()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LoopMind_Memory HSG v4 API",
    description="Production-grade memory platform for agents and applications",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(memories.router)
app.include_router(query.router)
app.include_router(actions.router)
app.include_router(analytics.router)
app.include_router(monitoring.router)
app.include_router(batch.router)
app.include_router(history.router)
app.include_router(export.router)
app.include_router(webhooks.router)
app.include_router(temporal.router)
app.include_router(fast_query.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LoopMind_Memory HSG v4 API",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.api_log_level,
    )

