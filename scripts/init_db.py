"""Initialize database and run migrations."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.db.session import init_db, close_db
from api.config import settings
import structlog

logger = structlog.get_logger()


async def main():
    """Initialize database."""
    logger.info("Initializing database", database_url=settings.database_url)
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        sys.exit(1)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())

