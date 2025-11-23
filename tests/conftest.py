"""Pytest configuration and fixtures."""

import pytest
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from core.models.memory import Base
from core.db.session import get_async_session


# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


async def override_get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Override dependency for test database session."""
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest.fixture
async def test_db():
    """Create test database tables."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Get test database session."""
    async for session in override_get_async_session():
        yield session


@pytest.fixture
def client():
    """Create test client."""
    app.dependency_overrides[get_async_session] = override_get_async_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

