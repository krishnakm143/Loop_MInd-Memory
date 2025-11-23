"""Tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "LoopMind_Memory-api"


def test_health_db(client: TestClient):
    """Test database health check endpoint."""
    response = client.get("/health/db")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert data["service"] == "postgres"


def test_health_redis(client: TestClient):
    """Test Redis health check endpoint."""
    response = client.get("/health/redis")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert data["service"] == "redis"


def test_health_qdrant(client: TestClient):
    """Test Qdrant health check endpoint."""
    response = client.get("/health/qdrant")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert data["service"] == "qdrant"


def test_health_neo4j(client: TestClient):
    """Test Neo4j health check endpoint."""
    response = client.get("/health/neo4j")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert data["service"] == "neo4j"

