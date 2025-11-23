"""Canonical memory database model."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import JSON, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Memory(Base):
    """Canonical memory model stored in PostgreSQL with JSONB."""

    __tablename__ = "memories"

    # Primary key
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Tenant and user identification
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Memory type and content
    type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Memory type: episodic, semantic, emotional, procedural, workspace",
    )
    content: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, comment="Memory content as JSONB"
    )

    # Structured data
    structured: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True, comment="Structured facts, entities, triples"
    )

    # Metadata
    tags: Mapped[List[str]] = mapped_column(
        JSON, nullable=True, default=list, comment="Tags for categorization"
    )
    importance: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False, comment="Importance score 0.0-1.0"
    )
    trust_score: Mapped[float] = mapped_column(
        Float, default=0.5, nullable=False, comment="Trust score 0.0-1.0"
    )

    # Versioning
    version: Mapped[int] = mapped_column(
        Integer, default=1, nullable=False, comment="Memory version number"
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="ingested",
        nullable=False,
        index=True,
        comment="Status: ingested, processed, merged, flagged, quarantined",
    )

    # Provenance
    provenance: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Provenance chain: list of sources and transformations",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        """String representation of Memory."""
        return (
            f"<Memory(memory_id={self.memory_id}, "
            f"tenant_id={self.tenant_id}, type={self.type}, "
            f"status={self.status}, version={self.version})>"
        )

