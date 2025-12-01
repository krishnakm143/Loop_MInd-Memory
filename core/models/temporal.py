"""Temporal Knowledge Graph models for time-bound facts."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import String, Float, Integer, DateTime, Text, Index, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from core.models.memory import Base


def utc_now() -> datetime:
    """Return current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class TemporalFact(Base):
    """
    Time-bound fact model for the Temporal Knowledge Graph.
    
    Stores subject-predicate-object triples with temporal validity windows.
    Supports automatic fact evolution where new facts close old ones.
    
    Example:
        subject: "CompanyX"
        predicate: "has_CEO"
        object: "Alice"
        valid_from: 2021-01-01
        valid_to: 2024-04-10  (auto-closed when Bob became CEO)
    """

    __tablename__ = "temporal_facts"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Tenant isolation
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )

    # Triple components (Subject-Predicate-Object)
    subject: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Entity subject (e.g., 'CompanyX', 'John')",
    )
    predicate: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Relationship type (e.g., 'has_CEO', 'works_at')",
    )
    object: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Object value (e.g., 'Alice', 'New York')",
    )

    # Temporal validity window
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When this fact became true",
    )
    valid_to: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When this fact ended (None = currently true)",
    )

    # Confidence and decay
    confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False,
        comment="Confidence score 0.0-1.0, decays over time",
    )
    initial_confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False,
        comment="Original confidence when fact was created",
    )

    # Source tracking
    source_memory_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memories.memory_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Source memory this fact was extracted from",
    )
    source_type: Mapped[str] = mapped_column(
        String(50),
        default="extracted",
        nullable=False,
        comment="Source type: extracted, user_defined, inferred",
    )

    # Extra data (renamed from metadata - SQLAlchemy reserved)
    extra_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional data (context, evidence, etc.)",
    )

    # Change tracking
    change_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of times this subject-predicate has changed",
    )
    superseded_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="ID of fact that superseded this one",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    # Composite indices for common query patterns
    __table_args__ = (
        Index("idx_facts_subject_predicate", "subject", "predicate"),
        Index("idx_facts_tenant_subject", "tenant_id", "subject"),
        Index("idx_facts_time_range", "valid_from", "valid_to"),
        Index("idx_facts_subject_time", "subject", "valid_from", "valid_to"),
        Index("idx_facts_active", "tenant_id", "valid_to"),  # For current facts
    )

    def __repr__(self) -> str:
        """String representation of TemporalFact."""
        return (
            f"<TemporalFact({self.subject} {self.predicate} {self.object}, "
            f"valid: {self.valid_from} - {self.valid_to or 'current'})>"
        )

    @property
    def is_current(self) -> bool:
        """Check if this fact is currently valid."""
        return self.valid_to is None

    @property
    def duration_days(self) -> Optional[int]:
        """Get the duration this fact was/is valid in days."""
        if self.valid_to:
            return (self.valid_to - self.valid_from).days
        return (datetime.now(timezone.utc) - self.valid_from).days


class FactChange(Base):
    """
    Audit log for fact changes.
    
    Tracks when facts evolve (new value replaces old).
    """

    __tablename__ = "fact_changes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # What changed
    subject: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    predicate: Mapped[str] = mapped_column(String(255), nullable=False)

    # Change details
    old_fact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("temporal_facts.id", ondelete="CASCADE"),
        nullable=False,
    )
    new_fact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("temporal_facts.id", ondelete="CASCADE"),
        nullable=False,
    )
    old_value: Mapped[str] = mapped_column(Text, nullable=False)
    new_value: Mapped[str] = mapped_column(Text, nullable=False)

    # When it changed
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    # Reason for change
    change_reason: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Why the fact changed (e.g., 'new_information', 'correction')",
    )

    __table_args__ = (
        Index("idx_changes_subject_predicate", "subject", "predicate"),
        Index("idx_changes_tenant_time", "tenant_id", "changed_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<FactChange({self.subject} {self.predicate}: "
            f"'{self.old_value}' -> '{self.new_value}')>"
        )



