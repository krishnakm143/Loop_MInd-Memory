"""Audit ledger database model for append-only event logging."""

from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import String, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from core.models.memory import Base


class AuditLedger(Base):
    """Append-only audit ledger for tracking all system events with signatures."""

    __tablename__ = "audit_ledger"

    # Primary key
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Event metadata
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Entity type: memory, action, adapter, user, etc.",
    )
    entity_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True, comment="ID of the affected entity"
    )

    # Event data
    event: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, comment="Event data as JSONB"
    )

    # Signature
    signature: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="Ed25519 signature of the event"
    )

    def __repr__(self) -> str:
        """String representation of AuditLedger."""
        return (
            f"<AuditLedger(event_id={self.event_id}, "
            f"tenant_id={self.tenant_id}, entity_type={self.entity_type}, "
            f"entity_id={self.entity_id}, timestamp={self.timestamp})>"
        )

