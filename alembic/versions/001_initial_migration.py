"""Initial migration: create memories and audit_ledger tables

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create memories table
    op.create_table(
        'memories',
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(255), nullable=False),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('content', postgresql.JSONB, nullable=False),
        sa.Column('structured', postgresql.JSONB, nullable=True),
        sa.Column('tags', postgresql.JSON, nullable=True),
        sa.Column('importance', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('trust_score', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('status', sa.String(50), nullable=False, server_default='ingested'),
        sa.Column('provenance', postgresql.JSONB, nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_memories_memory_id', 'memories', ['memory_id'])
    op.create_index('ix_memories_tenant_id', 'memories', ['tenant_id'])
    op.create_index('ix_memories_user_id', 'memories', ['user_id'])
    op.create_index('ix_memories_type', 'memories', ['type'])
    op.create_index('ix_memories_status', 'memories', ['status'])

    # Create audit_ledger table
    op.create_table(
        'audit_ledger',
        sa.Column('event_id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('entity_type', sa.String(50), nullable=False),
        sa.Column('entity_id', sa.String(255), nullable=False),
        sa.Column('event', postgresql.JSONB, nullable=False),
        sa.Column('signature', sa.Text(), nullable=True),
    )
    op.create_index('ix_audit_ledger_event_id', 'audit_ledger', ['event_id'])
    op.create_index('ix_audit_ledger_timestamp', 'audit_ledger', ['timestamp'])
    op.create_index('ix_audit_ledger_tenant_id', 'audit_ledger', ['tenant_id'])
    op.create_index('ix_audit_ledger_entity_type', 'audit_ledger', ['entity_type'])
    op.create_index('ix_audit_ledger_entity_id', 'audit_ledger', ['entity_id'])


def downgrade() -> None:
    op.drop_table('audit_ledger')
    op.drop_table('memories')

