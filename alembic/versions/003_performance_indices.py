"""Add performance indices for common query patterns.

Revision ID: 003_performance_indices
Revises: 002_temporal_facts
Create Date: 2025-11-27

Performance optimization migration:
- Composite indices for common queries
- Covering indices for frequent access patterns
- Partial indices for filtered queries

Target: <100ms query latency, >350 QPS throughput.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '003_performance_indices'
down_revision = '002_temporal_facts'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ==========================================================================
    # Memories table - composite indices for common queries
    # ==========================================================================
    
    # Tenant + sector (type) lookup - most common query pattern
    op.create_index(
        'idx_memories_tenant_type',
        'memories',
        ['tenant_id', 'type'],
        postgresql_concurrently=False,  # Use False for migration safety
    )
    
    # Tenant + created_at for time-range queries
    op.create_index(
        'idx_memories_tenant_created',
        'memories',
        ['tenant_id', 'created_at'],
        postgresql_using='btree',
    )
    
    # Tenant + status for filtering by processing state
    op.create_index(
        'idx_memories_tenant_status',
        'memories',
        ['tenant_id', 'status'],
    )
    
    # Tenant + user + type for user-specific queries
    op.create_index(
        'idx_memories_tenant_user_type',
        'memories',
        ['tenant_id', 'user_id', 'type'],
    )
    
    # Importance score for priority ordering (high importance first)
    op.create_index(
        'idx_memories_importance_desc',
        'memories',
        [sa.text('importance DESC')],
    )
    
    # Created_at descending for recent memories
    op.create_index(
        'idx_memories_created_desc',
        'memories',
        [sa.text('created_at DESC')],
    )
    
    # GIN index on tags JSONB for tag-based queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_tags_gin 
        ON memories USING GIN (tags);
    """)
    
    # GIN index on content JSONB for full-text search on content
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_content_gin 
        ON memories USING GIN (content);
    """)
    
    # ==========================================================================
    # Audit ledger - composite indices for audit queries
    # ==========================================================================
    
    # Tenant + timestamp for tenant audit history
    op.create_index(
        'idx_audit_tenant_timestamp',
        'audit_ledger',
        ['tenant_id', 'timestamp'],
    )
    
    # Entity type + entity ID for entity-specific audit trail
    op.create_index(
        'idx_audit_entity',
        'audit_ledger',
        ['entity_type', 'entity_id'],
    )
    
    # Timestamp descending for recent events
    op.create_index(
        'idx_audit_timestamp_desc',
        'audit_ledger',
        [sa.text('timestamp DESC')],
    )
    
    # ==========================================================================
    # Temporal facts - additional performance indices
    # ==========================================================================
    
    # Confidence for filtering high-confidence facts
    op.create_index(
        'idx_facts_confidence',
        'temporal_facts',
        [sa.text('confidence DESC')],
    )
    
    # Created_at for recent facts
    op.create_index(
        'idx_facts_created_desc',
        'temporal_facts',
        [sa.text('created_at DESC')],
    )


def downgrade() -> None:
    # Drop temporal facts indices
    op.drop_index('idx_facts_created_desc', table_name='temporal_facts')
    op.drop_index('idx_facts_confidence', table_name='temporal_facts')
    
    # Drop audit ledger indices
    op.drop_index('idx_audit_timestamp_desc', table_name='audit_ledger')
    op.drop_index('idx_audit_entity', table_name='audit_ledger')
    op.drop_index('idx_audit_tenant_timestamp', table_name='audit_ledger')
    
    # Drop memories indices
    op.execute("DROP INDEX IF EXISTS idx_memories_content_gin;")
    op.execute("DROP INDEX IF EXISTS idx_memories_tags_gin;")
    op.drop_index('idx_memories_created_desc', table_name='memories')
    op.drop_index('idx_memories_importance_desc', table_name='memories')
    op.drop_index('idx_memories_tenant_user_type', table_name='memories')
    op.drop_index('idx_memories_tenant_status', table_name='memories')
    op.drop_index('idx_memories_tenant_created', table_name='memories')
    op.drop_index('idx_memories_tenant_type', table_name='memories')



