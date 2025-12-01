"""Add temporal knowledge graph tables.

Revision ID: 002_temporal_facts
Revises: 001_initial_migration
Create Date: 2025-11-27

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_temporal_facts'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create temporal_facts table
    op.create_table(
        'temporal_facts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(500), nullable=False),
        sa.Column('predicate', sa.String(255), nullable=False),
        sa.Column('object', sa.Text(), nullable=False),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False),
        sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, default=1.0),
        sa.Column('initial_confidence', sa.Float(), nullable=False, default=1.0),
        sa.Column('source_memory_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('source_type', sa.String(50), nullable=False, default='extracted'),
        sa.Column('extra_data', postgresql.JSONB(), nullable=True),
        sa.Column('change_count', sa.Integer(), nullable=False, default=0),
        sa.Column('superseded_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['source_memory_id'], ['memories.memory_id'], ondelete='SET NULL'),
    )
    
    # Create indices for temporal_facts
    op.create_index('idx_temporal_facts_id', 'temporal_facts', ['id'])
    op.create_index('idx_temporal_facts_tenant_id', 'temporal_facts', ['tenant_id'])
    op.create_index('idx_temporal_facts_subject', 'temporal_facts', ['subject'])
    op.create_index('idx_temporal_facts_predicate', 'temporal_facts', ['predicate'])
    op.create_index('idx_temporal_facts_valid_from', 'temporal_facts', ['valid_from'])
    op.create_index('idx_temporal_facts_valid_to', 'temporal_facts', ['valid_to'])
    op.create_index('idx_temporal_facts_source_memory_id', 'temporal_facts', ['source_memory_id'])
    
    # Composite indices for common query patterns
    op.create_index('idx_facts_subject_predicate', 'temporal_facts', ['subject', 'predicate'])
    op.create_index('idx_facts_tenant_subject', 'temporal_facts', ['tenant_id', 'subject'])
    op.create_index('idx_facts_time_range', 'temporal_facts', ['valid_from', 'valid_to'])
    op.create_index('idx_facts_subject_time', 'temporal_facts', ['subject', 'valid_from', 'valid_to'])
    op.create_index('idx_facts_active', 'temporal_facts', ['tenant_id', 'valid_to'])
    
    # Create fact_changes table
    op.create_table(
        'fact_changes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(500), nullable=False),
        sa.Column('predicate', sa.String(255), nullable=False),
        sa.Column('old_fact_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('new_fact_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('old_value', sa.Text(), nullable=False),
        sa.Column('new_value', sa.Text(), nullable=False),
        sa.Column('changed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('change_reason', sa.String(500), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['old_fact_id'], ['temporal_facts.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['new_fact_id'], ['temporal_facts.id'], ondelete='CASCADE'),
    )
    
    # Create indices for fact_changes
    op.create_index('idx_fact_changes_tenant_id', 'fact_changes', ['tenant_id'])
    op.create_index('idx_fact_changes_subject', 'fact_changes', ['subject'])
    op.create_index('idx_fact_changes_changed_at', 'fact_changes', ['changed_at'])
    op.create_index('idx_changes_subject_predicate', 'fact_changes', ['subject', 'predicate'])
    op.create_index('idx_changes_tenant_time', 'fact_changes', ['tenant_id', 'changed_at'])


def downgrade() -> None:
    # Drop fact_changes table and indices
    op.drop_index('idx_changes_tenant_time', table_name='fact_changes')
    op.drop_index('idx_changes_subject_predicate', table_name='fact_changes')
    op.drop_index('idx_fact_changes_changed_at', table_name='fact_changes')
    op.drop_index('idx_fact_changes_subject', table_name='fact_changes')
    op.drop_index('idx_fact_changes_tenant_id', table_name='fact_changes')
    op.drop_table('fact_changes')
    
    # Drop temporal_facts table and indices
    op.drop_index('idx_facts_active', table_name='temporal_facts')
    op.drop_index('idx_facts_subject_time', table_name='temporal_facts')
    op.drop_index('idx_facts_time_range', table_name='temporal_facts')
    op.drop_index('idx_facts_tenant_subject', table_name='temporal_facts')
    op.drop_index('idx_facts_subject_predicate', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_source_memory_id', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_valid_to', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_valid_from', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_predicate', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_subject', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_tenant_id', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_id', table_name='temporal_facts')
    op.drop_table('temporal_facts')



