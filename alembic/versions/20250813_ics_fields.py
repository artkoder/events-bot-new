"""add ics fields"""

from alembic import op
import sqlalchemy as sa

revision = '20250813_ics_fields'
down_revision = '20250812_job_outbox_coalesce_depends'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('event', sa.Column('ics_hash', sa.String(), nullable=True))
    op.add_column('event', sa.Column('ics_file_id', sa.String(), nullable=True))
    op.add_column('event', sa.Column('ics_updated_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('event', 'ics_updated_at')
    op.drop_column('event', 'ics_file_id')
    op.drop_column('event', 'ics_hash')
