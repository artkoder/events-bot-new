"""Add preview_3d_url to Event model.

Revision ID: 20260101_preview_3d_url
"""
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("preview_3d_url", sa.String(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.drop_column("preview_3d_url")
