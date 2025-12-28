"""Add ticket_status and linked_event_ids to Event model.

Revision ID: 20251228_ticket_status
"""
from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    # Add ticket_status column for tracking ticket availability
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("ticket_status", sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("linked_event_ids", sa.JSON(), nullable=True, server_default="[]")
        )


def downgrade() -> None:
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.drop_column("ticket_status")
        batch_op.drop_column("linked_event_ids")
