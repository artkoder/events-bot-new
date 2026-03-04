"""Alembic migration: telegram_source_title

- add `title` column to telegram_source for operator-friendly display
"""

from alembic import op
import sqlalchemy as sa


revision = "20260216_telegram_source_title"
down_revision = "20260215_festival_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.add_column(sa.Column("title", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.drop_column("title")

