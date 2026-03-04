"""Alembic migration: festival_queue

- add unified festival queue table for festival-post sources
"""

from alembic import op
import sqlalchemy as sa


revision = "20260215_festival_queue"
down_revision = "20260215_festival_source_channels"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "festival_queue",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("source_kind", sa.String(), nullable=False),
        sa.Column("source_url", sa.String(), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("source_chat_username", sa.String(), nullable=True),
        sa.Column("source_chat_id", sa.Integer(), nullable=True),
        sa.Column("source_message_id", sa.Integer(), nullable=True),
        sa.Column("source_group_id", sa.Integer(), nullable=True),
        sa.Column("source_post_id", sa.Integer(), nullable=True),
        sa.Column("festival_context", sa.String(), nullable=True),
        sa.Column("festival_name", sa.String(), nullable=True),
        sa.Column("festival_full", sa.String(), nullable=True),
        sa.Column("festival_series", sa.String(), nullable=True),
        sa.Column("dedup_links_json", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
        sa.Column("signals_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("result_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index(
        "ix_festival_queue_status_next_run",
        "festival_queue",
        ["status", "next_run_at"],
        unique=False,
    )
    op.create_index(
        "ix_festival_queue_source_kind",
        "festival_queue",
        ["source_kind"],
        unique=False,
    )
    op.create_index(
        "ix_festival_queue_source_url",
        "festival_queue",
        ["source_url"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_festival_queue_source_url", table_name="festival_queue")
    op.drop_index("ix_festival_queue_source_kind", table_name="festival_queue")
    op.drop_index("ix_festival_queue_status_next_run", table_name="festival_queue")
    op.drop_table("festival_queue")

