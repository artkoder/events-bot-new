"""Alembic migration: post_metrics

- add `telegram_post_metric` for daily views/likes snapshots per (source_id, message_id, age_day)
- add `vk_post_metric` for daily views/likes snapshots per (group_id, post_id, age_day)
"""

from alembic import op
import sqlalchemy as sa


revision = "20260216_post_metrics"
down_revision = "20260216_telegram_source_title"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telegram_post_metric",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "source_id",
            sa.Integer(),
            sa.ForeignKey("telegram_source.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("age_day", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("message_ts", sa.Integer(), nullable=True),
        sa.Column("collected_ts", sa.Integer(), nullable=False),
        sa.Column("views", sa.Integer(), nullable=True),
        sa.Column("likes", sa.Integer(), nullable=True),
        sa.Column("reactions_json", sa.JSON(), nullable=True),
        sa.UniqueConstraint("source_id", "message_id", "age_day", name="ux_tg_metric_source_message_age"),
    )
    op.create_index(
        "ix_tg_metric_source_age",
        "telegram_post_metric",
        ["source_id", "age_day"],
        unique=False,
    )
    op.create_index(
        "ix_tg_metric_source_message",
        "telegram_post_metric",
        ["source_id", "message_id"],
        unique=False,
    )

    op.create_table(
        "vk_post_metric",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("group_id", sa.Integer(), nullable=False),
        sa.Column("post_id", sa.Integer(), nullable=False),
        sa.Column("age_day", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("post_ts", sa.Integer(), nullable=True),
        sa.Column("collected_ts", sa.Integer(), nullable=False),
        sa.Column("views", sa.Integer(), nullable=True),
        sa.Column("likes", sa.Integer(), nullable=True),
        sa.UniqueConstraint("group_id", "post_id", "age_day", name="ux_vk_metric_group_post_age"),
    )
    op.create_index(
        "ix_vk_metric_group_age",
        "vk_post_metric",
        ["group_id", "age_day"],
        unique=False,
    )
    op.create_index(
        "ix_vk_metric_group_post",
        "vk_post_metric",
        ["group_id", "post_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_vk_metric_group_post", table_name="vk_post_metric")
    op.drop_index("ix_vk_metric_group_age", table_name="vk_post_metric")
    op.drop_table("vk_post_metric")

    op.drop_index("ix_tg_metric_source_message", table_name="telegram_post_metric")
    op.drop_index("ix_tg_metric_source_age", table_name="telegram_post_metric")
    op.drop_table("telegram_post_metric")

