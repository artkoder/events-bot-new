"""Alembic migration: telegram_source_metadata

- add Telegram source metadata fields (about/links/hash/fetched_at)
- add source-level festival suggestions for operator UI
"""

from alembic import op
import sqlalchemy as sa


revision = "20260217_telegram_source_metadata"
down_revision = "20260216_post_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.add_column(sa.Column("about", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("about_links_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("meta_hash", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("meta_fetched_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("suggested_festival_series", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("suggested_website_url", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("suggestion_confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("suggestion_rationale", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("telegram_source", schema=None) as batch_op:
        batch_op.drop_column("suggestion_rationale")
        batch_op.drop_column("suggestion_confidence")
        batch_op.drop_column("suggested_website_url")
        batch_op.drop_column("suggested_festival_series")
        batch_op.drop_column("meta_fetched_at")
        batch_op.drop_column("meta_hash")
        batch_op.drop_column("about_links_json")
        batch_op.drop_column("about")
