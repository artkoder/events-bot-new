"""partial index for channel daily_time"""

from alembic import op

revision = "20250808_channel_daily_idx"
down_revision = "20250807_festival_source_text"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_channel_daily_time_notnull "
        "ON channel(daily_time) WHERE daily_time IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_channel_daily_time_notnull")
