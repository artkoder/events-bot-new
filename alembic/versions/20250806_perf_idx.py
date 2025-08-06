"""performance indexes for event table"""

from alembic import op
from sqlalchemy import text

revision = "20250806_perf_idx"
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_index("idx_event_date", "event", ["date"])
    op.create_index("idx_event_added_at", "event", ["added_at"])
    op.create_index("idx_event_date_city", "event", ["date", "city"])
    op.create_index(
        "idx_event_festival_date",
        "event",
        ["festival", "date"],
        postgresql_where=text("festival IS NOT NULL"),
    )

def downgrade() -> None:
    op.drop_index("idx_event_date", table_name="event")
    op.drop_index("idx_event_added_at", table_name="event")
    op.drop_index("idx_event_date_city", table_name="event")
    op.drop_index("idx_event_festival_date", table_name="event")
