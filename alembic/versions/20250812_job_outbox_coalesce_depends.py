from alembic import op
import sqlalchemy as sa

revision = "20250812_job_outbox_coalesce_depends"
down_revision = "20250811_festival_ticket_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("joboutbox", sa.Column("coalesce_key", sa.Text(), nullable=True))
    op.add_column("joboutbox", sa.Column("depends_on", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("joboutbox", "coalesce_key")
    op.drop_column("joboutbox", "depends_on")
