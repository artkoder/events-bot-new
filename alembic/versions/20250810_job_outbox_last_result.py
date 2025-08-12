from alembic import op
import sqlalchemy as sa

revision = "20250810_job_outbox_last_result"
down_revision = "20250809_job_outbox_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("joboutbox", sa.Column("last_result", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("joboutbox", "last_result")
