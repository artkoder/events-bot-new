from alembic import op

revision = "20250809_job_outbox_idx"
down_revision = "20250808_channel_daily_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_job_outbox_status_next_run_at",
        "job_outbox",
        ["status", "next_run_at"],
    )
    op.create_index(
        "ix_job_outbox_event_task",
        "job_outbox",
        ["event_id", "task"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_outbox_status_next_run_at", table_name="job_outbox")
    op.drop_index("ix_job_outbox_event_task", table_name="job_outbox")
