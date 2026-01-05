"""Alembic migration: festival_parser_fields

Add fields to Festival model for Universal Festival Parser:
- source_url: Original URL of the festival site
- source_type: "canonical" | "official" | "external"
- parser_run_id: Last parser run ID
- parser_version: Parser version used
- last_parsed_at: Timestamp of last parse
- uds_storage_path: Path in Supabase Storage
- contacts_phone: Phone contact
- contacts_email: Email contact
- is_annual: Is this an annual festival?
- audience: Target audience description
"""
from alembic import op
import sqlalchemy as sa


revision = "20260104_festival_parser_fields"
down_revision = "20260101_preview_3d_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("festival", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_url", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source_type", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("parser_run_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("parser_version", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column("last_parsed_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(sa.Column("uds_storage_path", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("contacts_phone", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("contacts_email", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("is_annual", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("audience", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("festival", schema=None) as batch_op:
        batch_op.drop_column("audience")
        batch_op.drop_column("is_annual")
        batch_op.drop_column("contacts_email")
        batch_op.drop_column("contacts_phone")
        batch_op.drop_column("uds_storage_path")
        batch_op.drop_column("last_parsed_at")
        batch_op.drop_column("parser_version")
        batch_op.drop_column("parser_run_id")
        batch_op.drop_column("source_type")
        batch_op.drop_column("source_url")
