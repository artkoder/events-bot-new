"""poster OCR cache tables"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "20250903_poster_ocr_cache"
down_revision: Union[str, None] = "20250902_festival_photo_urls"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "posterocrcache",
        sa.Column("hash", sa.String(), primary_key=True),
        sa.Column("detail", sa.String(), primary_key=True),
        sa.Column("model", sa.String(), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_table(
        "ocrusage",
        sa.Column("date", sa.String(), primary_key=True),
        sa.Column("spent_tokens", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("ocrusage")
    op.drop_table("posterocrcache")
