"""add created_at column to festival"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20250908_festival_created_at"
down_revision: Union[str, None] = "20250907_new_tourist_factor_vocab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "festival",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.execute("UPDATE festival SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
    op.alter_column("festival", "created_at", nullable=False)


def downgrade() -> None:
    op.drop_column("festival", "created_at")
