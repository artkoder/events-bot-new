"""add aliases column to festival"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "20250906_festival_aliases"
down_revision: Union[str, None] = "20250905_tourist_factor_codes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE festival ADD COLUMN aliases JSON DEFAULT '[]'")


def downgrade() -> None:
    op.execute("ALTER TABLE festival DROP COLUMN aliases")
