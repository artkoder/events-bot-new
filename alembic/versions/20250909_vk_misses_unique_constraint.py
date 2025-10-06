"""add unique constraint to vk_misses_sample"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "20250909_vk_misses_unique_constraint"
down_revision: Union[str, None] = "20250908_festival_created_at"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE vk_misses_sample "
        "ADD CONSTRAINT IF NOT EXISTS uniq_vk_miss "
        "UNIQUE (group_id, post_id)"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE vk_misses_sample "
        "DROP CONSTRAINT IF EXISTS uniq_vk_miss"
    )
