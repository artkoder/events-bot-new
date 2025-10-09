"""add festival activities_json"""

from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "20250911_festival_activities_json"
down_revision: Union[str, None] = "20250910_vk_misses_unique_constraint_if_not_exists"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE festival "
        "ADD COLUMN activities_json JSONB NOT NULL DEFAULT '[]'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE festival DROP COLUMN activities_json")
