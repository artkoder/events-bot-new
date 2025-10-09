"""add festival activities_json"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "20250911_festival_activities_json"
down_revision: Union[str, None] = "20250910_vk_misses_unique_constraint_if_not_exists"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "festival",
        sa.Column(
            "activities_json",
            sa.JSON().with_variant(JSONB, "postgresql"),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("festival", "activities_json")
