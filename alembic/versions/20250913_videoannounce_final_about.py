"""add final_about to videoannounce_item"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20250913_videoannounce_final_about"
down_revision: Union[str, None] = "20250912_videoannounce_channels"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("videoannounce_item", sa.Column("final_about", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("videoannounce_item", "final_about")
