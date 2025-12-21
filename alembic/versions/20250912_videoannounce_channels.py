"""video announce channel choices"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20250912_videoannounce_channels"
down_revision: Union[str, None] = "20250911_festival_activities_json"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "videoannounce_session",
        sa.Column("profile_key", sa.String(), nullable=True),
    )
    op.add_column(
        "videoannounce_session",
        sa.Column("test_chat_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "videoannounce_session",
        sa.Column("main_chat_id", sa.BigInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("videoannounce_session", "main_chat_id")
    op.drop_column("videoannounce_session", "test_chat_id")
    op.drop_column("videoannounce_session", "profile_key")
