"""add festival source post fields"""

from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20250902_festival_source_post'
down_revision: Union[str, None] = '20250901_festday_city_fix'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE festival ADD COLUMN source_post_url TEXT")
    op.execute("ALTER TABLE festival ADD COLUMN source_chat_id INTEGER")
    op.execute("ALTER TABLE festival ADD COLUMN source_message_id INTEGER")


def downgrade() -> None:
    pass
