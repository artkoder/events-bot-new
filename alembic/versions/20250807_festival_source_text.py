"""add festival source text"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20250807_festival_source_text'
down_revision: Union[str, None] = '20250806_festival_location'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('source_text', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('festival', 'source_text')
