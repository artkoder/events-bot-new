"""add festival program_url"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '20250815_festival_program_url'
down_revision: Union[str, None] = '20250814_festival_nav_hash'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('program_url', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('festival', 'program_url')

