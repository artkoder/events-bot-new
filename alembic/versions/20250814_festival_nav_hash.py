"""add festival nav_hash"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20250814_festival_nav_hash'
down_revision: Union[str, None] = '20250813_ics_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('nav_hash', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('festival', 'nav_hash')
