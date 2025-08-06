"""add festival location fields"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '20250806_festival_location'
down_revision: Union[str, None] = '20250806_perf_idx'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('location_name', sa.String(), nullable=True))
    op.add_column('festival', sa.Column('location_address', sa.String(), nullable=True))
    op.add_column('festival', sa.Column('city', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('festival', 'city')
    op.drop_column('festival', 'location_address')
    op.drop_column('festival', 'location_name')
