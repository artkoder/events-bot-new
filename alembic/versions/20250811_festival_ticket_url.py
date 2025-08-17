"""add festival ticket_url"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '20250811_festival_ticket_url'
down_revision: Union[str, None] = '20250810_job_outbox_last_result'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('ticket_url', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('festival', 'ticket_url')
