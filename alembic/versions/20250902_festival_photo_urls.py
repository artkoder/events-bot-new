"""add festival photo_urls"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import json

revision: str = '20250902_festival_photo_urls'
down_revision: Union[str, None] = '20250901_festday_city_fix'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('festival', sa.Column('photo_urls', sa.JSON(), nullable=False, server_default='[]'))
    conn = op.get_bind()
    festival = sa.table('festival', sa.column('id', sa.Integer), sa.column('photo_url', sa.String), sa.column('photo_urls', sa.JSON))
    rows = conn.execute(sa.select(festival.c.id, festival.c.photo_url)).fetchall()
    for fid, url in rows:
        if url:
            conn.execute(
                sa.update(festival).where(festival.c.id == fid).values(photo_urls=json.dumps([url]))
            )


def downgrade() -> None:
    op.drop_column('festival', 'photo_urls')
