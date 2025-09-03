"""fix city mixups for festdays"""

from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20250901_festday_city_fix'
down_revision: Union[str, None] = '20250815_festival_program_url'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
UPDATE events SET city='Зеленоградск'
WHERE festival='День города Зеленоградск' AND city='Черняховск';

UPDATE events SET city='Черняховск'
WHERE festival='День города Черняховск' AND city='Зеленоградск';

UPDATE events
SET telegraph_url=NULL, telegraph_path=NULL
WHERE festival='День города Зеленоградск'
  AND telegraph_path LIKE 'Den-goroda-CHernyahovsk---den-%';

UPDATE events
SET source_text = festival || ' — ' || date
WHERE (source_text IS NULL OR TRIM(source_text)='')
  AND festival IS NOT NULL AND festival != ''
  AND title LIKE 'День города %';
        """
    )


def downgrade() -> None:
    pass
