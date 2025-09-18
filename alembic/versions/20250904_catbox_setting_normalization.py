"""normalize catbox setting values"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20250904_catbox_setting_normalization"
down_revision: Union[str, None] = "20250903_poster_ocr_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE setting
               SET value='1'
             WHERE key='catbox_enabled'
               AND (
                    value IS NULL
                    OR trim(value) = ''
                    OR lower(trim(value)) IN ('true', 't', 'on', 'yes')
               )
            """
        )
    )
    conn.execute(
        sa.text(
            """
            UPDATE setting
               SET value='0'
             WHERE key='catbox_enabled'
               AND lower(trim(value)) IN ('false', 'f', 'off', 'no')
            """
        )
    )


def downgrade() -> None:
    pass
