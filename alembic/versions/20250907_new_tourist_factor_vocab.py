import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20250907_new_tourist_factor_vocab"
down_revision: Union[str, None] = "20250906_festival_aliases"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


NEW_CODES: list[str] = [
    "targeted_for_tourists",
    "unique_to_region",
    "iconic_location",
    "shows_local_life",
    "scenic_nature",
    "local_cuisine",
    "family_friendly",
]
NEW_CODE_SET = set(NEW_CODES)
ALIAS_TO_NEW: dict[str, str] = {
    "history": "unique_to_region",
    "culture": "unique_to_region",
    "atmosphere": "shows_local_life",
    "city": "shows_local_life",
    "sea": "scenic_nature",
    "water": "scenic_nature",
    "nature": "scenic_nature",
    "food": "local_cuisine",
    "gastronomy": "local_cuisine",
    "family": "family_friendly",
    "events": "targeted_for_tourists",
    "event": "targeted_for_tourists",
}

DOWN_CODES: list[str] = [
    "culture",
    "atmosphere",
    "nature",
    "water",
    "food",
    "family",
    "events",
]
DOWN_CODE_SET = set(DOWN_CODES)
NEW_TO_OLD: dict[str, str] = {
    "targeted_for_tourists": "events",
    "unique_to_region": "culture",
    "iconic_location": "culture",
    "shows_local_life": "atmosphere",
    "scenic_nature": "nature",
    "local_cuisine": "food",
    "family_friendly": "family",
}


def _normalize(values: Sequence[str], *, forward: bool) -> list[str]:
    if forward:
        mapping = ALIAS_TO_NEW
        allowed = NEW_CODES
        allowed_set = NEW_CODE_SET
    else:
        mapping = NEW_TO_OLD
        allowed = DOWN_CODES
        allowed_set = DOWN_CODE_SET
    seen: set[str] = set()
    for code in values:
        mapped = mapping.get(code, code)
        if mapped in allowed_set:
            seen.add(mapped)
    return [code for code in allowed if code in seen]


def _load_factors(payload: object) -> list[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except ValueError:
            return []
    else:
        data = payload
    if isinstance(data, list):
        return [item for item in data if isinstance(item, str)]
    return []


def upgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, tourist_factors FROM event WHERE tourist_factors IS NOT NULL")
    ).fetchall()
    for row in rows:
        factors = _load_factors(row.tourist_factors)
        normalized = _normalize(factors, forward=True)
        if normalized == factors:
            continue
        conn.execute(
            sa.text("UPDATE event SET tourist_factors = :factors WHERE id = :id"),
            {"factors": json.dumps(normalized), "id": row.id},
        )


def downgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, tourist_factors FROM event WHERE tourist_factors IS NOT NULL")
    ).fetchall()
    for row in rows:
        factors = _load_factors(row.tourist_factors)
        normalized = _normalize(factors, forward=False)
        if normalized == factors:
            continue
        conn.execute(
            sa.text("UPDATE event SET tourist_factors = :factors WHERE id = :id"),
            {"factors": json.dumps(normalized), "id": row.id},
        )
