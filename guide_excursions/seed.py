from __future__ import annotations

import json
import logging
from typing import Any

import aiosqlite

from .sources import GuideSourceSpec, canonical_guide_sources

logger = logging.getLogger(__name__)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


async def seed_guide_sources(
    conn: aiosqlite.Connection,
    *,
    sources: tuple[GuideSourceSpec, ...] | None = None,
) -> dict[str, int]:
    if sources is None:
        sources = canonical_guide_sources()

    metrics = {
        "profiles_inserted": 0,
        "profiles_updated": 0,
        "sources_inserted": 0,
        "sources_updated": 0,
    }

    for spec in sources:
        cur = await conn.execute(
            "SELECT id, display_name, marketing_name, source_links_json, base_region FROM guide_profile WHERE slug=?",
            (spec.profile_slug,),
        )
        profile_row = await cur.fetchone()
        source_link = f"https://t.me/{spec.username}"
        links_json = _json_dumps([source_link])
        if profile_row:
            await conn.execute(
                """
                UPDATE guide_profile
                SET
                    profile_kind=?,
                    display_name=?,
                    marketing_name=COALESCE(NULLIF(marketing_name, ''), ?),
                    source_links_json=?,
                    base_region=COALESCE(NULLIF(base_region, ''), ?),
                    last_seen_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    spec.profile_kind,
                    spec.display_name,
                    spec.marketing_name or spec.display_name,
                    links_json,
                    spec.base_region,
                    int(profile_row[0]),
                ),
            )
            profile_id = int(profile_row[0])
            metrics["profiles_updated"] += 1
        else:
            cur = await conn.execute(
                """
                INSERT INTO guide_profile(
                    slug,
                    profile_kind,
                    display_name,
                    marketing_name,
                    source_links_json,
                    base_region,
                    first_seen_at,
                    last_seen_at
                )
                VALUES(?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    spec.profile_slug,
                    spec.profile_kind,
                    spec.display_name,
                    spec.marketing_name or spec.display_name,
                    links_json,
                    spec.base_region,
                ),
            )
            profile_id = int(cur.lastrowid or 0)
            metrics["profiles_inserted"] += 1

        cur = await conn.execute(
            "SELECT id FROM guide_source WHERE platform='telegram' AND username=?",
            (spec.username,),
        )
        existing = await cur.fetchone()
        flags_json = _json_dumps(spec.flags or {})
        if existing:
            await conn.execute(
                """
                UPDATE guide_source
                SET
                    primary_profile_id=?,
                    source_kind=?,
                    trust_level=?,
                    priority_weight=?,
                    enabled=1,
                    flags_json=?,
                    base_region=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    int(profile_id),
                    spec.source_kind,
                    spec.trust_level,
                    float(spec.priority_weight or 1.0),
                    flags_json,
                    spec.base_region,
                    int(existing[0]),
                ),
            )
            metrics["sources_updated"] += 1
        else:
            await conn.execute(
                """
                INSERT INTO guide_source(
                    platform,
                    username,
                    primary_profile_id,
                    source_kind,
                    trust_level,
                    priority_weight,
                    enabled,
                    flags_json,
                    base_region,
                    added_via,
                    created_at,
                    updated_at
                )
                VALUES('telegram', ?, ?, ?, ?, ?, 1, ?, ?, 'seed', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    spec.username,
                    int(profile_id),
                    spec.source_kind,
                    spec.trust_level,
                    float(spec.priority_weight or 1.0),
                    flags_json,
                    spec.base_region,
                ),
            )
            metrics["sources_inserted"] += 1

    logger.info("guide_source seed applied: %s", metrics)
    return metrics
