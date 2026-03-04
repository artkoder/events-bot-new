from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import aiosqlite

from telegram_sources import (
    TelegramSourceSpec,
    canonical_tg_sources,
    normalize_tg_username,
    trust_priority,
)

logger = logging.getLogger(__name__)


def _json_obj(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        if isinstance(data, dict):
            return dict(data)
        return {}
    return {}


async def seed_telegram_sources(
    conn: aiosqlite.Connection,
    *,
    sources: Iterable[TelegramSourceSpec] | None = None,
) -> dict[str, int]:
    """Idempotently upsert canonical telegram_source rows (best-effort, safe on repeats).

    Policy:
    - Insert missing sources (enabled=1).
    - Upgrade trust_level if desired > existing.
    - Fill default_location only if empty.
    - Fill festival_series only if empty.
    - Merge filters_json by adding missing keys (never removes/overwrites existing truthy flags).
    - Normalize existing usernames to canonical lowercase (merges duplicates if needed).
    """

    if sources is None:
        sources = canonical_tg_sources()

    # Backward-compatible schema evolution (older DB snapshots).
    try:
        await conn.execute("ALTER TABLE telegram_source ADD COLUMN filters_json TEXT")
    except Exception as exc:
        if "duplicate column name" not in str(exc).lower():
            raise

    metrics: dict[str, int] = {
        "normalized": 0,
        "merged_duplicates": 0,
        "inserted": 0,
        "trust_upgraded": 0,
        "default_location_filled": 0,
        "festival_filled": 0,
        "filters_merged": 0,
    }

    # 1) Normalize existing usernames and merge duplicates by normalized key.
    rows = await conn.execute_fetchall("SELECT id, username FROM telegram_source")
    by_key: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        try:
            source_id = int(row[0])
            username = str(row[1] or "")
        except Exception:
            continue
        key = normalize_tg_username(username)
        if not key:
            continue
        by_key.setdefault(key, []).append((source_id, username))

    for key, items in by_key.items():
        items.sort(key=lambda x: x[0])
        keep_id, keep_username = items[0]
        if keep_username != key:
            # Rename keep row to canonical key (safe after duplicate merge below).
            await conn.execute(
                "UPDATE telegram_source SET username=? WHERE id=?",
                (key, keep_id),
            )
            metrics["normalized"] += 1
        if len(items) <= 1:
            continue
        for dup_id, _dup_username in items[1:]:
            # Merge scanned marks (PK=(source_id,message_id)) without collisions.
            await conn.execute(
                """
                INSERT OR IGNORE INTO telegram_scanned_message(
                    source_id, message_id, message_date, processed_at, status, events_extracted, events_imported, error
                )
                SELECT ?, message_id, message_date, processed_at, status, events_extracted, events_imported, error
                FROM telegram_scanned_message
                WHERE source_id=?
                """,
                (keep_id, dup_id),
            )
            await conn.execute(
                "DELETE FROM telegram_scanned_message WHERE source_id=?",
                (dup_id,),
            )
            await conn.execute(
                """
                INSERT OR IGNORE INTO telegram_source_force_message(
                    source_id, message_id, created_at
                )
                SELECT ?, message_id, created_at
                FROM telegram_source_force_message
                WHERE source_id=?
                """,
                (keep_id, dup_id),
            )
            await conn.execute(
                "DELETE FROM telegram_source_force_message WHERE source_id=?",
                (dup_id,),
            )
            await conn.execute("DELETE FROM telegram_source WHERE id=?", (dup_id,))
            metrics["merged_duplicates"] += 1

    before_rows = await conn.execute_fetchall("SELECT username FROM telegram_source")
    before_set = {
        normalize_tg_username(r[0]) for r in before_rows if r and normalize_tg_username(r[0])
    }

    # 2) Insert missing canonical sources.
    for spec in sources:
        username = normalize_tg_username(spec.username)
        if not username:
            continue
        filters_json = (
            json.dumps(spec.filters or {}, ensure_ascii=False) if (spec.filters is not None) else None
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO telegram_source(
                username, enabled, default_location, default_ticket_link, trust_level,
                festival_source, festival_series, filters_json
            )
            VALUES(?, 1, ?, NULL, ?, ?, ?, ?)
            """,
            (
                username,
                spec.default_location,
                (spec.trust_level or "").strip().lower() or None,
                1 if (spec.festival_series or "").strip() else 0,
                (spec.festival_series or "").strip() or None,
                filters_json,
            ),
        )

    # 3) Fetch current state after inserts and apply safe upgrades/fills.
    current = await conn.execute_fetchall(
        """
        SELECT id, username, enabled, trust_level, default_location, festival_series, filters_json
        FROM telegram_source
        """
    )
    current_map: dict[str, dict[str, Any]] = {}
    for row in current:
        try:
            source_id = int(row[0])
            username = normalize_tg_username(row[1])
        except Exception:
            continue
        if not username:
            continue
        current_map[username] = {
            "id": source_id,
            "enabled": int(row[2] or 0),
            "trust_level": (str(row[3] or "").strip().lower() or None),
            "default_location": (str(row[4] or "").strip() or None),
            "festival_series": (str(row[5] or "").strip() or None),
            "filters_json": row[6],
        }

    for spec in sources:
        username = normalize_tg_username(spec.username)
        if not username:
            continue
        row = current_map.get(username)
        if not row:
            # Should not happen after INSERT OR IGNORE, but keep it safe.
            continue
        updates: dict[str, Any] = {}

        desired_trust = (spec.trust_level or "").strip().lower()
        if desired_trust:
            existing_trust = (row.get("trust_level") or "").strip().lower()
            if trust_priority(desired_trust) > trust_priority(existing_trust):
                updates["trust_level"] = desired_trust
                metrics["trust_upgraded"] += 1

        desired_loc = (spec.default_location or "").strip()
        if desired_loc and not (row.get("default_location") or "").strip():
            updates["default_location"] = desired_loc
            metrics["default_location_filled"] += 1

        desired_series = (spec.festival_series or "").strip()
        if desired_series and not (row.get("festival_series") or "").strip():
            updates["festival_series"] = desired_series
            updates["festival_source"] = 1
            metrics["festival_filled"] += 1

        desired_filters = spec.filters or None
        if desired_filters is not None:
            existing_filters = _json_obj(row.get("filters_json"))
            merged = dict(existing_filters)
            changed = False
            for k, v in dict(desired_filters).items():
                # Never overwrite an explicit false/other value; only add missing keys
                # or upgrade falsy -> truthy.
                if k not in merged:
                    merged[k] = v
                    changed = True
                elif not merged.get(k) and v:
                    merged[k] = v
                    changed = True
            if changed:
                updates["filters_json"] = json.dumps(merged, ensure_ascii=False)
                metrics["filters_merged"] += 1

        if not updates:
            continue
        set_sql = ", ".join([f"{k}=?" for k in updates.keys()])
        params = list(updates.values()) + [username]
        await conn.execute(
            f"UPDATE telegram_source SET {set_sql} WHERE username=?",
            params,
        )

    after_rows = await conn.execute_fetchall("SELECT username FROM telegram_source")
    after_set = {
        normalize_tg_username(r[0]) for r in after_rows if r and normalize_tg_username(r[0])
    }
    want = {normalize_tg_username(s.username) for s in sources if normalize_tg_username(s.username)}
    metrics["inserted"] = len((after_set & want) - (before_set & want))

    logger.info("telegram_source seed applied: %s", metrics)
    return metrics
