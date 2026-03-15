from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import aiosqlite
from aiogram import Bot, types
from aiogram.types import BufferedInputFile

from admin_chat import resolve_superadmin_chat_id
from db import Database
from heavy_ops import heavy_operation
from ops_run import finish_ops_run, start_ops_run

from .dedup import deduplicate_occurrence_rows
from .digest import MAX_MEDIA_ITEMS, build_digest_messages, build_media_caption, format_date_time
from .editorial import refine_digest_rows
from .parser import GuideParsedOccurrence, collapse_ws, parse_post_occurrences
from .scanner import GuideScannedPost, scan_source_posts
from .seed import seed_guide_sources
from .telethon_client import create_telethon_runtime_client

logger = logging.getLogger(__name__)

GUIDE_DIGEST_TARGET_CHAT = (os.getenv("GUIDE_DIGEST_TARGET_CHAT") or "@keniggpt").strip() or "@keniggpt"
GUIDE_SCAN_LIMIT_FULL = max(10, min(int((os.getenv("GUIDE_SCAN_LIMIT_FULL") or "60") or 60), 200))
GUIDE_SCAN_LIMIT_LIGHT = max(5, min(int((os.getenv("GUIDE_SCAN_LIMIT_LIGHT") or "25") or 25), 120))
GUIDE_DAYS_BACK_FULL = max(3, min(int((os.getenv("GUIDE_DAYS_BACK_FULL") or "21") or 21), 90))
GUIDE_DAYS_BACK_LIGHT = max(2, min(int((os.getenv("GUIDE_DAYS_BACK_LIGHT") or "7") or 7), 30))
GUIDE_DIGEST_WINDOW_DAYS = max(3, min(int((os.getenv("GUIDE_DIGEST_WINDOW_DAYS") or "30") or 30), 90))
GUIDE_MEDIA_TELETHON_FALLBACK_MAX_MB = max(
    1,
    min(int((os.getenv("GUIDE_MEDIA_TELETHON_FALLBACK_MAX_MB") or "20") or 20), 100),
)

_RUN_LOCK = asyncio.Lock()


@dataclass(slots=True)
class GuideMonitorResult:
    run_id: str
    ops_run_id: int | None
    trigger: str
    mode: str
    metrics: dict[str, int]
    errors: list[str]
    latest_preview_issue_id: int | None = None


def _json_load(value: Any, *, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    raw = str(value).strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today_iso() -> str:
    return _utc_now().date().isoformat()


def _future_cutoff_iso(days: int = GUIDE_DIGEST_WINDOW_DAYS) -> str:
    return (_utc_now().date() + timedelta(days=int(days))).isoformat()


def _content_hash(post: GuideScannedPost) -> str:
    payload = {
        "source": post.source_username,
        "message_id": post.message_id,
        "grouped_id": post.grouped_id,
        "text": post.text,
        "media_refs": post.media_refs,
    }
    return hashlib.sha256(_json_dump(payload).encode("utf-8")).hexdigest()


def _parse_json_array(value: Any) -> list[Any]:
    data = _json_load(value, fallback=[])
    return data if isinstance(data, list) else []


def _median(values: Sequence[int]) -> int | None:
    items = sorted(int(v) for v in values if isinstance(v, int))
    if not items:
        return None
    mid = len(items) // 2
    if len(items) % 2 == 1:
        return items[mid]
    return (items[mid - 1] + items[mid]) // 2


def _popularity_mark(*, views: int | None, likes: int | None, median_views: int | None, median_likes: int | None) -> str:
    if isinstance(likes, int) and isinstance(median_likes, int) and likes > median_likes:
        return "❤️"
    if isinstance(views, int) and isinstance(median_views, int) and views > median_views:
        return "⭐"
    return ""


def _safe_time_sort(value: str | None) -> str:
    raw = collapse_ws(value)
    if not raw:
        return "99:99"
    return raw


def _light_or_full(mode: str) -> tuple[int, int]:
    mode_key = (mode or "full").strip().lower()
    if mode_key == "light":
        return GUIDE_SCAN_LIMIT_LIGHT, GUIDE_DAYS_BACK_LIGHT
    return GUIDE_SCAN_LIMIT_FULL, GUIDE_DAYS_BACK_FULL


def _booking_url_for_digest(value: str | None) -> str | None:
    raw = collapse_ws(value)
    if not raw:
        return None
    if raw.startswith("tel:"):
        return raw
    return raw


async def _enable_row_factory(conn: aiosqlite.Connection) -> None:
    conn.row_factory = aiosqlite.Row


async def _get_enabled_sources(conn: aiosqlite.Connection) -> list[aiosqlite.Row]:
    cur = await conn.execute(
        """
        SELECT
            gs.id,
            gs.username,
            gs.title,
            gs.primary_profile_id,
            gs.source_kind,
            gs.trust_level,
            gs.priority_weight,
            gs.flags_json,
            gs.base_region,
            gp.display_name,
            gp.marketing_name
        FROM guide_source gs
        LEFT JOIN guide_profile gp ON gp.id = gs.primary_profile_id
        WHERE gs.platform='telegram' AND COALESCE(gs.enabled, 1) = 1
        ORDER BY gs.priority_weight DESC, gs.username ASC
        """
    )
    return list(await cur.fetchall())


async def _update_source_runtime_meta(
    conn: aiosqlite.Connection,
    *,
    source_id: int,
    title: str | None,
    about_text: str | None,
    about_links: Sequence[str] | None,
    last_scanned_message_id: int | None,
) -> None:
    await conn.execute(
        """
        UPDATE guide_source
        SET
            title=COALESCE(NULLIF(?, ''), title),
            about_text=COALESCE(NULLIF(?, ''), about_text),
            about_links_json=CASE
                WHEN ? IS NULL OR ? = '' THEN about_links_json
                ELSE ?
            END,
            last_scanned_message_id=CASE
                WHEN ? IS NULL THEN last_scanned_message_id
                WHEN last_scanned_message_id IS NULL THEN ?
                WHEN ? > last_scanned_message_id THEN ?
                ELSE last_scanned_message_id
            END,
            last_scan_at=CURRENT_TIMESTAMP,
            updated_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
            collapse_ws(title),
            collapse_ws(about_text),
            _json_dump(list(about_links or [])) if about_links else None,
            _json_dump(list(about_links or [])) if about_links else None,
            _json_dump(list(about_links or [])) if about_links else None,
            int(last_scanned_message_id) if last_scanned_message_id is not None else None,
            int(last_scanned_message_id) if last_scanned_message_id is not None else None,
            int(last_scanned_message_id) if last_scanned_message_id is not None else None,
            int(last_scanned_message_id) if last_scanned_message_id is not None else None,
            int(source_id),
        ),
    )


async def _upsert_monitor_post(
    conn: aiosqlite.Connection,
    *,
    source_id: int,
    post: GuideScannedPost,
    post_kind: str,
    prefilter_passed: bool,
) -> int:
    content_hash = _content_hash(post)
    cur = await conn.execute(
        "SELECT id FROM guide_monitor_post WHERE source_id=? AND message_id=?",
        (int(source_id), int(post.message_id)),
    )
    row = await cur.fetchone()
    payloads = (
        int(source_id),
        int(post.message_id),
        int(post.grouped_id) if post.grouped_id is not None else None,
        post.post_date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        post.source_url,
        post.text,
        int(post.views) if post.views is not None else None,
        int(post.forwards) if post.forwards is not None else None,
        int(post.reactions_total) if post.reactions_total is not None else None,
        _json_dump(post.reactions_json or {}),
        content_hash,
        _json_dump(post.media_refs),
        post_kind,
        1 if prefilter_passed else 0,
    )
    if row:
        await conn.execute(
            """
            UPDATE guide_monitor_post
            SET
                grouped_id=?,
                post_date=?,
                source_url=?,
                text=?,
                views=?,
                forwards=?,
                reactions_total=?,
                reactions_json=?,
                content_hash=?,
                media_refs_json=?,
                post_kind=?,
                prefilter_passed=?,
                last_scanned_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            payloads[2:] + (int(row["id"]),),
        )
        return int(row["id"])

    cur = await conn.execute(
        """
        INSERT INTO guide_monitor_post(
            source_id,
            message_id,
            grouped_id,
            post_date,
            source_url,
            text,
            views,
            forwards,
            reactions_total,
            reactions_json,
            content_hash,
            media_refs_json,
            post_kind,
            prefilter_passed,
            llm_status,
            last_scanned_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'heuristic_only', CURRENT_TIMESTAMP)
        """,
        payloads,
    )
    return int(cur.lastrowid or 0)


async def _ensure_template(
    conn: aiosqlite.Connection,
    *,
    profile_id: int | None,
    parsed: GuideParsedOccurrence,
) -> int | None:
    if profile_id is None:
        return None
    cur = await conn.execute(
        "SELECT id FROM guide_template WHERE profile_id=? AND title_normalized=?",
        (int(profile_id), parsed.title_normalized),
    )
    row = await cur.fetchone()
    aliases_json = _json_dump([parsed.canonical_title])
    audience_json = _json_dump(parsed.audience_fit)
    guide_names_json = _json_dump(parsed.guide_names)
    if row:
        await conn.execute(
            """
            UPDATE guide_template
            SET
                canonical_title=COALESCE(NULLIF(?, ''), canonical_title),
                aliases_json=?,
                availability_mode=COALESCE(NULLIF(?, ''), availability_mode),
                audience_fit_json=?,
                participant_profiles_json=?,
                summary_short=COALESCE(NULLIF(?, ''), summary_short),
                last_seen_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                parsed.canonical_title,
                aliases_json,
                parsed.availability_mode,
                audience_json,
                guide_names_json,
                parsed.summary_one_liner,
                int(row["id"]),
            ),
        )
        return int(row["id"])
    cur = await conn.execute(
        """
        INSERT INTO guide_template(
            profile_id,
            canonical_title,
            title_normalized,
            aliases_json,
            base_city,
            availability_mode,
            audience_fit_json,
            participant_profiles_json,
            summary_short,
            first_seen_at,
            last_seen_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            int(profile_id),
            parsed.canonical_title,
            parsed.title_normalized,
            aliases_json,
            parsed.city,
            parsed.availability_mode,
            audience_json,
            guide_names_json,
            parsed.summary_one_liner,
        ),
    )
    return int(cur.lastrowid or 0)


async def _sync_occurrence_aggregator_flag(conn: aiosqlite.Connection, occurrence_id: int) -> None:
    cur = await conn.execute(
        """
        SELECT COALESCE(MAX(CASE WHEN gs.source_kind != 'aggregator' THEN 1 ELSE 0 END), 0) AS has_non_agg
        FROM guide_occurrence_source gos
        JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
        JOIN guide_source gs ON gs.id = gmp.source_id
        WHERE gos.occurrence_id=?
        """,
        (int(occurrence_id),),
    )
    row = await cur.fetchone()
    has_non_agg = int((row["has_non_agg"] if row else 0) or 0)
    await conn.execute(
        "UPDATE guide_occurrence SET aggregator_only=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (0 if has_non_agg else 1, int(occurrence_id)),
    )


async def _insert_occurrence_claims(
    conn: aiosqlite.Connection,
    *,
    occurrence_id: int,
    post_id: int,
    parsed: GuideParsedOccurrence,
) -> None:
    await conn.execute("DELETE FROM guide_fact_claim WHERE entity_kind='occurrence' AND entity_id=? AND source_post_id=?", (int(occurrence_id), int(post_id)))
    claims: list[tuple[str, str | None]] = [
        ("date", parsed.date_iso),
        ("time", parsed.time_text),
        ("meeting_point", parsed.meeting_point),
        ("price_text", parsed.price_text),
        ("booking_text", parsed.booking_text),
        ("booking_url", parsed.booking_url),
        ("status", parsed.status),
        ("seats_text", parsed.seats_text),
        ("summary_one_liner", parsed.summary_one_liner),
    ]
    for audience in parsed.audience_fit[:5]:
        claims.append(("audience_fit", audience))
    for fact_key, fact_value in claims:
        if not collapse_ws(fact_value):
            continue
        await conn.execute(
            """
            INSERT INTO guide_fact_claim(entity_kind, entity_id, fact_key, fact_value, confidence, source_post_id)
            VALUES('occurrence', ?, ?, ?, ?, ?)
            """,
            (int(occurrence_id), fact_key, collapse_ws(fact_value), 0.75, int(post_id)),
        )


async def _upsert_occurrence(
    conn: aiosqlite.Connection,
    *,
    source_row: Mapping[str, Any],
    post_id: int,
    post: GuideScannedPost,
    parsed: GuideParsedOccurrence,
    template_id: int | None,
) -> tuple[int, bool]:
    cur = await conn.execute(
        """
        SELECT
            go.id,
            go.primary_source_id,
            ps.source_kind AS primary_source_kind
        FROM guide_occurrence go
        LEFT JOIN guide_source ps ON ps.id = go.primary_source_id
        WHERE go.source_fingerprint=?
        """,
        (parsed.source_fingerprint,),
    )
    row = await cur.fetchone()
    source_id = int(source_row["id"])
    source_kind = str(source_row["source_kind"] or "")
    guide_names_json = _json_dump(parsed.guide_names)
    organizer_names_json = _json_dump(parsed.organizer_names)
    audience_json = _json_dump(parsed.audience_fit)
    created = False
    if row:
        occurrence_id = int(row["id"])
        existing_primary_kind = str(row["primary_source_kind"] or "")
        should_promote_primary = existing_primary_kind == "aggregator" and source_kind != "aggregator"
        await conn.execute(
            """
            UPDATE guide_occurrence
            SET
                template_id=COALESCE(?, template_id),
                canonical_title=COALESCE(NULLIF(?, ''), canonical_title),
                title_normalized=?,
                participant_profiles_json=?,
                guide_names_json=CASE
                    WHEN ? != '' THEN ?
                    ELSE guide_names_json
                END,
                organizer_names_json=CASE
                    WHEN ? != '' THEN ?
                    ELSE organizer_names_json
                END,
                digest_eligible=?,
                digest_eligibility_reason=?,
                is_last_call=?,
                date=COALESCE(?, date),
                time=COALESCE(?, time),
                city=COALESCE(NULLIF(?, ''), city),
                meeting_point=COALESCE(NULLIF(?, ''), meeting_point),
                audience_fit_json=?,
                price_text=COALESCE(NULLIF(?, ''), price_text),
                booking_text=COALESCE(NULLIF(?, ''), booking_text),
                booking_url=COALESCE(NULLIF(?, ''), booking_url),
                channel_url=COALESCE(NULLIF(?, ''), channel_url),
                status=CASE
                    WHEN ? != '' THEN ?
                    ELSE status
                END,
                seats_text=COALESCE(NULLIF(?, ''), seats_text),
                summary_one_liner=COALESCE(NULLIF(?, ''), summary_one_liner),
                digest_blurb=COALESCE(NULLIF(?, ''), digest_blurb),
                views=CASE
                    WHEN views IS NULL OR (? IS NOT NULL AND ? >= views) THEN ?
                    ELSE views
                END,
                likes=CASE
                    WHEN likes IS NULL OR (? IS NOT NULL AND ? >= likes) THEN ?
                    ELSE likes
                END,
                primary_source_id=CASE WHEN ? THEN ? ELSE primary_source_id END,
                primary_message_id=CASE WHEN ? THEN ? ELSE primary_message_id END,
                last_seen_post_at=?,
                updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                int(template_id) if template_id is not None else None,
                parsed.canonical_title,
                parsed.title_normalized,
                guide_names_json,
                collapse_ws(",".join(parsed.guide_names)),
                guide_names_json,
                collapse_ws(",".join(parsed.organizer_names)),
                organizer_names_json,
                1 if parsed.digest_eligible else 0,
                parsed.digest_eligibility_reason,
                1 if parsed.is_last_call else 0,
                parsed.date_iso,
                parsed.time_text,
                parsed.city,
                parsed.meeting_point,
                audience_json,
                parsed.price_text,
                parsed.booking_text,
                _booking_url_for_digest(parsed.booking_url),
                parsed.channel_url,
                parsed.status,
                parsed.status,
                parsed.seats_text,
                parsed.summary_one_liner,
                parsed.digest_blurb,
                int(post.views) if post.views is not None else None,
                int(post.views) if post.views is not None else None,
                int(post.views) if post.views is not None else None,
                int(post.reactions_total) if post.reactions_total is not None else None,
                int(post.reactions_total) if post.reactions_total is not None else None,
                int(post.reactions_total) if post.reactions_total is not None else None,
                1 if should_promote_primary else 0,
                source_id,
                1 if should_promote_primary else 0,
                int(post.message_id),
                post.post_date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                occurrence_id,
            ),
        )
    else:
        cur = await conn.execute(
            """
            INSERT INTO guide_occurrence(
                template_id,
                primary_source_id,
                primary_message_id,
                source_fingerprint,
                canonical_title,
                title_normalized,
                participant_profiles_json,
                guide_names_json,
                organizer_names_json,
                digest_eligible,
                digest_eligibility_reason,
                is_last_call,
                aggregator_only,
                date,
                time,
                city,
                meeting_point,
                audience_fit_json,
                price_text,
                booking_text,
                booking_url,
                channel_url,
                status,
                seats_text,
                summary_one_liner,
                digest_blurb,
                views,
                likes,
                first_seen_at,
                updated_at,
                last_seen_post_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
            (
                int(template_id) if template_id is not None else None,
                source_id,
                int(post.message_id),
                parsed.source_fingerprint,
                parsed.canonical_title,
                parsed.title_normalized,
                guide_names_json,
                guide_names_json,
                organizer_names_json,
                1 if parsed.digest_eligible else 0,
                parsed.digest_eligibility_reason,
                1 if parsed.is_last_call else 0,
                1 if source_kind == "aggregator" else 0,
                parsed.date_iso,
                parsed.time_text,
                parsed.city,
                parsed.meeting_point,
                audience_json,
                parsed.price_text,
                parsed.booking_text,
                _booking_url_for_digest(parsed.booking_url),
                parsed.channel_url,
                parsed.status,
                parsed.seats_text,
                parsed.summary_one_liner,
                parsed.digest_blurb,
                int(post.views) if post.views is not None else None,
                int(post.reactions_total) if post.reactions_total is not None else None,
                post.post_date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        occurrence_id = int(cur.lastrowid or 0)
        created = True

    await conn.execute(
        """
        INSERT OR IGNORE INTO guide_occurrence_source(occurrence_id, post_id, role)
        VALUES(?, ?, ?)
        """,
        (
            int(occurrence_id),
            int(post_id),
            "aggregator" if source_kind == "aggregator" else "primary",
        ),
    )
    await _sync_occurrence_aggregator_flag(conn, occurrence_id)
    await _insert_occurrence_claims(conn, occurrence_id=occurrence_id, post_id=post_id, parsed=parsed)
    return occurrence_id, created


def _should_prefilter(post: GuideScannedPost, source_kind: str) -> bool:
    text = collapse_ws(post.text).lower()
    if not text:
        return False
    positive = (
        "экскурс",
        "прогул",
        "маршрут",
        "путешеств",
        "тур ",
        "авторская экскурсия",
        "место встречи",
        "запись",
        "записаться",
        "выезд",
        "пешеходная",
        "квест-экскурсия",
    )
    if not any(token in text for token in positive):
        return False
    if source_kind == "aggregator" and not any(token in text for token in ("авторская экскурсия", "пешеходная", "приглашаем")):
        return False
    return True


async def _scan_and_import_source(
    conn: aiosqlite.Connection,
    client: Any,
    *,
    source_row: Mapping[str, Any],
    limit: int,
    days_back: int,
) -> dict[str, int]:
    metrics = {
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
    }
    username = str(source_row["username"])
    source_meta, posts = await scan_source_posts(client, username=username, limit=limit, days_back=days_back)
    await _update_source_runtime_meta(
        conn,
        source_id=int(source_row["id"]),
        title=source_meta.source_title,
        about_text=source_meta.about_text,
        about_links=source_meta.about_links,
        last_scanned_message_id=max((post.message_id for post in posts), default=None),
    )
    source_title = source_meta.source_title
    fallback_name = collapse_ws(str(source_row["display_name"] or source_title or username))
    for post in posts:
        metrics["posts_scanned"] += 1
        prefilter = _should_prefilter(post, str(source_row["source_kind"] or ""))
        post_id = await _upsert_monitor_post(
            conn,
            source_id=int(source_row["id"]),
            post=post,
            post_kind="mixed_or_non_target",
            prefilter_passed=prefilter,
        )
        if not prefilter:
            continue
        metrics["posts_prefiltered"] += 1
        occurrences = parse_post_occurrences(
            text=post.text,
            post_date=post.post_date,
            source_kind=str(source_row["source_kind"] or ""),
            source_title=source_title or collapse_ws(str(source_row["marketing_name"] or source_row["display_name"] or "")),
            channel_url=post.source_url,
            fallback_guide_name=fallback_name,
        )
        if not occurrences:
            continue
        await conn.execute(
            "UPDATE guide_monitor_post SET post_kind=?, title_hint=?, raw_facts_json=?, last_scanned_at=CURRENT_TIMESTAMP WHERE id=?",
            (
                occurrences[0].post_kind,
                occurrences[0].canonical_title,
                _json_dump(
                    [
                        {
                            "title": occ.canonical_title,
                            "date": occ.date_iso,
                            "time": occ.time_text,
                            "status": occ.status,
                            "eligible": occ.digest_eligible,
                        }
                        for occ in occurrences
                    ]
                ),
                int(post_id),
            ),
        )
        for parsed in occurrences:
            template_id = await _ensure_template(
                conn,
                profile_id=int(source_row["primary_profile_id"]) if source_row["primary_profile_id"] is not None else None,
                parsed=parsed,
            )
            if template_id is not None:
                metrics["templates_touched"] += 1
            occurrence_id, created = await _upsert_occurrence(
                conn,
                source_row=source_row,
                post_id=post_id,
                post=post,
                parsed=parsed,
                template_id=template_id,
            )
            if occurrence_id:
                if created:
                    metrics["occurrences_created"] += 1
                else:
                    metrics["occurrences_updated"] += 1
    return metrics


async def run_guide_monitor(
    db: Database,
    bot: Bot | None,
    *,
    chat_id: int | None,
    operator_id: int | None,
    trigger: str,
    mode: str = "full",
    send_progress: bool = True,
) -> GuideMonitorResult:
    run_id = uuid.uuid4().hex[:12]
    metrics = {
        "sources_scanned": 0,
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
        "errors": 0,
        "duration_sec": 0,
    }
    errors: list[str] = []
    started_monotonic = time.monotonic()
    ops_run_id: int | None = None
    limit, days_back = _light_or_full(mode)

    async with _RUN_LOCK:
        async with heavy_operation(
            kind="guide_monitoring",
            trigger=trigger,
            run_id=run_id,
            operator_id=operator_id,
            chat_id=chat_id,
        ) as allowed:
            if not allowed:
                raise RuntimeError("guide_monitoring is already running")

            ops_run_id = await start_ops_run(
                db,
                kind="guide_monitoring",
                trigger=trigger,
                chat_id=chat_id,
                operator_id=operator_id,
                details={"mode": mode, "run_id": run_id},
            )
            if send_progress and bot and chat_id:
                await bot.send_message(
                    int(chat_id),
                    (
                        "🧭 Запускаю мониторинг экскурсий.\n"
                        f"mode={html.escape(mode)}\n"
                        f"limit={limit}, days_back={days_back}\n"
                        f"run_id={html.escape(run_id)}"
                    ),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

            client = None
            try:
                async with db.raw_conn() as conn:
                    await _enable_row_factory(conn)
                    await seed_guide_sources(conn)
                    await conn.commit()
                    sources = await _get_enabled_sources(conn)
                client = await create_telethon_runtime_client()
                async with db.raw_conn() as conn:
                    await _enable_row_factory(conn)
                    for source in sources:
                        try:
                            source_metrics = await _scan_and_import_source(
                                conn,
                                client,
                                source_row=source,
                                limit=limit,
                                days_back=days_back,
                            )
                            metrics["sources_scanned"] += 1
                            for key, value in source_metrics.items():
                                metrics[key] = metrics.get(key, 0) + int(value or 0)
                        except Exception as exc:
                            logger.exception("guide_monitor: source failed username=%s", source["username"])
                            metrics["errors"] += 1
                            errors.append(f"@{source['username']}: {type(exc).__name__}: {exc}")
                    await conn.commit()
            except Exception as exc:
                logger.exception("guide_monitor failed")
                metrics["errors"] += 1
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                if client is not None:
                    try:
                        await client.disconnect()
                    except Exception:
                        logger.warning("guide_monitor: failed to disconnect telethon client", exc_info=True)

    metrics["duration_sec"] = int(max(0, round(time.monotonic() - started_monotonic)))
    status = "success" if not errors else ("partial" if metrics["sources_scanned"] > 0 else "error")
    details = {"mode": mode, "run_id": run_id, "errors": errors[:20]}
    await finish_ops_run(db, run_id=ops_run_id, status=status, metrics=metrics, details=details)

    result = GuideMonitorResult(
        run_id=run_id,
        ops_run_id=ops_run_id,
        trigger=trigger,
        mode=mode,
        metrics=metrics,
        errors=errors,
    )
    if send_progress and bot and chat_id:
        lines = [
            "✅ Мониторинг экскурсий завершён" if not errors else "⚠️ Мониторинг экскурсий завершён с ошибками",
            f"run_id={run_id}",
            f"Источников: {metrics['sources_scanned']}",
            f"Постов: {metrics['posts_scanned']}",
            f"После prefilter: {metrics['posts_prefiltered']}",
            f"Новых выходов: {metrics['occurrences_created']}",
            f"Обновлений: {metrics['occurrences_updated']}",
        ]
        if errors:
            lines.append("")
            lines.append("Ошибки:")
            lines.extend(f"- {collapse_ws(err)}"[:350] for err in errors[:5])
        await bot.send_message(
            int(chat_id),
            "\n".join(lines),
            disable_web_page_preview=True,
        )
    return result


async def _fetch_digest_candidates(
    conn: aiosqlite.Connection,
    *,
    family: str,
    limit: int = 24,
) -> list[dict[str, Any]]:
    where = [
        "go.digest_eligible = 1",
        "go.date IS NOT NULL",
        "go.date >= ?",
        "go.date <= ?",
    ]
    params: list[Any] = [_today_iso(), _future_cutoff_iso()]
    if family == "new_occurrences":
        where.append("go.published_new_digest_issue_id IS NULL")
    elif family == "last_call":
        where.append("go.is_last_call = 1")
        where.append("go.published_last_call_digest_issue_id IS NULL")
    cur = await conn.execute(
        f"""
        SELECT
            go.*,
            gs.username AS source_username,
            gs.title AS source_title,
            gs.source_kind AS source_kind,
            gs.about_text AS source_about_text,
            gs.about_links_json AS source_about_links_json,
            gs.priority_weight AS priority_weight
        FROM guide_occurrence go
        LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
        WHERE {' AND '.join(where)}
        ORDER BY go.date ASC, COALESCE(go.time, '99:99') ASC, go.updated_at DESC
        LIMIT ?
        """,
        (*params, int(limit)),
    )
    rows = await cur.fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["guide_names"] = _parse_json_array(item.get("guide_names_json"))
        item["organizer_names"] = _parse_json_array(item.get("organizer_names_json"))
        item["audience_fit"] = _parse_json_array(item.get("audience_fit_json"))
        item["source_about_links"] = _parse_json_array(item.get("source_about_links_json"))
        item["priority_weight"] = float(item.get("priority_weight") or 1.0)
        item["aggregator_only"] = int(item.get("aggregator_only") or 0)
        item["views"] = int(item["views"]) if item.get("views") is not None else None
        item["likes"] = int(item["likes"]) if item.get("likes") is not None else None
        post_cur = await conn.execute(
            """
            SELECT gmp.text
            FROM guide_occurrence_source gos
            JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
            WHERE gos.occurrence_id=?
            ORDER BY CASE WHEN gos.role='primary' THEN 0 ELSE 1 END, gmp.post_date DESC, gmp.id DESC
            LIMIT 1
            """,
            (int(item.get("id") or 0),),
        )
        post_row = await post_cur.fetchone()
        item["dedup_source_text"] = collapse_ws(str((post_row["text"] if post_row else "") or ""))
        out.append(item)
    return out


async def _load_source_medians(conn: aiosqlite.Connection, source_id: int) -> tuple[int | None, int | None]:
    cur = await conn.execute(
        """
        SELECT views, reactions_total
        FROM guide_monitor_post
        WHERE source_id=? AND post_date >= datetime('now', '-90 days')
        """,
        (int(source_id),),
    )
    rows = await cur.fetchall()
    views = [int(row["views"]) for row in rows if row["views"] is not None]
    likes = [int(row["reactions_total"]) for row in rows if row["reactions_total"] is not None]
    return _median(views), _median(likes)


async def build_guide_digest_preview(
    db: Database,
    *,
    family: str,
    limit: int = 24,
    run_id: int | None = None,
) -> dict[str, Any]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        raw_limit = max(int(limit) * 3, 48)
        rows = await _fetch_digest_candidates(conn, family=family, limit=raw_limit)
        prepared: list[dict[str, Any]] = []
        for row in rows:
            median_views, median_likes = await _load_source_medians(conn, int(row["primary_source_id"] or 0))
            row["popularity_mark"] = _popularity_mark(
                views=row.get("views"),
                likes=row.get("likes"),
                median_views=median_views,
                median_likes=median_likes,
            )
            score = float(row.get("priority_weight") or 1.0)
            if row.get("popularity_mark"):
                score += 0.4
            if int(row.get("aggregator_only") or 0):
                score -= 0.5
            if family == "last_call":
                score += 1.5 if int(row.get("is_last_call") or 0) else 0.0
            row["_score"] = score
            prepared.append(row)
        prepared.sort(
            key=lambda item: (
                -float(item.get("_score") or 0),
                str(item.get("date") or ""),
                _safe_time_sort(str(item.get("time") or "")),
                int(item.get("id") or 0),
            )
        )
        dedup = await deduplicate_occurrence_rows(prepared, family=family, limit=limit)
        display_rows = list(dedup.display_rows)
        display_rows, editorial_suppressed_ids, _editorial_reasons = await refine_digest_rows(
            display_rows,
            family=family,
            date_formatter=format_date_time,
        )
        texts = build_digest_messages(display_rows, family=family)
        occurrence_ids = list(dict.fromkeys([*dedup.covered_occurrence_ids, *editorial_suppressed_ids]))
        media_items: list[dict[str, Any]] = []
        for row in display_rows[:MAX_MEDIA_ITEMS]:
            cur = await conn.execute(
                """
                SELECT
                    gmp.source_url,
                    gmp.media_refs_json,
                    gs.username
                FROM guide_occurrence_source gos
                JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
                JOIN guide_source gs ON gs.id = gmp.source_id
                WHERE gos.occurrence_id=?
                ORDER BY CASE WHEN gos.role='primary' THEN 0 ELSE 1 END, gmp.post_date DESC, gmp.id DESC
                LIMIT 1
                """,
                (int(row["id"]),),
            )
            media_row = await cur.fetchone()
            if not media_row:
                continue
            refs = _parse_json_array(media_row["media_refs_json"])
            if not refs:
                continue
            media_items.append(
                {
                    "occurrence_id": int(row["id"]),
                    "source_username": str(media_row["username"]),
                    "source_url": str(media_row["source_url"] or ""),
                    "media_ref": refs[0],
                }
            )
        cur = await conn.execute(
            """
            INSERT INTO guide_digest_issue(family, status, target_chat, title, text, items_json, media_items_json, run_id)
            VALUES(?, 'preview', ?, ?, ?, ?, ?, ?)
            """,
            (
                family,
                GUIDE_DIGEST_TARGET_CHAT,
                texts[0][:180] if texts else "",
                "\n\n---PART---\n\n".join(texts),
                _json_dump(occurrence_ids),
                _json_dump(media_items),
                int(run_id) if run_id is not None else None,
            ),
        )
        issue_id = int(cur.lastrowid or 0)
        await conn.commit()
    return {
        "issue_id": issue_id,
        "family": family,
        "texts": texts,
        "items": display_rows,
        "media_items": media_items,
        "covered_occurrence_ids": occurrence_ids,
        "suppressed_occurrence_ids": list(dict.fromkeys([*dedup.suppressed_occurrence_ids, *editorial_suppressed_ids])),
        "pair_decisions": list(dedup.pair_decisions),
    }


async def _bridge_media_file_id(
    bot: Bot,
    *,
    staging_chat_id: int,
    source_username: str,
    message_id: int,
) -> tuple[str | None, str | None]:
    forwarded = await bot.forward_message(
        chat_id=int(staging_chat_id),
        from_chat_id=f"@{source_username}",
        message_id=int(message_id),
        disable_notification=True,
    )
    try:
        if forwarded.photo:
            return "photo", forwarded.photo[-1].file_id
        if forwarded.video:
            return "video", forwarded.video.file_id
        document = getattr(forwarded, "document", None)
        if document:
            mime = str(getattr(document, "mime_type", None) or "").lower()
            if mime.startswith("video/"):
                return "video", document.file_id
            if mime.startswith("image/"):
                return "photo", document.file_id
        return None, None
    finally:
        try:
            await bot.delete_message(chat_id=int(staging_chat_id), message_id=forwarded.message_id)
        except Exception:
            logger.warning("guide_digest: failed to delete staging forwarded message", exc_info=True)


async def _download_media_via_telethon(
    client: Any,
    *,
    source_username: str,
    message_id: int,
) -> tuple[str | None, bytes | None, str | None]:
    entity = await client.get_entity(source_username)
    message = await client.get_messages(entity, ids=int(message_id))
    if not message:
        return None, None, None
    kind = "photo"
    if getattr(message, "video", None) or (
        getattr(message, "document", None) is not None
        and str(getattr(getattr(message, "document", None), "mime_type", "") or "").lower().startswith("video/")
    ):
        kind = "video"
    data = await client.download_media(message, file=bytes)
    if not data:
        return None, None, None
    payload = bytes(data)
    if len(payload) > GUIDE_MEDIA_TELETHON_FALLBACK_MAX_MB * 1024 * 1024:
        return None, None, None
    filename = f"{source_username}_{message_id}.{'mp4' if kind == 'video' else 'jpg'}"
    return kind, payload, filename


async def publish_guide_digest(
    db: Database,
    bot: Bot,
    *,
    family: str,
    chat_id: int | None,
    target_chat: str | None = None,
) -> dict[str, Any]:
    preview = await build_guide_digest_preview(db, family=family)
    issue_id = int(preview["issue_id"])
    target = collapse_ws(target_chat) or GUIDE_DIGEST_TARGET_CHAT
    texts: list[str] = list(preview["texts"])
    if not texts:
        return {"issue_id": issue_id, "published": False, "reason": "empty"}
    message_ids: list[int] = []
    media_payload: list[types.InputMediaPhoto | types.InputMediaVideo] = []
    staging_chat_id = int(chat_id or await resolve_superadmin_chat_id(db) or 0)
    telethon_client = None
    if staging_chat_id:
        try:
            for idx, item in enumerate(preview["media_items"][:MAX_MEDIA_ITEMS]):
                media_ref = dict(item.get("media_ref") or {})
                source_username = str(item.get("source_username") or "").strip().lstrip("@")
                source_message_id = int(media_ref.get("message_id") or 0)
                if not source_username or source_message_id <= 0:
                    continue
                kind = None
                file_id = None
                try:
                    kind, file_id = await _bridge_media_file_id(
                        bot,
                        staging_chat_id=staging_chat_id,
                        source_username=source_username,
                        message_id=source_message_id,
                    )
                except Exception:
                    logger.warning(
                        "guide_digest: media bridge failed source=%s message_id=%s",
                        source_username,
                        source_message_id,
                        exc_info=True,
                    )
                caption = build_media_caption(
                    family=family,
                    item_count=len(preview["items"]),
                    media_count=len(preview["media_items"]),
                ) if idx == 0 else None
                if kind and file_id:
                    if kind == "video":
                        media_payload.append(types.InputMediaVideo(media=file_id, caption=caption))
                    else:
                        media_payload.append(types.InputMediaPhoto(media=file_id, caption=caption))
                    continue

                if telethon_client is None:
                    try:
                        telethon_client = await create_telethon_runtime_client()
                    except Exception:
                        logger.warning("guide_digest: telethon media fallback unavailable", exc_info=True)
                        telethon_client = False
                if telethon_client:
                    try:
                        dl_kind, dl_data, dl_filename = await _download_media_via_telethon(
                            telethon_client,
                            source_username=source_username,
                            message_id=source_message_id,
                        )
                    except Exception:
                        logger.warning(
                            "guide_digest: telethon download fallback failed source=%s message_id=%s",
                            source_username,
                            source_message_id,
                            exc_info=True,
                        )
                        continue
                    if dl_kind and dl_data and dl_filename:
                        upload = BufferedInputFile(dl_data, filename=dl_filename)
                        if dl_kind == "video":
                            media_payload.append(types.InputMediaVideo(media=upload, caption=caption))
                        else:
                            media_payload.append(types.InputMediaPhoto(media=upload, caption=caption))
        finally:
            if telethon_client and telethon_client is not False:
                try:
                    await telethon_client.disconnect()
                except Exception:
                    logger.warning("guide_digest: failed to disconnect telethon fallback client", exc_info=True)
    if media_payload:
        sent = await bot.send_media_group(chat_id=target, media=media_payload)
        message_ids.extend(int(msg.message_id) for msg in sent if getattr(msg, "message_id", None))
    for text in texts:
        sent = await bot.send_message(target, text, parse_mode="HTML", disable_web_page_preview=True)
        if getattr(sent, "message_id", None):
            message_ids.append(int(sent.message_id))

    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        await conn.execute(
            """
            UPDATE guide_digest_issue
            SET
                status='published',
                target_chat=?,
                published_at=CURRENT_TIMESTAMP,
                published_message_ids_json=?
            WHERE id=?
            """,
            (target, _json_dump(message_ids), issue_id),
        )
        occurrence_ids = [int(item) for item in (preview.get("covered_occurrence_ids") or [])]
        if occurrence_ids:
            marks_sql = ",".join("?" for _ in occurrence_ids)
            column = "published_new_digest_issue_id" if family == "new_occurrences" else "published_last_call_digest_issue_id"
            await conn.execute(
                f"UPDATE guide_occurrence SET {column}=?, updated_at=CURRENT_TIMESTAMP WHERE id IN ({marks_sql})",
                (issue_id, *occurrence_ids),
            )
        await conn.commit()
    return {"issue_id": issue_id, "published": True, "target_chat": target, "message_ids": message_ids, "texts": texts}


async def fetch_guide_sources_summary(db: Database) -> list[dict[str, Any]]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            """
            SELECT
                gs.username,
                gs.title,
                gs.source_kind,
                gs.trust_level,
                gs.last_scan_at,
                COUNT(DISTINCT gmp.id) AS posts_seen,
                COUNT(DISTINCT go.id) AS occurrences_seen
            FROM guide_source gs
            LEFT JOIN guide_monitor_post gmp ON gmp.source_id = gs.id
            LEFT JOIN guide_occurrence go ON go.primary_source_id = gs.id
            WHERE gs.platform='telegram'
            GROUP BY gs.id
            ORDER BY gs.priority_weight DESC, gs.username ASC
            """
        )
        rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def render_guide_sources_summary(db: Database) -> str:
    rows = await fetch_guide_sources_summary(db)
    lines = ["🗂 Источники экскурсий"]
    for row in rows:
        uname = f"@{row['username']}"
        title = collapse_ws(str(row.get("title") or ""))
        kind = collapse_ws(str(row.get("source_kind") or ""))
        trust = collapse_ws(str(row.get("trust_level") or ""))
        posts = int(row.get("posts_seen") or 0)
        occ = int(row.get("occurrences_seen") or 0)
        label = f"{uname}"
        if title:
            label += f" — {title}"
        lines.append(f"- {label} [{kind}, trust={trust}] posts={posts}, occ={occ}")
    return "\n".join(lines)
