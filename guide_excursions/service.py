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
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiosqlite
from aiogram import Bot, types
from aiogram.types import BufferedInputFile

from admin_chat import resolve_superadmin_chat_id
from db import Database
from heavy_ops import heavy_operation
from kaggle_registry import list_jobs, remove_job
from ops_run import finish_ops_run, start_ops_run
from video_announce.kaggle_client import KaggleClient

from .dedup import deduplicate_occurrence_rows
from .digest import MAX_MEDIA_ITEMS, build_digest_messages, build_media_caption, format_date_time
from .digest_writer import apply_digest_batch_copy
from .editorial import refine_digest_rows
from .enrich import apply_occurrence_enrichment, apply_profile_enrichment
from .kaggle_service import (
    download_guide_results,
    extract_guide_failure_message,
    format_kaggle_status_message,
    run_guide_monitor_kaggle,
)
from .parser import (
    GuideParsedOccurrence,
    build_source_fingerprint,
    collapse_ws,
    normalize_title_key,
    parse_post_occurrences,
)
from .public_identity import resolve_public_guide_names
from .scanner import GuideScannedPost, scan_source_posts
from .seed import seed_guide_sources
from .telethon_client import create_telethon_runtime_client

logger = logging.getLogger(__name__)

GUIDE_DIGEST_TARGET_CHAT = (os.getenv("GUIDE_DIGEST_TARGET_CHAT") or "@keniggpt").strip() or "@keniggpt"
GUIDE_SCAN_LIMIT_FULL = max(10, min(int((os.getenv("GUIDE_SCAN_LIMIT_FULL") or "60") or 60), 200))
GUIDE_SCAN_LIMIT_LIGHT = max(5, min(int((os.getenv("GUIDE_SCAN_LIMIT_LIGHT") or "25") or 25), 120))
GUIDE_DAYS_BACK_FULL = max(3, min(int((os.getenv("GUIDE_DAYS_BACK_FULL") or "5") or 5), 90))
GUIDE_DAYS_BACK_BOOTSTRAP = max(
    GUIDE_DAYS_BACK_FULL,
    min(int((os.getenv("GUIDE_DAYS_BACK_BOOTSTRAP") or "14") or 14), 90),
)
GUIDE_DAYS_BACK_LIGHT = max(2, min(int((os.getenv("GUIDE_DAYS_BACK_LIGHT") or "3") or 3), 30))
GUIDE_DIGEST_WINDOW_DAYS = max(3, min(int((os.getenv("GUIDE_DIGEST_WINDOW_DAYS") or "45") or 45), 90))
GUIDE_MEDIA_TELETHON_FALLBACK_MAX_MB = max(
    1,
    min(int((os.getenv("GUIDE_MEDIA_TELETHON_FALLBACK_MAX_MB") or "20") or 20), 100),
)
GUIDE_EXCURSIONS_KAGGLE_ENABLED = (
    (os.getenv("GUIDE_EXCURSIONS_KAGGLE_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_EXCURSIONS_LOCAL_FALLBACK_ENABLED = (
    (os.getenv("GUIDE_EXCURSIONS_LOCAL_FALLBACK_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)
GUIDE_OCCURRENCES_PAGE_SIZE = max(5, min(int((os.getenv("GUIDE_OCCURRENCES_PAGE_SIZE") or "8") or 8), 20))
GUIDE_TEMPLATES_PAGE_SIZE = max(5, min(int((os.getenv("GUIDE_TEMPLATES_PAGE_SIZE") or "8") or 8), 20))
GUIDE_RECENT_CHANGES_DEFAULT_HOURS = max(
    1,
    min(int((os.getenv("GUIDE_RECENT_CHANGES_DEFAULT_HOURS") or "24") or 24), 720),
)

_RUN_LOCK = asyncio.Lock()
_GUIDE_MONITOR_RECOVERY_ACTIVE: set[str] = set()


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


def _mapping_value(mapping: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    try:
        return mapping[key]
    except Exception:
        if isinstance(mapping, dict):
            return mapping.get(key, default)
    return default


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


async def _should_use_bootstrap_horizon(db: Database, *, mode: str) -> bool:
    if (mode or "full").strip().lower() == "light":
        return False
    if GUIDE_DAYS_BACK_BOOTSTRAP <= GUIDE_DAYS_BACK_FULL:
        return False
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            """
            SELECT 1
            FROM ops_run
            WHERE kind='guide_monitoring' AND status IN ('success', 'partial')
            LIMIT 1
            """
        )
        row = await cur.fetchone()
    return row is None


async def _resolve_scan_window(db: Database, mode: str) -> tuple[int, int]:
    limit, days_back = _light_or_full(mode)
    if await _should_use_bootstrap_horizon(db, mode=mode):
        days_back = max(days_back, GUIDE_DAYS_BACK_BOOTSTRAP)
    return limit, days_back


def _booking_url_for_digest(value: str | None) -> str | None:
    raw = collapse_ws(value)
    if not raw:
        return None
    if raw.startswith("tel:"):
        return raw
    if raw.lower().startswith("t.me/"):
        return f"https://{raw}"
    if raw.lower().startswith("telegram.me/"):
        return f"https://{raw}"
    return raw


def _normalize_digest_eligibility(
    *,
    date_iso: str | None,
    availability_mode: str | None,
    status: str | None,
    time_text: str | None = None,
    city: str | None = None,
    meeting_point: str | None = None,
    route_summary: str | None = None,
    price_text: str | None = None,
    booking_text: str | None = None,
    booking_url: str | None = None,
    summary_one_liner: str | None = None,
    digest_blurb: str | None = None,
    digest_eligible: bool,
    digest_reason: str | None,
) -> tuple[bool, str | None]:
    if not collapse_ws(date_iso):
        return False, collapse_ws(digest_reason) or "missing_date"
    mode = collapse_ws(availability_mode)
    if mode in {"on_request_private", "private"}:
        return False, collapse_ws(digest_reason) or "not_scheduled_public"
    if collapse_ws(status) == "cancelled":
        return False, collapse_ws(digest_reason) or "cancelled"
    if digest_eligible:
        return True, collapse_ws(digest_reason) or None
    evidence_count = sum(
        1
        for value in (
            time_text,
            city,
            meeting_point,
            route_summary,
            price_text,
            booking_text,
            booking_url,
            summary_one_liner,
            digest_blurb,
        )
        if collapse_ws(value)
    )
    if mode in {"", "scheduled_public", "limited"} and evidence_count >= 2:
        return True, collapse_ws(digest_reason) or "fact_policy_promoted"
    return bool(digest_eligible), collapse_ws(digest_reason) or None


def _is_tg_post_url(url: str | None) -> bool:
    raw = collapse_ws(url)
    if not raw.startswith("https://t.me/"):
        return False
    parts = raw.rstrip("/").split("/")
    return len(parts) >= 5 and parts[-1].isdigit()


def _source_post_url(payload_url: str | None, fallback_url: str | None) -> str | None:
    primary = collapse_ws(payload_url)
    if _is_tg_post_url(primary):
        return primary
    fallback = collapse_ws(fallback_url)
    if _is_tg_post_url(fallback):
        return fallback
    return primary or fallback or None


def _parse_iso_date(value: str | None) -> date | None:
    raw = collapse_ws(value)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _is_past_occurrence(date_iso: str | None) -> bool:
    parsed = _parse_iso_date(date_iso)
    if not parsed:
        return False
    return parsed < _utc_now().date()


def _safe_json_object(value: Any) -> dict[str, Any]:
    data = _json_load(value, fallback={})
    return data if isinstance(data, dict) else {}


def _safe_json_list(value: Any) -> list[Any]:
    data = _json_load(value, fallback=[])
    return data if isinstance(data, list) else []


def _normalize_string_list(value: Any, *, limit: int = 8) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = _safe_json_list(value)
    for item in raw_items:
        text = collapse_ws(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_fact_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        raw = _json_dump(value)
        return raw if raw not in {"{}", "[]"} else None
    text = collapse_ws(value)
    return text or None


def _merge_json_objects(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (extra or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not collapse_ws(value):
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, dict) and not value:
            continue
        out[key] = value
    return out


def _build_occurrence_fact_pack(
    parsed: GuideParsedOccurrence,
    *,
    post: GuideScannedPost,
    source_row: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    digest_eligible, digest_reason = _normalize_digest_eligibility(
        date_iso=parsed.date_iso,
        availability_mode=parsed.availability_mode,
        status=parsed.status,
        time_text=parsed.time_text,
        city=parsed.city,
        meeting_point=parsed.meeting_point,
        route_summary=parsed.route_summary,
        price_text=parsed.price_text,
        booking_text=parsed.booking_text,
        booking_url=parsed.booking_url,
        summary_one_liner=parsed.summary_one_liner,
        digest_blurb=parsed.digest_blurb,
        digest_eligible=bool(parsed.digest_eligible),
        digest_reason=parsed.digest_eligibility_reason,
    )
    pack = {
        "canonical_title": parsed.canonical_title,
        "title_normalized": parsed.title_normalized,
        "date": parsed.date_iso,
        "time": parsed.time_text,
        "duration_text": parsed.duration_text,
        "city": parsed.city,
        "meeting_point": parsed.meeting_point,
        "route_summary": parsed.route_summary,
        "audience_fit": list(parsed.audience_fit),
        "group_format": parsed.group_format,
        "price_text": parsed.price_text,
        "booking_text": parsed.booking_text,
        "booking_url": _booking_url_for_digest(parsed.booking_url),
        "channel_url": parsed.channel_url,
        "status": parsed.status,
        "seats_text": parsed.seats_text,
        "summary_one_liner": parsed.summary_one_liner,
        "digest_blurb": parsed.digest_blurb,
        "digest_eligible": digest_eligible,
        "digest_eligibility_reason": digest_reason,
        "is_last_call": bool(parsed.is_last_call),
        "post_kind": parsed.post_kind,
        "availability_mode": parsed.availability_mode,
        "guide_names": list(parsed.guide_names),
        "organizer_names": list(parsed.organizer_names),
        "source_username": str(_mapping_value(source_row, "username") or ""),
        "source_title": str(_mapping_value(source_row, "title") or _mapping_value(source_row, "display_name") or ""),
        "source_kind": str(_mapping_value(source_row, "source_kind") or ""),
        "source_url": post.source_url,
        "source_post_url": post.source_url,
        "source_message_id": int(post.message_id),
        "views": int(post.views) if post.views is not None else None,
        "reactions_total": int(post.reactions_total) if post.reactions_total is not None else None,
        "media_refs": list(post.media_refs or []),
    }
    return _merge_json_objects(pack, dict(extra or {}))


def _default_claim_role(fact_key: str) -> str:
    if fact_key in {"canonical_title", "title_normalized", "date", "time", "booking_url"}:
        return "anchor"
    if fact_key in {"status", "seats_text"}:
        return "status_delta"
    return "support"


def _normalize_claim_payloads(claims: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for claim in claims or []:
        if not isinstance(claim, Mapping):
            continue
        fact_key = collapse_ws(claim.get("fact_key")) or collapse_ws(claim.get("fact_type"))
        fact_value = claim.get("fact_value")
        if fact_value is None:
            fact_value = collapse_ws(claim.get("claim_text")) or collapse_ws(claim.get("fact")) or None
        if not fact_key and fact_value is not None:
            fact_key = "claim_text"
        if not fact_key or _normalize_fact_value(fact_value) is None:
            continue
        payload = dict(claim)
        payload["fact_key"] = fact_key
        payload["fact_value"] = fact_value
        normalized.append(payload)
    return normalized


def _report_occurrence_entry(
    *,
    occurrence_id: int,
    created: bool,
    parsed: GuideParsedOccurrence,
    source_username: str,
    source_post_url: str,
) -> dict[str, Any]:
    return {
        "occurrence_id": int(occurrence_id),
        "action": "created" if created else "updated",
        "canonical_title": parsed.canonical_title,
        "date": parsed.date_iso,
        "time": parsed.time_text,
        "status": parsed.status,
        "source_username": source_username,
        "source_post_url": source_post_url,
    }


def _format_source_post_label(source_username: str | None, source_post_url: str | None) -> str:
    username = collapse_ws(source_username).lstrip("@")
    url = collapse_ws(source_post_url)
    if username and url and _is_tg_post_url(url):
        return f"@{username}/{url.rstrip('/').split('/')[-1]}"
    if username:
        return f"@{username}"
    return url or "—"


def _parse_db_timestamp(value: Any) -> datetime | None:
    raw = collapse_ws(value)
    if not raw:
        return None
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(candidate)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _format_compact_utc(value: Any) -> str:
    dt = _parse_db_timestamp(value)
    if not dt:
        return "?"
    return dt.astimezone(timezone.utc).strftime("%d.%m %H:%M UTC")


def _chunk_plain_lines(lines: Sequence[str], *, max_len: int = 3800) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for raw_line in lines:
        line = str(raw_line or "")
        line_len = len(line) + 1
        if current and current_len + line_len > max_len:
            chunks.append("\n".join(current).strip())
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len
    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


async def _fetch_sqlite_int(conn: aiosqlite.Connection, sql: str, params: Sequence[Any]) -> int:
    cur = await conn.execute(sql, tuple(params))
    row = await cur.fetchone()
    try:
        return int((row[0] if row else 0) or 0)
    except Exception:
        return 0


def _status_icon(status: str | None) -> str:
    value = collapse_ws(status).lower()
    return {
        "success": "✅",
        "ok": "✅",
        "completed": "✅",
        "partial": "⚠️",
        "running": "⏳",
        "error": "❌",
        "failed": "❌",
        "crashed": "💥",
    }.get(value, "•")


def _short_text(value: Any, *, limit: int = 88) -> str:
    text = collapse_ws(value)
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _page_state(*, total_count: int, page: int, page_size: int) -> tuple[int, int, int]:
    size = max(1, int(page_size))
    total_pages = max(1, (max(0, int(total_count)) + size - 1) // size)
    current = max(1, min(int(page), total_pages))
    offset = (current - 1) * size
    return current, total_pages, offset


def _format_occurrence_change_line(change: Mapping[str, Any]) -> str:
    action = collapse_ws(change.get("action")) or "updated"
    icon = {
        "created": "✅",
        "updated": "🔄",
        "past_skipped": "⏭️",
    }.get(action, "•")
    occurrence_id = int(change.get("occurrence_id") or 0)
    when = format_date_time(
        collapse_ws(change.get("date")) or None,
        collapse_ws(change.get("time")) or None,
    ) or collapse_ws(change.get("date")) or "без даты"
    title = _short_text(change.get("canonical_title") or "Экскурсия", limit=96)
    line = f"{icon} #{occurrence_id} {when} — {title}".strip()
    source_label = _format_source_post_label(
        change.get("source_username"),
        change.get("source_post_url"),
    )
    if source_label != "—":
        line += f" [{source_label}]"
    return line


def _primary_occurrence_source_label(
    row: Mapping[str, Any],
    post_rows: Sequence[Mapping[str, Any]],
    *,
    fact_pack: Mapping[str, Any] | None = None,
) -> str:
    for post in post_rows:
        label = _format_source_post_label(
            _mapping_value(post, "source_username"),
            _mapping_value(post, "source_url"),
        )
        if label != "—":
            return label
    payload = fact_pack or {}
    return _format_source_post_label(
        _mapping_value(row, "source_username"),
        _mapping_value(payload, "source_post_url") or _mapping_value(payload, "source_url") or _mapping_value(row, "channel_url"),
    )


def _derive_claims_from_fact_pack(fact_pack: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for fact_key in (
        "canonical_title",
        "title_normalized",
        "date",
        "time",
        "duration_text",
        "city",
        "meeting_point",
        "route_summary",
        "price_text",
        "booking_text",
        "booking_url",
        "status",
        "seats_text",
        "summary_one_liner",
        "digest_blurb",
        "availability_mode",
        "post_kind",
        "group_format",
        "digest_eligibility_reason",
    ):
        value = _normalize_fact_value(fact_pack.get(fact_key))
        if not value:
            continue
        claims.append(
            {
                "fact_key": fact_key,
                "fact_value": value,
                "claim_role": _default_claim_role(fact_key),
                "confidence": 0.7,
                "provenance": {"source": "fact_pack"},
            }
        )
    for audience in _normalize_string_list(fact_pack.get("audience_fit"), limit=6):
        claims.append(
            {
                "fact_key": "audience_fit",
                "fact_value": audience,
                "claim_role": "support",
                "confidence": 0.7,
                "provenance": {"source": "fact_pack"},
            }
        )
    for guide_name in _normalize_string_list(fact_pack.get("guide_names"), limit=4):
        claims.append(
            {
                "fact_key": "guide_name",
                "fact_value": guide_name,
                "claim_role": "support",
                "confidence": 0.7,
                "provenance": {"source": "fact_pack"},
            }
        )
    for organizer in _normalize_string_list(fact_pack.get("organizer_names"), limit=4):
        claims.append(
            {
                "fact_key": "organizer_name",
                "fact_value": organizer,
                "claim_role": "support",
                "confidence": 0.7,
                "provenance": {"source": "fact_pack"},
            }
        )
    return claims


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
    llm_status: str | None = None,
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
                llm_status=COALESCE(NULLIF(?, ''), llm_status),
                last_scanned_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            payloads[2:] + (collapse_ws(llm_status), int(row["id"])),
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
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        payloads + (collapse_ws(llm_status) or "heuristic_only",),
    )
    return int(cur.lastrowid or 0)


async def _insert_entity_claims(
    conn: aiosqlite.Connection,
    *,
    entity_kind: str,
    entity_id: int,
    source_post_id: int | None,
    claims: Sequence[Mapping[str, Any]],
    replace_source_post_claims: bool = True,
) -> None:
    if replace_source_post_claims and source_post_id is not None:
        await conn.execute(
            "DELETE FROM guide_fact_claim WHERE entity_kind=? AND entity_id=? AND source_post_id=?",
            (entity_kind, int(entity_id), int(source_post_id)),
        )
    for claim in _normalize_claim_payloads(claims):
        fact_key = collapse_ws(claim.get("fact_key"))
        fact_value = _normalize_fact_value(claim.get("fact_value"))
        if not fact_key or not fact_value:
            continue
        confidence_raw = claim.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.7
        except Exception:
            confidence = 0.7
        provenance = _safe_json_object(claim.get("provenance"))
        if claim.get("fact_refs"):
            provenance["fact_refs"] = list(_safe_json_list(claim.get("fact_refs")))
        await conn.execute(
            """
            INSERT INTO guide_fact_claim(
                entity_kind,
                entity_id,
                fact_key,
                fact_value,
                confidence,
                source_post_id,
                claim_role,
                provenance_json,
                observed_at,
                last_confirmed_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            (
                entity_kind,
                int(entity_id),
                fact_key,
                fact_value,
                confidence,
                int(source_post_id) if source_post_id is not None else None,
                collapse_ws(claim.get("claim_role")) or _default_claim_role(fact_key),
                _json_dump(provenance) if provenance else None,
            ),
        )


def _unique_limited(values: Sequence[Any], *, limit: int = 6) -> list[str]:
    out: list[str] = []
    for value in values:
        text = collapse_ws(value)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _count_labels(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        label = collapse_ws(value).lower()
        if label not in {"locals", "tourists", "mixed"}:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def _build_occurrence_enrichment_claims(enrichment: Mapping[str, Any]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    main_hook = collapse_ws(enrichment.get("main_hook"))
    if main_hook:
        claims.append(
            {
                "fact_key": "main_hook",
                "fact_value": main_hook,
                "claim_role": "enrich_hook",
                "confidence": float(enrichment.get("main_hook_confidence") or 0) / 100.0 or 0.7,
                "provenance": {
                    "source": "guide_occurrence_enrich",
                    "evidence": list(_safe_json_list(enrichment.get("main_hook_evidence"))),
                },
            }
        )
    label = collapse_ws(enrichment.get("audience_region_fit_label")).lower()
    if label in {"locals", "tourists", "mixed"}:
        claims.append(
            {
                "fact_key": "audience_region_fit_label",
                "fact_value": label,
                "claim_role": "audience_region_fit",
                "confidence": float(enrichment.get("audience_region_confidence") or 0) / 100.0 or 0.7,
                "provenance": {
                    "source": "guide_occurrence_enrich",
                    "local_score": int(enrichment.get("audience_region_local_score") or 0),
                    "tourist_score": int(enrichment.get("audience_region_tourist_score") or 0),
                    "evidence": list(_safe_json_list(enrichment.get("audience_region_evidence"))),
                    "ambiguity": collapse_ws(enrichment.get("audience_region_ambiguity")) or None,
                },
            }
        )
    return claims


async def _apply_occurrence_enrichment_results(
    conn: aiosqlite.Connection,
    *,
    enrich_targets: Sequence[Mapping[str, Any]],
    enrichments: Mapping[int, Mapping[str, Any]],
) -> set[int]:
    touched_template_ids: set[int] = set()
    for target in enrich_targets:
        occurrence_id = int(_mapping_value(target, "occurrence_id") or 0)
        if occurrence_id <= 0:
            continue
        enrichment = enrichments.get(occurrence_id)
        if not isinstance(enrichment, Mapping) or not enrichment:
            continue
        fact_pack = dict(_mapping_value(target, "fact_pack") or {})
        merged_fact_pack = _merge_json_objects(fact_pack, dict(enrichment))
        await conn.execute(
            "UPDATE guide_occurrence SET fact_pack_json=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (_json_dump(merged_fact_pack), int(occurrence_id)),
        )
        source_post_id = int(_mapping_value(target, "post_id") or 0) or None
        if source_post_id is not None:
            await conn.execute(
                """
                DELETE FROM guide_fact_claim
                WHERE entity_kind='occurrence'
                  AND entity_id=?
                  AND source_post_id=?
                  AND claim_role IN ('enrich_hook', 'audience_region_fit')
                """,
                (int(occurrence_id), int(source_post_id)),
            )
        claims = _build_occurrence_enrichment_claims(enrichment)
        if claims:
            await _insert_entity_claims(
                conn,
                entity_kind="occurrence",
                entity_id=int(occurrence_id),
                source_post_id=source_post_id,
                claims=claims,
                replace_source_post_claims=False,
            )
        template_id = int(_mapping_value(target, "template_id") or 0)
        if template_id > 0:
            touched_template_ids.add(template_id)
    return touched_template_ids


def _upsert_profile_enrich_target(
    targets: dict[int, dict[str, Any]],
    *,
    profile_id: int | None,
    source_row: Mapping[str, Any],
    sample_title: str | None = None,
    sample_summary: str | None = None,
    sample_hook: str | None = None,
    guide_names: Sequence[str] | None = None,
) -> None:
    if profile_id is None:
        return
    target = targets.setdefault(
        int(profile_id),
        {
            "profile_id": int(profile_id),
            "source_username": collapse_ws(_mapping_value(source_row, "username")),
            "source_title": collapse_ws(_mapping_value(source_row, "title")),
            "display_name": collapse_ws(_mapping_value(source_row, "display_name")),
            "marketing_name": collapse_ws(_mapping_value(source_row, "marketing_name")),
            "about_text": collapse_ws(_mapping_value(source_row, "about_text")),
            "sample_titles": [],
            "sample_summaries": [],
            "sample_hooks": [],
            "guide_names": [],
        },
    )
    target["about_text"] = collapse_ws(_mapping_value(source_row, "about_text")) or target.get("about_text") or ""
    target["source_title"] = collapse_ws(_mapping_value(source_row, "title")) or target.get("source_title") or ""
    for key, value, limit in (
        ("sample_titles", sample_title, 6),
        ("sample_summaries", sample_summary, 4),
        ("sample_hooks", sample_hook, 4),
    ):
        text = collapse_ws(value)
        if not text:
            continue
        current = _normalize_string_list(target.get(key), limit=limit)
        target[key] = _normalize_string_list([*current, text], limit=limit)
    if guide_names:
        target["guide_names"] = _normalize_string_list([*list(target.get("guide_names") or []), *list(guide_names)], limit=6)


async def _apply_profile_enrichment_results(
    conn: aiosqlite.Connection,
    *,
    profile_targets: Sequence[Mapping[str, Any]],
    enrichments: Mapping[int, Mapping[str, Any]],
) -> int:
    touched = 0
    for target in profile_targets:
        profile_id = int(_mapping_value(target, "profile_id") or 0)
        if profile_id <= 0:
            continue
        enrichment = enrichments.get(profile_id)
        if not isinstance(enrichment, Mapping) or not enrichment:
            continue
        facts_rollup = {
            "guide_line": collapse_ws(enrichment.get("guide_line")) or None,
            "credentials": _normalize_string_list(_safe_json_list(enrichment.get("credentials")), limit=6),
            "expertise_tags": _normalize_string_list(_safe_json_list(enrichment.get("expertise_tags")), limit=8),
        }
        await _update_profile_from_hint(
            conn,
            profile_id=profile_id,
            source_row=target,
            profile_hint={
                "display_name": collapse_ws(enrichment.get("display_name")) or None,
                "summary_short": collapse_ws(enrichment.get("summary_short")) or None,
                "facts_rollup": {key: value for key, value in facts_rollup.items() if value},
            },
            source_post_id=None,
        )
        claims: list[dict[str, Any]] = []
        guide_line = collapse_ws(enrichment.get("guide_line"))
        if guide_line:
            claims.append(
                {
                    "fact_key": "guide_line",
                    "fact_value": guide_line,
                    "claim_role": "guide_profile_enrich",
                    "confidence": float(enrichment.get("confidence") or 0) / 100.0 or 0.75,
                    "provenance": {"source": "guide_profile_enrich"},
                }
            )
        for credential in _normalize_string_list(_safe_json_list(enrichment.get("credentials")), limit=6):
            claims.append(
                {
                    "fact_key": "credential",
                    "fact_value": credential,
                    "claim_role": "guide_profile_enrich",
                    "confidence": float(enrichment.get("confidence") or 0) / 100.0 or 0.75,
                    "provenance": {"source": "guide_profile_enrich"},
                }
            )
        for expertise in _normalize_string_list(_safe_json_list(enrichment.get("expertise_tags")), limit=8):
            claims.append(
                {
                    "fact_key": "expertise_tag",
                    "fact_value": expertise,
                    "claim_role": "guide_profile_enrich",
                    "confidence": float(enrichment.get("confidence") or 0) / 100.0 or 0.75,
                    "provenance": {"source": "guide_profile_enrich"},
                }
            )
        if claims:
            await conn.execute(
                """
                DELETE FROM guide_fact_claim
                WHERE entity_kind='guide'
                  AND entity_id=?
                  AND source_post_id IS NULL
                  AND claim_role='guide_profile_enrich'
                """,
                (profile_id,),
            )
            await _insert_entity_claims(
                conn,
                entity_kind="guide",
                entity_id=profile_id,
                source_post_id=None,
                claims=claims,
                replace_source_post_claims=False,
            )
        touched += 1
    return touched


async def _refresh_template_rollups(
    conn: aiosqlite.Connection,
    *,
    template_ids: Sequence[int],
) -> None:
    normalized_ids = sorted({int(item) for item in template_ids if int(item) > 0})
    for template_id in normalized_ids:
        template_cur = await conn.execute(
            "SELECT facts_rollup_json FROM guide_template WHERE id=? LIMIT 1",
            (int(template_id),),
        )
        template_row = await template_cur.fetchone()
        existing_rollup = _safe_json_object(template_row["facts_rollup_json"] if template_row else None)
        cur = await conn.execute(
            """
            SELECT
                city,
                duration_text,
                meeting_point,
                guide_names_json,
                organizer_names_json,
                audience_fit_json,
                fact_pack_json
            FROM guide_occurrence
            WHERE template_id=?
            ORDER BY date ASC, COALESCE(time, '99:99') ASC, id ASC
            """,
            (int(template_id),),
        )
        rows = await cur.fetchall()
        cities: list[str] = []
        durations: list[str] = []
        meeting_points: list[str] = []
        routes: list[str] = []
        guide_names: list[str] = []
        organizers: list[str] = []
        audience_fit: list[str] = []
        main_hooks: list[str] = []
        region_labels: list[str] = []
        for row in rows:
            fact_pack = _safe_json_object(row["fact_pack_json"])
            cities.extend(_unique_limited([row["city"], fact_pack.get("city")], limit=2))
            durations.extend(_unique_limited([row["duration_text"], fact_pack.get("duration_text")], limit=2))
            meeting_points.extend(_unique_limited([row["meeting_point"], fact_pack.get("meeting_point")], limit=2))
            routes.extend(_unique_limited([fact_pack.get("route_summary")], limit=2))
            guide_names.extend(_normalize_string_list(_parse_json_array(row["guide_names_json"]), limit=6))
            guide_names.extend(_normalize_string_list(fact_pack.get("guide_names"), limit=6))
            organizers.extend(_normalize_string_list(_parse_json_array(row["organizer_names_json"]), limit=6))
            organizers.extend(_normalize_string_list(fact_pack.get("organizer_names"), limit=6))
            audience_fit.extend(_normalize_string_list(_parse_json_array(row["audience_fit_json"]), limit=6))
            audience_fit.extend(_normalize_string_list(fact_pack.get("audience_fit"), limit=6))
            main_hooks.extend(_unique_limited([fact_pack.get("main_hook")], limit=1))
            region_labels.extend(_unique_limited([fact_pack.get("audience_region_fit_label")], limit=1))
        facts_rollup = {
            "cities": _unique_limited(cities, limit=6),
            "durations": _unique_limited(durations, limit=6),
            "meeting_points": _unique_limited(meeting_points, limit=6),
            "route_summaries": _unique_limited(routes, limit=6),
            "guide_names": _unique_limited(guide_names, limit=8),
            "organizer_names": _unique_limited(organizers, limit=8),
            "audience_fit": _unique_limited(audience_fit, limit=8),
            "main_hooks": _unique_limited(main_hooks, limit=6),
            "audience_region_fit_counts": _count_labels(region_labels),
        }
        facts_rollup = {key: value for key, value in facts_rollup.items() if value not in (None, "", [], {})}
        facts_rollup = _merge_json_objects(existing_rollup, facts_rollup)
        base_city = next(iter(facts_rollup.get("cities") or []), None)
        await conn.execute(
            """
            UPDATE guide_template
            SET
                base_city=COALESCE(NULLIF(?, ''), base_city),
                facts_rollup_json=?,
                last_seen_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                collapse_ws(base_city) or None,
                _json_dump(facts_rollup) if facts_rollup else None,
                int(template_id),
            ),
        )


async def _update_profile_from_hint(
    conn: aiosqlite.Connection,
    *,
    profile_id: int | None,
    source_row: Mapping[str, Any],
    profile_hint: Mapping[str, Any] | None,
    source_post_id: int | None,
) -> bool:
    if profile_id is None:
        return False
    hint = dict(profile_hint or {})
    existing_cur = await conn.execute(
        "SELECT source_links_json, audience_strengths_json, facts_rollup_json, summary_short FROM guide_profile WHERE id=?",
        (int(profile_id),),
    )
    existing = await existing_cur.fetchone()
    source_links = _normalize_string_list(_safe_json_list((existing["source_links_json"] if existing else None)), limit=12)
    source_links.extend(_normalize_string_list(hint.get("source_links"), limit=12))
    source_links = _normalize_string_list(source_links, limit=12)
    audience_strengths = _normalize_string_list(_safe_json_list((existing["audience_strengths_json"] if existing else None)), limit=10)
    audience_strengths.extend(_normalize_string_list(hint.get("audience_strengths"), limit=10))
    audience_strengths = _normalize_string_list(audience_strengths, limit=10)
    facts_rollup = _merge_json_objects(
        _safe_json_object(existing["facts_rollup_json"] if existing else None),
        _safe_json_object(hint.get("facts_rollup")),
    )
    summary_short = collapse_ws(hint.get("summary_short")) or collapse_ws(existing["summary_short"] if existing else None)
    await conn.execute(
        """
        UPDATE guide_profile
        SET
            display_name=COALESCE(NULLIF(?, ''), display_name),
            marketing_name=COALESCE(NULLIF(?, ''), marketing_name),
            source_links_json=?,
            audience_strengths_json=?,
            summary_short=COALESCE(NULLIF(?, ''), summary_short),
            facts_rollup_json=?,
            last_seen_at=CURRENT_TIMESTAMP
        WHERE id=?
        """,
        (
                collapse_ws(hint.get("display_name")) or collapse_ws(_mapping_value(source_row, "display_name")),
                collapse_ws(hint.get("marketing_name")) or collapse_ws(_mapping_value(source_row, "marketing_name")),
            _json_dump(source_links) if source_links else None,
            _json_dump(audience_strengths) if audience_strengths else None,
            summary_short,
            _json_dump(facts_rollup) if facts_rollup else None,
            int(profile_id),
        ),
    )
    claims: list[dict[str, Any]] = []
    if summary_short:
        claims.append(
            {
                "fact_key": "summary_short",
                "fact_value": summary_short,
                "claim_role": "guide_profile_hint",
                "confidence": 0.7,
                "provenance": {"source": "profile_hint"},
            }
        )
    for item in audience_strengths[:6]:
        claims.append(
            {
                "fact_key": "audience_strength",
                "fact_value": item,
                "claim_role": "guide_profile_hint",
                "confidence": 0.7,
                "provenance": {"source": "profile_hint"},
            }
        )
    for link in source_links[:6]:
        claims.append(
            {
                "fact_key": "source_link",
                "fact_value": link,
                "claim_role": "guide_profile_hint",
                "confidence": 0.7,
                "provenance": {"source": "profile_hint"},
            }
        )
    if claims:
        await _insert_entity_claims(
            conn,
            entity_kind="guide",
            entity_id=int(profile_id),
            source_post_id=source_post_id,
            claims=claims,
        )
    return True


async def _ensure_template(
    conn: aiosqlite.Connection,
    *,
    profile_id: int | None,
    parsed: GuideParsedOccurrence,
    template_hint: Mapping[str, Any] | None = None,
) -> int | None:
    if profile_id is None:
        return None
    cur = await conn.execute(
        "SELECT id FROM guide_template WHERE profile_id=? AND title_normalized=?",
        (int(profile_id), parsed.title_normalized),
    )
    row = await cur.fetchone()
    hint = dict(template_hint or {})
    aliases = _normalize_string_list([parsed.canonical_title, *_normalize_string_list(hint.get("aliases"), limit=8)], limit=8)
    audience_values = _normalize_string_list([*parsed.audience_fit, *_normalize_string_list(hint.get("audience_fit"), limit=8)], limit=8)
    participant_profiles = _normalize_string_list([*parsed.guide_names, *_normalize_string_list(hint.get("participant_profiles"), limit=8)], limit=8)
    aliases_json = _json_dump(aliases)
    audience_json = _json_dump(audience_values)
    guide_names_json = _json_dump(participant_profiles)
    facts_rollup = _merge_json_objects({}, _safe_json_object(hint.get("facts_rollup")))
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
                facts_rollup_json=CASE
                    WHEN ? IS NULL OR ? = '' THEN facts_rollup_json
                    ELSE ?
                END,
                last_seen_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                collapse_ws(hint.get("canonical_title")) or parsed.canonical_title,
                aliases_json,
                collapse_ws(hint.get("availability_mode")) or parsed.availability_mode,
                audience_json,
                guide_names_json,
                collapse_ws(hint.get("summary_short")) or parsed.summary_one_liner,
                _json_dump(facts_rollup) if facts_rollup else None,
                _json_dump(facts_rollup) if facts_rollup else None,
                _json_dump(facts_rollup) if facts_rollup else None,
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
            facts_rollup_json,
            first_seen_at,
            last_seen_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            int(profile_id),
            collapse_ws(hint.get("canonical_title")) or parsed.canonical_title,
            parsed.title_normalized,
            aliases_json,
            collapse_ws(hint.get("base_city")) or parsed.city,
            collapse_ws(hint.get("availability_mode")) or parsed.availability_mode,
            audience_json,
            guide_names_json,
            collapse_ws(hint.get("summary_short")) or parsed.summary_one_liner,
            _json_dump(facts_rollup) if facts_rollup else None,
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
    fact_pack: Mapping[str, Any] | None = None,
    fact_claims: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    claims = _normalize_claim_payloads(fact_claims)
    if not claims:
        claims = _derive_claims_from_fact_pack(
            fact_pack
            or {
                "canonical_title": parsed.canonical_title,
                "title_normalized": parsed.title_normalized,
                "date": parsed.date_iso,
                "time": parsed.time_text,
                "duration_text": parsed.duration_text,
                "meeting_point": parsed.meeting_point,
                "route_summary": parsed.route_summary,
                "price_text": parsed.price_text,
                "booking_text": parsed.booking_text,
                "booking_url": parsed.booking_url,
                "status": parsed.status,
                "seats_text": parsed.seats_text,
                "summary_one_liner": parsed.summary_one_liner,
                "audience_fit": list(parsed.audience_fit),
                "group_format": parsed.group_format,
                "guide_names": list(parsed.guide_names),
                "organizer_names": list(parsed.organizer_names),
            }
        )
    await _insert_entity_claims(
        conn,
        entity_kind="occurrence",
        entity_id=int(occurrence_id),
        source_post_id=int(post_id),
        claims=claims,
    )


async def _upsert_occurrence(
    conn: aiosqlite.Connection,
    *,
    source_row: Mapping[str, Any],
    post_id: int,
    post: GuideScannedPost,
    parsed: GuideParsedOccurrence,
    template_id: int | None,
    fact_pack: Mapping[str, Any] | None = None,
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
    fact_pack_json = _json_dump(dict(fact_pack or {})) if fact_pack else None
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
                duration_text=COALESCE(NULLIF(?, ''), duration_text),
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
                fact_pack_json=CASE
                    WHEN ? IS NULL OR ? = '' THEN fact_pack_json
                    ELSE ?
                END,
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
                parsed.duration_text,
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
                fact_pack_json,
                fact_pack_json,
                fact_pack_json,
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
                duration_text,
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
                fact_pack_json,
                views,
                likes,
                first_seen_at,
                updated_at,
                last_seen_post_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
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
                parsed.duration_text,
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
                fact_pack_json,
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
    await _insert_occurrence_claims(
        conn,
        occurrence_id=occurrence_id,
        post_id=post_id,
        parsed=parsed,
        fact_pack=fact_pack,
    )
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
) -> tuple[dict[str, int], dict[str, Any]]:
    metrics = {
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
        "profiles_touched": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
    }
    username = str(source_row["username"])
    source_report: dict[str, Any] = {
        "username": username,
        "source_title": None,
        "source_kind": str(source_row["source_kind"] or ""),
        "source_status": "ok",
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
        "errors": [],
        "posts": [],
        "occurrence_changes": [],
    }
    source_meta, posts = await scan_source_posts(client, username=username, limit=limit, days_back=days_back)
    enrich_targets: list[dict[str, Any]] = []
    template_ids_seen: set[int] = set()
    await _update_source_runtime_meta(
        conn,
        source_id=int(source_row["id"]),
        title=source_meta.source_title,
        about_text=source_meta.about_text,
        about_links=source_meta.about_links,
        last_scanned_message_id=max((post.message_id for post in posts), default=None),
    )
    source_context = dict(source_row)
    source_context["title"] = source_meta.source_title
    source_context["about_text"] = source_meta.about_text
    source_context["about_links_json"] = _json_dump(source_meta.about_links) if source_meta.about_links else None
    source_title = source_meta.source_title
    source_report["source_title"] = source_title
    fallback_name = collapse_ws(str(source_row["display_name"] or source_title or username))
    profile_targets: dict[int, dict[str, Any]] = {}
    for post in posts:
        metrics["posts_scanned"] += 1
        source_report["posts_scanned"] += 1
        prefilter = _should_prefilter(post, str(source_row["source_kind"] or ""))
        post_report: dict[str, Any] = {
            "message_id": int(post.message_id),
            "source_url": post.source_url,
            "post_kind": "mixed_or_non_target",
            "prefilter_passed": bool(prefilter),
            "llm_status": "local_parser",
            "occurrences": [],
            "errors": [],
        }
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
        source_report["posts_prefiltered"] += 1
        occurrences = parse_post_occurrences(
            text=post.text,
            post_date=post.post_date,
            source_kind=str(source_row["source_kind"] or ""),
            source_title=source_title or collapse_ws(str(source_row["marketing_name"] or source_row["display_name"] or "")),
            channel_url=post.source_url,
            fallback_guide_name=fallback_name,
        )
        if not occurrences:
            source_report["posts"].append(post_report)
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
        post_report["post_kind"] = occurrences[0].post_kind
        for parsed in occurrences:
            if _is_past_occurrence(parsed.date_iso):
                metrics["past_occurrences_skipped"] += 1
                source_report["past_occurrences_skipped"] += 1
                post_report["occurrences"].append(
                    {
                        "action": "past_skipped",
                        "canonical_title": parsed.canonical_title,
                        "date": parsed.date_iso,
                        "time": parsed.time_text,
                    }
                )
                continue
            profile_id = int(source_row["primary_profile_id"]) if source_row["primary_profile_id"] is not None else None
            if await _update_profile_from_hint(
                conn,
                profile_id=profile_id,
                source_row=source_context,
                profile_hint=None,
                source_post_id=post_id,
            ):
                metrics["profiles_touched"] += 1
            _upsert_profile_enrich_target(
                profile_targets,
                profile_id=profile_id,
                source_row=source_context,
                sample_title=parsed.canonical_title,
                sample_summary=parsed.summary_one_liner,
                guide_names=list(parsed.guide_names),
            )
            template_id = await _ensure_template(
                conn,
                profile_id=profile_id,
                parsed=parsed,
            )
            if template_id is not None:
                metrics["templates_touched"] += 1
                template_ids_seen.add(int(template_id))
            fact_pack = _build_occurrence_fact_pack(parsed, post=post, source_row=source_row)
            occurrence_id, created = await _upsert_occurrence(
                conn,
                source_row=source_row,
                post_id=post_id,
                post=post,
                parsed=parsed,
                template_id=template_id,
                fact_pack=fact_pack,
            )
            if occurrence_id:
                enrich_targets.append(
                    {
                        "occurrence_id": int(occurrence_id),
                        "template_id": int(template_id) if template_id is not None else None,
                        "post_id": int(post_id),
                        "fact_pack": fact_pack,
                        "canonical_title": parsed.canonical_title,
                        "summary_one_liner": parsed.summary_one_liner,
                        "digest_blurb": parsed.digest_blurb,
                        "route_summary": parsed.route_summary,
                        "city": parsed.city,
                        "audience_fit": list(parsed.audience_fit),
                        "group_format": parsed.group_format,
                        "guide_names": list(parsed.guide_names),
                        "post_excerpt": post.text,
                    }
                )
                if created:
                    metrics["occurrences_created"] += 1
                    source_report["occurrences_created"] += 1
                else:
                    metrics["occurrences_updated"] += 1
                    source_report["occurrences_updated"] += 1
                change = _report_occurrence_entry(
                    occurrence_id=occurrence_id,
                    created=created,
                    parsed=parsed,
                    source_username=username,
                    source_post_url=post.source_url,
                )
                post_report["occurrences"].append(change)
                source_report["occurrence_changes"].append(change)
        if prefilter or post_report["occurrences"] or post_report["errors"]:
            source_report["posts"].append(post_report)
    if enrich_targets:
        try:
            enrichments = await apply_occurrence_enrichment(enrich_targets)
            touched_template_ids = await _apply_occurrence_enrichment_results(
                conn,
                enrich_targets=enrich_targets,
                enrichments=enrichments,
            )
            await _refresh_template_rollups(
                conn,
                template_ids=[*template_ids_seen, *touched_template_ids],
            )
        except Exception as exc:
            source_report["source_status"] = "partial"
            source_report["errors"].append(f"enrich_failed: {type(exc).__name__}: {exc}")
    if profile_targets:
        try:
            profile_enrichments = await apply_profile_enrichment(list(profile_targets.values()))
            metrics["profiles_touched"] += await _apply_profile_enrichment_results(
                conn,
                profile_targets=list(profile_targets.values()),
                enrichments=profile_enrichments,
            )
        except Exception as exc:
            source_report["source_status"] = "partial"
            source_report["errors"].append(f"profile_enrich_failed: {type(exc).__name__}: {exc}")
    return metrics, source_report


def _guide_post_from_payload(payload: Mapping[str, Any], *, source_username: str, source_title: str | None) -> GuideScannedPost:
    post_date_raw = collapse_ws(payload.get("post_date"))
    try:
        post_date = datetime.fromisoformat(post_date_raw.replace("Z", "+00:00")) if post_date_raw else _utc_now()
    except Exception:
        post_date = _utc_now()
    if post_date.tzinfo is None:
        post_date = post_date.replace(tzinfo=timezone.utc)
    return GuideScannedPost(
        source_username=source_username,
        source_title=source_title,
        message_id=int(payload.get("message_id") or 0),
        grouped_id=int(payload.get("grouped_id") or 0) or None,
        post_date=post_date.astimezone(timezone.utc),
        source_url=collapse_ws(payload.get("source_url")) or f"https://t.me/{source_username}/{int(payload.get('message_id') or 0)}",
        text=str(payload.get("text") or ""),
        views=int(payload["views"]) if payload.get("views") is not None else None,
        forwards=int(payload["forwards"]) if payload.get("forwards") is not None else None,
        reactions_total=int(payload["reactions_total"]) if payload.get("reactions_total") is not None else None,
        reactions_json=_safe_json_object(payload.get("reactions_json")) or None,
        media_refs=list(_safe_json_list(payload.get("media_refs"))),
    )


def _coerce_occurrence_from_payload(
    payload: Mapping[str, Any],
    *,
    post: GuideScannedPost,
    source_row: Mapping[str, Any],
) -> GuideParsedOccurrence | None:
    title = collapse_ws(payload.get("canonical_title")) or collapse_ws(payload.get("title"))
    title_normalized = collapse_ws(payload.get("title_normalized")) or normalize_title_key(title)
    if not title or not title_normalized:
        return None
    date_iso = collapse_ws(payload.get("date") or payload.get("date_iso")) or None
    time_text = collapse_ws(payload.get("time") or payload.get("time_text")) or None
    channel_url = _source_post_url(payload.get("channel_url"), post.source_url)
    digest_eligible, digest_reason = _normalize_digest_eligibility(
        date_iso=date_iso,
        availability_mode=collapse_ws(payload.get("availability_mode")) or "scheduled_public",
        status=collapse_ws(payload.get("status")) or "scheduled",
        time_text=time_text,
        city=collapse_ws(payload.get("city")) or None,
        meeting_point=collapse_ws(payload.get("meeting_point")) or None,
        route_summary=collapse_ws(payload.get("route_summary")) or None,
        price_text=collapse_ws(payload.get("price_text")) or None,
        booking_text=collapse_ws(payload.get("booking_text")) or None,
        booking_url=_booking_url_for_digest(payload.get("booking_url")),
        summary_one_liner=collapse_ws(payload.get("summary_one_liner")) or None,
        digest_blurb=collapse_ws(payload.get("digest_blurb")) or collapse_ws(payload.get("summary_one_liner")) or None,
        digest_eligible=bool(payload.get("digest_eligible")),
        digest_reason=collapse_ws(payload.get("digest_eligibility_reason")) or None,
    )
    return GuideParsedOccurrence(
        block_text=collapse_ws(payload.get("source_excerpt") or post.text),
        canonical_title=title,
        title_normalized=title_normalized,
        date_iso=date_iso,
        time_text=time_text,
        duration_text=collapse_ws(payload.get("duration_text")) or None,
        city=collapse_ws(payload.get("city")) or None,
        meeting_point=collapse_ws(payload.get("meeting_point")) or None,
        route_summary=collapse_ws(payload.get("route_summary")) or None,
        audience_fit=_normalize_string_list(payload.get("audience_fit"), limit=8),
        group_format=collapse_ws(payload.get("group_format")) or None,
        price_text=collapse_ws(payload.get("price_text")) or None,
        booking_text=collapse_ws(payload.get("booking_text")) or None,
        booking_url=_booking_url_for_digest(payload.get("booking_url")),
        channel_url=channel_url,
        status=collapse_ws(payload.get("status")) or "scheduled",
        seats_text=collapse_ws(payload.get("seats_text")) or None,
        summary_one_liner=collapse_ws(payload.get("summary_one_liner")) or None,
        digest_blurb=collapse_ws(payload.get("digest_blurb")) or collapse_ws(payload.get("summary_one_liner")) or None,
        digest_eligible=digest_eligible,
        digest_eligibility_reason=digest_reason,
        is_last_call=bool(payload.get("is_last_call")),
        post_kind=collapse_ws(payload.get("post_kind")) or "announce_single",
        availability_mode=collapse_ws(payload.get("availability_mode")) or "scheduled_public",
        guide_names=_normalize_string_list(
            payload.get("guide_names")
            or [_mapping_value(source_row, "display_name"), _mapping_value(source_row, "marketing_name")],
            limit=4,
        ),
        organizer_names=_normalize_string_list(payload.get("organizer_names"), limit=4),
        source_fingerprint=collapse_ws(payload.get("source_fingerprint"))
        or build_source_fingerprint(title_normalized=title_normalized, date_iso=date_iso, time_text=time_text),
    )


async def _import_source_payload(
    conn: aiosqlite.Connection,
    *,
    source_row: Mapping[str, Any],
    source_payload: Mapping[str, Any],
) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    metrics = {
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
        "profiles_touched": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
    }
    errors: list[str] = []
    posts = list(_safe_json_list(source_payload.get("posts") or source_payload.get("candidates")))
    username = str(source_row["username"] or "")
    source_title = collapse_ws(source_payload.get("source_title")) or collapse_ws(_mapping_value(source_row, "title")) or None
    source_report: dict[str, Any] = {
        "username": username,
        "source_title": source_title,
        "source_kind": str(source_row["source_kind"] or ""),
        "source_status": collapse_ws(source_payload.get("source_status")) or "ok",
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
        "errors": [],
        "posts": [],
        "occurrence_changes": [],
    }
    enrich_targets: list[dict[str, Any]] = []
    template_ids_seen: set[int] = set()
    profile_targets: dict[int, dict[str, Any]] = {}
    await _update_source_runtime_meta(
        conn,
        source_id=int(source_row["id"]),
        title=source_title,
        about_text=collapse_ws(source_payload.get("about_text")) or None,
        about_links=_normalize_string_list(source_payload.get("about_links"), limit=16),
        last_scanned_message_id=max((int(item.get("message_id") or 0) for item in posts), default=None) or None,
    )
    source_context = dict(source_row)
    source_context["title"] = source_title
    source_context["about_text"] = collapse_ws(source_payload.get("about_text")) or None
    source_context["about_links_json"] = _json_dump(_normalize_string_list(source_payload.get("about_links"), limit=16))
    for post_payload in posts:
        if not isinstance(post_payload, dict):
            continue
        metrics["posts_scanned"] += 1
        source_report["posts_scanned"] += 1
        try:
            post = _guide_post_from_payload(post_payload, source_username=username, source_title=source_title)
        except Exception as exc:
            errors.append(f"@{username}: invalid post payload: {exc}")
            continue
        screen = _safe_json_object(post_payload.get("screen"))
        post_kind = collapse_ws(screen.get("post_kind")) or collapse_ws(post_payload.get("post_kind")) or "mixed_or_non_target"
        prefilter_passed = bool(post_payload.get("prefilter_passed")) or str(screen.get("decision") or "").strip().lower() not in {"", "ignore"}
        llm_status = collapse_ws(post_payload.get("llm_status")) or "kaggle_imported"
        if llm_status == "ok":
            metrics["llm_ok"] += 1
            source_report["llm_ok"] += 1
        elif llm_status.startswith("llm_deferred"):
            metrics["llm_deferred"] += 1
            source_report["llm_deferred"] += 1
        elif prefilter_passed and llm_status not in {"kaggle_imported", "local_parser"}:
            metrics["llm_error"] += 1
            source_report["llm_error"] += 1
        post_id = await _upsert_monitor_post(
            conn,
            source_id=int(source_row["id"]),
            post=post,
            post_kind=post_kind,
            prefilter_passed=prefilter_passed,
            llm_status=llm_status,
        )
        occurrence_payloads = [item for item in _safe_json_list(post_payload.get("occurrences")) if isinstance(item, dict)]
        if prefilter_passed:
            metrics["posts_prefiltered"] += 1
            source_report["posts_prefiltered"] += 1
        post_report: dict[str, Any] = {
            "message_id": int(post.message_id),
            "source_url": post.source_url,
            "post_kind": post_kind,
            "prefilter_passed": bool(prefilter_passed),
            "llm_status": llm_status,
            "occurrences": [],
            "errors": [],
        }
        await conn.execute(
            "UPDATE guide_monitor_post SET post_kind=?, title_hint=?, raw_facts_json=?, last_scanned_at=CURRENT_TIMESTAMP WHERE id=?",
            (
                post_kind,
                collapse_ws((occurrence_payloads[0] if occurrence_payloads else {}).get("canonical_title")) or None,
                _json_dump(occurrence_payloads) if occurrence_payloads else None,
                int(post_id),
            ),
        )
        if not occurrence_payloads:
            if prefilter_passed or str(llm_status).startswith("error"):
                source_report["posts"].append(post_report)
            continue
        profile_id = int(source_row["primary_profile_id"]) if source_row["primary_profile_id"] is not None else None
        for occurrence_payload in occurrence_payloads:
            parsed = _coerce_occurrence_from_payload(occurrence_payload, post=post, source_row=source_row)
            if parsed is None:
                errors.append(f"@{username}/{post.message_id}: empty occurrence payload")
                post_report["errors"].append("empty_occurrence_payload")
                continue
            if _is_past_occurrence(parsed.date_iso):
                metrics["past_occurrences_skipped"] += 1
                source_report["past_occurrences_skipped"] += 1
                post_report["occurrences"].append(
                    {
                        "action": "past_skipped",
                        "canonical_title": parsed.canonical_title,
                        "date": parsed.date_iso,
                        "time": parsed.time_text,
                    }
                )
                continue
            profile_hint = _safe_json_object(occurrence_payload.get("profile_hint"))
            if await _update_profile_from_hint(
                conn,
                profile_id=profile_id,
                source_row=source_context,
                profile_hint=profile_hint,
                source_post_id=post_id,
            ):
                metrics["profiles_touched"] += 1
            _upsert_profile_enrich_target(
                profile_targets,
                profile_id=profile_id,
                source_row=source_context,
                sample_title=parsed.canonical_title,
                sample_summary=parsed.summary_one_liner,
                guide_names=list(parsed.guide_names),
            )
            template_hint = _safe_json_object(occurrence_payload.get("template_hint"))
            template_id = await _ensure_template(
                conn,
                profile_id=profile_id,
                parsed=parsed,
                template_hint=template_hint,
            )
            if template_id is not None:
                metrics["templates_touched"] += 1
                template_ids_seen.add(int(template_id))
                template_claims = list(_safe_json_list(occurrence_payload.get("template_claims")))
                if not template_claims:
                    template_claims = [
                        {
                            "fact_key": "canonical_title",
                            "fact_value": collapse_ws(template_hint.get("canonical_title")) or parsed.canonical_title,
                            "claim_role": "template_hint",
                            "confidence": 0.7,
                            "provenance": {"source": "template_hint"},
                        }
                    ]
                await _insert_entity_claims(
                    conn,
                    entity_kind="template",
                    entity_id=int(template_id),
                    source_post_id=int(post_id),
                    claims=template_claims,
                )
            fact_pack = _build_occurrence_fact_pack(
                parsed,
                post=post,
                source_row=source_row,
                extra=_safe_json_object(occurrence_payload.get("fact_pack")),
            )
            occurrence_claims = list(_safe_json_list(occurrence_payload.get("fact_claims")))
            occurrence_id, created = await _upsert_occurrence(
                conn,
                source_row=source_row,
                post_id=post_id,
                post=post,
                parsed=parsed,
                template_id=template_id,
                fact_pack=fact_pack,
            )
            if occurrence_id:
                enrich_targets.append(
                    {
                        "occurrence_id": int(occurrence_id),
                        "template_id": int(template_id) if template_id is not None else None,
                        "post_id": int(post_id),
                        "fact_pack": fact_pack,
                        "canonical_title": parsed.canonical_title,
                        "summary_one_liner": parsed.summary_one_liner,
                        "digest_blurb": parsed.digest_blurb,
                        "route_summary": parsed.route_summary,
                        "city": parsed.city,
                        "audience_fit": list(parsed.audience_fit),
                        "group_format": parsed.group_format,
                        "guide_names": list(parsed.guide_names),
                        "post_excerpt": post.text,
                    }
                )
                await _insert_occurrence_claims(
                    conn,
                    occurrence_id=occurrence_id,
                    post_id=post_id,
                    parsed=parsed,
                    fact_pack=fact_pack,
                    fact_claims=occurrence_claims,
                )
                if created:
                    metrics["occurrences_created"] += 1
                    source_report["occurrences_created"] += 1
                else:
                    metrics["occurrences_updated"] += 1
                    source_report["occurrences_updated"] += 1
                change = _report_occurrence_entry(
                    occurrence_id=occurrence_id,
                    created=created,
                    parsed=parsed,
                    source_username=username,
                    source_post_url=post.source_url,
                )
                post_report["occurrences"].append(change)
                source_report["occurrence_changes"].append(change)
        if prefilter_passed or str(llm_status).startswith("error") or post_report["occurrences"] or post_report["errors"]:
            source_report["posts"].append(post_report)
    if enrich_targets:
        try:
            enrichments = await apply_occurrence_enrichment(enrich_targets)
            touched_template_ids = await _apply_occurrence_enrichment_results(
                conn,
                enrich_targets=enrich_targets,
                enrichments=enrichments,
            )
            await _refresh_template_rollups(
                conn,
                template_ids=[*template_ids_seen, *touched_template_ids],
            )
        except Exception as exc:
            errors.append(f"@{username}: enrich_failed: {type(exc).__name__}: {exc}")
            source_report["errors"].append(f"enrich_failed: {type(exc).__name__}: {exc}")
            source_report["source_status"] = "partial"
    if profile_targets:
        try:
            profile_enrichments = await apply_profile_enrichment(list(profile_targets.values()))
            metrics["profiles_touched"] += await _apply_profile_enrichment_results(
                conn,
                profile_targets=list(profile_targets.values()),
                enrichments=profile_enrichments,
            )
        except Exception as exc:
            errors.append(f"@{username}: profile_enrich_failed: {type(exc).__name__}: {exc}")
            source_report["errors"].append(f"profile_enrich_failed: {type(exc).__name__}: {exc}")
            source_report["source_status"] = "partial"
    source_report["errors"] = list(errors[:20])
    if errors:
        source_report["source_status"] = "partial"
    return metrics, errors, source_report


async def _import_results_file(
    db: Database,
    *,
    results_path: str,
) -> tuple[dict[str, int], list[str], dict[str, Any]]:
    raw = Path(results_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("guide_excursions_results.json must contain JSON object")
    metrics = {
        "sources_scanned": 0,
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
        "profiles_touched": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
        "errors": 0,
    }
    errors: list[str] = []
    source_reports: list[dict[str, Any]] = []
    occurrence_changes: list[dict[str, Any]] = []
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        await seed_guide_sources(conn)
        await conn.commit()
        sources = await _get_enabled_sources(conn)
        source_by_username = {str(row["username"]): row for row in sources}
        for source_payload in _safe_json_list(data.get("sources")):
            if not isinstance(source_payload, dict):
                continue
            username = collapse_ws(source_payload.get("username"))
            if not username:
                continue
            source_row = source_by_username.get(username)
            if source_row is None:
                errors.append(f"@{username}: source not seeded")
                metrics["errors"] += 1
                continue
            source_status = collapse_ws(source_payload.get("source_status")) or "ok"
            if source_status not in {"ok", "partial", "completed"}:
                errors.extend(
                    f"@{username}: {collapse_ws(item)}"
                    for item in _normalize_string_list(source_payload.get("errors"), limit=6)
                )
                metrics["errors"] += 1
            source_metrics, source_errors, source_report = await _import_source_payload(
                conn,
                source_row=source_row,
                source_payload=source_payload,
            )
            metrics["sources_scanned"] += 1
            source_reports.append(source_report)
            occurrence_changes.extend(list(source_report.get("occurrence_changes") or []))
            for key, value in source_metrics.items():
                metrics[key] = metrics.get(key, 0) + int(value or 0)
            if source_errors:
                errors.extend(source_errors)
                metrics["errors"] += len(source_errors)
        await conn.commit()
    summary = {
        "run_id": collapse_ws(data.get("run_id")),
        "scan_mode": collapse_ws(data.get("scan_mode")),
        "partial": bool(data.get("partial")),
        "stats": _safe_json_object(data.get("stats")),
        "started_at": collapse_ws(data.get("started_at")),
        "finished_at": collapse_ws(data.get("finished_at")),
        "schema_version": int(data.get("schema_version") or 0),
        "sources": source_reports,
        "occurrence_changes": occurrence_changes,
    }
    return metrics, errors, summary


async def refresh_guide_profile_enrichment(
    db: Database,
    *,
    profile_ids: Sequence[int] | None = None,
) -> int:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        targets: dict[int, dict[str, Any]] = {}
        params: list[Any] = []
        where_sql = ""
        if profile_ids:
            normalized = [int(item) for item in profile_ids if int(item) > 0]
            if not normalized:
                return 0
            where_sql = f"WHERE gp.id IN ({','.join('?' for _ in normalized)})"
            params.extend(normalized)
        cur = await conn.execute(
            f"""
            SELECT
                gp.id AS profile_id,
                gp.display_name,
                gp.marketing_name,
                gs.username AS source_username,
                gs.title AS source_title,
                gs.about_text
            FROM guide_profile gp
            LEFT JOIN guide_source gs ON gs.primary_profile_id = gp.id
            {where_sql}
            ORDER BY gp.id ASC
            """,
            tuple(params),
        )
        rows = await cur.fetchall()
        for row in rows:
            profile_id = int(row["profile_id"] or 0)
            if profile_id <= 0:
                continue
            _upsert_profile_enrich_target(targets, profile_id=profile_id, source_row=row)
            occ_cur = await conn.execute(
                """
                SELECT canonical_title, summary_one_liner, guide_names_json, fact_pack_json
                FROM guide_occurrence
                WHERE primary_source_id IN (
                    SELECT id FROM guide_source WHERE primary_profile_id=?
                )
                ORDER BY date ASC, COALESCE(time, '99:99') ASC, updated_at DESC
                LIMIT 6
                """,
                (profile_id,),
            )
            occurrences = await occ_cur.fetchall()
            for occurrence in occurrences:
                fact_pack = _safe_json_object(occurrence["fact_pack_json"])
                _upsert_profile_enrich_target(
                    targets,
                    profile_id=profile_id,
                    source_row=row,
                    sample_title=collapse_ws(occurrence["canonical_title"]),
                    sample_summary=collapse_ws(occurrence["summary_one_liner"]),
                    sample_hook=collapse_ws(fact_pack.get("main_hook")),
                    guide_names=_normalize_string_list(_safe_json_list(occurrence["guide_names_json"]), limit=6),
                )
        if not targets:
            return 0
        profile_enrichments = await apply_profile_enrichment(list(targets.values()))
        touched = await _apply_profile_enrichment_results(
            conn,
            profile_targets=list(targets.values()),
            enrichments=profile_enrichments,
        )
        await conn.commit()
        return touched


def _guide_monitor_completion_lines(
    *,
    run_id: str,
    ops_run_id: int | None,
    metrics: Mapping[str, Any],
    errors: Sequence[str],
) -> list[str]:
    lines = [
        "✅ Мониторинг экскурсий завершён" if not errors else "⚠️ Мониторинг экскурсий завершён с ошибками",
        f"ops_run_id={ops_run_id or '—'}",
        f"run_id={run_id}",
        f"Источников: {int(metrics.get('sources_scanned') or 0)}",
        f"Постов: {int(metrics.get('posts_scanned') or 0)}",
        f"После prefilter: {int(metrics.get('posts_prefiltered') or 0)}",
        (
            "LLM ok/deferred/error: "
            f"{int(metrics.get('llm_ok') or 0)}/"
            f"{int(metrics.get('llm_deferred') or 0)}/"
            f"{int(metrics.get('llm_error') or 0)}"
        ),
        f"Новых выходов: {int(metrics.get('occurrences_created') or 0)}",
        f"Обновлений: {int(metrics.get('occurrences_updated') or 0)}",
        (
            "Профили/шаблоны: "
            f"{int(metrics.get('profiles_touched') or 0)}/"
            f"{int(metrics.get('templates_touched') or 0)}"
        ),
        f"Past skipped: {int(metrics.get('past_occurrences_skipped') or 0)}",
        f"Отчёт: /guide_report {ops_run_id}" if ops_run_id else "Отчёт недоступен",
    ]
    if errors:
        lines.append("")
        lines.append("Ошибки:")
        lines.extend(f"- {collapse_ws(err)}"[:350] for err in list(errors)[:5])
    return lines


async def run_guide_import_from_results(
    db: Database,
    *,
    results_path: str,
    bot: Bot | None,
    chat_id: int | None,
    run_id: str,
    trigger: str,
    operator_id: int | None = 0,
    mode: str = "full",
    transport: str = "kaggle_recovery",
    send_progress: bool = False,
    kaggle_meta: dict[str, Any] | None = None,
) -> GuideMonitorResult:
    started_monotonic = time.monotonic()
    metrics = {
        "sources_scanned": 0,
        "posts_scanned": 0,
        "posts_prefiltered": 0,
        "occurrences_created": 0,
        "occurrences_updated": 0,
        "templates_touched": 0,
        "profiles_touched": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
        "errors": 0,
        "duration_sec": 0,
    }
    errors: list[str] = []
    run_details: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "transport": transport,
        "results_path": str(results_path),
        "source_reports": [],
        "occurrence_changes": [],
    }
    if kaggle_meta:
        run_details["kaggle_meta"] = dict(kaggle_meta)

    ops_run_id = await start_ops_run(
        db,
        kind="guide_monitoring",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details=run_details,
    )
    try:
        import_metrics, import_errors, import_summary = await _import_results_file(
            db,
            results_path=str(results_path),
        )
        run_details["import_summary"] = dict(import_summary or {})
        run_details["source_reports"] = list(import_summary.get("sources") or [])
        run_details["occurrence_changes"] = list(import_summary.get("occurrence_changes") or [])
        for key, value in import_metrics.items():
            metrics[key] = metrics.get(key, 0) + int(value or 0)
        if import_errors:
            errors.extend(import_errors)
        if import_summary.get("partial"):
            errors.append("kaggle result marked as partial")
    except Exception as exc:
        logger.exception("guide_monitor.import_from_results_failed path=%s run_id=%s", results_path, run_id)
        metrics["errors"] += 1
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        metrics["duration_sec"] = int(max(0, round(time.monotonic() - started_monotonic)))

    status = "success" if not errors else ("partial" if metrics["sources_scanned"] > 0 else "error")
    run_details["errors"] = errors[:20]
    await finish_ops_run(db, run_id=ops_run_id, status=status, metrics=metrics, details=run_details)

    if send_progress and bot and chat_id:
        await bot.send_message(
            int(chat_id),
            "\n".join(
                _guide_monitor_completion_lines(
                    run_id=run_id,
                    ops_run_id=ops_run_id,
                    metrics=metrics,
                    errors=errors,
                )
            ),
            disable_web_page_preview=True,
        )

    return GuideMonitorResult(
        run_id=run_id,
        ops_run_id=ops_run_id,
        trigger=trigger,
        mode=mode,
        metrics=metrics,
        errors=errors,
    )


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
        "profiles_touched": 0,
        "past_occurrences_skipped": 0,
        "llm_ok": 0,
        "llm_deferred": 0,
        "llm_error": 0,
        "errors": 0,
        "duration_sec": 0,
    }
    errors: list[str] = []
    started_monotonic = time.monotonic()
    ops_run_id: int | None = None
    limit, days_back = await _resolve_scan_window(db, mode)
    run_details: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "limit": limit,
        "days_back": days_back,
        "transport": "kaggle" if GUIDE_EXCURSIONS_KAGGLE_ENABLED else "local",
        "source_reports": [],
        "occurrence_changes": [],
    }

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
                details=run_details,
            )
            if send_progress and bot and chat_id:
                await bot.send_message(
                    int(chat_id),
                    (
                        "🧭 Запускаю мониторинг экскурсий.\n"
                        f"transport={'kaggle' if GUIDE_EXCURSIONS_KAGGLE_ENABLED else 'local'}\n"
                        f"mode={html.escape(mode)}\n"
                        f"limit={limit}, days_back={days_back}\n"
                        f"run_id={html.escape(run_id)}"
                    ),
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )

            client = None
            try:
                if GUIDE_EXCURSIONS_KAGGLE_ENABLED:
                    kaggle_status_message_id: int | None = None

                    async def _update_kaggle_status(phase: str, kernel_ref: str, status: dict | None) -> None:
                        nonlocal kaggle_status_message_id
                        if not (send_progress and bot and chat_id):
                            return
                        text = format_kaggle_status_message(phase, kernel_ref, status)
                        try:
                            if kaggle_status_message_id is None:
                                sent = await bot.send_message(
                                    int(chat_id),
                                    text,
                                    disable_web_page_preview=True,
                                )
                                kaggle_status_message_id = getattr(sent, "message_id", None)
                            else:
                                await bot.edit_message_text(
                                    chat_id=int(chat_id),
                                    message_id=int(kaggle_status_message_id),
                                    text=text,
                                    disable_web_page_preview=True,
                                )
                        except Exception:
                            logger.exception("guide_monitor: failed to update kaggle status")

                    try:
                        results_path, kaggle_meta = await run_guide_monitor_kaggle(
                            db,
                            run_id=run_id,
                            mode=mode,
                            limit=limit,
                            days_back=days_back,
                            chat_id=chat_id,
                            status_callback=_update_kaggle_status if send_progress and bot and chat_id else None,
                        )
                        run_details["kaggle_meta"] = dict(kaggle_meta or {})
                        run_details["results_path"] = str(results_path)
                        import_metrics, import_errors, import_summary = await _import_results_file(
                            db,
                            results_path=str(results_path),
                        )
                        run_details["import_summary"] = dict(import_summary or {})
                        run_details["source_reports"] = list(import_summary.get("sources") or [])
                        run_details["occurrence_changes"] = list(import_summary.get("occurrence_changes") or [])
                        for key, value in import_metrics.items():
                            metrics[key] = metrics.get(key, 0) + int(value or 0)
                        if import_errors:
                            errors.extend(import_errors)
                        if import_summary.get("partial"):
                            errors.append("kaggle result marked as partial")
                    except Exception as exc:
                        logger.exception("guide_monitor: kaggle path failed")
                        errors.append(f"Kaggle path failed: {type(exc).__name__}: {exc}")
                        metrics["errors"] += 1
                        if not GUIDE_EXCURSIONS_LOCAL_FALLBACK_ENABLED:
                            raise
                        run_details["transport"] = "local_fallback"
                        if send_progress and bot and chat_id:
                            await bot.send_message(
                                int(chat_id),
                                "⚠️ Kaggle путь не завершился, перехожу на локальный fallback scan/import.",
                                disable_web_page_preview=True,
                            )
                if (not GUIDE_EXCURSIONS_KAGGLE_ENABLED) or (
                    GUIDE_EXCURSIONS_LOCAL_FALLBACK_ENABLED and metrics["sources_scanned"] == 0 and bool(errors)
                ):
                    if not GUIDE_EXCURSIONS_KAGGLE_ENABLED:
                        run_details["transport"] = "local"
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
                                source_metrics, source_report = await _scan_and_import_source(
                                    conn,
                                    client,
                                    source_row=source,
                                    limit=limit,
                                    days_back=days_back,
                                )
                                run_details["source_reports"].append(source_report)
                                run_details["occurrence_changes"].extend(list(source_report.get("occurrence_changes") or []))
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
    run_details["errors"] = errors[:20]
    await finish_ops_run(db, run_id=ops_run_id, status=status, metrics=metrics, details=run_details)

    result = GuideMonitorResult(
        run_id=run_id,
        ops_run_id=ops_run_id,
        trigger=trigger,
        mode=mode,
        metrics=metrics,
        errors=errors,
    )
    if send_progress and bot and chat_id:
        await bot.send_message(
            int(chat_id),
            "\n".join(
                _guide_monitor_completion_lines(
                    run_id=run_id,
                    ops_run_id=ops_run_id,
                    metrics=metrics,
                    errors=errors,
                )
            ),
            disable_web_page_preview=True,
        )
    return result


async def resume_guide_monitor_jobs(
    db: Database,
    bot: Bot | None,
    *,
    chat_id: int | None = None,
) -> int:
    jobs = await list_jobs("guide_monitoring")
    if not jobs:
        return 0

    notify_chat_id = chat_id
    if notify_chat_id is None:
        notify_chat_id = await resolve_superadmin_chat_id(db)

    client = KaggleClient()
    recovered = 0
    for job in jobs:
        kernel_ref = str(job.get("kernel_ref") or "").strip()
        if not kernel_ref or kernel_ref in _GUIDE_MONITOR_RECOVERY_ACTIVE:
            continue
        _GUIDE_MONITOR_RECOVERY_ACTIVE.add(kernel_ref)
        try:
            meta = job.get("meta") if isinstance(job.get("meta"), dict) else {}
            owner_pid = meta.get("pid")
            if owner_pid == os.getpid():
                continue

            run_id = str(meta.get("run_id") or "").strip() or uuid.uuid4().hex[:12]
            mode = collapse_ws(meta.get("mode")) or "full"
            try:
                status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            except Exception:
                logger.exception("guide_monitor_recovery.status_failed kernel=%s", kernel_ref)
                continue

            state = str(status.get("status") or "").lower()
            if state in {"error", "failed", "cancelled"}:
                failure = collapse_ws(extract_guide_failure_message(status))
                await remove_job("guide_monitoring", kernel_ref)
                if bot and notify_chat_id:
                    await bot.send_message(
                        int(notify_chat_id),
                        (
                            "⚠️ guide_monitor recovery: "
                            f"kernel {kernel_ref} завершился ошибкой"
                            f"{': ' + failure if failure else ''}"
                        ),
                        disable_web_page_preview=True,
                    )
                continue
            if state != "complete":
                continue

            results_path = await download_guide_results(client, kernel_ref, run_id)
            await run_guide_import_from_results(
                db,
                results_path=str(results_path),
                bot=bot,
                chat_id=notify_chat_id,
                run_id=run_id,
                trigger="recovery_import",
                operator_id=0,
                mode=mode,
                transport="kaggle_recovery",
                send_progress=bool(bot and notify_chat_id),
                kaggle_meta={
                    "kernel_ref": kernel_ref,
                    "status": "complete",
                    "status_data": status or {},
                },
            )
            await remove_job("guide_monitoring", kernel_ref)
            recovered += 1
            if bot and notify_chat_id:
                await bot.send_message(
                    int(notify_chat_id),
                    (
                        "✅ guide_monitor recovery: "
                        f"kernel {kernel_ref} обработан"
                    ),
                    disable_web_page_preview=True,
                )
        finally:
            _GUIDE_MONITOR_RECOVERY_ACTIVE.discard(kernel_ref)

    return recovered


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
            gs.priority_weight AS priority_weight,
            gp.display_name AS guide_profile_display_name,
            gp.marketing_name AS guide_profile_marketing_name,
            gp.summary_short AS guide_profile_summary,
            gp.facts_rollup_json AS guide_profile_facts_rollup_json
        FROM guide_occurrence go
        LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
        LEFT JOIN guide_profile gp ON gp.id = gs.primary_profile_id
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
        item["fact_pack"] = _safe_json_object(item.get("fact_pack_json"))
        for key in (
            "canonical_title",
            "date",
            "time",
            "duration_text",
            "city",
            "meeting_point",
            "route_summary",
            "price_text",
            "booking_text",
            "booking_url",
            "status",
            "seats_text",
            "summary_one_liner",
            "digest_blurb",
            "availability_mode",
            "post_kind",
            "group_format",
        ):
            if not item.get(key) and item["fact_pack"].get(key) is not None:
                item[key] = item["fact_pack"].get(key)
        if not item["guide_names"] and item["fact_pack"].get("guide_names"):
            item["guide_names"] = _normalize_string_list(item["fact_pack"].get("guide_names"), limit=4)
        if not item["organizer_names"] and item["fact_pack"].get("organizer_names"):
            item["organizer_names"] = _normalize_string_list(item["fact_pack"].get("organizer_names"), limit=4)
        if not item["audience_fit"] and item["fact_pack"].get("audience_fit"):
            item["audience_fit"] = _normalize_string_list(item["fact_pack"].get("audience_fit"), limit=8)
        item["guide_profile_facts"] = _safe_json_object(item.get("guide_profile_facts_rollup_json"))
        item["source_about_links"] = _parse_json_array(item.get("source_about_links_json"))
        item["priority_weight"] = float(item.get("priority_weight") or 1.0)
        item["aggregator_only"] = int(item.get("aggregator_only") or 0)
        item["views"] = int(item["views"]) if item.get("views") is not None else None
        item["likes"] = int(item["likes"]) if item.get("likes") is not None else None
        post_cur = await conn.execute(
            """
            SELECT gmp.text, gmp.source_url
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
        item["source_post_url"] = _source_post_url(
            item["fact_pack"].get("source_post_url") if isinstance(item["fact_pack"], dict) else None,
            (post_row["source_url"] if post_row else None) or item.get("channel_url"),
        )
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
        display_rows = await resolve_public_guide_names(display_rows)
        display_rows = await apply_digest_batch_copy(
            display_rows,
            family=family,
            date_formatter=format_date_time,
        )
        texts = build_digest_messages(display_rows, family=family)
        occurrence_ids = list(dict.fromkeys([*dedup.covered_occurrence_ids, *editorial_suppressed_ids]))
        media_items: list[dict[str, Any]] = []
        media_positions: dict[str, int] = {}
        for row in display_rows:
            if len(media_items) >= MAX_MEDIA_ITEMS:
                break
            cur = await conn.execute(
                """
                SELECT
                    gmp.id AS post_id,
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
            media_key = collapse_ws(media_row["source_url"]) or f"post:{int(media_row['post_id'])}"
            position = int(media_positions.get(media_key) or 0)
            if position >= len(refs):
                continue
            media_positions[media_key] = position + 1
            media_items.append(
                {
                    "occurrence_id": int(row["id"]),
                    "source_username": str(media_row["username"]),
                    "source_url": str(media_row["source_url"] or ""),
                    "media_ref": refs[position],
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
                f"UPDATE guide_occurrence SET {column}=? WHERE id IN ({marks_sql})",
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


def _guide_occurrence_action_markup(
    rows: Sequence[Mapping[str, Any]],
    *,
    page: int,
    total_pages: int,
) -> types.InlineKeyboardMarkup | None:
    keyboard: list[list[types.InlineKeyboardButton]] = []
    for row in rows:
        occurrence_id = int(_mapping_value(row, "id") or 0)
        if occurrence_id <= 0:
            continue
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text=f"❌ {occurrence_id}",
                    callback_data=f"guide:occdel:{occurrence_id}:{int(page)}",
                ),
                types.InlineKeyboardButton(
                    text=f"🧾 {occurrence_id}",
                    callback_data=f"guide:occfacts:{occurrence_id}",
                ),
                types.InlineKeyboardButton(
                    text=f"📎 {occurrence_id}",
                    callback_data=f"guide:occlog:{occurrence_id}",
                ),
            ]
        )
    nav: list[types.InlineKeyboardButton] = []
    if page > 1:
        nav.append(
            types.InlineKeyboardButton(
                text="◀",
                callback_data=f"guide:future:{int(page) - 1}",
            )
        )
    if total_pages > 1:
        nav.append(
            types.InlineKeyboardButton(
                text=f"{int(page)}/{int(total_pages)}",
                callback_data=f"guide:future:{int(page)}",
            )
        )
    if page < total_pages:
        nav.append(
            types.InlineKeyboardButton(
                text="▶",
                callback_data=f"guide:future:{int(page) + 1}",
            )
        )
    if nav:
        keyboard.append(nav)
    return types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None


def _guide_template_action_markup(
    rows: Sequence[Mapping[str, Any]],
    *,
    page: int,
    total_pages: int,
) -> types.InlineKeyboardMarkup | None:
    keyboard: list[list[types.InlineKeyboardButton]] = []
    for row in rows:
        template_id = int(_mapping_value(row, "id") or 0)
        if template_id <= 0:
            continue
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text=f"🧩 {template_id}",
                    callback_data=f"guide:tplshow:{template_id}",
                ),
                types.InlineKeyboardButton(
                    text=f"❌ {template_id}",
                    callback_data=f"guide:tpldel:{template_id}:{int(page)}",
                )
            ]
        )
    nav: list[types.InlineKeyboardButton] = []
    if page > 1:
        nav.append(
            types.InlineKeyboardButton(
                text="◀",
                callback_data=f"guide:templates:{int(page) - 1}",
            )
        )
    if total_pages > 1:
        nav.append(
            types.InlineKeyboardButton(
                text=f"{int(page)}/{int(total_pages)}",
                callback_data=f"guide:templates:{int(page)}",
            )
        )
    if page < total_pages:
        nav.append(
            types.InlineKeyboardButton(
                text="▶",
                callback_data=f"guide:templates:{int(page) + 1}",
            )
        )
    if nav:
        keyboard.append(nav)
    return types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None


async def build_guide_future_occurrences_message(
    db: Database,
    *,
    page: int = 1,
    page_size: int = GUIDE_OCCURRENCES_PAGE_SIZE,
) -> tuple[str, types.InlineKeyboardMarkup | None]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        total_count = await _fetch_sqlite_int(
            conn,
            """
            SELECT COUNT(*)
            FROM guide_occurrence
            WHERE date IS NOT NULL AND date >= ?
            """,
            (_today_iso(),),
        )
        current_page, total_pages, offset = _page_state(
            total_count=total_count,
            page=page,
            page_size=page_size,
        )
        cur = await conn.execute(
            """
            SELECT
                go.id,
                go.template_id,
                go.canonical_title,
                go.date,
                go.time,
                go.status,
                go.digest_eligible,
                go.first_seen_at,
                go.updated_at,
                go.meeting_point,
                go.price_text,
                gs.username AS source_username,
                gs.title AS source_title,
                (
                    SELECT gmp.source_url
                    FROM guide_occurrence_source gos
                    JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
                    WHERE gos.occurrence_id = go.id
                    ORDER BY
                        CASE gos.role
                            WHEN 'primary' THEN 0
                            WHEN 'aggregator' THEN 1
                            ELSE 2
                        END,
                        gmp.post_date DESC,
                        gmp.id DESC
                    LIMIT 1
                ) AS source_post_url
            FROM guide_occurrence go
            LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
            WHERE go.date IS NOT NULL AND go.date >= ?
            ORDER BY go.date ASC, COALESCE(go.time, '99:99') ASC, go.updated_at DESC, go.id DESC
            LIMIT ? OFFSET ?
            """,
            (_today_iso(), int(page_size), int(offset)),
        )
        rows = await cur.fetchall()

    lines = [
        "🗓 Будущие экскурсии гидов",
        f"Всего: {int(total_count)} | page {int(current_page)}/{int(total_pages)}",
        "Команды: /guide_facts <id> | /guide_log <id>",
        "",
    ]
    if not rows:
        lines.append("Будущих экскурсий пока нет.")
        return "\n".join(lines), None

    for row in rows:
        when = format_date_time(str(row["date"] or ""), str(row["time"] or "")) or collapse_ws(row["date"]) or "без даты"
        title = collapse_ws(row["canonical_title"]) or "Экскурсия"
        lines.append(f"{int(row['id'])}. {when} — {title}")
        meta = [
            f"status={collapse_ws(row['status']) or '—'}",
            f"digest={1 if row['digest_eligible'] else 0}",
        ]
        if row["template_id"] is not None:
            meta.append(f"tpl={int(row['template_id'])}")
        source_label = _format_source_post_label(row["source_username"], row["source_post_url"])
        if source_label != "—":
            meta.append(source_label)
        lines.append("   " + " | ".join(meta))
        source_post_url = collapse_ws(row["source_post_url"])
        if source_post_url:
            lines.append(f"   🔗 {source_post_url}")
        created = _format_compact_utc(row["first_seen_at"])
        updated = _format_compact_utc(row["updated_at"])
        lines.append(f"   created={created} | updated={updated}")
        meeting_point = _short_text(row["meeting_point"], limit=90)
        if meeting_point:
            lines.append(f"   📍 {meeting_point}")
        price_text = _short_text(row["price_text"], limit=60)
        if price_text:
            lines.append(f"   💸 {price_text}")
        lines.append("")
    return "\n".join(lines).strip(), _guide_occurrence_action_markup(rows, page=current_page, total_pages=total_pages)


async def build_guide_templates_message(
    db: Database,
    *,
    page: int = 1,
    page_size: int = GUIDE_TEMPLATES_PAGE_SIZE,
) -> tuple[str, types.InlineKeyboardMarkup | None]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        total_count = await _fetch_sqlite_int(conn, "SELECT COUNT(*) FROM guide_template", ())
        current_page, total_pages, offset = _page_state(
            total_count=total_count,
            page=page,
            page_size=page_size,
        )
        cur = await conn.execute(
            """
            SELECT
                gt.id,
                gt.profile_id,
                gt.canonical_title,
                gt.availability_mode,
                gt.summary_short,
                gt.last_seen_at,
                gp.display_name AS profile_display_name,
                (
                    SELECT COUNT(*)
                    FROM guide_occurrence go
                    WHERE go.template_id = gt.id
                ) AS occurrences_total,
                (
                    SELECT COUNT(*)
                    FROM guide_occurrence go
                    WHERE go.template_id = gt.id
                      AND go.date IS NOT NULL
                      AND go.date >= ?
                ) AS future_occurrences
            FROM guide_template gt
            LEFT JOIN guide_profile gp ON gp.id = gt.profile_id
            ORDER BY gt.last_seen_at DESC, gt.id DESC
            LIMIT ? OFFSET ?
            """,
            (_today_iso(), int(page_size), int(offset)),
        )
        rows = await cur.fetchall()

    lines = [
        "🧩 Типовые экскурсии",
        f"Всего шаблонов: {int(total_count)} | page {int(current_page)}/{int(total_pages)}",
        "Команды: /guide_template <id> | /guide_templates [page]",
        "",
    ]
    if not rows:
        lines.append("Шаблонов пока нет.")
        return "\n".join(lines), None

    for row in rows:
        lines.append(f"{int(row['id'])}. {collapse_ws(row['canonical_title']) or 'Шаблон экскурсии'}")
        meta: list[str] = []
        profile_name = collapse_ws(row["profile_display_name"])
        if profile_name:
            meta.append(f"profile={profile_name}")
        availability = collapse_ws(row["availability_mode"])
        if availability:
            meta.append(f"availability={availability}")
        meta.append(f"future={int(row['future_occurrences'] or 0)}")
        meta.append(f"all={int(row['occurrences_total'] or 0)}")
        lines.append("   " + " | ".join(meta))
        lines.append(f"   last_seen={_format_compact_utc(row['last_seen_at'])}")
        summary = _short_text(row["summary_short"], limit=140)
        if summary:
            lines.append(f"   {summary}")
        lines.append("")
    return "\n".join(lines).strip(), _guide_template_action_markup(rows, page=current_page, total_pages=total_pages)


def _format_audience_region_counts(counts: Mapping[str, Any]) -> str | None:
    locals_count = int(counts.get("locals") or 0)
    tourists_count = int(counts.get("tourists") or 0)
    mixed_count = int(counts.get("mixed") or 0)
    parts: list[str] = []
    if locals_count:
        parts.append(f"locals={locals_count}")
    if tourists_count:
        parts.append(f"tourists={tourists_count}")
    if mixed_count:
        parts.append(f"mixed={mixed_count}")
    return ", ".join(parts) or None


async def render_guide_template_detail(
    db: Database,
    *,
    template_id: int,
) -> str:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            """
            SELECT
                gt.*,
                gp.display_name AS profile_display_name,
                gp.marketing_name AS profile_marketing_name
            FROM guide_template gt
            LEFT JOIN guide_profile gp ON gp.id = gt.profile_id
            WHERE gt.id=?
            LIMIT 1
            """,
            (int(template_id),),
        )
        row = await cur.fetchone()
        if not row:
            return f"Шаблон экскурсии #{int(template_id)} не найден."
        occ_cur = await conn.execute(
            """
            SELECT
                go.id,
                go.canonical_title,
                go.date,
                go.time,
                gs.username AS source_username,
                (
                    SELECT gmp.source_url
                    FROM guide_occurrence_source gos
                    JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
                    WHERE gos.occurrence_id = go.id
                    ORDER BY CASE gos.role WHEN 'primary' THEN 0 ELSE 1 END, gmp.post_date DESC, gmp.id DESC
                    LIMIT 1
                ) AS source_post_url
            FROM guide_occurrence go
            LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
            WHERE go.template_id=?
            ORDER BY go.date DESC, COALESCE(go.time, '99:99') DESC, go.id DESC
            LIMIT 12
            """,
            (int(template_id),),
        )
        occurrence_rows = await occ_cur.fetchall()

    facts_rollup = _safe_json_object(row["facts_rollup_json"])
    aliases = _parse_json_array(row["aliases_json"])
    audience_fit = _parse_json_array(row["audience_fit_json"])
    participants = _parse_json_array(row["participant_profiles_json"])

    lines = [
        f"🧩 Guide template #{int(row['id'])}",
        f"Название: {collapse_ws(row['canonical_title']) or '—'}",
        (
            f"profile={collapse_ws(row['profile_display_name']) or collapse_ws(row['profile_marketing_name']) or '—'}"
            f" | availability={collapse_ws(row['availability_mode']) or '—'}"
            f" | base_city={collapse_ws(row['base_city']) or '—'}"
        ),
        (
            f"first_seen={_format_compact_utc(row['first_seen_at'])}"
            f" | last_seen={_format_compact_utc(row['last_seen_at'])}"
        ),
    ]
    summary_short = _short_text(row["summary_short"], limit=220)
    if summary_short:
        lines.append(f"Кратко: {summary_short}")
    if aliases:
        lines.append(f"Алиасы: {', '.join(aliases[:8])}")
    if participants:
        lines.append(f"Гиды/участники: {', '.join(participants[:8])}")
    if audience_fit:
        lines.append(f"Audience fit: {', '.join(audience_fit[:8])}")

    lines.append("")
    lines.append("Накопленные template facts:")
    if not facts_rollup:
        lines.append("- Пока только skeleton: summary/aliases/participants без насыщенного rollup.")
    else:
        cities = _parse_json_array(facts_rollup.get("cities"))
        durations = _parse_json_array(facts_rollup.get("durations"))
        routes = _parse_json_array(facts_rollup.get("route_summaries"))
        meeting_points = _parse_json_array(facts_rollup.get("meeting_points"))
        main_hooks = _parse_json_array(facts_rollup.get("main_hooks"))
        rollup_guides = _parse_json_array(facts_rollup.get("guide_names"))
        region_counts = _format_audience_region_counts(_safe_json_object(facts_rollup.get("audience_region_fit_counts")))
        route_anchor = collapse_ws(facts_rollup.get("route_anchor"))
        if cities:
            lines.append(f"- Города: {', '.join(cities[:6])}")
        if durations:
            lines.append(f"- Продолжительность: {', '.join(durations[:6])}")
        if route_anchor:
            lines.append(f"- Route anchor: {route_anchor}")
        if routes:
            lines.append(f"- Типовые маршруты: {', '.join(routes[:4])}")
        if meeting_points:
            lines.append(f"- Типовые места сбора: {', '.join(meeting_points[:4])}")
        if main_hooks:
            lines.append(f"- Главные фишки: {', '.join(main_hooks[:4])}")
        if rollup_guides:
            lines.append(f"- Гиды в rollup: {', '.join(rollup_guides[:6])}")
        if region_counts:
            lines.append(f"- Locals/tourists/mixed: {region_counts}")
        if len(lines) > 0 and lines[-1] == "Накопленные template facts:":
            lines.append("- Rollup есть, но пока в нём нет насыщенных полей.")

    lines.append("")
    lines.append("Связанные occurrences:")
    if not occurrence_rows:
        lines.append("- —")
    else:
        for occ in occurrence_rows:
            schedule = format_date_time(str(occ["date"] or ""), str(occ["time"] or "")) or collapse_ws(occ["date"]) or "без даты"
            label = _format_source_post_label(occ["source_username"], occ["source_post_url"])
            lines.append(
                f"- #{int(occ['id'])} {schedule} — {_short_text(occ['canonical_title'], limit=100)} [{label}]"
            )
    return "\n".join(lines)


async def delete_guide_occurrence(db: Database, occurrence_id: int) -> dict[str, Any]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            "SELECT id, canonical_title FROM guide_occurrence WHERE id=?",
            (int(occurrence_id),),
        )
        row = await cur.fetchone()
        if not row:
            return {"deleted": False, "reason": "not_found"}
        await conn.execute(
            "DELETE FROM guide_fact_claim WHERE entity_kind='occurrence' AND entity_id=?",
            (int(occurrence_id),),
        )
        await conn.execute(
            "DELETE FROM guide_occurrence_source WHERE occurrence_id=?",
            (int(occurrence_id),),
        )
        await conn.execute(
            "DELETE FROM guide_occurrence WHERE id=?",
            (int(occurrence_id),),
        )
        await conn.commit()
    return {
        "deleted": True,
        "occurrence_id": int(occurrence_id),
        "canonical_title": collapse_ws(row["canonical_title"]) or "Экскурсия",
    }


async def delete_guide_template(db: Database, template_id: int) -> dict[str, Any]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            "SELECT id, canonical_title FROM guide_template WHERE id=?",
            (int(template_id),),
        )
        row = await cur.fetchone()
        if not row:
            return {"deleted": False, "reason": "not_found"}
        await conn.execute(
            "DELETE FROM guide_fact_claim WHERE entity_kind='template' AND entity_id=?",
            (int(template_id),),
        )
        await conn.execute(
            "UPDATE guide_occurrence SET template_id=NULL WHERE template_id=?",
            (int(template_id),),
        )
        await conn.execute(
            "DELETE FROM guide_template WHERE id=?",
            (int(template_id),),
        )
        await conn.commit()
    return {
        "deleted": True,
        "template_id": int(template_id),
        "canonical_title": collapse_ws(row["canonical_title"]) or "Шаблон экскурсии",
    }


async def render_guide_recent_changes(
    db: Database,
    *,
    hours: int = GUIDE_RECENT_CHANGES_DEFAULT_HOURS,
    limit: int = 20,
) -> list[str]:
    hours = max(1, min(int(hours), 720))
    limit = max(1, min(int(limit), 50))
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        created_cur = await conn.execute(
            """
            SELECT
                go.id,
                go.canonical_title,
                go.date,
                go.time,
                go.first_seen_at AS changed_at,
                gs.username AS source_username,
                (
                    SELECT gmp.source_url
                    FROM guide_occurrence_source gos
                    JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
                    WHERE gos.occurrence_id = go.id
                    ORDER BY
                        CASE gos.role
                            WHEN 'primary' THEN 0
                            WHEN 'aggregator' THEN 1
                            ELSE 2
                        END,
                        gmp.post_date DESC,
                        gmp.id DESC
                    LIMIT 1
                ) AS source_post_url
            FROM guide_occurrence go
            LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
            WHERE datetime(go.first_seen_at) >= datetime('now', ?)
              AND datetime(go.first_seen_at) < datetime('now')
            ORDER BY datetime(go.first_seen_at) DESC, go.id DESC
            LIMIT ?
            """,
            (f"-{int(hours)} hours", int(limit)),
        )
        created_rows = await created_cur.fetchall()
        updated_cur = await conn.execute(
            """
            SELECT
                go.id,
                go.canonical_title,
                go.date,
                go.time,
                go.updated_at AS changed_at,
                gs.username AS source_username,
                (
                    SELECT gmp.source_url
                    FROM guide_occurrence_source gos
                    JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
                    WHERE gos.occurrence_id = go.id
                    ORDER BY
                        CASE gos.role
                            WHEN 'primary' THEN 0
                            WHEN 'aggregator' THEN 1
                            ELSE 2
                        END,
                        gmp.post_date DESC,
                        gmp.id DESC
                    LIMIT 1
                ) AS source_post_url
            FROM guide_occurrence go
            LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
            WHERE datetime(go.updated_at) >= datetime('now', ?)
              AND datetime(go.updated_at) < datetime('now')
              AND datetime(go.first_seen_at) < datetime('now', ?)
            ORDER BY datetime(go.updated_at) DESC, go.id DESC
            LIMIT ?
            """,
            (f"-{int(hours)} hours", f"-{int(hours)} hours", int(limit)),
        )
        updated_rows = await updated_cur.fetchall()

    def _row_line(row: Mapping[str, Any], *, stamp_key: str) -> str:
        when = format_date_time(str(_mapping_value(row, "date") or ""), str(_mapping_value(row, "time") or "")) or collapse_ws(_mapping_value(row, "date")) or "без даты"
        label = _format_source_post_label(_mapping_value(row, "source_username"), _mapping_value(row, "source_post_url"))
        changed = _format_compact_utc(_mapping_value(row, stamp_key))
        tail = f" [{label}]" if label != "—" else ""
        return f"- #{int(_mapping_value(row, 'id') or 0)} {when} — {_short_text(_mapping_value(row, 'canonical_title') or 'Экскурсия', limit=110)}{tail} | {changed}"

    lines = [
        "🕘 Изменения экскурсий",
        f"Окно: последние {int(hours)} ч.",
        f"Новых: {len(created_rows)} | обновлённых: {len(updated_rows)}",
        "Команды: /guide_events | /guide_facts <id> | /guide_log <id>",
        "",
        "Новые:",
    ]
    if not created_rows:
        lines.append("- —")
    else:
        for row in created_rows:
            lines.append(_row_line(row, stamp_key="changed_at"))
    lines.append("")
    lines.append("Обновлённые:")
    if not updated_rows:
        lines.append("- —")
    else:
        for row in updated_rows:
            lines.append(_row_line(row, stamp_key="changed_at"))
    return _chunk_plain_lines(lines)


async def _fetch_guide_occurrence_snapshot(
    conn: aiosqlite.Connection,
    *,
    occurrence_id: int,
) -> tuple[aiosqlite.Row | None, list[aiosqlite.Row], list[aiosqlite.Row]]:
    cur = await conn.execute(
        """
        SELECT
            go.id,
            go.canonical_title,
            go.date,
            go.time,
            go.city,
            go.meeting_point,
            go.status,
            go.digest_eligible,
            go.digest_eligibility_reason,
            go.booking_text,
            go.booking_url,
            go.channel_url,
            go.summary_one_liner,
            go.digest_blurb,
            go.fact_pack_json,
            go.updated_at,
            go.first_seen_at,
            gs.username AS source_username,
            gs.title AS source_title
        FROM guide_occurrence go
        LEFT JOIN guide_source gs ON gs.id = go.primary_source_id
        WHERE go.id=?
        """,
        (int(occurrence_id),),
    )
    row = await cur.fetchone()
    if not row:
        return None, [], []

    posts_cur = await conn.execute(
        """
        SELECT
            gos.role,
            gos.created_at AS linked_at,
            gmp.id AS post_id,
            gmp.message_id,
            gmp.source_url,
            gmp.post_kind,
            gmp.llm_status,
            gmp.prefilter_passed,
            gmp.post_date,
            gmp.title_hint,
            gs.username AS source_username,
            gs.title AS source_title
        FROM guide_occurrence_source gos
        JOIN guide_monitor_post gmp ON gmp.id = gos.post_id
        JOIN guide_source gs ON gs.id = gmp.source_id
        WHERE gos.occurrence_id=?
        ORDER BY
            CASE gos.role
                WHEN 'primary' THEN 0
                WHEN 'aggregator' THEN 1
                ELSE 2
            END,
            gmp.post_date ASC,
            gmp.id ASC
        """,
        (int(occurrence_id),),
    )
    post_rows = await posts_cur.fetchall()

    claims_cur = await conn.execute(
        """
        SELECT
            gfc.id,
            gfc.fact_key,
            gfc.fact_value,
            gfc.claim_role,
            gfc.confidence,
            gfc.provenance_json,
            gfc.observed_at,
            gfc.last_confirmed_at,
            gmp.message_id,
            gmp.source_url,
            gs.username AS source_username,
            gs.title AS source_title
        FROM guide_fact_claim gfc
        LEFT JOIN guide_monitor_post gmp ON gmp.id = gfc.source_post_id
        LEFT JOIN guide_source gs ON gs.id = gmp.source_id
        WHERE gfc.entity_kind='occurrence' AND gfc.entity_id=?
        ORDER BY
            CASE gfc.claim_role
                WHEN 'anchor' THEN 0
                WHEN 'status_delta' THEN 1
                WHEN 'support' THEN 2
                ELSE 3
            END,
            gfc.observed_at ASC,
            gfc.id ASC
        """,
        (int(occurrence_id),),
    )
    claim_rows = await claims_cur.fetchall()
    return row, post_rows, claim_rows


async def render_guide_runs_summary(
    db: Database,
    *,
    hours: int = 48,
    limit: int = 12,
) -> list[str]:
    hours = max(1, min(int(hours), 720))
    limit = max(1, min(int(limit), 30))
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        cur = await conn.execute(
            """
            SELECT id, trigger, started_at, finished_at, status, metrics_json, details_json
            FROM ops_run
            WHERE kind='guide_monitoring'
              AND started_at >= datetime('now', ?)
            ORDER BY started_at DESC, id DESC
            LIMIT ?
            """,
            (f"-{int(hours)} hours", int(limit)),
        )
        rows = await cur.fetchall()

    lines = [
        "🧭 Recent guide monitoring runs",
        f"Окно: последние {hours} ч.",
    ]
    if not rows:
        lines.append("")
        lines.append("За это окно guide monitoring runs не найдено.")
        return _chunk_plain_lines(lines)

    lines.append("")
    for row in rows:
        metrics = _safe_json_object(row["metrics_json"])
        details = _safe_json_object(row["details_json"])
        icon = _status_icon(row["status"])
        started = _format_compact_utc(row["started_at"])
        transport = collapse_ws(details.get("transport")) or "—"
        mode = collapse_ws(details.get("mode")) or "—"
        lines.append(
            f"{icon} #{int(row['id'])} {started} | {collapse_ws(row['status']) or '—'} | {transport}/{mode}"
        )
        lines.append(
            "   "
            + ", ".join(
                [
                    f"sources={int(metrics.get('sources_scanned') or 0)}",
                    f"posts={int(metrics.get('posts_scanned') or 0)}",
                    f"llm_ok={int(metrics.get('llm_ok') or 0)}",
                    f"llm_def={int(metrics.get('llm_deferred') or 0)}",
                    f"created={int(metrics.get('occurrences_created') or 0)}",
                    f"updated={int(metrics.get('occurrences_updated') or 0)}",
                    f"errors={int(metrics.get('errors') or 0)}",
                ]
            )
        )
        lines.append(f"   /guide_report {int(row['id'])}")
    return _chunk_plain_lines(lines)


async def render_guide_run_report(
    db: Database,
    ops_run_id: int | None = None,
) -> list[str]:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        if ops_run_id is None:
            cur = await conn.execute(
                """
                SELECT *
                FROM ops_run
                WHERE kind='guide_monitoring'
                ORDER BY started_at DESC, id DESC
                LIMIT 1
                """
            )
        else:
            cur = await conn.execute(
                """
                SELECT *
                FROM ops_run
                WHERE id=? AND kind='guide_monitoring'
                LIMIT 1
                """,
                (int(ops_run_id),),
            )
        row = await cur.fetchone()

    if not row:
        target = f"#{int(ops_run_id)}" if ops_run_id is not None else "latest"
        return [f"Guide monitoring report {target} не найден."]

    metrics = _safe_json_object(row["metrics_json"])
    details = _safe_json_object(row["details_json"])
    source_reports = [item for item in _safe_json_list(details.get("source_reports")) if isinstance(item, dict)]
    occurrence_changes = [item for item in _safe_json_list(details.get("occurrence_changes")) if isinstance(item, dict)]
    kaggle_meta = _safe_json_object(details.get("kaggle_meta"))
    lines = [
        "🧭 Guide monitoring report",
        f"{_status_icon(row['status'])} ops_run_id={int(row['id'])} | status={collapse_ws(row['status']) or '—'} | trigger={collapse_ws(row['trigger']) or 'manual'}",
        (
            f"started={_format_compact_utc(row['started_at'])}"
            f" | finished={_format_compact_utc(row['finished_at']) if row['finished_at'] else '—'}"
        ),
        (
            f"transport={collapse_ws(details.get('transport')) or '—'}"
            f" | mode={collapse_ws(details.get('mode')) or '—'}"
            f" | run_id={collapse_ws(details.get('run_id')) or '—'}"
        ),
        (
            f"sources={int(metrics.get('sources_scanned') or 0)}, "
            f"posts={int(metrics.get('posts_scanned') or 0)}, "
            f"prefilter={int(metrics.get('posts_prefiltered') or 0)}, "
            f"llm_ok={int(metrics.get('llm_ok') or 0)}, "
            f"llm_deferred={int(metrics.get('llm_deferred') or 0)}, "
            f"llm_error={int(metrics.get('llm_error') or 0)}, "
            f"created={int(metrics.get('occurrences_created') or 0)}, "
            f"updated={int(metrics.get('occurrences_updated') or 0)}, "
            f"past_skipped={int(metrics.get('past_occurrences_skipped') or 0)}, "
            f"errors={int(metrics.get('errors') or 0)}"
        ),
        "Команды: /guide_runs 48 | /guide_report {id} | /guide_facts <occurrence_id> | /guide_log <occurrence_id>".format(
            id=int(row["id"])
        ),
    ]
    if kaggle_meta:
        lines.append(
            (
                f"Kaggle: {collapse_ws(kaggle_meta.get('kernel_ref')) or '—'}"
                f" | status={collapse_ws(kaggle_meta.get('status')) or '—'}"
                f" | duration={int(kaggle_meta.get('duration_sec') or 0)}s"
                f" | sources={int(kaggle_meta.get('sources_count') or 0)}"
            )
        )
    if occurrence_changes:
        lines.append("")
        lines.append("Изменённые карточки:")
        for change in occurrence_changes[:24]:
            lines.append(_format_occurrence_change_line(change))
        if len(occurrence_changes) > 24:
            lines.append(f"... ещё карточек: {len(occurrence_changes) - 24}")
    if source_reports:
        for source_report in source_reports[:12]:
            username = collapse_ws(source_report.get("username")).lstrip("@") or "—"
            title = collapse_ws(source_report.get("source_title"))
            source_kind = collapse_ws(source_report.get("source_kind")) or "—"
            source_status = collapse_ws(source_report.get("source_status")) or "—"
            errors = _normalize_string_list(source_report.get("errors"), limit=8)
            posts = [item for item in _safe_json_list(source_report.get("posts")) if isinstance(item, dict)]
            lines.append("")
            header = f"{_status_icon(source_status)} @{username}"
            if title:
                header += f" — {title}"
            lines.append(header)
            lines.append(
                (
                    f"kind={source_kind}, status={source_status}, "
                    f"posts={int(source_report.get('posts_scanned') or 0)}, "
                    f"prefilter={int(source_report.get('posts_prefiltered') or 0)}, "
                    f"llm_ok={int(source_report.get('llm_ok') or 0)}, "
                    f"llm_deferred={int(source_report.get('llm_deferred') or 0)}, "
                    f"llm_error={int(source_report.get('llm_error') or 0)}, "
                    f"created={int(source_report.get('occurrences_created') or 0)}, "
                    f"updated={int(source_report.get('occurrences_updated') or 0)}, "
                    f"past_skipped={int(source_report.get('past_occurrences_skipped') or 0)}"
                )
            )
            for post in posts[:10]:
                label = _format_source_post_label(username, post.get("source_url"))
                lines.append(
                    (
                        f"post {int(post.get('message_id') or 0)} | {label} | "
                        f"kind={collapse_ws(post.get('post_kind')) or '—'} | "
                        f"prefilter={1 if post.get('prefilter_passed') else 0} | "
                        f"llm={collapse_ws(post.get('llm_status')) or '—'}"
                    )
                )
                post_occurrences = [
                    item
                    for item in _safe_json_list(post.get("occurrences"))
                    if isinstance(item, dict)
                ]
                for change in post_occurrences[:6]:
                    lines.append("  " + _format_occurrence_change_line(change))
                post_errors = _normalize_string_list(post.get("errors"), limit=3)
                for error_text in post_errors:
                    lines.append(f"  error: {_short_text(error_text, limit=140)}")
            if len(posts) > 10:
                lines.append(f"... ещё постов у @{username}: {len(posts) - 10}")
            for error_text in errors[:4]:
                lines.append(f"source_error: {_short_text(error_text, limit=160)}")
        if len(source_reports) > 12:
            lines.append("")
            lines.append(f"... ещё источников: {len(source_reports) - 12}")
    detail_errors = _normalize_string_list(details.get("errors"), limit=10)
    if detail_errors:
        lines.append("")
        lines.append("Ошибки run:")
        for error_text in detail_errors:
            lines.append(f"- {_short_text(error_text, limit=180)}")
    return _chunk_plain_lines(lines)


async def render_guide_occurrence_log(db: Database, occurrence_id: int) -> str:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        row, post_rows, claim_rows = await _fetch_guide_occurrence_snapshot(conn, occurrence_id=int(occurrence_id))

    if not row:
        return f"Экскурсия #{int(occurrence_id)} не найдена."

    fact_pack = _safe_json_object(row["fact_pack_json"])
    lines = [
        f"🧾 Guide source log #{int(row['id'])}",
        f"Заголовок: {collapse_ws(row['canonical_title']) or '—'}",
        f"Дата/время: {format_date_time(str(row['date'] or ''), str(row['time'] or '')) or '—'}",
        f"Статус: {collapse_ws(row['status']) or '—'}",
        f"Primary source: {_primary_occurrence_source_label(row, post_rows, fact_pack=fact_pack)}",
        "",
        "Связанные посты:",
    ]
    if not post_rows:
        lines.append("- —")
    else:
        for post in post_rows[:30]:
            label = _format_source_post_label(post["source_username"], post["source_url"])
            linked_at = _format_compact_utc(post["post_date"] or post["linked_at"])
            lines.append(
                (
                    f"- {linked_at} | role={collapse_ws(post['role']) or 'source'} | {label}"
                    f" | kind={collapse_ws(post['post_kind']) or '—'}"
                    f" | llm={collapse_ws(post['llm_status']) or '—'}"
                    f" | prefilter={1 if post['prefilter_passed'] else 0}"
                )
            )
            title_hint = _short_text(post["title_hint"], limit=120)
            if title_hint:
                lines.append(f"  title_hint: {title_hint}")
        if len(post_rows) > 30:
            lines.append(f"... ещё связанных постов: {len(post_rows) - 30}")

    lines.append("")
    lines.append("Claim log:")
    if not claim_rows:
        lines.append("- —")
    else:
        for claim in claim_rows[:60]:
            observed = _format_compact_utc(claim["observed_at"])
            role = collapse_ws(claim["claim_role"]) or "support"
            conf = f"{float(claim['confidence']):.2f}" if claim["confidence"] is not None else "—"
            label = _format_source_post_label(claim["source_username"], claim["source_url"])
            provenance = _safe_json_object(claim["provenance_json"])
            refs = _normalize_string_list(provenance.get("fact_refs"), limit=4)
            line = f"- {observed} [{role}] {claim['fact_key']} = {claim['fact_value']} | conf={conf}"
            if label != "—":
                line += f" | {label}"
            if refs:
                line += f" | refs={','.join(refs)}"
            lines.append(line)
        if len(claim_rows) > 60:
            lines.append(f"... ещё claim rows: {len(claim_rows) - 60}")
    return "\n".join(lines)


async def render_guide_occurrence_facts(db: Database, occurrence_id: int) -> str:
    async with db.raw_conn() as conn:
        await _enable_row_factory(conn)
        row, post_rows, claim_rows = await _fetch_guide_occurrence_snapshot(conn, occurrence_id=int(occurrence_id))
    if not row:
        return f"Экскурсия #{int(occurrence_id)} не найдена."
    fact_pack = _safe_json_object(row["fact_pack_json"])
    lines = [
        f"🧾 Facts for guide occurrence #{int(row['id'])}",
        f"Заголовок: {collapse_ws(row['canonical_title']) or '—'}",
        f"Дата/время: {format_date_time(str(row['date'] or ''), str(row['time'] or '')) or '—'}",
        f"Статус: {collapse_ws(row['status']) or '—'}",
        f"Primary source: {_primary_occurrence_source_label(row, post_rows, fact_pack=fact_pack)}",
        (
            f"digest_eligible={1 if row['digest_eligible'] else 0}"
            f" | reason={collapse_ws(row['digest_eligibility_reason']) or '—'}"
        ),
        f"Проверка источников: /guide_log {int(row['id'])}",
        "",
        "Source posts:",
    ]
    if not post_rows:
        lines.append("- —")
    else:
        for post in post_rows[:12]:
            label = _format_source_post_label(post["source_username"], post["source_url"])
            lines.append(
                f"- role={collapse_ws(post['role']) or 'source'} | {label} | kind={collapse_ws(post['post_kind']) or '—'}"
            )
    lines.extend(
        [
            "",
        "Materialized fact pack:",
        ]
    )
    for key in sorted(fact_pack.keys()):
        value = fact_pack.get(key)
        normalized = _normalize_fact_value(value)
        if normalized:
            lines.append(f"- {key}: {normalized}")
    if not fact_pack:
        lines.append("- —")
    lines.append("")
    lines.append("Claims:")
    if not claim_rows:
        lines.append("- —")
    else:
        for claim in claim_rows[:40]:
            role = collapse_ws(claim["claim_role"]) or "support"
            conf = f"{float(claim['confidence']):.2f}" if claim["confidence"] is not None else "—"
            label = _format_source_post_label(claim["source_username"], claim["source_url"])
            observed = _format_compact_utc(claim["observed_at"])
            line = f"- {observed} [{role}] {claim['fact_key']} = {claim['fact_value']} (conf={conf})"
            if label != "—":
                line += f" | {label}"
            lines.append(line)
    return "\n".join(lines)
