from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from runtime import require_main_attr

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)

popular_posts_router = Router(name="popular_posts")

_MAX_TG_MESSAGE_LEN = 3800  # conservative: keep room for entities/HTML

_TG_POST_URL_RE = re.compile(r"(?:https?://)?t\.me/(?:s/)?([^/\s]+)/(\d+)", re.IGNORECASE)
_VK_WALL_PATH_RE = re.compile(r"^/wall-?(\d+)_([0-9]+)$", re.IGNORECASE)
_POPULAR_POSTS_THREE_DAY_MAX_AGE = 2
_POPULAR_POSTS_SEVEN_DAY_MAX_AGE = 6


def _utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _median_int(values: list[int]) -> int | None:
    if not values:
        return None
    data = sorted(int(v) for v in values)
    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        return int(data[mid])
    return int((data[mid - 1] + data[mid]) // 2)


def _chunk_lines(lines: list[str], *, max_len: int = _MAX_TG_MESSAGE_LEN) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line = str(line or "")
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
    return [c for c in chunks if c]


def _strip_url_query_fragment(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        return raw
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw.split("#", 1)[0].split("?", 1)[0].strip()
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _canonical_tg_post_url(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if "t.me/" not in raw.lower():
        return None
    m = _TG_POST_URL_RE.search(raw)
    if not m:
        return None
    from telegram_sources import normalize_tg_username

    username = normalize_tg_username(m.group(1))
    try:
        message_id = int(m.group(2))
    except Exception:
        message_id = 0
    if not username or message_id <= 0:
        return None
    return f"https://t.me/{username}/{message_id}"


def _canonical_vk_wall_url(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if "vk.com/" not in raw.lower():
        return None
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw.lstrip('/')}"
    try:
        parsed = urlsplit(raw)
    except Exception:
        return _strip_url_query_fragment(raw) or None
    host = (parsed.netloc or "").strip().lower()
    if host.startswith(("www.", "m.")):
        host = host.split(".", 1)[1]
    if host != "vk.com":
        return None
    m = _VK_WALL_PATH_RE.match(str(parsed.path or ""))
    if not m:
        return _strip_url_query_fragment(raw) or None
    try:
        group_id = int(m.group(1))
        post_id = int(m.group(2))
    except Exception:
        return _strip_url_query_fragment(raw) or None
    if group_id <= 0 or post_id <= 0:
        return _strip_url_query_fragment(raw) or None
    return f"https://vk.com/wall-{group_id}_{post_id}"


def _canonical_source_url_for_lookup(url: str | None) -> str | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if "t.me/" in low:
        return _canonical_tg_post_url(raw) or _strip_url_query_fragment(raw) or None
    if "vk.com/" in low:
        return _canonical_vk_wall_url(raw) or _strip_url_query_fragment(raw) or None
    return _strip_url_query_fragment(raw) or None


@dataclass(slots=True, frozen=True)
class _Baseline:
    median_views: int | None
    median_likes: int | None
    sample: int


@dataclass(slots=True)
class _PostItem:
    platform: str  # "tg" | "vk"
    source_key: int  # tg: source_id, vk: group_id
    source_label: str
    post_id: int
    post_url: str
    published_ts: int
    views: int | None
    likes: int | None
    baseline: _Baseline
    popularity: str
    score: float


@dataclass(slots=True, frozen=True)
class _EventRef:
    event_id: int
    title: str
    telegraph_url: str | None


@dataclass(slots=True, frozen=True)
class _EventLinks:
    events: tuple[_EventRef, ...]  # capped (see _resolve_telegraph_map)
    total: int


def _safe_ratio(value: int, denom: int | None) -> float:
    d = int(denom or 0)
    if d <= 0:
        d = 1
    return float(value) / float(d)


async def _require_superadmin(db: Database, user_id: int) -> bool:
    from models import User

    async with db.get_session() as session:
        user = await session.get(User, int(user_id))
        return bool(user and not user.blocked and user.is_superadmin)


async def _resolve_telegraph_map(db: Database, *, source_urls: list[str]) -> dict[str, _EventLinks]:
    """Map post_url -> event links bundle (Telegraph URL + title + event id).

    Returned `events` list is capped for UI readability. `total` preserves the true
    count of matching events for the given post.
    """
    from sqlalchemy import and_, or_, select
    from models import Event, EventSource

    urls = [
        str(u).strip()
        for u in (source_urls or [])
        if str(u or "").strip().startswith(("http://", "https://"))
    ]
    if not urls:
        return {}
    uniq_urls = list(dict.fromkeys(urls))

    def _event_telegraph_url(event: Any) -> str | None:
        url = getattr(event, "telegraph_url", None)
        if url and str(url).strip().startswith(("http://", "https://")):
            return str(url).strip()
        path = getattr(event, "telegraph_path", None)
        if path and str(path).strip():
            return f"https://telegra.ph/{str(path).strip().lstrip('/')}"
        return None

    # Normalize input URLs (strip ?single, /s/ etc.) so we can reliably match EventSource rows.
    orig_to_key: dict[str, str] = {}
    lookup_keys: list[str] = []
    tg_pairs: list[tuple[str, int]] = []
    for raw in uniq_urls:
        key = _canonical_source_url_for_lookup(raw) or ""
        if not key:
            continue
        orig_to_key[raw] = key
        if key not in lookup_keys:
            lookup_keys.append(key)
        if "t.me/" in key.lower():
            m = re.search(r"t\.me/([^/\s]+)/(\d+)", key, flags=re.IGNORECASE)
            if not m:
                continue
            uname = str(m.group(1) or "").strip().lower()
            try:
                mid = int(m.group(2))
            except Exception:
                continue
            if uname and mid > 0:
                tg_pairs.append((uname, mid))

    async with db.get_session() as session:
        key_to_event_ids: dict[str, list[int]] = {}
        event_ids: set[int] = set()

        def _add_mapping(source_url: str, event_id: int) -> None:
            su = str(source_url or "").strip()
            if not su:
                return
            key = _canonical_source_url_for_lookup(su) or ""
            if not key:
                return
            key_to_event_ids.setdefault(key, [])
            if int(event_id) not in key_to_event_ids[key]:
                key_to_event_ids[key].append(int(event_id))
            event_ids.add(int(event_id))

        # Primary: match by URL (canonical form).
        try:
            rows = (
                await session.execute(
                    select(EventSource.source_url, EventSource.event_id).where(
                        EventSource.source_url.in_(lookup_keys)
                    )
                )
            ).all()
            for source_url, eid in rows:
                if eid is None:
                    continue
                try:
                    ev_id = int(eid)
                except Exception:
                    continue
                _add_mapping(str(source_url or ""), ev_id)
        except Exception:
            logger.debug("popular_posts: failed to resolve event_source mapping by url", exc_info=True)

        # Fallback for TG when URL variants prevent direct matching.
        try:
            conds = [
                and_(
                    EventSource.source_chat_username == uname,
                    EventSource.source_message_id == int(mid),
                )
                for uname, mid in tg_pairs
            ]
            if conds:
                rows2 = (
                    await session.execute(
                        select(EventSource.source_url, EventSource.event_id).where(or_(*conds))
                    )
                ).all()
                for source_url, eid in rows2:
                    if eid is None:
                        continue
                    try:
                        ev_id = int(eid)
                    except Exception:
                        continue
                    _add_mapping(str(source_url or ""), ev_id)
        except Exception:
            logger.debug("popular_posts: failed to resolve event_source mapping by tg fields", exc_info=True)

        if not event_ids:
            return {}

        try:
            events = (
                await session.execute(select(Event).where(Event.id.in_(sorted(event_ids))))
            ).scalars().all()
        except Exception:
            logger.debug("popular_posts: failed to fetch events for telegraph map", exc_info=True)
            return {}

        id_to_ref: dict[int, _EventRef] = {}
        for ev in events:
            try:
                ev_id = int(getattr(ev, "id", 0) or 0)
            except Exception:
                continue
            title = str(getattr(ev, "title", "") or "").strip() or "событие"
            id_to_ref[ev_id] = _EventRef(
                event_id=ev_id,
                title=title,
                telegraph_url=_event_telegraph_url(ev),
            )

        out: dict[str, _EventLinks] = {}
        for raw in uniq_urls:
            key = orig_to_key.get(raw) or _canonical_source_url_for_lookup(raw) or ""
            if not key:
                continue
            eids = list(key_to_event_ids.get(key) or [])
            if not eids:
                continue
            refs: list[_EventRef] = []
            for ev_id in eids:
                ref = id_to_ref.get(int(ev_id))
                if not ref:
                    continue
                refs.append(ref)
                if len(refs) >= 3:
                    break
            out[raw] = _EventLinks(events=tuple(refs), total=int(len(eids)))
        return out


async def _table_exists(conn: Any, *, name: str) -> bool:
    try:
        cur = await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (str(name),),
        )
        row = await cur.fetchone()
        return bool(row)
    except Exception:
        return False


async def _load_top_items(
    db: Database,
    *,
    window_days: int,
    age_day: int,
    latest_age_day_max: int | None = None,
    limit: int,
) -> tuple[list[_PostItem], dict[str, Any]]:
    now_ts = _utc_now_ts()
    since_ts = now_ts - max(1, int(window_days)) * 86400
    min_sample = max(2, _env_int("POST_POPULARITY_MIN_SAMPLE", 2))
    from source_parsing.post_metrics import (
        PopularityBaseline,
        load_telegram_popularity_baseline,
        load_vk_popularity_baseline,
        popularity_marks,
    )

    tg_rows: list[tuple] = []
    vk_rows: list[tuple] = []
    tg_metrics_total = 0
    tg_metrics_sources = 0
    vk_metrics_total = 0
    vk_metrics_sources = 0
    async with db.raw_conn() as conn:
        tg_ready = all(
            [
                await _table_exists(conn, name="telegram_post_metric"),
                await _table_exists(conn, name="telegram_scanned_message"),
                await _table_exists(conn, name="telegram_source"),
            ]
        )
        if tg_ready:
            try:
                if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                    cur0 = await conn.execute(
                        """
                        SELECT
                            COUNT(*) AS posts,
                            COUNT(DISTINCT source_id) AS sources
                        FROM (
                            SELECT source_id, message_id
                            FROM telegram_post_metric
                            WHERE age_day <= ?
                              AND message_ts IS NOT NULL
                              AND message_ts >= ?
                            GROUP BY source_id, message_id
                        )
                        """,
                        (int(latest_age_day_max), int(since_ts)),
                    )
                else:
                    cur0 = await conn.execute(
                        """
                        SELECT
                            COUNT(*) AS posts,
                            COUNT(DISTINCT source_id) AS sources
                        FROM (
                            SELECT DISTINCT source_id, message_id
                            FROM telegram_post_metric
                            WHERE age_day = ?
                              AND message_ts IS NOT NULL
                              AND message_ts >= ?
                        )
                        """,
                        (int(age_day), int(since_ts)),
                    )
                row0 = await cur0.fetchone()
                if row0:
                    tg_metrics_total = int(row0[0] or 0)
                    tg_metrics_sources = int(row0[1] or 0)

                if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                    cur = await conn.execute(
                        """
                        WITH latest AS (
                            SELECT
                                source_id,
                                message_id,
                                MAX(age_day) AS max_age_day
                            FROM telegram_post_metric
                            WHERE age_day <= ?
                              AND message_ts IS NOT NULL
                              AND message_ts >= ?
                            GROUP BY source_id, message_id
                        )
                        SELECT
                            m.source_id,
                            m.message_id,
                            m.age_day,
                            COALESCE(m.source_url, '') AS source_url,
                            COALESCE(m.message_ts, 0) AS message_ts,
                            m.views,
                            m.likes,
                            COALESCE(s.events_imported, 0) AS events_imported,
                            COALESCE(t.username, '') AS username,
                            COALESCE(t.title, '') AS title
                        FROM telegram_post_metric m
                        JOIN latest l
                          ON l.source_id = m.source_id
                         AND l.message_id = m.message_id
                         AND l.max_age_day = m.age_day
                        JOIN telegram_scanned_message s
                          ON s.source_id = m.source_id
                         AND s.message_id = m.message_id
                        JOIN telegram_source t
                          ON t.id = m.source_id
                        WHERE COALESCE(s.events_imported, 0) > 0
                        """,
                        (int(latest_age_day_max), int(since_ts)),
                    )
                else:
                    cur = await conn.execute(
                        """
                        SELECT
                            m.source_id,
                            m.message_id,
                            m.age_day,
                            COALESCE(m.source_url, '') AS source_url,
                            COALESCE(m.message_ts, 0) AS message_ts,
                            m.views,
                            m.likes,
                            COALESCE(s.events_imported, 0) AS events_imported,
                            COALESCE(t.username, '') AS username,
                            COALESCE(t.title, '') AS title
                        FROM telegram_post_metric m
                        JOIN telegram_scanned_message s
                          ON s.source_id = m.source_id
                         AND s.message_id = m.message_id
                        JOIN telegram_source t
                          ON t.id = m.source_id
                        WHERE m.age_day = ?
                          AND m.message_ts IS NOT NULL
                          AND m.message_ts >= ?
                          AND COALESCE(s.events_imported, 0) > 0
                        """,
                        (int(age_day), int(since_ts)),
                    )
                tg_rows = await cur.fetchall()
            except sqlite3.OperationalError as exc:
                logger.info("popular_posts: tg query skipped: %s", exc)
                tg_rows = []
        else:
            tg_rows = []

        vk_ready = all(
            [
                await _table_exists(conn, name="vk_post_metric"),
                await _table_exists(conn, name="vk_inbox"),
                await _table_exists(conn, name="vk_inbox_import_event"),
            ]
        )
        if vk_ready:
            try:
                if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                    cur0 = await conn.execute(
                        """
                        SELECT
                            COUNT(*) AS posts,
                            COUNT(DISTINCT group_id) AS sources
                        FROM (
                            SELECT group_id, post_id
                            FROM vk_post_metric
                            WHERE age_day <= ?
                              AND post_ts IS NOT NULL
                              AND post_ts >= ?
                            GROUP BY group_id, post_id
                        )
                        """,
                        (int(latest_age_day_max), int(since_ts)),
                    )
                else:
                    cur0 = await conn.execute(
                        """
                        SELECT
                            COUNT(*) AS posts,
                            COUNT(DISTINCT group_id) AS sources
                        FROM (
                            SELECT DISTINCT group_id, post_id
                            FROM vk_post_metric
                            WHERE age_day = ?
                              AND post_ts IS NOT NULL
                              AND post_ts >= ?
                        )
                        """,
                        (int(age_day), int(since_ts)),
                    )
                row0 = await cur0.fetchone()
                if row0:
                    vk_metrics_total = int(row0[0] or 0)
                    vk_metrics_sources = int(row0[1] or 0)

                vk_has_source = await _table_exists(conn, name="vk_source")
                if vk_has_source:
                    if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                        sql = """
                            WITH latest AS (
                                SELECT
                                    group_id,
                                    post_id,
                                    MAX(age_day) AS max_age_day
                                FROM vk_post_metric
                                WHERE age_day <= ?
                                  AND post_ts IS NOT NULL
                                  AND post_ts >= ?
                                GROUP BY group_id, post_id
                            )
                            SELECT DISTINCT
                                m.group_id,
                                m.post_id,
                                m.age_day,
                                COALESCE(m.source_url, '') AS source_url,
                                COALESCE(m.post_ts, 0) AS post_ts,
                                m.views,
                                m.likes,
                                COALESCE(v.screen_name, '') AS screen_name,
                                COALESCE(v.name, '') AS name
                            FROM vk_post_metric m
                            JOIN latest l
                              ON l.group_id = m.group_id
                             AND l.post_id = m.post_id
                             AND l.max_age_day = m.age_day
                            JOIN vk_inbox i
                              ON i.group_id = m.group_id
                             AND i.post_id = m.post_id
                            JOIN vk_inbox_import_event ie
                              ON ie.inbox_id = i.id
                            LEFT JOIN vk_source v
                              ON v.group_id = m.group_id
                        """
                    else:
                        sql = """
                            SELECT DISTINCT
                                m.group_id,
                                m.post_id,
                                m.age_day,
                                COALESCE(m.source_url, '') AS source_url,
                                COALESCE(m.post_ts, 0) AS post_ts,
                                m.views,
                                m.likes,
                                COALESCE(v.screen_name, '') AS screen_name,
                                COALESCE(v.name, '') AS name
                            FROM vk_post_metric m
                            JOIN vk_inbox i
                              ON i.group_id = m.group_id
                             AND i.post_id = m.post_id
                            JOIN vk_inbox_import_event ie
                              ON ie.inbox_id = i.id
                            LEFT JOIN vk_source v
                              ON v.group_id = m.group_id
                            WHERE m.age_day = ?
                              AND m.post_ts IS NOT NULL
                              AND m.post_ts >= ?
                        """
                else:
                    if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                        sql = """
                            WITH latest AS (
                                SELECT
                                    group_id,
                                    post_id,
                                    MAX(age_day) AS max_age_day
                                FROM vk_post_metric
                                WHERE age_day <= ?
                                  AND post_ts IS NOT NULL
                                  AND post_ts >= ?
                                GROUP BY group_id, post_id
                            )
                            SELECT DISTINCT
                                m.group_id,
                                m.post_id,
                                m.age_day,
                                COALESCE(m.source_url, '') AS source_url,
                                COALESCE(m.post_ts, 0) AS post_ts,
                                m.views,
                                m.likes,
                                '' AS screen_name,
                                '' AS name
                            FROM vk_post_metric m
                            JOIN latest l
                              ON l.group_id = m.group_id
                             AND l.post_id = m.post_id
                             AND l.max_age_day = m.age_day
                            JOIN vk_inbox i
                              ON i.group_id = m.group_id
                             AND i.post_id = m.post_id
                            JOIN vk_inbox_import_event ie
                              ON ie.inbox_id = i.id
                        """
                    else:
                        sql = """
                            SELECT DISTINCT
                                m.group_id,
                                m.post_id,
                                m.age_day,
                                COALESCE(m.source_url, '') AS source_url,
                                COALESCE(m.post_ts, 0) AS post_ts,
                                m.views,
                                m.likes,
                                '' AS screen_name,
                                '' AS name
                            FROM vk_post_metric m
                            JOIN vk_inbox i
                              ON i.group_id = m.group_id
                             AND i.post_id = m.post_id
                            JOIN vk_inbox_import_event ie
                              ON ie.inbox_id = i.id
                            WHERE m.age_day = ?
                              AND m.post_ts IS NOT NULL
                              AND m.post_ts >= ?
                        """
                if isinstance(latest_age_day_max, int) and latest_age_day_max >= 0:
                    cur2 = await conn.execute(sql, (int(latest_age_day_max), int(since_ts)))
                else:
                    cur2 = await conn.execute(sql, (int(age_day), int(since_ts)))
                vk_rows = await cur2.fetchall()
            except sqlite3.OperationalError as exc:
                logger.info("popular_posts: vk query skipped: %s", exc)
                vk_rows = []
        else:
            vk_rows = []

    tg_sample: dict[int, set[int]] = {}
    tg_candidate_age_day: dict[tuple[int, int], int] = {}
    for source_id, message_id, candidate_age_day, _url, _ts, _v, _l, _imp, _u, _t in tg_rows:
        try:
            sid = int(source_id)
            mid = int(message_id)
            cad = int(candidate_age_day)
        except Exception:
            continue
        tg_sample.setdefault(sid, set()).add(mid)
        tg_candidate_age_day[(sid, mid)] = cad

    vk_sample: dict[int, set[int]] = {}
    vk_candidate_age_day: dict[tuple[int, int], int] = {}
    for group_id, post_id, candidate_age_day, _url, _ts, _v, _l, _sn, _nm in vk_rows:
        try:
            gid = int(group_id)
            pid = int(post_id)
            cad = int(candidate_age_day)
        except Exception:
            continue
        vk_sample.setdefault(gid, set()).add(pid)
        vk_candidate_age_day[(gid, pid)] = cad

    # Baselines: same source + same candidate age bucket over the popularity horizon
    # (default 90 days). The report window only controls which posts are candidates.
    tg_baseline: dict[tuple[int, int], _Baseline] = {}
    for sid_mid, candidate_age_day in tg_candidate_age_day.items():
        sid, _mid = sid_mid
        base = await load_telegram_popularity_baseline(
            db,
            source_id=int(sid),
            age_day=int(candidate_age_day),
            now_ts=int(now_ts),
        )
        tg_baseline[sid_mid] = _Baseline(
            median_views=base.median_views,
            median_likes=base.median_likes,
            sample=int(base.sample),
        )
    vk_baseline: dict[tuple[int, int], _Baseline] = {}
    for gid_pid, candidate_age_day in vk_candidate_age_day.items():
        gid, _pid = gid_pid
        base = await load_vk_popularity_baseline(
            db,
            group_id=int(gid),
            age_day=int(candidate_age_day),
            now_ts=int(now_ts),
        )
        vk_baseline[gid_pid] = _Baseline(
            median_views=base.median_views,
            median_likes=base.median_likes,
            sample=int(base.sample),
        )

    # Build candidates: strictly above median on at least one metric.
    candidates: list[_PostItem] = []
    skipped: dict[str, Any] = {
        "tg_available": bool(tg_ready),
        "vk_available": bool(vk_ready),
        "tg_total": int(len(tg_rows)),
        "vk_total": int(len(vk_rows)),
        "tg_sources": int(len(tg_sample)),
        "vk_sources": int(len(vk_sample)),
        "tg_metrics_total": int(tg_metrics_total),
        "vk_metrics_total": int(vk_metrics_total),
        "tg_metrics_sources": int(tg_metrics_sources),
        "vk_metrics_sources": int(vk_metrics_sources),
        "min_sample": int(min_sample),
        # Diagnostics: among posts that had enough data to compare (passed sample/median/metrics),
        # count how many are above the per-source median by each metric.
        "checked_posts": 0,
        "above_median_views": 0,
        "above_median_likes": 0,
        "above_median_both": 0,
        "skipped_small_sample": 0,
        "skipped_missing_median": 0,
        "skipped_missing_metrics": 0,
        "skipped_not_above_median": 0,
    }

    def _add_candidate(
        *,
        platform: str,
        source_key: int,
        source_label: str,
        post_id: int,
        post_url: str,
        published_ts: int,
        views: int | None,
        likes: int | None,
        baseline: _Baseline,
    ) -> None:
        if int(baseline.sample or 0) < min_sample:
            skipped["skipped_small_sample"] += 1
            return
        if baseline.median_views is None or baseline.median_likes is None:
            skipped["skipped_missing_median"] += 1
            return
        if not isinstance(views, int) or not isinstance(likes, int) or views < 0 or likes < 0:
            skipped["skipped_missing_metrics"] += 1
            return
        is_above_views = views > int(baseline.median_views)
        is_above_likes = likes > int(baseline.median_likes)
        skipped["checked_posts"] += 1
        if is_above_views:
            skipped["above_median_views"] += 1
        if is_above_likes:
            skipped["above_median_likes"] += 1
        if is_above_views and is_above_likes:
            skipped["above_median_both"] += 1
        if not (is_above_views or is_above_likes):
            skipped["skipped_not_above_median"] += 1
            return
        popularity = ""
        try:
            popularity = popularity_marks(
                views=int(views),
                likes=int(likes),
                baseline=PopularityBaseline(
                    median_views=baseline.median_views,
                    median_likes=baseline.median_likes,
                    sample=int(baseline.sample),
                ),
            ).text
        except Exception:
            popularity = ""
        v_ratio = _safe_ratio(views, baseline.median_views)
        l_ratio = _safe_ratio(likes, baseline.median_likes)
        score = float(min(v_ratio, l_ratio)) + 0.01 * float(v_ratio + l_ratio)
        candidates.append(
            _PostItem(
                platform=platform,
                source_key=int(source_key),
                source_label=str(source_label).strip() or str(source_key),
                post_id=int(post_id),
                post_url=str(post_url).strip(),
                published_ts=int(published_ts),
                views=int(views),
                likes=int(likes),
                baseline=baseline,
                popularity=str(popularity or "").strip(),
                score=float(score),
            )
        )

    for source_id, message_id, _candidate_age_day, url, ts, v, l, _imp, username, title in tg_rows:
        try:
            sid = int(source_id)
            mid = int(message_id)
            published_ts = int(ts or 0)
        except Exception:
            continue
        base = tg_baseline.get((sid, mid)) or _Baseline(None, None, 0)
        label = ""
        u = str(username or "").strip()
        t = str(title or "").strip()
        if t and u:
            label = f"{t} (@{u})"
        elif u:
            label = f"@{u}"
        elif t:
            label = t
        else:
            label = f"tg:{sid}"
        post_url = str(url or "").strip()
        if not post_url and u:
            post_url = f"https://t.me/{u}/{mid}"
        _add_candidate(
            platform="tg",
            source_key=sid,
            source_label=label,
            post_id=mid,
            post_url=post_url,
            published_ts=published_ts,
            views=v if isinstance(v, int) else None,
            likes=l if isinstance(l, int) else None,
            baseline=base,
        )

    for group_id, post_id, _candidate_age_day, url, ts, v, l, screen_name, name in vk_rows:
        try:
            gid = int(group_id)
            pid = int(post_id)
            published_ts = int(ts or 0)
        except Exception:
            continue
        base = vk_baseline.get((gid, pid)) or _Baseline(None, None, 0)
        sn = str(screen_name or "").strip()
        nm = str(name or "").strip()
        label = nm or (f"vk:{sn}" if sn else f"vk:{gid}")
        post_url = str(url or "").strip()
        if not post_url:
            post_url = f"https://vk.com/wall-{gid}_{pid}"
        _add_candidate(
            platform="vk",
            source_key=gid,
            source_label=label,
            post_id=pid,
            post_url=post_url,
            published_ts=published_ts,
            views=v if isinstance(v, int) else None,
            likes=l if isinstance(l, int) else None,
            baseline=base,
        )

    # Sort by score (normalized), then raw views/likes.
    candidates_sorted = sorted(
        candidates,
        key=lambda it: (
            float(it.score),
            int(it.views or 0),
            int(it.likes or 0),
        ),
        reverse=True,
    )
    return (candidates_sorted[: max(1, int(limit or 10))], skipped)


def _fmt_int(value: int | None) -> str:
    if not isinstance(value, int):
        return "—"
    if value < 0:
        return "—"
    return f"{value:,}".replace(",", " ")


def _fmt_ratio(value: int | None, median: int | None) -> str:
    if not isinstance(value, int) or not isinstance(median, int):
        return ""
    if median <= 0:
        return "×?"
    return f"×{(float(value) / float(median)):.2f}"


def _fmt_platform(platform: str) -> str:
    p = (platform or "").strip().lower()
    if p == "tg":
        return "TG"
    if p == "vk":
        return "VK"
    return p.upper() or "?"


async def _send_popular_posts_report(message: Message, db: Database, *, limit: int = 10) -> None:
    seven_day, seven_dbg = await _load_top_items(
        db,
        window_days=7,
        age_day=_POPULAR_POSTS_SEVEN_DAY_MAX_AGE,
        latest_age_day_max=_POPULAR_POSTS_SEVEN_DAY_MAX_AGE,
        limit=limit,
    )
    three_day, three_dbg = await _load_top_items(
        db,
        window_days=3,
        age_day=_POPULAR_POSTS_THREE_DAY_MAX_AGE,
        latest_age_day_max=_POPULAR_POSTS_THREE_DAY_MAX_AGE,
        limit=limit,
    )
    one_day, one_dbg = await _load_top_items(db, window_days=1, age_day=0, limit=limit)

    urls = [it.post_url for it in (seven_day + three_day + one_day) if it.post_url]
    telegraph_map = await _resolve_telegraph_map(db, source_urls=urls)

    def _render_section(title: str, items: list[_PostItem], dbg: dict[str, Any]) -> list[str]:
        lines: list[str] = [f"🔥 <b>{html.escape(title)}</b>", ""]
        min_sample = int(dbg.get("min_sample", 0) or 0)
        if min_sample <= 0:
            min_sample = max(2, _env_int("POST_POPULARITY_MIN_SAMPLE", 2))

        def _append_debug() -> None:
            lines.append(
                "Данные (посты с импортом событий): "
                f"TG постов={int(dbg.get('tg_total', 0))} (источников={int(dbg.get('tg_sources', 0))}), "
                f"VK постов={int(dbg.get('vk_total', 0))} (источников={int(dbg.get('vk_sources', 0))})."
            )
            lines.append(
                "Данные (метрики, включая посты без импортов): "
                f"TG постов={int(dbg.get('tg_metrics_total', 0))} (источников={int(dbg.get('tg_metrics_sources', 0))}), "
                f"VK постов={int(dbg.get('vk_metrics_total', 0))} (источников={int(dbg.get('vk_metrics_sources', 0))})."
            )
            lines.append(
                f"Фильтры/скипы: min_sample={min_sample}, "
                f"skip(sample)={int(dbg.get('skipped_small_sample', 0))}, "
                f"skip(median)={int(dbg.get('skipped_missing_median', 0))}, "
                f"skip(metrics)={int(dbg.get('skipped_missing_metrics', 0))}, "
                f"skip(&lt;=median)={int(dbg.get('skipped_not_above_median', 0))}."
            )
            checked = int(dbg.get("checked_posts", 0) or 0)
            if checked > 0:
                lines.append(
                    "Выше медианы (после фильтров): "
                    f"views={int(dbg.get('above_median_views', 0) or 0)}, "
                    f"likes={int(dbg.get('above_median_likes', 0) or 0)}, "
                    f"оба={int(dbg.get('above_median_both', 0) or 0)} "
                    f"(из {checked})."
                )

        if not items:
            lines.append("Нет постов, которые выше хотя бы одной из медиан и имеют достаточную выборку для расчёта.")
            if not bool(dbg.get("tg_available", True)):
                lines.append("Платформа TG: таблицы метрик не найдены (возможен DB_INIT_MINIMAL или старый дамп).")
            if not bool(dbg.get("vk_available", True)):
                lines.append("Платформа VK: таблицы метрик не найдены (возможен DB_INIT_MINIMAL или старый дамп).")
            _append_debug()
            return lines

        for idx, it in enumerate(items, start=1):
            head_marks = str(it.popularity or "").strip()
            head_marks = f"{html.escape(head_marks)} " if head_marks else ""
            post_url = str(it.post_url or "").strip()
            post_link = f'<a href="{html.escape(post_url)}">исходник</a>' if post_url else "исходник"
            lines.append(
                f"{idx}. {head_marks}[{_fmt_platform(it.platform)}] <b>{html.escape(it.source_label)}</b> — {post_link}"
            )
            lines.append(
                f"  ⭐ views: <b>{_fmt_int(it.views)}</b> (median { _fmt_int(it.baseline.median_views) }, n={int(it.baseline.sample)}) "
                f"{html.escape(_fmt_ratio(it.views, it.baseline.median_views))}"
            )
            lines.append(
                f"  👍 likes: <b>{_fmt_int(it.likes)}</b> (median { _fmt_int(it.baseline.median_likes) }, n={int(it.baseline.sample)}) "
                f"{html.escape(_fmt_ratio(it.likes, it.baseline.median_likes))}"
            )
            bundle = telegraph_map.get(post_url)
            if bundle and int(getattr(bundle, "total", 0) or 0) > 0:
                evs = list(getattr(bundle, "events", None) or [])
                total = int(getattr(bundle, "total", 0) or 0)
                if len(evs) == 1:
                    ev = evs[0]
                    ev_title = str(getattr(ev, "title", "") or "").strip() or "событие"
                    ev_id = int(getattr(ev, "event_id", 0) or 0)
                    turl = getattr(ev, "telegraph_url", None)
                    if turl:
                        lines.append(
                            f'  Событие: <a href="{html.escape(str(turl))}">{html.escape(ev_title)}</a> (id={ev_id})'
                        )
                    else:
                        lines.append(f"  Событие: {html.escape(ev_title)} (id={ev_id})")
                else:
                    lines.append(f"  События: {total}")
                    for ev in evs:
                        ev_title = str(getattr(ev, "title", "") or "").strip() or "событие"
                        ev_id = int(getattr(ev, "event_id", 0) or 0)
                        turl = getattr(ev, "telegraph_url", None)
                        if turl:
                            lines.append(
                                f'  • <a href="{html.escape(str(turl))}">{html.escape(ev_title)}</a> (id={ev_id})'
                            )
                        else:
                            lines.append(f"  • {html.escape(ev_title)} (id={ev_id})")
                extra = max(0, int(total) - int(len(evs)))
                if extra:
                    lines.append(f"  … ещё {extra}")
            else:
                lines.append("  Событие: —")
            lines.append("")

        _append_debug()
        return lines

    lines: list[str] = [
        "📊 <b>Популярные посты → события</b>",
        "Фильтр: views или likes строго выше медианы внутри канала/сообщества; медиана считается по тому же age_day за окно popularity horizon источника (обычно 90 дней), а в ТОП попадают посты из окна отчёта.",
        "Примечание: метрики пишутся только для постов, где были извлечены события (events_extracted>0/forced/existing); отчёт ниже дополнительно требует импортов (events_imported>0).",
        "",
        "Окно 7 суток: берём <b>последний доступный</b> снапшот метрик для постов последних ~7 суток (`age_day=0..6`) и сравниваем его с медианой того же `age_day`.",
        "",
    ]
    lines.extend(_render_section("ТОП-10 за 7 суток", seven_day, seven_dbg))
    lines.append("")
    lines.extend(
        [
        "Окно 3 суток: берём <b>последний доступный</b> снапшот метрик для постов последних ~3 суток (`age_day=0..2`) и сравниваем его с медианой того же `age_day`.",
        "",
        ]
    )
    lines.extend(_render_section("ТОП-10 за 3 суток", three_day, three_dbg))
    lines.append("")
    lines.append("Окно 24 часа: берём снапшоты метрик <b>age_day=0</b> (посты опубликованы в последние ~24 часа).")
    lines.append("")
    lines.extend(_render_section("ТОП-10 за 24 часа", one_day, one_dbg))

    for chunk in _chunk_lines(lines):
        await message.answer(chunk, parse_mode="HTML", disable_web_page_preview=True)


@popular_posts_router.message(Command("popular_posts"))
async def cmd_popular_posts(message: Message) -> None:
    get_db = require_main_attr("get_db")
    db = get_db()
    if db is None:
        await message.answer("❌ База данных ещё не инициализирована. Попробуйте позже.")
        return
    try:
        user_id = int(message.from_user.id)
    except Exception:
        await message.answer("❌ Не удалось определить пользователя.")
        return
    if not await _require_superadmin(db, user_id):
        await message.answer("❌ Команда доступна только администраторам.")
        return

    limit = 10
    try:
        parts = (message.text or "").strip().split()
        if len(parts) >= 2 and parts[1].isdigit():
            limit = int(parts[1])
        if limit < 1:
            limit = 1
        if limit > 30:
            limit = 30
    except Exception:
        limit = 10

    try:
        await _send_popular_posts_report(message, db, limit=limit)
    except Exception:
        logger.exception("popular_posts: failed")
        await message.answer("❌ Не удалось построить отчёт. Проверьте логи.")
