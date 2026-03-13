from __future__ import annotations

import html
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
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


def _utc_now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _parse_iso_date(value: Any) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _local_today() -> date:
    try:
        tz = require_main_attr("LOCAL_TZ")
    except Exception:
        tz = timezone.utc
    return datetime.now(tz).date()


def _event_is_current_or_future(event: Any, *, today: date | None = None) -> bool:
    current_day = today or _local_today()
    start_date = _parse_iso_date(getattr(event, "date", None))
    if start_date is None:
        return True
    if start_date >= current_day:
        return True
    end_date = _parse_iso_date(getattr(event, "end_date", None))
    if end_date is not None:
        return end_date >= current_day
    return False


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


@dataclass(slots=True, frozen=True)
class _WindowSpec:
    title: str
    window_days: int
    preferred_age_day: int
    intro: str


_WINDOW_SPECS: tuple[_WindowSpec, ...] = (
    _WindowSpec(
        title="ТОП-10 за 7 суток",
        window_days=7,
        preferred_age_day=6,
        intro=(
            "Окно 7 суток: посты опубликованы в последние ~7 суток; "
            "предпочитаем снапшот метрик <b>age_day=6</b>, а если его ещё нет — "
            "берём последний доступный <b>age_day&lt;=6</b>."
        ),
    ),
    _WindowSpec(
        title="ТОП-10 за 3 суток",
        window_days=3,
        preferred_age_day=2,
        intro=(
            "Окно 3 суток: посты опубликованы в последние ~3 суток; "
            "предпочитаем снапшот метрик <b>age_day=2</b>, а если его ещё нет — "
            "берём последний доступный <b>age_day&lt;=2</b>."
        ),
    ),
    _WindowSpec(
        title="ТОП-10 за 24 часа",
        window_days=1,
        preferred_age_day=0,
        intro=(
            "Окно 24 часа: берём снапшоты метрик <b>age_day=0</b> "
            "(посты опубликованы в последние ~24 часа)."
        ),
    ),
)


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


async def _resolve_telegraph_map(
    db: Database,
    *,
    source_urls: list[str],
) -> tuple[dict[str, _EventLinks], set[str]]:
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
        return {}, set()
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
            return {}, set()

        try:
            events = (
                await session.execute(select(Event).where(Event.id.in_(sorted(event_ids))))
            ).scalars().all()
        except Exception:
            logger.debug("popular_posts: failed to fetch events for telegraph map", exc_info=True)
            return {}, set()

        current_day = _local_today()
        active_id_to_ref: dict[int, _EventRef] = {}
        active_event_ids: set[int] = set()
        for ev in events:
            if not _event_is_current_or_future(ev, today=current_day):
                continue
            try:
                ev_id = int(getattr(ev, "id", 0) or 0)
            except Exception:
                continue
            title = str(getattr(ev, "title", "") or "").strip() or "событие"
            active_event_ids.add(ev_id)
            active_id_to_ref[ev_id] = _EventRef(
                event_id=ev_id,
                title=title,
                telegraph_url=_event_telegraph_url(ev),
            )

        out: dict[str, _EventLinks] = {}
        matched_urls: set[str] = set()
        for raw in uniq_urls:
            key = orig_to_key.get(raw) or _canonical_source_url_for_lookup(raw) or ""
            if not key:
                continue
            eids = list(key_to_event_ids.get(key) or [])
            if not eids:
                continue
            matched_urls.add(raw)
            refs: list[_EventRef] = []
            for ev_id in eids:
                ref = active_id_to_ref.get(int(ev_id))
                if not ref:
                    continue
                refs.append(ref)
                if len(refs) >= 3:
                    break
            if refs:
                total_active = sum(1 for ev_id in eids if int(ev_id) in active_event_ids)
                out[raw] = _EventLinks(events=tuple(refs), total=int(total_active))
        return out, matched_urls


def _prune_stale_only_items(
    items: list[_PostItem],
    *,
    telegraph_map: dict[str, _EventLinks],
    matched_urls: set[str],
    dbg: dict[str, Any] | None = None,
) -> list[_PostItem]:
    kept: list[_PostItem] = []
    skipped_past_event_only = 0
    for item in items:
        post_url = str(getattr(item, "post_url", "") or "").strip()
        if post_url and post_url in matched_urls and post_url not in telegraph_map:
            skipped_past_event_only += 1
            continue
        kept.append(item)
    if dbg is not None and skipped_past_event_only:
        dbg["skipped_past_event_only"] = int(dbg.get("skipped_past_event_only", 0) or 0) + int(
            skipped_past_event_only
        )
    return kept


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


def _dedupe_latest_age_rows(
    rows: list[tuple],
    *,
    key_indexes: tuple[int, int],
    age_idx: int,
) -> tuple[list[tuple], dict[int, int]]:
    best: dict[tuple[int, int], tuple[int, tuple]] = {}
    for row in rows:
        try:
            key = (int(row[key_indexes[0]]), int(row[key_indexes[1]]))
        except Exception:
            continue
        try:
            age = int(row[age_idx])
        except Exception:
            age = -1
        current = best.get(key)
        if current is None or age > current[0]:
            best[key] = (age, row)

    age_hist: dict[int, int] = {}
    deduped: list[tuple] = []
    for age, row in best.values():
        age_hist[int(age)] = int(age_hist.get(int(age), 0)) + 1
        deduped.append(row)
    return deduped, age_hist


async def _load_top_items(
    db: Database,
    *,
    window_days: int,
    age_day: int,
    limit: int,
) -> tuple[list[_PostItem], dict[str, Any]]:
    now_ts = _utc_now_ts()
    since_ts = now_ts - max(1, int(window_days)) * 86400
    min_sample = max(2, _env_int("POST_POPULARITY_MIN_SAMPLE", 2))
    preferred_age_day = max(0, int(age_day))
    configured_max_age_day = max(0, _env_int("POST_POPULARITY_MAX_AGE_DAY", 2))
    tg_monitoring_days_back = max(0, _env_int("TG_MONITORING_DAYS_BACK", 3))

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
                cur0 = await conn.execute(
                    """
                    SELECT
                        COUNT(*) AS posts,
                        COUNT(DISTINCT source_id) AS sources
                    FROM (
                        SELECT DISTINCT source_id, message_id
                        FROM telegram_post_metric
                        WHERE age_day <= ?
                          AND message_ts IS NOT NULL
                          AND message_ts >= ?
                    )
                    """,
                    (int(preferred_age_day), int(since_ts)),
                )
                row0 = await cur0.fetchone()
                if row0:
                    tg_metrics_total = int(row0[0] or 0)
                    tg_metrics_sources = int(row0[1] or 0)

                cur = await conn.execute(
                    """
                    SELECT
                        m.source_id,
                        m.message_id,
                        COALESCE(m.source_url, '') AS source_url,
                        COALESCE(m.message_ts, 0) AS message_ts,
                        m.age_day,
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
                    WHERE m.age_day <= ?
                      AND m.message_ts IS NOT NULL
                      AND m.message_ts >= ?
                      AND COALESCE(s.events_imported, 0) > 0
                    """,
                    (int(preferred_age_day), int(since_ts)),
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
                cur0 = await conn.execute(
                    """
                    SELECT
                        COUNT(*) AS posts,
                        COUNT(DISTINCT group_id) AS sources
                    FROM (
                        SELECT DISTINCT group_id, post_id
                        FROM vk_post_metric
                        WHERE age_day <= ?
                          AND post_ts IS NOT NULL
                          AND post_ts >= ?
                    )
                    """,
                    (int(preferred_age_day), int(since_ts)),
                )
                row0 = await cur0.fetchone()
                if row0:
                    vk_metrics_total = int(row0[0] or 0)
                    vk_metrics_sources = int(row0[1] or 0)

                vk_has_source = await _table_exists(conn, name="vk_source")
                if vk_has_source:
                    sql = """
                        SELECT DISTINCT
                            m.group_id,
                            m.post_id,
                            COALESCE(m.source_url, '') AS source_url,
                            COALESCE(m.post_ts, 0) AS post_ts,
                            m.age_day,
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
                        WHERE m.age_day <= ?
                          AND m.post_ts IS NOT NULL
                          AND m.post_ts >= ?
                    """
                else:
                    sql = """
                        SELECT DISTINCT
                            m.group_id,
                            m.post_id,
                            COALESCE(m.source_url, '') AS source_url,
                            COALESCE(m.post_ts, 0) AS post_ts,
                            m.age_day,
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
                        WHERE m.age_day <= ?
                          AND m.post_ts IS NOT NULL
                          AND m.post_ts >= ?
                    """
                cur2 = await conn.execute(sql, (int(preferred_age_day), int(since_ts)))
                vk_rows = await cur2.fetchall()
            except sqlite3.OperationalError as exc:
                logger.info("popular_posts: vk query skipped: %s", exc)
                vk_rows = []
        else:
            vk_rows = []

    tg_rows, tg_age_hist = _dedupe_latest_age_rows(tg_rows, key_indexes=(0, 1), age_idx=4)
    vk_rows, vk_age_hist = _dedupe_latest_age_rows(vk_rows, key_indexes=(0, 1), age_idx=4)

    # Baselines: per-source medians inside the same window and preferred age cap.
    tg_views: dict[int, list[int]] = {}
    tg_likes: dict[int, list[int]] = {}
    tg_sample: dict[int, set[int]] = {}
    for source_id, message_id, _url, _ts, _age, v, l, _imp, _u, _t in tg_rows:
        try:
            sid = int(source_id)
            mid = int(message_id)
        except Exception:
            continue
        tg_sample.setdefault(sid, set()).add(mid)
        if isinstance(v, int) and v >= 0:
            tg_views.setdefault(sid, []).append(int(v))
        if isinstance(l, int) and l >= 0:
            tg_likes.setdefault(sid, []).append(int(l))

    vk_views: dict[int, list[int]] = {}
    vk_likes: dict[int, list[int]] = {}
    vk_sample: dict[int, set[int]] = {}
    for group_id, post_id, _url, _ts, _age, v, l, _sn, _nm in vk_rows:
        try:
            gid = int(group_id)
            pid = int(post_id)
        except Exception:
            continue
        vk_sample.setdefault(gid, set()).add(pid)
        if isinstance(v, int) and v >= 0:
            vk_views.setdefault(gid, []).append(int(v))
        if isinstance(l, int) and l >= 0:
            vk_likes.setdefault(gid, []).append(int(l))

    tg_baseline: dict[int, _Baseline] = {}
    for sid, ids in tg_sample.items():
        sample = int(len(ids))
        tg_baseline[sid] = _Baseline(
            median_views=_median_int(tg_views.get(sid, [])),
            median_likes=_median_int(tg_likes.get(sid, [])),
            sample=sample,
        )
    vk_baseline: dict[int, _Baseline] = {}
    for gid, ids in vk_sample.items():
        sample = int(len(ids))
        vk_baseline[gid] = _Baseline(
            median_views=_median_int(vk_views.get(gid, [])),
            median_likes=_median_int(vk_likes.get(gid, [])),
            sample=sample,
        )

    # Build candidates: strictly above median on at least one metric.
    candidates: list[_PostItem] = []
    skipped: dict[str, Any] = {
        "tg_available": bool(tg_ready),
        "vk_available": bool(vk_ready),
        "preferred_age_day": int(preferred_age_day),
        "configured_max_age_day": int(configured_max_age_day),
        "tg_monitoring_days_back": int(tg_monitoring_days_back),
        "tg_total": int(len(tg_rows)),
        "vk_total": int(len(vk_rows)),
        "tg_sources": int(len(tg_sample)),
        "vk_sources": int(len(vk_sample)),
        "tg_metrics_total": int(tg_metrics_total),
        "vk_metrics_total": int(vk_metrics_total),
        "tg_metrics_sources": int(tg_metrics_sources),
        "vk_metrics_sources": int(vk_metrics_sources),
        "tg_selected_max_age_day": max(tg_age_hist) if tg_age_hist else None,
        "vk_selected_max_age_day": max(vk_age_hist) if vk_age_hist else None,
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
            from source_parsing.post_metrics import PopularityBaseline, popularity_marks

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
        primary_ratio = max(v_ratio if is_above_views else 0.0, l_ratio if is_above_likes else 0.0)
        score = float(primary_ratio) + 0.01 * float(v_ratio + l_ratio)
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

    for source_id, message_id, url, ts, _age, v, l, _imp, username, title in tg_rows:
        try:
            sid = int(source_id)
            mid = int(message_id)
            published_ts = int(ts or 0)
        except Exception:
            continue
        base = tg_baseline.get(sid) or _Baseline(None, None, 0)
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

    for group_id, post_id, url, ts, _age, v, l, screen_name, name in vk_rows:
        try:
            gid = int(group_id)
            pid = int(post_id)
            published_ts = int(ts or 0)
        except Exception:
            continue
        base = vk_baseline.get(gid) or _Baseline(None, None, 0)
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


def _window_diag_lines(dbg: dict[str, Any]) -> list[str]:
    preferred_age_day = int(dbg.get("preferred_age_day", 0) or 0)
    if preferred_age_day <= 0:
        return []

    lines: list[str] = []
    configured_max_age_day = int(dbg.get("configured_max_age_day", 0) or 0)
    if configured_max_age_day < preferred_age_day:
        lines.append(
            "Сбор зрелых снапшотов ограничен: "
            f"POST_POPULARITY_MAX_AGE_DAY={configured_max_age_day}, "
            f"поэтому <b>age_day={preferred_age_day}</b> пока не пишется; "
            f"окно использует последний доступный <b>age_day&lt;={preferred_age_day}</b>."
        )

    tg_days_back = int(dbg.get("tg_monitoring_days_back", 0) or 0)
    if tg_days_back and tg_days_back < (preferred_age_day + 1):
        lines.append(
            "Для Telegram Monitoring окно скана короче целевого бакета: "
            f"TG_MONITORING_DAYS_BACK={tg_days_back}, для <b>age_day={preferred_age_day}</b> "
            f"нужно как минимум {preferred_age_day + 1} суток."
        )

    observed_parts: list[str] = []
    tg_selected_max_age_day = dbg.get("tg_selected_max_age_day")
    vk_selected_max_age_day = dbg.get("vk_selected_max_age_day")
    if isinstance(tg_selected_max_age_day, int) and tg_selected_max_age_day < preferred_age_day:
        observed_parts.append(f"TG max age_day={tg_selected_max_age_day}")
    if isinstance(vk_selected_max_age_day, int) and vk_selected_max_age_day < preferred_age_day:
        observed_parts.append(f"VK max age_day={vk_selected_max_age_day}")
    if observed_parts:
        lines.append(
            f"Фактически в этом окне пока доступны более ранние снапшоты: {', '.join(observed_parts)}."
        )
    return lines


async def _send_popular_posts_report(message: Message, db: Database, *, limit: int = 10) -> None:
    window_results: list[tuple[_WindowSpec, list[_PostItem], dict[str, Any]]] = []
    for spec in _WINDOW_SPECS:
        items, dbg = await _load_top_items(
            db,
            window_days=spec.window_days,
            age_day=spec.preferred_age_day,
            limit=limit,
        )
        window_results.append((spec, items, dbg))

    urls = [it.post_url for _spec, items, _dbg in window_results for it in items if it.post_url]
    telegraph_map, matched_urls = await _resolve_telegraph_map(db, source_urls=urls)
    window_results = [
        (
            spec,
            _prune_stale_only_items(items, telegraph_map=telegraph_map, matched_urls=matched_urls, dbg=dbg),
            dbg,
        )
        for spec, items, dbg in window_results
    ]

    def _render_section(spec: _WindowSpec, items: list[_PostItem], dbg: dict[str, Any]) -> list[str]:
        lines: list[str] = [f"🔥 <b>{html.escape(spec.title)}</b>", ""]
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
                f"skip(&lt;=median)={int(dbg.get('skipped_not_above_median', 0))}, "
                f"skip(past_event_only)={int(dbg.get('skipped_past_event_only', 0))}."
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
            lines.append("Нет постов, которые выше медианы и имеют достаточную выборку для расчёта.")
            if not bool(dbg.get("tg_available", True)):
                lines.append("Платформа TG: таблицы метрик не найдены (возможен DB_INIT_MINIMAL или старый дамп).")
            if not bool(dbg.get("vk_available", True)):
                lines.append("Платформа VK: таблицы метрик не найдены (возможен DB_INIT_MINIMAL или старый дамп).")
            lines.extend(_window_diag_lines(dbg))
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

        lines.extend(_window_diag_lines(dbg))
        _append_debug()
        return lines

    lines: list[str] = [
        "📊 <b>Популярные посты → события</b>",
        "Показываем только события, которые сегодня или в будущем; посты, связанные только с уже завершившимися событиями, скрываются.",
        "Фильтр: views или likes строго выше медианы внутри канала/сообщества; медианы считаются по окну и платформе отдельно.",
        "Примечание: метрики пишутся только для постов, где были извлечены события (events_extracted>0/forced/existing); отчёт ниже дополнительно требует импортов (events_imported>0).",
    ]
    for idx, (spec, items, dbg) in enumerate(window_results):
        lines.append("")
        lines.append(spec.intro)
        lines.append("")
        lines.extend(_render_section(spec, items, dbg))
        if idx < len(window_results) - 1:
            lines.append("")

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
