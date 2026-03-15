"""Handlers for source parsing operations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse, urljoin

from aiogram import Bot
from sqlalchemy import select

from admin_chat import resolve_superadmin_chat_id
from db import Database
from ops_run import finish_ops_run, start_ops_run
from source_parsing.kaggle_runner import run_kaggle_kernel
from source_parsing.parser import (
    TheatreEvent,
    parse_theatre_json,
    normalize_location_name,
    find_existing_event,
    should_update_event,
    find_linked_events,
    limit_photos_for_source,
)
from poster_media import PosterMedia, is_supabase_storage_url, process_media
from kaggle_registry import list_jobs, remove_job
from video_announce.kaggle_client import KaggleClient
from models import Event, EventSource

logger = logging.getLogger(__name__)

PARSE_EVENT_TIMEOUT_SECONDS = int(os.getenv("SOURCE_PARSING_EVENT_TIMEOUT_SECONDS", "180"))
SOURCE_PARSING_OCR_TIMEOUT_SECONDS = int(os.getenv("SOURCE_PARSING_OCR_TIMEOUT_SECONDS", "60"))
SOURCE_PARSING_DIAG_TITLE = os.getenv("SOURCE_PARSING_DIAG_TITLE", "джотто").strip().lower()
SOURCE_PARSING_DISABLE_OCR_SOURCES = {
    s.strip().lower()
    for s in os.getenv("SOURCE_PARSING_DISABLE_OCR_SOURCES", "tretyakov").split(",")
    if s.strip()
}

# Delay between adding events to avoid overloading the system
EVENT_ADD_DELAY_SECONDS = 5  # Delay for Telegraph creation

# TEMPORARY: Limit events for debugging (set to None to disable)
DEBUG_MAX_EVENTS = None

_SOURCE_HOST_HINTS: dict[str, tuple[str, ...]] = {
    "dramteatr": ("dramteatr39.ru",),
    "muzteatr": ("muzteatr39.ru",),
    "sobor": ("sobor39.ru",),
    "tretyakov": ("tretyakovgallery.ru",),
    "philharmonia": ("filarmonia39.ru",),
    "qtickets": ("qtickets.events",),
    "pyramida": ("pyramida.info",),
    "dom_iskusstv": ("xn--b1admiilxbaki.xn--p1ai", "домискусств.рф"),
}


async def _fetch_og_image_for_dramteatr(page_url: str) -> str | None:
    """Best-effort cover extraction for dramteatr event pages.

    Kaggle JSON for dramteatr sometimes lacks `photos`. For operator UX and more
    stable Telegraph previews, try to grab `og:image` from the event page.
    """
    url = (page_url or "").strip()
    if not url or "dramteatr39.ru" not in url:
        return None
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
    except Exception:
        return None

    m = re.search(
        r'(?is)<meta\\s+[^>]*(?:property|name)=(\"|\\\')(?:(?:og:image)|(?:twitter:image))\\1[^>]*content=(\"|\\\')([^\"\\\']+)\\2',
        html,
    )
    if m:
        img = (m.group(3) or "").strip()
        if img:
            final = urljoin(url, img)
            if final.startswith(("http://", "https://")):
                return final

    # Fallback: dramteatr pages often embed galleries without OG tags.
    # Prefer event gallery images under `/storage/uploads/!spektakli/...`.
    img_srcs = re.findall(r'(?is)<img\\s+[^>]*src=(\"|\\\')([^\"\\\']+)\\1', html)
    candidates: list[str] = []
    for _q, src in img_srcs:
        s = (src or "").strip()
        if not s:
            continue
        s = urljoin(url, s)
        if not s.startswith(("http://", "https://")):
            continue
        low = s.lower()
        if "/storage/uploads/!spektakli/" not in low:
            continue
        if "/images/bimage/" in low:
            continue
        if not re.search(r"(?i)\\.(?:jpg|jpeg|png|webp)(?:\\?.*)?$", low):
            continue
        if s not in candidates:
            candidates.append(s)
    return candidates[0] if candidates else None


@dataclass
class SourceParsingStats:
    """Statistics for a source parsing run."""
    source: str
    total_received: int = 0
    new_added: int = 0
    ticket_updated: int = 0
    already_exists: int = 0
    failed: int = 0
    skipped: int = 0  # Explicitly skipped by Smart Update (e.g. no changes / promo filters)
    added_event_ids: list[int] = field(default_factory=list)
    updated_event_ids: list[int] = field(default_factory=list)  # For displaying Telegraph links


@dataclass
class SourceParsingResult:
    """Complete result of a source parsing run."""
    stats_by_source: dict[str, SourceParsingStats] = field(default_factory=dict)
    total_events: int = 0
    kernel_duration: float = 0.0
    processing_duration: float = 0.0
    log_file_path: str = ""
    json_file_paths: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chat_id: int | None = None  # For progress messages
    added_events: list["AddedEventInfo"] = field(default_factory=list)
    updated_events: list["UpdatedEventInfo"] = field(default_factory=list)


@dataclass
class AddedEventInfo:
    """Newly added event with Telegraph link for reporting."""
    event_id: int
    title: str
    telegraph_url: str
    ics_url: str | None
    log_cmd: str | None
    date: str | None
    time: str | None
    source: str | None
    source_url: str | None = None
    fact_stats: dict[str, int] | None = None
    source_ordinal: int | None = None
    source_total: int | None = None
    photo_count: int | None = None
    added_posters: int | None = None


@dataclass
class UpdatedEventInfo:
    """Updated event (ticket status change) with Telegraph link for reporting."""
    event_id: int
    title: str
    telegraph_url: str
    ics_url: str | None
    log_cmd: str | None
    date: str | None
    time: str | None
    source: str | None
    update_type: str  # 'ticket_status', 'full_update'
    source_url: str | None = None
    fact_stats: dict[str, int] | None = None
    source_ordinal: int | None = None
    source_total: int | None = None
    photo_count: int | None = None
    added_posters: int | None = None


def _event_telegraph_url(event) -> str | None:
    url = getattr(event, "telegraph_url", None)
    if url:
        return url
    path = getattr(event, "telegraph_path", None)
    if path:
        return f"https://telegra.ph/{path.lstrip('/')}"
    return None


def _normalize_source_url_for_match(url: str | None) -> str | None:
    if not url:
        return None
    value = str(url).strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        value = value.rstrip("/")
    return value


def _extract_host(url: str | None) -> str:
    normalized = _normalize_source_url_for_match(url)
    if not normalized:
        return ""
    try:
        return (urlparse(normalized).hostname or "").lower()
    except Exception:
        return ""


def _host_matches_source(host: str, source_name: str | None) -> bool:
    if not host or not source_name:
        return False
    hints = _SOURCE_HOST_HINTS.get(str(source_name).strip().lower(), ())
    return any(hint in host for hint in hints)


async def _load_latest_source_fact_stats(
    db: Database,
    *,
    event_id: int,
    source_url: str | None,
) -> dict[str, int] | None:
    """Return per-status fact counts for the most recent log batch for (event_id, source_url)."""
    if not event_id or not source_url:
        return None

    from sqlalchemy import func
    from models import EventSourceFact

    normalized = _normalize_source_url_for_match(source_url) or str(source_url).strip() or None
    if not normalized:
        return None

    async with db.get_session() as session:
        source = (
            await session.execute(
                select(EventSource).where(
                    EventSource.event_id == int(event_id),
                    EventSource.source_url.in_(
                        [normalized, str(source_url).strip(), normalized.rstrip("/")]
                    ),
                )
            )
        ).scalar_one_or_none()
        if not source:
            return None
        ts = await session.scalar(
            select(func.max(EventSourceFact.created_at)).where(
                EventSourceFact.event_id == int(event_id),
                EventSourceFact.source_id == int(source.id),
            )
        )
        if not ts:
            return None
        rows = (
            await session.execute(
                select(EventSourceFact.status, func.count())
                .where(
                    EventSourceFact.event_id == int(event_id),
                    EventSourceFact.source_id == int(source.id),
                    EventSourceFact.created_at == ts,
                )
                .group_by(EventSourceFact.status)
            )
        ).all()

    out: dict[str, int] = {}
    for status, cnt in rows:
        key = (str(status or "added")).strip().lower() or "added"
        out[key] = int(cnt or 0)
    return out or None


async def _load_event_source_ordinal(
    db: Database,
    *,
    event_id: int,
    source_url: str | None,
) -> tuple[int | None, int | None]:
    if not event_id or not source_url:
        return None, None
    normalized = _normalize_source_url_for_match(source_url) or str(source_url).strip() or None
    if not normalized:
        return None, None
    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(EventSource.id, EventSource.source_url)
                .where(EventSource.event_id == int(event_id))
                .order_by(EventSource.imported_at.asc(), EventSource.id.asc())
            )
        ).all()
    if not rows:
        return None, None
    urls = [str(url or "").strip() for _sid, url in rows]
    norm_urls = [_normalize_source_url_for_match(url) or url for url in urls]
    total = len(rows)
    for idx, url in enumerate(norm_urls):
        if not url:
            continue
        if url == normalized or url.rstrip("/") == normalized.rstrip("/"):
            return idx + 1, total
    # Fallback: try raw match.
    for idx, url in enumerate(urls):
        if url == normalized or url.rstrip("/") == normalized.rstrip("/"):
            return idx + 1, total
    return None, total


async def event_has_parser_source(
    db: Database,
    event_id: int,
    source_name: str | None,
    source_url: str | None = None,
) -> bool:
    """Check whether an event already has this parser source.

    We treat the source as present when:
    - event_source has matching `parser:<source_name>` source_type, or
    - event_source/source_post_url matches the same source URL, or
    - legacy source_post_url host belongs to this parser source.
    """
    if not event_id or not source_name:
        return False

    expected_type = f"parser:{str(source_name).strip().lower()}"
    expected_url = _normalize_source_url_for_match(source_url)
    expected_host = _extract_host(source_url)

    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        if not event:
            return False

        rows = (
            await session.execute(
                select(EventSource.source_type, EventSource.source_url).where(
                    EventSource.event_id == event_id
                )
            )
        ).all()

    for source_type, row_url in rows:
        stype = (source_type or "").strip().lower()
        if stype == expected_type:
            return True
        if expected_url and _normalize_source_url_for_match(row_url) == expected_url:
            return True
        if expected_host and _host_matches_source(_extract_host(row_url), source_name):
            return True

    legacy_url = _normalize_source_url_for_match(getattr(event, "source_post_url", None))
    if expected_url and legacy_url == expected_url:
        return True
    if _host_matches_source(_extract_host(legacy_url), source_name):
        return True
    return False


def unpack_add_event_result(
    raw: tuple[int | None, bool] | tuple[int | None, bool, str | None],
) -> tuple[int | None, bool, str]:
    """Normalize add_new_event_via_queue result.

    Backward-compatible with older 2-field tuples used in tests/mocks.
    """
    if not isinstance(raw, tuple):
        return None, False, "failed"
    if len(raw) >= 3:
        event_id, was_added, status = raw[0], bool(raw[1]), str(raw[2] or "")
        return event_id, was_added, status or ("created" if was_added else "merged")
    if len(raw) == 2:
        event_id, was_added = raw
        return event_id, bool(was_added), "created" if was_added else "merged"
    return None, False, "failed"


def classify_add_event_outcome(
    event_id: int | None,
    was_added: bool,
    status: str | None,
) -> str:
    if was_added or status == "created":
        return "added"
    st = (status or "").strip().lower()
    if st.startswith("skipped"):
        return "skipped"
    if event_id and st in {"merged", "updated"}:
        return "updated"
    if event_id and st:
        return "updated"
    return "failed"


def _format_parser_ticket_price(
    price_min: int | float | None,
    price_max: int | float | None,
) -> str:
    if price_min is None and price_max is None:
        return ""

    def _to_text(value: int | float | None) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)

    min_text = _to_text(price_min)
    max_text = _to_text(price_max)
    if min_text and max_text and min_text != max_text:
        return f"{min_text}-{max_text} RUB"
    if min_text:
        return f"от {min_text} RUB"
    if max_text:
        return f"до {max_text} RUB"
    return ""


def _format_parser_ticket_status(ticket_status: str | None) -> str:
    normalized = str(ticket_status or "").strip().lower()
    return {
        "available": "доступны",
        "sold_out": "распродано",
        "unknown": "неизвестно",
    }.get(normalized, normalized)


def _build_parser_source_text(
    theatre_event: TheatreEvent,
    *,
    full_description: str,
    location_name: str,
) -> str:
    """Build structured parser input for LLM draft extraction.

    Some site parsers, especially Qtickets, return short descriptions that omit
    date/time/venue entirely. Include parser facts explicitly so LLM extraction
    stays grounded and does not return an empty draft batch.
    """

    lines = [f"Название: {theatre_event.title}"]

    date_text = str(theatre_event.parsed_date or theatre_event.date_raw or "").strip()
    if date_text:
        lines.append(f"Дата: {date_text}")

    time_text = str(theatre_event.parsed_time or "").strip()
    if time_text and time_text != "00:00":
        lines.append(f"Время: {time_text}")

    venue_text = str(location_name or theatre_event.location or "").strip()
    if venue_text:
        lines.append(f"Площадка: {venue_text}")

    age_text = str(theatre_event.age_restriction or "").strip()
    if age_text:
        lines.append(f"Возраст: {age_text}")

    ticket_status_text = _format_parser_ticket_status(theatre_event.ticket_status)
    if ticket_status_text:
        lines.append(f"Статус билетов: {ticket_status_text}")

    price_text = _format_parser_ticket_price(
        theatre_event.ticket_price_min,
        theatre_event.ticket_price_max,
    )
    if price_text:
        lines.append(f"Цена: {price_text}")

    url_text = str(theatre_event.url or "").strip()
    if url_text:
        lines.append(f"Ссылка: {url_text}")

    body = str(full_description or "").strip()
    if body:
        lines.extend(["", "Описание:", body])

    return "\n".join(lines)


async def _ensure_telegraph_url(db: Database, event_id: int) -> str | None:
    import sys

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if not main_mod or not hasattr(main_mod, "update_telegraph_event_page"):
        logger.warning("source_parsing: main module missing for telegraph build")
        return None
    try:
        return await main_mod.update_telegraph_event_page(event_id, db, None)
    except Exception as e:
        logger.warning(
            "source_parsing: telegraph build failed event_id=%d error=%s",
            event_id,
            e,
        )
        return None


async def build_added_event_info(
    db: Database,
    event_id: int,
    source: str | None,
    *,
    source_url: str | None = None,
) -> AddedEventInfo | None:
    from models import Event

    async with db.get_session() as session:
        event = await session.get(Event, event_id)

    if not event:
        return None

    url = _event_telegraph_url(event)
    if not url:
        await _ensure_telegraph_url(db, event_id)
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
        if not event:
            return None
        url = _event_telegraph_url(event)

    if not url:
        logger.warning(
            "source_parsing: telegraph url missing after build event_id=%d",
            event_id,
        )

    fact_stats = None
    try:
        fact_stats = await _load_latest_source_fact_stats(
            db,
            event_id=int(event_id),
            source_url=source_url,
        )
    except Exception:
        fact_stats = None

    source_ordinal = None
    source_total = None
    try:
        source_ordinal, source_total = await _load_event_source_ordinal(
            db,
            event_id=int(event_id),
            source_url=source_url,
        )
    except Exception:
        source_ordinal = None
        source_total = None

    photo_count = None
    try:
        raw = getattr(event, "photo_count", None)
        if raw is None:
            urls = getattr(event, "photo_urls", None)
            if isinstance(urls, list):
                raw = len([u for u in urls if (str(u or "").strip())])
        if raw is not None:
            photo_count = int(raw or 0)
    except Exception:
        photo_count = None

    return AddedEventInfo(
        event_id=event_id,
        title=event.title or "",
        telegraph_url=url or "",
        ics_url=getattr(event, "ics_url", None),
        log_cmd=f"/log {event_id}",
        date=getattr(event, "date", None),
        time=getattr(event, "time", None),
        source=source,
        source_url=source_url,
        fact_stats=fact_stats,
        source_ordinal=source_ordinal,
        source_total=source_total,
        photo_count=photo_count,
    )


async def build_updated_event_info(
    db: Database,
    event_id: int,
    source: str | None,
    update_type: str,
    *,
    source_url: str | None = None,
) -> UpdatedEventInfo | None:
    """Build UpdatedEventInfo for a modified event."""
    from models import Event

    async with db.get_session() as session:
        event = await session.get(Event, event_id)

    if not event:
        return None

    url = _event_telegraph_url(event)
    # Don't force rebuild for updates - URL should already exist
    if not url:
        logger.debug(
            "source_parsing: no telegraph url for updated event_id=%d",
            event_id,
        )

    fact_stats = None
    try:
        fact_stats = await _load_latest_source_fact_stats(
            db,
            event_id=int(event_id),
            source_url=source_url,
        )
    except Exception:
        fact_stats = None

    source_ordinal = None
    source_total = None
    try:
        source_ordinal, source_total = await _load_event_source_ordinal(
            db,
            event_id=int(event_id),
            source_url=source_url,
        )
    except Exception:
        source_ordinal = None
        source_total = None

    photo_count = None
    try:
        raw = getattr(event, "photo_count", None)
        if raw is None:
            urls = getattr(event, "photo_urls", None)
            if isinstance(urls, list):
                raw = len([u for u in urls if (str(u or "").strip())])
        if raw is not None:
            photo_count = int(raw or 0)
    except Exception:
        photo_count = None

    return UpdatedEventInfo(
        event_id=event_id,
        title=event.title or "",
        telegraph_url=url or "",
        ics_url=getattr(event, "ics_url", None),
        log_cmd=f"/log {event_id}",
        date=getattr(event, "date", None),
        time=getattr(event, "time", None),
        source=source,
        source_url=source_url,
        fact_stats=fact_stats,
        update_type=update_type,
        source_ordinal=source_ordinal,
        source_total=source_total,
        photo_count=photo_count,
    )


async def download_images(urls: list[str]) -> list[tuple[bytes, str]]:
    """Download images from URLs."""
    import main as main_mod
    session = main_mod.get_http_session()
    result = []
    seen = set()
    
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        try:
            # Use short timeout to avoid blocking
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    # Try to get filename from URL or content-disposition
                    name = url.split("/")[-1] or "image.jpg"
                    result.append((data, name))
        except Exception as e:
            logger.warning("source_parsing: download failed url=%s error=%s", url, e)
            
    return result


async def update_event_ticket_status(
    db: Database,
    event_id: int,
    ticket_status: str,
    ticket_link: str | None = None,
) -> bool:
    """Update ticket status for an existing event.
    
    Args:
        db: Database instance
        event_id: Event ID to update
        ticket_status: New ticket status
        ticket_link: Optional ticket link to update
    
    Returns:
        True if updated successfully
    """
    from models import Event
    
    try:
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
            if not event:
                return False
            
            old_status = event.ticket_status
            event.ticket_status = ticket_status
            
            if ticket_link and not event.ticket_link:
                event.ticket_link = ticket_link
            
            await session.commit()
            
            logger.info(
                "source_parsing: updated ticket_status event_id=%d old=%s new=%s",
                event_id,
                old_status,
                ticket_status,
            )
            
            # Rebuild Telegraph page with updated data
            await _ensure_telegraph_url(db, event_id)
            
            return True
    except Exception as e:
        logger.error(
            "source_parsing: update failed event_id=%d error=%s",
            event_id,
            e,
        )
        return False


async def update_event_full(
    db: Database,
    event_id: int,
    theatre_event: TheatreEvent,
) -> bool:
    """Fully update an event with new data (for placeholder 00:00 events).
    
    Args:
        db: Database instance
        event_id: Event ID to update
        theatre_event: New event data
    
    Returns:
        True if updated successfully
    """
    from models import Event
    
    try:
        async with db.get_session() as session:
            event = await session.get(Event, event_id)
            if not event:
                return False
            
            # Update time
            if theatre_event.parsed_time and theatre_event.parsed_time != "00:00":
                event.time = theatre_event.parsed_time
            
            # Update ticket info
            event.ticket_status = theatre_event.ticket_status
            if theatre_event.url:
                event.ticket_link = theatre_event.url
            
            # Update pushkin card
            event.pushkin_card = theatre_event.pushkin_card
            
            # Don't overwrite description with raw parser text
            # Description should only come from LLM (short_description)
            # Keeping existing description if present
            
            # Update photos
            photos = limit_photos_for_source(
                theatre_event.photos,
                theatre_event.source_type,
            )
            if photos and not event.photo_urls:
                event.photo_urls = photos
                event.photo_count = len(photos)
            
            await session.commit()
            
            logger.info(
                "source_parsing: full update event_id=%d time=%s",
                event_id,
                event.time,
            )
            return True
    except Exception as e:
        logger.error(
            "source_parsing: full update failed event_id=%d error=%s",
            event_id,
            e,
        )
        return False


async def update_linked_events(
    db: Database,
    event_id: int,
    location_name: str,
    title: str,
) -> None:
    """Update linked_event_ids for an event and its related events.
    
    Args:
        db: Database instance
        event_id: Event ID to update
        location_name: Location name for finding linked events
        title: Title for fuzzy matching
    """
    # NOTE: location_name/title are kept for backward compatibility with callers.
    # The canonical recompute uses the stored Event fields to avoid drift.
    try:
        from linked_events import recompute_linked_event_ids

        res = await recompute_linked_event_ids(db, int(event_id))
        if res.changed_event_ids:
            logger.info(
                "source_parsing: linked events recomputed event_id=%d group=%d changed=%d capped=%s",
                event_id,
                len(res.group_event_ids or []),
                len(res.changed_event_ids or []),
                str(bool(getattr(res, "capped", False))).lower(),
            )
            # Refresh linked occurrences in Telegraph so the public "Другие даты" infoblock line
            # stays consistent across the whole group.
            refresh_ids = [
                int(x)
                for x in (res.changed_event_ids or [])
                if int(x) and int(x) != int(event_id)
            ]
            if refresh_ids:
                import sys

                main_mod = sys.modules.get("main") or sys.modules.get("__main__")
                if main_mod and hasattr(main_mod, "enqueue_job") and hasattr(main_mod, "JobTask"):
                    try:
                        enqueue_job = getattr(main_mod, "enqueue_job")
                        job_task = getattr(getattr(main_mod, "JobTask"), "telegraph_build")
                        for rid in refresh_ids[:80]:
                            await enqueue_job(db, int(rid), job_task, depends_on=None)
                    except Exception:
                        logger.warning(
                            "source_parsing: failed to enqueue linked telegraph refresh event_id=%d",
                            event_id,
                            exc_info=True,
                        )
    except Exception as e:
        logger.warning(
            "source_parsing: linking failed event_id=%d error=%s",
            event_id,
            e,
        )


async def schedule_existing_event_update(db: Database, event_id: int) -> None:
    """Enqueue deferred page rebuilds for an updated existing event."""
    import sys
    from models import Event

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if not main_mod or not hasattr(main_mod, "schedule_event_update_tasks"):
        logger.warning(
            "source_parsing: schedule_event_update_tasks unavailable event_id=%d",
            event_id,
        )
        return

    async with db.get_session() as session:
        event = await session.get(Event, event_id)
    if not event:
        return

    await main_mod.schedule_event_update_tasks(
        db,
        event,
        drain_nav=False,
        skip_vk_sync=True,
    )


async def add_new_event_via_queue(
    db: Database,
    bot: Bot | None,
    theatre_event: TheatreEvent,
    progress_current: int,
    progress_total: int,
    poster_media: Sequence[PosterMedia] | None = None,
) -> tuple[int | None, bool, str]:
    """Add a new event through the existing LLM queue system.
    
    Uses build_event_drafts_from_vk for consistent event creation.
    
    Args:
        db: Database instance
        bot: Telegram bot for notifications
        theatre_event: Event data from parser
        progress_current: Current progress number
        progress_total: Total events to process
    
    Returns:
        Tuple: (event_id, was_added, smart_update_status)
    """
    from vk_intake import build_event_drafts_from_vk
    import main as main_mod
    
    try:
        # Build description with all available info
        description_parts = []
        if theatre_event.description:
            description_parts.append(theatre_event.description)
        if theatre_event.age_restriction:
            description_parts.append(f"Возраст: {theatre_event.age_restriction}")
        if theatre_event.scene:
            description_parts.append(f"Сцена: {theatre_event.scene}")
        
        location_name = normalize_location_name(theatre_event.location, theatre_event.scene)
        full_description = "\n\n".join(description_parts) if description_parts else theatre_event.title
        source_text = _build_parser_source_text(
            theatre_event,
            full_description=full_description,
            location_name=location_name,
        )
        
        # Limit photos for source
        base_photos = list(theatre_event.photos or [])
        if not base_photos and theatre_event.url:
            og = await _fetch_og_image_for_dramteatr(theatre_event.url)
            if og:
                base_photos = [og]
        photos = limit_photos_for_source(
            base_photos,
            theatre_event.source_type,
        )
        
        # Prefer Catbox URLs from poster_media if available
        if poster_media:
            catbox_photos = [
                p.catbox_url for p in poster_media 
                if p.catbox_url
            ]
            if catbox_photos:
                # Use catbox photos, but still respect source limits if needed
                # (though usually we want all processed photos)
                photos = catbox_photos
        
        # Log progress
        logger.info(
            "source_parsing: adding event %d/%d title=%s location=%s",
            progress_current,
            progress_total,
            theatre_event.title[:50],
            location_name,
        )
        
        # Use existing event creation logic
        diag_enabled = bool(SOURCE_PARSING_DIAG_TITLE) and SOURCE_PARSING_DIAG_TITLE in theatre_event.title.lower()
        llm_start = time.monotonic()
        try:
            drafts, _ = await asyncio.wait_for(
                build_event_drafts_from_vk(
                    text=source_text,
                    source_name=f"theatre:{theatre_event.source_type}",
                    location_hint=location_name,
                    default_ticket_link=theatre_event.url,
                    poster_media=poster_media,
                ),
                timeout=PARSE_EVENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                "source_parsing: LLM timeout after %ss title=%s",
                PARSE_EVENT_TIMEOUT_SECONDS,
                theatre_event.title[:50],
            )
            return None, False, "llm_timeout"
        if diag_enabled:
            logger.info(
                "source_parsing: diag LLM done title=%s drafts=%d duration=%.2fs",
                theatre_event.title[:120],
                len(drafts or []),
                time.monotonic() - llm_start,
            )
        
        if not drafts:
            logger.warning(
                "source_parsing: no drafts returned title=%s",
                theatre_event.title,
            )
            return None, False, "llm_no_drafts"
        
        draft = drafts[0]
        
        # Override with parsed values
        if theatre_event.parsed_date:
            draft.date = theatre_event.parsed_date
        if theatre_event.parsed_time:
            draft.time = theatre_event.parsed_time
            
        # Override prices if available
        if theatre_event.ticket_price_min is not None:
            draft.ticket_price_min = theatre_event.ticket_price_min
        if theatre_event.ticket_price_max is not None:
            draft.ticket_price_max = theatre_event.ticket_price_max
        
        draft.venue = location_name
        draft.ticket_link = theatre_event.url
        draft.pushkin_card = theatre_event.pushkin_card
        
        # Use smart update (no VK posting here)
        try:
            import sys
            import hashlib
            from datetime import datetime, timezone
            from models import Event
            from smart_event_update import EventCandidate, PosterCandidate, smart_event_update
            
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if main_mod is None:
                logger.error("source_parsing: main module not found")
                return None, False, "main_module_missing"
            
            # Build final description - should come from LLM (short_description)
            # If LLM didn't return it, use title as fallback and log warning
            final_description = draft.description
            if not final_description:
                logger.warning(
                    "source_parsing: LLM returned empty short_description title=%s",
                    theatre_event.title[:50],
                )
                final_description = draft.title

            posters: list[PosterCandidate] = [
                PosterCandidate(
                    catbox_url=None
                    if is_supabase_storage_url(item.catbox_url)
                    else item.catbox_url,
                    supabase_url=item.supabase_url
                    or (
                        item.catbox_url
                        if is_supabase_storage_url(item.catbox_url)
                        else None
                    ),
                    sha256=item.digest,
                    phash=None,
                    ocr_text=item.ocr_text,
                    ocr_title=item.ocr_title,
                )
                for item in (poster_media or [])
            ]
            if not posters and photos:
                posters = [
                    PosterCandidate(
                        catbox_url=(url if not is_supabase_storage_url(url) else None),
                        supabase_url=(url if is_supabase_storage_url(url) else None),
                    )
                    for url in photos
                ]

            fallback_key = "|".join(
                [
                    theatre_event.source_type or "theatre",
                    draft.title or "",
                    str(draft.date or ""),
                    str(draft.time or ""),
                    location_name or "",
                ]
            )
            fallback_hash = hashlib.sha256(fallback_key.encode("utf-8")).hexdigest()[:16]
            source_url = theatre_event.url or f"parser:{theatre_event.source_type}:{fallback_hash}"

            candidate = EventCandidate(
                source_type=f"parser:{theatre_event.source_type}",
                source_url=source_url,
                source_text=full_description or draft.source_text or draft.title,
                title=draft.title,
                date=draft.date or datetime.now(timezone.utc).date().isoformat(),
                time=draft.time or "00:00",
                end_date=draft.end_date or None,
                festival=(draft.festival or None),
                location_name=draft.venue or "",
                location_address=draft.location_address or None,
                city=draft.city or None,
                # Site/parser sources are canonical: in conflicts they should override
                # lower-trust sources like Telegram.
                trust_level="high",
                ticket_price_min=draft.ticket_price_min,
                ticket_price_max=draft.ticket_price_max,
                ticket_link=theatre_event.url,  # Always use parser URL, not LLM links
                event_type=draft.event_type or None,
                emoji=draft.emoji or None,
                is_free=bool(draft.is_free),
                pushkin_card=bool(draft.pushkin_card),
                search_digest=draft.search_digest,
                raw_excerpt=final_description,
                posters=posters,
            )

            logger.info(
                "source_parsing: smart_update candidate source=%s url=%s title=%s date=%s time=%s location=%s photos=%d posters=%d",
                theatre_event.source_type,
                source_url,
                theatre_event.title[:80],
                candidate.date,
                candidate.time,
                location_name,
                len(photos or []),
                len(posters),
            )

            update_result = await smart_event_update(
                db,
                candidate,
                check_source_url=False,
                schedule_tasks=False,
            )
            event_id = update_result.event_id
            was_added = bool(update_result.created)

            if not event_id:
                logger.error(
                    "source_parsing: smart_update failed title=%s status=%s reason=%s",
                    theatre_event.title[:80],
                    update_result.status,
                    update_result.reason,
                )
                return None, False, update_result.status

            async with db.get_session() as session:
                saved = await session.get(Event, event_id)

            if saved:
                schedule_event_update_tasks = main_mod.schedule_event_update_tasks
                await schedule_event_update_tasks(
                    db,
                    saved,
                    drain_nav=False,
                    skip_vk_sync=True,
                )
            
            logger.info(
                "source_parsing: smart_update result event_id=%d status=%s created=%s merged=%s",
                event_id,
                update_result.status,
                int(update_result.created),
                int(update_result.merged),
            )
            
            # Update ticket status
            await update_event_ticket_status(
                db,
                event_id,
                theatre_event.ticket_status,
                theatre_event.url,
            )
            
            # Update linked events
            await update_linked_events(
                db,
                event_id,
                location_name,
                theatre_event.title,
            )

            # Final consistency pass: force Telegraph rebuild after all parser-side
            # mutations (smart merge + ticket/link updates + linked-event updates).
            # This avoids serving an intermediate page revision when async jobs
            # touched Telegraph earlier in the same flow.
            try:
                await _ensure_telegraph_url(db, event_id)
            except Exception:
                logger.warning(
                    "source_parsing: final telegraph rebuild failed event_id=%d",
                    event_id,
                    exc_info=True,
                )
            
            return event_id, was_added, update_result.status
                
        except Exception as persist_err:
            logger.error(
                "source_parsing: persist failed title=%s error=%s",
                theatre_event.title,
                persist_err,
                exc_info=True,
            )
            return None, False, "persist_failed"
        
    except Exception as e:
        logger.error(
            "source_parsing: add failed title=%s error=%s",
            theatre_event.title,
            e,
            exc_info=True,
        )
        return None, False, "add_failed"


def escape_md(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    chars = "_*[]()~`>#+-=|{}.!"
    for c in chars:
        text = text.replace(c, f"\\{c}")
    return text


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    failure_msg = status.get("failureMessage") or status.get("failure_message")
    if not state:
        return "неизвестен"
    result = str(state)
    if failure_msg:
        result += f" ({failure_msg})"
    return result


def _format_kaggle_phase(phase: str) -> str:
    labels = {
        "prepare": "подготовка",
        "pushed": "запуск в Kaggle",
        "poll": "выполнение",
        "complete": "завершено",
        "failed": "ошибка",
        "timeout": "таймаут",
        "not_found": "kernel не найден",
        "metadata_missing": "нет kernel-metadata.json",
        "metadata_error": "ошибка метаданных",
        "push_failed": "ошибка отправки",
    }
    return labels.get(phase, phase)


def _format_kaggle_status_message(
    phase: str,
    kernel_ref: str,
    status: dict | None,
) -> str:
    lines = [
        "🛰️ Kaggle: ParseTheatres",
        f"Kernel: {kernel_ref or '—'}",
        f"Этап: {_format_kaggle_phase(phase)}",
    ]
    if status is not None:
        lines.append(f"Статус Kaggle: {_format_kaggle_status(status)}")
    return "\n".join(lines)


async def format_parsing_report(
    result: SourceParsingResult,
    *,
    bot_username: str | None = None,
    db: Database | None = None,
) -> str:
    """Format parsing result as a human-readable report.
    
    Args:
        result: Parsing result
    
    Returns:
        Formatted summary string
    """
    lines = [
        f"🏁 **Парсинг источников завершен**",
        f"Продолжительность: {result.processing_duration:.1f} сек",
        f"Всего событий: {result.total_events}",
        "",
        "**По источникам:**"
    ]
    
    total_added = 0
    total_updated = 0
    total_failed = 0
    total_skipped = 0
    
    for source, stats in result.stats_by_source.items():
        total_added += stats.new_added
        total_updated += stats.ticket_updated
        total_failed += stats.failed
        total_skipped += stats.skipped
        
        # Use descriptive labels if available
        source_label = {
            "dramteatr": "Драмтеатр",
            "muzteatr": "Музтеатр",
            "sobor": "Собор",
            "tretyakov": "🎨 Третьяковка",
            "philharmonia": "🎵 Филармония",
        }.get(source, source)
        
        lines.append(f"• **{escape_md(source_label)}**:")
        lines.append(f"  ✅ Добавлено: {stats.new_added}")
        if stats.ticket_updated:
            lines.append(f"  🔄 Обновлено: {stats.ticket_updated}")
        if stats.failed:
            lines.append(f"  ❌ Ошибок: {stats.failed}")
        if stats.skipped:
            lines.append(f"  ⏭️ Пропущено: {stats.skipped}")
    
    lines.append("")
    lines.append(f"**Итого:**")
    lines.append(f"✅ Всего добавлено: {total_added}")
    if total_updated:
        lines.append(f"🔄 Всего обновлено: {total_updated}")
    if total_failed:
        lines.append(f"❌ Всего ошибок: {total_failed}")
    
    if result.errors:
        lines.append("")
        lines.append("**Ошибки выполнения:**")
        # Show first 3 errors to avoid overflow
        for err in result.errors[:3]:
            # Escape error text as it may contain underscores/paths
            lines.append(f"⚠️ {escape_md(str(err))}")
        if len(result.errors) > 3:
            lines.append(f"... и еще {len(result.errors) - 3}")

    # Add JSON file paths if available
    if result.json_file_paths:
        lines.append("")
        lines.append("**Сохраненные файлы:**")
        for path in result.json_file_paths:
            lines.append(f"📄 {escape_md(Path(path).name)}")

    # Unified per-event operator report (compact, actionable links).
    bot_username_clean = (bot_username or "").strip().lstrip("@") or None
    def _log_deeplink(event_id: int) -> str | None:
        if not bot_username_clean:
            return None
        return f"https://t.me/{bot_username_clean}?start=log_{int(event_id)}"

    def _format_facts_and_photos(item: object) -> str:
        stats = getattr(item, "fact_stats", None) or {}
        photo_count = getattr(item, "photo_count", None)
        try:
            photos = int(photo_count or 0)
        except Exception:
            photos = 0
        added_posters_raw = getattr(item, "added_posters", None)
        try:
            added_posters = int(added_posters_raw) if added_posters_raw is not None else None
        except Exception:
            added_posters = None
        if added_posters is None:
            photos_label = f"Иллюстрации: {'⚠️0' if photos == 0 else photos}"
        else:
            photos_label = f"Иллюстрации: +{added_posters}, всего {'⚠️0' if photos == 0 else photos}"
        if stats:
            added = int(stats.get("added") or 0)
            dup = int(stats.get("duplicate") or 0)
            conf = int(stats.get("conflict") or 0)
            note = int(stats.get("note") or 0)
            return f"  Факты: ✅{added} ↩️{dup} ⚠️{conf} ℹ️{note} | {photos_label}"
        return f"  Факты: — | {photos_label}"

    added_items = list(result.added_events or [])
    updated_items = list(result.updated_events or [])
    ctx = None
    if db and (added_items or updated_items):
        try:
            from source_parsing.smart_update_report import build_smart_update_report_context

            all_items = added_items + updated_items
            eids = [int(getattr(i, "event_id", 0) or 0) for i in all_items]
            urls = [str(getattr(i, "source_url", "") or "").strip() for i in all_items]
            ctx = await build_smart_update_report_context(db, event_ids=eids, source_urls=urls)
        except Exception:
            ctx = None
    tz = getattr(ctx, "tz", None)
    sources_by_eid = getattr(ctx, "sources_by_event_id", None) or {}
    video_counts = getattr(ctx, "video_count_by_event_id", None) or {}
    ticket_queue_by_eid = getattr(ctx, "ticket_queue_by_event_id", None) or {}
    fest_queue_by_src = getattr(ctx, "festival_queue_by_source_url", None) or {}

    def _ics_line_md(url: str | None, *, has_time: bool) -> str:
        value = (url or "").strip()
        if value:
            return f"  ICS: [ics]({value})"
        return "  ICS: ⏳" if has_time else "  ICS: —"

    def _sources_lines_md(eid: int) -> list[str]:
        rows = list(sources_by_eid.get(int(eid)) or [])
        if not rows or not tz:
            return []
        from source_parsing.smart_update_report import format_dt_compact

        out: list[str] = ["  Источники:"]
        limit = 24
        shown = rows[:limit]
        for imported_at, url in shown:
            stamp = format_dt_compact(imported_at, tz)
            out.append(f"  {stamp} {escape_md(str(url))}")
        if len(rows) > limit:
            out.append(f"  … ещё {len(rows) - limit}")
        return out

    def _queue_lines_md(eid: int, source_url: str | None) -> list[str]:
        out: list[str] = []
        src = (source_url or "").strip()
        if src:
            fest = fest_queue_by_src.get(src)
            if fest:
                name = (
                    (getattr(fest, "festival_name", None) or "")
                    or (getattr(fest, "festival_full", None) or "")
                ).strip()
                ctx2 = (getattr(fest, "festival_context", None) or "").strip()
                status = (getattr(fest, "status", None) or "").strip()
                fid = getattr(fest, "id", None)
                tail = name or ctx2
                extra = f" {escape_md(tail)}" if tail else ""
                id_part = f" (id={int(fid)})" if isinstance(fid, int) and fid > 0 else ""
                st_part = f" {escape_md(status)}" if status else ""
                out.append(f"  🎪 festival_queue:{st_part}{extra}{id_part}".strip())

        tickets = list(ticket_queue_by_eid.get(int(eid)) or [])
        if tickets:
            first = tickets[0]
            url = str(getattr(first, "url", "") or "").strip()
            label = str(getattr(first, "site_kind", "") or "tickets").strip() or "tickets"
            extra = f" +{len(tickets)}" if len(tickets) > 1 else ""
            if url:
                out.append(f"  🎟 ticket_site_queue:{extra} [{escape_md(label)}]({url})")
            else:
                out.append(f"  🎟 ticket_site_queue:{extra}".strip())
        return out

    def _format_facts_photos_videos_md(item: object) -> str:
        stats = getattr(item, "fact_stats", None) or {}
        try:
            photos = int(getattr(item, "photo_count", None) or 0)
        except Exception:
            photos = 0
        added_posters_raw = getattr(item, "added_posters", None)
        try:
            added_posters = int(added_posters_raw) if added_posters_raw is not None else None
        except Exception:
            added_posters = None
        if added_posters is None:
            photos_label = f"Иллюстрации: {'⚠️0' if photos == 0 else photos}"
        else:
            photos_label = f"Иллюстрации: +{added_posters}, всего {'⚠️0' if photos == 0 else photos}"
        try:
            eid2 = int(getattr(item, "event_id", 0) or 0)
        except Exception:
            eid2 = 0
        try:
            videos_total = int(video_counts.get(int(eid2), 0) or 0)
        except Exception:
            videos_total = 0
        videos_label = f" | Видео: {videos_total}" if videos_total > 0 else ""
        stats = getattr(item, "fact_stats", None) or {}
        if stats:
            added = int(stats.get("added") or 0)
            dup = int(stats.get("duplicate") or 0)
            conf = int(stats.get("conflict") or 0)
            note = int(stats.get("note") or 0)
            return f"  Факты: ✅{added} ↩️{dup} ⚠️{conf} ℹ️{note} | {photos_label}{videos_label}"
        return f"  Факты: — | {photos_label}{videos_label}"

    if added_items or updated_items:
        lines.append("")
        lines.append("**Smart Update (детали событий):**")
        if added_items:
            lines.append(f"✅ Созданные события: {len(added_items)}")
            for item in added_items[:12]:
                title = escape_md(item.title or "Без названия")
                meta: list[str] = []
                if item.date:
                    meta.append(str(item.date))
                if item.time:
                    meta.append(str(item.time))
                if item.telegraph_url:
                    lines.append(
                        f"• [{title}]({item.telegraph_url}) (id={item.event_id})"
                        + (f" — {' '.join(meta)}" if meta else "")
                    )
                else:
                    lines.append(f"• {title} (id={item.event_id})" + (f" — {' '.join(meta)}" if meta else ""))
                if getattr(item, "source_url", None):
                    lines.append(f"  Источник: {escape_md(str(item.source_url))}")
                if item.log_cmd:
                    link = _log_deeplink(int(item.event_id))
                    if link:
                        cmd = escape_md(item.log_cmd)
                        lines.append(f"  Лог: [{cmd}]({link})")
                    else:
                        lines.append(f"  Лог: {escape_md(item.log_cmd)}")
                try:
                    eid_i = int(getattr(item, "event_id", 0) or 0)
                except Exception:
                    eid_i = 0
                if eid_i:
                    lines.extend(_sources_lines_md(eid_i))
                lines.append(_ics_line_md(item.ics_url, has_time=bool((item.time or "").strip())))
                lines.append(_format_facts_photos_videos_md(item))
                lines.extend(_queue_lines_md(eid_i, getattr(item, "source_url", None)))
            if len(added_items) > 12:
                lines.append(f"... ещё {len(added_items) - 12}")
        if updated_items:
            lines.append(f"🔄 Обновлённые события: {len(updated_items)}")
            for item in updated_items[:12]:
                title = escape_md(item.title or "Без названия")
                meta = []
                if item.date:
                    meta.append(str(item.date))
                if item.time:
                    meta.append(str(item.time))
                if item.telegraph_url:
                    lines.append(
                        f"• [{title}]({item.telegraph_url}) (id={item.event_id})"
                        + (f" — {' '.join(meta)}" if meta else "")
                    )
                else:
                    lines.append(f"• {title} (id={item.event_id})" + (f" — {' '.join(meta)}" if meta else ""))
                if getattr(item, "source_url", None):
                    lines.append(f"  Источник: {escape_md(str(item.source_url))}")
                if item.log_cmd:
                    link = _log_deeplink(int(item.event_id))
                    if link:
                        cmd = escape_md(item.log_cmd)
                        lines.append(f"  Лог: [{cmd}]({link})")
                    else:
                        lines.append(f"  Лог: {escape_md(item.log_cmd)}")
                try:
                    eid_i = int(getattr(item, "event_id", 0) or 0)
                except Exception:
                    eid_i = 0
                if eid_i:
                    lines.extend(_sources_lines_md(eid_i))
                lines.append(_ics_line_md(item.ics_url, has_time=bool((item.time or "").strip())))
                lines.append(_format_facts_photos_videos_md(item))
                lines.extend(_queue_lines_md(eid_i, getattr(item, "source_url", None)))
            if len(updated_items) > 12:
                lines.append(f"... ещё {len(updated_items) - 12}")
            
    return "\n".join(lines)


async def _process_parsing_files(
    db: Database,
    bot: Bot | None,
    *,
    chat_id: int | None,
    theatre_files: list[str],
    phil_files: list[str],
    qtickets_files: list[str],
    result: SourceParsingResult,
    only_sources: set[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    events_by_source: dict[str, list[TheatreEvent]] = {}

    def _normalize_source_name(raw_name: str) -> str:
        normalized = raw_name.strip().lower()
        normalized = re.sub(r"\s*\(\d+\)\s*$", "", normalized)
        if "dramteatr" in normalized or "драм" in normalized:
            return "dramteatr"
        if "muzteatr" in normalized or "муз" in normalized:
            return "muzteatr"
        if "sobor" in normalized or "собор" in normalized:
            return "sobor"
        if "tretyakov" in normalized or "трет" in normalized:
            return "tretyakov"
        return normalized

    for file_path_str in theatre_files:
        try:
            file_path = Path(file_path_str)
            source_name = _normalize_source_name(file_path.stem)
            raw_content = file_path.read_text(encoding="utf-8")
            events = parse_theatre_json(raw_content, source_name)
            if events:
                events_by_source[source_name] = events
        except Exception as e:
            logger.error("source_parsing: failed to parse theatre file %s: %s", file_path_str, e)
            result.errors.append(f"File {file_path_str}: {str(e)}")

    if phil_files:
        try:
            from source_parsing.philharmonia import parse_philharmonia_output
            p_events = parse_philharmonia_output(phil_files)
            if p_events:
                events_by_source["philharmonia"] = p_events
        except Exception as e:
            logger.error("source_parsing: failed to parse philharmonia files: %s", e)
            result.errors.append(f"Philharmonia parse error: {str(e)}")

    if qtickets_files:
        try:
            from source_parsing.qtickets import parse_qtickets_output
            q_events = parse_qtickets_output(qtickets_files)
            if q_events:
                events_by_source["qtickets"] = q_events
        except Exception as e:
            logger.error("source_parsing: failed to parse qtickets files: %s", e)
            result.errors.append(f"Qtickets parse error: {str(e)}")

    if only_sources:
        # Keep only requested sources to reduce LLM calls during E2E / targeted runs.
        # Note: kernels may still have produced other JSONs; we just skip processing.
        events_by_source = {
            src: evs
            for src, evs in events_by_source.items()
            if src in only_sources or "all" in only_sources or "theatres" in only_sources
        }

    if date_from or date_to:
        def _in_range(d: str | None) -> bool:
            if not d:
                return False
            try:
                # Lexicographic compare works for ISO dates, but keep parsing defensive.
                _ = datetime.fromisoformat(d)
            except Exception:
                return False
            if date_from and d < date_from:
                return False
            if date_to and d > date_to:
                return False
            return True

        filtered: dict[str, list[TheatreEvent]] = {}
        for src, evs in events_by_source.items():
            kept = [e for e in evs if _in_range(getattr(e, "parsed_date", None))]
            if kept:
                filtered[src] = kept
        events_by_source = filtered

    total_count = sum(len(ev) for ev in events_by_source.values())
    result.total_events = total_count

    current_index = 0
    progress_message_id = None

    for source, events in events_by_source.items():
        try:
            stats, progress_message_id = await process_source_events(
                db,
                bot,
                events,
                source,
                current_index,
                total_count,
                chat_id=chat_id,
                progress_message_id=progress_message_id,
                added_events=result.added_events if chat_id else None,
                updated_events=result.updated_events if chat_id else None,
            )
            result.stats_by_source[source] = stats
            current_index += len(events)
        except Exception as e:
            logger.error(
                "source_parsing: failed to process events from %s: %s",
                source,
                e,
                exc_info=True,
            )
            result.errors.append(f"Source {escape_md(source)}: {escape_md(str(e))}")

    months = sorted({
        str(event.parsed_date)[:7]
        for events in events_by_source.values()
        for event in events
        if event.parsed_date
    })
    if months:
        logger.info("source_parsing: ensuring month_pages tasks for months=%s", months)
        try:
            import sys
            from sqlalchemy import select
            from models import Event, JobTask
        except Exception as e:
            logger.error(
                "source_parsing: failed to prepare month_pages enqueue: %s",
                e,
            )
        else:
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if main_mod is None:
                logger.error("source_parsing: main module not found for month_pages enqueue")
            else:
                enqueue_job = main_mod.enqueue_job
                mark_pages_dirty = main_mod.mark_pages_dirty
                weekend_start_for_date = getattr(main_mod, "weekend_start_for_date", None)
                deferred_time = datetime.now(timezone.utc) + timedelta(minutes=15)

                def _coerce_date(value: object) -> date | None:
                    if isinstance(value, date):
                        return value
                    try:
                        raw = str(value or "").strip()
                        return date.fromisoformat(raw) if raw else None
                    except Exception:
                        return None

                month_event_ids: dict[str, int] = {}
                weekend_event_ids: dict[str, int] = {}
                weekend_starts: set[str] = set()
                async with db.get_session() as session:
                    for month in months:
                        res = await session.execute(
                            select(Event.id)
                            .where(Event.date.like(f"{month}%"))
                            .order_by(Event.id.desc())
                            .limit(1)
                        )
                        event_id = res.scalar_one_or_none()
                        if event_id:
                            month_event_ids[month] = event_id
                    for events in events_by_source.values():
                        for parsed_event in events:
                            day = _coerce_date(getattr(parsed_event, "parsed_date", None))
                            if not day:
                                continue
                            weekend_start: date | None = None
                            if callable(weekend_start_for_date):
                                try:
                                    weekend_start = weekend_start_for_date(day)
                                except Exception:
                                    weekend_start = None
                            elif day.weekday() >= 5:
                                weekend_start = day - timedelta(days=day.weekday() - 5)
                            if weekend_start:
                                weekend_starts.add(weekend_start.isoformat())

                    for weekend_start in sorted(weekend_starts):
                        sunday = _coerce_date(weekend_start)
                        sunday_iso = (
                            (sunday + timedelta(days=1)).isoformat() if sunday else None
                        )
                        event_id = None
                        if sunday_iso:
                            sat_res = await session.execute(
                                select(Event.id)
                                .where(
                                    Event.date.like(f"{weekend_start}%")
                                    | Event.date.like(f"{sunday_iso}%")
                                )
                                .order_by(Event.id.desc())
                                .limit(1)
                            )
                            event_id = sat_res.scalar_one_or_none()
                        if event_id:
                            weekend_event_ids[weekend_start] = event_id

                for month in months:
                    event_id = month_event_ids.get(month)
                    if not event_id:
                        continue
                    await enqueue_job(
                        db,
                        event_id,
                        JobTask.month_pages,
                        coalesce_key=f"month_pages:{month}",
                        next_run_at=deferred_time,
                    )
                    await mark_pages_dirty(db, month)
                for weekend_start in sorted(weekend_starts):
                    event_id = weekend_event_ids.get(weekend_start)
                    if not event_id:
                        continue
                    await enqueue_job(
                        db,
                        event_id,
                        JobTask.weekend_pages,
                        coalesce_key=f"weekend_pages:{weekend_start}",
                        next_run_at=deferred_time,
                    )
                    await mark_pages_dirty(db, f"weekend:{weekend_start}")

    if bot and chat_id and progress_message_id:
        try:
            total_new = sum(s.new_added for s in result.stats_by_source.values())
            total_fail = sum(s.failed for s in result.stats_by_source.values())

            final_text = (
                f"✅ Обработка завершена\n"
                f"Добавлено: {total_new}\n"
                f"Ошибок: {total_fail}"
            )
            await bot.edit_message_text(
                text=final_text,
                chat_id=chat_id,
                message_id=progress_message_id,
                parse_mode="Markdown",
            )
        except Exception:
            logger.warning(
                "source_parsing: failed final progress update",
                exc_info=True,
            )


async def run_source_parsing(
    db: Database,
    bot: Bot | None = None,
    chat_id: int | None = None,
    only_sources: Sequence[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    trigger: str = "manual",
    operator_id: int | None = None,
    run_id: str | None = None,
) -> SourceParsingResult:
    """Run full source parsing pipeline.
    
    Args:
        db: Database instance
        bot: Optional bot instance for progress updates
        chat_id: Optional chat ID for progress updates
    
    Returns:
        Result statistics
    """
    start_time = time.time()
    result = SourceParsingResult(chat_id=chat_id)
    kaggle_status_message_id: int | None = None
    kaggle_kernel_ref = ""
    parse_run_id = run_id or uuid.uuid4().hex[:8]
    ops_run_id = await start_ops_run(
        db,
        kind="parse",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={
            "run_id": run_id,
            "only_sources": list(only_sources or []),
            "date_from": date_from,
            "date_to": date_to,
        },
    )
    ops_status = "success"
    ops_error: str | None = None
    log_handler: logging.Handler | None = None

    class _SourceParsingFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.name.startswith("source_parsing")

    debug_dir_env = (os.getenv("SOURCE_PARSING_DEBUG_DIR") or "").strip()
    if debug_dir_env:
        log_dir = Path(debug_dir_env)
    else:
        # In Fly production /data is a writable volume; in local/dev it may be protected.
        if os.path.isdir("/data") and os.access("/data", os.W_OK):
            log_dir = Path("/data/parse_debug")
        else:
            log_dir = Path("artifacts/run/parse_debug")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("source_parsing: failed to init debug dir %s: %s", log_dir, e)
    log_path = log_dir / f"source_parsing_{parse_run_id}.log"
    try:
        log_handler = logging.FileHandler(log_path, encoding="utf-8")
        log_handler.setLevel(logging.INFO)
        log_handler.addFilter(_SourceParsingFilter())
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(log_handler)
        result.log_file_path = str(log_path)
    except Exception as e:
        logger.warning("source_parsing: failed to init log file: %s", e)

    async def _update_kaggle_status(
        phase: str,
        kernel_ref: str,
        status: dict | None,
    ) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        kaggle_kernel_ref = kernel_ref or kaggle_kernel_ref
        if not bot or not chat_id:
            return
        text = _format_kaggle_status_message(phase, kernel_ref, status)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text)
                kaggle_status_message_id = sent.message_id
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                )
        except Exception:
            logger.warning(
                "source_parsing: failed to update kaggle status message",
                exc_info=True,
            )
    
    # Import Philharmonia and Qtickets runners
    from source_parsing.philharmonia import (
        run_philharmonia_kaggle_kernel,
    )
    from source_parsing.qtickets import (
        run_qtickets_kaggle_kernel,
    )

    if DEBUG_MAX_EVENTS:
        logger.info(
            "source_parsing: DEBUG limit active max_new_events=%d",
            DEBUG_MAX_EVENTS,
        )
    
    def _norm_source(raw: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "", (raw or "").strip().lower())

    only_set: set[str] | None = None
    if only_sources:
        only_set = {_norm_source(s) for s in only_sources if _norm_source(s)}
        if not only_set:
            only_set = None

    theatre_sources = {"dramteatr", "muzteatr", "sobor", "tretyakov"}
    need_theatres = True if only_set is None else bool(only_set & (theatre_sources | {"theatres", "theatre", "theater"}))
    need_phil = True if only_set is None else ("philharmonia" in only_set)
    need_qtickets = True if only_set is None else ("qtickets" in only_set)

    try:
        # 1. Run Kaggle kernels (Parallel). In targeted mode, skip unrelated kernels.
        logger.info(
            "source_parsing: starting kernels theatres=%s philharmonia=%s qtickets=%s only_sources=%s",
            int(bool(need_theatres)),
            int(bool(need_phil)),
            int(bool(need_qtickets)),
            sorted(only_set) if only_set else None,
        )

        tasks: dict[str, asyncio.Task] = {}
        if need_theatres:
            tasks["theatres"] = asyncio.create_task(
                run_kaggle_kernel(status_callback=_update_kaggle_status)
            )
        if need_phil:
            tasks["philharmonia"] = asyncio.create_task(run_philharmonia_kaggle_kernel())
        if need_qtickets:
            tasks["qtickets"] = asyncio.create_task(run_qtickets_kaggle_kernel())

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        by_name = dict(zip(tasks.keys(), results))

        # Process Theatres Result
        theatre_files: list[str] = []
        res_theatres = by_name.get("theatres")
        if isinstance(res_theatres, Exception):
            logger.error("source_parsing: theatres kernel error: %s", res_theatres)
            result.errors.append(f"Theatres kernel error: {res_theatres}")
        elif res_theatres is not None:
            status_t, files_t, dur_t = res_theatres
            result.kernel_duration = max(result.kernel_duration, dur_t)
            if status_t == "complete":
                theatre_files = [f for f in files_t if Path(f).suffix.lower() == ".json"]
                result.json_file_paths.extend(theatre_files)
            else:
                result.errors.append(f"Theatres kernel failed: {status_t}")

        # Process Philharmonia Result
        phil_files: list[str] = []
        res_phil = by_name.get("philharmonia")
        if isinstance(res_phil, Exception):
            logger.error("source_parsing: philharmonia kernel error: %s", res_phil)
            result.errors.append(f"Philharmonia kernel error: {res_phil}")
        elif res_phil is not None:
            status_p, files_p, dur_p = res_phil
            result.kernel_duration = max(result.kernel_duration, dur_p)
            if status_p == "complete":
                phil_files = [f for f in files_p if f.endswith("philharmonia_results.json")]
                if not phil_files:
                    phil_files = [f for f in files_p if Path(f).suffix.lower() == ".json"]
                result.json_file_paths.extend(phil_files)
            else:
                result.errors.append(f"Philharmonia kernel failed: {status_p}")

        # Process Qtickets Result
        qtickets_files: list[str] = []
        res_qtickets = by_name.get("qtickets")
        if isinstance(res_qtickets, Exception):
            logger.error("source_parsing: qtickets kernel error: %s", res_qtickets)
            result.errors.append(f"Qtickets kernel error: {res_qtickets}")
        elif res_qtickets is not None:
            status_q, files_q, dur_q = res_qtickets
            result.kernel_duration = max(result.kernel_duration, dur_q)
            if status_q == "complete":
                qtickets_files = [f for f in files_q if f.endswith("qtickets_events.json")]
                if not qtickets_files:
                    qtickets_files = [f for f in files_q if Path(f).suffix.lower() == ".json"]
                result.json_file_paths.extend(qtickets_files)
            else:
                result.errors.append(f"Qtickets kernel failed: {status_q}")

        await _process_parsing_files(
            db,
            bot,
            chat_id=chat_id,
            theatre_files=theatre_files,
            phil_files=phil_files,
            qtickets_files=qtickets_files,
            result=result,
            only_sources=only_set,
            date_from=date_from,
            date_to=date_to,
        )
        
        result.processing_duration = time.time() - start_time
        
        logger.info(
            "source_parsing: complete total=%d kernel=%.1fs processing=%.1fs",
            int(getattr(result, "total_events", 0) or 0),
            result.kernel_duration,
            result.processing_duration,
        )
        
        return result
    except Exception as exc:
        ops_status = "error"
        ops_error = str(exc)
        raise
    finally:
        if log_handler:
            logging.getLogger().removeHandler(log_handler)
            log_handler.close()
        source_details = {
            source: {
                "processed": int(stats.total_received),
                "new_events": int(stats.new_added),
                "updated_events": int(stats.ticket_updated + stats.already_exists),
                "failed": int(stats.failed),
                "skipped": int(stats.skipped),
            }
            for source, stats in (result.stats_by_source or {}).items()
        }
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=ops_status,
            metrics={
                "total_events": int(result.total_events or 0),
                "sources_processed": int(len(result.stats_by_source or {})),
                "events_created": int(len(result.added_events or [])),
                "events_updated": int(len(result.updated_events or [])),
                "errors_count": int(len(result.errors or [])),
                "kernel_duration": round(float(result.kernel_duration or 0.0), 3),
                "processing_duration": round(float(result.processing_duration or 0.0), 3),
            },
            details={
                "run_id": run_id,
                "log_file_path": result.log_file_path,
                "sources": source_details,
                "errors": list(result.errors or [])[:40],
                "fatal_error": ops_error,
            },
        )


_source_parsing_recovery_active: set[str] = set()


async def resume_source_parsing_jobs(
    db: Database,
    bot: Bot | None,
    *,
    chat_id: int | None = None,
) -> int:
    jobs = await list_jobs()
    parse_jobs = [
        j
        for j in jobs
        if j.get("type") in {"parse_theatres", "parse_philharmonia"}
    ]
    if not parse_jobs:
        return 0
    notify_chat_id = chat_id
    if notify_chat_id is None:
        notify_chat_id = await resolve_superadmin_chat_id(db)
    client = KaggleClient()
    recovered = 0
    for job in parse_jobs:
        kernel_ref = str(job.get("kernel_ref") or "")
        job_type = str(job.get("type") or "")
        if not kernel_ref or kernel_ref in _source_parsing_recovery_active:
            continue
        _source_parsing_recovery_active.add(kernel_ref)
        try:
            meta = job.get("meta") if isinstance(job.get("meta"), dict) else {}
            owner_pid = meta.get("pid")
            if owner_pid == os.getpid():
                continue
            try:
                status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            except Exception:
                logger.exception("source_parsing_recovery: status fetch failed kernel=%s", kernel_ref)
                continue
            state = str(status.get("status") or "").lower()
            if state in {"error", "failed", "cancelled"}:
                await remove_job(job_type, kernel_ref)
                if bot and notify_chat_id:
                    await bot.send_message(
                        notify_chat_id,
                        f"⚠️ parse recovery: kernel {kernel_ref} завершился ошибкой",
                    )
                continue
            if state != "complete":
                continue
            output_dir = Path(tempfile.gettempdir()) / f"source_parsing_recovery_{abs(hash(kernel_ref))}"
            output_dir.mkdir(parents=True, exist_ok=True)
            files = await asyncio.to_thread(
                client.download_kernel_output,
                kernel_ref,
                path=str(output_dir),
                force=True,
            )
            file_paths = [str(output_dir / f) for f in files]
            theatre_files: list[str] = []
            phil_files: list[str] = []
            qtickets_files: list[str] = []
            if job_type == "parse_theatres":
                theatre_files = [f for f in file_paths if Path(f).suffix.lower() == ".json"]
            elif job_type == "parse_philharmonia":
                phil_files = [f for f in file_paths if Path(f).suffix.lower() == ".json"]
            elif job_type == "parse_qtickets":
                qtickets_files = [f for f in file_paths if Path(f).suffix.lower() == ".json"]
            if not theatre_files and not phil_files and not qtickets_files:
                logger.warning(
                    "source_parsing_recovery: no json files kernel=%s", kernel_ref
                )
                continue
            result = SourceParsingResult(chat_id=notify_chat_id)
            result.json_file_paths.extend(theatre_files + phil_files + qtickets_files)
            await _process_parsing_files(
                db,
                bot,
                chat_id=notify_chat_id,
                theatre_files=theatre_files,
                phil_files=phil_files,
                qtickets_files=qtickets_files,
                result=result,
            )
            await remove_job(job_type, kernel_ref)
            recovered += 1
            if bot and notify_chat_id:
                bot_username = None
                try:
                    me = await bot.get_me()
                    bot_username = (getattr(me, "username", None) or "").strip().lstrip("@") or None
                except Exception:
                    bot_username = None
                report = await format_parsing_report(result, bot_username=bot_username, db=db)
                await bot.send_message(
                    notify_chat_id,
                    f"✅ parse recovery: kernel {kernel_ref} обработан\n\n{report}",
                    parse_mode="Markdown",
                )
        finally:
            _source_parsing_recovery_active.discard(kernel_ref)
    return recovered



async def process_source_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    source: str,
    start_index: int,
    total_count: int,
    chat_id: int | None = None,
    progress_message_id: int | None = None,
    added_events: list[AddedEventInfo] | None = None,
    updated_events: list[UpdatedEventInfo] | None = None,
) -> tuple[SourceParsingStats, int | None]:
    """Process events from a single source.
    
    Args:
        db: Database instance
        bot: Telegram bot
        events: List of events to process
        source: Source identifier
        start_index: Starting index for progress
        total_count: Total events across all sources
        chat_id: Chat ID for progress messages
        progress_message_id: Message ID to edit for progress updates
        added_events: List to populate with newly added events
        updated_events: List to populate with updated events
    
    Returns:
        Tuple of (statistics, updated progress_message_id)
    """
    stats = SourceParsingStats(source=source, total_received=len(events))
    
    # Source label for messages
    source_label = {
        "dramteatr": "🎭 Драмтеатр",
        "muzteatr": "🎵 Музтеатр",
        "sobor": "⛪ Собор",
        "tretyakov": "🎨 Третьяковка",
    }.get(source, source)
    
    for i, event in enumerate(events):
        current_progress = start_index + i + 1
        event_start = time.monotonic()
        result_tag = "unknown"
        event_id: int | None = None
        llm_used = False
        diag_enabled = bool(SOURCE_PARSING_DIAG_TITLE) and SOURCE_PARSING_DIAG_TITLE in (event.title or "").lower()

        # Update progress message for every event (new or existing).
        if bot and chat_id:
            try:
                progress_text = f"📝 Обработка {current_progress}/{total_count}: {event.title[:40]}"
                if progress_message_id:
                    await bot.edit_message_text(
                        text=progress_text,
                        chat_id=chat_id,
                        message_id=progress_message_id,
                    )
                else:
                    msg = await bot.send_message(chat_id, progress_text)
                    progress_message_id = msg.message_id
            except Exception as e:
                logger.warning("source_parsing: failed to update progress: %s", e)
        
        if not event.parsed_date:
            logger.warning(
                "source_parsing: skipping event without date title=%s",
                event.title,
            )
            stats.failed += 1
            result_tag = "missing_date"
            logger.info(
                "source_parsing: event_result source=%s title=%s result=%s duration=%.2fs",
                source,
                event.title[:80],
                result_tag,
                time.monotonic() - event_start,
            )
            continue

        # Filter out past events
        from datetime import datetime, timezone
        
        # We use current UTC date for filtering. 
        # Ideally we should use local date, but UTC is a safe conservative baseline 
        # so we don't filter out "today" events that might be technically "yesterday" in UTC late at night,
        # or vice-versa. 
        # Actually proper logic: 
        # If event_date < current_date, it's in the past. 
        # To be safe for timezone edge cases (late night parsing), 
        # let's only filter if it's strictly before TODAY.
        
        current_date = datetime.now(timezone.utc).date()
        
        # For string dates, we can't easily compare, but they should be parsed objects here 
        # thanks to our previous fixes (or at least Philharmonia ones are).
        # We can try to convert if generic object.
        event_date_obj = event.parsed_date
        if isinstance(event_date_obj, str):
             try:
                 event_date_obj = datetime.strptime(event_date_obj, "%Y-%m-%d").date()
             except:
                 pass

        if isinstance(event_date_obj, type(current_date)) and event_date_obj < current_date:
             stats.skipped += 1
             result_tag = "past_event"
             logger.info(
                "source_parsing: skipping past event title=%s date=%s",
                event.title[:50],
                event_date_obj,
             )
             continue

        if diag_enabled:
            desc_len = len(event.description or "")
            photos_count = len(event.photos or [])
            logger.info(
                "source_parsing: diag title=%s date_raw=%s parsed=%s %s url=%s photos=%d desc_len=%d",
                event.title[:120],
                event.date_raw,
                event.parsed_date,
                event.parsed_time or "",
                event.url,
                photos_count,
                desc_len,
            )
        
        try:
            location_name = normalize_location_name(event.location, event.scene)
            if diag_enabled:
                logger.info(
                    "source_parsing: diag normalized_location=%s",
                    location_name,
                )
            
            # Check for existing event
            existing_id, needs_full_update = await find_existing_event(
                db,
                location_name,
                event.parsed_date,
                event.parsed_time or "00:00",
                event.title,
            )
            
            parser_source_present = False
            if existing_id:
                parser_source_present = await event_has_parser_source(
                    db,
                    existing_id,
                    event.source_type,
                    event.url,
                )

            if existing_id and parser_source_present:
                event_id = existing_id
                if needs_full_update:
                    # Update the placeholder event fully
                    success = await update_event_full(db, existing_id, event)
                    if success:
                        await schedule_existing_event_update(db, existing_id)
                        stats.ticket_updated += 1
                        result_tag = "existing_full_update"
                        # Track updated event for reporting
                        if updated_events is not None:
                            info = await build_updated_event_info(
                                db,
                                existing_id,
                                source,
                                "full_update",
                                source_url=event.url,
                            )
                            if info:
                                updated_events.append(info)
                    else:
                        stats.failed += 1
                        result_tag = "existing_full_update_failed"
                else:
                    # Source already imported via parser -> cheap status/ticket sync only.
                    success = await update_event_ticket_status(
                        db,
                        existing_id,
                        event.ticket_status,
                        event.url,
                    )
                    if success:
                        await schedule_existing_event_update(db, existing_id)
                        stats.ticket_updated += 1
                        result_tag = "existing_ticket_update"
                        # Track updated event for reporting
                        if updated_events is not None:
                            info = await build_updated_event_info(
                                db,
                                existing_id,
                                source,
                                "ticket_status",
                                source_url=event.url,
                            )
                            if info:
                                updated_events.append(info)
                    else:
                        stats.already_exists += 1
                        result_tag = "existing_ticket_update_failed"

                # Always update linked events
                await update_linked_events(db, existing_id, location_name, event.title)
            else:
                mode_prefix = "existing_source_missing" if existing_id else "new"
                if existing_id and diag_enabled:
                    logger.info(
                        "source_parsing: diag parser_source_missing source=%s event_id=%s title=%s",
                        event.source_type,
                        existing_id,
                        event.title[:120],
                    )
                if diag_enabled and not existing_id:
                    logger.info(
                        "source_parsing: diag no existing match title=%s",
                        event.title[:120],
                    )
                # Prepare images for OCR if any
                poster_media_list = []
                
                # Filter photos first
                target_photos = limit_photos_for_source(
                    event.photos,
                    event.source_type,
                )
                if (not target_photos) and event.url and event.source_type == "dramteatr":
                    # Kaggle output for dramteatr can miss `photos`. Pull at least one
                    # gallery/cover image so Smart Update can log/apply posters.
                    cover = await _fetch_og_image_for_dramteatr(event.url)
                    if cover:
                        target_photos = [cover]

                if event.source_type in SOURCE_PARSING_DISABLE_OCR_SOURCES:
                    if diag_enabled:
                        logger.info(
                            "source_parsing: diag ocr skipped source=%s",
                            event.source_type,
                        )
                    target_photos = []
                
                if target_photos:
                    try:
                        if diag_enabled:
                            logger.info(
                                "source_parsing: diag ocr start title=%s photos=%d",
                                event.title[:120],
                                len(target_photos),
                            )
                        if event.source_type == "dramteatr" and len(target_photos) > 3:
                            # Dramteatr pages can have large galleries; we only need a few
                            # posters for Smart Update and operator logs.
                            target_photos = target_photos[:3]
                        raw_images = await asyncio.wait_for(
                            download_images(target_photos),
                            timeout=SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                        )
                        
                        if raw_images:
                            # Catbox can be flaky for consumers (Telegram preview, local runtime TLS issues).
                            # For dramteatr, keep original URLs but still run OCR and store hashes.
                            need_catbox = event.source_type != "dramteatr"
                            # OCR is optional for source-parsed galleries; we must not drop posters
                            # just because OCR timed out or a provider was unavailable.
                            need_ocr = bool(need_catbox) and (event.source_type not in SOURCE_PARSING_DISABLE_OCR_SOURCES)
                            try:
                                poster_media_list, _ = await asyncio.wait_for(
                                    process_media(
                                        raw_images,
                                        need_catbox=need_catbox,
                                        need_ocr=need_ocr,
                                    ),
                                    timeout=SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                                )
                            except asyncio.TimeoutError:
                                # Preserve poster hashes/URLs even if OCR is too slow.
                                poster_media_list, _ = await asyncio.wait_for(
                                    process_media(
                                        raw_images,
                                        need_catbox=need_catbox,
                                        need_ocr=False,
                                    ),
                                    timeout=SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                                )
                            if not need_catbox:
                                for poster, url in zip(poster_media_list, target_photos):
                                    if not poster.catbox_url:
                                        poster.catbox_url = url
                        if diag_enabled:
                            logger.info(
                                "source_parsing: diag ocr done title=%s raw=%d posters=%d",
                                event.title[:120],
                                len(raw_images),
                                len(poster_media_list),
                            )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "source_parsing: ocr timeout title=%s after %ss",
                            event.title,
                            SOURCE_PARSING_OCR_TIMEOUT_SECONDS,
                        )
                    except Exception as e:
                        logger.warning("source_parsing: ocr failed event=%s error=%s", event.title, e)

                # Add/merge event through Smart Update.
                llm_used = True
                new_id, was_added, status = unpack_add_event_result(
                    await add_new_event_via_queue(
                        db,
                        bot,
                        event,
                        current_progress,
                        total_count,
                        poster_media=poster_media_list,
                    )
                )

                if new_id:
                    event_id = new_id

                outcome = classify_add_event_outcome(new_id, was_added, status)
                if outcome == "added":
                    if new_id:
                        stats.new_added += 1
                        if added_events is not None:
                            info = await build_added_event_info(
                                db,
                                new_id,
                                source,
                                source_url=event.url,
                            )
                            if info:
                                added_events.append(info)
                    result_tag = f"{mode_prefix}_added"
                elif outcome == "updated":
                    if new_id:
                        stats.ticket_updated += 1
                        if updated_events is not None:
                            info = await build_updated_event_info(
                                db,
                                new_id,
                                source,
                                "smart_merge",
                                source_url=event.url,
                            )
                            if info:
                                updated_events.append(info)
                    result_tag = f"{mode_prefix}_updated"
                elif outcome == "skipped":
                    stats.skipped += 1
                    result_tag = f"{mode_prefix}_skipped"
                else:
                    stats.failed += 1
                    result_tag = f"{mode_prefix}_failed"

                if new_id and outcome in {"added", "updated"}:
                    # Delay between smart-update writes to reduce rebuild pressure.
                    await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)

                    # DEBUG: Stop after max events
                    if DEBUG_MAX_EVENTS and stats.new_added >= DEBUG_MAX_EVENTS:
                        logger.info("source_parsing: DEBUG limit reached (%d events)", DEBUG_MAX_EVENTS)
                        break
        except Exception:
            stats.failed += 1
            result_tag = "exception"
            logger.exception(
                "source_parsing: event_exception source=%s title=%s",
                source,
                event.title[:80],
            )

        logger.info(
            "source_parsing: event_result source=%s title=%s date=%s time=%s result=%s event_id=%s llm=%s duration=%.2fs",
            source,
            event.title[:80],
            event.parsed_date,
            event.parsed_time,
            result_tag,
            event_id,
            int(llm_used),
            time.monotonic() - event_start,
        )
    
    return stats, progress_message_id
async def run_diagnostic_parse(
    bot: Bot,
    chat_id: int,
    source: str,
) -> None:
    """Run diagnostic parse for a specific source and send result JSON.
    
    Args:
        bot: Bot instance
        chat_id: Chat ID to send results to
        source: Source identifier (dramteatr, muzteatr, sobor, tretyakov, or all)
    """
    from aiogram.types import FSInputFile
    
    await bot.send_message(
        chat_id,
        f"🔍 Запуск диагностического парсинга: {source}...\nПожалуйста, подождите (около 2-5 минут)."
    )
    
    start_time = time.time()
    kaggle_status_message_id: int | None = None
    kaggle_kernel_ref = ""

    async def _update_kaggle_status(
        phase: str,
        kernel_ref: str,
        status: dict | None,
    ) -> None:
        nonlocal kaggle_status_message_id, kaggle_kernel_ref
        kaggle_kernel_ref = kernel_ref or kaggle_kernel_ref
        
        text = _format_kaggle_status_message(phase, kernel_ref, status)
        try:
            if kaggle_status_message_id is None:
                sent = await bot.send_message(chat_id, text)
                kaggle_status_message_id = sent.message_id
            else:
                await bot.edit_message_text(
                    text=text,
                    chat_id=chat_id,
                    message_id=kaggle_status_message_id,
                )
        except Exception:
            pass

    # Run kernel with config
    if source == "philharmonia":
        from source_parsing.philharmonia import run_philharmonia_kaggle_kernel
        status, output_files, duration = await run_philharmonia_kaggle_kernel(
            status_callback=_update_kaggle_status,
        )
    else:
        status, output_files, duration = await run_kaggle_kernel(
            status_callback=_update_kaggle_status,
            run_config={"target_source": source}
        )
    
    if status != "complete":
        await bot.send_message(chat_id, f"❌ Ошибка запуска: статус {status}")
        return

    # Find the specific JSON file
    target_filename = f"{source}.json"
    if source == "philharmonia":
        target_filename = "philharmonia_results.json"
        
    target_path = None
    
    # If source is 'tretyakov', look for 'tretyakov.json' etc.
    # Note: notebook saves lowercase filenames
    
    found_files = []
    
    for fpath in output_files:
        path = Path(fpath)
        if path.name == target_filename:
            target_path = fpath
        if path.suffix == ".json":
            found_files.append(path)
            
    if target_path and Path(target_path).exists():
        await bot.send_document(
            chat_id,
            FSInputFile(target_path),
            caption=f"✅ Диагностика {source} завершена за {duration:.1f}с.\nРезультат в файле."
        )
    elif found_files:
        # If exact match not found but other JSONs exist (maybe naming mismatch), send them
        await bot.send_message(chat_id, f"⚠️ Файл {target_filename} не найден, отправляю найденные JSON:")
        for fpath in found_files:
             await bot.send_document(
                chat_id,
                FSInputFile(fpath),
                caption=f"📄 {fpath.name}"
            )
    else:
        await bot.send_message(
            chat_id, 
            f"⚠️ Файл {target_filename} не найден в результатах.\nСтатус: {status}\nФайлы: {len(output_files)}"
        )
