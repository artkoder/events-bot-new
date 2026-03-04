from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlparse

from sqlalchemy import func, select

from db import Database
from models import EventMediaAsset, EventSource, FestivalQueueItem, TicketSiteQueueItem


@dataclass(frozen=True, slots=True)
class TicketQueueRow:
    id: int
    site_kind: str
    url: str
    status: str


@dataclass(frozen=True, slots=True)
class FestivalQueueRow:
    id: int
    status: str
    festival_context: str | None
    festival_name: str | None
    festival_full: str | None


@dataclass(frozen=True, slots=True)
class SmartUpdateReportContext:
    tz: timezone
    sources_by_event_id: dict[int, list[tuple[datetime | None, str]]]
    video_count_by_event_id: dict[int, int]
    ticket_queue_by_event_id: dict[int, list[TicketQueueRow]]
    festival_queue_by_source_url: dict[str, FestivalQueueRow]


_SCHEME_RE = re.compile(r"^https?://", re.I)


def short_url_label(url: str | None) -> str:
    """Minimize a URL for display while keeping it recognizable."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    s = _SCHEME_RE.sub("", raw).rstrip("/")
    # Drop common tracking tail noise in display label only.
    s = re.sub(r"[?#].*$", "", s)
    return s or raw


def format_dt_compact(dt: datetime | None, tz: timezone) -> str:
    if not dt:
        return "??.?? ??:??"
    value = dt
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    try:
        local = value.astimezone(tz)
    except Exception:
        local = value
    return local.strftime("%d.%m %H:%M")


def offset_to_timezone(value: str) -> timezone:
    raw = (value or "+00:00").strip()
    if len(raw) != 6 or raw[0] not in "+-" or raw[3] != ":":
        return timezone.utc
    try:
        sign = 1 if raw[0] == "+" else -1
        hours = int(raw[1:3])
        minutes = int(raw[4:6])
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    except Exception:
        return timezone.utc


async def load_local_tz(db: Database) -> timezone:
    try:
        async with db.raw_conn() as conn:
            cur = await conn.execute("SELECT value FROM setting WHERE key='tz_offset'")
            row = await cur.fetchone()
        offset = str(row[0]).strip() if row and row[0] else "+00:00"
    except Exception:
        offset = "+00:00"
    return offset_to_timezone(offset)


async def build_smart_update_report_context(
    db: Database,
    *,
    event_ids: Iterable[int],
    source_urls: Iterable[str] = (),
) -> SmartUpdateReportContext:
    eids = sorted({int(v) for v in (event_ids or []) if v})
    urls = [str(u).strip() for u in (source_urls or []) if str(u or "").strip()]
    # keep deterministic uniqueness
    urls = list(dict.fromkeys(urls))
    tz = await load_local_tz(db)

    sources_by_event_id: dict[int, list[tuple[datetime | None, str]]] = {eid: [] for eid in eids}
    video_count_by_event_id: dict[int, int] = {}
    ticket_queue_by_event_id: dict[int, list[TicketQueueRow]] = {}
    festival_queue_by_source_url: dict[str, FestivalQueueRow] = {}

    if not eids and not urls:
        return SmartUpdateReportContext(
            tz=tz,
            sources_by_event_id=sources_by_event_id,
            video_count_by_event_id=video_count_by_event_id,
            ticket_queue_by_event_id=ticket_queue_by_event_id,
            festival_queue_by_source_url=festival_queue_by_source_url,
        )

    async with db.get_session() as session:
        if eids:
            src_rows = (
                await session.execute(
                    select(EventSource.event_id, EventSource.imported_at, EventSource.source_url)
                    .where(EventSource.event_id.in_(eids))
                    .order_by(EventSource.imported_at, EventSource.id)
                )
            ).all()
            for event_id, imported_at, source_url in src_rows:
                try:
                    eid = int(event_id)
                except Exception:
                    continue
                su = str(source_url or "").strip()
                if not su:
                    continue
                sources_by_event_id.setdefault(eid, []).append((imported_at, su))

            v_rows = (
                await session.execute(
                    select(EventMediaAsset.event_id, func.count())
                    .where(
                        EventMediaAsset.event_id.in_(eids),
                        EventMediaAsset.kind == "video",
                    )
                    .group_by(EventMediaAsset.event_id)
                )
            ).all()
            for event_id, cnt in v_rows:
                try:
                    video_count_by_event_id[int(event_id)] = int(cnt or 0)
                except Exception:
                    continue

            t_rows = (
                await session.execute(
                    select(
                        TicketSiteQueueItem.event_id,
                        TicketSiteQueueItem.id,
                        TicketSiteQueueItem.site_kind,
                        TicketSiteQueueItem.url,
                        TicketSiteQueueItem.status,
                    )
                    .where(TicketSiteQueueItem.event_id.in_(eids))
                    .order_by(TicketSiteQueueItem.id)
                )
            ).all()
            for event_id, item_id, site_kind, url, status in t_rows:
                try:
                    eid = int(event_id) if event_id is not None else None
                except Exception:
                    eid = None
                if not eid:
                    continue
                row = TicketQueueRow(
                    id=int(item_id or 0),
                    site_kind=str(site_kind or "").strip(),
                    url=str(url or "").strip(),
                    status=str(status or "").strip(),
                )
                ticket_queue_by_event_id.setdefault(eid, []).append(row)

        if urls:
            # Use the latest row per source_url (by id).
            f_rows = (
                await session.execute(
                    select(
                        FestivalQueueItem.source_url,
                        FestivalQueueItem.id,
                        FestivalQueueItem.status,
                        FestivalQueueItem.festival_context,
                        FestivalQueueItem.festival_name,
                        FestivalQueueItem.festival_full,
                    )
                    .where(FestivalQueueItem.source_url.in_(urls))
                    .order_by(FestivalQueueItem.id.desc())
                )
            ).all()
            for source_url, item_id, status, context, name, full in f_rows:
                key = str(source_url or "").strip()
                if not key:
                    continue
                if key in festival_queue_by_source_url:
                    continue
                festival_queue_by_source_url[key] = FestivalQueueRow(
                    id=int(item_id or 0),
                    status=str(status or "").strip(),
                    festival_context=(str(context).strip() if context is not None else None),
                    festival_name=(str(name).strip() if name is not None else None),
                    festival_full=(str(full).strip() if full is not None else None),
                )

    return SmartUpdateReportContext(
        tz=tz,
        sources_by_event_id=sources_by_event_id,
        video_count_by_event_id=video_count_by_event_id,
        ticket_queue_by_event_id=ticket_queue_by_event_id,
        festival_queue_by_source_url=festival_queue_by_source_url,
    )


def host_from_url(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw if re.match(r"^https?://", raw, re.I) else "https://" + raw)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""

