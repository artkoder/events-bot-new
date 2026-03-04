from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal, Sequence

from sqlalchemy import and_, func, select

from db import Database
from models import TicketSiteQueueItem
from ops_run import finish_ops_run, start_ops_run

logger = logging.getLogger(__name__)

TicketSiteKind = Literal["pyramida", "dom_iskusstv", "qtickets"]

_QTICKETS_URL_RE = re.compile(
    r"https?://[^\s<>()\"']*qtickets\.(?:events|ru)/[^\s<>()\"']+",
    re.IGNORECASE,
)


def is_ticket_sites_queue_enabled() -> bool:
    raw = (os.getenv("ENABLE_TICKET_SITES_QUEUE") or "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def ticket_sites_queue_schedule_text() -> str:
    time_local = (os.getenv("TICKET_SITES_QUEUE_TIME_LOCAL") or "11:20").strip() or "11:20"
    tz = (os.getenv("TICKET_SITES_QUEUE_TZ") or "Europe/Kaliningrad").strip() or "Europe/Kaliningrad"
    return f"{time_local} {tz}"


def _clean_url(url: str | None) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    return raw.rstrip(").,;:!?\u00bb\u201d\u2019\"'")


def detect_ticket_site_kind(url: str) -> TicketSiteKind | None:
    u = _clean_url(url)
    if not u:
        return None
    try:
        from source_parsing.pyramida import PYRAMIDA_URL_PATTERN
    except Exception:
        PYRAMIDA_URL_PATTERN = None
    try:
        from source_parsing.dom_iskusstv import DOM_ISKUSSTV_URL_PATTERN
    except Exception:
        DOM_ISKUSSTV_URL_PATTERN = None

    if PYRAMIDA_URL_PATTERN and PYRAMIDA_URL_PATTERN.search(u):
        return "pyramida"
    if DOM_ISKUSSTV_URL_PATTERN and DOM_ISKUSSTV_URL_PATTERN.search(u):
        return "dom_iskusstv"
    if _QTICKETS_URL_RE.search(u):
        return "qtickets"
    return None


def extract_ticket_site_urls(
    *,
    text: str | None,
    links_payload: Any | None = None,
    events_payload: Any | None = None,
) -> list[str]:
    parts: list[str] = []
    if text:
        parts.append(str(text))

    urls: list[str] = []
    for chunk in parts:
        try:
            from source_parsing.pyramida import extract_pyramida_urls
        except Exception:
            extract_pyramida_urls = None
        try:
            from source_parsing.dom_iskusstv import extract_dom_iskusstv_urls
        except Exception:
            extract_dom_iskusstv_urls = None

        if extract_pyramida_urls:
            urls.extend(extract_pyramida_urls(chunk))
        if extract_dom_iskusstv_urls:
            urls.extend(extract_dom_iskusstv_urls(chunk))
        urls.extend(_QTICKETS_URL_RE.findall(chunk))

    def _add_any(raw: Any) -> None:
        if isinstance(raw, str) and raw.strip():
            urls.append(raw.strip())
        elif isinstance(raw, dict):
            for key in ("url", "href", "link"):
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    urls.append(val.strip())
        elif isinstance(raw, list):
            for item in raw:
                _add_any(item)

    if links_payload is not None:
        _add_any(links_payload)
    if isinstance(events_payload, list):
        for ev in events_payload:
            if isinstance(ev, dict):
                _add_any(ev.get("ticket_link"))
                _add_any(ev.get("links"))

    out: list[str] = []
    seen: set[str] = set()
    for raw in urls:
        u = _clean_url(raw)
        if not u:
            continue
        if detect_ticket_site_kind(u) is None:
            continue
        key = u.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


async def enqueue_ticket_site_urls(
    db: Database,
    *,
    urls: Sequence[str],
    event_id: int | None = None,
    source_post_url: str | None = None,
    source_chat_username: str | None = None,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
) -> int:
    async with db.get_session() as session:
        enqueued = await enqueue_ticket_site_urls_in_session(
            session,
            urls=urls,
            event_id=event_id,
            source_post_url=source_post_url,
            source_chat_username=source_chat_username,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
        )
        await session.commit()
        return int(enqueued or 0)


async def enqueue_ticket_site_urls_in_session(
    session: Any,
    *,
    urls: Sequence[str],
    event_id: int | None = None,
    source_post_url: str | None = None,
    source_chat_username: str | None = None,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
    now: datetime | None = None,
) -> int:
    """Enqueue ticket-site URLs using an existing DB session.

    This exists so higher-level pipelines (e.g. Smart Update) can enqueue URLs
    without opening a nested write transaction (SQLite would otherwise hit
    `database is locked` under a long-running merge/create transaction).
    """

    now = now or datetime.now(timezone.utc)
    clean_urls = [_clean_url(u) for u in urls]
    clean_urls = [u for u in clean_urls if u]
    if not clean_urls:
        return 0

    enqueued = 0
    for raw_url in clean_urls:
        kind = detect_ticket_site_kind(raw_url)
        if kind is None:
            continue
        url = _clean_url(raw_url)
        existing = (
            await session.execute(
                select(TicketSiteQueueItem).where(TicketSiteQueueItem.url == url)
            )
        ).scalar_one_or_none()
        if existing:
            if existing.status == "disabled":
                continue
            if event_id and not existing.event_id:
                existing.event_id = int(event_id)
            existing.source_post_url = source_post_url or existing.source_post_url
            existing.source_chat_username = source_chat_username or existing.source_chat_username
            if source_chat_id is not None and existing.source_chat_id is None:
                existing.source_chat_id = int(source_chat_id)
            if source_message_id is not None and existing.source_message_id is None:
                existing.source_message_id = int(source_message_id)
            existing.site_kind = str(kind)
            existing.updated_at = now
            existing.next_run_at = now
            session.add(existing)
            enqueued += 1
            continue

        item = TicketSiteQueueItem(
            status="active",
            site_kind=str(kind),
            url=url,
            event_id=int(event_id) if event_id is not None else None,
            source_post_url=source_post_url,
            source_chat_username=source_chat_username,
            source_chat_id=int(source_chat_id) if source_chat_id is not None else None,
            source_message_id=int(source_message_id) if source_message_id is not None else None,
            attempts=0,
            last_error=None,
            last_result_json={},
            last_run_at=None,
            created_at=now,
            updated_at=now,
            next_run_at=now,
        )
        session.add(item)
        enqueued += 1

    return int(enqueued or 0)


def _interval_delta() -> timedelta:
    raw = (os.getenv("TICKET_SITES_QUEUE_INTERVAL_HOURS") or "24").strip() or "24"
    try:
        hours = float(raw)
    except Exception:
        hours = 24.0
    if hours <= 0:
        hours = 24.0
    return timedelta(hours=hours)


def _error_backoff(attempts: int) -> timedelta:
    base_minutes = 30
    minutes = min(12 * 60, base_minutes * max(1, attempts))
    return timedelta(minutes=minutes)


def _unique_preserve(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        val = _clean_url(raw)
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


async def ticket_sites_queue_info_text(
    db: Database,
    *,
    site_kind: str | None = None,
    limit: int | None = None,
) -> str:
    clean_kind = (site_kind or "").strip().lower() or None
    if clean_kind and clean_kind not in {"pyramida", "dom_iskusstv", "qtickets"}:
        clean_kind = None

    sample_limit = 10
    if limit is not None and int(limit) > 0:
        sample_limit = min(int(limit), 25)

    async with db.get_session() as session:
        base_where = []
        if clean_kind:
            base_where.append(TicketSiteQueueItem.site_kind == clean_kind)

        status_stmt = select(TicketSiteQueueItem.status, func.count()).group_by(TicketSiteQueueItem.status)
        if base_where:
            status_stmt = status_stmt.where(and_(*base_where))
        status_rows = (await session.execute(status_stmt)).all()
        status_counts = {str(st or ""): int(cnt or 0) for st, cnt in status_rows}

        pending_where = [TicketSiteQueueItem.status.in_(["active", "error"])]
        if clean_kind:
            pending_where.append(TicketSiteQueueItem.site_kind == clean_kind)
        pending_stmt = (
            select(TicketSiteQueueItem)
            .where(and_(*pending_where))
            .order_by(TicketSiteQueueItem.next_run_at, TicketSiteQueueItem.created_at)
            .limit(sample_limit)
        )
        pending_items = (await session.execute(pending_stmt)).scalars().all()

        running_where = [TicketSiteQueueItem.status == "running"]
        if clean_kind:
            running_where.append(TicketSiteQueueItem.site_kind == clean_kind)
        running_stmt = (
            select(TicketSiteQueueItem)
            .where(and_(*running_where))
            .order_by(TicketSiteQueueItem.updated_at.desc(), TicketSiteQueueItem.created_at.desc())
            .limit(5)
        )
        running_items = (await session.execute(running_stmt)).scalars().all()

    total = sum(status_counts.values())
    active = status_counts.get("active", 0)
    running = status_counts.get("running", 0)
    error = status_counts.get("error", 0)
    disabled = status_counts.get("disabled", 0)

    lines: list[str] = ["🎫 Очередь ticket-sites (info)"]
    enabled = "вкл" if is_ticket_sites_queue_enabled() else "выкл"
    lines.append(f"Автозапуск очереди: {enabled} (расписание: {ticket_sites_queue_schedule_text()})")
    if clean_kind:
        lines.append(f"Фильтр: {clean_kind}")
    lines.append(
        f"Счётчики: total={total} active={active} running={running} error={error} disabled={disabled}"
    )

    if pending_items:
        lines.append("Следующие (active/error):")
        for item in pending_items:
            url = str(item.url or "").strip()
            kind = str(item.site_kind or "").strip().lower()
            attempts = int(item.attempts or 0)
            next_run = getattr(item, "next_run_at", None)
            next_txt = ""
            if next_run:
                try:
                    next_txt = next_run.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
                except Exception:
                    next_txt = str(next_run)
            extra = []
            if next_txt:
                extra.append(f"next={next_txt}")
            extra.append(f"attempts={attempts}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- [{kind}] {url}{suffix}")
    else:
        lines.append("Активных элементов не найдено.")

    if running_items:
        lines.append("Running (последние):")
        for item in running_items:
            url = str(item.url or "").strip()
            kind = str(item.site_kind or "").strip().lower()
            attempts = int(item.attempts or 0)
            lines.append(f"- [{kind}] {url} (attempts={attempts})")

    lines.append("Подсказка: /ticket_sites_queue для обработки очереди.")
    return "\n".join(lines)


@dataclass(slots=True)
class TicketSitesQueueRunReport:
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    details: list[str] = field(default_factory=list)


_TICKET_SITES_QUEUE_LOCK = asyncio.Lock()


def _poster_candidates_from_urls(urls: Sequence[str], *, limit: int = 8) -> list[Any]:
    try:
        from smart_event_update import PosterCandidate
    except Exception:
        return []
    out: list[Any] = []
    for url in urls[: max(0, int(limit))]:
        u = _clean_url(url)
        if not u:
            continue
        out.append(PosterCandidate(catbox_url=u))
    return out


async def _smart_update_from_theatre_event(
    db: Database,
    *,
    site_kind: TicketSiteKind,
    url: str,
    title: str,
    date_iso: str,
    time_str: str,
    location: str,
    description: str | None,
    ticket_price_min: int | None,
    ticket_price_max: int | None,
    ticket_status: str | None,
    photos: Sequence[str] | None,
) -> tuple[str, int | None]:
    from smart_event_update import EventCandidate, smart_event_update

    candidate = EventCandidate(
        source_type=f"parser:{site_kind}",
        source_url=url,
        source_text=(description or "").strip() or title,
        title=title,
        date=date_iso,
        time=time_str or "00:00",
        end_date=None,
        location_name=location,
        location_address=None,
        city="Калининград",
        trust_level="high",
        ticket_price_min=ticket_price_min,
        ticket_price_max=ticket_price_max,
        ticket_link=url,
        ticket_status=(str(ticket_status or "").strip() or "available"),
        event_type=None,
        emoji=None,
        is_free=None,
        pushkin_card=False,
        raw_excerpt=(description or "").strip()[:600] if description else None,
        posters=_poster_candidates_from_urls(list(photos or [])),
    )
    result = await smart_event_update(
        db,
        candidate,
        check_source_url=False,
        schedule_kwargs={"skip_vk_sync": True},
    )
    return str(result.status or ""), int(result.event_id) if result.event_id else None


async def process_ticket_sites_queue(
    db: Database,
    *,
    bot: Any | None = None,
    chat_id: int | None = None,
    limit: int | None = None,
    site_kind: str | None = None,
    only_url: str | None = None,
    trigger: str = "manual",
    operator_id: int | None = None,
    run_id: str | None = None,
) -> TicketSitesQueueRunReport:
    _ = run_id
    clean_kind = (site_kind or "").strip().lower() or None
    if clean_kind and clean_kind not in {"pyramida", "dom_iskusstv", "qtickets"}:
        clean_kind = None
    clean_only_url = _clean_url(only_url)
    if clean_only_url and not clean_kind:
        kind = detect_ticket_site_kind(clean_only_url)
        clean_kind = str(kind) if kind else None

    item_limit = 25
    if limit is not None:
        try:
            parsed = int(limit)
            if parsed > 0:
                item_limit = min(parsed, 100)
        except Exception:
            item_limit = 25

    report = TicketSitesQueueRunReport()
    now = datetime.now(timezone.utc)

    async with _TICKET_SITES_QUEUE_LOCK:
        ops_id = await start_ops_run(
            db,
            kind="ticket_sites_queue",
            trigger=trigger,
            chat_id=chat_id,
            operator_id=operator_id,
            metrics={"limit": item_limit, "site_kind": clean_kind or ""},
        )

        try:
            async with db.get_session() as session:
                if clean_only_url:
                    where = [
                        TicketSiteQueueItem.url == clean_only_url,
                        TicketSiteQueueItem.status != "disabled",
                    ]
                else:
                    where = [
                        TicketSiteQueueItem.status.in_(["active", "error"]),
                        TicketSiteQueueItem.next_run_at <= now,
                    ]
                if clean_kind:
                    where.append(TicketSiteQueueItem.site_kind == clean_kind)
                stmt = (
                    select(TicketSiteQueueItem)
                    .where(and_(*where))
                    .order_by(TicketSiteQueueItem.next_run_at, TicketSiteQueueItem.created_at)
                    .limit(item_limit)
                )
                items = (await session.execute(stmt)).scalars().all()

                if not items:
                    report.skipped += 1
                    return report

                for item in items:
                    item.status = "running"
                    item.updated_at = now
                    item.attempts = int(item.attempts or 0) + 1
                    session.add(item)
                await session.commit()

            by_kind: dict[str, list[TicketSiteQueueItem]] = {"pyramida": [], "dom_iskusstv": [], "qtickets": []}
            for item in items:
                kind = str(item.site_kind or "").strip().lower()
                if kind in by_kind:
                    by_kind[kind].append(item)

            interval = _interval_delta()

            async def _mark_done(item: TicketSiteQueueItem, *, status: str, result: dict, err: str | None) -> None:
                tnow = datetime.now(timezone.utc)
                async with db.get_session() as session:
                    row = await session.get(TicketSiteQueueItem, int(item.id))
                    if not row:
                        return
                    row.status = status
                    row.last_error = err
                    row.last_result_json = dict(result or {})
                    try:
                        res_event_id = row.last_result_json.get("event_id")
                        if res_event_id is not None and not row.event_id:
                            row.event_id = int(res_event_id)
                    except Exception:
                        pass
                    row.last_run_at = tnow
                    row.updated_at = tnow
                    if status == "active":
                        row.next_run_at = tnow + interval
                    else:
                        row.next_run_at = tnow + _error_backoff(int(row.attempts or 0))
                    session.add(row)
                    await session.commit()

            # Pyramida: URL-scoped kernel
            if by_kind["pyramida"]:
                urls = _unique_preserve([i.url for i in by_kind["pyramida"] if i.url])
                try:
                    from source_parsing.pyramida import parse_pyramida_output, run_pyramida_kaggle_kernel

                    status, file_paths, _dur = await run_pyramida_kaggle_kernel(urls)
                    if status != "complete" or not file_paths:
                        raise RuntimeError(f"pyramida kernel failed status={status}")
                    events = parse_pyramida_output(file_paths)
                    events_by_url = {str(e.url or "").strip(): e for e in events if str(e.url or "").strip()}
                    for item in by_kind["pyramida"]:
                        u = str(item.url or "").strip()
                        ev = events_by_url.get(u)
                        if not ev:
                            report.failed += 1
                            await _mark_done(
                                item,
                                status="error",
                                result={"site_kind": "pyramida", "url": u, "reason": "not_found_in_output"},
                                err="not_found_in_output",
                            )
                            continue
                        su_status, ev_id = await _smart_update_from_theatre_event(
                            db,
                            site_kind="pyramida",
                            url=u,
                            title=str(ev.title or "").strip() or "—",
                            date_iso=str(ev.parsed_date or "").strip(),
                            time_str=str(ev.parsed_time or "").strip() or "00:00",
                            location=str(ev.location or "").strip() or "Калининград",
                            description=str(ev.description or "").strip() or None,
                            ticket_price_min=getattr(ev, "ticket_price_min", None),
                            ticket_price_max=getattr(ev, "ticket_price_max", None),
                            ticket_status=getattr(ev, "ticket_status", None),
                            photos=list(getattr(ev, "photos", None) or []),
                        )
                        report.processed += 1
                        report.success += 1 if su_status in {"created", "merged"} else 0
                        report.failed += 1 if su_status not in {"created", "merged"} else 0
                        await _mark_done(
                            item,
                            status="active" if su_status in {"created", "merged"} else "error",
                            result={
                                "site_kind": "pyramida",
                                "url": u,
                                "smart_update_status": su_status,
                                "event_id": ev_id,
                            },
                            err=None if su_status in {"created", "merged"} else f"smart_update:{su_status}",
                        )
                except Exception as exc:
                    logger.warning("ticket_sites_queue: pyramida failed", exc_info=True)
                    for item in by_kind["pyramida"]:
                        report.failed += 1
                        await _mark_done(
                            item,
                            status="error",
                            result={"site_kind": "pyramida", "url": str(item.url or "").strip()},
                            err=str(exc),
                        )

            # Dom Iskusstv: URL-scoped kernel
            if by_kind["dom_iskusstv"]:
                urls = _unique_preserve([i.url for i in by_kind["dom_iskusstv"] if i.url])
                try:
                    from source_parsing.dom_iskusstv import parse_dom_iskusstv_output, run_dom_iskusstv_kaggle_kernel

                    status, file_paths, _dur = await run_dom_iskusstv_kaggle_kernel(urls)
                    if status != "complete" or not file_paths:
                        raise RuntimeError(f"dom_iskusstv kernel failed status={status}")
                    events = parse_dom_iskusstv_output(file_paths)
                    events_by_url = {str(e.url or "").strip(): e for e in events if str(e.url or "").strip()}
                    for item in by_kind["dom_iskusstv"]:
                        u = str(item.url or "").strip()
                        ev = events_by_url.get(u)
                        if not ev:
                            report.failed += 1
                            await _mark_done(
                                item,
                                status="error",
                                result={"site_kind": "dom_iskusstv", "url": u, "reason": "not_found_in_output"},
                                err="not_found_in_output",
                            )
                            continue
                        su_status, ev_id = await _smart_update_from_theatre_event(
                            db,
                            site_kind="dom_iskusstv",
                            url=u,
                            title=str(ev.title or "").strip() or "—",
                            date_iso=str(ev.parsed_date or "").strip(),
                            time_str=str(ev.parsed_time or "").strip() or "00:00",
                            location=str(ev.location or "").strip() or "Дом искусств",
                            description=str(ev.description or "").strip() or None,
                            ticket_price_min=getattr(ev, "ticket_price_min", None),
                            ticket_price_max=getattr(ev, "ticket_price_max", None),
                            ticket_status=getattr(ev, "ticket_status", None),
                            photos=list(getattr(ev, "photos", None) or []),
                        )
                        report.processed += 1
                        report.success += 1 if su_status in {"created", "merged"} else 0
                        report.failed += 1 if su_status not in {"created", "merged"} else 0
                        await _mark_done(
                            item,
                            status="active" if su_status in {"created", "merged"} else "error",
                            result={
                                "site_kind": "dom_iskusstv",
                                "url": u,
                                "smart_update_status": su_status,
                                "event_id": ev_id,
                            },
                            err=None if su_status in {"created", "merged"} else f"smart_update:{su_status}",
                        )
                except Exception as exc:
                    logger.warning("ticket_sites_queue: dom_iskusstv failed", exc_info=True)
                    for item in by_kind["dom_iskusstv"]:
                        report.failed += 1
                        await _mark_done(
                            item,
                            status="error",
                            result={"site_kind": "dom_iskusstv", "url": str(item.url or "").strip()},
                            err=str(exc),
                        )

            # Qtickets: full kernel, then filter by queued URLs.
            if by_kind["qtickets"]:
                urls = _unique_preserve([i.url for i in by_kind["qtickets"] if i.url])
                try:
                    from pathlib import Path

                    from source_parsing.qtickets import parse_qtickets_output, run_qtickets_kaggle_kernel

                    status, file_paths, _dur = await run_qtickets_kaggle_kernel()
                    if status != "complete" or not file_paths:
                        raise RuntimeError(f"qtickets kernel failed status={status}")
                    json_files = [f for f in file_paths if Path(f).suffix.lower() == ".json"]
                    events = parse_qtickets_output(json_files)
                    events_by_url = {str(e.url or "").strip().casefold(): e for e in events if str(e.url or "").strip()}

                    for item in by_kind["qtickets"]:
                        u = str(item.url or "").strip()
                        ev = events_by_url.get(u.casefold())
                        if not ev:
                            report.failed += 1
                            await _mark_done(
                                item,
                                status="error",
                                result={"site_kind": "qtickets", "url": u, "reason": "not_found_in_output"},
                                err="not_found_in_output",
                            )
                            continue
                        su_status, ev_id = await _smart_update_from_theatre_event(
                            db,
                            site_kind="qtickets",
                            url=u,
                            title=str(ev.title or "").strip() or "—",
                            date_iso=str(ev.parsed_date or "").strip(),
                            time_str=str(ev.parsed_time or "").strip() or "00:00",
                            location=str(ev.location or "").strip() or "Калининград",
                            description=str(ev.description or "").strip() or None,
                            ticket_price_min=getattr(ev, "ticket_price_min", None),
                            ticket_price_max=getattr(ev, "ticket_price_max", None),
                            ticket_status=getattr(ev, "ticket_status", None),
                            photos=list(getattr(ev, "photos", None) or []),
                        )
                        report.processed += 1
                        report.success += 1 if su_status in {"created", "merged"} else 0
                        report.failed += 1 if su_status not in {"created", "merged"} else 0
                        await _mark_done(
                            item,
                            status="active" if su_status in {"created", "merged"} else "error",
                            result={
                                "site_kind": "qtickets",
                                "url": u,
                                "smart_update_status": su_status,
                                "event_id": ev_id,
                            },
                            err=None if su_status in {"created", "merged"} else f"smart_update:{su_status}",
                        )
                except Exception as exc:
                    logger.warning("ticket_sites_queue: qtickets failed", exc_info=True)
                    for item in by_kind["qtickets"]:
                        report.failed += 1
                        await _mark_done(
                            item,
                            status="error",
                            result={"site_kind": "qtickets", "url": str(item.url or "").strip()},
                            err=str(exc),
                        )

            return report
        finally:
            await finish_ops_run(
                db,
                run_id=ops_id,
                status="success" if report.failed == 0 else "error",
                metrics={
                    "processed": report.processed,
                    "success": report.success,
                    "failed": report.failed,
                    "skipped": report.skipped,
                },
            )
