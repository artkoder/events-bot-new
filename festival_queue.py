from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

from sqlalchemy import and_, func, or_, select, update

from db import Database
from models import Festival, FestivalQueueItem, Setting, TelegramSource, TelegramSourceForceMessage
from ops_run import finish_ops_run, start_ops_run

logger = logging.getLogger(__name__)

FestivalContext = Literal["festival_post", "event_with_festival", "none"]

_FESTIVAL_CONTEXT_VALUES: set[str] = {"festival_post", "event_with_festival", "none"}
_PROGRAM_RE = re.compile(
    r"\b(программ\w*|расписан\w*|план мероприятий|план событ\w*|афиш\w*)\b",
    re.IGNORECASE,
)
_DAY_SERIES_RE = re.compile(r"\bдень\s+[а-яёa-z0-9][^,\n.;:]{1,80}", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-](?:19|20)\d{2})?\b")
_TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")
_DATE_RANGE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}\s*(?:-|–|—|to)\s*\d{1,2}[./-]\d{1,2}\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s<>()\"']+")
_VK_WALL_RE = re.compile(r"wall-([0-9]+)_([0-9]+)", re.IGNORECASE)
_QUOTED_TEXT_RE = re.compile(r"[«\"]\s*([^\"»\n]{3,140})\s*[»\"]", re.IGNORECASE)
_DAY_BIRTHDAY_RE = re.compile(
    r"\bдень\s+рождения\s+[а-яёa-z0-9][^,\n.;:]{1,90}",
    re.IGNORECASE,
)


def _is_vk_domain(url: str) -> bool:
    u = (url or "").strip().lower()
    return "vk.com/" in u or "vk.ru/" in u


def _is_tg_domain(url: str) -> bool:
    u = (url or "").strip().lower()
    return "t.me/" in u


def _looks_like_vk_wall(url: str) -> bool:
    return bool(_VK_WALL_RE.search(url or ""))


def _is_recoverable_tg_monitor_error(exc: Exception) -> bool:
    msg = str(exc or "").strip().lower()
    if not msg:
        return False
    markers = (
        "kaggle kernel failed",
        "tg monitor timeout",
        "telegram monitor timeout",
        "network error",
        "ssl",
        "timeout",
    )
    return any(marker in msg for marker in markers)


async def _infer_vk_source_url(db: Database, source_post_url: str | None) -> str | None:
    if not source_post_url:
        return None
    m = _VK_WALL_RE.search(source_post_url)
    if not m:
        return None
    try:
        group_id = int(m.group(1))
    except Exception:
        return None
    if group_id <= 0:
        return None
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT screen_name FROM vk_source WHERE group_id=? LIMIT 1",
            (group_id,),
        )
        row = await cur.fetchone()
        await cur.close()
    screen = str(row[0] or "").strip() if row else ""
    if screen:
        return f"https://vk.com/{screen}"
    return f"https://vk.com/club{group_id}"


def is_festival_queue_enabled() -> bool:
    raw = (os.getenv("ENABLE_FESTIVAL_QUEUE") or "").strip().lower()
    if not raw:
        return False
    return raw in {"1", "true", "yes", "on"}


def festival_queue_status_text() -> str:
    return "вкл" if is_festival_queue_enabled() else "выкл"


def festival_queue_schedule_text() -> str:
    times = (os.getenv("FESTIVAL_QUEUE_TIMES_LOCAL") or "03:30,16:30").strip() or "03:30,16:30"
    tz = (os.getenv("FESTIVAL_QUEUE_TZ") or "Europe/Kaliningrad").strip() or "Europe/Kaliningrad"
    return f"{times} {tz}"


def _unique_preserve(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        val = (raw or "").strip()
        if not val:
            continue
        key = val.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def _extract_festival_name(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    for key in ("festival", "festival_name", "name", "series"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_festival_full(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    for key in ("festival_full", "full_name", "edition"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_for_grounding(value: str | None) -> str:
    raw = str(value or "").strip().casefold().replace("ё", "е")
    if not raw:
        return ""
    raw = re.sub(r"[^0-9a-zа-я]+", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _extract_explicit_day_festival_name(source_text: str | None) -> str | None:
    text = str(source_text or "").strip()
    if not text:
        return None
    for match in _QUOTED_TEXT_RE.finditer(text):
        candidate = str(match.group(1) or "").strip(" \t\r\n.,;:!?'\"")
        if candidate and re.search(r"\bдень\b", candidate, flags=re.IGNORECASE):
            return candidate
    match = _DAY_BIRTHDAY_RE.search(text)
    if match:
        return str(match.group(0) or "").strip(" \t\r\n.,;:!?'\"")
    return None


def _ground_day_festival_name(source_text: str | None, festival: str | None) -> str | None:
    explicit = _extract_explicit_day_festival_name(source_text)
    if not explicit:
        return festival
    if not festival:
        return explicit
    fest_raw = str(festival or "").strip()
    if not re.match(r"^\s*день\b", fest_raw, flags=re.IGNORECASE):
        return fest_raw
    source_norm = _normalize_for_grounding(source_text)
    fest_norm = _normalize_for_grounding(fest_raw)
    if fest_norm and fest_norm in source_norm:
        return fest_raw
    return explicit


def _extract_festival_context(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    value = str(event.get("festival_context") or "").strip().lower()
    if value in _FESTIVAL_CONTEXT_VALUES:
        return value
    return None


def _extract_dedup_links_from_text(text: str | None) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    return _unique_preserve(_URL_RE.findall(raw))


@dataclass(slots=True)
class FestivalContextDecision:
    context: FestivalContext
    festival: str | None = None
    festival_full: str | None = None
    dedup_links: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


def detect_festival_context(
    *,
    parsed_events: Sequence[dict[str, Any]] | None,
    festival_payload: dict[str, Any] | None,
    source_text: str | None,
    poster_texts: Sequence[str] | None = None,
    source_is_festival: bool = False,
    source_series: str | None = None,
) -> FestivalContextDecision:
    events = [ev for ev in (parsed_events or []) if isinstance(ev, dict)]
    payload = festival_payload if isinstance(festival_payload, dict) else {}
    joined_text = "\n".join(
        [
            str(source_text or ""),
            *[str(txt or "") for txt in (poster_texts or [])],
        ]
    ).strip()
    text_l = joined_text.casefold()

    explicit_context = str(payload.get("festival_context") or "").strip().lower()
    if explicit_context not in _FESTIVAL_CONTEXT_VALUES:
        explicit_context = ""
    for ev in events:
        ev_ctx = _extract_festival_context(ev)
        if ev_ctx == "festival_post":
            explicit_context = "festival_post"
            break
        if ev_ctx == "event_with_festival" and not explicit_context:
            explicit_context = ev_ctx

    festival = _extract_festival_name(payload)
    if not festival and events:
        festival = _extract_festival_name(events[0])
    festival_full = _extract_festival_full(payload)
    if not festival_full and events:
        festival_full = _extract_festival_full(events[0])
    if not festival and source_series:
        festival = source_series.strip() or None
    # TelegramMonitor may mark a "festival-like" event with boolean `festival: true`
    # (without providing a string festival name). In that case, treat the event title
    # as the festival series name so the source can be queued for festival processing.
    has_festival_flag = False
    for ev in events:
        if ev.get("festival") is True:
            has_festival_flag = True
            if not festival:
                title = str(ev.get("title") or "").strip()
                if title:
                    festival = title
            break
    if not festival and events:
        # Fallback for extractors that use `event_type: festival` instead of `festival: true`.
        ev0 = events[0]
        event_type = str(ev0.get("event_type") or "").strip().lower()
        if event_type == "festival":
            title = str(ev0.get("title") or "").strip()
            if title:
                festival = title
    festival = _ground_day_festival_name(joined_text, festival)

    dedup_links = _extract_dedup_links_from_text(joined_text)
    for key in ("website_url", "program_url", "vk_url", "tg_url", "ticket_url"):
        val = payload.get(key)
        if isinstance(val, str):
            dedup_links.extend(_extract_dedup_links_from_text(val))
    dedup_links = _unique_preserve(dedup_links)

    date_count = len(_DATE_RE.findall(joined_text))
    time_count = len(_TIME_RE.findall(joined_text))
    list_lines = len(
        [
            ln
            for ln in joined_text.splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)]|[—-])\s+\S+", ln)
        ]
    )
    program_signal = bool(_PROGRAM_RE.search(joined_text))
    day_signal = bool(_DAY_SERIES_RE.search(joined_text))
    range_signal = bool(_DATE_RANGE_RE.search(joined_text))
    multi_signal = bool(range_signal or date_count >= 2 or time_count >= 2 or list_lines >= 3)
    # If extractor already provided a multi-day range, treat it as a strong festival signal.
    for ev in events:
        date_value = str(ev.get("date") or "").strip()
        end_date_value = str(ev.get("end_date") or "").strip()
        if date_value and end_date_value and end_date_value != date_value:
            range_signal = True
            multi_signal = True
            break

    strong_events = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        title = str(ev.get("title") or "").strip()
        date_value = str(ev.get("date") or "").strip()
        location = str(ev.get("location_name") or "").strip()
        has_type = str(ev.get("event_type") or "").strip()
        has_ticket = str(ev.get("ticket_link") or "").strip()
        has_time = str(ev.get("time") or "").strip()
        if title and date_value and location and (has_time or has_type or has_ticket):
            strong_events += 1

    context: FestivalContext = "none"
    if explicit_context == "festival_post":
        context = "festival_post"
    elif explicit_context == "event_with_festival":
        context = "event_with_festival"
    elif (has_festival_flag and festival) or (
        events
        and str(events[0].get("event_type") or "").strip().lower() == "festival"
        and festival
    ):
        # Explicit extractor signal: the item is itself a festival/holiday/program.
        context = "festival_post"
    elif day_signal and program_signal:
        # Product rule: "День <...>" + program should always be festival_post,
        # even for one-day short programs.
        context = "festival_post"
    elif day_signal and (multi_signal or list_lines >= 2):
        context = "festival_post"
    elif (program_signal and multi_signal and (festival or source_is_festival)):
        context = "festival_post"
    elif not events and festival:
        context = "festival_post"
    elif festival and strong_events > 0:
        context = "event_with_festival"
    elif source_is_festival and (program_signal or day_signal):
        context = "festival_post"

    if context == "festival_post" and not festival and source_series:
        festival = source_series.strip() or None

    signals = {
        "program_signal": int(program_signal),
        "day_signal": int(day_signal),
        "multi_signal": int(multi_signal),
        "range_signal": int(range_signal),
        "dates_found": int(date_count),
        "times_found": int(time_count),
        "list_lines": int(list_lines),
        "strong_events": int(strong_events),
        "source_is_festival": int(bool(source_is_festival)),
    }
    return FestivalContextDecision(
        context=context,
        festival=festival,
        festival_full=festival_full,
        dedup_links=dedup_links,
        signals=signals,
    )


async def enqueue_festival_source(
    db: Database,
    *,
    source_kind: str,
    source_url: str,
    source_text: str | None,
    festival_context: str,
    festival_name: str | None,
    festival_full: str | None,
    festival_series: str | None = None,
    dedup_links: Sequence[str] | None = None,
    signals: dict[str, Any] | None = None,
    source_chat_username: str | None = None,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
    source_group_id: int | None = None,
    source_post_id: int | None = None,
) -> FestivalQueueItem:
    now = datetime.now(timezone.utc)
    clean_kind = (source_kind or "").strip().lower() or "vk"
    clean_url = (source_url or "").strip()
    clean_ctx = (festival_context or "").strip().lower()
    if clean_ctx not in _FESTIVAL_CONTEXT_VALUES:
        clean_ctx = "festival_post"

    links = _unique_preserve(list(dedup_links or []))
    signals_payload = dict(signals or {})

    async with db.get_session() as session:
        existing = (
            await session.execute(
                select(FestivalQueueItem).where(
                    FestivalQueueItem.source_kind == clean_kind,
                    FestivalQueueItem.source_url == clean_url,
                    FestivalQueueItem.status.in_(["pending", "running"]),
                )
            )
        ).scalar_one_or_none()
        if existing:
            existing.source_text = source_text or existing.source_text
            existing.festival_context = clean_ctx or existing.festival_context
            existing.festival_name = festival_name or existing.festival_name
            existing.festival_full = festival_full or existing.festival_full
            existing.festival_series = festival_series or existing.festival_series
            existing.dedup_links_json = links or list(existing.dedup_links_json or [])
            existing.signals_json = signals_payload or dict(existing.signals_json or {})
            existing.updated_at = now
            existing.next_run_at = now
            session.add(existing)
            await session.commit()
            await session.refresh(existing)
            logger.info(
                "festival_queue.enqueue requeued id=%s kind=%s url=%s",
                existing.id,
                clean_kind,
                clean_url,
            )
            return existing

        item = FestivalQueueItem(
            status="pending",
            source_kind=clean_kind,
            source_url=clean_url,
            source_text=source_text,
            source_chat_username=source_chat_username,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            source_group_id=source_group_id,
            source_post_id=source_post_id,
            festival_context=clean_ctx,
            festival_name=festival_name,
            festival_full=festival_full,
            festival_series=festival_series,
            dedup_links_json=links,
            signals_json=signals_payload,
            created_at=now,
            updated_at=now,
            next_run_at=now,
        )
        session.add(item)
        await session.commit()
        await session.refresh(item)
        logger.info(
            "festival_queue.enqueue created id=%s kind=%s url=%s",
            item.id,
            clean_kind,
            clean_url,
        )
        return item


def festival_queue_operator_lines(
    decision: FestivalContextDecision,
    *,
    queue_item_id: int | None = None,
) -> list[str]:
    lines = [f"festival_context: {decision.context}"]
    if decision.festival:
        lines.append(f"Фестиваль распознан: {decision.festival}")
    if decision.festival_full:
        lines.append(f"Выпуск: {decision.festival_full}")
    lines.append("Событие не создано")
    if queue_item_id:
        lines.append(f"Источник добавлен в фестивальную очередь (id={queue_item_id})")
    else:
        lines.append("Источник добавлен в фестивальную очередь")
    lines.append("Ручной запуск: /fest_queue")
    lines.append(f"Автозапуск очереди: {festival_queue_status_text()}")
    return lines


async def festival_queue_info_text(
    db: Database,
    *,
    source_kind: str | None = None,
    limit: int | None = None,
) -> str:
    clean_kind = (source_kind or "").strip().lower() or None
    if clean_kind and clean_kind not in {"vk", "tg", "url"}:
        clean_kind = None

    sample_limit = 10
    if limit is not None and int(limit) > 0:
        sample_limit = min(int(limit), 25)

    async with db.get_session() as session:
        base_where = []
        if clean_kind:
            base_where.append(FestivalQueueItem.source_kind == clean_kind)

        status_stmt = select(FestivalQueueItem.status, func.count()).group_by(FestivalQueueItem.status)
        if base_where:
            status_stmt = status_stmt.where(and_(*base_where))
        status_rows = (await session.execute(status_stmt)).all()
        status_counts = {str(st or ""): int(cnt or 0) for st, cnt in status_rows}

        active_where = [FestivalQueueItem.status.in_(["pending", "running"])]
        if clean_kind:
            active_where.append(FestivalQueueItem.source_kind == clean_kind)
        kind_stmt = (
            select(FestivalQueueItem.source_kind, func.count())
            .where(and_(*active_where))
            .group_by(FestivalQueueItem.source_kind)
        )
        kind_rows = (await session.execute(kind_stmt)).all()
        kind_counts = {str(k or ""): int(cnt or 0) for k, cnt in kind_rows}

        pending_where = [FestivalQueueItem.status == "pending"]
        if clean_kind:
            pending_where.append(FestivalQueueItem.source_kind == clean_kind)
        pending_stmt = (
            select(FestivalQueueItem)
            .where(and_(*pending_where))
            .order_by(FestivalQueueItem.next_run_at, FestivalQueueItem.created_at)
            .limit(sample_limit)
        )
        pending_items = (await session.execute(pending_stmt)).scalars().all()

        error_where = [FestivalQueueItem.status == "error"]
        if clean_kind:
            error_where.append(FestivalQueueItem.source_kind == clean_kind)
        error_stmt = (
            select(FestivalQueueItem)
            .where(and_(*error_where))
            .order_by(FestivalQueueItem.updated_at.desc(), FestivalQueueItem.created_at.desc())
            .limit(5)
        )
        error_items = (await session.execute(error_stmt)).scalars().all()

        running_where = [FestivalQueueItem.status == "running"]
        if clean_kind:
            running_where.append(FestivalQueueItem.source_kind == clean_kind)
        running_stmt = (
            select(FestivalQueueItem)
            .where(and_(*running_where))
            .order_by(FestivalQueueItem.updated_at.desc(), FestivalQueueItem.created_at.desc())
            .limit(5)
        )
        running_items = (await session.execute(running_stmt)).scalars().all()

    total = sum(status_counts.values())
    pending = status_counts.get("pending", 0)
    running = status_counts.get("running", 0)
    done = status_counts.get("done", 0)
    error = status_counts.get("error", 0)

    lines: list[str] = ["🎪 Фестивальная очередь (info)"]
    lines.append(f"Автозапуск очереди: {festival_queue_status_text()} (расписание: {festival_queue_schedule_text()})")
    if clean_kind:
        lines.append(f"Фильтр источника: {clean_kind}")
    lines.append(f"Счётчики: total={total} pending={pending} running={running} done={done} error={error}")
    if kind_counts:
        kind_parts = []
        for k in ["vk", "tg", "url"]:
            if k in kind_counts:
                kind_parts.append(f"{k}={kind_counts[k]}")
        if kind_parts:
            lines.append("Активные (pending+running) по источникам: " + " ".join(kind_parts))

    if pending_items:
        lines.append("Следующие pending:")
        for item in pending_items:
            url = str(item.source_url or "").strip()
            kind = str(item.source_kind or "").strip().lower()
            name = str(item.festival_name or item.festival_series or "").strip()
            attempts = int(item.attempts or 0)
            next_run = getattr(item, "next_run_at", None)
            next_run_txt = ""
            if next_run:
                try:
                    next_run_txt = next_run.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
                except Exception:
                    next_run_txt = str(next_run)
            extra = []
            if name:
                extra.append(f"festival={name}")
            if next_run_txt:
                extra.append(f"next={next_run_txt}")
            extra.append(f"attempts={attempts}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"- [{kind}] {url}{suffix}")
    else:
        lines.append("Pending элементов не найдено.")

    if running_items:
        lines.append("Running (последние):")
        for item in running_items:
            url = str(item.source_url or "").strip()
            kind = str(item.source_kind or "").strip().lower()
            attempts = int(item.attempts or 0)
            lines.append(f"- [{kind}] {url} (attempts={attempts})")

    if error_items:
        lines.append("Error (последние):")
        for item in error_items:
            url = str(item.source_url or "").strip()
            kind = str(item.source_kind or "").strip().lower()
            attempts = int(item.attempts or 0)
            err = str(item.last_error or "").strip()
            if len(err) > 180:
                err = err[:179].rstrip() + "…"
            err_txt = err or "—"
            lines.append(f"- [{kind}] {url} (attempts={attempts}) err={err_txt}")

    lines.append("Подсказка: /fest_queue для обработки очереди.")
    return "\n".join(lines)


@dataclass(slots=True)
class FestivalQueueRunReport:
    processed: int = 0
    success: int = 0
    failed: int = 0
    skipped: int = 0
    details: list[str] = field(default_factory=list)


def _festival_telegraph_url(fest: Festival | None) -> str | None:
    if not fest:
        return None
    url = str(getattr(fest, "telegraph_url", "") or "").strip()
    if url:
        return url
    path = str(getattr(fest, "telegraph_path", "") or "").strip().lstrip("/")
    if path:
        return f"https://telegra.ph/{path}"
    return None


async def _festival_report_links(
    db: Database,
    festival_name: str,
) -> tuple[str | None, str | None]:
    name = (festival_name or "").strip()
    if not name:
        return None, None
    fest_url: str | None = None
    index_url: str | None = None
    async with db.get_session() as session:
        fest = (
            await session.execute(select(Festival).where(Festival.name == name))
        ).scalar_one_or_none()
        fest_url = _festival_telegraph_url(fest)
        for key in ("festivals_index_url", "fest_index_url"):
            setting = await session.get(Setting, key)
            if setting and str(setting.value or "").strip():
                index_url = str(setting.value).strip()
                break
    return fest_url, index_url


async def _append_festival_activities(
    db: Database,
    *,
    festival_name: str,
    activities: Sequence[dict[str, Any]],
) -> int:
    clean_name = (festival_name or "").strip()
    if not clean_name or not activities:
        return 0
    payload: list[dict[str, Any]] = []
    seen: set[str] = set()
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        title = str(activity.get("title") or "").strip()
        if not title:
            continue
        key = "|".join(
            [
                title.casefold(),
                str(activity.get("date") or "").strip(),
                str(activity.get("time") or "").strip(),
                str(activity.get("location_name") or "").strip().casefold(),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        payload.append(
            {
                "title": title,
                "date": str(activity.get("date") or "").strip() or None,
                "time": str(activity.get("time") or "").strip() or None,
                "location_name": str(activity.get("location_name") or "").strip() or None,
                "program_only": True,
            }
        )
    if not payload:
        return 0

    async with db.get_session() as session:
        fest = (
            await session.execute(select(Festival).where(Festival.name == clean_name))
        ).scalar_one_or_none()
        if not fest:
            return 0
        existing = list(getattr(fest, "activities_json", None) or [])
        existing_keys = {
            "|".join(
                [
                    str(item.get("title") or "").strip().casefold(),
                    str(item.get("date") or "").strip(),
                    str(item.get("time") or "").strip(),
                    str(item.get("location_name") or "").strip().casefold(),
                ]
            )
            for item in existing
            if isinstance(item, dict)
        }
        added = 0
        for item in payload:
            key = "|".join(
                [
                    str(item.get("title") or "").strip().casefold(),
                    str(item.get("date") or "").strip(),
                    str(item.get("time") or "").strip(),
                    str(item.get("location_name") or "").strip().casefold(),
                ]
            )
            if key in existing_keys:
                continue
            existing_keys.add(key)
            existing.append(item)
            added += 1
        if added:
            fest.activities_json = existing
            session.add(fest)
            await session.commit()
        return added


def _looks_like_strong_program_event(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    title = str(item.get("title") or "").strip()
    date_value = str(item.get("date") or "").strip()
    location = str(item.get("location_name") or "").strip()
    time_value = str(item.get("time") or "").strip()
    event_type = str(item.get("event_type") or "").strip()
    ticket_link = str(item.get("ticket_link") or "").strip()
    if not (title and date_value and location):
        return False
    if time_value or event_type or ticket_link:
        return True
    return False


async def _process_vk_item(
    db: Database,
    item: FestivalQueueItem,
) -> dict[str, Any]:
    source_text = (item.source_text or "").strip()
    if not source_text:
        raise RuntimeError("vk queue item has no source_text")

    # Lightweight path: in live pipelines festival_queue items are usually created after
    # the event is already imported (Smart Update / VK intake). In that case we should
    # not re-run LLM parsing + smart_update again — just ensure the festival entity exists
    # and sync pages/index later in the queue runner.
    fest_hint = (item.festival_name or item.festival_series or "").strip()
    fest_hint = str(_ground_day_festival_name(source_text, fest_hint) or "").strip()
    if fest_hint:
        import main as main_mod

        website_url: str | None = None
        vk_url: str | None = None
        tg_url: str | None = None

        dedup_urls = _extract_dedup_links_from_text(source_text)
        for u in dedup_urls:
            if _is_vk_domain(u) or _is_tg_domain(u):
                continue
            website_url = u
            break
        vk_url = await _infer_vk_source_url(db, item.source_url)
        if vk_url and _looks_like_vk_wall(vk_url):
            vk_url = None

        fest_obj, _created, _updated = await main_mod.ensure_festival(
            db,
            fest_hint,
            full_name=item.festival_full,
            website_url=website_url,
            vk_url=vk_url,
            source_text=source_text,
            source_post_url=item.source_url,
        )
        return {
            "festival_name": fest_obj.name,
            "mode": "lightweight",
        }

    import main as main_mod
    from smart_event_update import EventCandidate, smart_event_update

    async with db.get_session() as session:
        fests = (await session.execute(select(Festival))).scalars().all()
    festival_names = sorted({(f.name or "").strip() for f in fests if (f.name or "").strip()})
    alias_pairs: list[tuple[str, int]] = []
    if festival_names:
        idx_map = {name: idx for idx, name in enumerate(festival_names)}
        for fest in fests:
            name = (fest.name or "").strip()
            if not name:
                continue
            idx = idx_map.get(name)
            if idx is None:
                continue
            base_norm = main_mod.normalize_alias(name)
            for alias in getattr(fest, "aliases", None) or []:
                norm = main_mod.normalize_alias(alias)
                if not norm or norm == base_norm:
                    continue
                alias_pairs.append((norm, idx))

    parsed = await main_mod.parse_event_via_llm(
        source_text,
        festival_names=festival_names,
        festival_alias_pairs=alias_pairs or None,
    )
    parsed_events = [ev for ev in list(parsed or []) if isinstance(ev, dict)]
    payload = getattr(parsed, "festival", None)
    decision = detect_festival_context(
        parsed_events=parsed_events,
        festival_payload=payload if isinstance(payload, dict) else None,
        source_text=source_text,
        source_is_festival=True,
        source_series=item.festival_series,
    )
    fest_name = (decision.festival or item.festival_name or item.festival_series or "").strip()
    if not fest_name:
        raise RuntimeError("festival name missing for vk queue item")

    start_date = None
    end_date = None
    website_url: str | None = None
    program_url: str | None = None
    ticket_url: str | None = None
    vk_url: str | None = None
    tg_url: str | None = None
    description: str | None = None
    photo_url: str | None = None
    photo_urls: list[str] | None = None
    if isinstance(payload, dict):
        start_date = main_mod.canonicalize_date(
            payload.get("start_date") or payload.get("date")
        )
        end_date = main_mod.canonicalize_date(payload.get("end_date"))
        website_url = str(payload.get("website_url") or "").strip() or None
        program_url = str(payload.get("program_url") or "").strip() or None
        ticket_url = str(payload.get("ticket_url") or "").strip() or None
        vk_url = str(payload.get("vk_url") or "").strip() or None
        tg_url = str(payload.get("tg_url") or "").strip() or None
        description = (
            str(payload.get("description") or payload.get("summary") or "").strip() or None
        )
        raw_photo_url = str(payload.get("photo_url") or "").strip() or None
        raw_photo_urls = payload.get("photo_urls")
        if isinstance(raw_photo_urls, list):
            cleaned = [str(x).strip() for x in raw_photo_urls if isinstance(x, str) and str(x).strip()]
            photo_urls = cleaned or None
        if raw_photo_url:
            photo_url = raw_photo_url
        elif photo_urls:
            photo_url = photo_urls[0]

    # Fallback: infer socials/website from the VK source + any URLs in the source text.
    dedup_urls = _extract_dedup_links_from_text(source_text)
    if not website_url:
        for u in dedup_urls:
            if _is_vk_domain(u) or _is_tg_domain(u):
                continue
            website_url = u
            break
    if not vk_url:
        vk_url = await _infer_vk_source_url(db, item.source_url)
    if vk_url and _looks_like_vk_wall(vk_url):
        vk_url = None

    fest_obj, _created, _updated = await main_mod.ensure_festival(
        db,
        fest_name,
        full_name=decision.festival_full or item.festival_full,
        photo_url=photo_url,
        photo_urls=photo_urls,
        website_url=website_url,
        program_url=program_url,
        vk_url=vk_url,
        tg_url=tg_url,
        ticket_url=ticket_url,
        description=description,
        start_date=start_date,
        end_date=end_date,
        source_text=source_text,
        source_post_url=item.source_url,
    )

    created_events = 0
    program_only: list[dict[str, Any]] = []
    for ev in parsed_events:
        if not _looks_like_strong_program_event(ev):
            program_only.append(ev)
            continue
        candidate = EventCandidate(
            source_type="vk",
            source_url=item.source_url,
            source_text=source_text,
            title=str(ev.get("title") or "").strip() or None,
            date=str(ev.get("date") or "").strip() or None,
            time=str(ev.get("time") or "").strip() or None,
            end_date=str(ev.get("end_date") or "").strip() or None,
            festival=fest_obj.name,
            location_name=str(ev.get("location_name") or "").strip() or None,
            location_address=str(ev.get("location_address") or "").strip() or None,
            city=str(ev.get("city") or "").strip() or None,
            ticket_link=str(ev.get("ticket_link") or "").strip() or None,
            ticket_price_min=ev.get("ticket_price_min"),
            ticket_price_max=ev.get("ticket_price_max"),
            event_type=str(ev.get("event_type") or "").strip() or None,
            emoji=str(ev.get("emoji") or "").strip() or None,
            is_free=ev.get("is_free"),
            pushkin_card=ev.get("pushkin_card"),
            search_digest=str(ev.get("search_digest") or "").strip() or None,
            raw_excerpt=str(ev.get("short_description") or "").strip() or None,
            festival_context="event_with_festival",
            festival_full=decision.festival_full or item.festival_full,
            festival_source=True,
            festival_series=item.festival_series,
        )
        result = await smart_event_update(db, candidate, check_source_url=False)
        if result.event_id:
            created_events += 1
        else:
            program_only.append(ev)

    added_activities = await _append_festival_activities(
        db,
        festival_name=fest_obj.name,
        activities=program_only,
    )
    return {
        "festival_name": fest_obj.name,
        "events_created": created_events,
        "activities_added": added_activities,
    }


async def _process_tg_item(
    db: Database,
    item: FestivalQueueItem,
    *,
    bot: Any | None = None,
    chat_id: int | None = None,
    trigger: str = "manual",
    operator_id: int | None = None,
) -> dict[str, Any]:
    from source_parsing.telegram.service import run_telegram_monitor

    username = str(item.source_chat_username or "").strip()
    message_id = item.source_message_id
    if (not username or not message_id) and item.source_url:
        m = re.search(r"t\.me/([^/]+)/([0-9]+)", item.source_url, flags=re.IGNORECASE)
        if m:
            username = username or str(m.group(1) or "").strip()
            message_id = message_id or int(m.group(2))
    if not username or not message_id:
        raise RuntimeError("tg queue item missing source username/message id")

    ctx = str(getattr(item, "festival_context", "") or "").strip().lower()
    if ctx == "event_with_festival" and (
        (item.festival_name or "").strip() or (item.festival_series or "").strip()
    ):
        # Lightweight path: for `event_with_festival` the event already exists in DB
        # (created by Smart Update). Do not re-run Telegram Monitoring/Kaggle here —
        # just ensure the festival page/index are created/updated.
        import main as main_mod

        fest_name = (item.festival_name or item.festival_series or "").strip()
        fest_obj, _created, _updated = await main_mod.ensure_festival(
            db,
            fest_name,
            full_name=item.festival_full,
            source_text=(item.source_text or "").strip() or None,
            source_post_url=item.source_url,
            source_chat_id=item.source_chat_id,
            source_message_id=item.source_message_id,
        )

        # Try to seed at least one cover image from the Telegram message itself.
        # Some TG posts don't get posters/images in DB (e.g. if they were imported earlier
        # without media), but DoD expects festival pages to have posters when available.
        if not getattr(fest_obj, "photo_url", None):
            try:
                import base64
                import json
                import os

                from telethon import TelegramClient
                from telethon.sessions import StringSession

                def _env(name: str) -> str:
                    return (os.getenv(name) or "").strip()

                api_id_raw = _env("TELEGRAM_API_ID") or _env("TG_API_ID")
                api_hash = _env("TELEGRAM_API_HASH") or _env("TG_API_HASH")
                bundle_b64 = _env("TELEGRAM_AUTH_BUNDLE_E2E")
                session_string = _env("TELEGRAM_SESSION")
                device_kwargs: dict[str, object] = {}
                if bundle_b64:
                    raw = base64.urlsafe_b64decode(bundle_b64.encode("ascii")).decode("utf-8")
                    bundle = json.loads(raw)
                    session_string = (bundle.get("session") or "").strip()
                    for key in (
                        "device_model",
                        "system_version",
                        "app_version",
                        "lang_code",
                        "system_lang_code",
                    ):
                        val = bundle.get(key)
                        if val:
                            device_kwargs[key] = val

                if api_id_raw and api_hash and session_string and username and message_id:
                    async with TelegramClient(
                        StringSession(session_string),
                        int(api_id_raw),
                        api_hash,
                        **device_kwargs,
                    ) as client:
                        msg = await client.get_messages(username, ids=int(message_id))
                        if msg and getattr(msg, "media", None):
                            data = await client.download_media(msg, file=bytes)
                            if isinstance(data, (bytes, bytearray)) and data:
                                urls, _msg = await main_mod.upload_images(
                                    [(bytes(data), f"tg_{username}_{int(message_id)}.jpg")],
                                    limit=1,
                                    force=True,
                                    event_hint=f"festival:{fest_obj.name}",
                                )
                                if urls:
                                    await main_mod.ensure_festival(
                                        db,
                                        fest_obj.name,
                                        photo_url=urls[0],
                                        photo_urls=list(urls),
                                    )
            except Exception:
                logger.warning(
                    "festival_queue: failed to seed cover from tg message for %s",
                    fest_obj.name,
                    exc_info=True,
                )

        return {
            "source_username": username,
            "message_id": int(message_id),
            "festival_name": fest_obj.name,
            "mode": "lightweight",
        }

    async with db.get_session() as session:
        source = (
            await session.execute(
                select(TelegramSource).where(TelegramSource.username == username)
            )
        ).scalar_one_or_none()
        if not source:
            source = TelegramSource(
                username=username,
                enabled=True,
                festival_source=True,
                festival_series=item.festival_series or item.festival_name,
            )
            session.add(source)
            await session.flush()
        else:
            source.enabled = True
            if item.festival_series and not source.festival_series:
                source.festival_series = item.festival_series
            if item.festival_series or item.festival_name:
                source.festival_source = True
        forced = await session.get(TelegramSourceForceMessage, (source.id, int(message_id)))
        if not forced:
            session.add(
                TelegramSourceForceMessage(
                    source_id=source.id,
                    message_id=int(message_id),
                )
            )
        await session.commit()

    monitor_error: str | None = None
    try:
        await run_telegram_monitor(
            db,
            bot=bot,
            chat_id=chat_id,
            send_progress=True,
            trigger=trigger,
            operator_id=operator_id,
        )
    except Exception as exc:
        fest_hint = (item.festival_name or item.festival_series or "").strip()
        if fest_hint and _is_recoverable_tg_monitor_error(exc):
            monitor_error = str(exc or "").strip() or "tg_monitor_error"
            logger.warning(
                "festival_queue.tg fallback to ensure_festival: festival=%s source=%s error=%s",
                fest_hint,
                item.source_url,
                monitor_error,
            )
        else:
            raise
    fest_name = (item.festival_name or item.festival_series or "").strip()
    if fest_name:
        # Safety net: Telegram monitoring may finish with 0 imported events for a forced
        # message. Keep queue processing deterministic by ensuring a festival stub exists
        # from the queue item itself, so page/index sync does not fail with
        # "festival page not created".
        import main as main_mod

        fest_obj, _created, _updated = await main_mod.ensure_festival(
            db,
            fest_name,
            full_name=item.festival_full,
            source_text=(item.source_text or "").strip() or None,
            source_post_url=item.source_url,
            source_chat_id=item.source_chat_id,
            source_message_id=item.source_message_id,
        )
        fest_name = fest_obj.name
    result = {
        "source_username": username,
        "message_id": int(message_id),
        "festival_name": fest_name,
    }
    if monitor_error:
        result["monitor_error"] = monitor_error
        result["mode"] = "fallback_ensure_festival"
    return result


async def _process_url_item(
    db: Database,
    item: FestivalQueueItem,
    *,
    bot: Any | None,
    chat_id: int | None,
) -> dict[str, Any]:
    from source_parsing.festival_parser import process_festival_url

    if not bot or chat_id is None:
        raise RuntimeError("bot/chat_id required for URL festival parser run")
    fest, uds_url, llm_url = await process_festival_url(
        db=db,
        bot=bot,
        chat_id=int(chat_id),
        url=item.source_url,
        status_callback=None,
    )
    return {
        "festival_name": fest.name,
        "telegraph_url": fest.telegraph_url,
        "uds_url": uds_url,
        "llm_log_url": llm_url,
    }


async def process_festival_queue(
    db: Database,
    *,
    bot: Any | None = None,
    chat_id: int | None = None,
    progress_message_id: int | None = None,
    source_kind: str | None = None,
    limit: int | None = None,
    trigger: str = "manual",
    operator_id: int | None = None,
    run_id: str | None = None,
) -> FestivalQueueRunReport:
    async def _progress(text: str) -> None:
        if not bot or chat_id is None or not progress_message_id:
            return
        payload = (text or "").strip()
        if not payload:
            return
        try:
            await bot.edit_message_text(
                chat_id=int(chat_id),
                message_id=int(progress_message_id),
                text=payload,
                disable_web_page_preview=True,
            )
        except Exception:
            return

    report = FestivalQueueRunReport()
    ops_run_id = await start_ops_run(
        db,
        kind="festival_queue",
        trigger=trigger,
        chat_id=chat_id,
        operator_id=operator_id,
        details={
            "run_id": run_id,
            "source_kind": source_kind,
            "limit": limit,
        },
    )
    status = "success"
    now = datetime.now(timezone.utc)
    try:
        stale_min_raw = (os.getenv("FESTIVAL_QUEUE_STALE_RUNNING_MINUTES") or "60").strip()
        try:
            stale_minutes = int(stale_min_raw)
        except Exception:
            stale_minutes = 60
        if stale_minutes > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
            async with db.get_session() as session:
                res = await session.execute(
                    select(func.count())
                    .select_from(FestivalQueueItem)
                    .where(
                        FestivalQueueItem.status == "running",
                        FestivalQueueItem.updated_at < cutoff,
                    )
                )
                stale_count = int(res.scalar() or 0)
                if stale_count > 0:
                    await session.execute(
                        update(FestivalQueueItem)
                        .where(
                            FestivalQueueItem.status == "running",
                            FestivalQueueItem.updated_at < cutoff,
                        )
                        .values(
                            status="pending",
                            last_error="stale running recovered",
                            next_run_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
                    await session.commit()
                    await _progress(
                        f"🎪 Фестивальная очередь: восстановлено элементов running→pending: {stale_count}. Продолжаю…"
                    )

        async with db.get_session() as session:
            stmt = select(FestivalQueueItem).where(
                FestivalQueueItem.status == "pending",
                FestivalQueueItem.next_run_at <= now,
            )
            if source_kind:
                stmt = stmt.where(FestivalQueueItem.source_kind == source_kind.strip().lower())
            stmt = stmt.order_by(FestivalQueueItem.created_at)
            if limit and int(limit) > 0:
                stmt = stmt.limit(int(limit))
            items = list((await session.execute(stmt)).scalars().all())

        def _clean_fest_name(raw: str | None) -> str:
            return str(raw or "").strip()

        def _festival_group_key(it: FestivalQueueItem) -> tuple[str, str] | None:
            kind = str(getattr(it, "source_kind", "") or "").strip().lower()
            if kind not in {"vk", "tg"}:
                return None
            name = _clean_fest_name(getattr(it, "festival_name", None)) or _clean_fest_name(
                getattr(it, "festival_series", None)
            )
            if not name:
                return None
            return kind, name.casefold()

        def _select_primary(group_items: list[FestivalQueueItem]) -> FestivalQueueItem:
            def _score(it: FestivalQueueItem) -> tuple[int, int, int]:
                # Prefer explicit festival posts, then items that contain actionable links,
                # then longer texts (more context), finally newest updated_at/created_at.
                ctx = str(getattr(it, "festival_context", "") or "").strip().lower()
                ctx_bonus = 100 if ctx == "festival_post" else 0
                txt = str(getattr(it, "source_text", "") or "")
                urls = _extract_dedup_links_from_text(txt)
                link_bonus = 10 * len(urls)
                len_bonus = min(50, int(len(txt) / 120))
                return (ctx_bonus + link_bonus + len_bonus, len(txt), int(getattr(it, "id", 0) or 0))

            return sorted(group_items, key=_score, reverse=True)[0]

        # Group queue items by (source_kind, festival_name) when possible.
        groups: list[dict[str, Any]] = []
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for it in items:
            gk = _festival_group_key(it)
            if not gk:
                groups.append({"key": ("item", int(it.id)), "items": [it], "primary": it})
                continue
            g = by_key.get(gk)
            if not g:
                g = {"key": gk, "items": [], "primary": None}
                by_key[gk] = g
                groups.append(g)
            g["items"].append(it)
        for g in groups:
            g["primary"] = _select_primary(list(g["items"]))

        total = len(groups)
        if total <= 0:
            await _progress("🎪 Фестивальная очередь: нет элементов для обработки.")
        else:
            raw_total = len(items)
            await _progress(
                f"🎪 Фестивальная очередь: найдено элементов: {raw_total}. Групп по фестивалям: {total}. Начинаю…"
            )

        import main as main_mod

        for idx, g in enumerate(groups, start=1):
            group_items: list[FestivalQueueItem] = list(g["items"])
            primary: FestivalQueueItem = g["primary"]
            report.processed += len(group_items)

            kind = str(getattr(primary, "source_kind", "") or "").strip().lower()
            fest_hint = _clean_fest_name(getattr(primary, "festival_name", None)) or _clean_fest_name(
                getattr(primary, "festival_series", None)
            ) or _clean_fest_name(getattr(primary, "festival_full", None))
            url = str(getattr(primary, "source_url", "") or "").strip()
            hint_line = f"\nФестиваль: {fest_hint}" if fest_hint else ""
            url_line = f"\nИсточник: {url[:300]}" if url else ""
            await _progress(
                f"🎪 Фестивальная очередь: {idx}/{total}\nСтатус: running\nkind={kind}{hint_line}{url_line}\nitems={len(group_items)}"
            )

            item_ids = [int(it.id) for it in group_items if getattr(it, "id", None) is not None]
            if not item_ids:
                report.skipped += len(group_items)
                continue

            async with db.get_session() as session:
                await session.execute(
                    update(FestivalQueueItem)
                    .where(FestivalQueueItem.id.in_(item_ids))
                    .values(
                        status="running",
                        last_error=None,
                        attempts=(func.coalesce(FestivalQueueItem.attempts, 0) + 1),
                        updated_at=datetime.now(timezone.utc),
                    )
                )
                await session.commit()

            try:
                # Apply festival metadata from a "best" item (links/context) and then
                # from the newest item (fresh excerpt/source_text), without re-running
                # heavy imports for every queue row.
                items_for_apply: list[FestivalQueueItem] = [primary]
                newest = max(
                    group_items,
                    key=lambda it: (
                        getattr(it, "created_at", None) or datetime.min.replace(tzinfo=timezone.utc),
                        int(getattr(it, "id", 0) or 0),
                    ),
                )
                if int(getattr(newest, "id", 0) or 0) != int(getattr(primary, "id", 0) or 0):
                    items_for_apply.append(newest)

                result: dict[str, Any] = {}
                fest_name = ""
                for apply_item in items_for_apply:
                    if apply_item.source_kind == "url":
                        result = await _process_url_item(
                            db, apply_item, bot=bot, chat_id=chat_id
                        )
                    elif apply_item.source_kind == "tg":
                        result = await _process_tg_item(
                            db,
                            apply_item,
                            bot=bot,
                            chat_id=chat_id,
                            trigger=trigger,
                            operator_id=operator_id,
                        )
                    elif apply_item.source_kind == "vk":
                        result = await _process_vk_item(db, apply_item)
                    else:
                        raise RuntimeError(f"unsupported source_kind={apply_item.source_kind}")
                    fest_name = str(result.get("festival_name") or fest_name or "").strip()

                if fest_name:
                    async with db.get_session() as session:
                        fest_obj = (
                            await session.execute(select(Festival).where(Festival.name == fest_name))
                        ).scalar_one_or_none()
                    if fest_obj:
                        try:
                            await main_mod.try_set_fest_cover_from_program(db, fest_obj)
                        except Exception:
                            logger.warning(
                                "festival_queue: failed to refresh cover for %s",
                                fest_name,
                                exc_info=True,
                            )
                    await main_mod.sync_festival_page(db, fest_name)
                    async with db.get_session() as session:
                        fresh = (
                            await session.execute(
                                select(Festival.telegraph_url, Festival.telegraph_path).where(
                                    Festival.name == fest_name
                                )
                            )
                        ).first()
                    url_ok = False
                    if fresh:
                        url_ok = bool((fresh[0] or "").strip()) or bool((fresh[1] or "").strip())
                    if not url_ok:
                        raise RuntimeError(f"festival page not created for '{fest_name}'")
                    await main_mod.sync_festivals_index_page(db)
                    index_url = await main_mod.get_setting_value(db, "fest_index_url")
                    if not index_url:
                        raise RuntimeError("festivals index url not set after sync")

                async with db.get_session() as session:
                    await session.execute(
                        update(FestivalQueueItem)
                        .where(FestivalQueueItem.id.in_(item_ids))
                        .values(
                            status="done",
                            festival_name=fest_name or primary.festival_name,
                            result_json=dict(result or {}),
                            last_error=None,
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
                    await session.commit()

                report.success += len(group_items)
                if fest_name:
                    fest_url, index_url = await _festival_report_links(db, fest_name)
                    lines = [f"Фестиваль обновлён: {fest_name}"]
                    if fest_url:
                        lines.append(f"Открыть страницу фестиваля: {fest_url}")
                    if index_url:
                        lines.append(f"Открыть страницу Фестивали: {index_url}")
                    report.details.append("\n".join(lines))
                await _progress(f"🎪 Фестивальная очередь: {idx}/{total}\nСтатус: ✅ done\nfestival={fest_name or '—'}")
            except Exception as exc:
                logger.exception("festival_queue group failed key=%s", g.get("key"))
                async with db.get_session() as session:
                    await session.execute(
                        update(FestivalQueueItem)
                        .where(FestivalQueueItem.id.in_(item_ids))
                        .values(
                            status="error",
                            last_error=str(exc),
                            updated_at=datetime.now(timezone.utc),
                        )
                    )
                    await session.commit()
                report.failed += len(group_items)
                report.details.append(
                    "\n".join(
                        [
                            "❌ Ошибка обработки фестивальной очереди",
                            f"kind={str(getattr(primary, 'source_kind', '') or '').strip().lower()}",
                            f"festival={fest_hint or '—'}",
                            f"url={url}" if url else "url=—",
                            f"error={exc}",
                            "Подсказка: /fest_queue -i (посмотреть состояние очереди)",
                        ]
                    )
                )
                await _progress(
                    f"🎪 Фестивальная очередь: {idx}/{total}\nСтатус: ❌ error\nerror={str(exc)[:280]}"
                )
            await asyncio.sleep(0)
        if total > 0:
            await _progress(
                "🎪 Фестивальная очередь: завершено.\n"
                f"processed={report.processed} success={report.success} failed={report.failed} skipped={report.skipped}"
            )
        return report
    except Exception:
        status = "error"
        raise
    finally:
        await finish_ops_run(
            db,
            run_id=ops_run_id,
            status=status,
            metrics={
                "processed": int(report.processed),
                "success": int(report.success),
                "failed": int(report.failed),
                "skipped": int(report.skipped),
                "limit": int(limit) if isinstance(limit, int) else (int(limit or 0) if str(limit or "").isdigit() else 0),
            },
            details={
                "run_id": run_id,
                "source_kind": source_kind,
                "details": list(report.details or [])[:30],
            },
        )
