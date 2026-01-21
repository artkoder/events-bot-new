from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import random
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from aiogram import types
from sqlalchemy import select, and_, or_

from db import Database
from main import ask_4o, format_day_pretty, parse_time_range
from models import (
    Event,
    EventPoster,
    VideoAnnounceEventHit,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceLLMTrace,
    VideoAnnounceSession,
)
from .about import normalize_about_with_fallback
from .kaggle_client import KaggleClient
from .prompts import selection_prompt, selection_response_format, about_fill_prompt, about_fill_response_format
from .custom_types import (
    RankedChoice,
    RankedEvent,
    RenderPayload,
    SelectionBuildResult,
    SelectionContext,
    VideoProfile,
)

logger = logging.getLogger(__name__)

TELEGRAPH_EXCERPT_LIMIT = 1200
POSTER_OCR_EXCERPT_LIMIT = 800
TRACE_MAX_LEN = 100_000


def _is_fair_event(ev: Event) -> bool:
    return (getattr(ev, "event_type", "") or "").strip().casefold() == "ярмарка"


def _format_fair_time(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text or text in {"00:00", "0:00"}:
        return None
    parsed = parse_time_range(text)
    if not parsed:
        return text
    start, end = parsed
    start_text = start.strftime("%H:%M")
    if end:
        return f"с {start_text} до {end.strftime('%H:%M')}"
    return f"с {start_text}"


def _build_fair_schedule_text(ev: Event) -> str | None:
    raw_end = getattr(ev, "end_date", None)
    if not raw_end:
        return None
    try:
        end_dt = date.fromisoformat(raw_end.split("..", 1)[0])
    except Exception:
        return None
    try:
        start_dt = date.fromisoformat((ev.date or "").split("..", 1)[0])
    except Exception:
        start_dt = None
    if start_dt and end_dt <= start_dt:
        return None
    end_label = format_day_pretty(end_dt)
    time_text = _format_fair_time(getattr(ev, "time", None))
    if time_text:
        return f"по {end_label} {time_text}"
    return f"по {end_label}"


def _build_about(
    *,
    about: str | None,
    event: Event,
    ocr_text: str | None = None,
    ocr_title: str | None = None,
    title: str | None = None,
) -> str:
    return normalize_about_with_fallback(
        about,
        ocr_text=ocr_title, # Pass ocr_title as ocr_text for deduplication as per requirements
    )


def _log_event_selection_stats(events: Sequence[Event]) -> None:
    if not events:
        logger.info("video_announce: selection stats count=0")
        return

    dates: list[date] = []
    by_date = defaultdict(int)
    for ev in events:
        try:
            raw_date = ev.date.split("..", 1)[0]
            d = date.fromisoformat(raw_date)
            dates.append(d)
            by_date[raw_date] += 1
        except (ValueError, AttributeError, IndexError):
            continue

    if not dates:
        return

    min_date = min(dates)
    max_date = max(dates)
    period = (
        f"{min_date.isoformat()}..{max_date.isoformat()}"
        if min_date != max_date
        else min_date.isoformat()
    )

    sorted_keys = sorted(by_date.keys())
    breakdown = ", ".join(f"{d}={by_date[d]}" for d in sorted_keys)

    logger.info(
        "video_announce: selection stats count=%d period=%s breakdown={%s}",
        len(events),
        period,
        breakdown,
    )


def _filter_events_with_posters(events: Sequence[Event]) -> list[Event]:
    filtered = [
        e
        for e in events
        if (getattr(e, "photo_count", 0) or 0) > 0
        and any((getattr(e, "photo_urls", []) or []))
    ]
    if len(filtered) != len(events):
        logger.info(
            "video_announce: dropped events without posters total=%d filtered=%d",  # noqa: G004
            len(events),
            len(filtered),
        )
    return filtered


def _filter_events_by_ticket_status(
    events: Sequence[Event], *, allow_sold_out: bool
) -> list[Event]:
    if allow_sold_out:
        return list(events)
    filtered = [e for e in events if getattr(e, "ticket_status", None) != "sold_out"]
    if len(filtered) != len(events):
        logger.info(
            "video_announce: dropped sold_out events total=%d filtered=%d",  # noqa: G004
            len(events),
            len(filtered),
        )
    return filtered


def _normalize_group_value(value: str | None) -> str:
    text = (value or "").strip().casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _event_group_key(ev: Event) -> str:
    return "|".join(
        [
            _normalize_group_value(getattr(ev, "title", "")),
            _normalize_group_value(getattr(ev, "location_name", "")),
            _normalize_group_value(getattr(ev, "city", "")),
        ]
    )


def _has_poster(ev: Event) -> bool:
    return (getattr(ev, "photo_count", 0) or 0) > 0 and any(
        (getattr(ev, "photo_urls", []) or [])
    )


def _short_time_text(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    first = (
        text.split(" ")[0]
        .split("-")[0]
        .split("–")[0]
        .split("—")[0]
    )
    if first in ("00:00", "0:00"):
        return None
    if ":" in text:
        return text[:5]
    return text


def _time_sort_key(value: str | None) -> int:
    if not value:
        return 24 * 60 + 1
    match = re.search(r"(\d{1,2}):(\d{2})", value)
    if not match:
        return 24 * 60 + 1
    return int(match.group(1)) * 60 + int(match.group(2))


def _event_sort_key(ev: Event) -> tuple[date, int, int]:
    try:
        base_date = date.fromisoformat(ev.date.split("..", 1)[0])
    except Exception:
        base_date = date.max
    return (base_date, _time_sort_key(_short_time_text(ev.time)), ev.id or 0)


def _build_schedule_info(
    events: Sequence[Event],
) -> tuple[str, list[dict[str, list[str]]]]:
    date_map: dict[date, set[str]] = defaultdict(set)
    date_seen: set[date] = set()
    for ev in events:
        raw_date = (ev.date or "").split("..", 1)[0]
        try:
            d = date.fromisoformat(raw_date)
        except Exception:
            continue
        date_seen.add(d)
        time_text = _short_time_text(ev.time)
        if time_text:
            date_map[d].add(time_text)
    parts: list[str] = []
    occurrences: list[dict[str, list[str]]] = []
    for d in sorted(date_seen):
        label = format_day_pretty(d)
        times = sorted(date_map.get(d, set()), key=_time_sort_key)
        occurrences.append({"date": d.isoformat(), "times": times})
        if times:
            parts.append(f"{label}: {', '.join(times)}")
        else:
            parts.append(label)
    return "\n".join(parts), occurrences


def _build_schedule_text(events: Sequence[Event]) -> str:
    return _build_schedule_info(events)[0]


def _dedupe_events(
    events: Sequence[Event],
    *,
    promoted_ids: set[int],
    primary_end: date,
) -> tuple[list[Event], dict[int, str], dict[int, list[dict[str, list[str]]]]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        grouped[_event_group_key(ev)].append(ev)

    primary: list[tuple[Event, str, list[dict[str, list[str]]]]] = []
    flexible: list[tuple[Event, str, list[dict[str, list[str]]]]] = []
    fallback: list[tuple[Event, str, list[dict[str, list[str]]]]] = []

    for group_events in grouped.values():
        schedule_events = [
            ev
            for ev in group_events
            if getattr(ev, "ticket_status", None) != "sold_out"
        ]
        if not schedule_events:
            continue
        poster_events = [ev for ev in schedule_events if _has_poster(ev)]
        if not poster_events:
            continue
        rep = sorted(poster_events, key=_event_sort_key)[0]
        schedule_text, occurrences = _build_schedule_info(schedule_events)
        fair_schedule = _build_fair_schedule_text(rep) if _is_fair_event(rep) else None
        if fair_schedule:
            schedule_text = fair_schedule
            occurrences = []
        include = any(
            (getattr(ev, "video_include_count", 0) or 0) > 0
            or ev.id in promoted_ids
            for ev in schedule_events
        )
        try:
            earliest = min(
                date.fromisoformat(ev.date.split("..", 1)[0])
                for ev in schedule_events
            )
        except Exception:
            earliest = date.max
        bucket = primary if include and earliest <= primary_end else flexible if include else fallback
        bucket.append((rep, schedule_text, occurrences))

    primary.sort(key=lambda pair: _event_sort_key(pair[0]))
    flexible.sort(key=lambda pair: _event_sort_key(pair[0]))
    fallback.sort(key=lambda pair: _event_sort_key(pair[0]))

    combined = primary + flexible
    selected = combined if combined else fallback
    schedule_map = {
        ev.id: schedule for ev, schedule, _ in selected if ev.id and schedule
    }
    occurrences_map = {
        ev.id: occ for ev, _, occ in selected if ev.id and occ
    }
    return [ev for ev, _, _ in selected], schedule_map, occurrences_map


async def fetch_profiles() -> list[VideoProfile]:
    from pathlib import Path
    import json

    profiles_path = Path(__file__).parent / "assets" / "profiles.json"
    profiles: list[VideoProfile] = []
    if profiles_path.exists():
        raw = json.loads(profiles_path.read_text(encoding="utf-8"))
        for item in raw:
            profiles.append(
                VideoProfile(
                    key=item.get("key", "default"),
                    title=item.get("title", "Профиль"),
                    description=item.get("description", ""),
                    prompt_name=item.get("prompt_name", "script"),
                    kaggle_dataset=item.get("kaggle_dataset"),
                )
            )
    if not profiles:
        profiles.append(
            VideoProfile(
                key="default", title="Быстрый обзор", description="Общий режим"
            )
        )
    return profiles


async def fetch_candidates(
    db: Database, ctx: SelectionContext
) -> tuple[
    list[Event],
    dict[int, str],
    dict[int, list[dict[str, list[str]]]],
]:
    today = ctx.target_date or datetime.now(ctx.tz).date()
    primary_end = today + timedelta(days=max(ctx.primary_window_days, 0))
    fallback_end = today + timedelta(days=max(ctx.fallback_window_days, ctx.primary_window_days))
    today_iso = today.isoformat()
    fallback_iso = fallback_end.isoformat()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                or_(
                    and_(Event.date >= today_iso, Event.date <= fallback_iso),
                    and_(
                        Event.event_type == "ярмарка",
                        Event.end_date.is_not(None),
                        Event.end_date >= today_iso,
                        Event.date <= fallback_iso,
                    ),
                )
            )
            .order_by(Event.date, Event.time, Event.id)
        )
        events = result.scalars().all()
    exclude_sold_out = ctx.profile is None or ctx.profile.key == "default"
    events = _filter_events_by_ticket_status(events, allow_sold_out=not exclude_sold_out)
    promoted_ids = set(ctx.promoted_event_ids or set())
    if ctx.profile is None or ctx.profile.key == "default":
        deduped, schedule_map, occurrences_map = _dedupe_events(
            events,
            promoted_ids=promoted_ids,
            primary_end=primary_end,
        )
        if not deduped:
            fallback = _filter_events_with_posters(events)[: ctx.candidate_limit]
            return fallback, {}, {}
        selected = deduped[: ctx.candidate_limit]
        filtered_map = {ev.id: schedule_map.get(ev.id, "") for ev in selected if ev.id}
        filtered_occurrences = {
            ev.id: occurrences_map.get(ev.id, [])
            for ev in selected
            if ev.id
        }
        return (
            selected,
            {k: v for k, v in filtered_map.items() if v},
            {k: v for k, v in filtered_occurrences.items() if v},
        )

    events = _filter_events_with_posters(events)
    filtered: list[Event] = []
    flexible: list[Event] = []
    for e in events:
        include = (e.video_include_count or 0) > 0 or e.id in promoted_ids
        if not include:
            continue
        event_date = date.fromisoformat(e.date.split("..", 1)[0])
        if event_date <= primary_end:
            filtered.append(e)
        else:
            flexible.append(e)
    combined = filtered + flexible
    selected = combined[: ctx.candidate_limit] if combined else events[: ctx.candidate_limit]
    schedule_map: dict[int, str] = {}
    for ev in selected:
        if ev.id and _is_fair_event(ev):
            fair_schedule = _build_fair_schedule_text(ev)
            if fair_schedule:
                schedule_map[ev.id] = fair_schedule
    if not combined:
        return selected, schedule_map, {}
    return selected, schedule_map, {}


def _score_events(client: KaggleClient, events: Iterable[Event]) -> list[RankedEvent]:
    scores = client.score(events)
    ranked: list[RankedEvent] = []
    for idx, event in enumerate(
        sorted(events, key=lambda e: (-scores.get(e.id, 0.0), e.date, e.time))
    ):
        ranked.append(RankedEvent(event=event, score=scores.get(event.id, 0.0), position=idx + 1))
    return ranked


async def _load_hits(db: Database, event_ids: Sequence[int]) -> set[int]:
    if not event_ids:
        return set()
    async with db.get_session() as session:
        result = await session.execute(
            select(VideoAnnounceEventHit.event_id).where(
                VideoAnnounceEventHit.event_id.in_(list(event_ids))
            )
        )
        rows = result.scalars().all()
    return set(rows)


async def _load_poster_ocr_texts(
    db: Database, event_ids: Sequence[int]
) -> tuple[dict[int, str], dict[int, str]]:
    if not event_ids:
        return {}, {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster)
            .where(EventPoster.event_id.in_(list(event_ids)))
            .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
        )
        posters = result.scalars().all()
    grouped_text: dict[int, list[str]] = defaultdict(list)
    titles: dict[int, str] = {}

    for poster in posters:
        text = (poster.ocr_text or "").strip()
        title = (poster.ocr_title or "").strip()
        if text:
            grouped_text[poster.event_id].append(text)
        if title and poster.event_id not in titles:
            titles[poster.event_id] = title

    excerpts: dict[int, str] = {}
    for event_id, texts in grouped_text.items():
        combined: list[str] = []
        remaining = POSTER_OCR_EXCERPT_LIMIT
        for text in texts:
            if remaining <= 0:
                break
            snippet = text[:remaining]
            combined.append(snippet)
            remaining -= len(snippet)
        excerpt = "\n\n".join(combined).strip()
        if excerpt:
            excerpts[event_id] = excerpt

    return excerpts, titles


def _apply_repeat_limit(
    candidates: Sequence[Event], *, limit: int, hits: set[int], promoted: set[int]
) -> list[Event]:
    seen: set[int] = set()
    selected: list[Event] = []
    repeated_allowed = max(0, math.floor(limit * 0.3))

    def _add(event: Event, *, allow_repeat: bool = False) -> None:
        if event.id in seen or len(selected) >= limit:
            return
        is_repeat = event.id in hits
        if is_repeat and not allow_repeat:
            current_repeats = sum(1 for ev in selected if ev.id in hits)
            if current_repeats >= repeated_allowed:
                return
        selected.append(event)
        seen.add(event.id)

    promoted_items = [e for e in candidates if e.id in promoted]
    fresh = [e for e in candidates if e.id not in hits and e.id not in promoted]
    repeats = [e for e in candidates if e.id in hits and e.id not in promoted]

    for ev in promoted_items:
        _add(ev, allow_repeat=True)
    for ev in fresh:
        _add(ev)
    for ev in repeats:
        _add(ev)

    if len(selected) < limit:
        for ev in candidates:
            if len(selected) >= limit:
                break
            _add(ev, allow_repeat=True)
    return selected


def _validate_intro_text(text: str | None) -> bool:
    if not text:
        return False
    # Check for date patterns: "24 ДЕКАБРЯ", "24-26 ДЕКАБРЯ", "С 29 ДЕКАБРЯ ПО 3 ЯНВАРЯ"
    # Months allowed: ЯНВАРЯ..ДЕКАБРЯ (full names)
    months = r"(ЯНВАРЯ|ФЕВРАЛЯ|МАРТА|АПРЕЛЯ|МАЯ|ИЮНЯ|ИЮЛЯ|АВГУСТА|СЕНТЯБРЯ|ОКТЯБРЯ|НОЯБРЯ|ДЕКАБРЯ)"

    # Pattern 1: Simple date "24 ДЕКАБРЯ"
    p1 = rf"\b\d{{1,2}}\s+{months}\b"
    # Pattern 2: Range "24-26 ДЕКАБРЯ" or "24–26 ДЕКАБРЯ"
    p2 = rf"\b\d{{1,2}}\s*[\-–]\s*\d{{1,2}}\s+{months}\b"
    # Pattern 3: Range across months "С 29 ДЕКАБРЯ ПО 3 ЯНВАРЯ"
    p3 = rf"\bС\s+\d{{1,2}}\s+{months}\s+ПО\s+\d{{1,2}}\s+{months}\b"
    # Also allow "29 ДЕКАБРЯ - 3 ЯНВАРЯ" just in case
    p4 = rf"\b\d{{1,2}}\s+{months}\s*[\-–]\s*\d{{1,2}}\s+{months}\b"

    combined = f"({p1})|({p2})|({p3})|({p4})"
    return bool(re.search(combined, text, re.IGNORECASE))

def _validate_about(text: str | None) -> bool:
    if not text:
        return False
    # Max 12 words
    # Clean first
    cleaned = text.strip()
    words = cleaned.split()
    if len(words) > 12:
        return False
    if not cleaned:
        return False
    return True


def _parse_llm_ranking(raw: str, known_ids: set[int]) -> tuple[bool, str | None, list[RankedChoice]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("video_announce: failed to parse ranking JSON")
        return (False, None, [])
    intro_text = None
    if isinstance(data, dict):
        raw_intro = data.get("intro_text")
        intro_text = raw_intro.strip() if isinstance(raw_intro, str) else None
    items = data.get("items") if isinstance(data, dict) else None
    parsed: list[RankedChoice] = []
    if not isinstance(items, list):
        return (False, intro_text, parsed)

    seen_ids = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        if not isinstance(event_id, int) or event_id not in known_ids or event_id in seen_ids:
            continue
        seen_ids.add(event_id)

        score_val = item.get("score")
        score = float(score_val) if isinstance(score_val, (int, float)) else None

        # Parse about
        about = item.get("about")
        if isinstance(about, str):
            about = about.strip()
        else:
            about = None

        parsed.append(
            RankedChoice(
                event_id=event_id,
                score=score if score is not None else 0.0,
                reason=item.get("reason"),
                about=about,
                selected=True,  # Explicitly selected by LLM
            )
        )
    return True, intro_text, parsed


def _describe_period(events: Sequence[Event]) -> str | None:
    dates: list[date] = []
    for ev in events:
        try:
            start = ev.date.split("..", 1)[0]
            dates.append(date.fromisoformat(start))
        except Exception:
            continue
    if not dates:
        return None
    start, end = min(dates), max(dates)
    if start == end:
        return start.isoformat()
    return f"{start.isoformat()}..{end.isoformat()}"


async def _store_llm_trace(
    db: Database,
    *,
    session_id: int | None,
    stage: str,
    model: str,
    request_json: str,
    response_json: str,
) -> None:
    trimmed_request = request_json
    trimmed_response = response_json
    if len(trimmed_request) > TRACE_MAX_LEN:
        logger.warning(
            "video_announce: request_json too long len=%d limit=%d, trimming",  # noqa: G004
            len(trimmed_request),
            TRACE_MAX_LEN,
        )
        trimmed_request = trimmed_request[:TRACE_MAX_LEN]
    if len(trimmed_response) > TRACE_MAX_LEN:
        logger.warning(
            "video_announce: response_json too long len=%d limit=%d, trimming",  # noqa: G004
            len(trimmed_response),
            TRACE_MAX_LEN,
        )
        trimmed_response = trimmed_response[:TRACE_MAX_LEN]
    try:
        async with db.get_session() as session:
            session.add(
                VideoAnnounceLLMTrace(
                    session_id=session_id,
                    stage=stage,
                    model=model,
                    request_json=trimmed_request,
                    response_json=trimmed_response,
                )
            )
            await session.commit()
    except Exception:
        logger.exception("video_announce: failed to store llm trace")


async def _rank_with_llm(
    db: Database,
    client: KaggleClient,
    events: Sequence[Event],
    *,
    promoted: set[int],
    mandatory_ids: set[int],
    schedule_map: dict[int, str] | None = None,
    occurrences_map: dict[int, list[dict[str, list[str]]]] | None = None,
    session_id: int | None = None,
    instruction: str | None = None,
    bot: Any | None = None,
    notify_chat_id: int | None = None,
    limit: int = 8,
) -> tuple[list[RankedEvent], str | None, bool]:
    if not events:
        return ([], None, True)

    # Pre-calculate base ranking
    base_ranked = _score_events(client, events)
    base_map = {r.event.id: r for r in base_ranked}

    event_ids = [e.id for e in events]
    poster_texts, poster_titles = await _load_poster_ocr_texts(db, event_ids)
    schedule_map = schedule_map or {}
    occurrences_map = occurrences_map or {}

    payload = []
    # Russian day of week names
    day_names_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    
    # Sort for consistent LLM input
    for ev in sorted(events, key=lambda e: (e.date, e.time, e.id)):
        # Calculate day of week from date
        try:
            event_date = date.fromisoformat(ev.date.split("..", 1)[0])
            day_of_week = day_names_ru[event_date.weekday()]
        except (ValueError, IndexError):
            day_of_week = None
        
        payload.append(
            {
                "event_id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "day_of_week": day_of_week,
                "time": ev.time,
                "schedule_text": schedule_map.get(ev.id),
                "occurrences": occurrences_map.get(ev.id),
                "city": ev.city,
                "location": ev.location_name,
                "topics": getattr(ev, "topics", []),
                "is_free": ev.is_free,
                "promoted": ev.id in promoted,
                "search_digest": getattr(ev, "search_digest", None),
                "poster_ocr_text": poster_texts.get(ev.id),
                "poster_ocr_title": poster_titles.get(ev.id),
            }
        )

    intro_text: str | None = None
    intro_valid: bool = True
    parsed: list[RankedChoice] = []
    parse_ok = False

    try:
        created_at = datetime.now(timezone.utc).isoformat()
        request_version = "selection_v2"
        system_prompt_text = selection_prompt()
        response_format = selection_response_format(max_items=limit)

        meta = {
            "source": "video_announce.selection",
            "count": len(payload),
            "system_prompt_id": "selection_prompt_v2",
            "period": _describe_period(events),
        }

        # User requested simplified input structure
        request_details = {
            "instruction": instruction,
            "candidates": payload,
            "meta": meta, # Optional but useful for debugging
        }

        request_json = json.dumps(request_details, ensure_ascii=False, indent=2)

        # Logging
        llm_input_preview = request_json[:200]
        logger.info(
            "video_announce: llm selection request items=%d promoted=%d limit=%d preview=%s",
            len(payload),
            len(promoted),
            limit,
            llm_input_preview,
        )

        if bot and notify_chat_id:
            try:
                filename = f"selection_request_{session_id or 'session'}.json"
                document = types.BufferedInputFile(
                    request_json.encode("utf-8"),
                    filename=filename,
                )
                await bot.send_document(
                    notify_chat_id,
                    document,
                    caption="Запрос на выборку и ранжирование (v2)",
                    disable_notification=True,
                )
            except Exception:
                logger.exception("video_announce: failed to send ranking request document")

        raw = await ask_4o(
            request_json,
            system_prompt=system_prompt_text,
            response_format=response_format,
            meta=meta,
            temperature=1.0,  # High temperature for creative variety in intro_text
        )

        if bot and notify_chat_id:
            try:
                filename = f"selection_response_{session_id or 'session'}.json"
                
                # Try to reformat JSON for pretty printing
                try:
                    response_obj = json.loads(raw)
                    pretty_response = json.dumps(response_obj, ensure_ascii=False, indent=2)
                except Exception:
                    pretty_response = raw
                
                document = types.BufferedInputFile(
                    pretty_response.encode("utf-8"),
                    filename=filename,
                )
                await bot.send_document(
                    notify_chat_id,
                    document,
                    caption="Ответ LLM на выборку (v2)",
                    disable_notification=True,
                )
            except Exception:
                logger.exception("video_announce: failed to send ranking response document")

        parse_ok, intro_text, parsed = _parse_llm_ranking(raw, {e.id for e in events})

        # Validation
        intro_valid = _validate_intro_text(intro_text)
        if not intro_valid:
            logger.warning("video_announce: invalid intro_text format: %r", intro_text)
            # Do NOT fix, pass as is or empty? Requirement says: "simply save as comes (or save empty - see recommendation... use flag)"
            # Recommendation: "intro_text_is_valid flag... do not change intro_text"
            pass

        await _store_llm_trace(
            db,
            session_id=session_id,
            stage="selection",
            model="gpt-4o",
            request_json=request_json,
            response_json=raw,
        )
    except Exception:
        logger.exception("video_announce: llm selection failed")
        parsed = []
        intro_valid = False

    # Construction of final ranked list
    final_ranked: list[RankedEvent] = []
    seen_ids: set[int] = set()

    # 1. Add LLM selections
    for choice in parsed:
        if choice.event_id in seen_ids:
            continue
        base = base_map.get(choice.event_id)
        if base:
            final_ranked.append(
                replace(
                    base,
                    score=choice.score if choice.score is not None else base.score,
                    reason=choice.reason,
                    about=choice.about, # Store raw about from LLM
                    selected=True,
                    position=len(final_ranked) + 1,
                    poster_ocr_text=poster_texts.get(base.event.id),
                    poster_ocr_title=poster_titles.get(base.event.id),
                )
            )
            seen_ids.add(choice.event_id)

    # Fallback if empty selection from LLM (Variant B)
    # We trigger fallback only if parsing FAILED. Valid empty selection (parse_ok=True, items=[]) is respected.
    if not final_ranked and not parse_ok:
        logger.warning("video_announce: LLM selection failed, using fallback top-N")
        fallback_candidates = sorted(base_ranked, key=lambda r: -r.score) # Sort by base score
        for base in fallback_candidates[:limit]:
             if base.event.id in seen_ids: continue
             final_ranked.append(
                 replace(
                     base,
                     selected=True, # Auto-select
                     reason="Fallback selection",
                     position=len(final_ranked) + 1,
                     poster_ocr_text=poster_texts.get(base.event.id),
                     poster_ocr_title=poster_titles.get(base.event.id),
                 )
             )
             seen_ids.add(base.event.id)

    # 2. Add remaining candidates
    # We use base_ranked order for remaining items
    for base in base_ranked:
        if base.event.id not in seen_ids:
            final_ranked.append(
                replace(
                    base,
                    selected=False,
                    position=len(final_ranked) + 1,
                    poster_ocr_text=poster_texts.get(base.event.id),
                    poster_ocr_title=poster_titles.get(base.event.id),
                )
            )
            seen_ids.add(base.event.id)

    return final_ranked, intro_text, intro_valid


async def prepare_session_items(
    db: Database,
    session_obj: VideoAnnounceSession,
    ranked: Iterable[RankedEvent],
    *,
    default_ready_ids: set[int],
) -> list[VideoAnnounceItem]:
    stored: list[VideoAnnounceItem] = []
    async with db.get_session() as session:
        for r in ranked:
            event = r.event
            item = VideoAnnounceItem(
                session_id=session_obj.id,
                event_id=event.id,
                position=r.position,
                status=(
                    VideoAnnounceItemStatus.READY
                    if event.id in default_ready_ids
                    else VideoAnnounceItemStatus.SKIPPED
                ),
                llm_score=r.score,
                llm_reason=r.reason,
                is_mandatory=r.mandatory,
                include_count=getattr(event, "video_include_count", 0) or 0,
            )

            # Validate about text logic
            # "Check about after normalization for non-empty and <= 12 words"
            # Note: _build_about calls normalize_about_with_fallback, which enforces limits.
            # But the requirement says "If after dedup/limits about becomes empty... simply save empty"
            # So _build_about needs to respect that.

            about_text = _build_about(
                about=r.about,
                event=event,
                ocr_text=r.poster_ocr_text,
                ocr_title=r.poster_ocr_title, # Pass title for dedup
            )

            description_text = (r.description or "").strip()
            if item.status == VideoAnnounceItemStatus.READY:
                item.final_about = about_text
                if description_text:
                    item.final_description = description_text
            session.add(item)
            stored.append(item)
        await session.commit()
    return stored


async def fill_missing_about(
    db: Database,
    session_id: int,
    items: Sequence[VideoAnnounceItem],
    events: dict[int, Event],
    *,
    bot: Any | None = None,
    notify_chat_id: int | None = None,
) -> dict[int, str]:
    """Request about field from LLM for items that don't have final_about.
    
    Returns a dict mapping event_id to generated about text.
    """
    missing = [item for item in items if not item.final_about]
    if not missing:
        return {}
    
    # Load OCR data for missing events
    missing_ids = [item.event_id for item in missing]
    poster_texts, poster_titles = await _load_poster_ocr_texts(db, missing_ids)
    
    # Build payload for LLM
    payload = []
    for item in missing:
        ev = events.get(item.event_id)
        if not ev:
            continue
        payload.append({
            "event_id": ev.id,
            "title": ev.title,
            "search_digest": getattr(ev, "search_digest", None),
            "ocr_title": poster_titles.get(ev.id),
            "poster_ocr_text": poster_texts.get(ev.id),
        })
    
    if not payload:
        return {}
    
    logger.info(
        "video_announce: requesting about for %d events without it",
        len(payload),
    )
    
    request_json = json.dumps({"events": payload}, ensure_ascii=False, indent=2)
    
    if bot and notify_chat_id:
        try:
            document = types.BufferedInputFile(
                request_json.encode("utf-8"),
                filename=f"about_fill_request_{session_id}.json",
            )
            await bot.send_document(
                notify_chat_id,
                document,
                caption="Запрос на генерацию about для недостающих событий",
                disable_notification=True,
            )
        except Exception:
            logger.exception("video_announce: failed to send about_fill request document")
    
    result: dict[int, str] = {}
    try:
        raw = await ask_4o(
            request_json,
            system_prompt=about_fill_prompt(),
            response_format=about_fill_response_format(len(payload)),
            meta={"source": "video_announce.fill_missing_about", "count": len(payload)},
        )
        
        if bot and notify_chat_id:
            try:
                response_obj = json.loads(raw)
                pretty_response = json.dumps(response_obj, ensure_ascii=False, indent=2)
                document = types.BufferedInputFile(
                    pretty_response.encode("utf-8"),
                    filename=f"about_fill_response_{session_id}.json",
                )
                await bot.send_document(
                    notify_chat_id,
                    document,
                    caption="Ответ LLM на генерацию about",
                    disable_notification=True,
                )
            except Exception:
                logger.exception("video_announce: failed to send about_fill response document")
        
        await _store_llm_trace(
            db,
            session_id=session_id,
            stage="about_fill",
            model="gpt-4o",
            request_json=request_json,
            response_json=raw,
        )
        
        data = json.loads(raw)
        items_data = data.get("items", [])
        known_ids = {ev_id for ev_id in events.keys()}
        
        for item_data in items_data:
            event_id = item_data.get("event_id")
            about = item_data.get("about")
            if event_id in known_ids and isinstance(about, str):
                # Normalize the about text
                ev = events.get(event_id)
                if ev:
                    normalized = normalize_about_with_fallback(
                        about.strip(),
                        ocr_text=poster_titles.get(event_id),
                    )
                    result[event_id] = normalized
        
        logger.info(
            "video_announce: generated about for %d/%d events",
            len(result),
            len(payload),
        )
    except Exception:
        logger.exception("video_announce: failed to fill missing about")
    
    return result


def build_payload(
    session: VideoAnnounceSession,
    ranked: list[RankedEvent],
    *,
    tz: timezone,
    items: Sequence[VideoAnnounceItem] | None = None,
    ) -> RenderPayload:
    events = [r.event for r in ranked]
    scores = {r.event.id: r.score for r in ranked}
    payload_items = list(items) if items is not None else [
        VideoAnnounceItem(
            session_id=session.id,
            event_id=r.event.id,
            position=r.position,
            status=VideoAnnounceItemStatus.READY,
        )
        for r in ranked
    ]
    return RenderPayload(
        session=session, items=payload_items, events=events, scores=scores
    )


def payload_as_json(payload: RenderPayload, tz: timezone) -> str:
    def _poster_urls(ev: Event) -> list[str]:
        urls = []
        for url in getattr(ev, "photo_urls", []) or []:
            if url.startswith("http:"):
                url = "https:" + url[5:]
            urls.append(url)
        return urls[:1]

    def _is_missing_time(t: str) -> bool:
        if not t:
            return True
        first = t.strip().split(" ")[0].split("-")[0].split("–")[0]
        return first in ("00:00", "0:00")

    def _format_scene_date(ev: Event) -> str:
        if isinstance(schedule_map, dict) and ev.id is not None:
            raw = schedule_map.get(str(ev.id))
            if raw is None:
                raw = schedule_map.get(ev.id)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        base_date = ev.date.split("..", 1)[0]
        try:
            pretty_date = format_day_pretty(date.fromisoformat(base_date))
        except Exception:
            pretty_date = base_date

        time_text = (ev.time or "").strip()
        if time_text and not _is_missing_time(time_text):
            short_time = time_text[:5] if ":" in time_text else time_text
            return f"{pretty_date} {short_time}"
        return pretty_date

    selection_params = (
        payload.session.selection_params
        if isinstance(payload.session.selection_params, dict)
        else {}
    )
    schedule_map = selection_params.get("dedup_schedule")

    # Simplified Intro Logic: Directly use stored value
    intro_text_override = (
        str(selection_params.get("intro_text_override") or "").strip()
        or str(selection_params.get("intro_text") or "").strip()
        or None
    )
    # Fallback to AFISHA if completely missing (e.g. invalid and empty)
    intro_str = intro_text_override or "АФИША"

    event_map = {ev.id: ev for ev in payload.events}
    
    # Use render_order if available, otherwise sort by position
    render_order = list(selection_params.get("render_order", []))
    if render_order:
        # Build order map: event_id -> order index
        order_map = {eid: idx for idx, eid in enumerate(render_order)}
        sorted_items = sorted(
            payload.items,
            key=lambda it: order_map.get(it.event_id, 999999)
        )
    else:
        sorted_items = sorted(payload.items, key=lambda it: it.position)
    
    scenes = []
    for item in sorted_items:
        ev = event_map.get(item.event_id)
        if not ev:
            continue
        location = ", ".join(part for part in [ev.city, ev.location_name] if part)

        # NOTE: final_about is already computed and stored in prepare_session_items via _build_about
        # But here we might re-compute if items are fresh?
        # Ideally we use item.final_about if present.
        if item.final_about is not None:
            about_text = item.final_about
        else:
             # Fallback calculation if not stored
             about_text = _build_about(
                about=item.final_about, # Should be None here if branch taken
                event=ev,
                ocr_text=item.poster_text,
                title=item.final_title or ev.title,
            )

        scene = {
            "about": about_text,
            "description": item.final_description or "",
            "date": _format_scene_date(ev),
            "location": location,
            "images": _poster_urls(ev),
            "is_free": bool(getattr(ev, "is_free", False)),
        }
        scenes.append(scene)

    # Extract date range and cities from events for intro
    # Use payload.events directly to get cities (events have .city attribute)
    event_cities: list[str] = []
    for ev in payload.events:
        city = getattr(ev, "city", None)
        if city and city not in event_cities:
            event_cities.append(city)
    
    logger.info("video_announce: extracted cities from events: %s", event_cities)

    # Build date string from events
    intro_date_str = ""
    min_date = None
    max_date = None
    if payload.events:
        dates_list: list[date] = []
        for ev in payload.events:
            try:
                d = date.fromisoformat(ev.date.split("..", 1)[0])
                dates_list.append(d)
            except Exception:
                pass
        if dates_list:
            min_date = min(dates_list)
            max_date = max(dates_list)
            if min_date == max_date:
                intro_date_str = format_day_pretty(min_date).upper()
            else:
                intro_date_str = f"{format_day_pretty(min_date)} - {format_day_pretty(max_date)}".upper()

    # Extract intro_pattern from selection_params
    intro_pattern = str(selection_params.get("intro_pattern") or "STICKER")

    intro_payload = {
        "count": len(scenes),
        "text": intro_str,
        "date": intro_date_str,
        "cities": event_cities[:4],  # Limit to 4 cities
        "pattern": intro_pattern,  # Add pattern for notebook
        "date_start": min_date.isoformat() if min_date else None,
        "date_end": max_date.isoformat() if max_date else None,
    }

    selection_meta = {}
    for key in ("mode", "test", "is_test"):
        if key in selection_params:
            selection_meta[key] = selection_params.get(key)

    obj = {
        "intro": intro_payload,
        "scenes": scenes,
    }
    if selection_meta:
        obj["selection_params"] = selection_meta
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _choose_default_ready(
    ranked: Sequence[RankedEvent],
    mandatory_ids: set[int],
    *,
    min_count: int,
    max_count: int,
) -> set[int]:
    ready: set[int] = set()

    # Filter only those marked as selected or mandatory
    candidates = [
        row for row in ranked if row.selected is True or row.event.id in mandatory_ids
    ]

    # First pass: Mandatory
    for row in candidates:
        if row.event.id in mandatory_ids:
            ready.add(row.event.id)

    # Second pass: Fill up to max_count with selected
    for row in candidates:
        if len(ready) >= max_count:
            break
        ready.add(row.event.id)

    return ready


async def build_selection(
    db: Database,
    ctx: SelectionContext,
    *,
    client: KaggleClient | None = None,
    session_id: int | None = None,
    candidates: Sequence[Event] | None = None,
    schedule_map: dict[int, str] | None = None,
    occurrences_map: dict[int, list[dict[str, list[str]]]] | None = None,
    bot: Any | None = None,
    notify_chat_id: int | None = None,
) -> SelectionBuildResult:
    client = client or KaggleClient()

    # Removed auto-expansion loop (Task Requirement: Отключить любое авторасширение дат/периодов)
    if candidates is not None:
        events = _filter_events_with_posters(list(candidates))
        schedule_map = schedule_map or {}
        occurrences_map = occurrences_map or {}
    else:
        events, schedule_map, occurrences_map = await fetch_candidates(db, ctx)
        schedule_map = schedule_map or {}
        occurrences_map = occurrences_map or {}

    random_ocr_texts: dict[int, str] | None = None
    random_ocr_titles: dict[int, str] | None = None
    if ctx.random_order and events:
        event_ids = [e.id for e in events if e.id is not None]
        ocr_texts, ocr_titles = await _load_poster_ocr_texts(db, event_ids)
        events = [e for e in events if e.id is not None and e.id in ocr_texts]
        random_ocr_texts = ocr_texts
        random_ocr_titles = ocr_titles

    async def _rank_events(
        current_events: Sequence[Event],
    ) -> tuple[list[RankedEvent], set[int], set[int], str | None, bool]:
        _log_event_selection_stats(current_events)
        filtered_events = list(current_events)
        mandatory_ids_local = {
            e.id
            for e in filtered_events
            if (getattr(e, "video_include_count", 0) or 0) > 0
        }
        promoted_local = set(ctx.promoted_event_ids or set()) | set(mandatory_ids_local)
        mandatory_ids_local = mandatory_ids_local | set(ctx.promoted_event_ids or set())
        hits_local = await _load_hits(db, [e.id for e in filtered_events])
        selected_local = (
            list(filtered_events)
            if candidates is not None
            else _apply_repeat_limit(
                filtered_events,
                limit=ctx.candidate_limit,
                hits=hits_local,
                promoted=promoted_local,
            )
        )
        if ctx.random_order:
            ranked_local: list[RankedEvent] = []
            selected_ids_local: set[int] = set()
            intro_text_local: str | None = None
            intro_valid_local: bool = True

            ocr_texts = random_ocr_texts or {}
            ocr_titles = random_ocr_titles or {}

            ocr_events = [e for e in selected_local if e.id is not None and e.id in ocr_texts]
            if not ocr_events:
                return [], mandatory_ids_local, set(), None, True

            mandatory_ocr = [e for e in ocr_events if e.id in mandatory_ids_local]
            others_ocr = [e for e in ocr_events if e.id not in mandatory_ids_local]

            others_with_title = [e for e in others_ocr if e.id in ocr_titles]
            others_without_title = [e for e in others_ocr if e.id not in ocr_titles]
            random.shuffle(others_with_title)
            random.shuffle(others_without_title)

            picked: list[Event] = []
            seen: set[int] = set()
            for ev in mandatory_ocr:
                if ev.id is None or ev.id in seen:
                    continue
                picked.append(ev)
                seen.add(ev.id)
                if len(picked) >= ctx.default_selected_max:
                    break

            if len(picked) < ctx.default_selected_max:
                for ev in others_with_title + others_without_title:
                    if ev.id is None or ev.id in seen:
                        continue
                    picked.append(ev)
                    seen.add(ev.id)
                    if len(picked) >= ctx.default_selected_max:
                        break

            picked_ids = {e.id for e in picked if e.id is not None}
            selected_ids_local = set(picked_ids)

            remaining = [e for e in ocr_events if e.id is not None and e.id not in picked_ids]
            remaining_sorted = sorted(remaining, key=_event_sort_key)
            ordered = picked + remaining_sorted

            scores = client.score(ordered)
            for pos, ev in enumerate(ordered, start=1):
                if ev.id is None:
                    continue
                is_selected = ev.id in picked_ids
                ranked_local.append(
                    RankedEvent(
                        event=ev,
                        score=scores.get(ev.id, 0.0),
                        position=pos,
                        reason="Random selection" if is_selected else None,
                        mandatory=ev.id in mandatory_ids_local,
                        selected=is_selected,
                        about=(ocr_titles.get(ev.id) or ev.title) if is_selected else None,
                        poster_ocr_text=ocr_texts.get(ev.id),
                        poster_ocr_title=ocr_titles.get(ev.id),
                    )
                )

            return (
                ranked_local,
                mandatory_ids_local,
                selected_ids_local,
                intro_text_local,
                intro_valid_local,
            )

        selected_ids_local = {ev.id for ev in selected_local}
        ranked_local, intro_text_local, intro_valid_local = await _rank_with_llm(
            db,
            client,
            selected_local,
            promoted=promoted_local,
            mandatory_ids=mandatory_ids_local,
            schedule_map=schedule_map,
            occurrences_map=occurrences_map,
            session_id=session_id,
            instruction=ctx.instruction,
            bot=bot,
            notify_chat_id=notify_chat_id,
            limit=ctx.default_selected_max,
        )

        return ranked_local, mandatory_ids_local, selected_ids_local, intro_text_local, intro_valid_local

    ranked, mandatory_ids, selected_ids, intro_text, intro_valid = await _rank_events(events)
    default_ready_ids = _choose_default_ready(
        ranked,
        mandatory_ids,
        min_count=ctx.default_selected_min,
        max_count=ctx.default_selected_max,
    )
    event_ids = {ev.id for ev in events if ev.id is not None}
    schedule_map = {eid: txt for eid, txt in schedule_map.items() if eid in event_ids}
    occurrences_map = {
        eid: occ for eid, occ in occurrences_map.items() if eid in event_ids
    }
    logger.info(
        "video_announce ranked events=%d candidates=%d ready=%d mandatory=%d top=%s",
        len(events),
        len(ranked),
        len(default_ready_ids),
        len(mandatory_ids),
        [r.event.id for r in ranked[:3]],
    )
    for row in ranked:
        status = "READY" if row.event.id in default_ready_ids else "CANDIDATE"
        # Only log first few or it spams
        if row.position <= 10:
             logger.info(
                "video_announce selection %s id=%s score=%.2f mandatory=%s reason=%s",
                status,
                row.event.id,
                row.score,
                row.mandatory,
                (row.reason or "")[0:200],
            )
    return SelectionBuildResult(
        ranked=ranked,
        default_ready_ids=default_ready_ids,
        mandatory_ids=mandatory_ids,
        candidates=events,
        selected_ids=default_ready_ids,
        schedule_map=schedule_map,
        occurrences_map=occurrences_map,
        intro_text=intro_text,
        intro_text_valid=intro_valid,
    )
