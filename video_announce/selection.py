from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from aiogram import types
from sqlalchemy import select

from db import Database
from main import ask_4o, format_day_pretty
from main import get_source_page_text
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
from .prompts import selection_prompt, selection_response_format
from .types import (
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

def _build_about(
    *,
    about: str | None,
    event: Event,
    ocr_text: str | None = None,
    title: str | None = None,
) -> str:
    return normalize_about_with_fallback(
        about,
        title=title or getattr(event, "title", None),
        ocr_text=ocr_text,
        fallback_parts=(getattr(event, "location_name", None), getattr(event, "city", None)),
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


async def fetch_candidates(db: Database, ctx: SelectionContext) -> list[Event]:
    today = ctx.target_date or datetime.now(ctx.tz).date()
    primary_end = today + timedelta(days=max(ctx.primary_window_days, 0))
    fallback_end = today + timedelta(days=max(ctx.fallback_window_days, ctx.primary_window_days))
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date >= today.isoformat())
            .where(Event.date <= fallback_end.isoformat())
            .order_by(Event.date, Event.time, Event.id)
        )
        events = result.scalars().all()
    events = _filter_events_with_posters(events)
    promoted_ids = set(ctx.promoted_event_ids or set())
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
    if not combined:
        return events[: ctx.candidate_limit]
    return combined[: max(ctx.candidate_limit * 2, ctx.candidate_limit)]


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


async def _fetch_telegraph_text(ev: Event) -> str | None:
    path = (ev.telegraph_path or "").strip()
    url = (ev.telegraph_url or "").strip()
    resolved_path = ""
    if path:
        resolved_path = path.lstrip("/")
    elif url:
        parsed = urlparse(url)
        resolved_path = parsed.path.lstrip("/")
    if not resolved_path:
        logger.warning(
            "telegraph_event_fetch no_path",
            extra={
                "event_id": ev.id,
                "telegraph_url": url,
                "telegraph_path": path,
            },
        )
        return None
    logger.info(
        "telegraph_event_fetch start",
        extra={
            "event_id": ev.id,
            "telegraph_url": url,
            "telegraph_path": path,
            "resolved_path": resolved_path,
        },
    )
    try:
        text = await get_source_page_text(resolved_path)
    except Exception:
        logger.exception("video_announce: failed to fetch telegraph text event=%s", ev.id)
        return None
    cleaned = (text or "").strip()
    if not cleaned:
        logger.warning(
            "telegraph_text_empty",
            extra={
                "event_id": ev.id,
                "telegraph_url": url,
                "telegraph_path": path,
                "resolved_path": resolved_path,
                "text_len": 0,
            },
        )
        return None
    return cleaned


async def _load_poster_ocr_texts(
    db: Database, event_ids: Sequence[int]
) -> dict[int, str]:
    if not event_ids:
        return {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster)
            .where(EventPoster.event_id.in_(list(event_ids)))
            .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
        )
        posters = result.scalars().all()
    grouped: dict[int, list[str]] = defaultdict(list)
    for poster in posters:
        text = (poster.ocr_text or "").strip()
        if text:
            grouped[poster.event_id].append(text)
    excerpts: dict[int, str] = {}
    for event_id, texts in grouped.items():
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
    return excerpts


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


def _parse_llm_ranking(raw: str, known_ids: set[int]) -> tuple[str | None, list[RankedChoice]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("video_announce: failed to parse ranking JSON")
        return (None, [])
    intro_text = None
    if isinstance(data, dict):
        raw_intro = data.get("intro_text")
        intro_text = raw_intro.strip() if isinstance(raw_intro, str) else None
    items = data.get("items") if isinstance(data, dict) else None
    parsed: list[RankedChoice] = []
    if not isinstance(items, list):
        return intro_text, parsed
    for item in items:
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        if not isinstance(event_id, int) or event_id not in known_ids:
            continue
        score = float(item.get("score") or 0)
        selected = item.get("selected")
        if not isinstance(selected, bool):
            selected = None
        about = item.get("about") if isinstance(item.get("about"), str) else None
        description_raw = item.get("description") if "description" in item else None
        parsed.append(
            RankedChoice(
                event_id=event_id,
                score=score,
                reason=item.get("reason"),
                selected=selected,
                selected_reason=item.get("selected_reason"),
                about=about,
                description=str(description_raw).strip() if description_raw else None,
            )
        )
    return intro_text, parsed


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
    session_id: int | None = None,
    instruction: str | None = None,
    bot: Any | None = None,
    notify_chat_id: int | None = None,
) -> tuple[list[RankedEvent], str | None]:
    if not events:
        return ([], None)
    event_ids = [e.id for e in events]
    poster_texts = await _load_poster_ocr_texts(db, event_ids)
    telegraph_tasks: dict[int, asyncio.Task[str | None]] = {}
    for ev in events:
        if ev.telegraph_url or ev.telegraph_path:
            telegraph_tasks[ev.id] = asyncio.create_task(_fetch_telegraph_text(ev))
    telegraph_full_texts: dict[int, str] = {}
    if telegraph_tasks:
        results = await asyncio.gather(*telegraph_tasks.values())
        for event_id, text in zip(telegraph_tasks.keys(), results):
            if text:
                telegraph_full_texts[event_id] = text
    payload = []
    for ev in sorted(events, key=lambda e: (e.date, e.time, e.id)):
        payload.append(
            {
                "event_id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "time": ev.time,
                "city": ev.city,
                "location": ev.location_name,
                "topics": getattr(ev, "topics", []),
                "is_free": ev.is_free,
                "include_count": getattr(ev, "video_include_count", 0) or 0,
                "promoted": ev.id in promoted,
                "full_event_description": telegraph_full_texts.get(ev.id),
                "poster_ocr_text": poster_texts.get(ev.id),
            }
        )
    intro_text: str | None = None
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        request_version = "selection_v1"
        system_prompt_text = selection_prompt()
        response_format = selection_response_format(len(payload))
        meta = {
            "source": "video_announce.selection",
            "count": len(payload),
            "system_prompt_id": "selection_prompt_v1",
            "system_prompt_name": "video_announce_selection",
            "period": _describe_period(events),
            "candidate_ids": event_ids,
        }
        request_details = {
            "request_version": request_version,
            "created_at": created_at,
            "system_prompt": system_prompt_text,
            "user_instruction": instruction,
            "user_message": instruction,
            "response_format": response_format,
            "meta": meta,
            "candidates": payload,
        }
        request_json = json.dumps(request_details, ensure_ascii=False)
        llm_input_digest = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        request_details["meta"]["llm_input_digest"] = llm_input_digest
        request_details["meta"]["llm_input_preview"] = request_json[:200]
        request_json = json.dumps(request_details, ensure_ascii=False, indent=2)
        preview = json.dumps(payload[:3], ensure_ascii=False)
        logger.info(
            "video_announce: llm selection request items=%d promoted=%d preview=%s instruction=%s",
            len(payload),
            len(promoted),
            preview,
            bool(instruction),
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
                    caption="Запрос на выборку и ранжирование",
                    disable_notification=True,
                )
            except Exception:
                logger.exception("video_announce: failed to send ranking request document")
        raw = await ask_4o(
            request_json,
            system_prompt=system_prompt_text,
            response_format=response_format,
            meta=meta,
        )
        if bot and notify_chat_id:
            try:
                filename = f"selection_response_{session_id or 'session'}.json"
                document = types.BufferedInputFile(
                    raw.encode("utf-8"),
                    filename=filename,
                )
                await bot.send_document(
                    notify_chat_id,
                    document,
                    caption="Ответ LLM на выборку и ранжирование",
                    disable_notification=True,
                )
            except Exception:
                logger.exception("video_announce: failed to send ranking response document")

        intro_text, parsed = _parse_llm_ranking(raw, {e.id for e in events})
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
    ranked: list[RankedEvent] = []
    if parsed:
        event_map = {ev.id: ev for ev in events}
        parsed_ids = {row.event_id for row in parsed}
        missing = [ev.id for ev in events if ev.id not in parsed_ids]
        if missing:
            logger.warning(
                "video_announce: ranking_incomplete returned=%d expected=%d missing=%s",
                len(parsed),
                len(events),
                missing,
            )
        for idx, row in enumerate(parsed):
            event = event_map.get(row.event_id)
            if not event:
                continue
            ranked.append(
                RankedEvent(
                    event=event,
                    score=row.score,
                    position=idx + 1,
                    reason=row.reason,
                    mandatory=event.id in mandatory_ids,
                    selected=row.selected,
                    selected_reason=row.selected_reason,
                    about=row.about,
                    description=row.description,
                    poster_ocr_text=poster_texts.get(event.id),
                )
            )
    else:
        ranked = [
            RankedEvent(
                event=row.event,
                score=row.score,
                position=idx + 1,
                mandatory=row.event.id in mandatory_ids,
                poster_ocr_text=poster_texts.get(row.event.id),
            )
            for idx, row in enumerate(_score_events(client, events))
        ]
    return ranked, intro_text


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
            about_text = _build_about(
                about=r.about,
                event=event,
                ocr_text=r.poster_ocr_text,
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
    def _poster_name(item: VideoAnnounceItem, ev: Event) -> str:
        ext = ".jpg"
        for url in getattr(ev, "photo_urls", []) or []:
            low_path = url.lower().split("?", 1)[0]
            for candidate in (".png", ".jpg", ".jpeg", ".webp"):
                if low_path.endswith(candidate):
                    ext = candidate
                    break
            if ext != ".jpg":
                break
        return f"{item.position}{ext}"

    def _intro_text(events: Sequence[Event]) -> str:
        dates: list[date] = []
        for ev in events:
            try:
                dt = ev.date.split("..", 1)[0]
                dates.append(date.fromisoformat(dt))
            except ValueError:
                continue
        if not dates:
            return "Видео афиша"
        start = min(dates)
        end = max(dates)
        pretty_start = format_day_pretty(start)
        if start == end:
            return f"Афиша на {pretty_start}"
        pretty_end = format_day_pretty(end)
        return f"Афиша {pretty_start} – {pretty_end}"

    def _format_scene_date(ev: Event) -> str:
        base_date = ev.date.split("..", 1)[0]
        try:
            pretty_date = format_day_pretty(date.fromisoformat(base_date))
        except Exception:
            pretty_date = base_date
        time_text = (ev.time or "").strip()
        if time_text:
            short_time = time_text[:5] if ":" in time_text else time_text
            return f"{pretty_date} {short_time}"
        return pretty_date

    selection_params = (
        payload.session.selection_params
        if isinstance(payload.session.selection_params, dict)
        else {}
    )
    intro_text_override = None
    if isinstance(selection_params, dict):
        intro_text_override = (
            str(selection_params.get("intro_text_override") or "").strip()
            or str(selection_params.get("intro_text") or "").strip()
            or None
        )

    event_map = {ev.id: ev for ev in payload.events}
    scenes = []
    for item in sorted(payload.items, key=lambda it: it.position):
        ev = event_map.get(item.event_id)
        if not ev:
            continue
        location = ", ".join(part for part in [ev.city, ev.location_name] if part)
        about_text = _build_about(
            about=item.final_about,
            event=ev,
            ocr_text=item.poster_text,
            title=item.final_title or ev.title,
        )
        scene = {
            "about": about_text,
            "description": item.final_description or "",
            "date": _format_scene_date(ev),
            "location": location,
            "images": [_poster_name(item, ev)],
            "is_free": bool(getattr(ev, "is_free", False)),
        }
        scenes.append(scene)

    obj = {
        "intro": {
            "count": len(scenes),
            "text": intro_text_override or _intro_text(payload.events),
        },
        "scenes": scenes,
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _choose_default_ready(
    ranked: Sequence[RankedEvent],
    mandatory_ids: set[int],
    *,
    min_count: int,
    max_count: int,
) -> set[int]:
    ready: set[int] = set()
    selected_rows = [
        row for row in ranked if row.selected is True or row.event.id in mandatory_ids
    ]

    # Always include mandatory/promoted events even if that exceeds the max target
    for row in selected_rows:
        if row.event.id in mandatory_ids:
            ready.add(row.event.id)

    target_count = min(max_count, len(selected_rows))
    if len(selected_rows) >= min_count:
        target_count = max(target_count, min_count)

    for row in selected_rows:
        if len(ready) >= target_count:
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
    bot: Any | None = None,
    notify_chat_id: int | None = None,
) -> SelectionBuildResult:
    client = client or KaggleClient()
    expanded_ctx = ctx
    events = (
        _filter_events_with_posters(list(candidates))
        if candidates is not None
        else await fetch_candidates(db, expanded_ctx)
    )

    while candidates is None and len(events) < ctx.default_selected_min:
        next_ctx = replace(
            expanded_ctx,
            fallback_window_days=expanded_ctx.fallback_window_days + 3,
        )
        more_events = await fetch_candidates(db, next_ctx)
        if len(more_events) <= len(events):
            break
        logger.info(
            "video_announce: expanding selection window fallback_days=%d -> %d due to low candidate count=%d",
            expanded_ctx.fallback_window_days,
            next_ctx.fallback_window_days,
            len(events),
        )
        events = more_events
        expanded_ctx = next_ctx

    async def _rank_events(
        current_events: Sequence[Event],
    ) -> tuple[list[RankedEvent], set[int], set[int], str | None]:
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
        selected_ids_local = {ev.id for ev in selected_local}
        ranked_local, intro_text_local = await _rank_with_llm(
            db,
            client,
            selected_local,
            promoted=promoted_local,
            mandatory_ids=mandatory_ids_local,
            session_id=session_id,
            instruction=ctx.instruction,
            bot=bot,
            notify_chat_id=notify_chat_id,
        )
        return ranked_local, mandatory_ids_local, selected_ids_local, intro_text_local

    ranked, mandatory_ids, selected_ids, intro_text = await _rank_events(events)
    default_ready_ids = _choose_default_ready(
        ranked,
        mandatory_ids,
        min_count=ctx.default_selected_min,
        max_count=ctx.default_selected_max,
    )
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
        intro_text=intro_text,
    )
