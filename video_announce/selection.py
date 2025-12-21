from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

from sqlalchemy import select

from db import Database
from main import ask_4o
from models import (
    Event,
    VideoAnnounceEventHit,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
)
from markup import format_day_pretty
from .kaggle_client import KaggleClient
from .prompts import RANKING_RESPONSE_FORMAT, ranking_prompt
from .types import RankedChoice, RankedEvent, RenderPayload, SelectionContext, VideoProfile

logger = logging.getLogger(__name__)


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
        return events[: ctx.limit]
    return combined[: max(ctx.limit * 2, ctx.limit)]


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


def _parse_llm_ranking(raw: str, known_ids: set[int]) -> list[RankedChoice]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("video_announce: failed to parse ranking JSON")
        return []
    items = data.get("items") if isinstance(data, dict) else None
    parsed: list[RankedChoice] = []
    if not isinstance(items, list):
        return parsed
    for item in items:
        if not isinstance(item, dict):
            continue
        event_id = item.get("event_id")
        if not isinstance(event_id, int) or event_id not in known_ids:
            continue
        score = float(item.get("score") or 0)
        parsed.append(
            RankedChoice(
                event_id=event_id,
                score=score,
                reason=item.get("reason"),
                use_ocr=item.get("use_ocr"),
                poster_source=item.get("poster_source"),
            )
        )
    return parsed


async def _rank_with_llm(
    client: KaggleClient, events: Sequence[Event], *, promoted: set[int]
) -> list[RankedEvent]:
    if not events:
        return []
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
                "poster_count": ev.photo_count,
                "promoted": ev.id in promoted,
            }
        )
    try:
        raw = await ask_4o(
            json.dumps(payload, ensure_ascii=False),
            system_prompt=ranking_prompt(),
            response_format=RANKING_RESPONSE_FORMAT,
            meta={"source": "video_announce.ranking"},
        )
        parsed = _parse_llm_ranking(raw, {e.id for e in events})
    except Exception:
        logger.exception("video_announce: llm ranking failed")
        parsed = []
    ranked: list[RankedEvent] = []
    if parsed:
        score_map = {row.event_id: row.score for row in parsed}
        ordering = {row.event_id: idx for idx, row in enumerate(parsed)}
        for ev in sorted(events, key=lambda e: ordering.get(e.id, len(events))):
            ranked.append(
                RankedEvent(
                    event=ev,
                    score=score_map.get(ev.id, 0.0),
                    position=len(ranked) + 1,
                )
            )
    else:
        ranked = _score_events(client, events)
    if len(ranked) < len(events):
        existing_ids = {r.event.id for r in ranked}
        remainder = [ev for ev in events if ev.id not in existing_ids]
        if remainder:
            fallback = _score_events(client, remainder)
            ranked.extend(
                RankedEvent(event=row.event, score=row.score, position=len(ranked) + idx + 1)
                for idx, row in enumerate(fallback)
            )
    return ranked


async def prepare_session_items(
    db: Database,
    session_obj: VideoAnnounceSession,
    ranked: Iterable[RankedEvent],
) -> list[VideoAnnounceItem]:
    stored: list[VideoAnnounceItem] = []
    async with db.get_session() as session:
        for r in ranked:
            item = VideoAnnounceItem(
                session_id=session_obj.id,
                event_id=r.event.id,
                position=r.position,
                status=VideoAnnounceItemStatus.READY,
            )
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

    event_map = {ev.id: ev for ev in payload.events}
    scenes = []
    for item in sorted(payload.items, key=lambda it: it.position):
        ev = event_map.get(item.event_id)
        if not ev:
            continue
        location = ", ".join(part for part in [ev.city, ev.location_name] if part)
        scene = {
            "title": item.final_title or ev.title,
            "description": item.final_description or ev.description,
            "date": ev.date.split("..", 1)[0],
            "location": location,
            "images": [_poster_name(item, ev)],
        }
        scenes.append(scene)

    obj = {
        "intro": {"count": len(scenes), "text": _intro_text(payload.events)},
        "scenes": scenes,
    }
    return json.dumps(obj, ensure_ascii=False, indent=2)


async def build_selection(
    db: Database,
    ctx: SelectionContext,
    *,
    client: KaggleClient | None = None,
) -> list[RankedEvent]:
    client = client or KaggleClient()
    events = await fetch_candidates(db, ctx)
    hits = await _load_hits(db, [e.id for e in events])
    promoted = set(ctx.promoted_event_ids or set())
    selected = _apply_repeat_limit(events, limit=ctx.limit, hits=hits, promoted=promoted)
    ranked = await _rank_with_llm(client, selected, promoted=promoted)
    logger.info(
        "video_announce ranked events=%d selected=%d top=%s",
        len(events),
        len(ranked),
        [r.event.id for r in ranked[:3]],
    )
    return ranked
