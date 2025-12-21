from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Iterable

from sqlalchemy import select

from db import Database
from models import (
    Event,
    VideoAnnounceEventHit,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
)
from markup import format_day_pretty
from .kaggle_client import KaggleClient
from .types import RankedEvent, RenderPayload, SelectionContext, VideoProfile

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
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date >= today.isoformat())
            .order_by(Event.date, Event.time, Event.id)
        )
        events = result.scalars().all()
    filtered: list[Event] = []
    for e in events:
        if e.video_include_count == 0:
            continue
        filtered.append(e)
        if len(filtered) >= ctx.limit:
            break
    if not filtered:
        return events[: ctx.limit]
    return filtered


def _score_events(client: KaggleClient, events: Iterable[Event]) -> list[RankedEvent]:
    scores = client.score(events)
    ranked: list[RankedEvent] = []
    for idx, event in enumerate(
        sorted(events, key=lambda e: (-scores.get(e.id, 0.0), e.date, e.time))
    ):
        ranked.append(RankedEvent(event=event, score=scores.get(event.id, 0.0), position=idx + 1))
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
            hit = VideoAnnounceEventHit(session_id=session_obj.id, event_id=r.event.id)
            session.add(item)
            session.add(hit)
            stored.append(item)
        await session.commit()
    return stored


def build_payload(
    session: VideoAnnounceSession,
    ranked: list[RankedEvent],
    *,
    tz: timezone,
) -> RenderPayload:
    items = [
        VideoAnnounceItem(
            session_id=session.id,
            event_id=r.event.id,
            position=r.position,
            status=VideoAnnounceItemStatus.READY,
        )
        for r in ranked
    ]
    events = [r.event for r in ranked]
    scores = {r.event.id: r.score for r in ranked}
    return RenderPayload(session=session, items=items, events=events, scores=scores)


def payload_as_json(payload: RenderPayload, tz: timezone) -> str:
    def _format_event(ev: Event) -> dict:
        dt = ev.date.split("..", 1)[0]
        d = date.fromisoformat(dt)
        pretty = format_day_pretty(d)
        return {
            "id": ev.id,
            "title": ev.title,
            "emoji": ev.emoji,
            "time": ev.time,
            "date": ev.date,
            "pretty_date": pretty,
            "location": ev.location_name,
            "city": ev.city,
            "topics": getattr(ev, "topics", []),
            "is_free": ev.is_free,
            "score": payload.scores.get(ev.id, 0.0),
            "video_include_count": ev.video_include_count,
        }

    obj = {
        "session_id": payload.session.id,
        "prepared_at": payload.prepared_at.isoformat(),
        "items": [
            {"event_id": it.event_id, "position": it.position}
            for it in payload.items
        ],
        "events": [_format_event(e) for e in payload.events],
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
    ranked = _score_events(client, events)
    logger.info(
        "video_announce ranked events=%d top=%s",
        len(ranked),
        [r.event.id for r in ranked[:3]],
    )
    return ranked
