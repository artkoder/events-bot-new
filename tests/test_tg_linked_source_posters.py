from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

import main
from models import Event, EventSource
from source_parsing.telegram.handlers import process_telegram_results


@pytest.mark.asyncio
async def test_tg_linked_source_posters_enrich_candidate_and_attach_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TG_MONITORING_LINKED_SOURCES_TEXT", "0")
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime.now(timezone.utc)
    event_date = (date.today() + timedelta(days=5)).isoformat()
    linked_url = "https://t.me/urbanistikastudy/219"
    linked_poster_url = "https://files.catbox.moe/urban-poster.webp"

    results = {
        "schema_version": 2,
        "run_id": "r-linked-posters",
        "generated_at": now.isoformat(),
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 1,
        },
        "messages": [
            {
                "source_username": "meowafisha",
                "source_title": "Meow",
                "message_id": 6777,
                "message_date": now.isoformat(),
                "source_link": "https://t.me/meowafisha/6777",
                "text": "Пост без локальной афиши",
                "posters": [],
                "events": [
                    {
                        "title": "Школа городской модерации",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Калининград",
                        "source_text": "Источник события",
                        "linked_source_urls": [linked_url],
                    }
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    fallback_calls: list[tuple[str, int]] = []
    captured_posters: list[str] = []
    created_event_id = 0

    async def fake_link_fallback(*, username: str, message_id: int, limit: int = 3):
        fallback_calls.append((username, int(message_id)))
        if username == "urbanistikastudy" and int(message_id) == 219:
            return [
                SimpleNamespace(
                    catbox_url=linked_poster_url,
                    supabase_url=None,
                    supabase_path=None,
                    sha256="linked-poster-sha",
                    phash=None,
                    ocr_text=None,
                    ocr_title=None,
                )
            ]
        return []

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        nonlocal created_event_id
        captured_posters[:] = [
            str(getattr(p, "supabase_url", None) or getattr(p, "catbox_url", None) or "").strip()
            for p in (candidate.posters or [])
        ]
        async with db_obj.get_session() as session:
            ev = Event(
                title=candidate.title or "",
                date=candidate.date,
                time=candidate.time,
                location_name=candidate.location_name or "",
                city=candidate.city or "Калининград",
                description="desc",
                source_text=candidate.source_text or "",
                source_post_url=candidate.source_url,
                source_message_id=candidate.source_message_id,
            )
            session.add(ev)
            await session.commit()
            await session.refresh(ev)
            created_event_id = int(ev.id or 0)
        return SimpleNamespace(status="created", event_id=created_event_id, added_posters=len(captured_posters), reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "_fallback_fetch_posters_from_public_tg_page", fake_link_fallback)
    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    await process_telegram_results(results_path, db, bot=None, progress_callback=None)

    assert ("urbanistikastudy", 219) in fallback_calls
    assert linked_poster_url in captured_posters
    assert created_event_id > 0

    async with db.get_session() as session:
        rows = (
            await session.execute(
                select(EventSource.source_url).where(EventSource.event_id == int(created_event_id))
            )
        ).all()
    source_urls = {str(r[0]) for r in rows if r and r[0]}
    assert linked_url in source_urls
