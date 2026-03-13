from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from models import Event
from smart_event_update import PosterCandidate
from source_parsing.telegram.handlers import process_telegram_results


@pytest.mark.asyncio
async def test_tg_multi_event_public_page_poster_fallback_keeps_image_only_posters(
    tmp_path,
    monkeypatch,
) -> None:
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime.now(timezone.utc)
    first_date = (date.today() + timedelta(days=3)).isoformat()
    second_date = (date.today() + timedelta(days=10)).isoformat()

    results = {
        "schema_version": 2,
        "run_id": "r-multi-event-poster-fallback",
        "generated_at": now.isoformat(),
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 2,
        },
        "messages": [
            {
                "source_username": "kaliningradartmuseum",
                "source_title": "Art Museum",
                "message_id": 7747,
                "message_date": now.isoformat(),
                "source_link": "https://t.me/kaliningradartmuseum/7747",
                "text": (
                    "Лекторий о красном цвете\n\n"
                    "14.03 17:00 Красный цвет в русской традиции\n"
                    "19.03 18:30 Психология цвета в маркетинге"
                ),
                "posters": [],
                "events": [
                    {
                        "title": "Красный цвет в русской традиции",
                        "date": first_date,
                        "time": "17:00",
                        "location_name": "Калининградский музей изобразительных искусств",
                    },
                    {
                        "title": "Психология цвета в маркетинге",
                        "date": second_date,
                        "time": "18:30",
                        "location_name": "Калининградский музей изобразительных искусств",
                    },
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    fallback_calls: list[tuple[str, int, int, bool]] = []
    captured_posters: dict[str, list[str]] = {}

    async def fake_fallback(*, username: str, message_id: int, limit: int = 3, need_ocr: bool = False):
        fallback_calls.append((username, int(message_id), int(limit), bool(need_ocr)))
        return [
            PosterCandidate(
                catbox_url="https://files.catbox.moe/red-photo.jpg",
                sha256="red-photo",
                ocr_text="",
                ocr_title="",
            )
        ]

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        captured_posters[str(candidate.title or "")] = [
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
                photo_urls=list(captured_posters[str(candidate.title or "")]),
                photo_count=len(captured_posters[str(candidate.title or "")]),
            )
            session.add(ev)
            await session.commit()
            await session.refresh(ev)
            event_id = int(ev.id or 0)
        return SimpleNamespace(
            status="created",
            event_id=event_id,
            added_posters=len(captured_posters[str(candidate.title or "")]),
            reason=None,
        )

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "_fallback_fetch_posters_from_public_tg_page", fake_fallback)
    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    await process_telegram_results(results_path, db, bot=None, progress_callback=None)

    assert fallback_calls == [("kaliningradartmuseum", 7747, 5, True)]
    assert captured_posters["Красный цвет в русской традиции"] == [
        "https://files.catbox.moe/red-photo.jpg"
    ]
    assert captured_posters["Психология цвета в маркетинге"] == [
        "https://files.catbox.moe/red-photo.jpg"
    ]

