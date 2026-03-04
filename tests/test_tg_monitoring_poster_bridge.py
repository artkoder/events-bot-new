from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

import main
from models import Event, TelegramScannedMessage, TelegramSource
from source_parsing.telegram.handlers import process_telegram_results


@pytest.mark.asyncio
async def test_tg_poster_bridge_skips_unrelated_posters_by_ocr(tmp_path, monkeypatch) -> None:
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    started = datetime.now(timezone.utc)
    event_date = (date.today() + timedelta(days=7)).isoformat()

    results = {
        "schema_version": 2,
        "run_id": "r-bridge-ocr-skip",
        "generated_at": started.isoformat(),
        "stats": {
            "sources_total": 1,
            "messages_scanned": 2,
            "messages_with_events": 1,
            "events_extracted": 1,
        },
        "messages": [
            {
                "source_username": "sobor39",
                "source_title": "Some Channel",
                "message_id": 100,
                "message_date": started.isoformat(),
                "source_link": "https://t.me/sobor39/100",
                "text": "Пост про событие",
                "posters": [],
                "events": [
                    {
                        "title": "Сказка о солдате",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                    }
                ],
            },
            {
                "source_username": "sobor39",
                "source_title": "Some Channel",
                "message_id": 101,
                "message_date": (started + timedelta(seconds=90)).isoformat(),
                "source_link": "https://t.me/sobor39/101",
                "text": "",
                "posters": [
                    {
                        "sha256": "p1",
                        "catbox_url": "https://files.catbox.moe/p1.jpg",
                        "ocr_text": "Modern Music Center 10 марта 19:00",
                    }
                ],
                "events": [],
            },
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    calls: list[object] = []

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        calls.append(candidate)
        if len(calls) == 1:
            async with db_obj.get_session() as session:
                ev = Event(
                    title=candidate.title or "",
                    date=candidate.date,
                    time=candidate.time,
                    location_name=candidate.location_name or "",
                    city=candidate.city or "Калининград",
                    description="desc",
                    source_text=candidate.source_text or "",
                )
                session.add(ev)
                await session.commit()
                await session.refresh(ev)
                eid = int(ev.id or 0)
            return SimpleNamespace(status="created", event_id=eid, added_posters=0, reason=None)
        raise AssertionError("poster bridge should not call Smart Update for unrelated posters")

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    await process_telegram_results(results_path, db, bot=None, progress_callback=None)

    assert len(calls) == 1
    async with db.get_session() as session:
        evs = (await session.execute(select(Event))).scalars().all()
    assert len(evs) == 1
    assert int(evs[0].photo_count or 0) == 0


@pytest.mark.asyncio
async def test_tg_poster_bridge_db_fallback_is_gated_by_caption_and_time_delta(tmp_path, monkeypatch) -> None:
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime.now(timezone.utc)
    event_date = (date.today() + timedelta(days=3)).isoformat()

    async with db.get_session() as session:
        src = TelegramSource(username="testchan", enabled=True)
        session.add(src)
        await session.commit()
        await session.refresh(src)

        prev_mid = 200
        prev_link = f"https://t.me/testchan/{prev_mid}"
        ev = Event(
            title="Target",
            date=event_date,
            time="19:00",
            location_name="Venue",
            city="Калининград",
            description="desc",
            source_text="src",
            source_post_url=prev_link,
            source_message_id=prev_mid,
        )
        session.add(ev)
        session.add(
            TelegramScannedMessage(
                source_id=int(src.id or 0),
                message_id=prev_mid,
                message_date=now,
                status="done",
                events_extracted=1,
                events_imported=1,
            )
        )
        await session.commit()

    async def fail_smart_event_update(*_args, **_kwargs):
        raise AssertionError("poster bridge DB fallback should not attach in this test")

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fail_smart_event_update)

    # Case 1: long caption should disable DB fallback bridging.
    results_long = {
        "schema_version": 2,
        "run_id": "r-bridge-db-long",
        "generated_at": now.isoformat(),
        "stats": {"sources_total": 1, "messages_scanned": 1, "messages_with_events": 0, "events_extracted": 0},
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Test",
                "message_id": prev_mid + 1,
                "message_date": (now + timedelta(seconds=60)).isoformat(),
                "source_link": f"https://t.me/testchan/{prev_mid + 1}",
                "text": "Это длинная подпись к пересланной афише, которая не должна запускать bridge fallback.",
                "posters": [{"sha256": "p2", "catbox_url": "https://files.catbox.moe/p2.jpg", "ocr_text": "Target"}],
                "events": [],
            }
        ],
    }
    p1 = Path(tmp_path) / "telegram_long.json"
    p1.write_text(json.dumps(results_long, ensure_ascii=False), encoding="utf-8")
    await process_telegram_results(p1, db, bot=None, progress_callback=None)

    # Case 2: large time delta should disable DB fallback bridging.
    results_delta = {
        "schema_version": 2,
        "run_id": "r-bridge-db-delta",
        "generated_at": now.isoformat(),
        "stats": {"sources_total": 1, "messages_scanned": 1, "messages_with_events": 0, "events_extracted": 0},
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Test",
                "message_id": prev_mid + 1,
                "message_date": (now + timedelta(hours=1)).isoformat(),
                "source_link": f"https://t.me/testchan/{prev_mid + 1}",
                "text": "",
                "posters": [{"sha256": "p3", "catbox_url": "https://files.catbox.moe/p3.jpg", "ocr_text": "Target"}],
                "events": [],
            }
        ],
    }
    p2 = Path(tmp_path) / "telegram_delta.json"
    p2.write_text(json.dumps(results_delta, ensure_ascii=False), encoding="utf-8")
    await process_telegram_results(p2, db, bot=None, progress_callback=None)

