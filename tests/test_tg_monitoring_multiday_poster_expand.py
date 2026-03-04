from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import main


@pytest.mark.asyncio
async def test_tg_monitoring_expands_multiday_poster_into_two_events(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc).isoformat()

    # Kaggle can occasionally collapse multi-day posters into a single extracted `date`.
    # Poster OCR contains two concrete "DD <month> ... HH:MM" pairs, so the importer expands.
    results = {
        "run_id": "r1",
        "generated_at": generated_at,
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 1,
        },
        "messages": [
            {
                "source_username": "yantarholl",
                "source_title": "Янтарь-холл",
                "message_id": 4224,
                "message_date": generated_at,
                "source_link": "https://t.me/yantarholl/4224",
                "text": "12 и 13 июня в Янтарь-холл. Шоу «Великие иллюзии».",
                "posters": [
                    {
                        "sha256": "p1",
                        "catbox_url": "https://files.catbox.moe/x.webp",
                        "ocr_title": "Братья Сафроновы — Великие иллюзии",
                        "ocr_text": "12 июня начало в 19:00\n13 июня начало в 15:00\nВеликие иллюзии",
                    }
                ],
                "events": [
                    {
                        "title": "Великие иллюзии",
                        "date": "2026-06-12",
                        "time": "19:00",
                        "location_name": "Янтарь-холл",
                        "raw_excerpt": "Шоу иллюзионистов",
                    }
                ],
            }
        ],
    }

    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    created: list[tuple[str, str]] = []

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        from models import Event

        async with db_obj.get_session() as session:
            ev = Event(
                title=candidate.title or "",
                description="desc",
                source_text=candidate.source_text or "",
                date=candidate.date,
                time=candidate.time,
                location_name=candidate.location_name or "",
                city=candidate.city or "Калининград",
                source_post_url=candidate.source_url,
                source_message_id=candidate.source_message_id,
                telegraph_path=f"e-{candidate.date}",
                telegraph_url=f"https://telegra.ph/e-{candidate.date}",
            )
            session.add(ev)
            await session.commit()
            await session.refresh(ev)
            created.append((candidate.date or "", candidate.time or ""))
            return SimpleNamespace(status="created", event_id=int(ev.id), added_posters=0, reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    report = await tg_handlers.process_telegram_results(results_path, db, bot=None)

    assert len(created) == 2
    assert ("2026-06-12", "19:00") in created
    assert ("2026-06-13", "15:00") in created

    # Report should reflect expanded extraction/import.
    assert sum(1 for _ in report.created_events) == 2

