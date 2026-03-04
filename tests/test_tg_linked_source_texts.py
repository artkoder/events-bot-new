from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from models import Event
from source_parsing.telegram.handlers import process_telegram_results


@pytest.mark.asyncio
async def test_tg_linked_source_texts_scan_runs_smart_update_for_linked_sources(
    tmp_path, monkeypatch
) -> None:
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime.now(timezone.utc)
    event_date = (date.today() + timedelta(days=5)).isoformat()
    linked_url = "https://t.me/tatianaabar/384"
    linked_text = "Большой текст из связанного поста\nЕщё строка"

    public_html = f"""
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="tatianaabar/384">
    <div class="tgme_widget_message_text js-message_text" dir="auto">{linked_text.replace(chr(10), "<br>")}</div>
  </div>
</div>
"""

    async def fake_http_call(_label, _method, url, **_kwargs):
        if "https://t.me/s/tatianaabar/384" in str(url):
            return SimpleNamespace(status_code=200, content=public_html.encode("utf-8"))
        return SimpleNamespace(status_code=404, content=b"")

    import net

    monkeypatch.setattr(net, "http_call", fake_http_call)

    calls: list[object] = []
    created_event_id = 0

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        nonlocal created_event_id
        calls.append(candidate)
        if created_event_id <= 0:
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
            return SimpleNamespace(
                status="created",
                event_id=created_event_id,
                added_posters=0,
                reason=None,
                queue_notes=[],
            )
        return SimpleNamespace(
            status="merged",
            event_id=created_event_id,
            added_posters=0,
            reason=None,
            queue_notes=[],
        )

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    results = {
        "schema_version": 2,
        "run_id": "r-linked-text",
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
                "message_id": 6814,
                "message_date": now.isoformat(),
                "source_link": "https://t.me/meowafisha/6814",
                "text": "Анонс",
                "posters": [],
                "events": [
                    {
                        "title": "Татьяна танцует",
                        "date": event_date,
                        "time": "22:00",
                        "location_name": "Venue",
                        "source_text": "Короткий текст",
                        "linked_source_urls": [linked_url],
                    }
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    await process_telegram_results(results_path, db, bot=object(), progress_callback=None)

    assert len(calls) >= 2, "expected Smart Update to be called for linked source text"
    linked_candidate = calls[1]
    assert getattr(linked_candidate, "source_url", None) == linked_url
    assert "Большой текст из связанного поста" in str(getattr(linked_candidate, "source_text", ""))
    assert getattr(linked_candidate, "source_chat_username", None) == "tatianaabar"
    assert int(getattr(linked_candidate, "source_message_id", 0) or 0) == 384

