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
async def test_tg_monitoring_fetches_full_text_for_truncated_message(tmp_path, monkeypatch) -> None:
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    published = datetime.now(timezone.utc)
    event_date = (date.today() + timedelta(days=5)).isoformat()

    # Message text ends with "..." and is long enough to trigger the truncation heuristic.
    base = ("Основной текст " + ("x" * 180)).strip()
    truncated = f"{base}..."
    full_text = f"{base}\n\nИх поддержит: Max’s Stereo Film Night\nЕщё строка"

    public_html = f"""
<div class="tgme_widget_message_wrap js-widget_message_wrap">
  <div class="tgme_widget_message text_not_supported_wrap js-widget_message" data-post="meowafisha/6773">
    <div class="tgme_widget_message_text js-message_text" dir="auto">{full_text.replace(chr(10), "<br>")}</div>
  </div>
</div>
"""

    async def fake_http_call(_label, _method, _url, **_kwargs):
        return SimpleNamespace(status_code=200, content=public_html.encode("utf-8"))

    import net

    monkeypatch.setattr(net, "http_call", fake_http_call)

    captured_source_texts: list[str] = []

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        captured_source_texts.append(candidate.source_text or "")
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

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    results = {
        "schema_version": 2,
        "run_id": "r-full-text-fallback",
        "generated_at": published.isoformat(),
        "stats": {"sources_total": 1, "messages_scanned": 1, "messages_with_events": 1, "events_extracted": 1},
        "messages": [
            {
                "source_username": "meowafisha",
                "source_title": "Meow",
                "message_id": 6773,
                "message_date": published.isoformat(),
                "source_link": "https://t.me/meowafisha/6773",
                "text": truncated,
                "posters": [{"sha256": "p", "catbox_url": "https://files.catbox.moe/p.jpg", "ocr_text": "афиша"}],
                "events": [{"title": "INDIE ДИСКОТЕКА", "date": event_date, "time": "19:00", "location_name": "Venue"}],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    await process_telegram_results(results_path, db, bot=object(), progress_callback=None)

    assert captured_source_texts, "expected Smart Update to be called"
    assert "Их поддержит" in captured_source_texts[0]
    assert not captured_source_texts[0].rstrip().endswith("...")
