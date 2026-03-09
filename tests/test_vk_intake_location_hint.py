from __future__ import annotations

from datetime import datetime, timezone

import pytest

from db import Database
import vk_intake
import main


@pytest.mark.asyncio
async def test_vk_intake_injects_location_hint_into_llm_text(tmp_path, monkeypatch):
    captured: dict[str, str] = {}

    async def fake_parse_event_via_llm(text: str, *args, **kwargs):
        captured["text"] = text
        return [
            {
                "title": "T",
                "short_description": "Описание события без логистики.",
                "festival": "",
                "festival_full": "",
                "date": "2026-03-11",
                "time": "11:00",
                "location_name": "Дворец спорта «Юность»",
                "location_address": "Маршала Баграмяна 2",
                "city": "Калининград",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "ticket_link": "",
                "is_free": False,
                "pushkin_card": False,
                "event_type": "спорт",
                "emoji": None,
                "end_date": None,
                "search_digest": "Спортивное мероприятие и соревнования ГТО среди школьных команд.",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse_event_via_llm)

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    drafts, _fest = await vk_intake.build_event_drafts_from_vk(
        "11 марта во Дворце спорта «Юность» в Калининграде пройдёт зимний фестиваль ВФСК ГТО.",
        source_name="test",
        location_hint="Дворец спорта «Юность», Маршала Баграмяна 2, Калининград",
        default_time=None,
        default_ticket_link=None,
        operator_extra=None,
        publish_ts=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
        event_ts_hint=None,
        festival_names=None,
        festival_alias_pairs=None,
        festival_hint=False,
        poster_media=None,
        ocr_tokens_spent=0,
        ocr_tokens_remaining=None,
    )

    assert drafts
    assert "Хинт по локации" in (captured.get("text") or "")
    assert "Дворец спорта «Юность»" in (captured.get("text") or "")
