from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_vk_intake_strips_time_that_is_copied_from_dd_mm_date(monkeypatch):
    async def fake_parse(*_args, **_kwargs):
        # The source contains "21.02" as a date, but the model mistakenly emits "21:02" as time.
        return [
            {
                "title": "Тест",
                "short_description": "Описание.",
                "date": "2026-02-21",
                "time": "21:02",
                "location_name": "Локация",
                "event_type": "встреча",
                "emoji": "",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        "Показ пройдёт 21.02. Подробности в посте.",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert len(drafts) == 1
    assert drafts[0].date == "2026-02-21"
    assert drafts[0].time is None


@pytest.mark.asyncio
async def test_vk_intake_collapses_program_schedule_drafts_into_time_range(monkeypatch):
    async def fake_parse(*_args, **_kwargs):
        # Model splits program items into separate events (undesired for umbrella holiday programs).
        return [
            {
                "title": "Душа моя, Масленица!",
                "short_description": "Праздничные гуляния с программой.",
                "date": "2026-02-22",
                "time": "10:00",
                "location_name": "Площадка",
                "event_type": "встреча",
                "emoji": "",
                "ticket_link": "",
            },
            {
                "title": "Душа моя, Масленица!",
                "short_description": "Праздничные гуляния с программой.",
                "date": "2026-02-22",
                "time": "11:00",
                "location_name": "Площадка",
                "event_type": "встреча",
                "emoji": "",
                "ticket_link": "",
            },
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        "Душа моя, Масленица!\n\n"
        "Программа:\n"
        "10:00 — сбор гостей\n"
        "11:00 — игры\n"
        "14:00 — финал\n",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert len(drafts) == 1
    assert drafts[0].date == "2026-02-22"
    assert drafts[0].time == "10:00..14:00"


@pytest.mark.asyncio
async def test_vk_intake_fills_default_time_without_leaking_it_into_prompt(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_parse(*_args, **_kwargs):
        captured["prompt"] = str(_args[0] if _args else "")
        return [
            {
                "title": "Тест",
                "short_description": "Описание.",
                "date": "2026-04-04",
                "time": "",
                "location_name": "Драматический театр",
                "event_type": "спектакль",
                "emoji": "",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        "Анонс спектакля «Гараж». 4 апреля.",
        default_time="19:00",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert len(drafts) == 1
    assert drafts[0].date == "2026-04-04"
    assert drafts[0].time == "19:00"
    assert drafts[0].time_is_default is True
    assert "предположи начало" not in captured.get("prompt", "").lower()
