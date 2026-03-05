from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_prefilters_long_historical_non_event(monkeypatch):
    async def should_not_parse(*_args, **_kwargs):
        raise AssertionError("parse_event_via_llm must not be called for obvious non-events")

    monkeypatch.setattr(main, "parse_event_via_llm", should_not_parse)
    monkeypatch.setattr(vk_intake, "extract_event_ts_hint", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("VK_AUTO_IMPORT_PREFILTER_HISTORY_MIN_CHARS", "500")

    text = (
        "В 1945 году в Кёнигсберге и Раушене формировалась новая городская среда. "
        "Этот исторический очерк рассказывает о первых школах, улицах и послевоенной жизни региона. "
    ) * 12

    drafts, festival = await vk_intake.build_event_drafts_from_vk(
        text,
        publish_ts=datetime.now(main.LOCAL_TZ),
        prefilter_obvious_non_events=True,
    )

    assert festival is None
    assert len(drafts) == 1
    assert "исторический/справочный" in str(drafts[0].reject_reason or "").lower()


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_prefilter_keeps_future_history_lecture(monkeypatch):
    captured: dict[str, int] = {"calls": 0}

    async def fake_parse(*_args, **_kwargs):
        captured["calls"] += 1
        return [
            {
                "title": "Лекция о Кёнигсберге",
                "short_description": "Публичная лекция о городе и его послевоенной истории.",
                "date": "2026-03-12",
                "time": "19:00",
                "location_name": "Историко-художественный музей",
                "event_type": "лекция",
                "emoji": "🏛️",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)
    monkeypatch.setenv("VK_AUTO_IMPORT_PREFILTER_HISTORY_MIN_CHARS", "500")

    text = (
        "Лекция о Кёнигсберге 1945 года состоится 12 марта в 19:00 в музее. "
        "Историк расскажет о первых послевоенных годах и покажет редкие документы. "
    ) * 6

    drafts, festival = await vk_intake.build_event_drafts_from_vk(
        text,
        publish_ts=datetime.now(main.LOCAL_TZ),
        prefilter_obvious_non_events=True,
    )

    assert festival is None
    assert captured["calls"] == 1
    assert len(drafts) == 1
    assert not (drafts[0].reject_reason or "").strip()
