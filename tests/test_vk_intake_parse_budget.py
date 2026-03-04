from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake
from poster_media import PosterMedia


def _poster(text: str) -> PosterMedia:
    return PosterMedia(
        data=b"x",
        name="poster.jpg",
        ocr_text=text,
        prompt_tokens=10,
        completion_tokens=1,
        total_tokens=11,
    )


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_skips_poster_ocr_for_long_posts(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_parse(*_args, **kwargs):
        captured["poster_texts"] = kwargs.get("poster_texts")
        captured["poster_summary"] = kwargs.get("poster_summary")
        return [
            {
                "title": "Весенний концерт",
                "short_description": "Программа музея к 8 марта.",
                "date": "2026-03-08",
                "time": "15:00",
                "location_name": "Калининградский историко-художественный музей",
                "event_type": "концерт",
                "emoji": "🎻",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    long_text = "Афиша музея.\n" + ("Подробная программа мероприятия.\n" * 90)
    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        long_text,
        publish_ts=datetime.now(main.LOCAL_TZ),
        poster_media=[
            _poster("OCR блок 1 с датой 8 марта 15:00"),
            _poster("OCR блок 2 с датой 8 марта 11:00"),
        ],
    )

    assert len(drafts) == 1
    assert captured["poster_texts"] is None
    assert captured["poster_summary"] == "Posters processed: 2. Tokens — prompt: 20, completion: 2, total: 22."


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_keeps_poster_ocr_for_short_posts(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_parse(*_args, **kwargs):
        captured["poster_texts"] = kwargs.get("poster_texts")
        return [
            {
                "title": "Весенний концерт",
                "short_description": "Программа музея к 8 марта.",
                "date": "2026-03-08",
                "time": "15:00",
                "location_name": "Калининградский историко-художественный музей",
                "event_type": "концерт",
                "emoji": "🎻",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        "Короткий анонс.\n8 марта в 15:00.",
        publish_ts=datetime.now(main.LOCAL_TZ),
        poster_media=[_poster("OCR блок 1 с датой 8 марта 15:00")],
    )

    assert len(drafts) == 1
    assert captured["poster_texts"] == ["OCR блок 1 с датой 8 марта 15:00"]
