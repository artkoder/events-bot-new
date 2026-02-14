from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_vk_intake_marks_recap_title_as_low_confidence(monkeypatch) -> None:
    # VK post: recap about a past concert with an explicit title + a short mention of a future
    # concert without a title. LLM may incorrectly reuse the past title for the future date.
    text = (
        "Мы знаем, что вы любите Миядзаки!\n"
        "Поэтому 12 февраля «Молодежная академия искусств» вновь исполнила программу "
        "«Волшебный мир Хаяо Миядзаки» для всех поклонников.\n"
        "На сцене филармонии выступили музыканты...\n"
        "🔔 19 марта для фанатов музыки, компьютерных игр и аниме музыканты "
        "исполнят тематический концерт.\n"
    )

    async def fake_parse(*_args, **_kwargs):
        return [
            {
                "title": "Волшебный мир Хаяо Миядзаки",
                "date": "2026-03-19",
                "time": None,
                "location_name": "Калининградская областная филармония",
                "short_description": "stub",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    publish_dt = datetime(2026, 2, 14, 18, 0, tzinfo=main.LOCAL_TZ)
    drafts, _ = await vk_intake.build_event_drafts_from_vk(text, publish_ts=publish_dt)

    assert len(drafts) == 1
    assert drafts[0].reject_reason, "expected a low-confidence reject_reason"
    assert "Низкая уверенность" in drafts[0].reject_reason


@pytest.mark.asyncio
async def test_vk_intake_allows_title_when_repeated_near_future_date(monkeypatch) -> None:
    text = (
        "12 февраля прошёл концерт «Волшебный мир Хаяо Миядзаки».\n"
        "А 19 марта состоится концерт «Волшебный мир Хаяо Миядзаки» снова.\n"
    )

    async def fake_parse(*_args, **_kwargs):
        return [
            {
                "title": "Волшебный мир Хаяо Миядзаки",
                "date": "2026-03-19",
                "time": None,
                "location_name": "Филармония",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    publish_dt = datetime(2026, 2, 14, 18, 0, tzinfo=main.LOCAL_TZ)
    drafts, _ = await vk_intake.build_event_drafts_from_vk(text, publish_ts=publish_dt)

    assert len(drafts) == 1
    assert not (drafts[0].reject_reason or "").strip()

