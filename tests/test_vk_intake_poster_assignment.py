from __future__ import annotations

from datetime import datetime, timezone

import pytest

from poster_media import PosterMedia

import vk_intake
import main


@pytest.mark.asyncio
async def test_vk_intake_assigns_posters_to_matching_drafts(monkeypatch) -> None:
    async def fake_parse_event_via_4o(text: str, *args, **kwargs):
        return [
            {
                "title": "Disco Party",
                "short_description": "Танцуем.",
                "festival": "",
                "date": "2026-02-22",
                "time": "19:00",
                "location_name": "Club X",
                "location_address": "Mira 1",
                "city": "Kaliningrad",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "ticket_link": "",
                "is_free": False,
                "pushkin_card": False,
                "event_type": "вечеринка",
                "emoji": None,
                "end_date": None,
                "search_digest": "",
            },
            {
                "title": "Тай-дай мастер-класс",
                "short_description": "Рисуем.",
                "festival": "Гаражка",
                "date": "2026-02-15",
                "time": "12:00",
                "location_name": "",
                "location_address": "",
                "city": "",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "ticket_link": "",
                "is_free": True,
                "pushkin_card": False,
                "event_type": "мастер-класс",
                "emoji": None,
                "end_date": None,
                "search_digest": "",
            },
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse_event_via_4o)

    poster1 = PosterMedia(data=b"1", name="p1")
    poster1.ocr_title = "DISCO PARTY"
    poster1.ocr_text = "22.02 19:00"

    poster2 = PosterMedia(data=b"2", name="p2")
    poster2.ocr_title = "Тай-дай"
    poster2.ocr_text = "15 февраля 12:00"

    generic = PosterMedia(data=b"3", name="p3")
    generic.ocr_title = "Афиша"
    generic.ocr_text = "Вход 350 руб."

    drafts, _ = await vk_intake.build_event_drafts_from_vk(
        "post text",
        source_name="test",
        publish_ts=datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
        poster_media=[poster1, poster2, generic],
        ocr_tokens_spent=0,
        ocr_tokens_remaining=None,
    )

    assert len(drafts) == 2
    assert [p.name for p in drafts[0].poster_media] == ["p1"]
    assert [p.name for p in drafts[1].poster_media] == ["p2"]

