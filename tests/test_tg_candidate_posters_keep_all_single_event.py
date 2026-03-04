import pytest


@pytest.mark.asyncio
async def test_tg_build_candidate_keeps_all_posters_for_single_event():
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    # Minimal TelegramSource stub (only fields accessed by _build_candidate).
    src = SimpleNamespace(default_location=None, default_ticket_link=None, trust_level=None)

    message = {
        "source_username": "testchannel",
        "message_id": 123,
        "source_link": "https://t.me/testchannel/123",
        "text": "Text",
        "events": [
            {
                "title": "Event",
                "date": "2026-02-21",
                "time": "19:00",
            }
        ],
        "posters": [
            {"sha256": "a", "catbox_url": "https://files.catbox.moe/a.jpg"},
            {"sha256": "b", "catbox_url": "https://files.catbox.moe/b.jpg"},
            {"sha256": "c", "catbox_url": "https://files.catbox.moe/c.jpg"},
            {"sha256": "d", "catbox_url": "https://files.catbox.moe/d.jpg"},
        ],
    }

    cand = _build_candidate(src, message, message["events"][0])
    assert cand.posters is not None
    assert len(cand.posters) == 4
    assert [p.sha256 for p in cand.posters] == ["a", "b", "c", "d"]

