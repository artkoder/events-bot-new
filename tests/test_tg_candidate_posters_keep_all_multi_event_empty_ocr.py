import pytest


@pytest.mark.asyncio
async def test_tg_build_candidate_keeps_posters_for_multi_event_when_ocr_empty():
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
            {"title": "Event A", "date": "2026-02-21", "time": "19:00"},
            {"title": "Event B", "date": "2026-02-21", "time": "20:00"},
        ],
        # Posters without OCR text/title (e.g., plain photos).
        "posters": [
            {"sha256": "a", "catbox_url": "https://files.catbox.moe/a.jpg", "ocr_text": "", "ocr_title": ""},
            {"sha256": "b", "catbox_url": "https://files.catbox.moe/b.jpg", "ocr_text": None, "ocr_title": None},
            {"sha256": "c", "catbox_url": "https://files.catbox.moe/c.jpg"},
        ],
    }

    cand0 = _build_candidate(src, message, message["events"][0])
    cand1 = _build_candidate(src, message, message["events"][1])
    assert [p.sha256 for p in cand0.posters] == ["a", "b", "c"]
    assert [p.sha256 for p in cand1.posters] == ["a", "b", "c"]

