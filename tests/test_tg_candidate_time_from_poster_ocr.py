from types import SimpleNamespace

from source_parsing.telegram.handlers import _build_candidate


def test_build_candidate_infers_time_from_poster_ocr_when_missing() -> None:
    source = SimpleNamespace(
        default_location=None,
        default_ticket_link=None,
        trust_level="medium",
    )
    message = {
        "source_username": "meowafisha",
        "message_id": 1,
        "source_link": "https://t.me/meowafisha/1",
        "text": "В воскресенье — DJ-сет от Mamoru.",
        "posters": [
            {
                "sha256": "a" * 64,
                "ocr_title": "DJ-set от Mamoru",
                "ocr_text": "22.02 | 20:00 HEART ROCK BAR",
            }
        ],
    }
    event_data = {
        "title": "DJ-сет от Mamoru",
        "date": "2026-02-22",
        "time": "",
        "location_name": "Heart Rock Bar",
        "location_address": "Советский проспект 21а, Калининград",
        "city": "Калининград",
    }
    cand = _build_candidate(source, message, event_data)
    assert cand.time == "20:00"

