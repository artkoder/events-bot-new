from __future__ import annotations

from types import SimpleNamespace

from source_parsing.telegram import handlers as tg_handlers


def _source(**overrides):
    base = {
        "default_location": None,
        "default_ticket_link": None,
        "trust_level": "medium",
        "festival_source": False,
        "festival_series": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_candidate_normalizes_bar_sovetov_alias_to_canonical_reference() -> None:
    candidate = tg_handlers._build_candidate(
        _source(),
        {
            "source_username": "test_channel",
            "message_id": 101,
            "text": "DJ Supaisky в Bar Sovetov.",
        },
        {
            "title": "DJ Supaisky",
            "date": "2026-03-14",
            "time": "21:00",
            "location_name": "Bar Sovetov",
            "city": "Калининград",
        },
    )

    assert candidate.location_name == "Бар Советов, Мира 118, Калининград"
    assert candidate.location_address == "Мира 118"
    assert candidate.city == "Калининград"


def test_build_candidate_normalizes_known_location_by_address_even_with_typo() -> None:
    candidate = tg_handlers._build_candidate(
        _source(),
        {
            "source_username": "test_channel",
            "message_id": 102,
            "text": "Панки и пост-панки в Суспирии.",
        },
        {
            "title": "Панки и пост-панки",
            "date": "2026-03-14",
            "time": "20:00",
            "location_name": "бар “Сусппирия”",
            "location_address": "Коперника 21",
            "city": "Калининград",
        },
    )

    assert candidate.location_name == "Бар Суспирия, Коперника 21, Калининград"
    assert candidate.location_address == "Коперника 21"
    assert candidate.city == "Калининград"
