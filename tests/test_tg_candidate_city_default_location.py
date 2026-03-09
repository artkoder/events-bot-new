from __future__ import annotations


def test_tg_build_candidate_overrides_extracted_city_with_default_location_city() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(
        default_location="Заря, Мира 41-43, Калининград",
        default_ticket_link=None,
        trust_level="high",
    )
    message = {
        "source_username": "zaryakinoteatr",
        "message_id": 801,
        "source_link": "https://t.me/zaryakinoteatr/801",
        "text": "На сцене — актёры театра и кино (г. Москва)",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {
        "title": "Event",
        "date": "2026-03-07",
        "time": "20:00",
        "city": "Москва",
        "location_name": None,
    }
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Калининград"
    assert cand.location_name == "Заря"
    assert cand.location_address == "Мира 41-43"


def test_tg_build_candidate_default_location_city_parses_hash_city() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(
        default_location="Кафедральный собор, Остров Канта, #Калининград",
        default_ticket_link=None,
        trust_level="high",
    )
    message = {
        "source_username": "sobor39",
        "message_id": 1,
        "source_link": "https://t.me/sobor39/1",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {"title": "Event", "date": "2026-03-07", "time": "20:00", "city": "Москва"}
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Калининград"
    assert cand.location_name == "Кафедральный собор"
    assert cand.location_address == "Остров Канта"


def test_tg_build_candidate_uses_structured_default_location_when_only_address_extracted() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(
        default_location="Научная библиотека, Мира 9, Калининград",
        default_ticket_link=None,
        trust_level="high",
    )
    message = {
        "source_username": "kaliningradlibrary",
        "message_id": 2,
        "source_link": "https://t.me/kaliningradlibrary/2",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {
        "title": "Event",
        "date": "2026-03-07",
        "time": "20:00",
        "location_address": "Мира 9, Калининград",
    }
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Калининград"
    assert cand.location_name == "Научная библиотека"
    assert cand.location_address == "Мира 9"


def test_tg_build_candidate_without_default_location_keeps_extracted_city() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(default_location=None, default_ticket_link=None, trust_level=None)
    message = {
        "source_username": "testchannel",
        "message_id": 1,
        "source_link": "https://t.me/testchannel/1",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {"title": "Event", "date": "2026-03-07", "time": "20:00", "city": "Москва"}
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Москва"


def test_tg_build_candidate_does_not_infer_city_from_plain_venue_name() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(default_location=None, default_ticket_link=None, trust_level=None)
    message = {
        "source_username": "testchannel",
        "message_id": 1,
        "source_link": "https://t.me/testchannel/1",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {"title": "Event", "date": "2026-03-07", "time": "20:00", "location_name": "ЗАРЯ"}
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Калининград"


def test_tg_build_candidate_does_not_infer_city_from_venue_and_street() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(default_location=None, default_ticket_link=None, trust_level=None)
    message = {
        "source_username": "testchannel",
        "message_id": 1,
        "source_link": "https://t.me/testchannel/1",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-03-07", "time": "20:00"}],
        "posters": [],
    }
    ev = {
        "title": "Event",
        "date": "2026-03-07",
        "time": "20:00",
        "location_name": "Заря, Мира 41-43",
        "city": "Москва",
    }
    cand = _build_candidate(src, message, ev)
    assert cand.city == "Москва"
