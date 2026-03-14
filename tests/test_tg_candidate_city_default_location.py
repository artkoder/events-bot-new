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
    assert cand.location_name == "Заря, Мира 41-43, Калининград"


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


def test_tg_build_candidate_keeps_explicit_offsite_location_over_default_location() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(
        default_location="Замок Инстербург, Замковая 1, Черняховск",
        default_ticket_link=None,
        trust_level="high",
    )
    message = {
        "source_username": "zamokinsterburg",
        "message_id": 5441,
        "source_link": "https://t.me/zamokinsterburg/5441",
        "text": (
            "16 марта в Центральной городской библиотеке им. А. Лунина "
            "состоится презентация книги. Адрес: ул. Калинина, 4."
        ),
        "events": [{"title": "Event", "date": "2026-03-16", "time": "14:00"}],
        "posters": [],
    }
    ev = {
        "title": "Event",
        "date": "2026-03-16",
        "time": "14:00",
        "city": "Черняховск",
        "location_name": "Библиотека им. Лунина",
        "location_address": "Калинина 4",
        "source_text": message["text"],
    }
    cand = _build_candidate(src, message, ev)
    assert cand.location_name == "Библиотека им. Лунина"
    assert cand.location_address == "Калинина 4"
    assert cand.city == "Черняховск"
    assert cand.metrics["tg_location_overridden_by_default"] is False
    assert cand.metrics["tg_location_kept_extracted"] is True


def test_tg_build_candidate_keeps_default_location_when_mismatch_not_grounded_in_text() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(
        default_location="Заря, Мира 41-43, Калининград",
        default_ticket_link=None,
        trust_level="high",
    )
    message = {
        "source_username": "zaryakinoteatr",
        "message_id": 802,
        "source_link": "https://t.me/zaryakinoteatr/802",
        "text": "На сцене — актёры театра и кино. Билеты в кассе театра.",
        "events": [{"title": "Event", "date": "2026-03-08", "time": "20:00"}],
        "posters": [],
    }
    ev = {
        "title": "Event",
        "date": "2026-03-08",
        "time": "20:00",
        "city": "Калининград",
        "location_name": "Клуб X",
        "location_address": "Примерная 7",
        "source_text": "На сцене — актёры театра и кино.",
    }
    cand = _build_candidate(src, message, ev)
    assert cand.location_name == "Заря, Мира 41-43, Калининград"
    assert cand.location_address is None
    assert cand.city == "Калининград"
    assert cand.metrics["tg_location_overridden_by_default"] is True
