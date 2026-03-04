def test_tg_ticket_link_bare_url_is_coerced_to_https() -> None:
    from types import SimpleNamespace

    from source_parsing.telegram.handlers import _build_candidate

    src = SimpleNamespace(default_location="Somewhere", default_ticket_link=None, trust_level=None)
    message = {
        "source_username": "testchannel",
        "message_id": 1,
        "source_link": "https://t.me/testchannel/1",
        "text": "Text",
        "events": [{"title": "Event", "date": "2026-02-21", "time": "19:00"}],
        "posters": [],
    }
    ev = {"title": "Event", "date": "2026-02-21", "time": "19:00", "ticket_link": "clck.ru/ABC123"}
    cand = _build_candidate(src, message, ev)
    assert cand.ticket_link == "https://clck.ru/ABC123"

