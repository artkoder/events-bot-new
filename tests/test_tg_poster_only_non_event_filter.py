def test_tg_poster_only_non_event_is_detected() -> None:
    from source_parsing.telegram.handlers import _looks_like_poster_only_non_event

    message = {
        "text": "В посте нет ни слова про Сергея.",
        "posters": [
            {"ocr_title": "Sergey Smitanin. Academic Study", "ocr_text": "СЕРГЕЙ СМИТАНИН. АКАДЕМИЧЕСКИЙ ЭТЮД"},
        ],
    }
    event_data = {
        "title": "Сергей Смитанин. Академический этюд",
        "date": "2026-02-19",
        "time": None,
        "ticket_link": None,
        "end_date": None,
    }
    assert _looks_like_poster_only_non_event(message, event_data) is True


def test_tg_poster_only_event_with_date_on_poster_is_not_dropped() -> None:
    from source_parsing.telegram.handlers import _looks_like_poster_only_non_event

    message = {
        "text": "В посте нет текста, только афиша.",
        "posters": [
            {"ocr_title": "Лекция", "ocr_text": "22.02 19:00 Лекция о море"},
        ],
    }
    event_data = {
        "title": "Лекция о море",
        "date": "2026-02-22",
        "time": "19:00",
        "ticket_link": None,
        "end_date": None,
    }
    assert _looks_like_poster_only_non_event(message, event_data) is False

