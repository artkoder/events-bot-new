from __future__ import annotations


def test_venue_status_update_detected_as_non_event() -> None:
    from smart_event_update import _looks_like_venue_status_update_not_event

    title = "Ворота // арт-пространство"
    text = (
        "Ситуация крайне сложная. Город мог потерять «Ворота» уже с 1 мая. "
        "Пока дана отсрочка до 1 июня."
    )
    assert _looks_like_venue_status_update_not_event(title, text) is True


def test_venue_status_update_kept_when_event_signals_present() -> None:
    from smart_event_update import _looks_like_venue_status_update_not_event

    title = "Концерт перед закрытием"
    text = (
        "Закрытие площадки обсуждается, но концерт состоится 1 мая в 19:00. "
        "Площадка работает до 1 июня."
    )
    assert _looks_like_venue_status_update_not_event(title, text) is False


def test_venue_status_update_requires_deadline_anchor() -> None:
    from smart_event_update import _looks_like_venue_status_update_not_event

    title = "Новости площадки"
    text = "Ситуация крайне сложная, нужна поддержка, но без конкретных дат."
    assert _looks_like_venue_status_update_not_event(title, text) is False

