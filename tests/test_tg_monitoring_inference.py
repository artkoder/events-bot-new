from source_parsing.telegram.handlers import (
    _extract_ticket_link_from_text,
    _infer_location_from_text,
    _infer_title_from_message_text,
)


def test_infer_location_splits_venue_and_address() -> None:
    text = """
17.02 КВИЗ: СУМЕРКИ

...

Ресторан «Мушкино», Сергеева 14
Сбор гостей 19:00, начало 19:30
"""
    loc, addr = _infer_location_from_text(text)
    assert loc == "Ресторан «Мушкино»"
    assert addr == "Сергеева 14"


def test_infer_location_from_pin_line() -> None:
    text = "📍 Арт Гармония, ул. Старорусская, 29"
    loc, addr = _infer_location_from_text(text)
    assert loc == "Арт Гармония"
    assert addr == "ул. Старорусская, 29"


def test_infer_title_skips_date_prefix() -> None:
    text = "18.02 | ШОУ «КЛУБ ЗНАКОМСТВ»\n\nПервое комедийное шоу..."
    title = _infer_title_from_message_text(text)
    assert title == "ШОУ «КЛУБ ЗНАКОМСТВ»"


def test_infer_ticket_link_from_booking_handle() -> None:
    text = "Чтобы забронировать место, напиши: @owlet0226"
    ticket = _extract_ticket_link_from_text(text)
    assert ticket == "https://t.me/owlet0226"

