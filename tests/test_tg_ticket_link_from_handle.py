def test_tg_ticket_link_from_handle_is_detected() -> None:
    from source_parsing.telegram.handlers import _extract_ticket_link_from_text

    text = "Билеты у @Masha_v_sety"
    assert _extract_ticket_link_from_text(text) == "https://t.me/Masha_v_sety"


def test_tg_ticket_link_from_handle_is_detected_in_colon_format() -> None:
    from source_parsing.telegram.handlers import _extract_ticket_link_from_text

    assert _extract_ticket_link_from_text("Билеты: @olua503") == "https://t.me/olua503"


def test_tg_ticket_link_from_tme_link_is_detected() -> None:
    from source_parsing.telegram.handlers import _extract_ticket_link_from_text

    assert _extract_ticket_link_from_text("Запись: t.me/owlet0226") == "https://t.me/owlet0226"
