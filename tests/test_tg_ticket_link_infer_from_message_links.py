from __future__ import annotations


def test_infer_ticket_link_from_message_links_single_external() -> None:
    from source_parsing.telegram.handlers import _infer_ticket_link_from_message_links

    assert (
        _infer_ticket_link_from_message_links(["https://signalcommunity.timepad.ru/event/3821867/"])
        == "https://signalcommunity.timepad.ru/event/3821867/"
    )


def test_infer_ticket_link_from_message_links_ignores_multiple_unknowns() -> None:
    from source_parsing.telegram.handlers import _infer_ticket_link_from_message_links

    assert (
        _infer_ticket_link_from_message_links(
            [
                "https://example.com/a",
                "https://example.com/b",
            ]
        )
        is None
    )


def test_infer_ticket_link_from_message_links_picks_single_strong_domain() -> None:
    from source_parsing.telegram.handlers import _infer_ticket_link_from_message_links

    assert (
        _infer_ticket_link_from_message_links(
            [
                "https://example.com/a",
                "https://foo.timepad.ru/event/1/",
                "https://t.me/somechannel/123",
            ]
        )
        == "https://foo.timepad.ru/event/1/"
    )

