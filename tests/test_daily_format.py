from datetime import date

import main


def make_event(**kwargs: object) -> main.Event:
    base = {
        "title": "Event",
        "description": "Описание",
        "source_text": "source",
        "date": date(2024, 1, 1).isoformat(),
        "time": "18:00",
        "location_name": "Place",
    }
    base.update(kwargs)
    return main.Event(**base)


def test_format_event_daily_uses_partner_vk_link() -> None:
    event = make_event(
        source_post_url="https://vk.com/wall-1_1",
        creator_id=123,
    )

    rendered = main.format_event_daily(event, partner_creator_ids={123})

    assert '<a href="https://vk.com/wall-1_1">' in rendered


def test_format_event_daily_prefers_telegraph_for_vk_queue() -> None:
    event = make_event(
        source_vk_post_url="https://vk.com/wall-1_2",
        telegraph_url="https://telegra.ph/test",
    )

    rendered = main.format_event_daily(event)

    assert '<a href="https://telegra.ph/test">' in rendered
