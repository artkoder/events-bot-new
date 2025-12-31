import pytest
from datetime import date

import main_part2
from main_part2 import Event, event_to_nodes, _build_month_page_content_sync


def _make_event(event_id: int) -> Event:
    return Event(
        id=event_id,
        title=f"E{event_id}",
        description="Test Description",
        date="2026-05-15",
        time="19:00",
        location_name="Test Location",
        source_text="source",
        photo_urls=["http://img"],
    )

@pytest.mark.asyncio
async def test_event_to_nodes_rendering():
    ev = Event(
        id=1,
        title="Test Event",
        description="Test Description",
        date="2026-05-15",
        time="19:00",
        location_name="Test Location",
        source_text="source",
        photo_urls=["https://example.com/image.jpg"],
        telegraph_url="https://telegra.ph/Test-Event",
    )
    
    # Case 1: show_image=True
    nodes_with_image = event_to_nodes(ev, show_image=True)
    # Check if first node is figure
    assert nodes_with_image[0]["tag"] == "figure"
    assert nodes_with_image[0]["children"][0]["tag"] == "img"
    assert nodes_with_image[0]["children"][0]["attrs"]["src"] == "https://example.com/image.jpg"
    
    # Case 2: show_image=False
    nodes_no_image = event_to_nodes(ev, show_image=False)
    assert nodes_no_image[0]["tag"] != "figure"
    assert nodes_no_image[0]["tag"] == "h4"

@pytest.mark.asyncio
async def test_build_month_page_threshold(monkeypatch):
    captured = []

    def fake_add_day_sections(days, by_day, fest_map, add_many, **kwargs):
        captured.append(kwargs.get("show_images"))

    monkeypatch.setattr(main_part2, "add_day_sections", fake_add_day_sections)
    monkeypatch.setattr(
        main_part2, "parse_iso_date", lambda value: date.fromisoformat(value), raising=False
    )
    monkeypatch.setattr(main_part2, "rough_size", lambda nodes: 0, raising=False)

    events_small = [_make_event(i) for i in range(9)]
    _build_month_page_content_sync(
        "2026-05",
        events_small,
        [],
        {},
        None,
        None,
        None,
        True,
        True,
    )

    events_large = [_make_event(i) for i in range(10)]
    _build_month_page_content_sync(
        "2026-05",
        events_large,
        [],
        {},
        None,
        None,
        None,
        True,
        True,
    )

    assert captured == [True, False]
