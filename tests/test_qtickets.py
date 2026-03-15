from __future__ import annotations

import json

from source_parsing.qtickets import parse_qtickets_output


def test_parse_qtickets_output_accepts_current_kernel_contract(tmp_path) -> None:
    payload = [
        {
            "title": "DJ Supaisky",
            "description": "Танцевальная вечеринка.",
            "age_restriction": "18+",
            "date_raw": "2026-03-14T21:00:00+02:00",
            "parsed_date": "2026-03-14",
            "parsed_time": "21:00",
            "location": "Бар Советов",
            "url": "https://kaliningrad.qtickets.events/123-dj-supaisky",
            "photos": ["https://example.com/poster.jpg"],
            "ticket_price_min": 0,
            "ticket_price_max": 0,
            "ticket_status": "available",
            "source_type": "qtickets",
        }
    ]
    path = tmp_path / "qtickets_events.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    events = parse_qtickets_output([str(path)])

    assert len(events) == 1
    event = events[0]
    assert event.title == "DJ Supaisky"
    assert event.date_raw == "2026-03-14T21:00:00+02:00"
    assert event.parsed_date == "2026-03-14"
    assert event.parsed_time == "21:00"
    assert event.photos == ["https://example.com/poster.jpg"]
    assert event.ticket_price_min == 0
    assert event.ticket_price_max == 0
    assert event.ticket_status == "available"


def test_parse_qtickets_output_keeps_legacy_json_contract_compatible(tmp_path) -> None:
    payload = [
        {
            "title": "Legacy Qtickets Event",
            "description": "Legacy payload.",
            "date": "2026-03-15",
            "time": "20:30",
            "location": "Бар Советов",
            "url": "https://kaliningrad.qtickets.events/legacy",
            "image_url": "https://example.com/legacy.jpg",
            "price_min": 400,
            "price_max": 900,
        }
    ]
    path = tmp_path / "qtickets_events.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    events = parse_qtickets_output([str(path)])

    assert len(events) == 1
    event = events[0]
    assert event.date_raw == "2026-03-15"
    assert event.parsed_date == "2026-03-15"
    assert event.parsed_time == "20:30"
    assert event.photos == ["https://example.com/legacy.jpg"]
    assert event.ticket_price_min == 400
    assert event.ticket_price_max == 900
    assert event.ticket_status == "available"
