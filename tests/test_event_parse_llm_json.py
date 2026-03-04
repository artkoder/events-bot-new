import pytest

import main


def test_event_parse_extract_json_handles_code_fences() -> None:
    raw = "```json\n[{\"title\":\"A\"}]\n```"
    data = main._event_parse_extract_json(raw)
    assert isinstance(data, list)
    assert data[0]["title"] == "A"


def test_event_parse_extract_json_recovers_embedded_json() -> None:
    raw = "Here is the result:\n\n```json\n{\"events\":[{\"title\":\"A\"}]}\n```\nThanks!"
    data = main._event_parse_extract_json(raw)
    assert isinstance(data, dict)
    assert isinstance(data.get("events"), list)
    assert data["events"][0]["title"] == "A"


def test_event_parse_normalize_dict_with_events_and_festival() -> None:
    payload = {
        "festival": "Test Fest",
        "events": [
            {
                "title": "E1",
                "location_name": "Some Venue",
                "city": "Калининград",
            }
        ],
    }
    out = main._event_parse_normalize_parsed_events(payload)
    assert isinstance(out, main.ParsedEvents)
    assert len(out) == 1
    assert out[0]["title"] == "E1"
    assert isinstance(out.festival, dict)
    assert out.festival.get("name") == "Test Fest"


def test_event_parse_normalize_list() -> None:
    out = main._event_parse_normalize_parsed_events([{"title": "E1"}, {"title": "E2"}])
    assert len(out) == 2
    assert out[1]["title"] == "E2"


def test_event_parse_normalize_rejects_bad_types() -> None:
    with pytest.raises(RuntimeError):
        main._event_parse_normalize_parsed_events("not json")  # type: ignore[arg-type]

