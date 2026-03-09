from __future__ import annotations

import main


def test_known_venue_matching_accepts_common_alias_filharmonia() -> None:
    venue = main._match_known_venue("Калининградская филармония", city="Калининград")
    assert venue is not None
    assert venue.canonical_line.startswith("Филармония им. Светланова,")


def test_known_venue_matching_by_address_accepts_moskovsky_ave_abbr_variants() -> None:
    venue = main._match_known_venue_by_address("Московский пр-т, 39.", city="Калининград")
    assert venue is not None
    assert venue.canonical_line.startswith("Библиотека Чехова,")


def test_location_normalisation_prefers_known_address_over_mismatched_known_venue() -> None:
    obj = {
        "location_name": "Научная библиотека, Мира 9, Калининград",
        "location_address": "Московский пр-т, 39.",
        "city": "Калининград",
    }
    main._normalise_event_location_from_reference(obj)
    assert obj.get("location_name") == "Библиотека Чехова"
    assert obj.get("location_address") == "Московский пр-кт 39"
    assert obj.get("city") == "Калининград"


def test_location_normalisation_splits_known_venue_into_structured_fields() -> None:
    obj = {
        "location_name": "Историко-художественный музей, Клиническая 21, Калининград",
        "location_address": "Клиническая 21",
        "city": "Калининград",
    }
    main._normalise_event_location_from_reference(obj)
    assert obj.get("location_name") == "Историко-художественный музей"
    assert obj.get("location_address") == "Клиническая 21"
    assert obj.get("city") == "Калининград"


def test_known_venue_matching_gumbinnen_is_in_gusev() -> None:
    venue = main._match_known_venue("Дизайн-резиденция Gumbinnen", city="Гусев")
    assert venue is not None
    assert venue.canonical_line == "Дизайн-резиденция Gumbinnen, Ленина 29, Гусев"
