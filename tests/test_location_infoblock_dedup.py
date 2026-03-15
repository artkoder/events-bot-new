from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_source_summary_location_avoids_embedded_address_city_dupes() -> None:
    import main

    summary = main.SourcePageEventSummary(
        date="2026-02-15",
        time="16:00",
        location_name="Bar Sovetov, Mira 118, Kaliningrad",
        location_address="Mira 118",
        city="Kaliningrad",
        is_free=False,
    )

    html = await main._build_source_summary_block(summary)
    assert "📍" in html
    assert "Bar Sovetov" in html
    assert html.count("Kaliningrad") == 1
    assert "Mira 118, Kaliningrad, Mira 118" not in html


@pytest.mark.asyncio
async def test_exhibition_summary_hides_inferred_end_date() -> None:
    import main

    summary = main.SourcePageEventSummary(
        date="2026-03-05",
        end_date="2026-04-05",
        end_date_is_inferred=True,
        event_type="выставка",
        is_free=False,
    )

    html = await main._build_source_summary_block(summary)
    assert "🗓 с 5 марта" in html
    assert "по 5 апреля" not in html


@pytest.mark.asyncio
async def test_exhibition_summary_shows_confirmed_future_range() -> None:
    import main

    summary = main.SourcePageEventSummary(
        date="2099-03-05",
        end_date="2099-04-05",
        end_date_is_inferred=False,
        event_type="выставка",
        is_free=False,
    )

    html = await main._build_source_summary_block(summary)
    assert "🗓 с 5 марта по 5 апреля" in html


def test_known_venue_matching_does_not_guess_unknown_school_by_generic_token() -> None:
    import main

    venue = main._match_known_venue("Школа им. М.С. Любушкина", city="Янтарный")
    assert venue is None


def test_location_normalisation_keeps_raw_unknown_venue_when_address_conflicts() -> None:
    import main

    obj = {
        "location_name": "Школа им. М.С. Любушкина",
        "location_address": "пгт Янтарный, ул. Лесная, 10А",
        "city": "Янтарный",
    }
    main._normalise_event_location_from_reference(obj)
    assert obj["location_name"] == "Школа им. М.С. Любушкина"
    assert obj["location_address"] == "пгт Янтарный, ул. Лесная, 10А"
    assert obj["city"] == "Янтарный"


def test_location_normalisation_overrides_wrong_city_for_known_venue() -> None:
    import main

    obj = {
        "location_name": "Заря",
        "location_address": None,
        "city": "Москва",
    }
    main._normalise_event_location_from_reference(obj)
    assert obj["location_name"] == "Заря, Мира 41-43, Калининград"
    assert obj["city"] == "Калининград"
