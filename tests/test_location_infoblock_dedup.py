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

