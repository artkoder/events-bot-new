from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_source_summary_renders_other_dates_line() -> None:
    import main

    summary = main.SourcePageEventSummary(
        date="2026-03-07",
        time="11:00",
        location_name="Студия",
        is_free=False,
        other_dates=[
            main.RelatedEventDate(
                date="2026-03-12",
                time="10:30",
                url="https://telegra.ph/test-03-12",
            ),
            main.RelatedEventDate(
                date="2026-03-26",
                time="10:30",
                lifecycle_status="cancelled",
            ),
            main.RelatedEventDate(
                date="2026-03-29",
                time="14:00",
                lifecycle_status="postponed",
                url="https://telegra.ph/test-03-29",
            ),
        ],
        other_dates_more=2,
    )

    html = await main._build_source_summary_block(summary)

    assert "Другие даты" in html
    assert "12 марта 10:30" in html
    assert 'href="https://telegra.ph/test-03-12"' in html
    assert "❌ 26 марта 10:30" in html
    assert "⏸ 29 марта 14:00" in html
    assert 'href="https://telegra.ph/test-03-29"' in html
    assert "и ещё 2" in html


@pytest.mark.asyncio
async def test_source_summary_hides_other_dates_for_exhibitions() -> None:
    import main

    summary = main.SourcePageEventSummary(
        date="2026-03-07",
        end_date="2026-03-29",
        end_date_is_inferred=False,
        event_type="выставка",
        is_free=False,
        other_dates=[
            main.RelatedEventDate(
                date="2026-03-12",
                time="10:30",
                url="https://telegra.ph/test-03-12",
            ),
        ],
        other_dates_more=0,
    )

    html = await main._build_source_summary_block(summary)
    assert "Другие даты" not in html

