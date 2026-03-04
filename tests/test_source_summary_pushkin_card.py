from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_source_summary_renders_pushkin_card_only_when_true() -> None:
    import main

    summary_yes = main.SourcePageEventSummary(
        date="2026-03-07",
        time="11:00",
        location_name="Студия",
        pushkin_card=True,
    )
    html_yes = await main._build_source_summary_block(summary_yes)
    assert "Пушкинская карта" in html_yes

    summary_no = main.SourcePageEventSummary(
        date="2026-03-07",
        time="11:00",
        location_name="Студия",
        pushkin_card=False,
    )
    html_no = await main._build_source_summary_block(summary_no)
    assert "Пушкинская карта" not in html_no

