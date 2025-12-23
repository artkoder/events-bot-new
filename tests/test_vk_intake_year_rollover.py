import pytest
import os
import sys
from datetime import datetime
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import vk_intake
import main

@pytest.mark.asyncio
async def test_year_rollover_when_date_has_passed(monkeypatch):
    # Setup
    local_tz = main.LOCAL_TZ
    # publish date: Dec 23, 2025
    publish_dt = datetime(2025, 12, 23, 20, 0, tzinfo=local_tz)

    # Text implies "7 January"
    text = "❄ 7 января будет весело..."
    operator_extra = "7 января 19:00"

    # LLM returns 2025-01-07 (naive year assumption)
    async def fake_parse(text_in, *args, **kwargs):
        return [
            {
                "title": "New Year Event",
                "date": "2025-01-07",  # Passed date relative to Dec 2025
                "time": "19:00",
                "short_description": "Desc",
                "location_name": "Venue",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    # Act
    drafts, _ = await vk_intake.build_event_drafts_from_vk(
        text,
        operator_extra=operator_extra,
        publish_ts=publish_dt,
    )

    # Assert
    assert len(drafts) == 1
    # Should rollover to 2026 because Jan 7 < Dec 23
    assert drafts[0].date == "2026-01-07"

@pytest.mark.asyncio
async def test_no_rollover_when_explicit_year_in_text(monkeypatch):
    local_tz = main.LOCAL_TZ
    publish_dt = datetime(2025, 12, 23, 20, 0, tzinfo=local_tz)

    text = "Событие 7 января 2025 года (архив)..."

    # LLM returns 2025-01-07
    async def fake_parse(text_in, *args, **kwargs):
        return [
            {
                "title": "Old Event",
                "date": "2025-01-07",
                "time": "19:00",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    drafts, _ = await vk_intake.build_event_drafts_from_vk(
        text,
        publish_ts=publish_dt,
    )

    # Should remain 2025 because year is explicit in text
    assert drafts[0].date == "2025-01-07"

@pytest.mark.asyncio
async def test_rollover_with_hint_application(monkeypatch):
    local_tz = main.LOCAL_TZ
    publish_dt = datetime(2025, 12, 23, 20, 0, tzinfo=local_tz)

    # Hint says 2026-01-07
    hint_dt = datetime(2026, 1, 7, 19, 0, tzinfo=local_tz)
    hint_ts = int(hint_dt.timestamp())

    text = "7 января"

    # LLM returns 2025-01-07
    async def fake_parse(text_in, *args, **kwargs):
        return [
            {
                "title": "Future Event",
                "date": "2025-01-07",
                "time": None, # Time missing
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    drafts, _ = await vk_intake.build_event_drafts_from_vk(
        text,
        publish_ts=publish_dt,
        event_ts_hint=hint_ts,
    )

    # Should use hint year AND hint time
    assert drafts[0].date == "2026-01-07"
    assert drafts[0].time == "19:00"
