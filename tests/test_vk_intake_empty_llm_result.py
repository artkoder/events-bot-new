from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_returns_empty_instead_of_raising(monkeypatch):
    async def fake_parse(*_args, **_kwargs):
        return []  # no events, no festival payload

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    drafts, festival = await vk_intake.build_event_drafts_from_vk(
        "Текст без событий",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert drafts == []
    assert festival is None

