from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_injects_giveaway_prize_hint(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_parse_event_via_llm(text: str, *args, **kwargs):
        captured["text"] = text
        return []

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse_event_via_llm)

    drafts, festival = await vk_intake.build_event_drafts_from_vk(
        (
            "РОЗЫГРЫШ билетов на матч «Балтика» — «ЦСКА».\n"
            "Главный приз — два билета на матч, который состоится 14 марта на Ростех Арене.\n"
            "Для участия подпишись, поставь лайк и напиши комментарий.\n"
            "Итоги подведём 10 марта."
        ),
        publish_ts=datetime.now(main.LOCAL_TZ),
    )

    assert drafts == []
    assert festival is None
    assert "только как приз" in (captured.get("text") or "")
    assert "верни `[]`" in (captured.get("text") or "")


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_does_not_inject_giveaway_prize_hint_for_regular_post(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_parse_event_via_llm(text: str, *args, **kwargs):
        captured["text"] = text
        return []

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse_event_via_llm)

    drafts, festival = await vk_intake.build_event_drafts_from_vk(
        "14 марта в 19:00 в Ростех Арене пройдёт матч «Балтика» — «ЦСКА».",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )

    assert drafts == []
    assert festival is None
    assert "только как приз" not in (captured.get("text") or "")
