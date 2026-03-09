from __future__ import annotations

import pytest

import smart_event_update as su
from models import Event
from smart_event_update import EventCandidate


def _candidate_with_text(text: str) -> EventCandidate:
    return EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text=text,
        raw_excerpt=text,
        title="Черновик",
        date="2026-03-08",
        time="22:00",
        location_name="Бар Советов",
        city="Калининград",
    )


def test_title_grounding_rejects_editorial_title_with_single_token_overlap():
    candidate = _candidate_with_text(
        "8 марта в 22:00 в Баре Советов пройдут стендап-вечер и DJ-сет от DJ LUNADON."
    )

    assert (
        su._is_title_grounded_in_candidate_sources(  # type: ignore[attr-defined]
            "8 марта — День раскрепощения женщин",
            candidate,
        )
        is False
    )


def test_title_grounding_accepts_concrete_title_with_multiple_grounded_tokens():
    candidate = _candidate_with_text(
        "8 марта в 22:00 в Баре Советов пройдут стендап-вечер и DJ-сет от DJ LUNADON."
    )

    assert (
        su._is_title_grounded_in_candidate_sources(  # type: ignore[attr-defined]
            "Стендап-вечер и DJ-сет",
            candidate,
        )
        is True
    )


@pytest.mark.asyncio
async def test_match_create_bundle_prompt_blocks_editorial_titles(monkeypatch):
    seen = {}

    async def fake_ask(prompt, _schema, *, max_tokens, label):  # noqa: ANN001
        seen["prompt"] = prompt
        seen["max_tokens"] = max_tokens
        seen["label"] = label
        return {
            "action": "create",
            "match_event_id": None,
            "confidence": 0.2,
            "reason_short": "new",
            "bundle": {
                "title": "Стендап-вечер в Баре Советов",
                "description": "Описание.",
                "facts": [],
                "search_digest": None,
                "short_description": None,
            },
        }

    monkeypatch.setattr(su, "_ask_gemma_json", fake_ask)
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    candidate = _candidate_with_text(
        "8 марта в 22:00 в Баре Советов пройдут стендап-вечер и DJ-сет от DJ LUNADON."
    )
    event = Event(
        id=1,
        title="Старое событие",
        description="Описание.",
        date="2026-03-01",
        time="22:00",
        location_name="Бар Советов",
        city="Калининград",
        source_text="Старый анонс.",
    )

    result = await su._llm_match_or_create_bundle(  # type: ignore[attr-defined]
        candidate,
        [event],
        threshold=0.72,
        clean_title="8 марта",
        clean_source_text=candidate.source_text,
        clean_raw_excerpt=candidate.raw_excerpt,
        normalized_event_type="party",
    )

    assert result is not None
    prompt = seen["prompt"]
    assert "НЕ придумывай тематические, редакционные или идеологические заголовки" in prompt
    assert "Описывай ЧТО происходит (формат, участники, жанр)" in prompt
