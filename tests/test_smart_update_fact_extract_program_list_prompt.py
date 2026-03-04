from __future__ import annotations

import pytest

import smart_event_update as su
from smart_event_update import EventCandidate


VK_FILM_PROGRAM_TEXT = """
Не знаешь, чем заняться в четверг? 😏

Тогда приглашаем тебя на показ короткометражного кино.

Программа «Звёзды в коротком метре» (84 мин.):
«Замуж второй раз» — Радда Новикова
«Не Дед Мороз» — Степан Азарян
«Про корову» — Николай Алексеев и Антон Симухин
«Фак Ап» — Александра Розовская
«Я не помню» — Евгений Цыганов
""".strip()


@pytest.mark.asyncio
async def test_llm_extract_candidate_facts_prompt_instructs_compact_program_lists(monkeypatch) -> None:
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    captured = {"prompt": ""}

    async def _fake_json(prompt: str, schema: dict, *, max_tokens: int, label: str):  # noqa: ANN001,ARG001
        captured["prompt"] = prompt
        return {
            "facts": [
                "Программа «Звёзды в коротком метре» (84 мин.)",
                "«Замуж второй раз» — Радда Новикова",
                "«Я не помню» — Евгений Цыганов",
            ]
        }

    monkeypatch.setattr(su, "_ask_gemma_json", _fake_json)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-212598834_532",
        source_text=VK_FILM_PROGRAM_TEXT,
        title="Звёзды в коротком метре — Дни короткометражного кино",
        date="2026-03-12",
        time="19:00",
        city="Калининград",
    )

    facts = await su._llm_extract_candidate_facts(candidate)
    assert su.SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE in captured["prompt"]
    assert "«Замуж второй раз» — Радда Новикова" in facts
    assert "«Я не помню» — Евгений Цыганов" in facts


@pytest.mark.asyncio
async def test_create_bundle_prompt_includes_compact_program_rule(monkeypatch) -> None:
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    captured = {"prompt": ""}

    async def _fake_json(prompt: str, schema: dict, *, max_tokens: int, label: str):  # noqa: ANN001,ARG001
        captured["prompt"] = prompt
        return {
            "title": "Тест",
            "description": "Описание.",
            "facts": ["Факт 1"],
            "search_digest": "Короткий дайджест без логистики.",
            "short_description": "Короткое нейтральное описание события без лишних деталей, одним предложением.",
        }

    monkeypatch.setattr(su, "_ask_gemma_json", _fake_json)

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-1_1",
        source_text=VK_FILM_PROGRAM_TEXT,
        raw_excerpt="",
        title="Тест",
        date="2026-03-12",
        time="19:00",
        location_name="Дом молодёжи",
        city="Калининград",
    )

    out = await su._llm_create_description_facts_and_digest(
        candidate,
        clean_title="Тест",
        clean_source_text=candidate.source_text,
        clean_raw_excerpt=candidate.raw_excerpt,
        normalized_event_type="кинопоказ",
    )
    assert out is not None
    assert su.SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE in captured["prompt"]
