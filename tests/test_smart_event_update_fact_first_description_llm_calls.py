from __future__ import annotations

import pytest

import smart_event_update as su


@pytest.mark.asyncio
async def test_fact_first_description_makes_bounded_llm_calls(monkeypatch) -> None:
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    calls = {"text": 0, "json": 0}

    async def _fake_text(prompt: str, *, max_tokens: int, label: str, temperature: float = 0.0) -> str:  # noqa: ARG001
        calls["text"] += 1
        if calls["text"] == 1:
            # Bad structure: only one heading + micro-section.
            return "Лид.\n\n### Программа\nКоротко."
        return (
            "Лид одним абзацем.\n\n"
            "### Что будет\n"
            "- Пункт 1\n"
            "- Пункт 2\n\n"
            "### Условия участия\n"
            "Фраза первая. Фраза вторая."
        )

    async def _fake_json(prompt: str, schema: dict, *, max_tokens: int, label: str):  # noqa: ANN001,ARG001
        calls["json"] += 1
        return {"missing": ["Факт 2"], "extra": []}

    monkeypatch.setattr(su, "_ask_gemma_text", _fake_text)
    monkeypatch.setattr(su, "_ask_gemma_json", _fake_json)

    out = await su._llm_fact_first_description_md(
        title="Тест",
        event_type="лекция",
        facts_text_clean=["Факт 1", "Факт 2"],
        anchors=[],
        label="t",
    )
    assert out is not None
    assert "### Что будет" in out
    assert calls["json"] == 1
    assert calls["text"] == 2


@pytest.mark.asyncio
async def test_fact_first_description_allows_one_extra_policy_revise(monkeypatch) -> None:
    monkeypatch.setattr(su, "SMART_UPDATE_LLM_DISABLED", False)

    calls = {"text": 0, "json": 0}

    async def _fake_text(prompt: str, *, max_tokens: int, label: str, temperature: float = 0.0) -> str:  # noqa: ARG001
        calls["text"] += 1
        if calls["text"] == 1:
            return "Лид.\n\n### Программа\nКоротко."
        if calls["text"] == 2:
            # Still bad after first revise: one heading + micro-section.
            return "Лид.\n\n### Что будет\nКоротко."
        return (
            "Лид одним абзацем.\n\n"
            "### Что будет\n"
            "- Пункт 1\n"
            "- Пункт 2\n\n"
            "### Условия участия\n"
            "Фраза первая. Фраза вторая."
        )

    async def _fake_json(prompt: str, schema: dict, *, max_tokens: int, label: str):  # noqa: ANN001,ARG001
        calls["json"] += 1
        return {"missing": [], "extra": []}

    monkeypatch.setattr(su, "_ask_gemma_text", _fake_text)
    monkeypatch.setattr(su, "_ask_gemma_json", _fake_json)

    out = await su._llm_fact_first_description_md(
        title="Тест",
        event_type="лекция",
        facts_text_clean=["Факт 1"],
        anchors=[],
        label="t",
    )
    assert out is not None
    assert "### Условия участия" in out
    assert calls["json"] == 1
    assert calls["text"] == 3

