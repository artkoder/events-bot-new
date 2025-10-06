import pytest

import main


def test_four_o_pitch_prompt_contains_new_guidance():
    prompt = main.FOUR_O_PITCH_PROMPT
    assert "любопыт" in prompt
    assert "гипербол" in prompt
    assert "эмодзи" in prompt
    assert "огранич" in prompt
    assert "следуй инструкциям оператора" in prompt.lower()


def test_four_o_editor_prompt_mentions_constraints():
    prompt = main.FOUR_O_EDITOR_PROMPT
    assert "инструкц" in prompt
    assert "безусловно" in prompt
    assert "опустить" in prompt or "опусти" in prompt.lower()


@pytest.mark.asyncio
async def test_compose_story_pitch_via_4o_fallback_on_error(monkeypatch):
    async def fake_ask(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    text = "Первая строка\nВторая строка"

    result = await main.compose_story_pitch_via_4o(text, title="История")

    assert result == "Первая строка"


@pytest.mark.asyncio
async def test_compose_story_pitch_via_4o_fallback_on_empty(monkeypatch):
    async def fake_ask(*_args, **_kwargs):
        return "  \n  "

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    text = "\n Лидовая строка\nПродолжение"

    result = await main.compose_story_pitch_via_4o(text)

    assert result == "Лидовая строка"


@pytest.mark.asyncio
async def test_compose_story_pitch_via_4o_includes_instructions(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_ask(prompt_text, *, system_prompt=None, max_tokens=None, **_kwargs):
        captured["prompt"] = prompt_text
        return "Готовый ответ"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    result = await main.compose_story_pitch_via_4o(
        "Первый абзац", instructions="Не использовать эмодзи"
    )

    assert result == "Готовый ответ"
    prompt = captured["prompt"]
    assert "Дополнительные инструкции редактору" in prompt
    assert "Не использовать эмодзи" in prompt


@pytest.mark.asyncio
async def test_compose_story_editorial_via_4o_includes_instructions(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_ask(prompt_text, *, system_prompt=None, max_tokens=None, **_kwargs):
        captured["prompt"] = prompt_text
        return "<p>Текст</p>"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    result = await main.compose_story_editorial_via_4o(
        "История", instructions="Оставить теги <b>"
    )

    assert result == "<p>Текст</p>"
    prompt = captured["prompt"]
    assert "Дополнительные инструкции редактору" in prompt
    assert "Оставить теги <b>" in prompt
