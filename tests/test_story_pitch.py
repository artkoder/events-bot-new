import pytest

import main


def test_four_o_pitch_prompt_contains_new_guidance():
    prompt = main.FOUR_O_PITCH_PROMPT
    assert "любопыт" in prompt
    assert "гипербол" in prompt
    assert "эмодзи" in prompt


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
