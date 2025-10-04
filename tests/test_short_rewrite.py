import os, sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main


@pytest.mark.asyncio
async def test_build_short_vk_text_uses_description_when_source_missing(monkeypatch):
    calls = {}

    async def fake_ask(prompt, **kwargs):
        calls["prompt"] = prompt
        calls["system_prompt"] = kwargs.get("system_prompt")
        return "Сжатый текст. Второе предложение."

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(
        description="Длинное описание события. Подробности о программе и времени.",
        title="Название события",
    )

    result = await main.build_short_vk_text(
        event,
        "",
        max_sentences=2,
        poster_texts=["OCR блок"],
    )

    assert "Длинное описание события" in calls["prompt"]
    assert "OCR блок" in calls["prompt"]
    assert "в первой строке не повторяй название" in calls["prompt"]
    assert "Название проекта или события можно упомянуть позже" in calls["prompt"]
    assert "в первой строке не повторяй название" in calls["system_prompt"]
    assert result == "Сжатый текст. Второе предложение."


@pytest.mark.asyncio
async def test_build_short_vk_text_falls_back_to_title(monkeypatch):
    async def fake_ask(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("ask_4o should not be called when only title is available")

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(description="", title="Лаконичное событие")

    result = await main.build_short_vk_text(event, "", max_sentences=3)

    assert result == "Лаконичное событие"


@pytest.mark.asyncio
async def test_build_short_vk_text_handles_missing_text_reply(monkeypatch):
    async def fake_ask(prompt, **kwargs):
        return "Пожалуйста, предоставьте текст, который нужно сократить."

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(description="", title="Название события")
    original = "Первое предложение. Второе предложение. Третье предложение."

    result = await main.build_short_vk_text(event, original, max_sentences=3)

    assert result == "Первое предложение. Второе предложение."


@pytest.mark.asyncio
async def test_build_short_vk_text_preserves_paragraphs(monkeypatch):
    async def fake_ask(prompt, **kwargs):
        return (
            "Первое предложение. Второе предложение.\n\n"
            "Третье предложение. Четвертое предложение."
        )

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(description="Описание события", title="Название события")

    result = await main.build_short_vk_text(event, "Исходный текст", max_sentences=3)

    assert result == "Первое предложение. Второе предложение.\n\nТретье предложение."
    assert "\n\n" in result
