import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main


def test_event_topic_prompt_mentions_topics():
    prompt = main.EVENT_TOPIC_SYSTEM_PROMPT
    for key, label in main.TOPIC_LABELS.items():
        assert key in prompt
        assert label in prompt
    assert "Бесплатно" in prompt
    assert "Фестивали" in prompt


@pytest.mark.asyncio
async def test_classify_event_topics_filters_and_limits(monkeypatch):
    monkeypatch.setenv("FOUR_O_MINI", "1")
    captured: dict[str, object] = {}

    async def fake_ask(text, **kwargs):
        captured["text"] = text
        captured["kwargs"] = kwargs
        return json.dumps(
            {
                "topics": [
                    "Музыка",
                    "неизвестная",
                    "искусство",
                    "кино",
                    "музыка",
                    "урбанистика",
                ]
            }
        )

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(
        title="Большой концерт",
        description="Подробное описание мероприятия.",
        source_text="Анонс события #музыка #концерт",
        location_name="Главная сцена",
        location_address="Невский проспект, 1",
        city="Санкт-Петербург",
    )

    result = await main.classify_event_topics(event)

    assert result == ["музыка", "искусство", "кино"]
    assert "#музыка" in captured["text"]
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["system_prompt"] == main.EVENT_TOPIC_SYSTEM_PROMPT
    enum_values = kwargs["response_format"]["json_schema"]["schema"]["properties"]["topics"]["items"][
        "enum"
    ]
    assert sorted(enum_values) == sorted(main.TOPIC_LABELS.keys())


@pytest.mark.asyncio
async def test_classify_event_topics_handles_invalid_json(monkeypatch):
    async def fake_ask(*args, **kwargs):
        return "{invalid_json}"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(
        title="Лекция",
        description="Интересное событие",
        source_text="",
        location_name="",
        location_address="",
        city="",
    )

    result = await main.classify_event_topics(event)

    assert result == []


@pytest.mark.asyncio
async def test_classify_event_topics_handles_exception(monkeypatch):
    async def fake_ask(*args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    event = SimpleNamespace(
        title="Мастер-класс",
        description="",
        source_text="",
        location_name="",
        location_address="",
        city="",
    )

    result = await main.classify_event_topics(event)

    assert result == []
