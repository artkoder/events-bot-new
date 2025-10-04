import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
import models


def test_event_topic_prompt_mentions_topics():
    prompt = main.EVENT_TOPIC_SYSTEM_PROMPT
    for key, label in main.TOPIC_LABELS.items():
        assert key in prompt
        assert label in prompt
    assert "Бесплатно" in prompt
    assert "Фестивали" in prompt
    assert "ярмарк" in prompt.casefold()
    assert "пьесы классических авторов" in prompt
    assert any(name in prompt for name in ("Шекспир", "Мольер", "Пушкин", "Гоголь"))
    assert "исторические или мифологические сюжеты" in prompt
    assert "новой драме" in prompt
    assert "экспериментальным, иммерсивным" in prompt
    assert "ставь обе темы" in prompt


def test_topic_labels_include_theatre_subtypes():
    assert main.TOPIC_LABELS["THEATRE_CLASSIC"] == "Классический театр и драма"
    assert (
        main.TOPIC_LABELS["THEATRE_MODERN"]
        == "Современный и экспериментальный театр"
    )


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
                    "HISTORICAL_IMMERSION",
                    "неизвестная",
                    "art",
                    "CINEMA",
                    "MUSIC",
                    "URBANISM",
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

    assert result == ["HISTORICAL_IMMERSION", "EXHIBITIONS", "MOVIES"]
    assert "#музыка" in captured["text"]
    kwargs = captured["kwargs"]
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["system_prompt"] == main.EVENT_TOPIC_SYSTEM_PROMPT
    enum_values = kwargs["response_format"]["json_schema"]["schema"]["properties"]["topics"]["items"][
        "enum"
    ]
    assert enum_values == list(main.TOPIC_LABELS.keys())
    assert "HISTORICAL_IMMERSION" in enum_values


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


def test_normalize_topic_identifier_legacy_aliases():
    cases = {
        "handmade": "HANDMADE",
        "Нетворкинг": "NETWORKING",
        "спорт": "ACTIVE",
        "Personalities": "PERSONALITIES",
        "дети": "KIDS_SCHOOL",
        "семейные": "FAMILY",
        "психология": "PSYCHOLOGY",
        "Psychology": "PSYCHOLOGY",
        "mental health": "PSYCHOLOGY",
        "классический спектакль": "THEATRE_CLASSIC",
        "Драма": "THEATRE_CLASSIC",
        "современный театр": "THEATRE_MODERN",
        "experimental theatre": "THEATRE_MODERN",
        "средневековье": "HISTORICAL_IMMERSION",
        "исторические костюмы": "HISTORICAL_IMMERSION",
    }
    for raw, expected in cases.items():
        assert models.normalize_topic_identifier(raw) == expected


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
