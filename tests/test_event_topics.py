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
    assert "Калининградской области" in prompt
    assert "краеведении" in prompt


def test_topic_labels_include_theatre_subtypes():
    assert main.TOPIC_LABELS["THEATRE_CLASSIC"] == "Классический театр и драма"
    assert (
        main.TOPIC_LABELS["THEATRE_MODERN"]
        == "Современный и экспериментальный театр"
    )


def test_topic_labels_include_fashion():
    assert main.TOPIC_LABELS["FASHION"] == "Мода и стиль"


def test_topic_labels_include_kaliningrad_local_history():
    assert (
        main.TOPIC_LABELS["KRAEVEDENIE_KALININGRAD_OBLAST"]
        == "Краеведение Калининградской области"
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
        "мода": "FASHION",
        "fashion": "FASHION",
        "Fashion Week": "FASHION",
        "показ мод": "FASHION",
        "fashion show": "FASHION",
        "styling": "FASHION",
        "стиль": "FASHION",
        "краеведение": "KRAEVEDENIE_KALININGRAD_OBLAST",
        "локальная история": "KRAEVEDENIE_KALININGRAD_OBLAST",
        "калининградская область": "KRAEVEDENIE_KALININGRAD_OBLAST",
        "Калининград": "KRAEVEDENIE_KALININGRAD_OBLAST",
        "URBANISM": "KRAEVEDENIE_KALININGRAD_OBLAST",
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


@pytest.mark.asyncio
async def test_assign_event_topics_adds_kaliningrad_topic(monkeypatch):
    async def fake_classify(event):
        return ["LECTURES"]

    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    event = SimpleNamespace(
        title="Локальная история",
        description="Встреча об урбанистике",
        source_text="",
        location_name="Музей",
        location_address="Калининград, Ленинский проспект, 1",
        city="Калининград",
        topics=[],
        topics_manual=False,
    )

    topics, text_length, error_text, manual = await main.assign_event_topics(event)

    assert "KRAEVEDENIE_KALININGRAD_OBLAST" in topics
    assert topics[0] == "LECTURES"
    assert text_length > 0
    assert error_text is None
    assert manual is False


@pytest.mark.asyncio
async def test_assign_event_topics_detects_region_via_hashtags(monkeypatch):
    async def fake_classify(event):
        return []

    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    event = SimpleNamespace(
        title="",
        description="Разговор про #урбанистика и #Калининград",
        source_text="",
        location_name="",
        location_address="",
        city="",
        topics=[],
        topics_manual=False,
    )

    topics, _, _, _ = await main.assign_event_topics(event)

    assert topics == ["KRAEVEDENIE_KALININGRAD_OBLAST"]


@pytest.mark.asyncio
async def test_assign_event_topics_keeps_manual_mode(monkeypatch):
    event = SimpleNamespace(
        title="",
        description="",
        source_text="",
        location_name="",
        location_address="",
        city="Калининград",
        topics=["EXHIBITIONS"],
        topics_manual=True,
    )

    topics, text_length, error_text, manual = await main.assign_event_topics(event)

    assert topics == ["EXHIBITIONS"]
    assert text_length == 0
    assert error_text is None
    assert manual is True


@pytest.mark.asyncio
async def test_assign_event_topics_does_not_duplicate_topic(monkeypatch):
    async def fake_classify(event):
        return ["KRAEVEDENIE_KALININGRAD_OBLAST"]

    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    event = SimpleNamespace(
        title="История Калининграда",
        description="",
        source_text="",
        location_name="",
        location_address="",
        city="Калининград",
        topics=[],
        topics_manual=False,
    )

    topics, _, _, _ = await main.assign_event_topics(event)

    assert topics == ["KRAEVEDENIE_KALININGRAD_OBLAST"]
