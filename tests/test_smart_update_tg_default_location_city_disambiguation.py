from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_tg_default_location_city_disambiguation_overrides_to_extracted_city(
    monkeypatch,
) -> None:
    import smart_event_update as seu

    calls = {"n": 0}

    async def fake_ask(prompt, schema, *, max_tokens: int, label: str):
        _ = (prompt, schema, max_tokens)
        calls["n"] += 1
        assert label == "tg_city_disambiguation"
        return {
            "decision": "extracted",
            "confidence": 0.85,
            "reason_short": "Явно указаны площадка/адрес в Москве.",
        }

    monkeypatch.setattr(seu, "_ask_gemma_json", fake_ask)

    cand = seu.EventCandidate(
        source_type="telegram",
        source_url="https://t.me/test/1",
        source_text="Концерт пройдёт в Москве, площадка: Клуб X.",
        title="Event",
        date="2026-03-07",
        time="20:00",
        location_name="Заря, Мира 41-43, Калининград",
        location_address=None,
        city="Калининград",
        metrics={
            "tg_default_location": "Заря, Мира 41-43, Калининград",
            "tg_default_city": "Калининград",
            "tg_extracted_city": "Москва",
            "tg_extracted_location_name": "Клуб X, Москва",
            "tg_extracted_location_address": "Москва, ул. Пример 1",
        },
    )

    await seu._maybe_disambiguate_telegram_default_location_city(cand)
    assert calls["n"] == 1
    assert cand.city == "Москва"
    assert cand.location_name == "Клуб X, Москва"
    assert cand.location_address == "Москва, ул. Пример 1"


@pytest.mark.asyncio
async def test_tg_default_location_city_disambiguation_keeps_default_on_low_confidence(
    monkeypatch,
) -> None:
    import smart_event_update as seu

    async def fake_ask(prompt, schema, *, max_tokens: int, label: str):
        _ = (prompt, schema, max_tokens, label)
        return {
            "decision": "extracted",
            "confidence": 0.55,
            "reason_short": "Не уверен, город может быть контекстом.",
        }

    monkeypatch.setattr(seu, "_ask_gemma_json", fake_ask)

    cand = seu.EventCandidate(
        source_type="telegram",
        source_url="https://t.me/test/2",
        source_text="На сцене — актёры (г. Москва).",
        title="Event",
        date="2026-03-07",
        time="20:00",
        location_name="Заря, Мира 41-43, Калининград",
        city="Калининград",
        metrics={
            "tg_default_location": "Заря, Мира 41-43, Калининград",
            "tg_default_city": "Калининград",
            "tg_extracted_city": "Москва",
        },
    )

    await seu._maybe_disambiguate_telegram_default_location_city(cand)
    assert cand.city == "Калининград"

