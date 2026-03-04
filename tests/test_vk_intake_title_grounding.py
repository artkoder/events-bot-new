from __future__ import annotations

from datetime import datetime

import pytest

import main
import vk_intake


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_replaces_suspicious_title_tokens(monkeypatch):
    async def fake_parse(*_args, **_kwargs):
        return [
            {
                "title": "🔬 Энергия Утя",
                "short_description": "Интерактивная экспозиция о разных технологиях и энергиях.",
                "date": "2026-02-14",
                "time": "10:00",
                "location_name": "Музей Мирового океана",
                "event_type": "выставка",
                "emoji": "🔬",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        'Там — билеты в "Планету Океан"! Форматы посещения.',
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert len(drafts) == 1
    assert "Утя" not in (drafts[0].title or "")
    assert "Музей" in (drafts[0].title or "")


@pytest.mark.asyncio
async def test_build_event_drafts_from_vk_injects_standup_hint_into_llm_prompt(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_parse(*_args, **_kwargs):
        captured["prompt"] = str(_args[0] if _args else "")
        return [
            {
                "title": "🎤 Медитации для женщин с Олегом Ашуровым",
                "short_description": "Стендап-концерт с комиком Олегом Ашуровым.",
                "date": "2026-02-20",
                "time": "19:00",
                "location_name": "Тестовая площадка",
                "event_type": "концерт",
                "emoji": "🎤",
                "ticket_link": "",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _festival = await vk_intake.build_event_drafts_from_vk(
        "Стендап-концерт. Медитации для женщин с Олегом Ашуровым.\nНачало 20 февраля в 19:00.",
        publish_ts=datetime.now(main.LOCAL_TZ),
    )
    assert len(drafts) == 1
    assert "Если это стендап/комедия" in captured.get("prompt", "")
