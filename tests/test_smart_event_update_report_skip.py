from __future__ import annotations

import pytest
from sqlalchemy import select

from db import Database
from models import Event
from smart_event_update import EventCandidate, smart_event_update


@pytest.mark.asyncio
async def test_smart_update_skips_completed_event_report_from_vk(tmp_path) -> None:
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    candidate = EventCandidate(
        source_type="vk",
        source_url="https://vk.com/wall-2051396_23431",
        source_text=(
            "💥 Один на один с учениками МАОУ СОШ № 33\n\n"
            "Мы отправились к ребятам, чтобы поговорить о самом важном – о выборе будущего. "
            "И сделали это в формате профориентационного квиза «Востребованные профессии».\n\n"
            "Вместе с учениками 9-11 классов мы:\n"
            "🔍 Исследовали современные и перспективные профессиональные направления.\n"
            "🧠 Решали практические задачи, где пригодилась и логика, и смекалка.\n"
            "🤝 Работали в командах – ведь умение договариваться и быстро принимать решения пригодится в любой сфере.\n\n"
            "Было здорово видеть горящие глаза ребят, их вовлечённость и неподдельный интерес к теме.\n\n"
            "Огромное спасибо администрации и педагогам 33 школы за тёплый приём и сотрудничество. "
            "И, конечно, скоро увидимся вновь, ведь это не последняя наша встреча!"
        ),
        raw_excerpt="Профориентационный квиз для учеников 9-11 классов школы № 33.",
        title="Профориентационная игра с «Ораторами России»",
        date="2026-03-14",
        city="Калининград",
        location_name="МАОУ СОШ № 33",
        trust_level="medium",
    )

    result = await smart_event_update(
        db,
        candidate,
        check_source_url=True,
        schedule_tasks=False,
    )
    assert result.status == "skipped_non_event"
    assert result.reason == "completed_event_report"

    async with db.get_session() as session:
        rows = (await session.execute(select(Event))).scalars().all()
        assert rows == []
