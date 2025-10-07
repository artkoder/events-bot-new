import asyncio
import types
from datetime import date

import pytest

import main
import vk_intake
from db import Database


def _record(value: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(date=value)


def test_holiday_date_range_single_dot():
    record = _record("31.10")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-31"
    assert end == "2024-10-31"


def test_holiday_date_range_dot_range():
    record = _record("31.10-02.11")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-31"
    assert end == "2024-11-02"


def test_holiday_date_range_long_dash():
    record = _record("31.10 — 02.11")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-31"
    assert end == "2024-11-02"


def test_holiday_date_range_textual_months():
    record = _record("31 октября — 2 ноября")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-31"
    assert end == "2024-11-02"


def test_holiday_date_range_textual_same_month():
    record = _record("1-2 мая")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-05-01"
    assert end == "2024-05-02"


def test_holiday_date_range_rollover_next_year():
    record = _record("25.12-07.01")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-12-25"
    assert end == "2025-01-07"


def test_holiday_date_range_legacy_single_mm_dd():
    record = _record("10-31")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-31"
    assert end == "2024-10-31"


@pytest.mark.asyncio
async def test_persist_event_passes_holiday_range_to_ensure(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_assign(event):
        return [], len(event.description or ""), "", False

    scheduled: list[str] = []

    async def fake_schedule(db_obj, event_obj, drain_nav: bool = True, skip_vk_sync: bool = False):
        scheduled.append(event_obj.festival)
        return {}

    async def fake_rebuild(*_args, **_kwargs):
        return False

    sync_calls: list[str] = []

    async def fake_sync_page(db_obj, name: str):
        sync_calls.append(name)

    captured: dict[str, str | None] = {}

    async def fake_ensure(db_obj, name: str | None = None, **kwargs):
        captured["name"] = name
        captured["start_date"] = kwargs.get("start_date")
        captured["end_date"] = kwargs.get("end_date")
        captured["description"] = kwargs.get("description")
        captured["aliases"] = kwargs.get("aliases")
        return types.SimpleNamespace(name=name), True, True

    record = types.SimpleNamespace(
        date="31 октября — 2 ноября",
        canonical_name="Тестовый фестиваль",
        description="Описание",
        normalized_aliases=("тест",),
    )

    monkeypatch.setattr(main, "assign_event_topics", fake_assign)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)
    monkeypatch.setattr(main, "ensure_festival", fake_ensure)
    monkeypatch.setattr(main, "get_holiday_record", lambda value: record)

    draft = vk_intake.EventDraft(
        title="Праздник",
        date="2025-10-30",
        time="20:00",
        festival="Тестовый фестиваль",
        source_text="text",
    )

    await vk_intake.persist_event_and_pages(draft, [], db)
    await asyncio.sleep(0)

    current_year = date.today().year
    assert captured["name"] == "Тестовый фестиваль"
    assert captured["start_date"] == f"{current_year}-10-31"
    assert captured["end_date"] == f"{current_year}-11-02"
    assert captured["description"] == "Описание"
    assert captured["aliases"] == ["тест"]

    assert scheduled == ["Тестовый фестиваль"]
    assert sync_calls == ["Тестовый фестиваль"]


