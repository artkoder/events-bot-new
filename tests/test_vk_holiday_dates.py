import types
from datetime import date

import pytest

import main
import smart_event_update as smart_event_update_mod
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


def test_holiday_date_range_partial_numeric_month():
    record = _record("20-31.10")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-20"
    assert end == "2024-10-31"


def test_holiday_date_range_partial_textual_month():
    record = _record("20 -31 октября")
    start, end = vk_intake._holiday_date_range(record, 2024)
    assert start == "2024-10-20"
    assert end == "2024-10-31"


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


def test_holiday_date_range_movable_maslenitsa():
    record = _record("movable:maslenitsa")
    start, end = vk_intake._holiday_date_range(record, 2026)
    assert start == "2026-02-16"
    assert end == "2026-02-22"


def test_event_date_matches_holiday_with_tolerance_before():
    record = _record("31.10")
    assert vk_intake._event_date_matches_holiday(record, "2025-10-30", None, 1)


def test_event_date_matches_holiday_with_tolerance_after():
    record = _record("31.10")
    assert vk_intake._event_date_matches_holiday(record, "2025-11-01", None, 1)


def test_event_date_matches_cross_year_within_tolerance():
    record = _record("25.12-07.01")
    assert vk_intake._event_date_matches_holiday(record, "2025-01-08", None, 1)


def test_event_date_outside_tolerance_is_skipped():
    record = _record("31.10")
    assert not vk_intake._event_date_matches_holiday(record, "2025-10-28", None, 1)


def test_event_date_matches_partial_range_with_tolerance():
    record = _record("20-31.10")
    assert vk_intake._event_date_matches_holiday(record, "2024-10-19", None, 1)
    assert vk_intake._event_date_matches_holiday(record, "2024-11-01", None, 1)


@pytest.mark.asyncio
async def test_smart_update_passes_holiday_range_to_ensure(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    monkeypatch.setenv("REGION_FILTER_ENABLED", "0")
    monkeypatch.setattr(smart_event_update_mod, "SMART_UPDATE_LLM_DISABLED", True)

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
        tolerance_days=1,
        canonical_name="Тестовый фестиваль",
        description="Описание",
        normalized_aliases=("тест",),
    )

    monkeypatch.setattr(main, "ensure_festival", fake_ensure)
    monkeypatch.setattr(main, "get_holiday_record", lambda value: record)

    candidate = smart_event_update_mod.EventCandidate(
        source_type="telegram",
        source_url="test://holiday/1",
        source_text="text",
        raw_excerpt="text",
        title="Праздник",
        date="2025-10-30",
        time="20:00",
        location_name="Калининград",
        city="Калининград",
        festival="Тестовый фестиваль",
    )

    res = await smart_event_update_mod.smart_event_update(
        db,
        candidate,
        check_source_url=False,
        schedule_tasks=False,
    )
    assert res.status == "created"

    assert captured["name"] == "Тестовый фестиваль"
    assert captured["start_date"] == "2025-10-31"
    assert captured["end_date"] == "2025-11-02"
    assert captured["description"] == "Описание"
    assert captured["aliases"] == ["тест"]
