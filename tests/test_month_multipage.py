"""Tests for multi-page month split functionality."""
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import main
from db import Database
from models import Event, MonthPage


@pytest.mark.asyncio
async def test_build_month_page_content_page_1_has_intro(tmp_path: Path, monkeypatch):
    """Page 1 should include the intro paragraph."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 1, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 1, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    month = "2026-01"
    events = [
        Event(
            id=1,
            title="Test Event",
            description="Desc",
            source_text="src",
            date="2026-01-15",
            time="18:00",
            location_name="Hall",
            added_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    ]

    title, content, size = await main.build_month_page_content(
        db, month, events, [], page_number=1
    )

    # Check title is full format
    assert "полный анонс" in title
    
    # Check intro paragraph is present
    from telegraph.utils import nodes_to_html
    html = nodes_to_html(content)
    assert "Планируйте свой месяц" in html


@pytest.mark.asyncio
async def test_build_month_page_content_page_2_no_intro(tmp_path: Path, monkeypatch):
    """Page 2+ should NOT include the intro paragraph."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 1, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 1, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    month = "2026-01"
    events = [
        Event(
            id=1,
            title="Test Event",
            description="Desc",
            source_text="src",
            date="2026-01-15",
            time="18:00",
            location_name="Hall",
            added_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    ]

    title, content, size = await main.build_month_page_content(
        db, month, events, [],
        page_number=2,
        first_date=date(2026, 1, 15),
        last_date=date(2026, 1, 25),
    )

    # Check title uses date range format
    assert "с 15 по 25" in title
    assert "января" in title
    assert "2026" in title
    
    # Check intro paragraph is NOT present
    from telegraph.utils import nodes_to_html
    html = nodes_to_html(content)
    assert "Планируйте свой месяц" not in html


@pytest.mark.asyncio
async def test_build_month_page_content_page_2_fallback_title(tmp_path: Path, monkeypatch):
    """Page 2+ without dates should use fallback title."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 1, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 1, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)

    month = "2026-01"
    events = [
        Event(
            id=1,
            title="Test Event",
            description="Desc",
            source_text="src",
            date="2026-01-15",
            time="18:00",
            location_name="Hall",
            added_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    ]

    # No first_date/last_date provided
    title, content, size = await main.build_month_page_content(
        db, month, events, [], page_number=2
    )

    # Check title uses fallback format
    assert "продолжение" in title


@pytest.mark.asyncio
async def test_build_month_page_content_debug_prefix(tmp_path: Path, monkeypatch):
    """In DEBUG mode, titles should have ТЕСТ prefix."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FakeDate(date):
        @classmethod
        def today(cls) -> date:
            return date(2026, 1, 1)

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return datetime(2026, 1, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(main, "date", FakeDate)
    monkeypatch.setattr(main, "datetime", FakeDatetime)
    monkeypatch.setenv("EVBOT_DEBUG", "1")

    month = "2026-01"
    events = [
        Event(
            id=1,
            title="Test Event",
            description="Desc",
            source_text="src",
            date="2026-01-15",
            time="18:00",
            location_name="Hall",
            added_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    ]

    title, _, _ = await main.build_month_page_content(
        db, month, events, [], page_number=1
    )

    # Check title has TEST prefix
    assert title.startswith("ТЕСТ ")

