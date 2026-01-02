"""Tests for festivals_fix_nav filtering of past festivals."""
from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from db import Database
from models import Festival, Event


@pytest.mark.asyncio
async def test_festivals_fix_nav_skips_past_festivals_without_future_events(
    tmp_path: Path, monkeypatch
):
    """Festivals with past end_date and no future events should be skipped."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    past_week = (date.today() - timedelta(days=7)).isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    
    async with db.get_session() as session:
        # Past festival with no future events - should be skipped
        past_fest = Festival(
            name="PastFest",
            telegraph_path="past_fest",
            start_date=past_week,
            end_date=yesterday,
        )
        session.add(past_fest)
        
        # Future festival - should be processed
        future_fest = Festival(
            name="FutureFest",
            telegraph_path="future_fest",
            start_date=tomorrow,
            end_date=tomorrow,
        )
        session.add(future_fest)
        await session.commit()
    
    calls = {"tg": [], "vk": []}
    
    async def fake_tg(eid, db_obj, bot_obj):
        calls["tg"].append(eid)
        return main.NavUpdateResult(True, 0, False)
    
    async def fake_vk(eid, db_obj, bot_obj):
        calls["vk"].append(eid)
        return True
    
    monkeypatch.setattr(main, "update_festival_tg_nav", fake_tg)
    monkeypatch.setattr(main, "update_festival_vk_nav", fake_vk)
    
    pages, changed, dup, legacy = await main.festivals_fix_nav(db, None)
    
    # Only FutureFest should be processed (has telegraph_path and is in future)
    assert pages == 1, f"Expected 1 page, got {pages}"
    assert len(calls["tg"]) == 1, f"Expected 1 TG call, got {len(calls['tg'])}"
    assert len(calls["vk"]) == 1, f"Expected 1 VK call, got {len(calls['vk'])}"


@pytest.mark.asyncio
async def test_festivals_fix_nav_includes_past_festival_with_future_event(
    tmp_path: Path, monkeypatch
):
    """Past festival with future event should be processed."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    past_week = (date.today() - timedelta(days=7)).isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    
    async with db.get_session() as session:
        # Past festival end_date but has a future event - should be processed
        past_fest_with_event = Festival(
            name="PastFestWithEvent",
            telegraph_path="past_fest_with_event",
            start_date=past_week,
            end_date=yesterday,
        )
        session.add(past_fest_with_event)
        await session.commit()
        
        # Future event for this festival
        future_event = Event(
            id=1,
            title="Future Event",
            description="Test description",
            location_name="Test Location",
            source_text="Test source",
            festival="PastFestWithEvent",
            date=tomorrow,
            time="19:00",
        )
        session.add(future_event)
        await session.commit()
    
    calls = {"tg": [], "vk": []}
    
    async def fake_tg(eid, db_obj, bot_obj):
        calls["tg"].append(eid)
        return main.NavUpdateResult(True, 0, False)
    
    async def fake_vk(eid, db_obj, bot_obj):
        calls["vk"].append(eid)
        return True
    
    monkeypatch.setattr(main, "update_festival_tg_nav", fake_tg)
    monkeypatch.setattr(main, "update_festival_vk_nav", fake_vk)
    
    pages, changed, dup, legacy = await main.festivals_fix_nav(db, None)
    
    # PastFestWithEvent should be processed because it has a future event
    assert pages == 1, f"Expected 1 page, got {pages}"
    assert len(calls["tg"]) == 1, f"Expected 1 TG call, got {len(calls['tg'])}"
    assert len(calls["vk"]) == 1, f"Expected 1 VK call, got {len(calls['vk'])}"


@pytest.mark.asyncio
async def test_festivals_fix_nav_includes_festival_ending_today(
    tmp_path: Path, monkeypatch
):
    """Festival ending today should be processed."""
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    today = date.today().isoformat()
    past_week = (date.today() - timedelta(days=7)).isoformat()
    
    async with db.get_session() as session:
        # Festival ending today - should be processed
        today_fest = Festival(
            name="TodayFest",
            telegraph_path="today_fest",
            start_date=past_week,
            end_date=today,
        )
        session.add(today_fest)
        await session.commit()
    
    calls = {"tg": [], "vk": []}
    
    async def fake_tg(eid, db_obj, bot_obj):
        calls["tg"].append(eid)
        return main.NavUpdateResult(True, 0, False)
    
    async def fake_vk(eid, db_obj, bot_obj):
        calls["vk"].append(eid)
        return True
    
    monkeypatch.setattr(main, "update_festival_tg_nav", fake_tg)
    monkeypatch.setattr(main, "update_festival_vk_nav", fake_vk)
    
    pages, changed, dup, legacy = await main.festivals_fix_nav(db, None)
    
    # TodayFest should be processed (end_date >= today)
    assert pages == 1, f"Expected 1 page, got {pages}"
    assert len(calls["tg"]) == 1, f"Expected 1 TG call, got {len(calls['tg'])}"
