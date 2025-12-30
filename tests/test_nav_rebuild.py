from unittest.mock import AsyncMock, patch
import pytest
from datetime import date
from sqlmodel import Session, select
from main import (
    Database,
    Event,
    MonthPage,
    build_month_nav_html,
    sync_month_page,
    refresh_month_nav,
    schedule_event_update_tasks, # We need to test this integration
    JobTask,
)
from sqlmodel.pool import StaticPool
from sqlmodel import create_engine, SQLModel

@pytest.fixture
def db_memory(tmp_path):
    # Use disk file to avoid :memory: issues with multiple connections/threads in aiosqlite
    db_file = tmp_path / "test.db"
    # The actual Database class in db.py takes a path string, not an engine.
    # It manages its own engine internally.
    db = Database(str(db_file))
    
    # Initialize the DB schema manually since we aren't running main.startup
    # We can use the internal engine property or raw connection to create tables if needed.
    # But Database.init() is async.
    # Simplest way for test is to use SQLModel.metadata.create_all with a sync engine first
    # OR rely on the test running init() if possible.
    
    # Let's create tables synchronously for setup
    engine = create_engine(f"sqlite:///{db_file}")
    SQLModel.metadata.create_all(engine)
    
    return db



@pytest.mark.asyncio
async def test_nav_update_on_new_month(db_memory: Database):
    """
    Test that when a NEW month page is created, existing pages get their footer updated.
    """
    # Mock refresh_month_nav to avoid real API calls and check invocation
    with patch("main.refresh_month_nav", new_callable=AsyncMock) as mock_refresh:
        
        # 1. Start with only January 2026
        async with db_memory.get_session() as session:
            ev1 = Event(id=1, title="Jan Event", date="2026-01-01", description="desc", time="12:00", location_name="loc", source_text="src")
            session.add(ev1)
            mp1 = MonthPage(month="2026-01", url="http://jan2026", hash="old_hash", path="path/to/jan")
            session.add(mp1)
            await session.commit()
            
        # 2. Add April event and call schedule_event_update_tasks
        async with db_memory.get_session() as session:
            ev2 = Event(id=2, title="Apr Event", date="2026-04-01", description="desc", time="12:00", location_name="loc", source_text="src")
            session.add(ev2)
            await session.commit()
            
            # Note: We haven't created MonthPage for April yet in DB, 
            # so schedule_event_update_tasks should detect it as NEW (if we check existing MPs).
            # Wait, our logic checks `MonthPage` table.
            # If we don't insert MonthPage for April manually, `schedule_event_update_tasks` sees it's missing.
            # BUT `month_pages` job is what creates it.
            # The check `if not existing_mp` should be True.
            
            await schedule_event_update_tasks(db_memory, ev2)
            
        # 3. Verify that refresh_month_nav was scheduled (called)
        # Since we used asyncio.create_task, it's fire-and-forget.
        # But method patch should capture it.
        # We might need to yield to event loop to let it run if it was awaited.
        # But `create_task` schedules it.
        
        # We assertions must handle that it might be called asynchronously.
        # `mock_refresh.assert_called_once()` might fail if task hasn't started.
        # But usually in test loop it's fast enough or we just check call.
        
        assert mock_refresh.call_count >= 1, "refresh_month_nav should have been triggered for new month"
        assert mock_refresh.await_count >= 1 or mock_refresh.call_count >= 1
        # Depending on how mock works with create_task. 
        # Actually create_task(coro) calls the coro function immediately to get a coroutine object?
        # No, `refresh_month_nav(db)` is called to create the coroutine.
        # So mocks call count increments.
        
@pytest.mark.asyncio
async def test_year_suffix_logic(db_memory: Database):
    # Setup simple data for build_month_nav_html check
    # We re-use logic from previous test but simpler setup
    events = [
        Event(id=1, title="Jan", date="2026-01-01", description="d", time="12:00", location_name="l", source_text="s"),
        Event(id=2, title="Feb", date="2026-02-01", description="d", time="12:00", location_name="l", source_text="s"),
        Event(id=3, title="Jan27", date="2027-01-01", description="d", time="12:00", location_name="l", source_text="s"),
    ]
    async with db_memory.get_session() as session:
        for e in events:
            session.add(e)
            mp = MonthPage(month=e.date[:7], url=f"http://{e.date[:7]}", path="p", hash="h")
            session.add(mp)
        await session.commit()

    html = await build_month_nav_html(db_memory)
    # 2026-01 -> Январь 2026 (first month, has year)
    # 2026-02 -> Февраль (same year, no year)
    # 2027-01 -> Январь 2027 (new year, has year)
    
    assert "Январь 2026" in html
    assert "Февраль</a>" in html
    assert "Февраль 2026" not in html
    assert "Январь 2027" in html
