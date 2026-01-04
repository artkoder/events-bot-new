import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from main import (
    Database, 
    Event, 
    JobOutbox, 
    JobTask, 
    JobStatus, 
    update_telegraph_event_page,
    update_month_pages_for
)
from sqlalchemy import select

@pytest.mark.asyncio
async def test_skip_immediate_rebuild_if_deferred_exists(tmp_path):
    """
    Verify that update_telegraph_event_page skips calling update_month_pages_for
    if a deferred month_pages job exists for the same month.
    """
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    
    # Setup: Create an event
    async with db.get_session() as session:
        ev = Event(
            title="Test Event",
            description="Test Description",
            date="2026-05-15",
            time="10:00",
            location_name="Location",
            source_text="src",
            telegraph_url="https://telegra.ph/existing",
            content_hash="old_hash"
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        event_id = ev.id

    # Mock dependencies
    # We need to mock build_source_page_content, telegraph, etc. to avoid external calls
    # but strictly speaking we only care about update_month_pages_for call.
    
    with patch("main.build_source_page_content") as mock_build, \
         patch("main.Telegraph", new_callable=MagicMock) as mock_tg_cls, \
         patch("main.update_month_pages_for", new_callable=AsyncMock) as mock_update_month, \
         patch("main.get_telegraph_token", return_value="fake_token"), \
         patch("main.content_hash", return_value="new_hash_diff"), \
         patch("main.telegraph_create_page", new_callable=AsyncMock) as mock_create:
            
        mock_build.return_value = ("<p>Hello</p>", [], [])
        mock_create.return_value = {"url": "https://t.ph/new", "path": "new_path"}
        
        # Case 1: No deferred job -> Should call update_month_pages_for
        await update_telegraph_event_page(event_id, db, None)
        mock_update_month.assert_called_once()
        mock_update_month.reset_mock()
        
        # Case 2: Deferred job exists -> Should SKIP update_month_pages_for
        # Reset hash to force update
        async with db.get_session() as session:
            ev = await session.get(Event, event_id)
            ev.content_hash = "reset_hash"
            session.add(ev)
            # Add deferred job
            future = datetime.now(timezone.utc) + timedelta(minutes=15)
            job = JobOutbox(
                event_id=event_id,
                task=JobTask.month_pages,
                status=JobStatus.pending,
                coalesce_key="month_pages:2026-05",
                next_run_at=future,
                updated_at=datetime.now(timezone.utc)
            )
            session.add(job)
            await session.commit()
            
        # Run again
        await update_telegraph_event_page(event_id, db, None)
        mock_update_month.assert_not_called()
        
        # Case 3: Deferred job exists but executed/failed (not pending) -> Should call
        async with db.get_session() as session:
            ev = await session.get(Event, event_id)
            ev.content_hash = "reset_hash_2"
            session.add(ev)
            
            jobs = (await session.execute(select(JobOutbox))).scalars().all()
            for j in jobs:
                j.status = JobStatus.done
                session.add(j)
            await session.commit()
            
        mock_update_month.reset_mock()
        await update_telegraph_event_page(event_id, db, None)
        mock_update_month.assert_called_once()

