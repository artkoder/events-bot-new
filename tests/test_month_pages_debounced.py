import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from main import Database, Event, JobOutbox, JobTask, JobStatus, job_month_pages_debounced


@pytest.mark.asyncio
async def test_job_month_pages_debounced_uses_coalesce_key(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        ev = Event(
            title="Test Event",
            description="desc",
            date="2026-05-15",
            time="10:00",
            location_name="loc",
            source_text="src",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        event_id = ev.id

        session.add(
            JobOutbox(
                event_id=event_id,
                task=JobTask.month_pages,
                status=JobStatus.running,
                coalesce_key="month_pages:2026-05",
                next_run_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    with patch("main.sync_month_page", new_callable=AsyncMock) as mock_sync:
        ok = await job_month_pages_debounced(event_id, db, None)
        assert ok is True
        mock_sync.assert_awaited_once_with(db, "2026-05")

