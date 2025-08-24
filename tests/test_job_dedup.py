import logging
import pytest
import main
from main import Database, JobTask, JobOutbox
from sqlalchemy import select


@pytest.mark.asyncio
async def test_enqueue_job_dedup(tmp_path, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    caplog.set_level(logging.INFO)
    await main.enqueue_job(db, 1, JobTask.week_pages)
    await main.enqueue_job(db, 1, JobTask.week_pages)
    await main.enqueue_job(db, 1, JobTask.month_pages)
    await main.enqueue_job(db, 1, JobTask.month_pages)
    async with db.get_session() as session:
        res = await session.execute(select(JobOutbox).where(JobOutbox.event_id == 1))
        jobs = res.scalars().all()
    kinds = [j.task for j in jobs]
    assert kinds.count(JobTask.week_pages) == 1
    assert kinds.count(JobTask.month_pages) == 1
    assert any("merged job_key=week_pages:1" in r.message for r in caplog.records)
