import logging
import pytest
import main
from main import Database, JobTask, JobOutbox, JobStatus
from sqlalchemy import select
from datetime import datetime, timedelta


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
    assert any(
        r.message.startswith("ENQ") and "job_key=week_pages:1" in r.message and "merged" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_enqueue_job_requeue_and_skip(tmp_path, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(
            JobOutbox(
                event_id=1,
                task=JobTask.month_pages,
                status=JobStatus.done,
                attempts=1,
                last_error="err",
                updated_at=datetime.utcnow() - timedelta(days=1),
                next_run_at=datetime.utcnow() - timedelta(days=1),
            )
        )
        session.add(
            JobOutbox(
                event_id=1,
                task=JobTask.vk_sync,
                status=JobStatus.done,
            )
        )
        await session.commit()
    caplog.set_level(logging.INFO)
    action1 = await main.enqueue_job(db, 1, JobTask.month_pages)
    action2 = await main.enqueue_job(db, 1, JobTask.vk_sync)
    assert action1 == "requeued"
    assert action2 == "skipped"
    async with db.get_session() as session:
        res = await session.execute(
            select(JobOutbox).where(JobOutbox.event_id == 1)
        )
        jobs = {j.task: j for j in res.scalars().all()}
    assert jobs[JobTask.month_pages].status == JobStatus.pending
    assert jobs[JobTask.month_pages].attempts == 0
    assert jobs[JobTask.month_pages].last_error is None
    assert jobs[JobTask.vk_sync].status == JobStatus.done
    assert any(
        r.message.startswith("ENQ") and "job_key=month_pages:1" in r.message and "requeued" in r.message
        for r in caplog.records
    )
    assert any(
        r.message.startswith("ENQ") and "job_key=vk_sync:1" in r.message and "skipped" in r.message
        for r in caplog.records
    )
