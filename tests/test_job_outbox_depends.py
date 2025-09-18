import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from main import Database, enqueue_job
from models import Event, JobOutbox, JobStatus, JobTask


@pytest.mark.asyncio
async def test_enqueue_job_merges_depends_on(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        event = Event(
            title="Title",
            description="Description",
            date="2025-01-01",
            time="10:00",
            location_name="Location",
            source_text="Source",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id

        job = JobOutbox(
            event_id=event_id,
            task=JobTask.month_pages,
            status=JobStatus.pending,
            depends_on="initial",
            coalesce_key="month_pages:2025-01",
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        job_id = job.id

    async with db.get_session() as session:
        job = await session.get(JobOutbox, job_id)
        assert job is not None
        assert job.depends_on == "initial"

    result = await enqueue_job(
        db,
        event_id=event_id,
        task=JobTask.month_pages,
        depends_on=["followup"],
    )

    assert result == "merged-rearmed"

    async with db.get_session() as session:
        job = await session.get(JobOutbox, job_id)
        assert job is not None
        assert job.depends_on == "followup,initial"
        assert job.status == JobStatus.pending
