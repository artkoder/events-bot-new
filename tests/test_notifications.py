import os
import sys

import pytest
from datetime import datetime
from sqlmodel import select

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import main
from main import (
    Database,
    Event,
    JobOutbox,
    JobStatus,
    JobTask,
    schedule_event_update_tasks,
    run_event_update_jobs,
)


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(text)


@pytest.mark.asyncio
async def test_progress_notifications(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot()

    ev = Event(
        title="t",
        description="d",
        date="2025-01-04",
        time="12:00",
        location_name="loc",
        source_text="src",
    )
    async with db.get_session() as session:
        session.add(ev)
        await session.commit()
        await session.refresh(ev)

    await schedule_event_update_tasks(db, ev)

    async def noop(eid, db_obj, bot_obj):
        return None

    monkeypatch.setattr(main, "update_telegraph_event_page", noop)
    monkeypatch.setattr(main, "job_sync_vk_source_post", noop)
    monkeypatch.setattr(main, "update_month_pages_for", noop)
    monkeypatch.setattr(main, "update_weekend_pages_for", noop)
    monkeypatch.setattr(main, "update_festival_pages_for_event", noop)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": noop,
            "vk_sync": noop,
            "month_pages": noop,
            "weekend_pages": noop,
            "festival_pages": noop,
        },
    )

    async def fake_link(task, eid, db_obj):
        mapping = {
            JobTask.telegraph_build: "http://t",
            JobTask.vk_sync: "http://v",
            JobTask.month_pages: "http://m",
            JobTask.weekend_pages: "http://w",
        }
        return mapping.get(task)

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)

    assert "Telegraph (событие): OK — http://t" in bot.messages
    assert "Страница месяца: OK — http://m" in bot.messages
    assert "Выходные: OK — http://w" in bot.messages
    assert "VK: OK — http://v" in bot.messages

    bot.messages.clear()
    async with db.get_session() as session:
        res = await session.execute(select(JobOutbox))
        for job in res.scalars():
            job.status = JobStatus.pending
            job.next_run_at = datetime.utcnow()
            session.add(job)
        await session.commit()

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)
    assert bot.messages == []
