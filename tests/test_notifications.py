import os
import sys

import pytest
from datetime import date, datetime, timedelta, timezone
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
    publish_event_progress,
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

    await schedule_event_update_tasks(db, ev, drain_nav=False)

    async def ok_handler(eid, db_obj, bot_obj):
        return True

    async def nochange_handler(eid, db_obj, bot_obj):
        return False

    monkeypatch.setattr(main, "update_telegraph_event_page", ok_handler)
    monkeypatch.setattr(main, "job_sync_vk_source_post", ok_handler)
    monkeypatch.setattr(main, "update_month_pages_for", nochange_handler)
    monkeypatch.setattr(main, "update_weekend_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_week_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_festival_pages_for_event", ok_handler)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": ok_handler,
            "vk_sync": ok_handler,
            "month_pages": nochange_handler,
            "week_pages": ok_handler,
            "weekend_pages": ok_handler,
            "festival_pages": ok_handler,
        },
    )

    async def fake_link(task, eid, db_obj):
        mapping = {
            JobTask.telegraph_build: "http://t",
            JobTask.vk_sync: "http://v",
            JobTask.month_pages: "http://m",
            JobTask.week_pages: "http://wk",
            JobTask.weekend_pages: "http://w",
        }
        return mapping.get(task)

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)

    assert "Telegraph (событие): OK — http://t" in bot.messages
    assert "Страница месяца: без изменений" in bot.messages
    assert "VK (неделя): OK — http://wk" in bot.messages
    assert "VK (выходные): OK — http://w" in bot.messages
    assert "VK (событие): OK — http://v" in bot.messages

    bot.messages.clear()
    async with db.get_session() as session:
        res = await session.execute(select(JobOutbox))
        for job in res.scalars():
            job.status = JobStatus.pending
            job.next_run_at = datetime.now(timezone.utc)
            session.add(job)
        await session.commit()

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)
    assert bot.messages == []


@pytest.mark.asyncio
async def test_progress_notifications_forced_rebuild(tmp_path, monkeypatch):
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

    await schedule_event_update_tasks(db, ev, drain_nav=False)

    async def month_handler(eid, db_obj, bot_obj):
        return "rebuild"

    async def ok_handler(eid, db_obj, bot_obj):
        return True

    monkeypatch.setattr(main, "update_month_pages_for", month_handler)
    monkeypatch.setattr(main, "update_telegraph_event_page", ok_handler)
    monkeypatch.setattr(main, "job_sync_vk_source_post", ok_handler)
    monkeypatch.setattr(main, "update_weekend_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_week_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_festival_pages_for_event", ok_handler)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": ok_handler,
            "vk_sync": ok_handler,
            "month_pages": month_handler,
            "week_pages": ok_handler,
            "weekend_pages": ok_handler,
            "festival_pages": ok_handler,
        },
    )

    async def fake_link(task, eid, db_obj):
        return "http://m"

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)

    assert any(
        m.startswith("Страница месяца: OK — http://m (forced rebuild)")
        for m in bot.messages
    )


@pytest.mark.asyncio
async def test_progress_notifications_error(tmp_path, monkeypatch):
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

    await schedule_event_update_tasks(db, ev, drain_nav=False)

    async def ok_handler(eid, db_obj, bot_obj):
        return True

    async def err_handler(eid, db_obj, bot_obj):
        raise Exception("boom")

    monkeypatch.setattr(main, "update_telegraph_event_page", ok_handler)
    monkeypatch.setattr(main, "job_sync_vk_source_post", ok_handler)
    monkeypatch.setattr(main, "update_month_pages_for", err_handler)
    monkeypatch.setattr(main, "update_weekend_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_week_pages_for", ok_handler)
    monkeypatch.setattr(main, "update_festival_pages_for_event", ok_handler)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": ok_handler,
            "vk_sync": ok_handler,
            "month_pages": err_handler,
            "week_pages": ok_handler,
            "weekend_pages": ok_handler,
            "festival_pages": ok_handler,
        },
    )

    await run_event_update_jobs(db, bot, notify_chat_id=1, event_id=ev.id)

    assert any(m.startswith("Страница месяца: ERROR: boom") for m in bot.messages)


class ProgBot:
    def __init__(self):
        self.messages = []
        self.edits = []
        self._mid = 0

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(text)
        self._mid += 1
        class M:
            pass

        m = M()
        m.message_id = self._mid
        return m

    async def edit_message_text(self, text, chat_id=None, message_id=None, **kwargs):
        self.edits.append(text)


class CapBot:
    def __init__(self):
        self.messages = []
        self.markups = []
        self._mid = 0

    async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
        self.messages.append(text)
        self.markups.append(reply_markup)
        self._mid += 1
        class M:
            pass

        m = M()
        m.message_id = self._mid
        return m

    async def edit_message_text(self, text, chat_id=None, message_id=None, reply_markup=None, **kwargs):
        pass


@pytest.mark.asyncio
async def test_publish_event_progress_single_message(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

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

    async with db.get_session() as session:
        session.add_all(
            [
                JobOutbox(event_id=ev.id, task=JobTask.telegraph_build),
                JobOutbox(event_id=ev.id, task=JobTask.vk_sync),
                JobOutbox(event_id=ev.id, task=JobTask.week_pages),
            ]
        )
        await session.commit()

    async def ok_handler(eid, db_obj, bot_obj):
        return True

    monkeypatch.setattr(main, "update_telegraph_event_page", ok_handler)
    monkeypatch.setattr(main, "job_sync_vk_source_post", ok_handler)
    monkeypatch.setattr(main, "update_week_pages_for", ok_handler)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": ok_handler,
            "vk_sync": ok_handler,
            "week_pages": ok_handler,
        },
    )

    async def fake_link(task, eid, db_obj):
        mapping = {
            JobTask.telegraph_build: "http://t",
            JobTask.vk_sync: "http://v",
            JobTask.week_pages: "http://wk",
        }
        return mapping.get(task)

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    bot = ProgBot()
    await publish_event_progress(ev, db, bot, chat_id=1)

    assert any("Идёт процесс публикации" in m for m in bot.messages)
    assert bot.edits
    final = bot.edits[-1]
    assert final.startswith("Готово")
    assert "✅ Telegraph (событие) — http://t" in final
    assert "✅ VK (событие) — http://v" in final
    assert "✅ VK (неделя) — http://wk" in final


@pytest.mark.asyncio
async def test_publish_event_progress_waits_for_pending(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

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

    async with db.get_session() as session:
        session.add_all(
            [
                JobOutbox(event_id=ev.id, task=JobTask.telegraph_build),
                JobOutbox(
                    event_id=ev.id,
                    task=JobTask.week_pages,
                    next_run_at=datetime.now(timezone.utc) + timedelta(milliseconds=100),
                ),
            ]
        )
        await session.commit()

    async def ok_handler(eid, db_obj, bot_obj):
        return True

    monkeypatch.setattr(main, "update_telegraph_event_page", ok_handler)
    monkeypatch.setattr(main, "update_week_pages_for", ok_handler)
    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {
            "telegraph_build": ok_handler,
            "week_pages": ok_handler,
        },
    )

    async def fake_link(task, eid, db_obj):
        mapping = {
            JobTask.telegraph_build: "http://t",
            JobTask.week_pages: "http://wk",
        }
        return mapping.get(task)

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    bot = ProgBot()
    await publish_event_progress(ev, db, bot, chat_id=1)

    final = bot.edits[-1]
    assert final.startswith("Готово")
    assert "✅ Telegraph (событие) — http://t" in final
    assert "✅ VK (неделя) — http://wk" in final


@pytest.mark.asyncio
async def test_publish_event_progress_ics(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

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
        session.add(JobOutbox(event_id=ev.id, task=JobTask.ics_publish))
        session.add(JobOutbox(event_id=ev.id, task=JobTask.tg_ics_post))
        await session.commit()

    async def ics_handler(eid, db_obj, bot_obj, progress):
        progress.mark("ics_supabase", "done", "https://sup")
        return True

    async def tg_handler(eid, db_obj, bot_obj, progress):
        progress.mark("ics_telegram", "done", "https://tg")
        return True

    monkeypatch.setattr(
        main,
        "JOB_HANDLERS",
        {"ics_publish": ics_handler, "tg_ics_post": tg_handler},
    )

    async def fake_link(task, eid, db_obj):
        return None

    monkeypatch.setattr(main, "_job_result_link", fake_link)

    bot = ProgBot()
    await publish_event_progress(ev, db, bot, chat_id=1)

    final = bot.edits[-1]
    assert "✅ ICS (Supabase) — https://sup" in final
    assert "✅ ICS (Telegram) — https://tg" in final
    assert "<a href" not in final


@pytest.mark.asyncio
async def test_publish_event_progress_captcha_flag(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

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
        session.add(
            JobOutbox(
                event_id=ev.id,
                task=JobTask.vk_sync,
                next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
        )
        await session.commit()

    main._vk_captcha_needed = True
    bot = CapBot()
    await publish_event_progress(ev, db, bot, chat_id=1)

    assert bot.messages
    text = bot.messages[0]
    assert "⏸ VK — требуется капча; нажмите «Ввести код»" in text
    assert text.count("VK") == 1
    markup = bot.markups[0]
    assert markup.inline_keyboard[0][0].callback_data == "captcha_input"
    main._vk_captcha_needed = False


@pytest.mark.asyncio
async def test_publish_event_progress_fest_index(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    future_date = (date.today() + timedelta(days=1)).isoformat()
    ev = Event(
        title="t",
        description="d",
        date=future_date,
        time="12:00",
        location_name="loc",
        source_text="src",
        festival="Fest",
    )
    async with db.get_session() as session:
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(JobOutbox(event_id=ev.id, task=JobTask.festival_pages))
        await session.commit()

    async def fake_sync_festival_page(db_obj, name, **kwargs):
        return "http://fest"

    async def fake_sync_festival_vk_post(db_obj, name, bot_obj, nav_only=False, nav_lines=None, strict=False):
        return False

    async def fake_rebuild(db_obj):
        return True

    async def fake_get_setting(db_obj, key):
        if key == "fest_index_url":
            return "http://index"
        return None

    monkeypatch.setattr(main, "sync_festival_page", fake_sync_festival_page)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_festival_vk_post)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild)
    monkeypatch.setattr(main, "get_setting_value", fake_get_setting)
    monkeypatch.setattr(main, "_job_result_link", lambda task, eid, db_obj: "http://vk")

    bot = ProgBot()
    await publish_event_progress(ev, db, bot, chat_id=1)

    final = bot.edits[-1]
    assert "✅ Telegraph (фестиваль) — http://fest" in final
    assert "✅ Все фестивали (Telegraph) — http://index" in final
