import asyncio
import time
import os
import pytest
from aiogram import Bot, types
import main
from main import Database, Event, Channel, JobOutbox, JobTask
from sqlalchemy import select


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.docs = []

    async def send_document(self, chat_id, document, caption=None, **kwargs):
        self.docs.append((chat_id, caption, kwargs.get("parse_mode")))
        from types import SimpleNamespace
        chat = SimpleNamespace(id=chat_id)
        return SimpleNamespace(
            document=SimpleNamespace(file_id="file" + str(len(self.docs))),
            message_id=len(self.docs),
            chat=chat,
        )


class FakeClient:
    def __init__(self):
        self.storage = self
        self.uploaded = []

    def from_(self, bucket):
        self.bucket = bucket
        return self

    def upload(self, path, content, options):
        self.uploaded.append((path, content, options))

    def get_public_url(self, path):
        return f"https://supabase/{path}"


class Progress:
    def __init__(self):
        self.marks = []

    def mark(self, key, status, detail):
        self.marks.append((key, status, detail))


@pytest.mark.asyncio
async def test_publish_ics_both_channels_success(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="Concert",
                description="desc",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    called = {}
    async def fake_update(*a, **k):
        called["v"] = True
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)
    monkeypatch.setattr(main, "update_source_post_keyboard", lambda *a, **k: None)
    await main.ics_publish(1, db, bot)
    assert fake.uploaded
    assert bot.docs
    assert called.get("v")
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.ics_hash and ev.ics_url and ev.ics_file_id
        assert ev.ics_post_url and ev.ics_post_id


@pytest.mark.asyncio
async def test_ics_skips_when_no_change(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    async def fake_update(*a, **k):
        pass
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)
    monkeypatch.setattr(main, "update_source_post_keyboard", lambda *a, **k: None)
    await main.ics_publish(1, db, bot)
    fake.uploaded.clear()
    bot.docs.clear()
    pr = Progress()
    await main.ics_publish(1, db, bot, pr)
    assert not fake.uploaded
    assert not bot.docs
    assert ("ics_supabase", "skipped_nochange", "no change") in pr.marks
    assert ("ics_telegram", "skipped_nochange", "no change") in pr.marks


@pytest.mark.asyncio
async def test_ics_updates_on_change(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    async def fake_update(*a, **k):
        pass
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)
    monkeypatch.setattr(main, "update_source_post_keyboard", lambda *a, **k: None)
    await main.ics_publish(1, db, bot)
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        h1, u1, f1 = ev.ics_hash, ev.ics_url, ev.ics_file_id
        ev.date = "2025-07-19"
        await session.commit()
    await main.ics_publish(1, db, bot)
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.ics_hash != h1
        assert ev.ics_url != u1
        assert ev.ics_file_id != f1


@pytest.mark.asyncio
async def test_supabase_error_does_not_block_telegram(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    class BadClient(FakeClient):
        def upload(self, *a, **k):
            raise RuntimeError("fail")
    fake = BadClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    with pytest.raises(RuntimeError):
        await main.ics_publish(1, db, bot)
    assert bot.docs
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.ics_file_id
        assert ev.ics_url is None


@pytest.mark.asyncio
async def test_telegram_error_does_not_block_supabase(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    class BadBot(DummyBot):
        async def send_document(self, *a, **k):
            raise RuntimeError("tg fail")
    bot = BadBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    with pytest.raises(RuntimeError):
        await main.ics_publish(1, db, bot)
    assert fake.uploaded
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        assert ev.ics_url
        assert ev.ics_file_id is None


@pytest.mark.asyncio
async def test_ics_coalesced_jobs_and_semaphore(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add_all(
            [
                Event(
                    id=1,
                    title="A",
                    description="d",
                    source_text="s",
                    date="2025-07-18",
                    time="19:00",
                    location_name="Hall",
                    city="Town",
                ),
                Event(
                    id=2,
                    title="B",
                    description="d",
                    source_text="s",
                    date="2025-07-19",
                    time="19:00",
                    location_name="Hall",
                    city="Town",
                ),
            ]
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    order = []
    orig = main.build_ics_content
    async def fake_build(db_arg, ev_arg):
        order.append((ev_arg.id, time.perf_counter()))
        await asyncio.sleep(0.1)
        return await orig(db_arg, ev_arg)
    monkeypatch.setattr(main, "build_ics_content", fake_build)
    await asyncio.gather(
        main.ics_publish(1, db, bot),
        main.ics_publish(2, db, bot),
    )
    assert order[0][0] == 1 and order[1][0] == 2
    assert order[1][1] >= order[0][1]


@pytest.mark.asyncio
async def test_ics_publish_enqueues_page_jobs(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")
    async with db.get_session() as session:
        session.add(Channel(channel_id=-100, title="Asset", is_admin=True, is_asset=True))
        session.add(
            Event(
                id=1,
                title="A",
                description="d",
                source_text="s",
                date="2025-07-18",
                time="19:00",
                location_name="Hall",
                city="Town",
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    async def fake_update(*a, **k):
        pass
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)
    monkeypatch.setattr(main, "update_source_post_keyboard", lambda *a, **k: None)
    await main.ics_publish(1, db, bot)
    async with db.get_session() as session:
        res = await session.execute(select(JobOutbox.task))
        tasks = {row[0] for row in res.all()}
    assert JobTask.month_pages in tasks
    assert JobTask.weekend_pages in tasks
    assert JobTask.week_pages in tasks
