import os
import pytest
from aiogram import Bot, types
import main
from main import Database, Event, Channel


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

    def from_(self, bucket):
        self.bucket = bucket
        return self

    def upload(self, path, content, options):
        pass

    def get_public_url(self, path):
        return f"https://supabase/{path}"


class Progress:
    def __init__(self):
        self.marks = []

    def mark(self, key, status, detail):
        self.marks.append((key, status, detail))


@pytest.mark.asyncio
async def test_progress_lines_for_ics_states(tmp_path, monkeypatch):
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
                telegraph_url="https://tg/1",
                silent=True,
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    async def fake_update(*a, **k):
        pass
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)

    pr = Progress()
    await main.ics_publish(1, db, bot, pr)
    assert ("ics_supabase", "done", "https://supabase/event-1-2025-07-18.ics") in pr.marks
    assert any(
        m[0] == "ics_telegram" and m[1] == "done" and m[2].startswith("https://t.me/c/")
        for m in pr.marks
    )
    caption, parse_mode = bot.docs[0][1], bot.docs[0][2]
    assert parse_mode == "HTML"
    lines = caption.splitlines()
    assert lines[0] == "A"
    assert lines[1] == '<a href="https://tg/1">Подробнее</a>'
    assert "Hall" in lines[2] and "#Town" in lines[2] and ", 19:00" in lines[2]

    os.environ["SUPABASE_DISABLED"] = "1"
    pr = Progress()
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        ev.time = "20:00"
        await session.commit()
    await main.ics_publish(1, db, bot, pr)
    assert ("ics_supabase", "skipped_disabled", "disabled") in pr.marks
    assert any(m[0] == "ics_telegram" and m[1] == "done" for m in pr.marks)
    del os.environ["SUPABASE_DISABLED"]

    pr = Progress()
    await main.ics_publish(1, db, bot, pr)
    assert ("ics_supabase", "skipped_nochange", "no change") in pr.marks
    assert ("ics_telegram", "skipped_nochange", "no change") in pr.marks

    class BadClient(FakeClient):
        def upload(self, *a, **k):
            raise RuntimeError("fail")
    monkeypatch.setattr(main, "get_supabase_client", lambda: BadClient())
    pr = Progress()
    async with db.get_session() as session:
        ev = await session.get(Event, 1)
        ev.time = "21:00"
        await session.commit()
    with pytest.raises(RuntimeError):
        await main.ics_publish(1, db, bot, pr)
    assert any(m[0] == "ics_supabase" and m[1] == "error" for m in pr.marks)
    assert any(m[0] == "ics_telegram" and m[1] == "done" for m in pr.marks)


class HTMLFailBot(DummyBot):
    async def send_document(self, chat_id, document, caption=None, parse_mode=None, **kwargs):
        if parse_mode == "HTML":
            from aiogram.exceptions import TelegramBadRequest
            raise TelegramBadRequest(object(), "fail")
        return await super().send_document(chat_id, document, caption=caption, **kwargs)


@pytest.mark.asyncio
async def test_html_fallback(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = HTMLFailBot("123:abc")
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
                telegraph_url="https://tg/1",
                silent=True,
            )
        )
        await session.commit()
    fake = FakeClient()
    monkeypatch.setattr(main, "get_supabase_client", lambda: fake)
    async def fake_update(*a, **k):
        pass
    monkeypatch.setattr(main, "update_source_page_ics", fake_update)
    pr = Progress()
    await main.ics_publish(1, db, bot, pr)
    caption, parse_mode = bot.docs[0][1], bot.docs[0][2]
    assert parse_mode is None
    assert "Подробнее: https://tg/1" in caption
    assert any(
        m[0] == "ics_telegram" and m[1] == "done" and m[2].startswith("https://t.me/c/")
        for m in pr.marks
    )
