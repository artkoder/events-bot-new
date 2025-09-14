import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from types import SimpleNamespace

import main
from main import Database, User


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg = SimpleNamespace(
            message_id=len(self.messages) + 1,
            date=0,
            chat=SimpleNamespace(id=chat_id, type="private"),
            from_user=SimpleNamespace(id=0, is_bot=True, first_name="B"),
            text=text,
            reply_markup=kwargs.get("reply_markup"),
        )
        self.messages.append(msg)
        return msg

    async def send_media_group(self, chat_id, media):
        self.media = media


@pytest.mark.asyncio
async def test_vkrev_show_next_adds_blank_line_and_group_name(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?)",
            (1, 10, 0, "text", None, 1, 9999999999, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    bot = DummyBot()
    await main._vkrev_show_next(1, "batch1", 1, db, bot)
    assert bot.messages, "no message sent"
    lines = bot.messages[0].text.splitlines()
    assert lines[1] == "Test Community"
    assert lines[2] == ""  # blank line before the link
    assert lines[3] == "https://vk.com/wall-1_10"
