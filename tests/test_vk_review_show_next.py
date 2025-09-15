import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from types import SimpleNamespace

from aiogram import types

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


@pytest.mark.asyncio
async def test_vkrev_show_next_prompts_finish_on_empty_queue(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("batch1", 1, "2025-09"),
        )
        await conn.commit()
    bot = DummyBot()
    await main._vkrev_show_next(1, "batch1", 1, db, bot)
    assert len(bot.messages) == 2
    empty_msg, finish_msg = bot.messages
    assert empty_msg.text == "Очередь пуста"
    assert isinstance(finish_msg.reply_markup, types.InlineKeyboardMarkup)
    button = finish_msg.reply_markup.inline_keyboard[0][0]
    assert button.callback_data == "vkrev:finish:batch1"
    assert button.text.startswith("🧹 Завершить")
    assert "обновления" in finish_msg.text.lower()


@pytest.mark.asyncio
async def test_handle_vk_check_creates_new_batch(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()

    captured: dict[str, str] = {}

    async def fake_show_next(chat_id, batch_id, operator_id, db_obj, bot_obj):
        captured["batch_id"] = batch_id
        captured["operator_id"] = operator_id

    monkeypatch.setattr(main, "_vkrev_show_next", fake_show_next)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "🔎 Проверить события",
        }
    )

    bot = DummyBot()
    await main.handle_vk_check(msg, db, bot)
    assert captured["operator_id"] == 1
    assert captured["batch_id"].endswith(":1")
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT batch_id FROM vk_review_batch")
        rows = await cur.fetchall()
    assert len(rows) == 1 and rows[0][0] == captured["batch_id"]


@pytest.mark.asyncio
async def test_handle_vk_check_reuses_existing_batch(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("existing", 1, "2025-09"),
        )
        await conn.commit()

    captured: dict[str, str] = {}

    async def fake_show_next(chat_id, batch_id, operator_id, db_obj, bot_obj):
        captured["batch_id"] = batch_id
        captured["operator_id"] = operator_id

    monkeypatch.setattr(main, "_vkrev_show_next", fake_show_next)

    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "🔎 Проверить события",
        }
    )

    bot = DummyBot()
    await main.handle_vk_check(msg, db, bot)
    assert captured["operator_id"] == 1
    assert captured["batch_id"] == "existing"
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM vk_review_batch")
        (count,) = await cur.fetchone()
    assert count == 1
