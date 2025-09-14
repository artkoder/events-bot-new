import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from aiogram import types
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


@pytest.mark.asyncio
async def test_vk_requeue_imported_returns_items(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        await session.commit()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
            ("b1", 1, "2025-09"),
        )
        rows = [
            (1, 1, 0, "t", None, 1, None, "imported", "b1"),
            (1, 2, 0, "t", None, 1, None, "imported", "b1"),
            (1, 3, 0, "t", None, 1, None, "imported", "b1"),
            (1, 4, 0, "t", None, 1, None, "imported", "b2"),
        ]
        await conn.executemany(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status, review_batch) VALUES(?,?,?,?,?,?,?,?,?)",
            rows,
        )
        await conn.commit()
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "text": "/vk_requeue_imported 2",
        }
    )
    bot = DummyBot()
    await main.handle_vk_requeue_imported(msg, db, bot)
    assert bot.messages, "no message sent"
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT id, status, review_batch FROM vk_inbox ORDER BY id")
        rows = await cur.fetchall()
    statuses = {r[0]: (r[1], r[2]) for r in rows}
    assert statuses[3][0] == "pending"
    assert statuses[2][0] == "pending"
    assert statuses[1][0] == "imported"
    assert statuses[4][0] == "imported"
