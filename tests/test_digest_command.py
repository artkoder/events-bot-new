import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, timedelta
import pytest
from aiogram import types
from types import SimpleNamespace

import main
from main import Database, User, Event

class DummyBot:
    def __init__(self):
        self.messages = []
        self.media_groups = []

    async def send_message(self, chat_id, text, **kwargs):
        msg_id = len(self.messages) + 1
        data = {
            "message_id": msg_id,
            "date": 0,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": 0, "is_bot": True, "first_name": "B"},
            "text": text,
        }
        if "reply_markup" in kwargs:
            data["reply_markup"] = kwargs["reply_markup"]
        msg = types.Message.model_validate(data)
        self.messages.append(msg)
        return msg

    async def send_media_group(self, chat_id, media, **kwargs):
        self.media_groups.append((chat_id, media, kwargs))
        msgs = []
        for m in media:
            msg_id = len(self.messages) + len(msgs) + 1
            data = {
                "message_id": msg_id,
                "date": 0,
                "chat": {"id": chat_id, "type": "private"},
                "from": {"id": 0, "is_bot": True, "first_name": "B"},
            }
            if getattr(m, "caption", None):
                data["caption"] = m.caption
            msg = types.Message.model_validate(data)
            msgs.append(msg)
        self.messages.extend(msgs)
        return msgs


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_handle_digest_sends_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.utcnow() + timedelta(days=1)
        ev = Event(
            title="L1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
            telegraph_url="https://telegra.ph/test",
        )
        session.add(ev)
        await session.commit()
    msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "U"},
        "text": "/digest",
    })
    bot = DummyBot()

    async def fake_ask(prompt, max_tokens=0):
        return "Интро"

    async def fake_extract(url, event_id=None):
        return "https://example.com/img.jpg"

    class DummyClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def head(self, url):
            return SimpleNamespace(headers={"content-type": "image/jpeg"})

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_telegra_ph_cover_url", fake_extract)
    monkeypatch.setattr(main.httpx, "AsyncClient", DummyClient)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = menu_msg.reply_markup.inline_keyboard[0][0].callback_data.split(":")[-1]
    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "message": menu_msg.model_dump(),
            "chat_instance": "1",
            "data": f"digest:select:lectures:{digest_id}",
        }
    )
    await main.handle_digest_select_lectures(cb, db, bot)
    assert bot.media_groups
    caption = bot.media_groups[0][1][0].caption
    assert "<a href=" in caption


def test_help_contains_digest():
    assert any(cmd["usage"].startswith("/digest") for cmd in main.HELP_COMMANDS)
