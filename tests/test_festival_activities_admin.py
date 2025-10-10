from __future__ import annotations

from datetime import datetime, timezone

import pytest
from aiogram import types

import main
from db import Database
from models import Festival, User


@pytest.mark.asyncio
async def test_handle_festival_activities_upload(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(User(user_id=1))
        fest = Festival(name="Fest")
        session.add(fest)
        await session.commit()
        fid = fest.id

    main.festival_edit_sessions[1] = (fid, main.FESTIVAL_EDIT_FIELD_ACTIVITIES)

    preview_messages: list[str] = []

    async def fake_sync(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "sync_festival_page", fake_sync)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_sync)

    class DummyBot:
        def __init__(self):
            self.sent: list[dict[str, object]] = []

        async def send_message(self, chat_id, text, reply_markup=None, **kwargs):
            self.sent.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})
            preview_messages.append(text)

        async def download(self, *a, **k):  # pragma: no cover
            raise AssertionError("download should not be called")

    bot = DummyBot()

    yaml_text = (
        "version: 2\n"
        "festival_site: https://fest.example\n"
        "always_on:\n"
        "  - title: Экспозиция\n"
        "by_request: []\n"
    )

    message = types.Message.model_validate(
        {
            "message_id": 10,
            "date": datetime.now(timezone.utc),
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "Admin"},
            "text": yaml_text,
        }
    )

    await main.handle_festival_edit_message(message, db, bot)

    assert main.festival_edit_sessions[1] == (fid, None)
    assert any("Сохранено активностей" in text for text in preview_messages)

    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        assert fest.website_url == "https://fest.example"
        assert fest.activities_json

    main.festival_edit_sessions.pop(1, None)
