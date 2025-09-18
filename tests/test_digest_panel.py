import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from aiogram import types

import main
from main import Database, User, Event, Channel
from test_digest_command import DummyBot

@pytest.mark.asyncio
async def test_digest_toggle_refresh_publish(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        ev1 = Event(
            title="L1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
            telegraph_url="https://telegra.ph/1",
        )
        ev2 = Event(
            title="L2",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="13:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
            telegraph_url="https://telegra.ph/2",
        )
        session.add_all([ev1, ev2, Channel(channel_id=-100, title="Chan", daily_time="08:00")])
        await session.commit()
    bot = DummyBot()

    async def fake_ask(prompt, max_tokens=0):
        return "Интро"

    async def fake_extract(url, **kw):
        return ["https://example.com/img.jpg"]

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    msg = types.Message.model_validate({
        "message_id": 1,
        "date": 0,
        "chat": {"id": 1, "type": "private"},
        "from": {"id": 1, "is_bot": False, "first_name": "U"},
        "text": "/digest",
    })
    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = menu_msg.reply_markup.inline_keyboard[0][0].callback_data.split(":")[-1]
    async def ans(**kw):
        return None
    cb_select = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:lectures:{digest_id}",
        answer=ans,
    )
    await main.handle_digest_select_lectures(cb_select, db, bot)

    session_data = main.digest_preview_sessions[digest_id]
    assert len(session_data["items"]) == 2
    assert session_data["excluded"] == set()

    panel_msg = bot.messages[-1]
    async def ans2(**kw):
        return None
    cb_toggle = SimpleNamespace(
        data=f"dg:t:{digest_id}:1",
        from_user=SimpleNamespace(id=1),
        message=panel_msg,
        answer=ans2,
        id="2",
    )
    prev_groups = len(bot.media_groups)
    await main.handle_digest_toggle(cb_toggle, bot)
    assert len(bot.media_groups) == prev_groups
    updated_panel = next(m for m in bot.messages if m.message_id == panel_msg.message_id)
    assert updated_panel.reply_markup.inline_keyboard[0][0].text.startswith("❌")
    assert 0 in session_data["excluded"]

    async def ans3(**kw):
        return None
    cb_refresh = SimpleNamespace(
        data=f"dg:r:{digest_id}",
        from_user=SimpleNamespace(id=1),
        message=panel_msg,
        answer=ans3,
        id="3",
    )
    await main.handle_digest_refresh(cb_refresh, bot)
    assert len(bot.media_groups) == prev_groups + 1
    caption = bot.media_groups[-1][1][0].caption
    assert "L2" in caption and "L1" not in caption

    new_panel = bot.messages[-1]
    async def ans4(*args, **kw):
        return None
    cb_send = SimpleNamespace(
        data=f"dg:s:{digest_id}:-100",
        from_user=SimpleNamespace(id=1),
        message=new_panel,
        answer=ans4,
        id="4",
    )
    await main.handle_digest_send(cb_send, db, bot)
    # check that link and status messages were sent
    assert bot.messages[-2].text.startswith("https://")
    assert "нет событий" in bot.messages[-1].text
