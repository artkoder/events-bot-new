import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from aiogram import types

import main
from main import Database, User, Event, Channel
from test_digest_command import DummyBot


@pytest.mark.asyncio
async def test_notify_partners(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        session.add(User(user_id=2, username="p1", is_partner=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        ev = Event(
            title="L1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
            telegraph_url="https://telegra.ph/1",
            creator_id=2,
        )
        session.add_all([ev, Channel(channel_id=-100, title="Chan", daily_time="08:00")])
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

    panel_msg = bot.messages[-1]
    async def ans2(*args, **kw):
        return None
    cb_send = SimpleNamespace(
        data=f"dg:s:{digest_id}:-100",
        from_user=SimpleNamespace(id=1),
        message=panel_msg,
        answer=ans2,
        id="2",
    )
    await main.handle_digest_send(cb_send, db, bot)
    status_msg = bot.messages[-1]
    assert status_msg.reply_markup is not None
    cb_notify = SimpleNamespace(
        data=f"dg:np:{digest_id}:-100",
        from_user=SimpleNamespace(id=1),
        message=status_msg,
        answer=ans2,
        id="3",
    )
    await main.handle_digest_notify_partners(cb_notify, db, bot)
    partner_msgs = [m for m in bot.messages if m.chat.id == 2]
    assert len(partner_msgs) == 1
    assert "Ваше событие попало в дайджест" in partner_msgs[0].text
    assert bot.messages[-1].text.startswith("Уведомлено:")
    prev = len(partner_msgs)
    await main.handle_digest_notify_partners(cb_notify, db, bot)
    partner_msgs = [m for m in bot.messages if m.chat.id == 2]
    assert len(partner_msgs) == prev
