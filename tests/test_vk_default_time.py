from types import SimpleNamespace

import pytest

import main
import vk_intake


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg = SimpleNamespace(text=text, reply_markup=kwargs.get("reply_markup"))
        self.messages.append(msg)
        return msg


@pytest.mark.asyncio
async def test_vk_list_shows_numbers_and_default_time(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "One", None, "19:00"),
        )
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (2, "club2", "Two", None, None),
        )
        await conn.commit()

    bot = DummyBot()
    msg = SimpleNamespace(chat=SimpleNamespace(id=1))
    await main.handle_vk_list(msg, db, bot)

    assert bot.messages, "no message sent"
    lines = bot.messages[0].text.splitlines()
    assert lines[0].startswith("1.")
    assert "типовое время: 19:00" in lines[0]
    assert lines[1].startswith("2.")
    assert "типовое время: -" in lines[1]
    buttons = bot.messages[0].reply_markup.inline_keyboard
    assert buttons[0][0].text == "❌ 1"
    assert buttons[0][1].text == "🕒 1"
    assert buttons[1][0].text == "❌ 2"
    assert buttons[1][1].text == "🕒 2"


@pytest.mark.asyncio
async def test_vk_default_time_message_updates_db(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "One", None, None),
        )
        await conn.commit()
        cur = await conn.execute("SELECT id FROM vk_source")
        (vid,) = await cur.fetchone()

    bot = DummyBot()
    main.vk_default_time_sessions[1] = vid
    message = SimpleNamespace(
        chat=SimpleNamespace(id=1),
        from_user=SimpleNamespace(id=1),
        text="20:30",
    )
    await main.handle_vk_dtime_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT default_time FROM vk_source WHERE id=?", (vid,))
        (val,) = await cur.fetchone()
    assert val == "20:30"

    main.vk_default_time_sessions[1] = vid
    message.text = "-"
    await main.handle_vk_dtime_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT default_time FROM vk_source WHERE id=?", (vid,))
        (val,) = await cur.fetchone()
    assert val is None


@pytest.mark.asyncio
async def test_build_event_payload_includes_default_time(monkeypatch):
    captured = {}

    async def fake_parse(text, **kwargs):
        captured["text"] = text
        captured["festival_names"] = kwargs.get("festival_names")
        return [{"title": "T", "date": "2099-01-01"}]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    draft = await vk_intake.build_event_payload_from_vk("text", default_time="19:00")

    assert "19:00" in captured["text"]
    assert captured["festival_names"] is None
    assert draft.time == "19:00"
