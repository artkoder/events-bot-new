from types import SimpleNamespace

import pytest

import main
import vk_intake


class DummyMessage:
    def __init__(self, chat_id, text, reply_markup):
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.reply_markup = reply_markup

    async def edit_text(self, text, **kwargs):
        self.text = text
        self.reply_markup = kwargs.get("reply_markup")
        return self


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg = DummyMessage(chat_id, text, kwargs.get("reply_markup"))
        self.messages.append(msg)
        return msg


@pytest.mark.asyncio
async def test_vk_list_shows_numbers_and_default_time(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        for idx in range(1, 13):
            await conn.execute(
                "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
                (
                    idx,
                    f"club{idx}",
                    f"Group {idx}",
                    None,
                    "19:00" if idx == 1 else None,
                ),
            )
        # inbox stats for first two groups
        for post_id in range(2):
            await conn.execute(
                "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
                (1, post_id, 0, "text", None, 1, "pending"),
            )
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
            (1, 100, 0, "text", None, 1, "skipped"),
        )
        for post_id in range(12):
            await conn.execute(
                "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
                (2, 200 + post_id, 0, "text", None, 1, "imported"),
            )
        await conn.execute(
            "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
            (2, 400, 0, "text", None, 1, "rejected"),
        )
        await conn.commit()

    bot = DummyBot()
    msg = SimpleNamespace(chat=SimpleNamespace(id=1))
    await main.handle_vk_list(msg, db, bot)

    assert bot.messages, "no message sent"
    lines = bot.messages[0].text.splitlines()
    assert lines[0].startswith("1.")
    assert "типовое время: 19:00" in lines[0]
    assert lines[1] == "    Pending | Skipped | Imported | Rejected"
    assert lines[2] == "          2 |       1 |        0 |        0"
    assert lines[3].startswith("2.")
    assert "типовое время: -" in lines[3]
    assert lines[4] == "    Pending | Skipped | Imported | Rejected"
    assert lines[5] == "          0 |       0 |       12 |        1"
    buttons = bot.messages[0].reply_markup.inline_keyboard
    assert buttons[0][0].text == "❌ 1"
    assert buttons[0][0].callback_data.endswith(":1")
    assert buttons[0][1].text == "🕒 1"
    assert buttons[1][0].text == "❌ 2"
    assert buttons[1][0].callback_data.endswith(":1")
    assert buttons[-1][-1].callback_data == "vksrcpage:2"

    class DummyCallback:
        def __init__(self, data, message):
            self.data = data
            self.message = message
            self.answered = False

        async def answer(self, *args, **kwargs):
            self.answered = True

    callback = DummyCallback("vksrcpage:2", bot.messages[0])
    await main.handle_vk_list_page_callback(callback, db, bot)
    assert callback.answered
    page2_lines = bot.messages[0].text.splitlines()
    assert page2_lines[0].startswith("11.")
    assert page2_lines[1] == "    Pending | Skipped | Imported | Rejected"
    assert page2_lines[2] == "          0 |       0 |        0 |        0"
    nav_row = bot.messages[0].reply_markup.inline_keyboard[-1]
    assert nav_row[0].callback_data == "vksrcpage:1"


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
    main.vk_default_time_sessions[1] = main.VkDefaultTimeSession(
        source_id=vid,
        page=1,
    )
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

    main.vk_default_time_sessions[1] = main.VkDefaultTimeSession(
        source_id=vid,
        page=1,
    )
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
