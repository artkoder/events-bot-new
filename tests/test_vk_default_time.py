from types import SimpleNamespace

import pytest

import main
import vk_intake


class DummyMessage:
    def __init__(self, chat_id, text, reply_markup):
        self.chat = SimpleNamespace(id=chat_id)
        self.text = text
        self.reply_markup = reply_markup
        self.extra_kwargs: dict[str, object] = {}

    async def edit_text(self, text, **kwargs):
        self.text = text
        self.reply_markup = kwargs.get("reply_markup")
        return self


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        msg = DummyMessage(chat_id, text, kwargs.get("reply_markup"))
        msg.extra_kwargs = kwargs
        self.messages.append(msg)
        return msg


class DummyCallback:
    def __init__(self, data, message):
        self.data = data
        self.message = message
        self.answers: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def answer(self, *args, **kwargs):
        self.answers.append((args, kwargs))


@pytest.mark.asyncio
async def test_vk_list_shows_numbers_and_default_time(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        for idx in range(1, 13):
            await conn.execute(
                "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
                (
                    idx,
                    f"club{idx}",
                    f"Group {idx}",
                    None,
                    "19:00" if idx == 1 else None,
                    "https://tickets.example/club1" if idx == 1 else None,
                ),
            )
        await conn.execute(
            "INSERT INTO vk_crawl_cursor(group_id, updated_at) VALUES(?, ?)",
            (1, "2024-05-31 12:34:56"),
        )
        await conn.execute(
            "INSERT INTO vk_crawl_cursor(group_id, updated_at) VALUES(?, ?)",
            (2, 1717156496),
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
    assert "билеты: https://tickets.example/club1" in lines[0]
    assert "последняя проверка: 2024-05-31 12:34" in lines[0]
    assert lines[1] == " Pending | Skipped | Imported | Rejected "
    assert (
        lines[2]
        == "       2        |       1        |        0        |        0        "
    )
    assert lines[3].startswith("2.")
    assert "типовое время: -" in lines[3]
    assert ", последняя проверка: 2024-05-31 11:54" in lines[3]
    assert lines[4] == " Pending | Skipped | Imported | Rejected "
    assert (
        lines[5]
        == "       0        |       0        |       12        |        1        "
    )
    buttons = bot.messages[0].reply_markup.inline_keyboard
    assert buttons[0][0].text == "❌ 1"
    assert buttons[0][0].callback_data.endswith(":1")
    assert buttons[0][1].text == "⚙️ 1"
    assert buttons[1][0].text == "🕒 1"
    assert buttons[1][1].text == "🎟 1"
    assert buttons[1][2].text == "📍 1"
    assert buttons[2][0].text == "❌ 2"
    assert buttons[2][0].callback_data.endswith(":1")
    assert buttons[2][1].text == "⚙️ 2"
    assert buttons[3][0].text == "🕒 2"
    assert buttons[3][1].text == "🎟 2"
    assert buttons[3][2].text == "📍 2"
    rejected_buttons = [btn for row in buttons for btn in row if btn.text.startswith("🚫 Rejected")]
    assert rejected_buttons, "expected rejected button"
    assert rejected_buttons[0].text == "🚫 Rejected: 1"
    assert rejected_buttons[0].callback_data == "vkrejected:2:1"
    assert buttons[-1][-1].callback_data == "vksrcpage:2"

    callback = DummyCallback("vksrcpage:2", bot.messages[0])
    await main.handle_vk_list_page_callback(callback, db, bot)
    assert callback.answers
    page2_lines = bot.messages[0].text.splitlines()
    assert page2_lines[0].startswith("11.")
    assert page2_lines[1] == " Pending | Skipped | Imported | Rejected "
    assert (
        page2_lines[2]
        == "       0        |       0        |        0        |        0        "
    )
    nav_row = bot.messages[0].reply_markup.inline_keyboard[-1]
    assert nav_row[0].callback_data == "vksrcpage:1"


@pytest.mark.asyncio
async def test_vk_rejected_callback_sends_links(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Group 1", None, None, None),
        )
        for idx in range(2):
            await conn.execute(
                "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
                (1, 100 + idx, 0, "text", None, 1, "rejected"),
            )
        await conn.commit()

    bot = DummyBot()
    list_message = DummyMessage(1, "", None)
    callback = DummyCallback("vkrejected:1:1", list_message)

    await main.handle_vk_rejected_callback(callback, db, bot)

    assert callback.answers
    assert len(bot.messages) == 1
    sent = bot.messages[0]
    assert sent.extra_kwargs.get("disable_web_page_preview") is True
    lines = sent.text.splitlines()
    assert lines[0] == "🚫 Отклонённые посты — Group 1 (vk.com/club1)"
    assert set(lines[1:]) == {
        "https://vk.com/wall-1_100",
        "https://vk.com/wall-1_101",
    }


@pytest.mark.asyncio
async def test_vk_rejected_callback_alerts_when_empty(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Group 1", None, None, None),
        )
        await conn.commit()

    bot = DummyBot()
    list_message = DummyMessage(1, "", None)
    callback = DummyCallback("vkrejected:1:1", list_message)

    await main.handle_vk_rejected_callback(callback, db, bot)

    assert not bot.messages
    assert callback.answers
    assert callback.answers[0][0] == ("Нет отклонённых постов",)
    assert callback.answers[0][1].get("show_alert") is True


@pytest.mark.asyncio
async def test_vk_default_time_message_updates_db(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "One", None, None, None),
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
async def test_vk_default_ticket_link_message_updates_db(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "One", None, None, None),
        )
        await conn.commit()
        cur = await conn.execute("SELECT id FROM vk_source")
        (vid,) = await cur.fetchone()

    bot = DummyBot()
    list_message = DummyMessage(1, "", None)
    main.vk_default_ticket_link_sessions.clear()
    main.vk_default_ticket_link_sessions[1] = main.VkDefaultTicketLinkSession(
        source_id=vid,
        page=1,
        message=list_message,
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=1),
        from_user=SimpleNamespace(id=1),
        text="https://tickets.new",
    )
    await main.handle_vk_ticket_link_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        (link_val,) = await cur.fetchone()
    assert link_val == "https://tickets.new"
    assert "https://tickets.new" in bot.messages[-1].text
    assert "https://tickets.new" in list_message.text

    main.vk_default_ticket_link_sessions[1] = main.VkDefaultTicketLinkSession(
        source_id=vid,
        page=1,
        message=list_message,
    )
    message.text = "-"
    await main.handle_vk_ticket_link_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        (link_val,) = await cur.fetchone()
    assert link_val is None

    main.vk_default_ticket_link_sessions[1] = main.VkDefaultTicketLinkSession(
        source_id=vid,
        page=1,
        message=list_message,
    )
    message.text = "ftp://invalid"
    await main.handle_vk_ticket_link_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        (link_val,) = await cur.fetchone()
    assert link_val is None
    assert "Неверный формат" in bot.messages[-1].text
    assert 1 in main.vk_default_ticket_link_sessions


@pytest.mark.asyncio
async def test_vk_default_location_message_updates_db(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "One", None, None, None),
        )
        await conn.commit()
        cur = await conn.execute("SELECT id FROM vk_source")
        (vid,) = await cur.fetchone()

    bot = DummyBot()
    list_message = DummyMessage(1, "", None)
    main.vk_default_location_sessions.clear()
    main.vk_default_location_sessions[1] = main.VkDefaultLocationSession(
        source_id=vid,
        page=1,
        message=list_message,
    )
    message = SimpleNamespace(
        chat=SimpleNamespace(id=1),
        from_user=SimpleNamespace(id=1),
        text="Калининград",
    )
    await main.handle_vk_location_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT location FROM vk_source WHERE id=?", (vid,))
        (location_val,) = await cur.fetchone()
    assert location_val == "Калининград"
    assert "Калининград" in bot.messages[-1].text
    assert "Калининград" in list_message.text

    main.vk_default_location_sessions[1] = main.VkDefaultLocationSession(
        source_id=vid,
        page=1,
        message=list_message,
    )
    message.text = "-"
    await main.handle_vk_location_message(message, db, bot)
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT location FROM vk_source WHERE id=?", (vid,))
        (location_val,) = await cur.fetchone()
    assert location_val is None
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


@pytest.mark.asyncio
async def test_build_event_payload_uses_default_ticket_link(monkeypatch):
    captured = {}

    async def fake_parse(text, **kwargs):
        captured["text"] = text
        return [{"title": "T", "date": "2099-01-01"}]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    draft = await vk_intake.build_event_payload_from_vk(
        "text", default_ticket_link="https://tickets.example"
    )

    assert "https://tickets.example" in captured["text"]
    assert draft.links == ["https://tickets.example"]


@pytest.mark.asyncio
async def test_build_event_payload_preserves_llm_ticket_link(monkeypatch):
    captured = {}

    async def fake_parse(text, **kwargs):
        captured["text"] = text
        return [
            {
                "title": "T",
                "date": "2099-01-01",
                "ticket_link": "https://real.example",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    draft = await vk_intake.build_event_payload_from_vk(
        "text", default_ticket_link="https://tickets.example"
    )

    assert "https://tickets.example" in captured["text"]
    assert draft.links == ["https://real.example"]
