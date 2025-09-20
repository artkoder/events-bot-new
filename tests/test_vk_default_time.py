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
        for idx in range(1, 12):
            await conn.execute(
                "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
                (
                    idx,
                    f"club{idx}",
                    f"Name {idx}",
                    "Loc" if idx == 1 else None,
                    "19:00" if idx == 1 else None,
                ),
            )
        status_bases = {
            "pending": 1000,
            "locked": 2000,
            "skipped": 3000,
            "imported": 4000,
            "rejected": 5000,
        }
        for status, count in (
            ("pending", 2),
            ("locked", 1),
            ("skipped", 1),
            ("imported", 1),
            ("rejected", 1),
        ):
            for extra in range(count):
                await conn.execute(
                    "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, has_date, status) VALUES(?,?,?,?,?,?,?)",
                    (
                        1,
                        status_bases[status] + extra,
                        0,
                        "text",
                        None,
                        0,
                        status,
                    ),
                )
        await conn.commit()

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT id FROM vk_source ORDER BY id")
        source_ids = [row[0] for row in await cur.fetchall()]

    bot = DummyBot()
    msg = SimpleNamespace(chat=SimpleNamespace(id=1))
    await main.handle_vk_list(msg, db, bot)

    assert bot.messages, "no message sent"
    first_page = bot.messages[0]
    lines = first_page.text.splitlines()
    assert lines[0] == "Страница 1/2"
    assert lines[1].startswith("1. Name 1 (vk.com/club1) — id=1")
    assert "типовое время: 19:00" in lines[1]
    header_parts = [part.strip() for part in lines[2].split("|")]
    assert header_parts == ["Pending", "Locked", "Skipped", "Imported", "Rejected"]
    counts_parts = [part.strip() for part in lines[3].split("|")]
    assert counts_parts == ["2", "1", "1", "1", "1"]
    second_header = [part.strip() for part in lines[5].split("|")]
    assert second_header == header_parts
    second_counts = [part.strip() for part in lines[6].split("|")]
    assert second_counts == ["0", "0", "0", "0", "0"]

    buttons = first_page.reply_markup.inline_keyboard
    assert buttons[0][0].text == "❌ 1"
    assert buttons[0][0].callback_data == f"vkdel:1:{source_ids[0]}"
    assert buttons[0][1].text == "🕒 1"
    assert buttons[0][1].callback_data == f"vkdt:1:{source_ids[0]}"
    assert buttons[-1][0].text == "➡️"
    assert buttons[-1][0].callback_data == "vksrcpage:2"

    await main.handle_vk_list(msg, db, bot, page=2)
    assert len(bot.messages) == 2
    second_page = bot.messages[1]
    second_lines = second_page.text.splitlines()
    assert second_lines[0] == "Страница 2/2"
    assert second_lines[1].startswith("11. Name 11")
    second_counts_parts = [part.strip() for part in second_lines[3].split("|")]
    assert second_counts_parts == ["0", "0", "0", "0", "0"]
    buttons_page2 = second_page.reply_markup.inline_keyboard
    assert buttons_page2[0][0].callback_data == f"vkdel:2:{source_ids[-1]}"
    assert buttons_page2[0][1].callback_data == f"vkdt:2:{source_ids[-1]}"
    assert buttons_page2[-1][0].callback_data == "vksrcpage:1"


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
    main.vk_default_time_sessions[1] = main.VkDefaultTimeSession(source_id=vid)
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

    main.vk_default_time_sessions[1] = main.VkDefaultTimeSession(source_id=vid)
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
