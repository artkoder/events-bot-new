import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from datetime import datetime, timedelta, timezone
import pytest
from aiogram import types
from types import SimpleNamespace

import main
import digests
from main import Database, User, Event

class DummyBot:
    def __init__(self):
        self.messages = []
        self.media_groups = []
        self.edited = []
        self.deleted = []

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

    async def edit_message_reply_markup(self, chat_id, message_id, reply_markup):
        for idx, m in enumerate(self.messages):
            if m.message_id == message_id:
                data = m.model_dump()
                data["reply_markup"] = reply_markup
                self.messages[idx] = types.Message.model_validate(data)
                break
        self.edited.append((chat_id, message_id, reply_markup))

    async def delete_message(self, chat_id, message_id):
        self.deleted.append((chat_id, message_id))


def get_menu_button(markup: types.InlineKeyboardMarkup, digest_type: str) -> types.InlineKeyboardButton:
    for row in markup.inline_keyboard:
        for btn in row:
            data = getattr(btn, "callback_data", "") or ""
            if data.startswith(f"digest:select:{digest_type}:"):
                return btn
    raise AssertionError(f"Button for digest type {digest_type!r} not found")


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_handle_digest_sends_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
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

    async def fake_extract(url, **kw):
        return ["https://example.com/img.jpg"]

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "lectures").callback_data.split(":")[-1]
    async def answer(**kw):
        return None
    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:lectures:{digest_id}",
        answer=answer,
    )
    await main.handle_digest_select_lectures(cb, db, bot)
    assert bot.media_groups
    caption = bot.media_groups[0][1][0].caption
    assert "<a href=" in caption
    panel = bot.messages[-1]
    assert panel.text.startswith("Управление дайджестом лекций")
    assert panel.reply_markup.inline_keyboard[0][0].callback_data.startswith("dg:t:")


@pytest.mark.asyncio
async def test_handle_digest_sends_masterclasses_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        ev = Event(
            title="M1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="мастер-класс",
            telegraph_url="https://telegra.ph/test2",
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

    async def fake_extract(url, **kw):
        return ["https://example.com/img2.jpg"]

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "masterclasses").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:masterclasses:{digest_id}",
        answer=answer,
    )
    await main.handle_digest_select_masterclasses(cb, db, bot)
    assert bot.media_groups
    panel = bot.messages[-1]
    assert panel.text.startswith("Управление дайджестом мастер-классов")
    session = main.digest_preview_sessions[digest_id]
    assert session["items_noun"] == "мастер-классов"
    assert session["items"][0]["event_type"] == "мастер-класс"
    assert session["items"][0]["norm_description"] == "d"
    assert session["items"][0]["date"]
    assert session["items"][0]["end_date"] is None


@pytest.mark.asyncio
async def test_handle_digest_sends_exhibitions_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        ev = Event(
            title="E1",
            description="d",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="выставка",
            telegraph_url="https://telegra.ph/test3",
            end_date=dt.strftime("%Y-%m-%d"),
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

    async def fake_extract(url, **kw):
        return ["https://example.com/img3.jpg"]

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "exhibitions").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:exhibitions:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_exhibitions(cb, db, bot)
    assert bot.media_groups
    panel = bot.messages[-1]
    assert panel.text.startswith("Управление дайджестом выставок")
    session = main.digest_preview_sessions[digest_id]
    assert session["items_noun"] == "выставок"
    item = session["items"][0]
    assert item["event_type"] == "выставка"
    assert item["norm_description"] == "d"
    assert item["date"]
    assert item["end_date"] == ev.date


@pytest.mark.asyncio
async def test_handle_digest_sends_psychology_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        ev = Event(
            title="P1",
            description="Описание",
            date=dt.strftime("%Y-%m-%d"),
            time="12:00",
            location_name="loc",
            source_text="s",
            event_type="лекция",
            telegraph_url="https://telegra.ph/test4",
            topics=["Психология"],
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

    async def fake_extract(url, **kw):
        return ["https://example.com/img4.jpg"]

    recorded_payloads: list[list[dict[str, object]]] = []

    async def fake_psych_intro(n, horizon_days, payload):
        recorded_payloads.append(payload)
        return "Интро"

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)
    monkeypatch.setattr(main, "compose_psychology_intro_via_4o", fake_psych_intro)
    monkeypatch.setattr(digests, "compose_psychology_intro_via_4o", fake_psych_intro)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "psychology").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:psychology:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_psychology(cb, db, bot)
    assert bot.media_groups
    panel = bot.messages[-1]
    expected_panel_text = (
        "Управление дайджестом психологии\n"
        "Выключите лишнее и нажмите «Обновить превью»."
    )
    assert panel.text == expected_panel_text
    session_data = main.digest_preview_sessions[digest_id]
    assert session_data["items_noun"] == "психологических событий"
    assert session_data["items"][0]["norm_topics"] == ["PSYCHOLOGY"]
    assert session_data["intro_html"] == "Интро"
    assert "Интро" in session_data.get("current_caption_html", "")
    assert panel.reply_markup.inline_keyboard[0][0].text.startswith("✅ 1")
    assert panel.reply_markup.inline_keyboard[-2][0].text == "🔄 Обновить превью"
    assert panel.reply_markup.inline_keyboard[-1][0].text == "🗑 Скрыть панель"

    cb_refresh = SimpleNamespace(
        data=f"dg:r:{digest_id}",
        from_user=SimpleNamespace(id=1),
        message=panel,
        answer=answer,
        id="2",
    )

    await main.handle_digest_refresh(cb_refresh, bot)
    assert recorded_payloads
    assert recorded_payloads[-1][0]["topics"] == ["PSYCHOLOGY"]


@pytest.mark.asyncio
async def test_handle_digest_sends_science_pop_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        session.add(
            Event(
                title="Science",
                description="d",
                date=dt.strftime("%Y-%m-%d"),
                time="18:00",
                location_name="loc",
                source_text="s",
                event_type="лекция",
                telegraph_url="https://telegra.ph/science",
                topics=["SCIENCE_POP"],
            )
        )
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

    async def fake_extract(url, **kw):
        return ["https://example.com/science.jpg"]

    async def fake_intro(count, horizon, titles, *, event_noun):
        return f"{count}|{horizon}|{event_noun}"

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)
    monkeypatch.setattr(main, "compose_digest_intro_via_4o", fake_intro)
    monkeypatch.setattr(digests, "compose_digest_intro_via_4o", fake_intro)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = (
        get_menu_button(menu_msg.reply_markup, "science_pop")
        .callback_data.split(":")[-1]
    )

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:science_pop:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_science_pop(cb, db, bot)
    assert bot.media_groups
    panel = bot.messages[-1]
    expected_panel_text = (
        "Управление дайджестом научпопа\n"
        "Выключите лишнее и нажмите «Обновить превью»."
    )
    assert panel.text == expected_panel_text
    session_data = main.digest_preview_sessions[digest_id]
    assert session_data["items_noun"] == "научно-популярных событий"
    assert session_data["digest_type"] == "science_pop"
    assert session_data["items"][0]["norm_topics"] == ["SCIENCE_POP"]
    assert "научно-популярных событий" in session_data["intro_html"]


@pytest.mark.asyncio
async def test_handle_digest_sends_networking_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        session.add(
            Event(
                title="Business Breakfast",
                description="Networking",
                date=dt.strftime("%Y-%m-%d"),
                time="12:00",
                location_name="Cafe",
                source_text="s",
                event_type="лекция",
                telegraph_url="https://telegra.ph/networking",
                topics=["NETWORKING"],
            )
        )
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

    async def fake_extract(url, **kw):
        return []

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "networking").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:networking:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_networking(cb, db, bot)
    session_data = main.digest_preview_sessions[digest_id]
    assert session_data["items_noun"] == "нетворкингов"
    assert session_data["digest_type"] == "networking"
    assert session_data["items"][0]["norm_topics"] == ["NETWORKING"]


@pytest.mark.asyncio
async def test_handle_digest_sends_entertainment_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        base = datetime.now(timezone.utc)

        def add(offset, title, topics):
            dt = base + timedelta(days=offset)
            session.add(
                Event(
                    title=title,
                    description="Fun",
                    date=dt.strftime("%Y-%m-%d"),
                    time="12:00",
                    location_name="Club",
                    source_text="s",
                    event_type="вечеринка",
                    telegraph_url="https://telegra.ph/ent",
                    topics=topics,
                )
            )

        add(1, "Standup", ["STANDUP"])
        add(2, "Quiz", ["QUIZ_GAMES"])
        session.add(
            Event(
                title="Lecture",
                description="No",
                date=(base + timedelta(days=3)).strftime("%Y-%m-%d"),
                time="12:00",
                location_name="Hall",
                source_text="s",
                event_type="лекция",
                topics=["LECTURES"],
            )
        )
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

    async def fake_extract(url, **kw):
        return []

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "entertainment").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:entertainment:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_entertainment(cb, db, bot)
    session_data = main.digest_preview_sessions[digest_id]
    assert session_data["items_noun"] == "развлечений"
    assert session_data["digest_type"] == "entertainment"
    titles = [item["title"] for item in session_data["items"]]
    assert "Lecture" not in titles
    assert set(titles) == {"Standup", "Quiz"}


@pytest.mark.asyncio
async def test_handle_digest_sends_movies_preview(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        session.add(
            Event(
                title="Cinema Night",
                description="Film",
                date=dt.strftime("%Y-%m-%d"),
                time="18:00",
                location_name="Cinema",
                source_text="s",
                event_type="кинопоказ",
                telegraph_url="https://telegra.ph/movies",
            )
        )
        session.add(
            Event(
                title="Other",
                description="No",
                date=dt.strftime("%Y-%m-%d"),
                time="18:00",
                location_name="Hall",
                source_text="s",
                event_type="лекция",
            )
        )
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

    async def fake_extract(url, **kw):
        return []

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "extract_catbox_covers_from_telegraph", fake_extract)

    await main.show_digest_menu(msg, db, bot)
    menu_msg = bot.messages[0]
    digest_id = get_menu_button(menu_msg.reply_markup, "movies").callback_data.split(":")[-1]

    async def answer(**kw):
        return None

    cb = SimpleNamespace(
        id="1",
        from_user=SimpleNamespace(id=1),
        message=menu_msg,
        data=f"digest:select:movies:{digest_id}",
        answer=answer,
    )

    await main.handle_digest_select_movies(cb, db, bot)
    session_data = main.digest_preview_sessions[digest_id]
    assert session_data["items_noun"] == "кинопоказов"
    assert session_data["digest_type"] == "movies"
    assert [item["title"] for item in session_data["items"]] == ["Cinema Night"]


@pytest.mark.asyncio
async def test_digest_preview_deduplicates_media(monkeypatch):
    session = {
        "chat_id": 123,
        "intro_html": "Intro",
        "footer_html": "Footer",
        "items": [
            {
                "index": 1,
                "line_html": "<b>Event 1</b>",
                "cover_url": "https://example.com/img.jpg",
            },
            {
                "index": 2,
                "line_html": "<b>Event 2</b>",
                "cover_url": "https://example.com/img.jpg",
            },
        ],
        "channels": [],
    }

    async def fake_compose(intro_html, lines_html, footer_html, *, digest_id):
        parts = [intro_html] if intro_html else []
        parts.extend(lines_html)
        if footer_html:
            parts.append(footer_html)
        return "\n".join(parts), lines_html

    def fake_attach(media, caption_html):
        return False, len(caption_html)

    monkeypatch.setattr(main, "compose_digest_caption", fake_compose)
    monkeypatch.setattr(main, "attach_caption_if_fits", fake_attach)

    bot = DummyBot()
    await main._send_preview(session, "digest-test", bot)

    assert session["current_media_urls"] == ["https://example.com/img.jpg"]
    caption = session["current_caption_html"]
    assert "Event 1" in caption
    assert "Event 2" in caption


def test_help_contains_digest():
    assert any(cmd["usage"].startswith("/digest") for cmd in main.HELP_COMMANDS)
