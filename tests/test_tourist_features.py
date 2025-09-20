import pytest

from aiogram import types

import main
from main import Database, Event, User, append_tourist_block, build_event_card_message


@pytest.fixture(autouse=True)
def _clear_tourist_sessions():
    main.tourist_reason_sessions.clear()
    main.tourist_note_sessions.clear()


class DummyBot:
    def __init__(self):
        self.edited_text_calls = []
        self.edited_markup_calls = []
        self.sent_messages = []

    async def edit_message_text(self, *, chat_id, message_id, text, reply_markup=None):
        self.edited_text_calls.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": reply_markup,
            }
        )

    async def edit_message_caption(self, *, chat_id, message_id, caption, reply_markup=None):
        await self.edit_message_text(
            chat_id=chat_id, message_id=message_id, text=caption, reply_markup=reply_markup
        )

    async def edit_message_reply_markup(self, *, chat_id, message_id, reply_markup=None):
        self.edited_markup_calls.append(
            {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup}
        )

    async def send_message(self, chat_id, text):
        self.sent_messages.append({"chat_id": chat_id, "text": text})


def patch_answer(monkeypatch):
    calls = []

    async def fake_answer(self, text=None, show_alert=False):
        calls.append({"callback": self, "text": text, "show_alert": show_alert})

    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer, raising=False)
    return calls


def make_callback(data: str, message: types.Message, user_id: int = 1) -> types.CallbackQuery:
    return types.CallbackQuery.model_validate(
        {
            "id": "cb",
            "data": data,
            "from": {"id": user_id, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "message": message.model_dump(),
        }
    )


@pytest.mark.parametrize(
    "base_rows,source",
    [
        ([[types.InlineKeyboardButton(text="A", callback_data="x")]], "tg"),
        (
            [
                [types.InlineKeyboardButton(text="A", callback_data="x")],
                [types.InlineKeyboardButton(text="B", callback_data="y")],
            ],
            "vk",
        ),
    ],
)
def test_tourist_block_appended(base_rows, source):
    event = Event(
        id=1,
        title="T",
        description="",
        date="2025-09-01",
        time="10:00",
        location_name="L",
        source_text="S",
    )
    rows = append_tourist_block(base_rows, event, source)
    flat = [btn.callback_data for row in rows for btn in row]
    assert any(cb and cb.startswith("tourist:yes:") for cb in flat)
    assert any(cb and cb.startswith("tourist:note:start:") for cb in flat)


@pytest.mark.asyncio
async def test_tourist_yes_callback_updates_event(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        event = Event(
            title="Title",
            description="",
            date="2025-09-01",
            time="10:00",
            location_name="Loc",
            source_text="Src",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        base_rows = append_tourist_block([[types.InlineKeyboardButton(text="Edit", callback_data="edit")]], event, "tg")
        markup = types.InlineKeyboardMarkup(inline_keyboard=base_rows)
        text = build_event_card_message("Event added", event, ["title: Title"])
    message = types.Message.model_validate(
        {
            "message_id": 100,
            "date": 0,
            "chat": {"id": 10, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    callback = make_callback(f"tourist:yes:{event_id}:tg", message)
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(callback, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_label == 1
        assert updated.tourist_label_by == 1
        assert updated.tourist_label_source == "tg"
    assert any(call["text"] == "Отмечено" for call in answers)
    assert bot.edited_text_calls
    assert "🌍 Туристам: Да" in bot.edited_text_calls[-1]["text"]


@pytest.mark.asyncio
async def test_tourist_factor_flow(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        event = Event(
            title="Title",
            description="",
            date="2025-09-01",
            time="10:00",
            location_name="Loc",
            source_text="Src",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=append_tourist_block([[types.InlineKeyboardButton(text="Edit", callback_data="edit")]], event, "tg")
        )
        text = build_event_card_message("Event added", event, ["title: Title"])
    message = types.Message.model_validate(
        {
            "message_id": 200,
            "date": 0,
            "chat": {"id": 20, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    cb_menu = make_callback(f"tourist:fxmenu:{event_id}:tg", message)
    await main.process_request(cb_menu, db, bot)
    assert main.tourist_reason_sessions
    cb_toggle = make_callback(f"tourist:fx:{event_id}:history:tg", message)
    await main.process_request(cb_toggle, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_factors == ["history"]
    cb_back = make_callback(f"tourist:fxback:{event_id}:tg", message)
    await main.process_request(cb_back, db, bot)
    assert not main.tourist_reason_sessions
    assert any(call["text"] == "Отмечено" for call in answers)


@pytest.mark.asyncio
async def test_tourist_factor_timeout(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        event = Event(
            title="Title",
            description="",
            date="2025-09-01",
            time="10:00",
            location_name="Loc",
            source_text="Src",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=append_tourist_block([[types.InlineKeyboardButton(text="Edit", callback_data="edit")]], event, "tg")
        )
        text = build_event_card_message("Event added", event, ["title: Title"])
    message = types.Message.model_validate(
        {
            "message_id": 300,
            "date": 0,
            "chat": {"id": 30, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    cb = make_callback(f"tourist:fx:{event_id}:history:tg", message)
    bot = DummyBot()

    patch_answer(monkeypatch)
    await main.process_request(cb, db, bot)
    assert bot.sent_messages
    assert "Сессия" in bot.sent_messages[-1]["text"]


@pytest.mark.asyncio
async def test_tourist_note_flow(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        event = Event(
            title="Title",
            description="",
            date="2025-09-01",
            time="10:00",
            location_name="Loc",
            source_text="Src",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=append_tourist_block([[types.InlineKeyboardButton(text="Edit", callback_data="edit")]], event, "tg")
        )
        text = build_event_card_message("Event added", event, ["title: Title"])
    message = types.Message.model_validate(
        {
            "message_id": 400,
            "date": 0,
            "chat": {"id": 40, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    cb = make_callback(f"tourist:note:start:{event_id}:tg", message)
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(cb, db, bot)
    assert main.tourist_note_sessions
    note_message = types.Message.model_validate(
        {
            "message_id": 500,
            "date": 0,
            "chat": {"id": 40, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Замечательное событие",
        }
    )
    await main.handle_tourist_note_message(note_message, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_note == "Замечательное событие"
    assert bot.sent_messages and "Комментарий сохранён" in bot.sent_messages[-1]["text"]
    assert bot.edited_text_calls
    assert "📝 есть комментарий" in bot.edited_text_calls[-1]["text"]
    assert answers and answers[0]["text"] == "Ожидаю"


@pytest.mark.asyncio
async def test_tourist_note_timeout(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    message = types.Message.model_validate(
        {
            "message_id": 600,
            "date": 0,
            "chat": {"id": 60, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Комментарий",
        }
    )
    bot = DummyBot()
    await main.handle_tourist_note_message(message, db, bot)
    assert bot.sent_messages
    assert "Сессия" in bot.sent_messages[-1]["text"]


@pytest.mark.asyncio
async def test_tourist_note_clear(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.get_session() as session:
        session.add(User(user_id=1))
        event = Event(
            title="Title",
            description="",
            date="2025-09-01",
            time="10:00",
            location_name="Loc",
            source_text="Src",
            tourist_note="Старый комментарий",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)
        event_id = event.id
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=append_tourist_block([[types.InlineKeyboardButton(text="Edit", callback_data="edit")]], event, "tg")
        )
        text = build_event_card_message("Event added", event, ["title: Title"])
    message = types.Message.model_validate(
        {
            "message_id": 700,
            "date": 0,
            "chat": {"id": 70, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    cb = make_callback(f"tourist:note:clear:{event_id}:tg", message)
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(cb, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_note is None
    assert any(call["text"] == "Отмечено" for call in answers)
    assert bot.edited_text_calls
    assert "📝" not in bot.edited_text_calls[-1]["text"]
