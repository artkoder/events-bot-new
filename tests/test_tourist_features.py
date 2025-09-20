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
    texts = [btn.text for row in rows for btn in row]
    assert f"tourist:yes:{event.id}" in flat
    assert f"tourist:fxmenu:{event.id}" in flat
    assert f"tourist:note:start:{event.id}" in flat
    assert "Интересно туристам" in texts
    assert "Причины" in texts
    assert "✍️ Комментарий" in texts


def test_build_event_card_message_without_factors():
    event = Event(
        id=1,
        title="T",
        description="",
        date="2025-09-01",
        time="10:00",
        location_name="L",
        source_text="S",
        tourist_factors=[],
        tourist_label=1,
    )
    text = build_event_card_message("Event added", event, ["title: Title"])
    assert "🧩" not in text


def test_build_event_card_message_with_factors():
    event = Event(
        id=1,
        title="T",
        description="",
        date="2025-09-01",
        time="10:00",
        location_name="L",
        source_text="S",
        tourist_factors=["targeted_for_tourists", "local_cuisine"],
        tourist_label=1,
    )
    text = build_event_card_message("Event added", event, ["title: Title"])
    assert "🧩 2 причин" in text


def test_normalize_tourist_factors_handles_legacy_codes():
    normalized = main._normalize_tourist_factors(["culture", "food", "events"])
    assert normalized == [
        "targeted_for_tourists",
        "unique_to_region",
        "local_cuisine",
    ]


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
    callback = make_callback(f"tourist:yes:{event_id}", message)
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(callback, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_label == 1
        assert updated.tourist_label_by == 1
        assert updated.tourist_label_source == "operator"
    session_state = main.tourist_reason_sessions.get(1)
    assert session_state and session_state.event_id == event_id
    assert any(call["text"] == "Отмечено" for call in answers)
    assert bot.edited_text_calls
    last_call = bot.edited_text_calls[-1]
    assert "🌍 Туристам: Да" in last_call["text"]
    assert "🧩" not in last_call["text"]
    markup = last_call["reply_markup"]
    reason_callbacks = [
        btn.callback_data
        for row in markup.inline_keyboard
        for btn in row
        if btn.callback_data
    ]
    assert f"tourist:fxdone:{event_id}" in reason_callbacks
    factor_buttons = [
        btn
        for row in last_call["reply_markup"].inline_keyboard
        for btn in row
        if btn.callback_data and btn.callback_data.startswith("tourist:fx:")
    ]
    expected_callbacks = {
        f"tourist:fx:{factor.code}:{event_id}" for factor in main.TOURIST_FACTORS
    }
    assert expected_callbacks <= {btn.callback_data for btn in factor_buttons}
    assert any(
        btn.text.startswith("➕ 🎯 Специально для туристов") for btn in factor_buttons
    )


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
    cb_menu = make_callback(f"tourist:fxmenu:{event_id}", message)
    await main.process_request(cb_menu, db, bot)
    assert main.tourist_reason_sessions
    assert any(call["text"] == "Выберите причины" for call in answers)
    cb_toggle = make_callback(
        f"tourist:fx:targeted_for_tourists:{event_id}", message
    )
    await main.process_request(cb_toggle, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_factors == ["targeted_for_tourists"]
    cb_done = make_callback(f"tourist:fxdone:{event_id}", message)
    await main.process_request(cb_done, db, bot)
    assert not main.tourist_reason_sessions
    assert any(call["text"] == "Причины сохранены" for call in answers)
    assert bot.edited_text_calls
    last_text = bot.edited_text_calls[-1]["text"]
    assert "🧩 1 причин" in last_text


@pytest.mark.asyncio
async def test_tourist_factor_skip(tmp_path, monkeypatch):
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
            "message_id": 250,
            "date": 0,
            "chat": {"id": 25, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    cb_menu = make_callback(f"tourist:fxmenu:{event_id}", message)
    await main.process_request(cb_menu, db, bot)
    cb_skip = make_callback(f"tourist:fxskip:{event_id}", message)
    await main.process_request(cb_skip, db, bot)
    assert not main.tourist_reason_sessions
    assert any(call["text"] == "Причины можно выбрать позже" for call in answers)


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
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    cb_menu = make_callback(f"tourist:fxmenu:{event_id}", message)
    await main.process_request(cb_menu, db, bot)
    assert any(call["text"] == "Выберите причины" for call in answers)
    assert bot.edited_text_calls
    menu_markup = bot.edited_text_calls[-1]["reply_markup"]
    assert any(
        btn.callback_data and btn.callback_data.startswith("tourist:fx:")
        for row in menu_markup.inline_keyboard
        for btn in row
    )

    main.tourist_reason_sessions.clear()

    cb = make_callback(f"tourist:fx:targeted_for_tourists:{event_id}", message)
    await main.process_request(cb, db, bot)
    assert len(bot.edited_text_calls) >= 2
    restored_markup = bot.edited_text_calls[-1]["reply_markup"]
    assert not any(
        btn.callback_data and btn.callback_data.startswith("tourist:fx:")
        for row in restored_markup.inline_keyboard
        for btn in row
    )
    assert any(
        btn.callback_data == f"tourist:fxmenu:{event_id}"
        for row in restored_markup.inline_keyboard
        for btn in row
    )
    assert not main.tourist_reason_sessions
    assert len(answers) >= 2
    assert answers[-1]["text"] == "Сессия истекла, откройте причины заново"
    assert not answers[-1]["show_alert"]


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
    cb = make_callback(f"tourist:note:start:{event_id}", message)
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
async def test_tourist_note_trim_long_text(tmp_path, monkeypatch):
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
            "message_id": 450,
            "date": 0,
            "chat": {"id": 45, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )
    cb = make_callback(f"tourist:note:start:{event_id}", message)
    patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(cb, db, bot)
    long_note = "A" * 600
    note_message = types.Message.model_validate(
        {
            "message_id": 455,
            "date": 0,
            "chat": {"id": 45, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": long_note,
        }
    )
    await main.handle_tourist_note_message(note_message, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_note == long_note[:500]
    assert bot.sent_messages
    assert (
        bot.sent_messages[-1]["text"]
        == "Комментарий сохранён (обрезан до 500 символов)."
    )


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
    cb = make_callback(f"tourist:note:clear:{event_id}", message)
    answers = patch_answer(monkeypatch)
    bot = DummyBot()
    await main.process_request(cb, db, bot)
    async with db.get_session() as session:
        updated = await session.get(Event, event_id)
        assert updated.tourist_note is None
    assert any(call["text"] == "Отмечено" for call in answers)
    assert bot.edited_text_calls
    assert "📝" not in bot.edited_text_calls[-1]["text"]


@pytest.mark.asyncio
async def test_tourist_label_source_always_operator(tmp_path, monkeypatch):
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
            inline_keyboard=append_tourist_block(
                [[types.InlineKeyboardButton(text="Edit", callback_data="edit")]],
                event,
                "tg",
            )
        )
        text = build_event_card_message("Event added", event, ["title: Title"])

    message = types.Message.model_validate(
        {
            "message_id": 800,
            "date": 0,
            "chat": {"id": 80, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": text,
            "reply_markup": markup.model_dump(),
        }
    )

    patch_answer(monkeypatch)
    bot = DummyBot()

    async def _assert_source():
        async with db.get_session() as session:
            refreshed = await session.get(Event, event_id)
            assert refreshed and refreshed.tourist_label_source == "operator"

    cb_yes = make_callback(f"tourist:yes:{event_id}", message)
    await main.process_request(cb_yes, db, bot)
    await _assert_source()

    cb_factor = make_callback(
        f"tourist:fx:targeted_for_tourists:{event_id}", message
    )
    await main.process_request(cb_factor, db, bot)
    await _assert_source()

    cb_no = make_callback(f"tourist:no:{event_id}", message)
    await main.process_request(cb_no, db, bot)
    await _assert_source()

    cb_note_start = make_callback(f"tourist:note:start:{event_id}", message)
    await main.process_request(cb_note_start, db, bot)

    note_message = types.Message.model_validate(
        {
            "message_id": 801,
            "date": 0,
            "chat": {"id": 80, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "Комментарий для туристов",
        }
    )
    await main.handle_tourist_note_message(note_message, db, bot)
    await _assert_source()

    cb_note_clear = make_callback(f"tourist:note:clear:{event_id}", message)
    await main.process_request(cb_note_clear, db, bot)
    await _assert_source()
