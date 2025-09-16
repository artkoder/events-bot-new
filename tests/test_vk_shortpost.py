import os, sys
import pytest
import os, sys
from types import SimpleNamespace
from datetime import date as real_date

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
import main
from main import Database
from models import Event
from aiogram import types

class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(SimpleNamespace(chat_id=chat_id, text=text))
        return SimpleNamespace(message_id=1)


@pytest.mark.asyncio
async def test_vkrev_fetch_photos_uses_user_token(monkeypatch):
    calls = []

    async def fake_api(method, params, db=None, bot=None, token=None, token_kind=None, **kwargs):
        calls.append((token, token_kind))
        return {"response": []}

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "_vk_user_token", lambda: "user-token")

    photos = await main._vkrev_fetch_photos(1, 2, None, None)

    assert photos == []
    assert calls == [("user-token", "user")]


@pytest.mark.asyncio
async def test_vkrev_fetch_photos_no_user_token(monkeypatch):
    async def fake_api(*args, **kwargs):  # pragma: no cover
        raise AssertionError("_vk_api should not be called without user token")

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "_vk_user_token", lambda: None)

    photos = await main._vkrev_fetch_photos(1, 2, None, None)

    assert photos == []


@pytest.mark.asyncio
async def test_collect_photo_ids():
    items = [
        {
            "attachments": [
                {"type": "photo", "photo": {"owner_id": 1, "id": 1}},
                {"type": "photo", "photo": {"owner_id": 1, "id": 2, "access_key": "k"}},
            ],
            "copy_history": [
                {
                    "attachments": [
                        {"type": "photo", "photo": {"owner_id": 2, "id": 3}},
                        {"type": "photo", "photo": {"owner_id": 1, "id": 1}},
                    ]
                }
            ],
        }
    ]
    photos = main._vkrev_collect_photo_ids(items, 10)
    assert photos == ["photo2_3", "photo1_1", "photo1_2_k"]


@pytest.mark.asyncio
async def test_shortpost_wall_post(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, imported_event_id, review_batch) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, "imported", 77, "b"),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, city, source_text, telegraph_url) VALUES(?,?,?,?,?,?,?,?,?)",
            (77, "Test", "d", "2025-09-27", "19:00", "Place", "City", "source", "https://t")
        )
        await conn.commit()
    monkeypatch.setenv("VK_AFISHA_GROUP_ID", "-5")
    main.VK_AFISHA_GROUP_ID = "-5"
    calls = []

    async def fake_api(method, params, db=None, bot=None, token=None, **kwargs):
        calls.append((method, params))
        if method == "wall.post":
            assert params["from_group"] == 1
            assert params["attachments"] == "photo1_2,https://t"
            assert len(params["message"]) <= 4096
            tags = [w for w in params["message"].split() if w.startswith("#")]
            assert 5 <= len(tags) <= 7
            return {"response": {"post_id": 42}}
        raise AssertionError

    async def fake_vk_api(method, **params):
        if method == "wall.getById":
            return [
                {
                    "attachments": [
                        {"type": "photo", "photo": {"owner_id": 1, "id": 2}},
                    ]
                }
            ]
        raise AssertionError

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    async def fake_build(event, src, max_sent):
        return "short"
    monkeypatch.setattr(main, "build_short_vk_text", fake_build)
    async def fake_ask(prompt, **kwargs):
        return "#a #b #c #d #e"
    monkeypatch.setattr(main, "ask_4o", fake_ask)
    bot = DummyBot()
    async def fake_answer(self, *args, **kwargs):
        return None
    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)
    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "chat_instance": "1",
            "data": "vkrev:shortpost:77",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb._bot = bot
    await main.handle_vk_review_cb(cb, db, bot)
    # now simulate publish from the same chat
    cb_pub = types.CallbackQuery.model_validate(
        {
            "id": "2",
            "from": {"id": 10, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "data": "vkrev:shortpost_pub:77",
            "message": {"message_id": 2, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb_pub._bot = bot
    await main.handle_vk_review_cb(cb_pub, db, bot)
    assert any("✅ Опубликовано" in m.text for m in bot.messages)
    assert any(m == "wall.post" for m, _ in calls)
    async with db.get_session() as session:
        ev = await session.get(Event, 77)
        assert ev.vk_repost_url == "https://vk.com/wall-5_42"


@pytest.mark.asyncio
async def test_shortpost_publish_without_photo(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, imported_event_id, review_batch) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, "imported", 77, "b"),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, city, source_text, telegraph_url) VALUES(?,?,?,?,?,?,?,?,?)",
            (77, "Test", "d", "2025-09-27", "19:00", "Place", "City", "source", "https://t"),
        )
        await conn.commit()
    monkeypatch.setenv("VK_AFISHA_GROUP_ID", "-5")
    main.VK_AFISHA_GROUP_ID = "-5"

    async def fake_api(method, params, db=None, bot=None, token=None, **kwargs):
        if method == "wall.getById":
            return {"response": []}
        if method == "wall.post":
            assert "attachments" not in params
            return {"response": {"post_id": 43}}
        raise AssertionError

    monkeypatch.setattr(main, "_vk_api", fake_api)

    async def fake_build(event, src, max_sent):
        return "short"

    monkeypatch.setattr(main, "build_short_vk_text", fake_build)

    async def fake_ask(prompt, **kwargs):
        return "#a #b #c #d #e"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    bot = DummyBot()

    async def fake_answer(self, *args, **kwargs):
        return None

    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)

    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "chat_instance": "1",
            "data": "vkrev:shortpost:77",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb._bot = bot
    await main.handle_vk_review_cb(cb, db, bot)

    cb_pub = types.CallbackQuery.model_validate(
        {
            "id": "2",
            "from": {"id": 10, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "data": "vkrev:shortpost_pub:77",
            "message": {"message_id": 2, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb_pub._bot = bot
    await main.handle_vk_review_cb(cb_pub, db, bot)

    assert any("✅ Опубликовано" in m.text for m in bot.messages)

    async with db.get_session() as session:
        ev = await session.get(Event, 77)
        assert ev.vk_repost_url == "https://vk.com/wall-5_43"


@pytest.mark.asyncio
async def test_shortpost_captcha(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, imported_event_id, review_batch) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, "imported", 77, "b"),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, city, source_text) VALUES(?,?,?,?,?,?,?,?)",
            (77, "Test", "d", "2025-09-27", "19:00", "Place", "City", "source"),
        )
        await conn.commit()
    monkeypatch.setenv("VK_AFISHA_GROUP_ID", "-5")
    main.VK_AFISHA_GROUP_ID = "-5"
    async def fake_api(method, params, db=None, bot=None, token=None, **kwargs):
        if method == "wall.getById":
            return {
                "response": [
                    {
                        "attachments": [
                            {"type": "photo", "photo": {"owner_id": 1, "id": 2}}
                        ]
                    }
                ]
            }
        raise main.VKAPIError(14, "captcha")
    monkeypatch.setattr(main, "_vk_api", fake_api)
    async def fake_build(event, src, max_sent):
        return "short"
    monkeypatch.setattr(main, "build_short_vk_text", fake_build)
    async def fake_ask(prompt, **kwargs):
        return "#a #b #c #d #e"
    monkeypatch.setattr(main, "ask_4o", fake_ask)
    bot = DummyBot()
    async def fake_answer(self, *args, **kwargs):
        return None
    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)
    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "chat_instance": "1",
            "data": "vkrev:shortpost:77",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb._bot = bot
    await main.handle_vk_review_cb(cb, db, bot)
    cb_pub = types.CallbackQuery.model_validate(
        {
            "id": "2",
            "from": {"id": 10, "is_bot": False, "first_name": "A"},
            "chat_instance": "1",
            "data": "vkrev:shortpost_pub:77",
            "message": {"message_id": 2, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )
    cb_pub._bot = bot
    await main.handle_vk_review_cb(cb_pub, db, bot)
    texts = [m.text for m in bot.messages]
    assert "Капча, публикацию не делаем. Попробуйте позже" in texts


@pytest.mark.asyncio
async def test_shortpost_no_time(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="",
        location_name="Place",
        source_text="src",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")
    assert "⏰" not in msg
    assert "[https://vk.com/wall-1_1|Источник]" in msg


@pytest.mark.asyncio
async def test_shortpost_midnight_time_hidden(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="00:00",
        location_name="Place",
        source_text="src",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")
    assert "⏰" not in msg


@pytest.mark.asyncio
async def test_shortpost_city_not_duplicated(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="19:00",
        location_name="Place",
        location_address="City",
        city="City",
        source_text="src",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")
    location_line = next(
        line for line in msg.splitlines() if line.startswith("📍 Локация:")
    )
    assert location_line == "📍 Локация: Place, City"
    assert "City, City" not in msg


@pytest.mark.asyncio
async def test_shortpost_type_line_for_hyphenated_type(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "короткое описание"

    async def fake_ask(prompt, **kwargs):
        return "#доптег1 #доптег2"

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "ask_4o", fake_ask)

    ev = Event(
        id=1,
        title="Мастерство",
        description="d",
        date="2025-09-27",
        time="19:00",
        location_name="Place",
        city="Калининград",
        source_text="src",
        event_type="мастер-класс",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")

    lines = msg.splitlines()
    date_idx = next(i for i, line in enumerate(lines) if line.startswith("🗓"))
    assert lines[date_idx - 1] == "#мастеркласс"

    hashtags_line = lines[-1]
    assert "#мастеркласс" in hashtags_line
    assert "#мастер_класс" in hashtags_line


@pytest.mark.asyncio
async def test_shortpost_plain_type_hashtag_not_repeated(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "короткое описание"

    async def fake_ask(prompt, **kwargs):
        return "#доптег1 #доптег2"

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "ask_4o", fake_ask)

    ev = Event(
        id=1,
        title="Экспозиция",
        description="d",
        date="2025-09-27",
        time="18:00",
        location_name="Place",
        city="Калининград",
        source_text="src",
        event_type="выставка",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")

    lines = msg.splitlines()
    date_idx = next(i for i, line in enumerate(lines) if line.startswith("🗓"))
    assert lines[date_idx - 1] == "#выставка"

    hashtags_line = lines[-1]
    assert "#выставка" not in hashtags_line.split()


@pytest.mark.asyncio
async def test_shortpost_ongoing_exhibition(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)

    class FakeDate(real_date):
        @classmethod
        def today(cls):
            return cls(2025, 9, 28)

    monkeypatch.setattr(main, "date", FakeDate)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-20",
        end_date="2025-10-05",
        time="18:00",
        location_name="Place",
        source_text="src",
        event_type="выставка",
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")
    assert "🗓 по 5 октября" in msg
    assert "⏰" not in msg


@pytest.mark.asyncio
async def test_shortpost_preview_link(monkeypatch):
    async def fake_build_text(event, src, max_sent):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="",
        location_name="Place",
        source_text="src",
    )

    msg, _ = await main._vkrev_build_shortpost(
        ev, "https://vk.com/wall-1_1", for_preview=True
    )
    assert "[https://vk.com/wall-1_1|Источник]" not in msg
    assert "Источник\nhttps://vk.com/wall-1_1" in msg


@pytest.mark.asyncio
async def test_build_short_vk_tags_adds_city_hashtag(monkeypatch):
    async def fake_ask(prompt, **kwargs):
        return "#доптег1 #доптег2"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="19:00",
        location_name="Place",
        city="Санкт-Петербург",
        event_type="Лекция",
        source_text="src",
    )

    tags = await main.build_short_vk_tags(ev, "summary")

    assert "#санктпетербург" in tags
    assert tags.index("#санктпетербург") <= 2
