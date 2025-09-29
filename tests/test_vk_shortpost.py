import logging
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
async def test_vkrev_fetch_photos_prefers_service_token(monkeypatch):
    calls = []

    async def fake_api(
        method,
        params,
        db=None,
        bot=None,
        token=None,
        token_kind=None,
        skip_captcha=False,
        **kwargs,
    ):
        calls.append((method, params, token, token_kind, skip_captcha))
        return {"response": []}

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "VK_SERVICE_TOKEN", "service-token")
    monkeypatch.setattr(main, "_vk_user_token", lambda: "user-token")

    photos = await main._vkrev_fetch_photos(1, 2, None, None)

    assert photos == []
    assert calls == [
        ("wall.getById", {"posts": "-1_2"}, "service-token", "service", True)
    ]


@pytest.mark.asyncio
async def test_vkrev_fetch_photos_fallback_to_user_token(monkeypatch):
    calls = []

    async def fake_api(
        method,
        params,
        db=None,
        bot=None,
        token=None,
        token_kind=None,
        skip_captcha=False,
        **kwargs,
    ):
        calls.append((method, params, token, token_kind, skip_captcha))
        return {"response": []}

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "VK_SERVICE_TOKEN", None)
    monkeypatch.setattr(main, "_vk_user_token", lambda: "user-token")

    photos = await main._vkrev_fetch_photos(1, 2, None, None)

    assert photos == []
    assert calls == [
        ("wall.getById", {"posts": "-1_2"}, "user-token", "user", True)
    ]


@pytest.mark.asyncio
async def test_vkrev_fetch_photos_no_tokens(monkeypatch):
    async def fake_api(*args, **kwargs):  # pragma: no cover
        raise AssertionError("_vk_api should not be called without tokens")

    monkeypatch.setattr(main, "_vk_api", fake_api)
    monkeypatch.setattr(main, "VK_SERVICE_TOKEN", None)
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
    async def fake_build(event, src, max_sent, **kwargs):
        return "short"
    monkeypatch.setattr(main, "build_short_vk_text", fake_build)
    async def fake_ask(prompt, **kwargs):
        return "#a #b #c #d #e"
    monkeypatch.setattr(main, "ask_4o", fake_ask)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)
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
async def test_shortpost_publish_uses_cached_preview(tmp_path, monkeypatch):
    main.vk_shortpost_ops.clear()
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
    monkeypatch.setattr(main, "VK_TOKEN_AFISHA", "token")

    build_calls = 0

    async def fake_build_short_vk_text(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return "summary"

    async def fake_build_short_vk_tags(*args, **kwargs):
        return ["#tag"]

    async def fake_get_event_poster_texts(event_id, _db):
        return []

    async def fake_show_next(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_short_vk_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_build_short_vk_tags)
    monkeypatch.setattr(main, "get_event_poster_texts", fake_get_event_poster_texts)
    monkeypatch.setattr(main, "_vkrev_show_next", fake_show_next)

    posts: list[dict] = []

    async def fake_vk_api(method, **params):
        if method == "wall.getById":
            return []
        raise AssertionError

    async def fake__vk_api(method, params, db=None, bot=None, token=None, **kwargs):
        if method == "wall.post":
            posts.append(params)
            return {"response": {"post_id": 42}}
        raise AssertionError

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    monkeypatch.setattr(main, "_vk_api", fake__vk_api)

    bot = DummyBot()
    callback = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 1, "is_bot": False, "first_name": "U"},
            "chat_instance": "1",
            "data": "vkrev:shortpost:77",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 1, "type": "private"}},
        }
    )

    await main._vkrev_handle_shortpost(callback, 77, db, bot)
    assert build_calls == 1

    await main._vkrev_publish_shortpost(77, db, bot, actor_chat_id=1, operator_id=10)
    assert build_calls == 1
    assert posts and posts[0]["message"].startswith("TEST")
    lines = posts[0]["message"].split("\n")
    assert "Источник" not in lines
    assert "[https://vk.com/wall-1_2|Источник]" in lines

    main.vk_shortpost_ops.clear()


@pytest.mark.asyncio
async def test_shortpost_publish_rebuilds_without_cache(tmp_path, monkeypatch):
    main.vk_shortpost_ops.clear()
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
    monkeypatch.setattr(main, "VK_TOKEN_AFISHA", "token")

    build_calls = 0

    async def fake_build_short_vk_text(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return "summary"

    async def fake_build_short_vk_tags(*args, **kwargs):
        return ["#tag"]

    async def fake_get_event_poster_texts(event_id, _db):
        return []

    async def fake_show_next(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_short_vk_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_build_short_vk_tags)
    monkeypatch.setattr(main, "get_event_poster_texts", fake_get_event_poster_texts)
    monkeypatch.setattr(main, "_vkrev_show_next", fake_show_next)

    async def fake_vk_api(method, **params):
        if method == "wall.getById":
            return []
        raise AssertionError

    async def fake__vk_api(method, params, db=None, bot=None, token=None, **kwargs):
        if method == "wall.post":
            return {"response": {"post_id": 42}}
        raise AssertionError

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    monkeypatch.setattr(main, "_vk_api", fake__vk_api)

    bot = DummyBot()

    await main._vkrev_publish_shortpost(77, db, bot, actor_chat_id=1, operator_id=10)
    assert build_calls == 1

    main.vk_shortpost_ops.clear()
@pytest.mark.asyncio
async def test_handle_vk_review_accept_notifies_before_import(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(" "id, group_id, post_id, date, text, matched_kw, has_date, status, imported_event_id, review_batch"
            ") VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, 1, 1, 0, "", None, 0, "pending", None, "batch"),
        )
        await conn.commit()

    sent = []

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db_,
        bot_,
        operator_extra=None,
        *,
        force_festival=False,
    ):
        sent.append((chat_id, operator_id, inbox_id, batch_id))
        await bot_.send_message(chat_id, "import flow called")

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)

    bot = DummyBot()

    async def fake_answer(self, *args, **kwargs):
        return None

    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)

    cb = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 10, "is_bot": False, "first_name": "Op"},
            "chat_instance": "1",
            "data": "vkrev:accept:1",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 5, "type": "private"}},
        }
    )
    cb._bot = bot

    await main.handle_vk_review_cb(cb, db, bot)

    assert sent == [(5, 10, 1, "batch")]
    assert [m.text for m in bot.messages] == [
        "⏳ Начинаю импорт события…",
        "import flow called",
    ]

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

    async def fake_build(event, src, max_sent, **kwargs):
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
async def test_shortpost_publish_skips_group_only_photo_upload(tmp_path, monkeypatch, caplog):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, imported_event_id, review_batch) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, "imported", 77, "batch"),
        )
        await conn.execute(
            "INSERT INTO event(id, title, description, date, time, location_name, city, source_text, telegraph_url) VALUES(?,?,?,?,?,?,?,?,?)",
            (77, "Test", "d", "2025-09-27", "19:00", "Place", "City", "source", "https://t"),
        )
        await conn.commit()

    async with db.get_session() as session:
        ev = await session.get(Event, 77)
        ev.photo_urls = ["http://img1"]
        await session.commit()

    monkeypatch.setattr(main, "VK_AFISHA_GROUP_ID", "-5")
    monkeypatch.setattr(main, "VK_PHOTOS_ENABLED", True)
    monkeypatch.setattr(main, "VK_USER_TOKEN", None)
    monkeypatch.setattr(main, "VK_TOKEN_AFISHA", "group-token")
    monkeypatch.setattr(main, "VK_TOKEN", None)

    caplog.set_level(logging.INFO)

    calls: list[str] = []

    async def fake__vk_api(method, params, db=None, bot=None, token=None, **kwargs):
        calls.append(method)
        if method == "wall.post":
            return {"response": {"post_id": 99}}
        raise AssertionError(f"unexpected _vk_api call {method}")

    async def fake_vk_api(method, **params):
        if method == "wall.getById":
            return {"response": []}
        raise AssertionError(f"unexpected vk_api call {method}")

    async def fake_save_repost(db_obj, event_id, url):
        return None

    async def fake_show_next(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "_vk_api", fake__vk_api)
    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    monkeypatch.setattr(main.vk_review, "save_repost_url", fake_save_repost)
    monkeypatch.setattr(main, "_vkrev_show_next", fake_show_next)

    bot = DummyBot()

    await main._vkrev_publish_shortpost(
        77,
        db,
        bot,
        actor_chat_id=1,
        operator_id=2,
        text="text",
    )

    assert calls == ["wall.post"]
    assert not any("photos.getWallUploadServer" in rec.getMessage() for rec in caplog.records)


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
    async def fake_build(event, src, max_sent, **kwargs):
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
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
async def test_shortpost_free_event_ticket_line(monkeypatch):
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

    ev = Event(
        id=1,
        title="T",
        description="d",
        date="2025-09-27",
        time="",
        location_name="Place",
        source_text="src",
        ticket_link="https://tickets",
        is_free=True,
    )

    msg, _ = await main._vkrev_build_shortpost(ev, "https://vk.com/wall-1_1")
    assert "🆓 Бесплатно, по регистрации https://tickets" in msg
    assert "🎟 Билеты:" not in msg


@pytest.mark.asyncio
async def test_shortpost_midnight_time_hidden(monkeypatch):
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
        line for line in msg.splitlines() if line.startswith("📍 ")
    )
    assert location_line == "📍 Place, City"
    assert "City, City" not in msg


@pytest.mark.asyncio
async def test_shortpost_type_line_for_hyphenated_type(monkeypatch):
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "короткое описание"

    async def fake_ask(prompt, **kwargs):
        return "#доптег1 #доптег2"

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "ask_4o", fake_ask)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
    assert lines[date_idx + 1] == "#мастеркласс"

    hashtags_line = lines[-1]
    hashtags = hashtags_line.split()
    assert "#мастер_класс" in hashtags
    assert "#мастеркласс" not in hashtags


@pytest.mark.asyncio
async def test_shortpost_plain_type_hashtag_not_repeated(monkeypatch):
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "короткое описание"

    async def fake_ask(prompt, **kwargs):
        return "#доптег1 #доптег2"

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "ask_4o", fake_ask)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
    assert lines[date_idx + 1] == "#выставка"

    hashtags_line = lines[-1]
    assert "#выставка" not in hashtags_line.split()


@pytest.mark.asyncio
async def test_shortpost_ongoing_exhibition(monkeypatch):
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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
    async def fake_build_text(event, src, max_sent, **kwargs):
        return "short summary"

    async def fake_tags(event, summary, used_type_hashtag=None):
        return ["#a", "#b", "#c", "#d", "#e"]

    monkeypatch.setattr(main, "build_short_vk_text", fake_build_text)
    monkeypatch.setattr(main, "build_short_vk_tags", fake_tags)
    async def fake_location(parts):
        return ", ".join(filter(None, parts))
    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

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


@pytest.mark.asyncio
async def test_build_short_vk_tags_location_abbreviations(monkeypatch):
    async def fake_ask(prompt, **kwargs):
        return "#доп1 #доп2 #доп3"

    monkeypatch.setattr(main, "ask_4o", fake_ask)

    ev = Event(
        id=2,
        title="T",
        description="d",
        date="2025-09-27",
        time="19:00",
        location_name="ИЦАЭ (в КГТУ)",
        city="Калининград",
        event_type="Лекция",
        source_text="src",
    )

    tags = await main.build_short_vk_tags(ev, "summary")

    assert "#ИЦАЭ" in tags
    assert "#КГТУ" in tags
    assert len(tags) <= 7
