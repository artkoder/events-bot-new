import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from aiogram import types, Bot

import main


class DummyBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "https://vk.com/muzteatr39",
        "muzteatr39",
        "club231920894",
    ],
)
async def test_vk_resolve_group_variants(raw, monkeypatch):
    async def fake_vk_api(method, **params):
        if method == "utils.resolveScreenName":
            if params["screen_name"] == "muzteatr39":
                return {"type": "group", "object_id": 231920894}
            return {}
        if method == "groups.getById":
            assert params["group_ids"] in (231920894, "231920894")
            assert params.get("fields") == "screen_name"
            return [{"id": 231920894, "name": "Teatr", "screen_name": "muzteatr39"}]
        raise AssertionError("unexpected method")

    monkeypatch.setattr(main, "vk_api", fake_vk_api)
    gid, name, screen_name = await main.vk_resolve_group(raw)
    assert gid == 231920894
    assert name == "Teatr"
    assert screen_name == "muzteatr39"


@pytest.mark.asyncio
async def test_vk_add_message_error(monkeypatch, tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = DummyBot("123:abc")

    async def fake_resolve(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "vk_resolve_group", fake_resolve)
    user_id = 42
    main.vk_add_source_sessions.add(user_id)
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": user_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "U"},
            "text": "garbage",
        }
    )
    await main.handle_vk_add_message(msg, db, bot)
    assert bot.messages[-1][1] == (
        "Не удалось определить сообщество.\n"
        "Проверьте ссылку/скриннейм (пример: https://vk.com/muzteatr39).\n"
        "Технические детали: boom."
    )
