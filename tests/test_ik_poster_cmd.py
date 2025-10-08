from types import SimpleNamespace

import pytest

from handlers import ik_poster_cmd


class DummyState:
    def __init__(self, data: dict):
        self._data = data
        self.cleared = False

    async def get_data(self) -> dict:
        return self._data

    async def clear(self) -> None:
        self.cleared = True


class DummyBot:
    def __init__(self):
        self.photo_messages = []

    async def send_photo(self, chat_id, photo, **kwargs):
        self.photo_messages.append((chat_id, photo, kwargs))


class DummyMessage:
    def __init__(self, bot: DummyBot):
        self.bot = bot
        self.replies = []
        self.photos = []
        self.markup_cleared = False

    async def edit_reply_markup(self, *args, **kwargs):
        self.markup_cleared = True

    async def answer(self, text, **kwargs):
        self.replies.append((text, kwargs))

    async def answer_photo(self, photo, caption=None, **kwargs):
        self.photos.append((photo, caption, kwargs))


class DummyCallback:
    def __init__(self, bot: DummyBot, data: str):
        self._bot = bot
        self.data = data
        self.message = DummyMessage(bot)
        self.from_user = SimpleNamespace(full_name="Tester", username="tester")
        self.answered = False

    async def answer(self, *args, **kwargs):
        self.answered = True


@pytest.mark.asyncio
async def test_handle_mode_does_not_forward_when_operator_disabled(monkeypatch):
    monkeypatch.setenv("OPERATOR_CHAT_ID", "0")
    monkeypatch.setattr(ik_poster_cmd, "process_poster", lambda *_, **__: b"processed")

    state = DummyState({"image": b"img", "filename": "poster.jpg"})
    bot = DummyBot()
    callback = DummyCallback(bot, "ik-poster:smart")

    await ik_poster_cmd.handle_mode(callback, state)

    assert bot.photo_messages == []
