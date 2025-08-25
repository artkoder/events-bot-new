import pytest
import main
from aiogram import Bot, types
from pathlib import Path
from main import Database, User

class DummyResp:
    def __init__(self, data):
        self._data = data
    def json(self):
        return self._data


@pytest.mark.asyncio
async def test_vk_api_captcha_cached(monkeypatch):
    calls = 0
    async def fake_http_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return DummyResp({
            "error": {
                "error_code": 14,
                "error_msg": "Captcha needed",
                "captcha_sid": "sid",
                "captcha_img": "img",
            }
        })
    monkeypatch.setattr(main, "http_call", fake_http_call)
    monkeypatch.setattr(main, "_vk_captcha_needed", False)
    monkeypatch.setattr(main, "_vk_captcha_sid", None)
    monkeypatch.setattr(main, "_vk_captcha_img", None)
    monkeypatch.setattr(main, "_vk_captcha_method", None)
    monkeypatch.setattr(main, "_vk_captcha_params", None)
    with pytest.raises(main.VKAPIError) as e1:
        await main._vk_api("wall.get", {}, token="t")
    assert e1.value.code == 14
    assert e1.value.captcha_sid == "sid"
    assert e1.value.captcha_img == "img"
    assert calls == 1
    with pytest.raises(main.VKAPIError) as e2:
        await main._vk_api("wall.get", {}, token="t")
    assert e2.value.captcha_sid == "sid"
    assert e2.value.captcha_img == "img"
    assert calls == 1


class PhotoBot(Bot):
    def __init__(self, token: str):
        super().__init__(token)
        self.photos = []
        self.messages = []

    async def send_photo(self, chat_id, photo, **kwargs):
        self.photos.append((chat_id, photo, kwargs))

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs))


@pytest.mark.asyncio
async def test_handle_vk_captcha_flow(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = PhotoBot("123:abc")
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        await session.commit()
    main._vk_captcha_sid = "sid"
    main._vk_captcha_img = "img"
    main._vk_captcha_needed = True
    main._vk_captcha_method = "wall.post"
    main._vk_captcha_params = {}

    async def fake_superadmin_id(db):
        return 1

    monkeypatch.setattr(main, "get_superadmin_id", fake_superadmin_id)
    class ImgResp:
        def __init__(self, data: bytes):
            self._data = data
        async def read(self):
            return self._data
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class ImgSession:
        def get(self, url):
            return ImgResp(b"img")
    monkeypatch.setattr(main, "get_http_session", lambda: ImgSession())
    await main.notify_vk_captcha(db, bot, "img")
    from aiogram.types import BufferedInputFile
    assert bot.photos and isinstance(bot.photos[0][1], BufferedInputFile)

    async def fake_vk_api(method, params, db=None, bot=None, **kwargs):
        assert method == "wall.post"
        assert params["captcha_sid"] == "sid"
        assert params["captcha_key"] == "1234"
        return {"response": 1}

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)
    msg = types.Message.model_validate(
        {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 1, "type": "private"},
            "from": {"id": 1, "is_bot": False, "first_name": "A"},
            "text": "/captcha 1234",
        }
    )
    await main.handle_vk_captcha(msg, db, bot)
    assert main._vk_captcha_sid is None
    assert bot.messages and "Captcha solved" in bot.messages[0][1]
