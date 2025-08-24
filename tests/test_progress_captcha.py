import pytest
from aiogram import Bot, types
import main
from main import Database, User
from scheduling import BatchProgress, CoalescingScheduler


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
async def test_progress_captcha_flow(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    bot = PhotoBot("123:abc")
    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        await session.commit()

    calls = []

    async def fake_vk_api(method, params, db=None, bot=None, **kwargs):
        if method == "captcha.force":
            assert params["captcha_sid"] == "sid"
            assert params["captcha_key"] == "1234"
            return {"response": 1}
        calls.append(method)
        if len(calls) == 1:
            main._vk_captcha_needed = True
            main._vk_captcha_sid = "sid"
            main._vk_captcha_img = "img"
            await main.notify_vk_captcha(db, bot, "img")
            raise main.VKAPIError(14, "Captcha needed", "sid", "img")
        if len(calls) == 2:
            return {"response": 1}
        raise main.VKAPIError(5, "fail")

    monkeypatch.setattr(main, "_vk_api", fake_vk_api)

    progress = BatchProgress(total_events=0)
    scheduler = CoalescingScheduler(progress, on_captcha=main.vk_captcha_paused)

    async def month_job(_):
        pass

    async def vk_job(_):
        await main._vk_api("wall.post", {}, db=db, bot=bot)

    scheduler.add_job("month_pages:2025-08", month_job)
    scheduler.add_job(
        "vk_week_post:2025-30", vk_job, depends_on=["month_pages:2025-08"]
    )
    scheduler.add_job(
        "vk_week_post:2025-31",
        vk_job,
        depends_on=["vk_week_post:2025-30", "month_pages:2025-08"],
    )

    await scheduler.run()
    assert progress.status["vk_week_post:2025-30"] == "captcha"
    assert progress.status["vk_week_post:2025-31"] == "pending"
    assert bot.photos and bot.photos[0][1] == "img"

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

    assert progress.status["vk_week_post:2025-30"] == "done"
    assert progress.status["vk_week_post:2025-31"] == "error"
    assert "captcha" not in progress.status.values()
    assert "pending" not in progress.status.values()
    assert calls == ["wall.post", "wall.post", "wall.post"]
