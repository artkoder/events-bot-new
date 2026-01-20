from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from db import Database
from main import LOCAL_TZ
from models import (
    Event,
    EventPoster,
    User,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
)
from video_announce.scenario import VideoAnnounceScenario


class _DummyBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs) -> None:  # noqa: ARG002
        self.messages.append((chat_id, text, kwargs))

    async def send_document(self, chat_id: int, document, **kwargs) -> None:  # noqa: ANN001,ARG002
        self.messages.append((chat_id, "document", kwargs))


@pytest.mark.asyncio
async def test_run_tomorrow_pipeline_creates_session_and_starts(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now_local = datetime.now(LOCAL_TZ)
    tomorrow = (now_local + timedelta(days=1)).date()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        ev = Event(
            title="Event",
            description="d",
            source_text="s",
            date=tomorrow.isoformat(),
            time="19:00",
            location_name="Loc",
            city="City",
            photo_urls=["https://example.com/1.jpg"],
            photo_count=1,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(
            EventPoster(
                event_id=ev.id,
                poster_hash="h1",
                ocr_text="TEXT",
                ocr_title="TITLE",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    started: dict[str, int] = {}

    async def _fake_start_render(self, session_id: int, message=None, *, limit_scenes=None) -> str:  # noqa: ANN001,ARG002
        started["session_id"] = session_id
        return "Рендеринг запущен"

    monkeypatch.setattr(VideoAnnounceScenario, "start_render", _fake_start_render)

    bot = _DummyBot()
    scenario = VideoAnnounceScenario(db, bot, chat_id=123, user_id=1)
    await scenario.run_tomorrow_pipeline()

    assert "session_id" in started

    async with db.get_session() as session:
        sess = await session.get(VideoAnnounceSession, started["session_id"])
        assert sess is not None
        assert isinstance(sess.selection_params, dict)
        assert sess.selection_params.get("random_order") is True
        assert sess.selection_params.get("target_date") == tomorrow.isoformat()
        assert sess.kaggle_kernel_ref

        res = await session.execute(
            select(VideoAnnounceItem).where(VideoAnnounceItem.session_id == sess.id)
        )
        items = list(res.scalars().all())
        assert any(it.status == VideoAnnounceItemStatus.READY for it in items)

    assert any(
        f"Сессия #{started['session_id']}" in text for _, text, _ in bot.messages
    )


@pytest.mark.asyncio
async def test_run_tomorrow_pipeline_test_mode_limits_scenes(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now_local = datetime.now(LOCAL_TZ)
    tomorrow = (now_local + timedelta(days=1)).date()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        ev = Event(
            title="Event",
            description="d",
            source_text="s",
            date=tomorrow.isoformat(),
            time="19:00",
            location_name="Loc",
            city="City",
            photo_urls=["https://example.com/1.jpg"],
            photo_count=1,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        session.add(
            EventPoster(
                event_id=ev.id,
                poster_hash="h1",
                ocr_text="TEXT",
                ocr_title="TITLE",
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()

    started: dict[str, int] = {}

    async def _fake_start_render(self, session_id: int, message=None, *, limit_scenes=None) -> str:  # noqa: ANN001,ARG002
        started["session_id"] = session_id
        started["limit_scenes"] = limit_scenes
        return "Рендеринг запущен"

    monkeypatch.setattr(VideoAnnounceScenario, "start_render", _fake_start_render)

    bot = _DummyBot()
    scenario = VideoAnnounceScenario(db, bot, chat_id=123, user_id=1)
    await scenario.run_tomorrow_pipeline(test_mode=True)

    assert started["limit_scenes"] == 1
