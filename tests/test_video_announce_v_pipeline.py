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
    VideoAnnounceSessionStatus,
)
from video_announce.scenario import TOMORROW_TEST_MIN_POSTERS, VideoAnnounceScenario


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

    assert started["limit_scenes"] == TOMORROW_TEST_MIN_POSTERS


@pytest.mark.asyncio
async def test_prepare_tomorrow_session_builds_manual_preflight_without_render(
    monkeypatch, tmp_path
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now_local = datetime.now(LOCAL_TZ)
    tomorrow = (now_local + timedelta(days=1)).date()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        ev = Event(
            title="Manual Event",
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

    async def _should_not_render(*args, **kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("prepare_tomorrow_session must not auto-start render")

    monkeypatch.setattr(VideoAnnounceScenario, "start_render", _should_not_render)

    bot = _DummyBot()
    scenario = VideoAnnounceScenario(db, bot, chat_id=123, user_id=1)
    await scenario.prepare_tomorrow_session()

    async with db.get_session() as session:
        res = await session.execute(select(VideoAnnounceSession))
        sessions = list(res.scalars().all())
        assert len(sessions) == 1
        sess = sessions[0]
        assert sess.status == VideoAnnounceSessionStatus.SELECTED
        assert isinstance(sess.selection_params, dict)
        assert sess.selection_params.get("target_date") == tomorrow.isoformat()
        assert sess.selection_params.get("render_scene_limit") == 12
        assert sess.kaggle_kernel_ref is None

        res_items = await session.execute(
            select(VideoAnnounceItem).where(VideoAnnounceItem.session_id == sess.id)
        )
        items = list(res_items.scalars().all())
        assert any(it.status == VideoAnnounceItemStatus.READY for it in items)

    texts = [text for _, text, _ in bot.messages]
    assert any("подготовлена" in text for text in texts)
    assert any("SELECTED" in text for text in texts)


@pytest.mark.asyncio
async def test_show_kernel_selection_blocks_when_ready_items_exceed_limit(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add(User(user_id=1, is_superadmin=True))
        sess = VideoAnnounceSession(
            status=VideoAnnounceSessionStatus.SELECTED,
            profile_key="default",
            selection_params={"render_scene_limit": 12},
        )
        session.add(sess)
        await session.commit()
        await session.refresh(sess)

        for idx in range(13):
            ev = Event(
                title=f"Event {idx}",
                description="d",
                source_text="s",
                date="2026-03-14",
                time="19:00",
                location_name=f"Loc {idx}",
                city="City",
                photo_urls=[f"https://example.com/{idx}.jpg"],
                photo_count=1,
            )
            session.add(ev)
            await session.flush()
            session.add(
                VideoAnnounceItem(
                    session_id=sess.id,
                    event_id=ev.id,
                    position=idx + 1,
                    status=VideoAnnounceItemStatus.READY,
                )
            )
        await session.commit()

    bot = _DummyBot()
    scenario = VideoAnnounceScenario(db, bot, chat_id=123, user_id=1)

    msg = await scenario.show_kernel_selection(sess.id)

    assert msg == (
        "Выбрано 13 событий, а текущий рендер поддерживает максимум 12. "
        "Снимите лишние в SELECTED перед запуском."
    )
