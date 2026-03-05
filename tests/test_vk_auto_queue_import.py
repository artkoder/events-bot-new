import os
import sys
import logging
from datetime import datetime, timezone

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
from main import Database

import vk_intake
import vk_auto_queue
import poster_ocr
from poster_media import PosterMedia
from source_parsing.handlers import AddedEventInfo


class DummyBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text, **_kwargs):
        self.messages.append((int(chat_id), str(text)))

    async def get_me(self):
        class Me:
            username = "eventsbotTestBot"
        return Me()


@pytest.mark.asyncio
async def test_vk_auto_import_requests_strict_chronological_pick_next(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    calls: list[dict[str, object]] = []

    async def fake_pick_next(_db, _operator_id, _batch_id, **kwargs):
        calls.append(dict(kwargs))
        return None

    monkeypatch.setattr(vk_auto_queue.vk_review, "pick_next", fake_pick_next)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert calls, "expected at least one pick_next call"
    assert calls[0].get("prefer_oldest") is True
    assert calls[0].get("strict_chronological") is True


@pytest.mark.asyncio
async def test_vk_auto_import_cancellation_notice_marks_existing_event_inactive(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Create an existing upcoming event that should be cancelled/hidden.
    async with db.get_session() as session:
        from models import Event

        session.add(
            Event(
            title="Manhattan Short Online",
            description="Описание",
            source_text="src",
            date="2026-02-15",
            time="16:00",
            location_name="арт-пространство «Сигнал»",
            location_address="ул. К. Леонова, 22",
            city="Калининград",
            )
        )
        await session.commit()
        # Reload the inserted event_id via query to avoid relying on ORM identity mechanics.
        from sqlalchemy import select

        res = await session.execute(select(Event.id).where(Event.title == "Manhattan Short Online"))
        event_id = int(res.scalar_one())

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (211997788, "signal", "Пространство Сигнал", "арт-пространство «Сигнал»", None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 211997788, 2754, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    cancel_text = (
        "Друзья, объявление для любителей кинофестиваля Manhattan Short Online. "
        "К сожалению, организаторы сдвинули сроки фестиваля, поэтому показ 15 февраля не состоится."
    )

    async def fake_fetch(*_args, **_kwargs):
        return (
            cancel_text,
            [],
            datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("build_event_drafts must not be called for cancellation notices")

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", should_not_be_called)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT silent, lifecycle_status FROM event WHERE id=?",
            (int(event_id),),
        )
        silent, lifecycle_status = await cur.fetchone()
        assert int(silent or 0) == 0
        assert str(lifecycle_status or "") in {"cancelled", "postponed"}

        cur = await conn.execute("SELECT status, imported_event_id FROM vk_inbox WHERE id=1")
        status, imported_event_id = await cur.fetchone()
        assert status == "imported"
        assert int(imported_event_id) == int(event_id)


@pytest.mark.asyncio
async def test_vk_auto_import_marks_inbox_imported_and_links_multiple_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    # Queue row - use OCR_PENDING sentinel so vk_review doesn't try to recompute ts_hint/reject.
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Test Community", "Научная библиотека", None, "https://tickets.local"),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            ["https://example.com/a.jpg"],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        d1 = vk_intake.EventDraft(title="E1", date="2026-12-31", time="18:30", venue="Научная библиотека")
        d2 = vk_intake.EventDraft(title="E2", date="2026-12-31", time="18:30", venue="Научная библиотека")
        return [d1, d2], None

    # Persist stub: we only need deterministic ids to verify mapping table; the events
    # themselves are not required for this unit test.
    counter = {"n": 0}

    async def fake_persist(*_args, **_kwargs):
        counter["n"] += 1
        return vk_intake.PersistResult(
            event_id=1000 + counter["n"],
            telegraph_url="",
            ics_supabase_url="",
            ics_tg_url="",
            event_date="2026-12-31",
            event_end_date=None,
            event_time="18:30",
            event_type=None,
            is_free=False,
            smart_status="created",
            smart_created=True,
            smart_merged=False,
        )

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=10, operator_id=123)

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status, imported_event_id FROM vk_inbox WHERE id=1")
        status, imported_event_id = await cur.fetchone()
        assert status == "imported"
        assert imported_event_id == 1001

        cur = await conn.execute(
            "SELECT event_id FROM vk_inbox_import_event WHERE inbox_id=1 ORDER BY event_id"
        )
        rows = await cur.fetchall()
        assert [r[0] for r in rows] == [1001, 1002]


@pytest.mark.asyncio
async def test_vk_auto_import_rejects_low_confidence_drafts(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (26560795, "club", "Калининградская областная филармония", None, None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 26560795, 11921, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        d1 = vk_intake.EventDraft(
            title="Волшебный мир Хаяо Миядзаки",
            date="2026-03-19",
            time=None,
            venue="Филармония",
            reject_reason="Низкая уверенность: заголовок взят из прошедшего концерта.",
        )
        return [d1], None

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("persist_event_and_pages must not be called for low-confidence drafts")

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", should_not_be_called)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=1")
        (status,) = await cur.fetchone()
    assert status == "rejected"


@pytest.mark.asyncio
async def test_vk_auto_import_enables_obvious_non_event_prefilter(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (20898960, "kukly", "Калининградский областной театр кукол", None, None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 20898960, 4794, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    captured: dict[str, object] = {}

    async def fake_fetch(*_args, **_kwargs):
        return (
            "Исторический очерк о послевоенном театре кукол.",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **kwargs):
        captured["prefilter_obvious_non_events"] = kwargs.get("prefilter_obvious_non_events")
        return [
            vk_intake.EventDraft(
                title="",
                reject_reason="Длинный исторический/справочный пост без признаков будущего посещаемого события",
            )
        ], None

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("persist_event_and_pages must not be called for reject-only drafts")

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", should_not_be_called)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert captured["prefilter_obvious_non_events"] is True
    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=1")
        (status,) = await cur.fetchone()
    assert status == "rejected"


@pytest.mark.asyncio
async def test_vk_auto_import_skips_festival_helper_for_regular_sources(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link, festival_source) VALUES(?,?,?,?,?,?,?)",
            (1, "club1", "Regular Community", None, None, None, 0),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    seen: dict[str, object] = {}

    async def fake_build_event_drafts(*_args, **kwargs):
        seen["festival_names"] = kwargs.get("festival_names")
        seen["festival_alias_pairs"] = kwargs.get("festival_alias_pairs")
        seen["festival_hint"] = kwargs.get("festival_hint")
        return [], None

    async def fake_load_festival_hints(_db):
        return ["Fest"], [("fest", 0)]

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_auto_queue, "_load_festival_hints", fake_load_festival_hints)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert seen["festival_names"] is None
    assert seen["festival_alias_pairs"] is None
    assert seen["festival_hint"] is False


@pytest.mark.asyncio
async def test_vk_auto_import_keeps_festival_helper_for_festival_sources(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link, festival_source) VALUES(?,?,?,?,?,?,?)",
            (1, "club1", "Festival Community", None, None, None, 1),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    seen: dict[str, object] = {}

    async def fake_build_event_drafts(*_args, **kwargs):
        seen["festival_names"] = kwargs.get("festival_names")
        seen["festival_alias_pairs"] = kwargs.get("festival_alias_pairs")
        seen["festival_hint"] = kwargs.get("festival_hint")
        return [], None

    async def fake_load_festival_hints(_db):
        return ["Fest"], [("fest", 0)]

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_auto_queue, "_load_festival_hints", fake_load_festival_hints)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert seen["festival_names"] == ["Fest"]
    assert seen["festival_alias_pairs"] == [("fest", 0)]
    assert seen["festival_hint"] is True


@pytest.mark.asyncio
async def test_vk_auto_import_include_skipped_requeues_and_imports(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Test Community", "Научная библиотека", None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "skipped"),
        )
        await conn.commit()

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        d1 = vk_intake.EventDraft(title="E1", date="2026-12-31", time="18:30", venue="Научная библиотека")
        return [d1], None

    async def fake_persist(*_args, **_kwargs):
        return vk_intake.PersistResult(
            event_id=1001,
            telegraph_url="",
            ics_supabase_url="",
            ics_tg_url="",
            event_date="2026-12-31",
            event_end_date=None,
            event_time="18:30",
            event_type=None,
            is_free=False,
            smart_status="created",
            smart_created=True,
            smart_merged=False,
        )

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)

    bot = DummyBot()
    report = await vk_auto_queue.run_vk_auto_import(
        db,
        bot,
        chat_id=1,
        limit=1,
        operator_id=123,
        include_skipped=True,
    )


@pytest.mark.asyncio
async def test_vk_auto_import_prefetch_does_not_reprocess_current_locked_row(tmp_path, monkeypatch):
    """Regression: prefetch must not pick the currently locked row again.

    Previously, run_vk_auto_import() picked the "next" post while the current post
    was still locked. vk_review.pick_next() prefers resuming locked rows for the
    operator, so it returned the same row and the importer processed it twice.
    """

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = int(datetime.now(timezone.utc).timestamp())
    future1 = now + 100_000
    future2 = now + 200_000

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Test Community", "Научная библиотека", None, None),
        )
        await conn.executemany(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (1, 1, 100, now, "Концерт 31.12.2099", "k", 1, future1, "pending"),
                (2, 1, 200, now + 1, "Концерт 01.01.2100", "k", 1, future2, "pending"),
            ],
        )
        await conn.commit()

    async def fake_fetch(group_id, post_id, *_args, **_kwargs):
        return (
            f"text {group_id}_{post_id}",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        return [vk_intake.EventDraft(title="E", date="2026-12-31", time="18:30", venue="Научная библиотека")], None

    calls: list[str] = []

    async def fake_persist(*_args, **kwargs):
        calls.append(str(kwargs.get("source_post_url") or ""))
        n = len(calls)
        return vk_intake.PersistResult(
            event_id=2000 + n,
            telegraph_url="",
            ics_supabase_url="",
            ics_tg_url="",
            event_date="2026-12-31",
            event_end_date=None,
            event_time="18:30",
            event_type=None,
            is_free=False,
            smart_status="created",
            smart_created=True,
            smart_merged=False,
        )

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=2, operator_id=123)

    assert calls == [
        "https://vk.com/wall-1_100",
        "https://vk.com/wall-1_200",
    ]

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=1")
        (status,) = await cur.fetchone()
        assert status == "imported"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inline_jobs_env", "expected_wait_for_telegraph"),
    [
        (None, False),
        ("0", True),
    ],
)
async def test_vk_auto_import_skips_redundant_telegraph_wait_when_inline_jobs_enabled(
    tmp_path,
    monkeypatch,
    inline_jobs_env,
    expected_wait_for_telegraph,
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Test Community", "Научная библиотека", None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    if inline_jobs_env is None:
        monkeypatch.delenv("VK_AUTO_IMPORT_INLINE_JOBS", raising=False)
    else:
        monkeypatch.setenv("VK_AUTO_IMPORT_INLINE_JOBS", inline_jobs_env)

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        return [vk_intake.EventDraft(title="E", date="2026-12-31", time="18:30", venue="Научная библиотека")], None

    captured_waits: list[bool] = []

    async def fake_persist(*_args, **kwargs):
        captured_waits.append(bool(kwargs.get("wait_for_telegraph_url", True)))
        return vk_intake.PersistResult(
            event_id=1001,
            telegraph_url="",
            ics_supabase_url="",
            ics_tg_url="",
            event_date="2026-12-31",
            event_end_date=None,
            event_time="18:30",
            event_type=None,
            is_free=False,
            smart_status="created",
            smart_created=True,
            smart_merged=False,
        )

    async def fake_run_jobs(*_args, **_kwargs):
        return None

    async def fake_report(*_args, **_kwargs):
        return True

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)
    monkeypatch.setattr(main, "run_event_update_jobs", fake_run_jobs)
    monkeypatch.setattr(vk_auto_queue, "_send_unified_event_report", fake_report)

    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert captured_waits == [expected_wait_for_telegraph]


@pytest.mark.asyncio
async def test_vk_auto_import_logs_stage_timings_for_slow_rows_without_pipeline_timings(
    tmp_path,
    monkeypatch,
    caplog,
):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (1, "club1", "Test Community", "Научная библиотека", None, None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 100, 0, "stub", vk_intake.OCR_PENDING_SENTINEL, 0, None, "pending"),
        )
        await conn.commit()

    monkeypatch.setenv("VK_AUTO_IMPORT_INLINE_JOBS", "0")
    monkeypatch.setenv("VK_AUTO_IMPORT_SLOW_ROW_LOG_SEC", "0")
    monkeypatch.delenv("PIPELINE_TIMINGS", raising=False)

    async def fake_fetch(*_args, **_kwargs):
        return (
            "text",
            [],
            datetime.now(timezone.utc),
            {"views": 10, "likes": 1},
            vk_auto_queue.VkFetchStatus(True, "ok"),
        )

    async def fake_build_event_drafts(*_args, **_kwargs):
        return [vk_intake.EventDraft(title="E", date="2026-12-31", time="18:30", venue="Научная библиотека")], None

    async def fake_persist(*_args, **_kwargs):
        return vk_intake.PersistResult(
            event_id=1001,
            telegraph_url="",
            ics_supabase_url="",
            ics_tg_url="",
            event_date="2026-12-31",
            event_end_date=None,
            event_time="18:30",
            event_type=None,
            is_free=False,
            smart_status="created",
            smart_created=True,
            smart_merged=False,
        )

    async def fake_report(*_args, **_kwargs):
        return True

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build_event_drafts)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)
    monkeypatch.setattr(vk_auto_queue, "_send_unified_event_report", fake_report)

    caplog.set_level(logging.INFO)
    bot = DummyBot()
    await vk_auto_queue.run_vk_auto_import(db, bot, chat_id=1, limit=1, operator_id=123)

    assert any(
        "timing vk_auto_import_row" in rec.message and "persist_total" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_vk_auto_report_is_unified_and_contains_fact_stats(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_added_info(_db, _event_id, _source, **_kwargs):
        return AddedEventInfo(
            event_id=2417,
            title="Фигаро",
            source="vk",
            telegraph_url="https://telegra.ph/Figaro-02-11",
            ics_url="https://example.test/figaro.ics",
            log_cmd="/log 2417",
            date="2026-02-12",
            time="19:00",
            source_url="https://vk.com/wall-30777579_14572",
            fact_stats={"added": 5, "duplicate": 3, "conflict": 1, "note": 2},
        )

    monkeypatch.setattr("source_parsing.handlers.build_added_event_info", fake_added_info)

    bot = DummyBot()
    await vk_auto_queue._send_unified_event_report(
        db,
        bot,
        1,
        created=[2417],
        updated=[],
        source_url="https://vk.com/wall-30777579_14572",
    )

    assert bot.messages, "VK auto report was not sent"
    _chat_id, text = bot.messages[-1]
    assert "Smart Update (детали событий)" in text
    assert "✅ Созданные события: 1" in text
    # Telegraph link is now embedded into the event title; the duplicated "Telegraph:" line
    # should not be present when telegraph_url exists.
    assert 'href="https://telegra.ph/Figaro-02-11"' in text
    assert "Telegraph:" not in text
    assert "Факты: ✅5 ↩️3 ⚠️1 ℹ️2" in text
    assert "Иллюстрации:" in text
    assert "start=log_2417" in text


@pytest.mark.asyncio
async def test_fetch_vk_post_text_and_photos_accepts_unwrapped_response(monkeypatch):
    async def fake_vk_api(_method, **_params):
        return {
            "items": [
                {
                    "text": "Тестовый пост",
                    "date": 1760000000,
                    "attachments": [
                        {
                            "type": "photo",
                            "photo": {
                                "sizes": [
                                    {
                                        "url": "https://img.test/p1.jpg",
                                        "width": 1200,
                                        "height": 900,
                                    }
                                ]
                            },
                        }
                    ],
                }
            ]
        }

    monkeypatch.setattr(main, "vk_api", fake_vk_api)

    text, photos, published_at, metrics, status = await vk_auto_queue.fetch_vk_post_text_and_photos(
        30777579, 14572
    )

    assert text == "Тестовый пост"
    assert photos == ["https://img.test/p1.jpg"]
    assert published_at is not None
    assert metrics is None
    assert status.ok is True
    assert status.kind == "ok"


@pytest.mark.asyncio
async def test_fetch_vk_post_text_and_photos_includes_repost_text(monkeypatch):
    async def fake_vk_api(_method, **_params):
        return {
            "items": [
                {
                    "text": "Комментарий к репосту",
                    "date": 1760000000,
                    "copy_history": [
                        {
                            "text": "Основной текст события в репосте",
                            "attachments": [],
                        }
                    ],
                    "attachments": [],
                }
            ]
        }

    monkeypatch.setattr(main, "vk_api", fake_vk_api)

    text, photos, published_at, metrics, status = await vk_auto_queue.fetch_vk_post_text_and_photos(1, 1)

    assert "Комментарий к репосту" in text
    assert "Основной текст события в репосте" in text
    assert photos == []
    assert published_at is not None
    assert metrics is None
    assert status.ok is True
    assert status.kind == "ok"


@pytest.mark.asyncio
async def test_vk_auto_queue_rejects_deleted_post_when_vk_fetch_not_found(tmp_path, monkeypatch):
    from types import SimpleNamespace

    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, status, locked_by, locked_at, review_batch)
            VALUES(?, ?, ?, ?, ?, NULL, 1, 'locked', 777, CURRENT_TIMESTAMP, 'batch-x')
            """,
            (1, 29891284, 12930, 0, "cached text"),
        )
        await conn.commit()

    async def fake_fetch_vk_post_text_and_photos(_group_id, _post_id, *, db, bot):  # noqa: ARG001
        return (
            "",
            [],
            None,
            None,
            vk_auto_queue.VkFetchStatus(False, "not_found", error_code=100, error="Post was deleted"),
        )

    async def should_not_be_called(*_args, **_kwargs):
        raise AssertionError("build_event_drafts must not be called when VK post is not found")

    monkeypatch.setattr(vk_auto_queue, "fetch_vk_post_text_and_photos", fake_fetch_vk_post_text_and_photos)
    monkeypatch.setattr(vk_auto_queue.vk_intake, "build_event_drafts", should_not_be_called)

    report = vk_auto_queue.VkAutoImportReport(batch_id="batch-x")
    post = SimpleNamespace(
        id=1,
        group_id=29891284,
        post_id=12930,
        date=0,
        text="cached text",
        event_ts_hint=None,
    )

    class DummyBot:
        pass

    await vk_auto_queue._process_vk_inbox_row(  # type: ignore[attr-defined]
        db,
        DummyBot(),
        chat_id=1,
        operator_id=1,
        batch_id="batch-x",
        post=post,
        source_url="https://vk.com/wall-29891284_12930",
        report=report,
        festival_names=None,
        festival_alias_pairs=None,
        progress_message_id=None,
        progress_current_no=1,
        progress_total_txt="1",
    )

    assert report.inbox_rejected == 1
    assert report.inbox_failed == 0

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=?", (1,))
        row = await cur.fetchone()
    assert row[0] == "rejected"


def test_build_smart_update_posters_falls_back_to_vk_photo_url_when_catbox_missing():
    draft = vk_intake.EventDraft(
        title="Тест",
        date="2026-02-20",
        time="19:00",
        venue="Локация",
    )
    draft.poster_media = [PosterMedia(data=b"img", name="poster.jpg")]
    photos = ["https://sun9-1.userapi.com/poster.jpg"]

    class _Poster:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    posters = vk_intake._build_smart_update_posters(
        draft,
        photos=photos,
        poster_cls=_Poster,
    )

    assert len(posters) == 1
    assert posters[0].catbox_url == photos[0]


def test_build_smart_update_posters_uses_source_photos_when_ocr_items_absent():
    draft = vk_intake.EventDraft(
        title="Тест",
        date="2026-02-20",
        time="19:00",
        venue="Локация",
    )
    draft.poster_media = []
    photos = ["https://sun9-1.userapi.com/p1.jpg", "https://sun9-1.userapi.com/p2.jpg"]

    class _Poster:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    posters = vk_intake._build_smart_update_posters(
        draft,
        photos=photos,
        poster_cls=_Poster,
    )

    assert [p.catbox_url for p in posters] == photos


def test_build_smart_update_posters_routes_supabase_urls_into_supabase_field():
    draft = vk_intake.EventDraft(
        title="Тест",
        date="2026-02-20",
        time="19:00",
        venue="Локация",
    )
    draft.poster_media = [
        PosterMedia(
            data=b"img",
            name="poster.jpg",
            catbox_url="https://project.supabase.co/storage/v1/object/public/events-media/p/dh16/ab/abc.webp",
        )
    ]

    class _Poster:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    posters = vk_intake._build_smart_update_posters(
        draft,
        photos=None,
        poster_cls=_Poster,
    )

    assert len(posters) == 1
    assert posters[0].catbox_url is None
    assert posters[0].supabase_url == draft.poster_media[0].catbox_url


@pytest.mark.asyncio
async def test_vk_build_event_drafts_does_not_fail_on_ocr_errors(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_download(_urls):
        return [(b"img", "p.jpg")]

    async def fake_process_media(_bytes, **_kwargs):
        return [PosterMedia(data=b"img", name="p.jpg")], None

    async def fake_recognize(_db, _photo_bytes, **_kwargs):
        raise RuntimeError("OCR request failed")

    class _Parsed(list):
        festival = None

    async def fake_parse(*_args, **_kwargs):
        return _Parsed(
            [
                {
                    "title": "Событие",
                    "date": "2026-02-20",
                    "time": "19:00",
                    "location_name": "Научная библиотека",
                    "short_description": "Тест",
                }
            ]
        )

    monkeypatch.setattr(vk_intake, "_download_photo_media", fake_download)
    monkeypatch.setattr(vk_intake, "process_media", fake_process_media)
    monkeypatch.setattr(poster_ocr, "recognize_posters", fake_recognize)
    monkeypatch.setattr(main, "parse_event_via_llm", fake_parse)

    drafts, _fest = await vk_intake.build_event_drafts(
        "Текст",
        photos=["https://example.com/a.jpg"],
        source_name="VK",
        location_hint=None,
        default_time=None,
        default_ticket_link=None,
        operator_extra=None,
        publish_ts=None,
        event_ts_hint=None,
        festival_names=[],
        festival_alias_pairs=[],
        festival_hint=False,
        db=db,
    )
    assert drafts
