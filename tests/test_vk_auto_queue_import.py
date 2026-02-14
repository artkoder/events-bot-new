import os
import sys
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
        return cancel_text, [], datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)

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
        return "text", ["https://example.com/a.jpg"], datetime.now(timezone.utc)

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
        return "text", [], datetime.now(timezone.utc)

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
        return "text", [], datetime.now(timezone.utc)

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
    assert report.skipped_requeued == 1

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT status FROM vk_inbox WHERE id=1")
        (status,) = await cur.fetchone()
        assert status == "imported"


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
    assert "Telegraph:" in text
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

    text, photos, published_at = await vk_auto_queue.fetch_vk_post_text_and_photos(30777579, 14572)

    assert text == "Тестовый пост"
    assert photos == ["https://img.test/p1.jpg"]
    assert published_at is not None


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

    text, photos, published_at = await vk_auto_queue.fetch_vk_post_text_and_photos(1, 1)

    assert "Комментарий к репосту" in text
    assert "Основной текст события в репосте" in text
    assert photos == []
    assert published_at is not None


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
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

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
