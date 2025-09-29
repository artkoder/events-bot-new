import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any

import pytest
from types import SimpleNamespace

from aiogram import types

import main
import poster_ocr
import vk_intake
import vk_review
from main import Database
from models import Event, EventPoster, JobTask, Festival, PosterOcrCache
from poster_media import PosterMedia
from sqlmodel import select
from markup import linkify_for_telegraph


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(SimpleNamespace(chat_id=chat_id, text=text))
        return SimpleNamespace(message_id=1)

    async def send_media_group(self, chat_id, media):
        pass


@pytest.mark.asyncio
async def test_download_photo_media_logs_mime(monkeypatch, caplog):
    urls = [
        "https://example.com/a.png",
        "https://example.com/b.jpg",
    ]
    png_payload = b"\x89PNG\r\n\x1a\nbinary"
    jpeg_payload = b"\xff\xd8\xffbinary"
    payloads = {
        urls[0]: (
            png_payload,
            {
                "Content-Type": "image/png",
                "Content-Length": str(len(png_payload)),
            },
        ),
        urls[1]: (
            jpeg_payload,
            {
                "Content-Type": "image/jpeg",
                "Content-Length": str(len(jpeg_payload)),
            },
        ),
    }

    class FakeContent:
        def __init__(self, data: bytes) -> None:
            self._data = data

        async def read(self, n: int = -1) -> bytes:
            return self._data

        async def iter_chunked(self, _n: int):
            yield self._data

    class FakeResponse:
        def __init__(self, data: bytes, headers: dict[str, str]) -> None:
            self.content = FakeContent(data)
            self.headers = headers
            self.content_length = len(data)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self, mapping: dict[str, tuple[bytes, dict[str, str]]]) -> None:
            self._mapping = mapping
            self.requests: list[tuple[str, dict[str, str] | None]] = []

        def get(self, url: str, **kwargs) -> FakeResponse:
            self.requests.append((url, kwargs.get("headers")))
            data, headers = self._mapping[url]
            return FakeResponse(data, headers)

    session = FakeSession(payloads)
    monkeypatch.setattr(main, "get_http_session", lambda: session)
    monkeypatch.setattr(main, "HTTP_SEMAPHORE", asyncio.Semaphore(5))
    monkeypatch.setattr(main, "HTTP_TIMEOUT", 1)
    monkeypatch.setattr(main, "MAX_DOWNLOAD_SIZE", 1024)
    monkeypatch.setattr(main, "MAX_ALBUM_IMAGES", 5)

    request_headers = {
        "User-Agent": "UnitTest UA",
        "Accept": "image/*",
        "Referer": "https://vk.com/test",
        "Sec-Fetch-Dest": "image",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
    }
    monkeypatch.setattr(main, "VK_PHOTO_FETCH_HEADERS", request_headers, raising=False)

    def fake_ensure_jpeg(data: bytes, name: str) -> tuple[bytes, str]:
        if data.startswith(b"\x89PNG"):
            return b"\xff\xd8\xffconverted", "converted.jpg"
        return data, name

    def fake_detect_image_type(data: bytes) -> str | None:
        if data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if data.startswith(b"\x89PNG"):
            return "png"
        return None

    monkeypatch.setattr(main, "ensure_jpeg", fake_ensure_jpeg)
    monkeypatch.setattr(main, "detect_image_type", fake_detect_image_type)
    monkeypatch.setattr(main, "validate_jpeg_markers", lambda _data: None)

    caplog.set_level(logging.INFO)
    results = await vk_intake._download_photo_media(urls)

    assert results == [
        (b"\xff\xd8\xffconverted", "converted.jpg"),
        (b"\xff\xd8\xffbinary", "vk_poster_2.jpg"),
    ]

    messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
    size_0 = len(payloads[urls[0]][0])
    size_1 = len(payloads[urls[1]][0])
    assert (
        "vk.photo_media processed idx=0 url="
        f"{urls[0]} size={size_0} subtype=jpeg filename=converted.jpg "
        f"content_type=image/png content_length={len(png_payload)}" in messages
    )
    assert (
        "vk.photo_media processed idx=1 url="
        f"{urls[1]} size={size_1} subtype=jpeg filename=vk_poster_2.jpg "
        f"content_type=image/jpeg content_length={len(jpeg_payload)}" in messages
    )

    assert [req[0] for req in session.requests] == urls
    for _, headers in session.requests:
        assert headers == request_headers


@pytest.mark.asyncio
async def test_vkrev_import_flow_persists_url_and_skips_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    draft = vk_intake.EventDraft(title="T", date="2025-09-02", time="10:00", source_text="T")

    async def fake_build(
        text,
        *,
        photos=None,
        source_name=None,
        location_hint=None,
        default_time=None,
        operator_extra=None,
        festival_names=None,
        festival_alias_pairs=None,
        festival_hint=False,
        db=None,
    ):
        captured["festival_names"] = festival_names
        captured["festival_alias_pairs"] = festival_alias_pairs
        captured["festival_hint"] = festival_hint
        return [draft]

    captured = {}

    async def fake_mark_imported(
        db_, inbox_id, batch_id, operator_id, event_id, event_date
    ):
        captured["event_id"] = event_id

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    async with db.get_session() as session:
        session.add(Festival(name="Fest One"))
        await session.commit()

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    async with db.get_session() as session:
        ev = await session.get(Event, captured["event_id"])
    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert captured["festival_names"] == ["Fest One"]
    assert captured["festival_alias_pairs"] is None
    assert captured["festival_hint"] is False
    assert JobTask.vk_sync not in tasks
    message_lines = bot.messages[-1].text.splitlines()
    assert message_lines[0] == "Импортировано"
    assert "Тип: —" in message_lines
    assert "Дата начала: 2025-09-02" in message_lines
    assert "Дата окончания: —" in message_lines
    assert "Время: 10:00" in message_lines
    assert "Бесплатное: нет" in message_lines


@pytest.mark.asyncio
async def test_vkrev_import_flow_reports_ocr_usage(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    poster_media = [PosterMedia(data=b"", name="p1")]
    draft = vk_intake.EventDraft(
        title="T",
        date="2025-09-02",
        time="10:00",
        source_text="T",
        poster_media=poster_media,
        ocr_tokens_spent=12,
        ocr_tokens_remaining=789,
    )

    async def fake_build(
        text,
        *,
        photos=None,
        source_name=None,
        location_hint=None,
        default_time=None,
        operator_extra=None,
        festival_names=None,
        festival_alias_pairs=None,
        festival_hint=False,
        db=None,
    ):
        return [draft]

    async def fake_mark_imported(db_, inbox_id, batch_id, operator_id, event_id, event_date):
        pass

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        return "job"

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    info_text = bot.messages[-1].text
    assert "OCR: потрачено 12, осталось 789" in info_text


@pytest.mark.asyncio
async def test_vkrev_import_flow_reports_ocr_limit(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    poster_media = [PosterMedia(data=b"", name="p1")]
    draft = vk_intake.EventDraft(
        title="T",
        date="2025-09-02",
        time="10:00",
        source_text="T",
        poster_media=poster_media,
        ocr_tokens_spent=0,
        ocr_tokens_remaining=0,
        ocr_limit_notice="OCR недоступен: лимит",
    )

    async def fake_build(*args, **kwargs):
        return [draft]

    async def fake_mark_imported(*args, **kwargs):
        pass

    async def fake_enqueue_job(*args, **kwargs):
        return "job"

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    info_text = bot.messages[-1].text
    assert "OCR недоступен" in info_text
    assert "OCR: потрачено 0, осталось 0" in info_text


@pytest.mark.asyncio
async def test_vkrev_import_flow_passes_festival_hint(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async with db.get_session() as session:
        session.add(Festival(name="Fest One", aliases=["Fest Alias"]))
        await session.commit()

    captured: dict[str, Any] = {}

    async def fake_fetch(*args, **kwargs):
        return []

    async def fake_download_photo_media(_photos):
        return []

    async def fake_recognize_posters(db_, photo_bytes, log_context=None):
        return [], 0, 0

    async def fake_parse(text: str, *args, **kwargs):
        captured["text"] = text
        captured["festival_names"] = kwargs.get("festival_names")
        captured["festival_alias_pairs"] = kwargs.get("festival_alias_pairs")
        return [
            {
                "title": "Fest Event",
                "date": "2025-09-02",
                "time": "10:00",
                "short_description": "Desc",
                "location_name": "Venue",
            }
        ]

    def fake_require(name: str):
        assert name == "parse_event_via_4o"
        return fake_parse

    async def fake_persist(draft, photos, db_, source_post_url=None):
        return vk_intake.PersistResult(
            event_id=1,
            telegraph_url="https://t",
            ics_supabase_url="https://s",
            ics_tg_url="https://tg",
            event_date="2025-09-02",
            event_end_date=None,
            event_time="10:00",
            event_type=None,
            is_free=False,
        )

    async def fake_mark_imported(*args, **kwargs):
        pass

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "_download_photo_media", fake_download_photo_media)
    monkeypatch.setattr(vk_intake.poster_ocr, "recognize_posters", fake_recognize_posters)
    monkeypatch.setattr(vk_intake, "require_main_attr", fake_require)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)

    bot = DummyBot()
    await main._vkrev_import_flow(
        1,
        1,
        1,
        "batch1",
        db,
        bot,
        festival_hint=True,
    )

    assert "Оператор подтверждает" in captured["text"]
    assert captured["festival_names"] == ["Fest One"]
    expected_alias = main.normalize_alias("Fest Alias")
    assert captured["festival_alias_pairs"] == [(expected_alias, 0)]


@pytest.mark.asyncio
async def test_vkrev_import_flow_handles_multiple_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    draft1 = vk_intake.EventDraft(
        title="First",
        date="2025-09-02",
        time="10:00",
        source_text="First",
    )
    draft2 = vk_intake.EventDraft(
        title="Second",
        date="2025-09-03",
        time="12:00",
        source_text="Second",
    )

    async def fake_build(*args, **kwargs):
        return [draft1, draft2]

    captured_mark_id: dict[str, int] = {}

    async def fake_mark_imported(
        db_, inbox_id, batch_id, operator_id, event_id, event_date
    ):
        captured_mark_id["event_id"] = event_id

    tasks: list[str] = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    async with db.get_session() as session:
        session.add(Festival(name="Fest Multi"))
        await session.commit()

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    async with db.get_session() as session:
        events = (await session.execute(select(Event))).scalars().all()

    assert len(events) == 2
    stored_ids = sorted(event.id for event in events)
    assert captured_mark_id["event_id"] == stored_ids[0]
    assert JobTask.vk_sync not in tasks

    assert len(bot.messages) == 3
    summary_text = bot.messages[0].text
    assert f"Событие 1: ID {stored_ids[0]}" in summary_text
    assert f"Событие 2: ID {stored_ids[1]}" in summary_text
    assert bot.messages[1].text.startswith("Импортировано #1")
    assert bot.messages[2].text.startswith("Импортировано #2")


@pytest.mark.asyncio
async def test_vkrev_import_flow_creates_festival_without_events(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "Festival post", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    async def fake_build(*args, **kwargs):
        return []

    sync_calls: list[tuple[str, str]] = []

    async def fake_sync_page(db_obj, name, **kwargs):
        sync_calls.append(("page", name))

    async def fake_sync_index(db_obj, **kwargs):
        sync_calls.append(("index", ""))

    async def fake_sync_vk(db_obj, name, bot_obj, **kwargs):
        sync_calls.append(("vk", name))
        return None

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    async def fake_rebuild_nav(db_obj):
        sync_calls.append(("nav", ""))
        return False

    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)
    monkeypatch.setattr(main, "sync_festivals_index_page", fake_sync_index)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_vk)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", fake_rebuild_nav)
    monkeypatch.setattr(main.parse_event_via_4o, "_festival", {
        "name": "Fest Solo",
        "start_date": "2025-07-01",
        "end_date": "2025-07-07",
        "location_name": "Main Park",
        "city": "Kaliningrad",
        "website_url": "https://fest.example",
    }, raising=False)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    async with db.get_session() as session:
        result = await session.execute(select(Festival).where(Festival.name == "Fest Solo"))
        fest = result.scalar_one()

    assert fest.start_date == "2025-07-01"
    assert fest.end_date == "2025-07-07"
    assert fest.location_name == "Main Park"
    assert fest.city == "Kaliningrad"
    assert fest.website_url == "https://fest.example"
    assert fest.photo_urls == []
    assert fest.source_text == "Festival post"

    message_text = bot.messages[-1].text
    assert "Импортировано только фестиваль" in message_text
    assert "Фестиваль: Fest Solo" in message_text
    assert "События не импортированы" in message_text

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, imported_event_id FROM vk_inbox WHERE id=?",
            (1,),
        )
        status, imported_event_id = await cur.fetchone()
    assert status == "imported"
    assert imported_event_id is None

    assert ("page", "Fest Solo") in sync_calls


@pytest.mark.asyncio
async def test_vkrev_import_flow_requires_festival_when_forced(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    draft = vk_intake.EventDraft(
        title="T",
        date="2025-09-02",
        time="10:00",
        source_text="T",
    )

    async def fake_build(*args, **kwargs):
        return [draft]

    mark_called = False

    async def fake_mark_imported(*args, **kwargs):
        nonlocal mark_called
        mark_called = True

    async def fake_persist(*args, **kwargs):
        raise AssertionError("persist should not be called when festival is required")

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(vk_intake, "persist_event_and_pages", fake_persist)

    setattr(main.parse_event_via_4o, "_festival", None)

    bot = DummyBot()
    await main._vkrev_import_flow(
        1,
        1,
        1,
        "batch1",
        db,
        bot,
        force_festival=True,
    )

    assert mark_called is False
    assert bot.messages[-1].text == "❌ Не удалось распознать фестиваль, импорт остановлен."
    assert getattr(main.parse_event_via_4o, "_festival", None) is None


@pytest.mark.asyncio
async def test_vkrev_import_flow_creates_festival_and_reports_status(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_source(group_id, screen_name, name, location, default_time) VALUES(?,?,?,?,?)",
            (1, "club1", "Test Community", "", None),
        )
        await conn.execute(
            "INSERT INTO vk_inbox(id, group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 2, 0, "text", None, 1, 0, "pending"),
        )
        await conn.commit()

    async def fake_fetch(*args, **kwargs):
        return []

    poster = PosterMedia(data=b"", name="p1", catbox_url="https://cat.box/image.jpg")
    draft = vk_intake.EventDraft(
        title="T",
        date="2025-09-02",
        time="10:00",
        source_text="Source",
        poster_media=[poster],
    )

    async def fake_build(*args, **kwargs):
        return [draft]

    async def fake_mark_imported(*args, **kwargs):
        pass

    async def fake_enqueue_job(*args, **kwargs):
        return "job"

    sync_calls: list[tuple[str, tuple]] = []

    async def fake_sync_page(db_obj, name):
        sync_calls.append(("page", (name,)))

    async def fake_sync_index(db_obj):
        sync_calls.append(("index", tuple()))

    async def fake_sync_vk(db_obj, name, bot_obj, *, strict=False):
        sync_calls.append(("vk", (name, strict)))

    fest_payload = {
        "name": "Fest Alpha",
        "full_name": "Fest Alpha International",
        "start_date": "2025-07-01",
        "end_date": "2025-07-05",
        "location_name": "Main Hall",
        "location_address": "Main Hall, Fest City",
        "city": "Fest City",
        "website_url": "https://fest.example",
        "program_url": "https://fest.example/program",
        "ticket_url": "https://fest.example/tickets",
    }

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_drafts", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)
    monkeypatch.setattr(main, "sync_festival_page", fake_sync_page)
    monkeypatch.setattr(main, "sync_festivals_index_page", fake_sync_index)
    monkeypatch.setattr(main, "rebuild_festivals_index_if_needed", fake_sync_index)
    monkeypatch.setattr(main, "sync_festival_vk_post", fake_sync_vk)

    setattr(main.parse_event_via_4o, "_festival", fest_payload)

    bot = DummyBot()
    await main._vkrev_import_flow(
        1,
        1,
        1,
        "batch1",
        db,
        bot,
        force_festival=True,
    )

    async with db.get_session() as session:
        fest = (
            await session.execute(select(Festival).where(Festival.name == "Fest Alpha"))
        ).scalar_one()
        event = (await session.execute(select(Event))).scalars().one()

    assert fest.full_name == "Fest Alpha International"
    assert fest.photo_urls == ["https://cat.box/image.jpg"]
    assert fest.website_url == "https://fest.example"
    assert fest.program_url == "https://fest.example/program"
    assert fest.ticket_url == "https://fest.example/tickets"
    assert fest.start_date == "2025-07-01"
    assert fest.end_date == "2025-07-05"
    assert fest.location_name == "Main Hall"
    assert fest.city == "Fest City"
    assert fest.source_post_url == "https://vk.com/wall-1_2"
    assert event.festival == "Fest Alpha"

    assert ("page", ("Fest Alpha",)) in sync_calls
    assert ("index", tuple()) in sync_calls
    assert ("vk", ("Fest Alpha", True)) in sync_calls

    detail_text = bot.messages[-1].text
    assert "Фестиваль: Fest Alpha (создан)" in detail_text

@pytest.mark.asyncio
async def test_vk_persist_event_updates_ocr_records(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_assign_event_topics(event_obj):
        return [], len(event_obj.description or ""), None, False

    async def fake_schedule_event_update_tasks(
        db_obj, event_obj, drain_nav=True, skip_vk_sync=False
    ):
        return {}

    monkeypatch.setattr(main, "assign_event_topics", fake_assign_event_topics)
    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)

    def _make_draft(catbox_url: str, text: str, prompt: int, completion: int, total: int):
        poster = PosterMedia(data=b"image-bytes", name="poster", catbox_url=catbox_url)
        poster.ocr_text = text
        poster.prompt_tokens = prompt
        poster.completion_tokens = completion
        poster.total_tokens = total
        return vk_intake.EventDraft(
            title="T",
            date="2025-09-02",
            time="10:00",
            source_text="T",
            poster_media=[poster],
            ocr_tokens_spent=total,
            ocr_tokens_remaining=10_000,
        )

    first = _make_draft("https://cat.box/a", "Первый текст", 2, 3, 5)
    result_first = await vk_intake.persist_event_and_pages(first, [], db)

    async with db.get_session() as session:
        posters = (
            await session.execute(
                select(EventPoster).where(EventPoster.event_id == result_first.event_id)
            )
        ).scalars().all()
        assert len(posters) == 1
        stored = posters[0]
        first_hash = stored.poster_hash
        assert stored.ocr_text == "Первый текст"
        assert stored.total_tokens == 5

    second = _make_draft("https://cat.box/b", "Обновлённый текст", 7, 8, 15)
    result_second = await vk_intake.persist_event_and_pages(second, [], db)

    assert result_second.event_id == result_first.event_id

    async with db.get_session() as session:
        posters = (
            await session.execute(
                select(EventPoster).where(EventPoster.event_id == result_first.event_id)
            )
        ).scalars().all()

    assert len(posters) == 1
    updated = posters[0]
    assert updated.poster_hash == first_hash
    assert updated.ocr_text == "Обновлённый текст"
    assert updated.catbox_url == "https://cat.box/b"
    assert updated.prompt_tokens == 7
    assert updated.completion_tokens == 8
    assert updated.total_tokens == 15


@pytest.mark.asyncio
async def test_build_event_draft_uses_cached_text_when_limit(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    captured: dict[str, list[str] | None] = {}

    async def fake_parse(text, *args, **kwargs):
        poster_texts = kwargs.get("poster_texts")
        captured["poster_texts"] = poster_texts
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
                "short_description": poster_texts[0] if poster_texts else "",
            }
        ]

    async def fake_process_media(images, *, need_catbox, need_ocr):
        posters = [
            PosterMedia(data=b"", name="poster1"),
            PosterMedia(data=b"", name="poster2"),
        ]
        return posters, ""

    digest_first = hashlib.sha256(b"img1").hexdigest()
    cached_result = PosterOcrCache(
        hash=digest_first,
        detail="auto",
        model="gpt-4o-mini",
        text="Poster text one",
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=3,
    )

    async def fake_ocr(
        db_obj, items, detail="auto", *, count_usage=True, log_context=None
    ):
        raise poster_ocr.PosterOcrLimitExceededError(
            "limit",
            spent_tokens=0,
            remaining=0,
            results=[cached_result],
        )

    async def fake_download(urls):
        return [(b"img1", "poster1.jpg"), (b"img2", "poster2.jpg")]

    monkeypatch.setattr(vk_intake, "process_media", fake_process_media)
    monkeypatch.setattr(poster_ocr, "recognize_posters", fake_ocr)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(vk_intake, "_download_photo_media", fake_download)

    draft = await vk_intake.build_event_draft(
        "text",
        photos=["one", "two"],
        source_name="Test",
        db=db,
    )

    assert captured["poster_texts"] == ["Poster text one"]
    assert draft.poster_media[0].ocr_text == "Poster text one"
    assert draft.poster_media[1].ocr_text is None
    assert draft.ocr_tokens_spent == 0
    assert draft.ocr_tokens_remaining == 0


@pytest.mark.asyncio
async def test_get_event_poster_texts_returns_saved(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    event = Event(
        title="T",
        description="",
        source_text="src",
        date="2025-09-02",
        time="18:00",
        location_name="Hall",
    )

    async with db.get_session() as session:
        session.add(event)
        await session.flush()
        session.add_all(
            [
                EventPoster(
                    event_id=event.id,
                    poster_hash="hash-new",
                    ocr_text="New text",
                    prompt_tokens=3,
                    completion_tokens=4,
                    total_tokens=7,
                    updated_at=datetime(2025, 2, 1),
                ),
                EventPoster(
                    event_id=event.id,
                    poster_hash="hash-old",
                    ocr_text="Old text",
                    prompt_tokens=1,
                    completion_tokens=1,
                    total_tokens=2,
                    updated_at=datetime(2025, 1, 1),
                ),
                EventPoster(
                    event_id=event.id,
                    poster_hash="hash-empty",
                    ocr_text="   ",
                    updated_at=datetime(2025, 3, 1),
                ),
            ]
        )
        await session.commit()

    texts = await main.get_event_poster_texts(event.id, db)

    assert texts == ["New text", "Old text"]
    assert await main.get_event_poster_texts(None, db) == []

@pytest.mark.asyncio
async def test_build_event_payload_includes_operator_extra(monkeypatch):
    async def fake_parse(text, *args, **kwargs):
        lower = text.lower()
        assert "original" in lower
        assert "extra" in lower
        assert kwargs.get("festival_names") is None
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    draft = await vk_intake.build_event_payload_from_vk(
        "Original announcement", operator_extra=" Extra context "
    )

    assert draft.source_text == "Original announcement\n\nExtra context"


@pytest.mark.asyncio
async def test_build_event_payload_uses_extra_when_text_missing(monkeypatch):
    async def fake_parse(text, *args, **kwargs):
        assert text.strip() == "Only extra"
        assert kwargs.get("festival_names") is None
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    draft = await vk_intake.build_event_payload_from_vk(
        "", operator_extra="  Only extra  "
    )

    assert draft.source_text == "Only extra"


@pytest.mark.asyncio
async def test_handle_vk_extra_message_exposes_text_links(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db,
        bot,
        *,
        operator_extra=None,
        force_festival=False,
    ):
        captured["chat_id"] = chat_id
        captured["operator_id"] = operator_id
        captured["operator_extra"] = operator_extra
        captured["force_festival"] = force_festival

    async def fake_parse(text, *args, **kwargs):
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    user_id = 4242
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=111),
        text="Check this link",
        caption=None,
        html_text=None,
        caption_html=None,
        entities=[
            types.MessageEntity(
                type="text_link",
                offset=11,
                length=4,
                url="https://example.com",
            )
        ],
        caption_entities=None,
    )

    main.vk_review_extra_sessions[user_id] = (7, "batch-7", False)
    await main.handle_vk_extra_message(message, db=object(), bot=object())

    assert user_id not in main.vk_review_extra_sessions
    operator_extra = captured.get("operator_extra")
    assert operator_extra == "Check this [link](https://example.com)"
    assert captured.get("force_festival") is False

    draft = await vk_intake.build_event_payload_from_vk(
        "Original announcement",
        operator_extra=operator_extra,
    )

    assert "Check this [link](https://example.com)" in draft.source_text
    html = linkify_for_telegraph(draft.source_text)
    assert '<a href="https://example.com">link</a>' in html


@pytest.mark.asyncio
async def test_handle_vk_extra_message_exposes_text_links_with_parentheses(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db,
        bot,
        *,
        operator_extra=None,
        force_festival=False,
    ):
        captured["chat_id"] = chat_id
        captured["operator_id"] = operator_id
        captured["operator_extra"] = operator_extra
        captured["force_festival"] = force_festival

    async def fake_parse(text, *args, **kwargs):
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    user_id = 5151
    url = "https://example.com/foo(bar)"
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=111),
        text="Check this link",
        caption=None,
        html_text=None,
        caption_html=None,
        entities=[
            types.MessageEntity(
                type="text_link",
                offset=11,
                length=4,
                url=url,
            )
        ],
        caption_entities=None,
    )

    main.vk_review_extra_sessions[user_id] = (8, "batch-8", False)
    await main.handle_vk_extra_message(message, db=object(), bot=object())

    assert user_id not in main.vk_review_extra_sessions
    operator_extra = captured.get("operator_extra")
    escaped_url = url.replace(")", "\\)")
    assert operator_extra == f"Check this [link]({escaped_url})"
    assert captured.get("force_festival") is False

    draft = await vk_intake.build_event_payload_from_vk(
        "Original announcement",
        operator_extra=operator_extra,
    )

    assert f"Check this [link]({escaped_url})" in draft.source_text
    html = linkify_for_telegraph(draft.source_text)
    assert '<a href="https://example.com/foo(bar)">link</a>' in html


@pytest.mark.asyncio
async def test_handle_vk_extra_message_preserves_emoji_offsets(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db,
        bot,
        *,
        operator_extra=None,
        force_festival=False,
    ):
        captured["operator_extra"] = operator_extra
        captured["force_festival"] = force_festival

    async def fake_parse(text, *args, **kwargs):
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    user_id = 5252
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=222),
        text="Check 😄 link",
        caption=None,
        html_text=None,
        caption_html=None,
        entities=[
            types.MessageEntity(
                type="text_link",
                offset=9,
                length=4,
                url="https://emoji.example",
            )
        ],
        caption_entities=None,
    )

    main.vk_review_extra_sessions[user_id] = (9, "batch-9", False)
    await main.handle_vk_extra_message(message, db=object(), bot=object())

    operator_extra = captured.get("operator_extra")
    assert operator_extra == "Check 😄 [link](https://emoji.example)"
    assert captured.get("force_festival") is False

    draft = await vk_intake.build_event_payload_from_vk(
        "Original announcement",
        operator_extra=operator_extra,
    )

    assert "Check 😄 [link](https://emoji.example)" in draft.source_text
    html = linkify_for_telegraph(draft.source_text)
    assert '<a href="https://emoji.example">link</a>' in html


@pytest.mark.asyncio
async def test_handle_vk_extra_message_forces_festival(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db,
        bot,
        *,
        operator_extra=None,
        force_festival=False,
    ):
        captured["force_festival"] = force_festival
        captured["operator_extra"] = operator_extra

    async def fake_parse(text, *args, **kwargs):
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)

    user_id = 5353
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=222),
        text="Доп.инфо",
        caption=None,
        html_text=None,
        caption_html=None,
        entities=[],
        caption_entities=None,
    )

    main.vk_review_extra_sessions[user_id] = (10, "batch-10", True)
    await main.handle_vk_extra_message(message, db=object(), bot=object())

    assert captured["force_festival"] is True
    assert captured["operator_extra"] == "Доп.инфо"
    main.vk_review_extra_sessions.clear()


@pytest.mark.asyncio
async def test_handle_vk_review_accept_fest_forces_flag(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox("
            "id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 1, 0, "", None, 0, "pending", "batch"),
        )
        await conn.commit()

    captured: dict[str, object] = {}

    async def fake_import_flow(
        chat_id,
        operator_id,
        inbox_id,
        batch_id,
        db_,
        bot_,
        *,
        operator_extra=None,
        force_festival=False,
    ):
        captured["force_festival"] = force_festival
        captured["args"] = (chat_id, operator_id, inbox_id, batch_id)

    monkeypatch.setattr(main, "_vkrev_import_flow", fake_import_flow)

    bot = DummyBot()

    async def fake_answer(self, *args, **kwargs):
        return None

    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)

    callback = types.CallbackQuery.model_validate(
        {
            "id": "1",
            "from": {"id": 10, "is_bot": False, "first_name": "Op"},
            "chat_instance": "1",
            "data": "vkrev:accept_fest:1",
            "message": {"message_id": 1, "date": 0, "chat": {"id": 5, "type": "private"}},
        }
    )
    callback._bot = bot

    await main.handle_vk_review_cb(callback, db, bot)

    assert captured["force_festival"] is True
    assert captured["args"] == (5, 10, 1, "batch")
    assert any("⏳" in msg.text for msg in bot.messages)


@pytest.mark.asyncio
async def test_handle_vk_review_accept_fest_extra_records_session(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO vk_inbox("
            "id, group_id, post_id, date, text, matched_kw, has_date, status, review_batch"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            (1, 1, 1, 0, "", None, 0, "pending", "batch"),
        )
        await conn.commit()

    main.vk_review_extra_sessions.clear()

    bot = DummyBot()

    async def fake_answer(self, *args, **kwargs):
        return None

    monkeypatch.setattr(types.CallbackQuery, "answer", fake_answer)

    callback = types.CallbackQuery.model_validate(
        {
            "id": "2",
            "from": {"id": 20, "is_bot": False, "first_name": "Op"},
            "chat_instance": "2",
            "data": "vkrev:accept_fest_extra:1",
            "message": {"message_id": 2, "date": 0, "chat": {"id": 7, "type": "private"}},
        }
    )
    callback._bot = bot

    await main.handle_vk_review_cb(callback, db, bot)

    assert main.vk_review_extra_sessions[20] == (1, "batch", True)
    assert bot.messages[-1].text == "Отправьте доп. информацию одним сообщением"
    main.vk_review_extra_sessions.clear()


@pytest.mark.asyncio
async def test_build_event_draft_handles_ocr_limit(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_download(urls):
        return [(b"img", "poster.jpg")]

    async def fake_process(media, need_catbox=True, need_ocr=False):
        return [PosterMedia(data=b"", name="poster", catbox_url="cat")], "ok"

    async def fake_parse(text, *args, **kwargs):
        return [
            {
                "title": "T",
                "date": "2025-09-02",
                "time": "10:00",
                "location_name": "Hall",
            }
        ]

    async def fake_recognize(
        db_obj, items, detail="auto", *, count_usage=True, log_context=None
    ):
        raise poster_ocr.PosterOcrLimitExceededError(
            "limit",
            spent_tokens=0,
            remaining=0,
        )

    monkeypatch.setattr(vk_intake, "_download_photo_media", fake_download)
    monkeypatch.setattr(vk_intake, "process_media", fake_process)
    monkeypatch.setattr(main, "parse_event_via_4o", fake_parse)
    monkeypatch.setattr(poster_ocr, "recognize_posters", fake_recognize)

    draft = await vk_intake.build_event_draft("text", photos=["url"], db=db)

    assert draft.poster_media
    assert draft.ocr_tokens_spent == 0
    assert draft.ocr_tokens_remaining == 0
    assert draft.ocr_limit_notice is not None
    assert "лимит" in draft.ocr_limit_notice.lower()
