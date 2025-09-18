import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import hashlib
from datetime import datetime

import pytest
from types import SimpleNamespace

import main
import poster_ocr
import vk_intake
import vk_review
from main import Database
from models import Event, EventPoster, JobTask, Festival, PosterOcrCache
from poster_media import PosterMedia
from sqlmodel import select


class DummyBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(SimpleNamespace(chat_id=chat_id, text=text))
        return SimpleNamespace(message_id=1)

    async def send_media_group(self, chat_id, media):
        pass


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
        db=None,
    ):
        captured["festival_names"] = festival_names
        return draft

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
    monkeypatch.setattr(vk_intake, "build_event_draft", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    async with db.get_session() as session:
        ev = await session.get(Event, captured["event_id"])
    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert captured["festival_names"] == ["Fest One"]
    assert JobTask.vk_sync not in tasks
    assert bot.messages[-1].text == (
        "Импортировано\n"
        "Тип: —\n"
        "Дата начала: 2025-09-02\n"
        "Дата окончания: —\n"
        "Время: 10:00\n"
        "Бесплатное: нет"
    )


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
        db=None,
    ):
        return draft

    async def fake_mark_imported(db_, inbox_id, batch_id, operator_id, event_id, event_date):
        pass

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        return "job"

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_draft", fake_build)
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
        return draft

    async def fake_mark_imported(*args, **kwargs):
        pass

    async def fake_enqueue_job(*args, **kwargs):
        return "job"

    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch)
    monkeypatch.setattr(vk_intake, "build_event_draft", fake_build)
    monkeypatch.setattr(vk_review, "mark_imported", fake_mark_imported)
    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    bot = DummyBot()
    await main._vkrev_import_flow(1, 1, 1, "batch1", db, bot)

    info_text = bot.messages[-1].text
    assert "OCR недоступен" in info_text
    assert "OCR: потрачено 0, осталось 0" in info_text


@pytest.mark.asyncio
async def test_vk_persist_event_updates_ocr_records(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_assign_event_topics(event_obj):
        return [], len(event_obj.description or ""), None, False

    async def fake_schedule_event_update_tasks(db_obj, event_obj, drain_nav=True):
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
