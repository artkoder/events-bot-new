import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
from types import SimpleNamespace

import main
import vk_intake
import vk_review
from main import Database
from models import Event, JobTask, Festival


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
