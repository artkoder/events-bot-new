import pytest

import main
import vk_intake
from main import Database
from models import Event, JobTask


@pytest.mark.asyncio
async def test_persist_event_and_pages_sets_source_post_url_and_skips_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)

    draft = vk_intake.EventDraft(title="T", date="2025-09-02", time="10:00", source_text="T")
    res = await vk_intake.persist_event_and_pages(
        draft, [], db, source_post_url="https://vk.com/wall-1_2"
    )

    async with db.get_session() as session:
        ev = await session.get(Event, res.event_id)
    assert ev.source_post_url == "https://vk.com/wall-1_2"
    assert JobTask.vk_sync not in tasks


@pytest.mark.asyncio
async def test_persist_event_and_pages_persists_extended_fields(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_schedule_event_update_tasks(db_, ev, drain_nav=True, skip_vk_sync=False):
        return {}

    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)

    draft = vk_intake.EventDraft(
        title="Выставка",
        date="2025-09-01",
        time="10:00",
        venue="Музей",
        description="Описание",
        festival="Фестиваль",
        location_address="Адрес",
        city="Калининград",
        ticket_price_min=100,
        ticket_price_max=250,
        event_type="выставка",
        emoji="🎨",
        end_date="2025-09-10",
        is_free=True,
        pushkin_card=True,
        source_text="Источник",
    )

    res = await vk_intake.persist_event_and_pages(draft, [], db)

    async with db.get_session() as session:
        saved = await session.get(Event, res.event_id)

    assert saved.event_type == "выставка"
    assert saved.end_date == "2025-09-10"
    assert saved.city == "Калининград"
    assert saved.ticket_price_min == 100
    assert saved.ticket_price_max == 250
    assert saved.is_free is True
    assert saved.pushkin_card is True
    assert saved.emoji == "🎨"
    assert saved.description == "Описание"
    assert saved.festival == "Фестиваль"


@pytest.mark.asyncio
async def test_persist_event_and_pages_classifies_topics(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async def fake_schedule_event_update_tasks(
        db_obj, event_obj, drain_nav=True, skip_vk_sync=False
    ):
        return {}

    calls = {"topics": 0}

    async def fake_classify(event: Event):
        calls["topics"] += 1
        return ["HANDMADE"]

    monkeypatch.setattr(main, "schedule_event_update_tasks", fake_schedule_event_update_tasks)
    monkeypatch.setattr(main, "classify_event_topics", fake_classify)

    draft = vk_intake.EventDraft(
        title="T",
        date="2025-09-01",
        time="10:00",
        venue="Club",
        source_text="Music",
    )

    res = await vk_intake.persist_event_and_pages(draft, [], db)

    async with db.get_session() as session:
        saved = await session.get(Event, res.event_id)

    assert calls["topics"] == 1
    assert saved.topics == ["HANDMADE"]
    assert saved.topics_manual is False


@pytest.mark.asyncio
async def test_schedule_event_update_tasks_enqueues_and_runs_vk_sync(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    tasks = []

    async def fake_enqueue_job(db_, eid, task, depends_on=None, coalesce_key=None):
        tasks.append(task)
        return "job"

    async def fake_sync_vk_source_post(ev, text, db_, bot, ics_url=None):
        return "https://vk.com/wall-1_1"

    monkeypatch.setattr(main, "enqueue_job", fake_enqueue_job)
    monkeypatch.setattr(main, "sync_vk_source_post", fake_sync_vk_source_post)

    ev = Event(
        title="T",
        description="",
        festival=None,
        date="2025-09-02",
        time="10:00",
        location_name="",
        source_text="T",
        source_post_url="http://example.com",
    )

    async with db.get_session() as session:
        saved, _ = await main.upsert_event(session, ev)

    await main.schedule_event_update_tasks(db, saved)

    assert JobTask.vk_sync in tasks

    await main.job_sync_vk_source_post(saved.id, db, None)

    async with db.get_session() as session:
        updated = await session.get(Event, saved.id)
    assert updated.source_vk_post_url == "https://vk.com/wall-1_1"


@pytest.mark.asyncio
async def test_upsert_event_preserves_manual_topics(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        existing = Event(
            title="T",
            description="D",
            festival=None,
            date="2025-09-02",
            time="10:00",
            location_name="Loc",
            location_address="Addr",
            city="City",
            source_text="Manual",
        topics=["EXHIBITIONS"],
            topics_manual=True,
        )
        session.add(existing)
        await session.commit()
        event_id = existing.id

    async with db.get_session() as session:
        new = Event(
            title="T",
            description="D",
            festival=None,
            date="2025-09-02",
            time="10:00",
            location_name="Loc",
            location_address="Addr",
            city="City",
            source_text="Manual",
            topics=["CONCERTS"],
            topics_manual=False,
        )
        saved, created = await main.upsert_event(session, new)
        assert created is False
        assert saved.id == event_id

    async with db.get_session() as session:
        refreshed = await session.get(Event, event_id)
        assert refreshed is not None
        assert refreshed.topics == ["EXHIBITIONS"]
        assert refreshed.topics_manual is True


@pytest.mark.asyncio
async def test_upsert_event_updates_topics_when_not_manual(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        existing = Event(
            title="T",
            description="D",
            festival=None,
            date="2025-09-03",
            time="11:00",
            location_name="Loc",
            location_address="Addr",
            city="City",
            source_text="Auto",
            topics=[],
            topics_manual=False,
        )
        session.add(existing)
        await session.commit()
        event_id = existing.id

    async with db.get_session() as session:
        new = Event(
            title="T",
            description="D",
            festival=None,
            date="2025-09-03",
            time="11:00",
            location_name="Loc",
            location_address="Addr",
            city="City",
            source_text="Auto",
            topics=["CONCERTS"],
            topics_manual=True,
        )
        saved, created = await main.upsert_event(session, new)
        assert created is False
        assert saved.id == event_id

    async with db.get_session() as session:
        refreshed = await session.get(Event, event_id)
        assert refreshed is not None
        assert refreshed.topics == ["CONCERTS"]
        assert refreshed.topics_manual is True
