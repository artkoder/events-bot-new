from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

import main
from models import Event, EventMediaAsset, JobOutbox, JobStatus, JobTask
from source_parsing.telegram.handlers import process_telegram_results


@pytest.mark.asyncio
async def test_tg_monitoring_attaches_message_videos_to_single_event(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()
    event_date = (date.today() + timedelta(days=2)).isoformat()
    results = {
        "schema_version": 2,
        "run_id": "r-video-single",
        "generated_at": generated_at,
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 1,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Video Channel",
                "message_id": 77,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/77",
                "text": "Пост с видео",
                "has_video": True,
                "video_status": "supabase",
                "videos": [
                    {
                        "sha256": "abc123",
                        "size_bytes": 123456,
                        "mime_type": "video/mp4",
                        "supabase_url": "https://example.supabase.co/storage/v1/object/public/events-media/v/abc123.mp4",
                        "supabase_path": "v/abc123.mp4",
                    }
                ],
                "events": [
                    {
                        "title": "Концерт",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                    }
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        from models import Event

        async with db_obj.get_session() as session:
            ev = Event(
                title=candidate.title or "",
                date=candidate.date,
                time=candidate.time,
                location_name=candidate.location_name or "",
                city=candidate.city or "Калининград",
                description="desc",
                source_text=candidate.source_text or "",
            )
            session.add(ev)
            await session.commit()
            await session.refresh(ev)
            event_id = int(ev.id or 0)
        return SimpleNamespace(status="created", event_id=event_id, added_posters=0, reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    progress_done = []

    async def capture_progress(progress):
        if progress.stage == "done":
            progress_done.append(progress)

    await process_telegram_results(results_path, db, bot=None, progress_callback=capture_progress)

    async with db.get_session() as session:
        rows = (await session.execute(select(EventMediaAsset))).scalars().all()
    assert len(rows) == 1
    asset = rows[0]
    assert asset.kind == "video"
    assert asset.supabase_path == "v/abc123.mp4"
    assert asset.sha256 == "abc123"
    assert int(asset.size_bytes or 0) == 123456
    assert asset.mime_type == "video/mp4"

    assert progress_done, "expected done progress callback"
    assert progress_done[0].post_video_status == "supabase"


@pytest.mark.asyncio
async def test_tg_monitoring_attaches_message_videos_to_nochange_event_and_requeues_telegraph(
    tmp_path, monkeypatch
):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()
    event_date = (date.today() + timedelta(days=2)).isoformat()

    async with db.get_session() as session:
        ev = Event(
            title="Existing",
            date=event_date,
            time="19:00",
            location_name="Venue",
            city="Калининград",
            description="desc",
            source_text="src",
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        eid = int(ev.id or 0)

    results = {
        "schema_version": 2,
        "run_id": "r-video-nochange",
        "generated_at": generated_at,
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 1,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Video Channel",
                "message_id": 79,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/79",
                "text": "Пост с видео (но событие уже есть и без изменений)",
                "has_video": True,
                "video_status": "supabase",
                "videos": [
                    {
                        "sha256": "zzz999",
                        "size_bytes": 111111,
                        "mime_type": "video/mp4",
                        "supabase_url": "https://example.supabase.co/storage/v1/object/public/events-media/v/zzz999.mp4",
                        "supabase_path": "v/zzz999.mp4",
                    }
                ],
                "events": [
                    {
                        "title": "Existing",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                    }
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    async def fake_smart_event_update(_db_obj, _candidate, **_kwargs):
        return SimpleNamespace(status="skipped_nochange", event_id=eid, added_posters=0, reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    progress_done = []

    async def capture_progress(progress):
        if progress.stage == "done":
            progress_done.append(progress)

    await process_telegram_results(results_path, db, bot=None, progress_callback=capture_progress)

    async with db.get_session() as session:
        rows = (await session.execute(select(EventMediaAsset))).scalars().all()
        jobs = (
            await session.execute(
                select(JobOutbox)
                .where(JobOutbox.event_id == eid, JobOutbox.task == JobTask.telegraph_build)
                .order_by(JobOutbox.id.desc())
                .limit(1)
            )
        ).scalars().all()

    assert len(rows) == 1
    assert rows[0].event_id == eid
    assert rows[0].kind == "video"

    assert jobs, "expected telegraph_build job to be enqueued"
    assert jobs[0].status == JobStatus.pending

    assert progress_done, "expected done progress callback"
    assert progress_done[0].post_video_status == "supabase"


@pytest.mark.asyncio
async def test_tg_monitoring_skips_message_videos_for_multi_event_posts(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()
    event_date = (date.today() + timedelta(days=2)).isoformat()
    results = {
        "schema_version": 2,
        "run_id": "r-video-multi",
        "generated_at": generated_at,
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 1,
            "events_extracted": 2,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Video Channel",
                "message_id": 78,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/78",
                "text": "Пост с двумя событиями и одним видео",
                "has_video": True,
                "videos": [
                    {
                        "sha256": "def456",
                        "size_bytes": 222222,
                        "mime_type": "video/mp4",
                        "supabase_url": "https://example.supabase.co/storage/v1/object/public/events-media/v/def456.mp4",
                        "supabase_path": "v/def456.mp4",
                    }
                ],
                "events": [
                    {
                        "title": "Событие 1",
                        "date": event_date,
                        "time": "18:00",
                        "location_name": "Venue",
                    },
                    {
                        "title": "Событие 2",
                        "date": event_date,
                        "time": "20:00",
                        "location_name": "Venue",
                    },
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        from models import Event

        async with db_obj.get_session() as session:
            ev = Event(
                title=candidate.title or "",
                date=candidate.date,
                time=candidate.time,
                location_name=candidate.location_name or "",
                city=candidate.city or "Калининград",
                description="desc",
                source_text=candidate.source_text or "",
            )
            session.add(ev)
            await session.commit()
            await session.refresh(ev)
            event_id = int(ev.id or 0)
        return SimpleNamespace(status="created", event_id=event_id, added_posters=0, reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    progress_done = []

    async def capture_progress(progress):
        if progress.stage == "done":
            progress_done.append(progress)

    await process_telegram_results(results_path, db, bot=None, progress_callback=capture_progress)

    async with db.get_session() as session:
        rows = (await session.execute(select(EventMediaAsset))).scalars().all()
    assert rows == []

    assert progress_done, "expected done progress callback"
    assert progress_done[0].post_video_status == "skipped:multi_event_message"
    assert progress_done[0].skip_breakdown.get("video_skipped_multi_event") == 1
