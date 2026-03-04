from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from models import TelegramSource
from source_parsing.telegram.handlers import process_telegram_results
from source_parsing.telegram.service import _format_skipped_posts_block


@pytest.mark.asyncio
async def test_tg_monitoring_collects_skipped_posts_and_persists_channel_title(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    today = date.today()
    future = (today + timedelta(days=2)).isoformat()
    past = (today - timedelta(days=3)).isoformat()
    generated_at = datetime.now(timezone.utc).isoformat()

    results = {
        "run_id": "r1",
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
                "source_title": "Test Channel Title",
                "message_id": 10,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/10",
                "text": "Текст поста про событие.\n📍 Venue",
                "events": [
                    {
                        "title": "Будущее событие",
                        "date": future,
                        "time": "19:00",
                        "location_name": "Venue",
                        "raw_excerpt": "Коротко про событие",
                    },
                    {
                        "title": "Прошедшее событие",
                        "date": past,
                        "time": "18:00",
                        "location_name": "Venue",
                        "raw_excerpt": "Коротко",
                    },
                ],
            }
        ],
    }

    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    created_ids: list[int] = []

    async def fake_smart_event_update(db_obj, candidate, **_kwargs):
        from models import Event

        if not created_ids:
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
                created_ids.append(int(ev.id))
            return SimpleNamespace(status="created", event_id=created_ids[0], added_posters=0, reason=None)
        return SimpleNamespace(status="skipped_nochange", event_id=None, added_posters=0, reason="no_changes")

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    report = await process_telegram_results(results_path, db, bot=None)

    assert report.created_events, "expected created event info"
    assert report.skipped_posts, "expected skipped/partial post list"

    skipped = report.skipped_posts[0]
    assert skipped.source_username == "testchan"
    assert skipped.source_title == "Test Channel Title"
    assert skipped.events_extracted == 2
    assert skipped.events_imported == 1
    assert skipped.skip_breakdown.get("past_event") == 1

    # Title is persisted to DB for sources list UX.
    async with db.get_session() as session:
        from sqlalchemy import select

        src = (
            await session.execute(
                select(TelegramSource).where(TelegramSource.username == "testchan")
            )
        ).scalar_one_or_none()
        assert src is not None
        assert (src.title or "").strip() == "Test Channel Title"

    # Formatting block contains link + reason breakdown.
    lines = _format_skipped_posts_block(report, limit=5)
    joined = "\n".join(lines)
    assert "Пропущенные/частично обработанные посты" in joined
    assert "past_event=1" in joined
    assert "https://t.me/testchan/10" in joined


@pytest.mark.asyncio
async def test_tg_monitoring_metrics_only_skips_smart_update_and_records_metrics(tmp_path, monkeypatch):
    from models import TelegramScannedMessage

    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    now = datetime.now(timezone.utc)
    generated_at = now.isoformat()
    message_ts = int(now.timestamp())

    async with db.get_session() as session:
        src = TelegramSource(username="testchan", enabled=True)
        session.add(src)
        await session.commit()
        await session.refresh(src)
        session.add(
            TelegramScannedMessage(
                source_id=int(src.id),
                message_id=10,
                status="done",
                events_extracted=2,
                events_imported=2,
            )
        )
        await session.commit()

    # Seed baseline metrics for the same channel so popularity scoring has a sample.
    async with db.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO telegram_post_metric(
                source_id, message_id, age_day, source_url, message_ts, collected_ts, views, likes
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (int(src.id), 9, 0, "https://t.me/testchan/9", message_ts, message_ts, 10, 1),
        )
        # Baseline is computed only for messages that are known to contain events.
        await conn.execute(
            """
            INSERT INTO telegram_scanned_message(
                source_id, message_id, status, events_extracted, events_imported, processed_at
            ) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP)
            """,
            (int(src.id), 9, "done", 1, 1),
        )
        await conn.commit()

    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "1")
    monkeypatch.setenv("POST_POPULARITY_VIEWS_MULT", "1.5")
    monkeypatch.setenv("POST_POPULARITY_LIKES_MULT", "1.5")

    results = {
        "run_id": "r2",
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
                "source_title": "Test Channel Title",
                "message_id": 10,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/10",
                "text": "Текст поста про событие.",
                "metrics": {"views": 100, "likes": 10},
                "events": [
                    {"title": "Событие 1", "date": now.date().isoformat(), "time": "19:00", "location_name": "Venue"},
                    {"title": "Событие 2", "date": now.date().isoformat(), "time": "20:00", "location_name": "Venue"},
                ],
            }
        ],
    }

    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    import source_parsing.telegram.handlers as tg_handlers

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("smart_event_update must not run for already-scanned message_id")

    monkeypatch.setattr(tg_handlers, "smart_event_update", should_not_run)

    report = await process_telegram_results(results_path, db, bot=None)

    assert report.messages_new == 0
    assert report.messages_metrics_only == 1
    assert len(report.metrics_only_posts) == 1
    assert report.metrics_only_posts[0].status == "metrics_only"
    assert report.popular_posts, "expected popular post marker when views/likes exceed baseline"
    assert "⭐" in (report.popular_posts[0].popularity or "")
    assert "👍" in (report.popular_posts[0].popularity or "")

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT views, likes FROM telegram_post_metric WHERE source_id=? AND message_id=? AND age_day=?",
            (int(src.id), 10, 0),
        )
        row = await cur.fetchone()
    assert row is not None
    assert row[0] == 100
    assert row[1] == 10


@pytest.mark.asyncio
async def test_tg_monitoring_ignores_new_messages_without_events(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()
    results = {
        "run_id": "r-noevents",
        "generated_at": generated_at,
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 0,
            "events_extracted": 0,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Test Channel Title",
                "message_id": 10,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/10",
                "text": "Пост без событий",
                "metrics": {"views": 100, "likes": 10},
                "events": [],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    import source_parsing.telegram.handlers as tg_handlers

    async def should_not_run(*_args, **_kwargs):
        raise AssertionError("smart_event_update must not run for empty events[] message")

    monkeypatch.setattr(tg_handlers, "smart_event_update", should_not_run)

    progress_log: list[str] = []

    async def capture_progress(progress) -> None:
        progress_log.append(str(progress.stage))

    await process_telegram_results(
        results_path,
        db,
        bot=None,
        progress_callback=capture_progress,
    )

    # No progress and no scanned-message mark for a new no-events post.
    assert progress_log == []
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT COUNT(1)
            FROM telegram_scanned_message s
            JOIN telegram_source src ON src.id = s.source_id
            WHERE src.username=? AND s.message_id=?
            """,
            ("testchan", 10),
        )
        (cnt,) = await cur.fetchone()
    assert int(cnt or 0) == 0


@pytest.mark.asyncio
async def test_tg_monitoring_schema_v2_persists_source_meta_and_keeps_manual_festival_series(tmp_path):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()

    async with db.get_session() as session:
        src = TelegramSource(
            username="testchan",
            enabled=True,
            title="Old title",
            festival_source=True,
            festival_series="Manual Series",
        )
        session.add(src)
        await session.commit()

    results = {
        "schema_version": 2,
        "run_id": "r3",
        "generated_at": generated_at,
        "sources_meta": [
            {
                "username": "testchan",
                "source_type": "channel",
                "title": "New Meta Title",
                "about": "Официальный канал фестиваля. Сайт: https://openfest.example.org",
                "about_links": ["https://openfest.example.org"],
                "fetched_at": generated_at,
                "meta_hash": "sha256:testhash",
                "suggestions": {
                    "festival_series": "Suggested Series",
                    "website_url": "https://openfest.example.org",
                    "confidence": 0.86,
                    "rationale_short": "В названии и описании явно указан фестиваль.",
                },
            }
        ],
        "stats": {
            "sources_total": 1,
            "messages_scanned": 1,
            "messages_with_events": 0,
            "events_extracted": 0,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Legacy Message Title",
                "message_id": 42,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/42",
                "text": "Служебный пост без событий",
                "events": [],
            }
        ],
    }

    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    report = await process_telegram_results(results_path, db, bot=None)
    assert report.messages_scanned >= 0

    async with db.get_session() as session:
        from sqlalchemy import select

        src = (
            await session.execute(
                select(TelegramSource).where(TelegramSource.username == "testchan")
            )
        ).scalar_one_or_none()
        assert src is not None
        assert src.title == "New Meta Title"
        assert src.about == "Официальный канал фестиваля. Сайт: https://openfest.example.org"
        assert src.about_links_json == ["https://openfest.example.org"]
        assert src.meta_hash == "sha256:testhash"
        assert src.meta_fetched_at is not None
        assert src.suggested_festival_series == "Suggested Series"
        assert src.suggested_website_url == "https://openfest.example.org"
        assert src.suggestion_confidence == pytest.approx(0.86, abs=1e-6)
        assert src.suggestion_rationale == "В названии и описании явно указан фестиваль."
        # Manual value should remain unchanged until operator explicitly accepts suggestion in /tg.
        assert src.festival_series == "Manual Series"


@pytest.mark.asyncio
async def test_tg_monitoring_emits_import_progress_per_message(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    generated_at = datetime.now(timezone.utc).isoformat()
    event_date = (date.today() + timedelta(days=1)).isoformat()

    results = {
        "run_id": "r-progress",
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
                "source_title": "Test Channel",
                "message_id": 99,
                "message_date": generated_at,
                "source_link": "https://t.me/testchan/99",
                "text": "Анонс события",
                "events": [
                    {
                        "title": "Новый концерт",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                        "raw_excerpt": "Коротко",
                    }
                ],
            }
        ],
    }
    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    created_ids: list[int] = []

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
            created_ids.append(int(ev.id))
        return SimpleNamespace(status="created", event_id=created_ids[-1], added_posters=2, reason=None)

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    progress_log: list[tuple[str, str, int, int, int]] = []

    async def capture_progress(progress) -> None:
        progress_log.append(
            (
                str(progress.stage),
                str(progress.status),
                int(progress.current_no),
                int(progress.total_no),
                int(progress.events_imported),
            )
        )

    report = await process_telegram_results(
        results_path,
        db,
        bot=None,
        progress_callback=capture_progress,
    )

    assert report.events_created == 1
    assert progress_log, "expected import progress callbacks"
    assert progress_log[0][0] == "start"
    assert progress_log[-1][0] == "done"
    assert progress_log[-1][1] in {"done", "partial"}
    assert progress_log[-1][2] == 1
    assert progress_log[-1][3] == 1
    assert progress_log[-1][4] >= 1


@pytest.mark.asyncio
async def test_tg_monitoring_processes_messages_in_chronological_order(tmp_path, monkeypatch):
    db = main.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    older_dt = datetime(2026, 2, 10, 10, 0, tzinfo=timezone.utc)
    newer_dt = datetime(2026, 2, 11, 10, 0, tzinfo=timezone.utc)
    event_date = (date.today() + timedelta(days=3)).isoformat()

    # Intentionally reversed order in payload: newer first, older second.
    results = {
        "run_id": "r-order",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "sources_total": 1,
            "messages_scanned": 2,
            "messages_with_events": 2,
            "events_extracted": 2,
        },
        "messages": [
            {
                "source_username": "testchan",
                "source_title": "Test Channel",
                "message_id": 200,
                "message_date": newer_dt.isoformat(),
                "source_link": "https://t.me/testchan/200",
                "text": "Новый пост",
                "events": [
                    {
                        "title": "Один и тот же ивент",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                    }
                ],
            },
            {
                "source_username": "testchan",
                "source_title": "Test Channel",
                "message_id": 100,
                "message_date": older_dt.isoformat(),
                "source_link": "https://t.me/testchan/100",
                "text": "Старый пост",
                "events": [
                    {
                        "title": "Один и тот же ивент",
                        "date": event_date,
                        "time": "19:00",
                        "location_name": "Venue",
                    }
                ],
            },
        ],
    }

    results_path = Path(tmp_path) / "telegram_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")

    processed_message_ids: list[int] = []

    async def fake_smart_event_update(_db_obj, candidate, **_kwargs):
        processed_message_ids.append(int(candidate.source_message_id or 0))
        return SimpleNamespace(status="skipped_nochange", event_id=None, added_posters=0, reason="no_changes")

    import source_parsing.telegram.handlers as tg_handlers

    monkeypatch.setattr(tg_handlers, "smart_event_update", fake_smart_event_update)

    await process_telegram_results(results_path, db, bot=None)

    assert processed_message_ids == [100, 200]
