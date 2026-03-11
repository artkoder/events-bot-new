import asyncio
import os
import sqlite3
import time
import json
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from db import Database
import kaggle_registry
from models import Event, EventSource, JobOutbox, JobTask, TelegramScannedMessage, TelegramSource
from source_parsing.telegram.handlers import TelegramMonitorReport
import source_parsing.telegram.service as tg_service
from source_parsing.telegram.service import (
    _import_results_with_retry,
    _build_secrets_payload,
    _is_auth_key_duplicated_failure,
    _preview_friendly_tg_post_url,
    _send_event_details,
    find_latest_telegram_results_json,
)


class _DummyMe:
    username = "eventsbotTestBot"


class _DummyBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str, dict]] = []

    async def get_me(self):
        return _DummyMe()

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append((int(chat_id), str(text), kwargs))


def _mk_auth_bundle(*, session: str) -> str:
    payload = {
        "session": session,
        "device_model": "Test Device",
        "system_version": "Test OS 1",
        "app_version": "Test App 1",
        "lang_code": "ru",
        "system_lang_code": "ru-RU",
    }
    raw = json.dumps(payload, ensure_ascii=False)
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


@pytest.mark.asyncio
async def test_send_event_details_reports_zero_changes():
    bot = _DummyBot()
    report = TelegramMonitorReport(
        run_id="test-run",
        messages_scanned=0,
        events_extracted=0,
        events_created=0,
        events_merged=0,
    )

    await _send_event_details(bot, 12345, report, db=None)

    assert bot.messages, "Expected a Smart Update detail message for zero-change run"
    _chat_id, text, kwargs = bot.messages[-1]
    assert "Smart Update (детали событий)" in text
    assert "Созданные события: 0" in text
    assert "Обновлённые события: 0" in text
    assert kwargs.get("parse_mode") == "HTML"


def test_resolve_tg_monitor_ops_status_distinguishes_error_empty_and_success():
    report = TelegramMonitorReport(run_id="r1")
    assert tg_service._resolve_tg_monitor_ops_status(report, report_loaded=False) == "error"

    empty_report = TelegramMonitorReport(run_id="r2", messages_scanned=0)
    assert tg_service._resolve_tg_monitor_ops_status(empty_report, report_loaded=True) == "empty"

    success_report = TelegramMonitorReport(run_id="r3", messages_scanned=4)
    assert tg_service._resolve_tg_monitor_ops_status(success_report, report_loaded=True) == "success"

    partial_report = TelegramMonitorReport(run_id="r4", messages_scanned=4, errors=["warn"])
    assert tg_service._resolve_tg_monitor_ops_status(partial_report, report_loaded=True) == "partial"


@pytest.mark.asyncio
async def test_import_results_with_retry_on_sqlite_lock(monkeypatch):
    attempts = {"n": 0}
    notices: list[str] = []

    async def fake_process(*_args, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OperationalError(
                "INSERT",
                {},
                sqlite3.OperationalError("database is locked"),
            )
        return TelegramMonitorReport(run_id="ok")

    async def fake_notify(text: str) -> None:
        notices.append(str(text))

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(tg_service, "process_telegram_results", fake_process)
    monkeypatch.setattr(tg_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(tg_service, "IMPORT_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(tg_service, "IMPORT_RETRY_BASE_DELAY_SEC", 0.01)

    report = await _import_results_with_retry(
        Path("dummy.json"),
        db=None,  # fake_process ignores db
        bot=None,
        run_id="retry-run",
        progress_callback=None,
        notify=fake_notify,
    )

    assert report.run_id == "ok"
    assert attempts["n"] == 2
    assert notices, "expected operator notification about retry"


@pytest.mark.asyncio
async def test_import_results_with_retry_raises_after_limit(monkeypatch):
    attempts = {"n": 0}

    async def fake_process(*_args, **_kwargs):
        attempts["n"] += 1
        raise OperationalError(
            "UPDATE",
            {},
            sqlite3.OperationalError("database is locked"),
        )

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(tg_service, "process_telegram_results", fake_process)
    monkeypatch.setattr(tg_service.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(tg_service, "IMPORT_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(tg_service, "IMPORT_RETRY_BASE_DELAY_SEC", 0.01)

    with pytest.raises(OperationalError):
        await _import_results_with_retry(
            Path("dummy.json"),
            db=None,  # fake_process ignores db
            bot=None,
            run_id="retry-fail",
            progress_callback=None,
            notify=None,
        )

    assert attempts["n"] == 2


def test_find_latest_telegram_results_json_picks_newest(tmp_path):
    older_dir = tmp_path / "tg-monitor-old"
    newer_dir = tmp_path / "tg-monitor-new"
    older_dir.mkdir()
    newer_dir.mkdir()

    older = older_dir / "telegram_results.json"
    newer = newer_dir / "telegram_results.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    now = time.time()
    older_ts = now - 120
    newer_ts = now - 5
    older.touch()
    newer.touch()
    os.utime(older, (older_ts, older_ts))
    os.utime(newer, (newer_ts, newer_ts))

    found = find_latest_telegram_results_json(tmp_path)
    assert found == newer


def test_find_latest_telegram_results_json_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_telegram_results_json(tmp_path)


def test_build_secrets_payload_prefers_s22_bundle_even_in_dev_mode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEV_MODE", "1")
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    s22 = _mk_auth_bundle(session="session_s22")
    e2e = _mk_auth_bundle(session="session_e2e")
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_S22", s22)
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", e2e)
    monkeypatch.delenv("TG_MONITORING_AUTH_BUNDLE_ENV", raising=False)

    payload = json.loads(_build_secrets_payload())
    assert payload["TELEGRAM_AUTH_BUNDLE_S22"] == s22
    assert payload["TG_SESSION"] == "session_s22"


def test_build_secrets_payload_uses_s22_bundle_outside_dev(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEV_MODE", raising=False)
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    s22 = _mk_auth_bundle(session="session_s22")
    e2e = _mk_auth_bundle(session="session_e2e")
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_S22", s22)
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", e2e)
    monkeypatch.delenv("TG_MONITORING_AUTH_BUNDLE_ENV", raising=False)

    payload = json.loads(_build_secrets_payload())
    assert payload["TELEGRAM_AUTH_BUNDLE_S22"] == s22
    assert payload["TG_SESSION"] == "session_s22"


def test_build_secrets_payload_can_use_explicit_override_bundle(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TG_API_ID", "123")
    monkeypatch.setenv("TG_API_HASH", "hash")
    monkeypatch.setenv("GOOGLE_API_KEY", "key")

    s22 = _mk_auth_bundle(session="session_s22")
    e2e = _mk_auth_bundle(session="session_e2e")
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_S22", s22)
    monkeypatch.setenv("TELEGRAM_AUTH_BUNDLE_E2E", e2e)
    monkeypatch.setenv("TG_MONITORING_AUTH_BUNDLE_ENV", "TELEGRAM_AUTH_BUNDLE_E2E")

    payload = json.loads(_build_secrets_payload())
    assert payload["TELEGRAM_AUTH_BUNDLE_S22"] == e2e
    assert payload["TG_SESSION"] == "session_e2e"


def test_is_auth_key_duplicated_failure_detects_telethon_error():
    assert _is_auth_key_duplicated_failure(
        {
            "status": "FAILED",
            "failureMessage": "AuthKeyDuplicatedError: The authorization key was used under two different IP addresses",
        }
    )


def test_preview_friendly_tg_post_url_adds_single_query():
    assert (
        _preview_friendly_tg_post_url("https://t.me/klassster/17729")
        == "https://t.me/klassster/17729?single"
    )
    assert (
        _preview_friendly_tg_post_url("https://t.me/klassster/17729?foo=1")
        == "https://t.me/klassster/17729?foo=1&single"
    )
    assert (
        _preview_friendly_tg_post_url("https://t.me/klassster/17729?single")
        == "https://t.me/klassster/17729?single"
    )
    # Non-canonical forms remain unchanged.
    assert _preview_friendly_tg_post_url("https://t.me/s/klassster/17729") == "https://t.me/s/klassster/17729"


@pytest.mark.asyncio
async def test_run_telegram_monitor_marks_error_on_cancelled(monkeypatch):
    finished: dict[str, object] = {}

    async def fake_start_ops_run(*_args, **_kwargs):
        return 777

    async def fake_finish_ops_run(*_args, **kwargs):
        finished.update(kwargs)

    @asynccontextmanager
    async def fake_global_lock(**_kwargs):
        yield

    async def fake_locked(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(tg_service, "start_ops_run", fake_start_ops_run)
    monkeypatch.setattr(tg_service, "finish_ops_run", fake_finish_ops_run)
    monkeypatch.setattr(tg_service, "_tg_monitor_global_lock", fake_global_lock)
    monkeypatch.setattr(tg_service, "_run_telegram_monitor_locked", fake_locked)

    with pytest.raises(asyncio.CancelledError):
        await tg_service.run_telegram_monitor(db=None, bot=None, chat_id=None, run_id="cancel-me")

    assert finished["status"] == "error"
    assert finished["details"]["run_id"] == "cancel-me"
    assert "cancelled" in finished["details"]["errors"]


@pytest.mark.asyncio
async def test_resume_telegram_monitor_jobs_imports_completed_kernel(monkeypatch, tmp_path):
    monkeypatch.setattr(kaggle_registry, "_REGISTRY_PATH", tmp_path / "kaggle_jobs.json")
    await kaggle_registry.register_job(
        "tg_monitoring",
        "owner/kernel",
        meta={"run_id": "run-123", "pid": 999999, "chat_id": 12345},
    )

    imported: dict[str, object] = {}

    class _DummyClient:
        def get_kernel_status(self, kernel_ref):
            assert kernel_ref == "owner/kernel"
            return {"status": "complete"}

    async def fake_download_results(_client, kernel_ref, run_id):
        imported["downloaded"] = (kernel_ref, run_id)
        path = tmp_path / "telegram_results.json"
        path.write_text("{}", encoding="utf-8")
        return path

    async def fake_import(
        _db,
        *,
        results_path,
        bot=None,
        chat_id=None,
        run_id=None,
        send_progress=False,
        trigger="manual_import_only",
        operator_id=None,
    ):
        imported["import"] = {
            "results_path": str(results_path),
            "chat_id": chat_id,
            "run_id": run_id,
            "send_progress": send_progress,
            "trigger": trigger,
            "operator_id": operator_id,
        }
        return TelegramMonitorReport(run_id=run_id, messages_scanned=1)

    monkeypatch.setattr(tg_service, "KaggleClient", _DummyClient)
    monkeypatch.setattr(tg_service, "_download_results", fake_download_results)
    monkeypatch.setattr(tg_service, "run_telegram_import_from_results", fake_import)

    recovered = await tg_service.resume_telegram_monitor_jobs(db=None, bot=None)

    assert recovered == 1
    assert imported["downloaded"] == ("owner/kernel", "run-123")
    assert imported["import"]["run_id"] == "run-123"
    assert imported["import"]["trigger"] == "recovery_import"
    assert await kaggle_registry.list_jobs("tg_monitoring") == []


@pytest.mark.asyncio
async def test_recreate_telegram_events_from_results_removes_event_joboutbox_and_scanned_marks(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    source_username = "devchan"
    message_id = 12345
    source_link = f"https://t.me/{source_username}/{message_id}"

    async with db.get_session() as session:
        source = TelegramSource(username=source_username, enabled=True)
        session.add(source)
        await session.commit()
        await session.refresh(source)

        event = Event(
            title="DEV test event",
            description="desc",
            source_text="source",
            date="2026-03-15",
            time="19:00",
            location_name="Venue",
        )
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(
            EventSource(
                event_id=int(event.id),
                source_type="telegram",
                source_url=source_link,
                source_chat_username=source_username,
                source_message_id=message_id,
            )
        )
        session.add(
            JobOutbox(
                event_id=int(event.id),
                task=JobTask.telegraph_build,
                payload={"kind": "test"},
            )
        )
        session.add(
            TelegramScannedMessage(
                source_id=int(source.id),
                message_id=message_id,
                status="done",
                events_extracted=1,
                events_imported=1,
            )
        )
        await session.commit()

    results_path = tmp_path / "telegram_results.json"
    results_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "messages": [
                    {
                        "source_username": source_username,
                        "message_id": message_id,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stats = await tg_service.recreate_telegram_events_from_results(db, results_path=results_path)

    assert stats.event_ids_found == 1
    assert stats.events_deleted == 1
    assert stats.joboutbox_deleted == 1
    assert stats.scanned_deleted == 1

    async with db.get_session() as session:
        assert await session.get(Event, int(event.id)) is None

        event_source_rows = (
            await session.execute(
                select(EventSource).where(EventSource.source_url == source_link)
            )
        ).scalars().all()
        assert event_source_rows == []

        job_rows = (
            await session.execute(
                select(JobOutbox).where(JobOutbox.event_id == int(event.id))
            )
        ).scalars().all()
        assert job_rows == []

        scanned = await session.get(TelegramScannedMessage, (int(source.id), message_id))
        assert scanned is None
