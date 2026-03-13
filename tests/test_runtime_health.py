from types import SimpleNamespace

import pytest

import main


class _RawConn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, query: str):
        return query


class _DbOk:
    def raw_conn(self):
        return _RawConn()


class _AliveTask:
    def done(self) -> bool:
        return False


class _FinishedTask:
    def __init__(self, exc: Exception | None = None):
        self._exc = exc

    def done(self) -> bool:
        return True

    def cancelled(self) -> bool:
        return False

    def exception(self):
        return self._exc


@pytest.mark.asyncio
async def test_runtime_health_report_ok(monkeypatch):
    now = 1000.0
    monkeypatch.setattr(main._time, "monotonic", lambda: now)
    runtime_health = {
        "boot_monotonic": now - 30,
        "last_tick_monotonic": now - 5,
        "ready": True,
    }
    app = {
        "runtime_health": runtime_health,
        "daily_scheduler": _AliveTask(),
        "add_event_watch": _AliveTask(),
        "job_outbox_worker": _AliveTask(),
    }

    status, payload = await main._runtime_health_report(
        app,
        _DbOk(),
        SimpleNamespace(session=SimpleNamespace(closed=False)),
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["db"] == "ok"
    assert payload["issues"] == []


@pytest.mark.asyncio
async def test_runtime_health_report_fails_for_dead_task(monkeypatch):
    now = 2000.0
    monkeypatch.setattr(main._time, "monotonic", lambda: now)
    app = {
        "runtime_health": {
            "boot_monotonic": now - 60,
            "last_tick_monotonic": now - 5,
            "ready": True,
        },
        "daily_scheduler": _FinishedTask(RuntimeError("boom")),
        "add_event_watch": _AliveTask(),
    }

    status, payload = await main._runtime_health_report(
        app,
        _DbOk(),
        SimpleNamespace(session=SimpleNamespace(closed=False)),
    )

    assert status == 503
    assert payload["ok"] is False
    assert "daily_scheduler:exception:RuntimeError" in payload["issues"]


@pytest.mark.asyncio
async def test_runtime_health_report_allows_startup_grace(monkeypatch):
    now = 2500.0
    monkeypatch.setattr(main._time, "monotonic", lambda: now)
    app = {
        "runtime_health": {
            "boot_monotonic": now - 20,
            "last_tick_monotonic": None,
            "ready": False,
        }
    }

    status, payload = await main._runtime_health_report(
        app,
        _DbOk(),
        SimpleNamespace(session=SimpleNamespace(closed=False)),
    )

    assert status == 200
    assert payload["ok"] is True
    assert payload["ready"] is False
    assert payload["issues"] == []


@pytest.mark.asyncio
async def test_runtime_health_report_fails_for_stale_heartbeat(monkeypatch):
    now = 3000.0
    monkeypatch.setattr(main._time, "monotonic", lambda: now)
    app = {
        "runtime_health": {
            "boot_monotonic": now - 120,
            "last_tick_monotonic": now - 90,
            "ready": True,
        },
        "daily_scheduler": _AliveTask(),
        "add_event_watch": _AliveTask(),
    }

    status, payload = await main._runtime_health_report(
        app,
        _DbOk(),
        SimpleNamespace(session=SimpleNamespace(closed=False)),
    )

    assert status == 503
    assert payload["ok"] is False
    assert "heartbeat:stale" in payload["issues"]
