import asyncio

import pytest

import main


async def _hang_forever() -> None:
    await asyncio.Future()


@pytest.mark.asyncio
async def test_watch_add_event_worker_restarts_stalled_worker(monkeypatch) -> None:
    queue: asyncio.Queue[object] = asyncio.Queue()
    queue.put_nowait(object())
    monkeypatch.setattr(main, "add_event_queue", queue)
    monkeypatch.setenv("STALL_GUARD_SECS", "1")

    started = asyncio.Event()

    async def fake_worker(_db, _bot, limit: int = 2) -> None:  # noqa: ANN001
        assert limit == 2
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(main, "add_event_queue_worker", fake_worker)
    main._ADD_EVENT_LAST_DEQUEUE_TS = main._time.monotonic() - 10.0

    old_worker = asyncio.create_task(_hang_forever())
    app = {"add_event_worker": old_worker}

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            main._watch_add_event_worker(app, db=object(), bot=object()),
            timeout=0.1,
        )

    new_worker = app["add_event_worker"]
    assert new_worker is not old_worker
    assert started.is_set()
    assert old_worker.cancelled() or old_worker.done()

    new_worker.cancel()
    with pytest.raises(asyncio.CancelledError):
        await new_worker
