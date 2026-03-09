from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json

import pytest

from db import Database
from models import VideoAnnounceSession, VideoAnnounceSessionStatus
import source_parsing.telegram.service as tg_service
import video_announce.poller as video_poller


@pytest.mark.asyncio
async def test_tg_monitor_waits_for_expected_kernel_dataset_binding(monkeypatch):
    class _FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def kernel_has_dataset_sources(self, kernel_ref: str, expected_sources: list[str]):
            self.calls += 1
            if self.calls == 1:
                return False, {"dataset_sources": ["zigomaro/old-dataset"]}
            return True, {"dataset_sources": ["zigomaro/old-dataset", *expected_sources]}

    seen: list[tuple[str, list[str]]] = []

    async def _fake_sleep(_seconds: float) -> None:
        return None

    async def _status_callback(phase: str, _kernel_ref: str, status: dict | None) -> None:
        seen.append((phase, list((status or {}).get("_actual_dataset_sources") or [])))

    monkeypatch.setattr(tg_service.asyncio, "sleep", _fake_sleep)

    meta = await tg_service._await_kernel_dataset_sources(  # noqa: SLF001
        _FakeClient(),
        "zigomaro/telegram-monitor-bot",
        ["zigomaro/cipher", "zigomaro/key"],
        run_id="test-run",
        timeout_seconds=5,
        poll_interval_seconds=1,
        status_callback=_status_callback,
    )

    assert meta["dataset_sources"][-2:] == ["zigomaro/cipher", "zigomaro/key"]
    assert [phase for phase, _ in seen] == ["dataset_bind", "dataset_bind"]


class _DummyBot:
    def __init__(self) -> None:
        self.sent_messages: list[str] = []
        self.edited_messages: list[str] = []
        self.sent_videos: list[tuple[int, str | None]] = []
        self._next_message_id = 1

    async def send_message(self, chat_id: int, text: str, **kwargs):  # noqa: ANN001,ARG002
        self.sent_messages.append(text)
        message_id = self._next_message_id
        self._next_message_id += 1
        return type(
            "SentMessage",
            (),
            {
                "chat": type("Chat", (), {"id": chat_id})(),
                "message_id": message_id,
            },
        )()

    async def edit_message_text(self, text: str, chat_id: int, message_id: int, **kwargs):  # noqa: ANN001,ARG002
        self.edited_messages.append(text)
        return None

    async def send_video(self, chat_id: int, video, caption: str | None = None, **kwargs):  # noqa: ANN001,ARG002
        self.sent_videos.append((chat_id, caption))
        return None


@pytest.mark.asyncio
async def test_video_poller_rejects_complete_if_kernel_binding_was_replaced(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        sess = VideoAnnounceSession(
            status=VideoAnnounceSessionStatus.RENDERING,
            kaggle_kernel_ref="zigomaro/crumple-video",
            kaggle_dataset="zigomaro/video-afisha-session-42",
            started_at=datetime.now(timezone.utc),
        )
        session.add(sess)
        await session.commit()
        await session.refresh(sess)
        session_id = sess.id

    class _FakeClient:
        def get_kernel_status(self, kernel_ref: str) -> dict:
            assert kernel_ref == "zigomaro/crumple-video"
            return {"status": "COMPLETE"}

        def kernel_has_dataset_sources(self, kernel_ref: str, expected_sources: list[str]):
            assert kernel_ref == "zigomaro/crumple-video"
            assert expected_sources == ["zigomaro/video-afisha-session-42"]
            return False, {"dataset_sources": ["zigomaro/video-afisha-session-99"]}

        def delete_dataset(self, dataset: str) -> None:
            assert dataset == "zigomaro/video-afisha-session-42"

    bot = _DummyBot()
    async with db.get_session() as session:
        sess = await session.get(VideoAnnounceSession, session_id)

    assert sess is not None

    await video_poller.run_kernel_poller(
        db,
        _FakeClient(),
        sess,
        bot=bot,
        notify_chat_id=123,
        test_chat_id=None,
        main_chat_id=None,
        poll_interval=1,
        timeout_minutes=1,
        dataset_slug="zigomaro/video-afisha-session-42",
    )

    async with db.get_session() as session:
        fresh = await session.get(VideoAnnounceSession, session_id)

    assert fresh is not None
    assert fresh.status == VideoAnnounceSessionStatus.FAILED
    assert "kernel superseded before output download" in str(fresh.error)
    assert any("kernel больше не привязан" in text for text in bot.sent_messages)


@pytest.mark.asyncio
async def test_tg_and_video_results_are_processed_independently(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    imported_run_ids: list[str] = []

    async def _fake_process_results(results_path, db_obj, bot=None, progress_callback=None):  # noqa: ANN001,ARG002
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        imported_run_ids.append(str(payload.get("run_id")))
        return tg_service.TelegramMonitorReport(
            run_id=str(payload.get("run_id") or ""),
            sources_total=int((payload.get("stats") or {}).get("sources_total") or 0),
        )

    monkeypatch.setattr(tg_service, "process_telegram_results", _fake_process_results)

    async with db.get_session() as session:
        video_session = VideoAnnounceSession(
            status=VideoAnnounceSessionStatus.RENDERING,
            kaggle_kernel_ref="zigomaro/crumple-video",
            kaggle_dataset="zigomaro/video-afisha-session-77",
            started_at=datetime.now(timezone.utc),
        )
        session.add(video_session)
        await session.commit()
        await session.refresh(video_session)
        video_session_id = int(video_session.id)

    class _SharedFakeClient:
        def get_kernel_status(self, kernel_ref: str) -> dict:
            if kernel_ref == "zigomaro/crumple-video":
                return {"status": "COMPLETE"}
            return {"status": "RUNNING"}

        def kernel_has_dataset_sources(self, kernel_ref: str, expected_sources: list[str]):
            if kernel_ref == "zigomaro/crumple-video":
                return True, {"dataset_sources": list(expected_sources)}
            if kernel_ref == "zigomaro/telegram-monitor-bot":
                return True, {"dataset_sources": list(expected_sources)}
            return False, {"dataset_sources": []}

        def download_kernel_output(self, kernel_ref: str, *, path, force=True, quiet=False):  # noqa: ANN001,ARG002
            from pathlib import Path

            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            if kernel_ref == "zigomaro/telegram-monitor-bot":
                payload = {
                    "run_id": "tg-run-123",
                    "generated_at": "2026-03-09T18:10:00+00:00",
                    "stats": {"sources_total": 2},
                    "messages": [],
                    "sources_meta": [],
                }
                target = out / "telegram_results.json"
                target.write_text(json.dumps(payload), encoding="utf-8")
                return ["telegram_results.json"]
            if kernel_ref == "zigomaro/crumple-video":
                target = out / "crumple_video_final.mp4"
                target.write_bytes(b"0" * 1024)
                return ["crumple_video_final.mp4"]
            raise AssertionError(kernel_ref)

        def delete_dataset(self, dataset: str) -> None:  # noqa: ARG002
            return None

    bot = _DummyBot()
    client = _SharedFakeClient()

    async with db.get_session() as session:
        video_session = await session.get(VideoAnnounceSession, video_session_id)
    assert video_session is not None

    async def _tg_flow() -> None:
        results_path = await tg_service._download_results(client, "zigomaro/telegram-monitor-bot", "tg-run-123")  # noqa: SLF001
        report = await tg_service._import_results_with_retry(  # noqa: SLF001
            results_path,
            db,
            bot=bot,
            run_id="tg-run-123",
            progress_callback=None,
        )
        assert report.run_id == "tg-run-123"

    async def _video_flow() -> None:
        await video_poller.run_kernel_poller(
            db,
            client,
            video_session,
            bot=bot,
            notify_chat_id=777,
            test_chat_id=None,
            main_chat_id=None,
            poll_interval=1,
            timeout_minutes=1,
            dataset_slug="zigomaro/video-afisha-session-77",
        )

    await asyncio.gather(_tg_flow(), _video_flow())

    async with db.get_session() as session:
        fresh = await session.get(VideoAnnounceSession, video_session_id)

    assert imported_run_ids == ["tg-run-123"]
    assert fresh is not None
    assert fresh.status == VideoAnnounceSessionStatus.PUBLISHED_TEST
    assert fresh.video_url == "crumple_video_final.mp4"
    assert bot.sent_videos
