from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from aiogram import types
from aiogram.types import FSInputFile
from sqlalchemy import select

from db import Database
from models import (
    Event,
    VideoAnnounceEventHit,
    VideoAnnounceItem,
    VideoAnnounceSession,
    VideoAnnounceSessionStatus,
)
from .kaggle_client import KaggleClient

logger = logging.getLogger(__name__)

_status_messages: dict[int, tuple[int, int]] = {}
_status_locks: dict[int, asyncio.Lock] = {}


def _read_positive_int(env_key: str, default: int) -> int:
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
        if value <= 0:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            "video_announce: invalid %s=%r, falling back to default %s",
            env_key,
            raw_value,
            default,
        )
        return default


VIDEO_MAX_MB = _read_positive_int("VIDEO_MAX_MB", 50)
VIDEO_KAGGLE_TIMEOUT_MINUTES = _read_positive_int("VIDEO_KAGGLE_TIMEOUT_MINUTES", 40)

logger.info(
    "video_announce: limits configured max_video_mb=%s kaggle_timeout_min=%s",
    VIDEO_MAX_MB,
    VIDEO_KAGGLE_TIMEOUT_MINUTES,
)


def _status_keyboard(session_id: int) -> types.InlineKeyboardMarkup:
    return types.InlineKeyboardMarkup(
        inline_keyboard=
        [[types.InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"vidkstat:{session_id}")]]
    )


def remember_status_message(session_id: int, chat_id: int, message_id: int) -> None:
    _status_messages[session_id] = (chat_id, message_id)


def get_status_message(session_id: int) -> tuple[int, int] | None:
    return _status_messages.get(session_id)


def _get_status_lock(session_id: int) -> asyncio.Lock:
    lock = _status_locks.get(session_id)
    if not lock:
        lock = asyncio.Lock()
        _status_locks[session_id] = lock
    return lock


def _format_kaggle_status(status: dict | None) -> str:
    if not status:
        return "неизвестен"
    state = status.get("status")
    failure_msg = status.get("failureMessage") or status.get("failure_message")
    if not state:
        return "неизвестен"
    result = str(state)
    if failure_msg:
        result += f" ({failure_msg})"
    return result


def _status_text(
    session_obj: VideoAnnounceSession,
    kaggle_status: dict | None,
    *,
    note: str | None = None,
) -> str:
    lines = [
        f"Сессия #{session_obj.id}: {session_obj.status}",
        f"Kernel: {session_obj.kaggle_kernel_ref or '—'}",
        f"Dataset: {session_obj.kaggle_dataset or '—'}",
        f"Статус Kaggle: {_format_kaggle_status(kaggle_status)}",
    ]
    if session_obj.video_url:
        lines.append(f"Видео: {session_obj.video_url}")
    if session_obj.error:
        lines.append(f"Ошибка: {session_obj.error}")
    if note:
        lines.append(note)
    return "\n".join(lines)


async def update_status_message(
    bot,
    session_obj: VideoAnnounceSession,
    kaggle_status: dict | None,
    *,
    chat_id: int | None = None,
    message_id: int | None = None,
    allow_send: bool = False,
    note: str | None = None,
) -> tuple[int, int] | None:
    text = _status_text(session_obj, kaggle_status, note=note)
    markup = _status_keyboard(session_obj.id)
    lock = _get_status_lock(session_obj.id)
    async with lock:
        stored = get_status_message(session_obj.id)
        if stored and (chat_id is None or message_id is None):
            chat_id, message_id = stored
        if message_id is None and not allow_send:
            return stored
        try:
            if message_id is None and chat_id is not None:
                sent = await bot.send_message(chat_id, text, reply_markup=markup)
                remember_status_message(session_obj.id, sent.chat.id, sent.message_id)
                return (sent.chat.id, sent.message_id)
            if chat_id is not None and message_id is not None:
                await bot.edit_message_text(
                    text=text, chat_id=chat_id, message_id=message_id, reply_markup=markup
                )
                remember_status_message(session_obj.id, chat_id, message_id)
                return (chat_id, message_id)
        except Exception:
            logger.exception(
                "video_announce: failed to update status message session_id=%s",
                session_obj.id,
            )
        return stored


def _find_video(files: Iterable[Path]) -> Path | None:
    for file in files:
        if file.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            return file
    return None


def _find_logs(files: Iterable[Path]) -> list[Path]:
    return [f for f in files if f.suffix.lower() in {".txt", ".log", ".json"}]


async def _update_status(
    db: Database,
    session_id: int,
    *,
    status: VideoAnnounceSessionStatus,
    error: str | None = None,
    video_url: str | None = None,
) -> VideoAnnounceSession | None:
    async with db.get_session() as session:
        obj = await session.get(VideoAnnounceSession, session_id)
        if not obj:
            return None
        obj.status = status
        if status in {VideoAnnounceSessionStatus.DONE, VideoAnnounceSessionStatus.FAILED}:
            obj.finished_at = datetime.now(timezone.utc)
        if video_url:
            obj.video_url = video_url
        obj.error = error
        await session.commit()
        await session.refresh(obj)
        return obj


async def _mark_published_main(db: Database, session_obj: VideoAnnounceSession) -> None:
    async with db.get_session() as session:
        fresh = await session.get(VideoAnnounceSession, session_obj.id)
        if not fresh:
            return
        fresh.status = VideoAnnounceSessionStatus.PUBLISHED_MAIN
        fresh.published_at = datetime.now(timezone.utc)
        res = await session.execute(
            select(VideoAnnounceItem).where(VideoAnnounceItem.session_id == session_obj.id)
        )
        items = res.scalars().all()
        event_ids = [it.event_id for it in items]
        if event_ids:
            ev_res = await session.execute(select(Event).where(Event.id.in_(event_ids)))
            events = ev_res.scalars().all()
        else:
            events = []
        existing_hits: set[int] = set()
        if event_ids:
            hit_res = await session.execute(
                select(VideoAnnounceEventHit.event_id).where(
                    VideoAnnounceEventHit.session_id == session_obj.id,
                    VideoAnnounceEventHit.event_id.in_(event_ids),
                )
            )
            existing_hits = set(hit_res.scalars().all())
        for ev in events:
            ev.video_include_count = max(0, (ev.video_include_count or 0) - 1)
            if ev.id not in existing_hits:
                session.add(
                    VideoAnnounceEventHit(session_id=session_obj.id, event_id=ev.id)
                )
        session.add(fresh)
        await session.commit()


async def _send_logs(bot, chat_id: int, files: list[Path], *, caption: str | None = None) -> None:
    for file in files:
        try:
            input_file = FSInputFile(file)
            await bot.send_document(
                chat_id, input_file, caption=caption, disable_notification=True
            )
        except Exception:
            logger.exception("video_announce: failed to send log %s", file)


async def _download_and_send_logs(
    client: KaggleClient,
    kernel_ref: str,
    bot,
    chat_id: int,
    session_id: int,
    *,
    download_dir: Path | None = None,
    caption_prefix: str = "Логи Kaggle",
) -> None:
    """Download kernel output and send any log files to the chat."""
    tmp_dir = download_dir or Path(os.getenv("TMPDIR", "/tmp"))
    output_dir = tmp_dir / f"videoannounce-logs-{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info(
            "video_announce: downloading kernel output for logs kernel=%s session=%s",
            kernel_ref,
            session_id,
        )
        files = await asyncio.to_thread(
            client.download_kernel_output,
            kernel_ref,
            path=output_dir,
            force=True,
            quiet=True,
        )
        paths = [output_dir / Path(f).name for f in files]
        log_files = _find_logs(paths)
        logger.info(
            "video_announce: found %s log files in output: %s",
            len(log_files),
            [f.name for f in log_files],
        )
        if log_files:
            await _send_logs(
                bot, chat_id, log_files, caption=f"{caption_prefix} сессии #{session_id}"
            )
        else:
            # Send all files if no .log/.txt/.json found
            all_files = list(output_dir.iterdir())
            logger.info(
                "video_announce: no log files found, sending all %s files",
                len(all_files),
            )
            if all_files:
                await _send_logs(
                    bot, chat_id, all_files, caption=f"{caption_prefix} сессии #{session_id}"
                )
            else:
                await bot.send_message(
                    chat_id, f"⚠️ Логи Kaggle для сессии #{session_id} не найдены"
                )
    except Exception:
        logger.exception(
            "video_announce: failed to download kernel output for logs session=%s",
            session_id,
        )
        await bot.send_message(
            chat_id, f"⚠️ Не удалось скачать логи Kaggle для сессии #{session_id}"
        )


async def _cleanup_dataset(client: KaggleClient, dataset_slug: str | None) -> None:
    """Delete the temporary Kaggle dataset after kernel completion."""
    if not dataset_slug:
        return
    try:
        logger.info("video_announce: deleting dataset %s", dataset_slug)
        await asyncio.to_thread(client.delete_dataset, dataset_slug)
        logger.info("video_announce: dataset %s deleted successfully", dataset_slug)
    except Exception:
        logger.exception("video_announce: failed to delete dataset %s", dataset_slug)

async def run_kernel_poller(
    db: Database,
    client: KaggleClient,
    session_obj: VideoAnnounceSession,
    *,
    bot,
    notify_chat_id: int,
    test_chat_id: int | None,
    main_chat_id: int | None,
    status_chat_id: int | None = None,
    status_message_id: int | None = None,
    poll_interval: int = 60,
    timeout_minutes: int = VIDEO_KAGGLE_TIMEOUT_MINUTES,
    download_dir: Path | None = None,
    dataset_slug: str | None = None,
) -> None:
    deadline = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
    kernel_ref = session_obj.kaggle_kernel_ref
    if not kernel_ref:
        await _update_status(
            db,
            session_obj.id,
            status=VideoAnnounceSessionStatus.FAILED,
            error="kernel reference missing",
        )
        await bot.send_message(notify_chat_id, "Не указан kernel для сессии")
        return
    status_message = await update_status_message(
        bot,
        session_obj,
        {},
        chat_id=status_chat_id,
        message_id=status_message_id,
        allow_send=True,
        note="Старт отслеживания Kaggle",
    )
    if status_message:
        status_chat_id, status_message_id = status_message
    
    # Track consecutive unknown statuses
    unknown_status_count = 0
    # Kaggle kernel can take a while to start, API returns None during startup
    # At ~1 min poll interval, 30 attempts = ~30 minutes before failing
    MAX_UNKNOWN_STATUS_COUNT = 30
    
    while datetime.now(timezone.utc) < deadline:
        try:
            status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            logger.info(
                "video_announce: kernel status poll session=%s kernel=%s status=%s",
                session_obj.id,
                kernel_ref,
                status.get("status"),
            )
        except Exception:
            logger.exception("video_announce: kernel status failed session=%s", session_obj.id)
            status = {}
        await update_status_message(
            bot,
            session_obj,
            status,
            chat_id=status_chat_id,
            message_id=status_message_id,
            allow_send=True,
        )
        state = str(status.get("status") or "").lower()
        
        # Handle unknown/empty status
        if not state or state in {"none", "unknown"}:
            unknown_status_count += 1
            logger.warning(
                "video_announce: unknown kernel status session=%s count=%s/%s full_response=%s",
                session_obj.id,
                unknown_status_count,
                MAX_UNKNOWN_STATUS_COUNT,
                status,
            )
            if unknown_status_count >= MAX_UNKNOWN_STATUS_COUNT:
                error_msg = f"Kaggle API returns unknown status after {MAX_UNKNOWN_STATUS_COUNT} attempts"
                session_obj = await _update_status(
                    db,
                    session_obj.id,
                    status=VideoAnnounceSessionStatus.FAILED,
                    error=error_msg,
                )
                if not session_obj:
                    return
                await update_status_message(
                    bot,
                    session_obj,
                    status,
                    chat_id=status_chat_id,
                    message_id=status_message_id,
                    allow_send=True,
                )
                await bot.send_message(
                    notify_chat_id,
                    f"⚠️ Сессия #{session_obj.id}: Kaggle API не возвращает статус.\n"
                    "Проверьте ноутбук вручную на kaggle.com",
                )
                await _download_and_send_logs(
                    client,
                    kernel_ref,
                    bot,
                    notify_chat_id,
                    session_obj.id,
                    download_dir=download_dir,
                    caption_prefix="⚠️ Логи (неизвестный статус)",
                )
                await _cleanup_dataset(client, dataset_slug)
                return
            await asyncio.sleep(poll_interval)
            continue
        else:
            # Reset counter if we get a valid status
            unknown_status_count = 0
        
        if state == "complete":
            break
        if state in {"error", "failed"}:
            failure_msg = status.get("failureMessage") or status.get("failure_message") or ""
            error_detail = f"{state}: {failure_msg}" if failure_msg else str(status)
            logger.warning(
                "video_announce: kernel failed session=%s kernel=%s error=%s",
                session_obj.id,
                kernel_ref,
                error_detail,
            )
            session_obj = await _update_status(
                db,
                session_obj.id,
                status=VideoAnnounceSessionStatus.FAILED,
                error=error_detail,
            )
            if not session_obj:
                return
            await update_status_message(
                bot,
                session_obj,
                status,
                chat_id=status_chat_id,
                message_id=status_message_id,
                allow_send=True,
            )
            await bot.send_message(
                notify_chat_id, f"❌ Сессия #{session_obj.id} завершилась ошибкой Kaggle: {state}"
            )
            # Download and send logs on failure
            await _download_and_send_logs(
                client,
                kernel_ref,
                bot,
                notify_chat_id,
                session_obj.id,
                download_dir=download_dir,
                caption_prefix="❌ Логи ошибки Kaggle",
            )
            await _cleanup_dataset(client, dataset_slug)
            return
        await asyncio.sleep(poll_interval)
    else:
        logger.warning(
            "video_announce: kernel timeout session=%s kernel=%s timeout_min=%s",
            session_obj.id,
            kernel_ref,
            timeout_minutes,
        )
        session_obj = await _update_status(
            db,
            session_obj.id,
            status=VideoAnnounceSessionStatus.FAILED,
            error=f"timeout after {timeout_minutes}min",
        )
        if not session_obj:
            return
        await update_status_message(
            bot,
            session_obj,
            status,
            chat_id=status_chat_id,
            message_id=status_message_id,
            allow_send=True,
        )
        await bot.send_message(
            notify_chat_id,
            f"⏱️ Сессия #{session_obj.id} не завершилась за {timeout_minutes} минут",
        )
        # Download and send logs on timeout
        await _download_and_send_logs(
            client,
            kernel_ref,
            bot,
            notify_chat_id,
            session_obj.id,
            download_dir=download_dir,
            caption_prefix="⏱️ Логи (таймаут) Kaggle",
        )
        await _cleanup_dataset(client, dataset_slug)
        return

    tmp_dir = download_dir or Path(os.getenv("TMPDIR", "/tmp"))
    output_dir = tmp_dir / f"videoannounce-{session_obj.id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = await asyncio.to_thread(
            client.download_kernel_output,
            kernel_ref,
            path=output_dir,
            force=True,
            quiet=True,
        )
        paths = [output_dir / Path(f).name for f in files]
        video_path = _find_video(paths)
        log_files = _find_logs(paths)
        if not video_path:
            logger.warning(
                "video_announce: no video in output session=%s files=%s",
                session_obj.id,
                [p.name for p in paths],
            )
            session_obj = await _update_status(
                db,
                session_obj.id,
                status=VideoAnnounceSessionStatus.FAILED,
                error="missing video output",
            )
            if not session_obj:
                return
            await update_status_message(
                bot,
                session_obj,
                status,
                chat_id=status_chat_id,
                message_id=status_message_id,
                allow_send=True,
            )
            await bot.send_message(notify_chat_id, "❌ Видео не найдено в выводе kernel")
            # Send logs even when video is missing
            if log_files:
                await _send_logs(
                    bot,
                    notify_chat_id,
                    log_files,
                    caption=f"❌ Логи (нет видео) сессии #{session_obj.id}",
                )
            return
        if video_path.stat().st_size > VIDEO_MAX_MB * 1024 * 1024:
            session_obj = await _update_status(
                db,
                session_obj.id,
                status=VideoAnnounceSessionStatus.FAILED,
                error=f"video exceeds {VIDEO_MAX_MB}MB",
            )
            if not session_obj:
                return
            await update_status_message(
                bot,
                session_obj,
                status,
                chat_id=status_chat_id,
                message_id=status_message_id,
                allow_send=True,
            )
            await bot.send_message(
                notify_chat_id,
                f"Видео из сессии #{session_obj.id} превышает {VIDEO_MAX_MB} MB",
            )
            return
        session_obj = await _update_status(
            db,
            session_obj.id,
            status=VideoAnnounceSessionStatus.DONE,
            video_url=video_path.name,
        )
        if not session_obj:
            return
        await update_status_message(
            bot,
            session_obj,
            status,
            chat_id=status_chat_id,
            message_id=status_message_id,
            allow_send=True,
        )
        caption = f"Видео-анонс #{session_obj.id}"
        target_test = test_chat_id or notify_chat_id
        video_input = FSInputFile(video_path)
        try:
            await bot.send_video(target_test, video_input, caption=caption)
        except Exception as e:
            logger.warning("video_announce: failed to send video to test chat %s: %s", target_test, e)
            # Fallback to notify_chat_id if test_chat_id fails
            if target_test != notify_chat_id:
                video_input = FSInputFile(video_path)
                await bot.send_video(notify_chat_id, video_input, caption=caption)
        await _send_logs(bot, notify_chat_id, log_files, caption=f"✅ Логи сессии #{session_obj.id}")
        session_obj = await _update_status(
            db,
            session_obj.id,
            status=VideoAnnounceSessionStatus.PUBLISHED_TEST,
        )
        if session_obj:
            await update_status_message(
                bot,
                session_obj,
                status,
                chat_id=status_chat_id,
                message_id=status_message_id,
                allow_send=True,
                note="Отправлено в тестовый канал",
            )
        if main_chat_id:
            try:
                video_input_main = FSInputFile(video_path)
                await bot.send_video(main_chat_id, video_input_main, caption=caption)
                await _mark_published_main(db, session_obj)
                async with db.get_session() as session:
                    refreshed = await session.get(VideoAnnounceSession, session_obj.id)
                if refreshed:
                    await update_status_message(
                        bot,
                        refreshed,
                        status,
                        chat_id=status_chat_id,
                        message_id=status_message_id,
                        allow_send=True,
                        note="Опубликовано в основном канале",
                    )
            except Exception as e:
                logger.warning("video_announce: failed to send video to main chat %s: %s", main_chat_id, e)
    finally:
        await _cleanup_dataset(client, dataset_slug)


async def reset_stuck_sessions(db: Database, *, max_age_minutes: int = 30) -> int:
    """Move long-running RENDERING sessions into FAILED state."""

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)
    updated = 0
    async with db.get_session() as session:
        res = await session.execute(
            select(VideoAnnounceSession).where(
                VideoAnnounceSession.status == VideoAnnounceSessionStatus.RENDERING,
                VideoAnnounceSession.started_at < cutoff,
            )
        )
        for sess in res.scalars():
            sess.status = VideoAnnounceSessionStatus.FAILED
            sess.finished_at = datetime.now(timezone.utc)
            sess.error = "stuck rendering watchdog"
            updated += 1
            logger.warning("video_announce reset stuck session_id=%s", sess.id)
        if updated:
            await session.commit()
    return updated
