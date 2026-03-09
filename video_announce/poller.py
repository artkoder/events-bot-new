from __future__ import annotations

import asyncio
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from aiogram import types
from aiogram.types import FSInputFile
from sqlalchemy import select

from db import Database
from main import format_day_pretty
from models import (
    Event,
    VideoAnnounceEventHit,
    VideoAnnounceItem,
    VideoAnnounceItemStatus,
    VideoAnnounceSession,
    VideoAnnounceSessionStatus,
)
from .kaggle_client import KaggleClient

logger = logging.getLogger(__name__)

_status_messages: dict[int, tuple[int, int]] = {}
_status_locks: dict[int, asyncio.Lock] = {}
_poller_tasks: dict[int, asyncio.Task] = {}


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
VIDEO_KAGGLE_TIMEOUT_MINUTES = _read_positive_int("VIDEO_KAGGLE_TIMEOUT_MINUTES", 150)
VIDEO_DATASET_BIND_WAIT_SECONDS = _read_positive_int("VIDEO_DATASET_BIND_WAIT_SECONDS", 120)
VIDEO_DATASET_BIND_POLL_SECONDS = _read_positive_int("VIDEO_DATASET_BIND_POLL_SECONDS", 10)

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


def _track_poller_task(session_id: int, task: asyncio.Task) -> None:
    _poller_tasks[session_id] = task

    def _cleanup(_task: asyncio.Task) -> None:
        _poller_tasks.pop(session_id, None)

    task.add_done_callback(_cleanup)


def _poller_active(session_id: int) -> bool:
    task = _poller_tasks.get(session_id)
    return bool(task and not task.done())


def start_kernel_poller_task(
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
) -> asyncio.Task:
    if _poller_active(session_obj.id):
        return _poller_tasks[session_obj.id]
    task = asyncio.create_task(
        run_kernel_poller(
            db,
            client,
            session_obj,
            bot=bot,
            notify_chat_id=notify_chat_id,
            test_chat_id=test_chat_id,
            main_chat_id=main_chat_id,
            status_chat_id=status_chat_id,
            status_message_id=status_message_id,
            poll_interval=poll_interval,
            timeout_minutes=timeout_minutes,
            download_dir=download_dir,
            dataset_slug=dataset_slug,
        )
    )
    _track_poller_task(session_obj.id, task)
    return task


def _parse_positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _format_date_range(dates: list[date]) -> str | None:
    if not dates:
        return None
    min_date = min(dates)
    max_date = max(dates)
    if min_date == max_date:
        return format_day_pretty(min_date)
    return f"{format_day_pretty(min_date)} - {format_day_pretty(max_date)}"


def _selection_render_limit(session_obj: VideoAnnounceSession) -> int | None:
    params = (
        session_obj.selection_params
        if isinstance(session_obj.selection_params, dict)
        else {}
    )
    return _parse_positive_int(params.get("render_scene_limit"))


def _resolve_notify_chat_id(session_obj: VideoAnnounceSession) -> int | None:
    params = (
        session_obj.selection_params
        if isinstance(session_obj.selection_params, dict)
        else {}
    )
    raw = params.get("notify_chat_id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _fallback_target_date_label(session_obj: VideoAnnounceSession) -> str | None:
    params = (
        session_obj.selection_params
        if isinstance(session_obj.selection_params, dict)
        else {}
    )
    raw = params.get("target_date")
    if not raw:
        return None
    try:
        parsed = date.fromisoformat(str(raw))
    except ValueError:
        return None
    return format_day_pretty(parsed)


async def _load_session_date_range(
    db: Database, session_obj: VideoAnnounceSession
) -> str | None:
    limit = _selection_render_limit(session_obj)
    async with db.get_session() as session:
        res = await session.execute(
            select(VideoAnnounceItem)
            .where(VideoAnnounceItem.session_id == session_obj.id)
            .where(VideoAnnounceItem.status == VideoAnnounceItemStatus.READY)
            .order_by(VideoAnnounceItem.position)
        )
        items = res.scalars().all()
        if limit:
            items = items[:limit]
        event_ids = [item.event_id for item in items]
        if not event_ids:
            return None
        ev_res = await session.execute(select(Event).where(Event.id.in_(event_ids)))
        events = ev_res.scalars().all()
    dates: list[date] = []
    for ev in events:
        try:
            raw_date = (ev.date or "").split("..", 1)[0]
            dates.append(date.fromisoformat(raw_date))
        except Exception:
            continue
    return _format_date_range(dates)


async def _build_video_caption(
    db: Database, session_obj: VideoAnnounceSession
) -> str:
    label = await _load_session_date_range(db, session_obj)
    if not label:
        label = _fallback_target_date_label(session_obj)
    if not label:
        label = format_day_pretty((datetime.now(timezone.utc) + timedelta(days=1)).date())
    return f"Видео-анонс #{session_obj.id} на завтра {label}"


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


async def await_kernel_dataset_binding(
    client: KaggleClient,
    kernel_ref: str,
    *,
    dataset_slug: str | None,
    session_id: int | None = None,
    timeout_seconds: int = VIDEO_DATASET_BIND_WAIT_SECONDS,
    poll_interval_seconds: int = VIDEO_DATASET_BIND_POLL_SECONDS,
) -> dict:
    expected = [str(dataset_slug).strip()] if str(dataset_slug or "").strip() else []
    if not expected:
        return {}

    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(timeout_seconds)))
    last_meta: dict | None = None
    last_error: str | None = None
    while datetime.now(timezone.utc) < deadline:
        try:
            matched, meta = await asyncio.to_thread(
                client.kernel_has_dataset_sources,
                kernel_ref,
                expected,
            )
            last_meta = meta or {}
            logger.info(
                "video_announce: kernel dataset bind session=%s kernel=%s matched=%s expected=%s actual=%s",
                session_id,
                kernel_ref,
                matched,
                expected,
                list(last_meta.get("dataset_sources") or []),
            )
            if matched:
                return last_meta
            last_error = None
        except Exception as exc:
            last_error = str(exc) or exc.__class__.__name__
            logger.warning(
                "video_announce: kernel dataset bind check failed session=%s kernel=%s err=%s",
                session_id,
                kernel_ref,
                last_error,
            )
        await asyncio.sleep(max(1, int(poll_interval_seconds)))

    actual_sources = list((last_meta or {}).get("dataset_sources") or [])
    details = f"expected={expected}"
    if actual_sources:
        details += f" actual={actual_sources}"
    if last_error:
        details += f" last_error={last_error}"
    raise RuntimeError(f"kernel dataset binding was not confirmed ({details})")


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
    candidates = [
        file
        for file in files
        if file.exists() and file.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
    ]
    if not candidates:
        return None
    preferred = [f for f in candidates if "final" in f.name.lower()]
    if preferred:
        return max(preferred, key=lambda f: f.stat().st_size if f.exists() else 0)
    return max(candidates, key=lambda f: f.stat().st_size if f.exists() else 0)


def _find_logs(files: Iterable[Path]) -> list[Path]:
    return [
        f
        for f in files
        if f.exists() and f.suffix.lower() in {".txt", ".log", ".json"}
    ]


def _expand_output_paths(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        if p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child not in seen:
                    files.append(child)
                    seen.add(child)
        elif p.is_file() and p not in seen:
            files.append(p)
            seen.add(p)
    return files


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
        # files is a list of relative paths as strings
        # Create full paths without flattening directories
        paths = [output_dir / f for f in files]
        
        # Recursively find all log files in the output directory
        log_candidates = []
        for p in paths:
             if p.is_dir():
                 log_candidates.extend(list(p.rglob("*")))
             else:
                 log_candidates.append(p)

        log_files = _find_logs(log_candidates)
        # Deduplicate paths just in case
        log_files = sorted(list(set(log_files)))

        logger.info(
            "video_announce: found %s log files in output: %s",
            len(log_files),
            [f.name for f in log_files],
        )

        MAX_FILES_TO_SEND = 10
        if log_files:
            if len(log_files) <= MAX_FILES_TO_SEND:
                await _send_logs(
                    bot, chat_id, log_files, caption=f"{caption_prefix} сессии #{session_id}"
                )
            else:
                # Too many files, zip them
                zip_path = output_dir / f"logs-{session_id}.zip"
                import zipfile
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for lf in log_files:
                        zipf.write(lf, lf.relative_to(output_dir))
                
                await _send_logs(
                    bot, chat_id, [zip_path], caption=f"{caption_prefix} сессии #{session_id} (архив)"
                )

        else:
            # Send all files if no .log/.txt/.json found
            all_files = []
            for p in output_dir.rglob("*"):
                if p.is_file():
                    all_files.append(p)
            
            logger.info(
                "video_announce: no log files found, sending all %s files",
                len(all_files),
            )
            if all_files:
                if len(all_files) <= MAX_FILES_TO_SEND:
                    await _send_logs(
                        bot, chat_id, all_files, caption=f"{caption_prefix} сессии #{session_id}"
                    )
                else:
                    zip_path = output_dir / f"output-{session_id}.zip"
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for f in all_files:
                            zipf.write(f, f.relative_to(output_dir))
                    await _send_logs(
                        bot, chat_id, [zip_path], caption=f"{caption_prefix} сессии #{session_id} (полный архив)"
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
            if dataset_slug:
                try:
                    await await_kernel_dataset_binding(
                        client,
                        kernel_ref,
                        dataset_slug=dataset_slug,
                        session_id=session_obj.id,
                        timeout_seconds=max(5, poll_interval),
                        poll_interval_seconds=max(2, min(poll_interval, VIDEO_DATASET_BIND_POLL_SECONDS)),
                    )
                except Exception as exc:
                    logger.warning(
                        "video_announce: kernel complete but dataset binding mismatch session=%s kernel=%s err=%s",
                        session_obj.id,
                        kernel_ref,
                        exc,
                    )
                    session_obj = await _update_status(
                        db,
                        session_obj.id,
                        status=VideoAnnounceSessionStatus.FAILED,
                        error=f"kernel superseded before output download: {exc}",
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
                        note="Kernel был перепривязан к другому dataset",
                    )
                    await bot.send_message(
                        notify_chat_id,
                        (
                            f"⚠️ Сессия #{session_obj.id}: kernel больше не привязан к dataset "
                            f"{dataset_slug}. Похоже, другой запуск перезаписал notebook."
                        ),
                    )
                    await _cleanup_dataset(client, dataset_slug)
                    return
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
        max_attempts = 3
        files: list[str] = []
        for attempt in range(1, max_attempts + 1):
            try:
                files = await asyncio.to_thread(
                    client.download_kernel_output,
                    kernel_ref,
                    path=output_dir,
                    force=True,
                    quiet=True,
                )
                break
            except Exception:
                logger.exception(
                    "video_announce: kernel output download failed attempt=%s/%s session=%s",
                    attempt,
                    max_attempts,
                    session_obj.id,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(5 * attempt)
                else:
                    raise
        paths = [output_dir / f for f in files]
        output_files = _expand_output_paths(paths)
        video_path = _find_video(output_files)
        log_files = _find_logs(output_files)
        if not video_path:
            logger.warning(
                "video_announce: no video in output session=%s files=%s",
                session_obj.id,
                files or [p.name for p in output_files],
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
        caption = await _build_video_caption(db, session_obj)
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
    except Exception:
        logger.exception(
            "video_announce: failed to download kernel output session=%s kernel=%s",
            session_obj.id,
            kernel_ref,
        )
        session_obj = await _update_status(
            db,
            session_obj.id,
            status=VideoAnnounceSessionStatus.FAILED,
            error="kernel output download failed",
        )
        if session_obj:
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
            f"⚠️ Сессия #{session_obj.id}: не удалось скачать вывод kernel",
        )
    finally:
        await _cleanup_dataset(client, dataset_slug)


async def resume_rendering_sessions(db: Database, bot, *, chat_id: int | None = None) -> int:
    async with db.get_session() as session:
        res = await session.execute(
            select(VideoAnnounceSession).where(
                VideoAnnounceSession.status == VideoAnnounceSessionStatus.RENDERING
            )
        )
        sessions = res.scalars().all()
    if not sessions:
        return 0
    recovered = 0
    admin_chat_id = None
    if chat_id is None:
        raw_admin = os.getenv("ADMIN_CHAT_ID")
        if raw_admin:
            try:
                admin_chat_id = int(raw_admin)
            except (TypeError, ValueError):
                admin_chat_id = None
    client = KaggleClient()
    for sess in sessions:
        if not sess.kaggle_kernel_ref:
            continue
        if _poller_active(sess.id):
            continue
        notify_chat_id = _resolve_notify_chat_id(sess) or chat_id or admin_chat_id or sess.test_chat_id or sess.main_chat_id
        if not notify_chat_id:
            continue
        start_kernel_poller_task(
            db,
            client,
            sess,
            bot=bot,
            notify_chat_id=notify_chat_id,
            test_chat_id=sess.test_chat_id,
            main_chat_id=sess.main_chat_id,
            poll_interval=60,
            timeout_minutes=VIDEO_KAGGLE_TIMEOUT_MINUTES,
            dataset_slug=sess.kaggle_dataset,
        )
        recovered += 1
    return recovered


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
