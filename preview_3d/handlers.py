"""Handlers for 3D preview generation command /3di."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from aiogram import types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from db import Database
from models import Event, MonthPage, User
from sqlmodel import select
from video_announce.kaggle_client import KaggleClient, KERNELS_ROOT_PATH
from kaggle_registry import register_job, remove_job, list_jobs

logger = logging.getLogger(__name__)

# Constants
MONTHS_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}
MAX_IMAGES_PER_EVENT = 7
KAGGLE_KERNEL_FOLDER = "Preview3D"
KAGGLE_DATASET_SLUG_PREFIX = "preview3d"
KAGGLE_POLL_INTERVAL_SECONDS = 20
KAGGLE_TIMEOUT_SECONDS = 4 * 60 * 60  # 4 hours for CPU fallback scenarios
KAGGLE_STARTUP_WAIT_SECONDS = 10
KAGGLE_DATASET_WAIT_SECONDS = 15

# Store active sessions (in production, use DB)
_active_sessions: dict[int, dict] = {}
_preview3d_lock = asyncio.Lock()

STATUS_LABELS = {
    "preparing": "подготовка",
    "queued": "в очереди",
    "dataset": "подготовка датасета",
    "dataset_wait": "Ожидание датасета...",
    "kernel_push": "запуск ядра Kaggle",
    "pushing": "запуск ядра Kaggle",
    "rendering": "рендеринг",
    "running": "рендеринг (Kaggle)",
    "complete": "завершено",
    "failed": "ошибка",
    "timeout": "таймаут",
    "download": "скачивание результатов",
    "apply_results": "применение результатов",
    "done": "готово",
    "error": "ошибка",
}


async def _is_authorized(db: Database, user_id: int) -> bool:
    """Check if user is superadmin."""
    async with db.get_session() as session:
        user = await session.get(User, user_id)
        return user is not None and user.is_superadmin


async def _get_events_for_month(db: Database, month: str, min_images: int = 1) -> list[Event]:
    """Get all events for a month that have images."""
    start = date.fromisoformat(f"{month}-01")
    next_start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.date >= start.isoformat(),
                Event.date < next_start.isoformat()
            )
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()
    
    # Filter events that have images
    return [e for e in events if e.photo_urls and len(e.photo_urls) >= min_images]


async def _get_events_without_preview(db: Database, month: str, min_images: int = 1) -> list[Event]:
    """Get events that don't have a 3D preview yet."""
    events = await _get_events_for_month(db, month, min_images=min_images)
    return [e for e in events if not e.preview_3d_url]


async def _get_all_future_events_without_preview(db: Database, min_images: int = 1) -> list[Event]:
    """Get ALL future events (date >= today) that don't have a 3D preview.
    
    Searches across all months, not limited to a single month.
    """
    today = datetime.now(timezone.utc).date()
    today_str = today.isoformat()
    
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.date >= today_str,
                (Event.preview_3d_url.is_(None)) | (Event.preview_3d_url == "")
            )
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()
    
    # Filter events that have enough images
    return [e for e in events if e.photo_urls and len(e.photo_urls) >= min_images]


async def _get_new_events_gap(db: Database, min_images: int = 1) -> list[Event]:
    """Get recent events that are missing a 3D preview.
    
    Checks events from the last 14 days.
    Does NOT stop at the first event with a preview, ensuring gaps are filled.
    """
    candidates: list[Event] = []
    # Look back 14 days to ensure we cover recent additions
    # (using current time which is fine for relative check)
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
    
    async with db.get_session() as session:
        # Fetch events from last 14 days, ordered by ID desc (newest first)
        query = (
            select(Event)
            .where(Event.date >= cutoff_date)
            .order_by(Event.id.desc())
        )
        
        result = await session.execute(query)
        events = result.scalars().all()
        
        for event in events:
            # If already has preview, skip
            if event.preview_3d_url:
                continue
            
            # Check image requirement
            urls = event.photo_urls or []
            if len(urls) >= min_images:
                candidates.append(event)
                
    return candidates


async def run_3di_new_only_scheduler(
    db: Database,
    bot,
    *,
    chat_id: int | None = None,
    min_images: int = 1,
    run_id: str | None = None,
) -> int:
    """Run 3D preview generation for new events without UI callbacks."""
    try:
        if run_id:
            logger.info("3di_scheduler: started run_id=%s", run_id)
        events = await _get_new_events_gap(db, min_images=min_images)
        if not events:
            logger.info("3di_scheduler: no new events to process")
            return 0

        session_id = int(datetime.now(timezone.utc).timestamp() * 1000)
        session = {
            "status": "preparing",
            "month": "New Events Gap",
            "mode": "new_only",
            "event_count": len(events),
            "created_at": datetime.now(timezone.utc),
            "chat_id": chat_id,
            "message_id": None,
        }
        _active_sessions[session_id] = session

        message_id = None
        if bot and chat_id:
            try:
                sent = await bot.send_message(
                    chat_id,
                    _format_status_text(session),
                    reply_markup=_build_status_keyboard(session_id),
                    parse_mode="HTML",
                )
                message_id = sent.message_id
            except Exception:
                logger.warning("3di_scheduler: failed to post status message", exc_info=True)
        session["message_id"] = message_id

        payload = {
            "events": [
                {
                    "event_id": e.id,
                    "title": e.title,
                    "images": (e.photo_urls or [])[:MAX_IMAGES_PER_EVENT],
                }
                for e in events
            ]
        }

        await _run_kaggle_render(
            db=db,
            bot=bot,
            chat_id=chat_id or 0,
            session_id=session_id,
            payload=payload,
            month="New Events Gap",
        )
        return len(events)
    except Exception:
        logger.exception("3di_scheduler failed")
        return 0


_preview3d_recovery_active: set[str] = set()


async def resume_preview3d_jobs(
    db: Database,
    bot,
    *,
    chat_id: int | None = None,
) -> int:
    jobs = await list_jobs("preview3d")
    if not jobs:
        return 0
    notify_chat_id = chat_id
    if notify_chat_id is None:
        admin_chat_id = os.getenv("ADMIN_CHAT_ID")
        if admin_chat_id:
            try:
                notify_chat_id = int(admin_chat_id)
            except ValueError:
                notify_chat_id = None
    client = KaggleClient()
    recovered = 0
    for job in jobs:
        kernel_ref = str(job.get("kernel_ref") or "")
        if not kernel_ref or kernel_ref in _preview3d_recovery_active:
            continue
        _preview3d_recovery_active.add(kernel_ref)
        try:
            try:
                status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
            except Exception:
                logger.exception("3di_recovery: status fetch failed kernel=%s", kernel_ref)
                continue
            state = str(status.get("status") or "").lower()
            meta = job.get("meta") if isinstance(job.get("meta"), dict) else {}
            owner_pid = meta.get("pid")
            if owner_pid == os.getpid():
                continue
            if state in {"error", "failed", "cancelled"}:
                await remove_job("preview3d", kernel_ref)
                if notify_chat_id and bot:
                    await bot.send_message(
                        notify_chat_id,
                        f"⚠️ 3di recovery: kernel {kernel_ref} завершился ошибкой",
                    )
                continue
            if state != "complete":
                continue
            raw_session_id = meta.get("session_id")
            try:
                session_id = int(raw_session_id)
            except (TypeError, ValueError):
                session_id = int(time.time() * 1000)
            try:
                results = await _download_kaggle_results(client, kernel_ref, session_id)
                updated, errors, skipped = await update_previews_from_results(db, results)
            except Exception:
                logger.exception("3di_recovery: failed to apply results kernel=%s", kernel_ref)
                continue
            updated_event_ids: list[int] = []
            seen_event_ids: set[int] = set()
            for result in results:
                status = (result.get("status") or "").lower()
                if status != "ok" or not result.get("preview_url"):
                    continue
                raw_event_id = result.get("event_id")
                try:
                    event_id = int(raw_event_id)
                except (TypeError, ValueError):
                    continue
                if event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)
                updated_event_ids.append(event_id)
            if updated_event_ids:
                try:
                    from main import schedule_event_update_tasks as schedule_tasks
                except Exception:
                    try:
                        from main_part2 import schedule_event_update_tasks as schedule_tasks
                    except Exception:
                        logger.exception("3di_recovery: schedule_event_update_tasks import failed")
                        schedule_tasks = None
                if schedule_tasks:
                    async with db.get_session() as db_session:
                        result = await db_session.execute(
                            select(Event).where(Event.id.in_(updated_event_ids))
                        )
                        updated_events = result.scalars().all()
                    for event in updated_events:
                        await schedule_tasks(db, event, skip_vk_sync=True)
            await remove_job("preview3d", kernel_ref)
            recovered += 1
            if notify_chat_id and bot:
                await bot.send_message(
                    notify_chat_id,
                    (
                        f"✅ 3di recovery: kernel {kernel_ref} обработан. "
                        f"updated={updated}, errors={errors}, skipped={skipped}"
                    ),
                )
        finally:
            _preview3d_recovery_active.discard(kernel_ref)
    return recovered


def _build_main_menu(is_multy: bool = False) -> InlineKeyboardMarkup:
    """Build main menu for /3di command."""
    suffix = ":multy" if is_multy else ""
    buttons = [
        [InlineKeyboardButton(text="🆕 Только новые", callback_data=f"3di:new_only{suffix}")],
        [InlineKeyboardButton(text="🌐 All missing", callback_data=f"3di:all_missing{suffix}")],
        [InlineKeyboardButton(text="⚡️ Сгенерировать (текущий мес)", callback_data=f"3di:new{suffix}")],
        [InlineKeyboardButton(text="🔄 Перегенерировать все", callback_data=f"3di:all{suffix}")],
        [InlineKeyboardButton(text="📅 Выбрать месяц", callback_data=f"3di:month_select{suffix}")],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _build_month_menu(is_multy: bool = False) -> InlineKeyboardMarkup:
    """Build month selection menu."""
    today = datetime.now(timezone.utc).date()
    suffix = ":multy" if is_multy else ""
    buttons = []
    
    for i in range(6):  # Show 6 months
        month_date = (today.replace(day=1) + timedelta(days=32*i)).replace(day=1)
        month_key = month_date.strftime("%Y-%m")
        month_name = MONTHS_RU[month_date.month]
        year = month_date.year
        buttons.append([
            InlineKeyboardButton(
                text=f"{month_name} {year}",
                callback_data=f"3di:gen:{month_key}{suffix}"
            )
        ])
    
    back_suffix = ":multy" if is_multy else ""
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=f"3di:back{back_suffix}")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _format_month_name(month: str) -> str:
    try:
        month_number = int(month.split("-")[1])
    except (AttributeError, IndexError, ValueError, TypeError):
        return month
    return MONTHS_RU.get(month_number, month)


def _format_status_text(session: dict) -> str:
    month_name = _format_month_name(session.get("month", ""))
    event_count = session.get("event_count", 0)
    status = session.get("status", "unknown")
    status_label = STATUS_LABELS.get(status, status)
    mode = session.get("mode", "")
    return (
        f"🎨 <b>3D-превью: {month_name}</b>\n\n"
        f"📊 Событий к обработке: {event_count}\n"
        f"🔄 Статус: {status_label}\n\n"
        f"Режим: {mode}"
    )


def _build_status_keyboard(session_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"3di:status:{session_id}")],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
    ])


async def _update_status_message(bot, chat_id: int, message_id: int, session_id: int) -> None:
    session = _active_sessions.get(session_id)
    if not session or not message_id:
        return
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=_format_status_text(session),
            reply_markup=_build_status_keyboard(session_id),
            parse_mode="HTML",
        )
    except Exception:
        logger.debug("3di: status message update skipped", exc_info=True)


async def _set_session_status(
    session_id: int,
    status: str,
    bot=None,
    chat_id: int | None = None,
    message_id: int | None = None,
) -> None:
    session = _active_sessions.get(session_id)
    if not session:
        return
    if session.get("status") == status:
        return
    session["status"] = status
    if bot and chat_id and message_id:
        await _update_status_message(bot, chat_id, message_id, session_id)


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


async def _create_preview3d_dataset(payload: dict, session_id: int) -> str:
    username = _require_kaggle_username()
    dataset_slug = f"{KAGGLE_DATASET_SLUG_PREFIX}-{session_id}"
    dataset_id = f"{username}/{dataset_slug}"
    meta = {
        "title": f"Preview 3D Payload {session_id}",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "dataset-metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (tmp_path / "payload.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        client = KaggleClient()
        try:
            await asyncio.to_thread(client.create_dataset, tmp_path)
        except Exception:
            logger.exception("3di: dataset create failed, retry after delete")
            await asyncio.to_thread(client.delete_dataset, dataset_id, no_confirm=True)
            await asyncio.to_thread(client.create_dataset, tmp_path)
    return dataset_id


async def _push_preview3d_kernel(client: KaggleClient, dataset_id: str) -> str:
    kernel_path = KERNELS_ROOT_PATH / KAGGLE_KERNEL_FOLDER
    if not kernel_path.exists():
        raise FileNotFoundError(f"Kernel folder not found: {kernel_path}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in kernel_path.iterdir():
            dest = tmp_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        meta_path = tmp_path / "kernel-metadata.json"
        meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
        slug = meta_data.get("slug") or "preview-3d"
        username = os.getenv("KAGGLE_USERNAME")
        if username:
            meta_data["id"] = f"{username}/{slug}"
            meta_data["slug"] = slug
            meta_data["title"] = meta_data.get("title") or "Preview 3D"
        meta_data["dataset_sources"] = [dataset_id]
        meta_data["enable_internet"] = True
        meta_path.write_text(
            json.dumps(meta_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        api = client._get_api()
        await asyncio.to_thread(api.kernels_push, str(tmp_path))
        kernel_ref = str(meta_data.get("id") or meta_data.get("slug") or slug)
    await asyncio.sleep(KAGGLE_STARTUP_WAIT_SECONDS)
    return kernel_ref


async def _poll_kaggle_kernel(
    client: KaggleClient,
    kernel_ref: str,
    session_id: int,
    bot,
    chat_id: int,
    message_id: int | None,
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    deadline = started + KAGGLE_TIMEOUT_SECONDS
    last_status: dict | None = None
    last_status_name = None
    while time.monotonic() < deadline:
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status
        status_name = (status.get("status") or "").upper()
        if status_name != last_status_name:
            last_status_name = status_name
            await _set_session_status(
                session_id,
                status_name.lower() or "running",
                bot=bot,
                chat_id=chat_id,
                message_id=message_id,
            )
        if status_name == "COMPLETE":
            return "complete", last_status, time.monotonic() - started
        if status_name in ("ERROR", "FAILED", "CANCELLED"):
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(KAGGLE_POLL_INTERVAL_SECONDS)
    return "timeout", last_status, time.monotonic() - started


async def _download_kaggle_results(
    client: KaggleClient,
    kernel_ref: str,
    session_id: int,
) -> list[dict]:
    output_dir = Path(tempfile.gettempdir()) / f"preview3d-{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        files = await asyncio.to_thread(
            client.download_kernel_output,
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        output_path = None
        for name in files:
            if Path(name).name == "output.json":
                output_path = output_dir / name
                break
        if output_path and output_path.exists():
            output_data = json.loads(output_path.read_text(encoding="utf-8"))
            results = output_data.get("results")
            if not isinstance(results, list):
                raise RuntimeError("Invalid output.json format: missing results")
            return results
        if files:
            logger.warning(
                "output.json not found in Kaggle output (attempt %s/%s). Files: %s",
                attempt,
                max_attempts,
                sorted(files),
            )
        else:
            logger.warning(
                "output.json not found in Kaggle output (attempt %s/%s). No files returned.",
                attempt,
                max_attempts,
            )
        if attempt < max_attempts:
            await asyncio.sleep(5)
    raise RuntimeError(
        f"output.json not found in Kaggle output after {max_attempts} attempts"
    )


async def _run_kaggle_render(
    db: Database,
    bot,
    chat_id: int,
    session_id: int,
    payload: dict,
    month: str,
) -> None:
    session = _active_sessions.get(session_id)
    if not session:
        return
    message_id = session.get("message_id")
    kernel_ref = ""
    registered = False
    try:
        year = month.split("-")[0]
        month_number = int(month.split("-")[1])
    except (AttributeError, IndexError, ValueError, TypeError):
        month_label = month
    else:
        month_label = f"{MONTHS_RU.get(month_number, month)} {year}"
    event_count = session.get("event_count", len(payload.get("events", [])))
    try:
        if _preview3d_lock.locked():
            await _set_session_status(
                session_id, "queued", bot=bot, chat_id=chat_id, message_id=message_id
            )
        async with _preview3d_lock:
            await _set_session_status(
                session_id, "dataset", bot=bot, chat_id=chat_id, message_id=message_id
            )
            dataset_id = await _create_preview3d_dataset(payload, session_id)
            session["kaggle_dataset"] = dataset_id
            await _set_session_status(
                session_id, "dataset_wait", bot=bot, chat_id=chat_id, message_id=message_id
            )
            await asyncio.sleep(KAGGLE_DATASET_WAIT_SECONDS)
            await _set_session_status(
                session_id, "pushing", bot=bot, chat_id=chat_id, message_id=message_id
            )
            client = KaggleClient()
            kernel_ref = await _push_preview3d_kernel(client, dataset_id)
            session["kaggle_kernel_ref"] = kernel_ref
            try:
                await register_job(
                    "preview3d",
                    kernel_ref,
                    meta={
                        "session_id": session_id,
                        "chat_id": chat_id,
                        "month": month,
                        "pid": os.getpid(),
                    },
                )
                registered = True
            except Exception:
                logger.warning("3di: failed to register recovery job", exc_info=True)
            await _set_session_status(
                session_id, "rendering", bot=bot, chat_id=chat_id, message_id=message_id
            )
            final_status, status_data, duration = await _poll_kaggle_kernel(
                client, kernel_ref, session_id, bot, chat_id, message_id
            )
            if final_status != "complete":
                failure = ""
                if status_data:
                    failure = status_data.get("failureMessage") or ""
                raise RuntimeError(f"Kaggle kernel failed ({final_status}) {failure}".strip())
            await _set_session_status(
                session_id, "download", bot=bot, chat_id=chat_id, message_id=message_id
            )
            output_dir = Path(tempfile.gettempdir()) / f"preview3d-{session_id}"
            try:
                results = await _download_kaggle_results(client, kernel_ref, session_id)
                await _set_session_status(
                    session_id, "apply_results", bot=bot, chat_id=chat_id, message_id=message_id
                )
                updated, errors, skipped = await update_previews_from_results(db, results)
                updated_event_ids: list[int] = []
                seen_event_ids: set[int] = set()
                for result in results:
                    status = (result.get("status") or "").lower()
                    if status != "ok" or not result.get("preview_url"):
                        continue
                    raw_event_id = result.get("event_id")
                    try:
                        event_id = int(raw_event_id)
                    except (TypeError, ValueError):
                        continue
                    if event_id in seen_event_ids:
                        continue
                    seen_event_ids.add(event_id)
                    updated_event_ids.append(event_id)

                updated_events: list[Event] = []
                month_url: str | None = None
                if updated_event_ids:
                    async with db.get_session() as db_session:
                        result = await db_session.execute(
                            select(Event).where(Event.id.in_(updated_event_ids))
                        )
                        updated_events = result.scalars().all()
                        month_page = await db_session.get(MonthPage, month)
                        if month_page and month_page.url:
                            month_url = month_page.url
                else:
                    async with db.get_session() as db_session:
                        month_page = await db_session.get(MonthPage, month)
                        if month_page and month_page.url:
                            month_url = month_page.url

                events_by_id = {event.id: event for event in updated_events}
                ordered_events = [
                    events_by_id[event_id]
                    for event_id in updated_event_ids
                    if event_id in events_by_id
                ]

                schedule_tasks = None
                try:
                    from main import schedule_event_update_tasks as schedule_tasks
                except Exception:
                    try:
                        from main_part2 import schedule_event_update_tasks as schedule_tasks
                    except Exception:
                        logger.exception("3di: schedule_event_update_tasks import failed")
                        schedule_tasks = None

                if schedule_tasks:
                    for event in ordered_events:
                        await schedule_tasks(db, event, skip_vk_sync=True)
                session["status"] = "done"
                if message_id:
                    lines = [
                        f"🎨 <b>3D-превью: {month_label}</b>",
                        "",
                        f"📊 Событий: {event_count}",
                        f"✅ Успешно: {updated}",
                        f"⚠️ Ошибок: {errors}",
                        f"⏭ Пропущено: {skipped}",
                        f"⏱ Длительность: {duration:.1f}с",
                        "",
                    ]
                    if month_url:
                        lines.append(
                            f"🔗 <a href=\"{html.escape(month_url)}\">Страница месяца</a>"
                        )
                    else:
                        lines.append("🔗 <i>Страница месяца не найдена</i>")
                    lines.append("")
                    lines.append("<b>Обновленные события:</b>")
                    listed_events = ordered_events[:5]
                    if listed_events:
                        for idx, event in enumerate(listed_events, 1):
                            title = html.escape(event.title)
                            if event.telegraph_url:
                                url = html.escape(event.telegraph_url)
                                lines.append(f"{idx}. <a href=\"{url}\">{title}</a>")
                            else:
                                lines.append(f"{idx}. {title}")
                        if len(ordered_events) > len(listed_events):
                            lines.append(
                                f"... и еще {len(ordered_events) - len(listed_events)}"
                            )
                    else:
                        lines.append("Нет обновленных событий.")
                    text = "\n".join(lines)
                    await bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                            [InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")],
                            [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
                        ]),
                        parse_mode="HTML",
                    )
            if registered and kernel_ref:
                await remove_job("preview3d", kernel_ref)
            finally:
                shutil.rmtree(output_dir, ignore_errors=True)
    except Exception as exc:
        session["status"] = "error"
        session["error"] = str(exc)
        if registered and kernel_ref:
            try:
                await remove_job("preview3d", kernel_ref)
            except Exception:
                logger.warning("3di: failed to remove recovery job after error", exc_info=True)
        if message_id:
            error_text = html.escape(str(exc))
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=f"❌ <b>Ошибка 3D-превью</b>\n\n{error_text}",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")],
                    [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
                ]),
                parse_mode="HTML",
            )


async def handle_3di_command(message: types.Message, db: Database, bot) -> None:
    """Handle /3di command - show main menu.
    
    Args:
        message: The message triggering the command (e.g., "/3di multy")
    """
    if not await _is_authorized(db, message.from_user.id):
        await bot.send_message(message.chat.id, "❌ Недостаточно прав")
        return
    
    # Parse arguments
    # Parse arguments
    full_text = message.text or message.caption or ""
    args = full_text.split()[1:]
    is_multy = "multy" in args or "multi" in args
    
    text = (
        "🎨 <b>3D-превью генератор</b>\n\n"
        "Генерация 3D-превью для событий с помощью Blender на Kaggle.\n\n"
    )
    if is_multy:
        text += "🎭 <b>Режим: MULTY</b> (только события с 2+ картинками)\n\n"
    
    text += "Выберите действие:"
    
    await bot.send_message(
        message.chat.id,
        text,
        reply_markup=_build_main_menu(is_multy=is_multy),
        parse_mode="HTML"
    )


async def handle_3di_callback(
    callback: types.CallbackQuery,
    db: Database,
    bot,
    *,
    start_kaggle_render: Callable | None = None,
) -> None:
    """Handle callbacks from /3di menu."""
    if not callback.data or not callback.data.startswith("3di:"):
        return
    
    if not await _is_authorized(db, callback.from_user.id):
        await callback.answer("❌ Недостаточно прав", show_alert=True)
        return
    
    data = callback.data
    chat_id = callback.message.chat.id
    message_id = callback.message.message_id
    
    is_multy = data.endswith(":multy")
    suffix = ":multy" if is_multy else ""
    # Strip suffix for logic processing steps that don't need it or handle it manually
    base_data = data.replace(":multy", "")
    
    if base_data == "3di:close":
        await bot.delete_message(chat_id, message_id)
        await callback.answer()
        return
    
    if base_data == "3di:back":
        text = (
            "🎨 <b>3D-превью генератор</b>\n\n"
            "Генерация 3D-превью для событий с помощью Blender на Kaggle.\n\n"
        )
        if is_multy:
            text += "🎭 <b>Режим: MULTY</b> (только события с 2+ картинками)\n\n"
        text += "Выберите действие:"
        
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=_build_main_menu(is_multy=is_multy),
            parse_mode="HTML"
        )
        await callback.answer()
        return
    
    if base_data == "3di:month_select":
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="📅 <b>Выберите месяц для генерации:</b>",
            reply_markup=_build_month_menu(is_multy=is_multy),
            parse_mode="HTML"
        )
        await callback.answer()
        return
    
    if base_data == "3di:new_only":
        # Generate for events added after the last one with preview
        min_images = 2 if is_multy else 1
        events = await _get_new_events_gap(db, min_images=min_images)
        
        if not events:
            await callback.answer("Нет новых событий (после последнего с превью)", show_alert=True)
            return
            
        mode_str = "new_only:multy" if is_multy else "new_only"
        # Use a generic label for the month/group since it's a gap fill
        label = "New Events Gap"
        await _start_generation(
            db, bot, callback, events, label, mode_str, start_kaggle_render
        )
        return

    if base_data == "3di:new":
        # Generate for all months - events without preview
        today = datetime.now(timezone.utc).date()
        month_key = today.strftime("%Y-%m")
        min_images = 2 if is_multy else 1
        events = await _get_events_without_preview(db, month_key, min_images=min_images)
        
        if not events:
            await callback.answer("Нет событий без превью (в текущем месяце)", show_alert=True)
            return
        
        mode_str = "new:multy" if is_multy else "new"
        await _start_generation(
            db, bot, callback, events, month_key, mode_str, start_kaggle_render
        )
        return
    
    if base_data == "3di:all_missing":
        # Generate for ALL future events (date >= today) without preview
        min_images = 2 if is_multy else 1
        events = await _get_all_future_events_without_preview(db, min_images=min_images)
        
        if not events:
            await callback.answer("Нет будущих событий без превью", show_alert=True)
            return
        
        mode_str = "all_missing:multy" if is_multy else "all_missing"
        # Use a descriptive label since this spans multiple months
        label = "All Missing"
        await _start_generation(
            db, bot, callback, events, label, mode_str, start_kaggle_render
        )
        return
    
    if base_data == "3di:all":
        # Regenerate all for current month
        today = datetime.now(timezone.utc).date()
        month_key = today.strftime("%Y-%m")
        min_images = 2 if is_multy else 1
        events = await _get_events_for_month(db, month_key, min_images=min_images)
        
        if not events:
            await callback.answer(f"Нет событий ({min_images}+ изображений)", show_alert=True)
            return
        
        mode_str = "all:multy" if is_multy else "all"
        await _start_generation(
            db, bot, callback, events, month_key, mode_str, start_kaggle_render
        )
        return
    
    if base_data.startswith("3di:gen:"):
        month_key = base_data.split(":")[2]
        min_images = 2 if is_multy else 1
        events = await _get_events_for_month(db, month_key, min_images=min_images)
        
        if not events:
            await callback.answer(f"Нет событий ({min_images}+ изображений) в этом месяце", show_alert=True)
            return
        
        mode_str = "month:multy" if is_multy else "month"
        await _start_generation(
            db, bot, callback, events, month_key, mode_str, start_kaggle_render
        )
        return
    
    if data.startswith("3di:status:"):
        session_id = int(data.split(":")[2])
        session = _active_sessions.get(session_id)
        if not session:
            await callback.answer("Сессия не найдена", show_alert=True)
            return
        await callback.answer(f"Статус: {session.get('status', 'unknown')}")
        return
    
    await callback.answer("Неизвестное действие", show_alert=True)


async def _start_generation(
    db: Database,
    bot,
    callback: types.CallbackQuery,
    events: list[Event],
    month: str,
    mode: str,
    start_kaggle_render: Callable | None,
) -> None:
    """Start 3D preview generation for events."""
    chat_id = callback.message.chat.id
    message_id = callback.message.message_id
    
    # Create session
    session_id = int(datetime.now(timezone.utc).timestamp() * 1000)
    _active_sessions[session_id] = {
        "status": "preparing",
        "month": month,
        "mode": mode,
        "event_count": len(events),
        "created_at": datetime.now(timezone.utc),
        "chat_id": chat_id,
        "message_id": message_id,
    }
    
    # Build payload
    payload = {
        "events": [
            {
                "event_id": e.id,
                "title": e.title,
                "images": (e.photo_urls or [])[:MAX_IMAGES_PER_EVENT]
            }
            for e in events
        ]
    }
    
    await bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text=_format_status_text(_active_sessions[session_id]),
        reply_markup=_build_status_keyboard(session_id),
        parse_mode="HTML"
    )
    await callback.answer("Генерация запущена!")

    if start_kaggle_render is None:
        start_kaggle_render = _run_kaggle_render
    try:
        await start_kaggle_render(
            db=db,
            bot=bot,
            chat_id=chat_id,
            session_id=session_id,
            payload=payload,
            month=month,
        )
    except Exception as e:
        logger.exception("3di: Kaggle render failed")
        _active_sessions[session_id]["status"] = "error"
        _active_sessions[session_id]["error"] = str(e)
        
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=f"❌ Ошибка: {html.escape(str(e))}",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")]
            ]),
            parse_mode="HTML"
        )


async def update_previews_from_results(
    db: Database,
    results: list[dict],
) -> tuple[int, int, int]:
    """Update Event.preview_3d_url from Kaggle results.
    
    Returns: (updated_count, error_count, skipped_count)
    """
    updated = 0
    errors = 0
    skipped = 0
    
    async with db.get_session() as session:
        for result in results:
            event_id = result.get("event_id")
            preview_url = result.get("preview_url")
            status = result.get("status", "")
            
            if not event_id:
                logger.warning("3di: Result without event_id: %s", result)
                continue
            
            if status == "ok" and preview_url:
                event = await session.get(Event, event_id)
                if event:
                    event.preview_3d_url = preview_url
                    updated += 1
                    logger.info("3di: Updated preview for event %d: %s", event_id, preview_url)
                else:
                    logger.warning("3di: Event %d not found in DB", event_id)
            elif status == "skip":
                skipped += 1
                logger.debug("3di: Skipped event %d: %s", event_id, result.get("error", "no images"))
            else:
                errors += 1
                error_msg = result.get("error", "unknown")
                logger.warning("3di: Failed for event %d: %s", event_id, error_msg)
        
        await session.commit()
    
    return updated, errors, skipped
