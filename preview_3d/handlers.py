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
from models import Event, User
from sqlmodel import select
from video_announce.kaggle_client import KaggleClient, KERNELS_ROOT_PATH

logger = logging.getLogger(__name__)

# Constants
MONTHS_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}
MAX_IMAGES_PER_EVENT = 7
KAGGLE_KERNEL_FOLDER = "Preview3D"
KAGGLE_DATASET_SLUG = "preview3d-dataset"
KAGGLE_POLL_INTERVAL_SECONDS = 20
KAGGLE_TIMEOUT_SECONDS = 30 * 60
KAGGLE_STARTUP_WAIT_SECONDS = 10

# Store active sessions (in production, use DB)
_active_sessions: dict[int, dict] = {}


async def _is_authorized(db: Database, user_id: int) -> bool:
    """Check if user is superadmin."""
    async with db.get_session() as session:
        user = await session.get(User, user_id)
        return user is not None and user.is_superadmin


async def _get_events_for_month(db: Database, month: str) -> list[Event]:
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
    return [e for e in events if e.photo_urls and len(e.photo_urls) > 0]


async def _get_events_without_preview(db: Database, month: str) -> list[Event]:
    """Get events that don't have a 3D preview yet."""
    events = await _get_events_for_month(db, month)
    return [e for e in events if not e.preview_3d_url]


def _build_main_menu() -> InlineKeyboardMarkup:
    """Build main menu for /3di command."""
    buttons = [
        [InlineKeyboardButton(text="🆕 Сгенерировать новые", callback_data="3di:new")],
        [InlineKeyboardButton(text="🔄 Перегенерировать все", callback_data="3di:all")],
        [InlineKeyboardButton(text="📅 Выбрать месяц", callback_data="3di:month_select")],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _build_month_menu() -> InlineKeyboardMarkup:
    """Build month selection menu."""
    today = datetime.now(timezone.utc).date()
    buttons = []
    
    for i in range(6):  # Show 6 months
        month_date = (today.replace(day=1) + timedelta(days=32*i)).replace(day=1)
        month_key = month_date.strftime("%Y-%m")
        month_name = MONTHS_RU[month_date.month]
        year = month_date.year
        buttons.append([
            InlineKeyboardButton(
                text=f"{month_name} {year}",
                callback_data=f"3di:gen:{month_key}"
            )
        ])
    
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def _require_kaggle_username() -> str:
    username = (os.getenv("KAGGLE_USERNAME") or "").strip()
    if not username:
        raise RuntimeError("KAGGLE_USERNAME not set")
    return username


async def _create_preview3d_dataset(payload: dict, session_id: int) -> str:
    username = _require_kaggle_username()
    dataset_id = f"{username}/{KAGGLE_DATASET_SLUG}"
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
) -> tuple[str, dict | None, float]:
    started = time.monotonic()
    deadline = started + KAGGLE_TIMEOUT_SECONDS
    last_status: dict | None = None
    while time.monotonic() < deadline:
        status = await asyncio.to_thread(client.get_kernel_status, kernel_ref)
        last_status = status
        status_name = (status.get("status") or "").upper()
        if session_id in _active_sessions:
            _active_sessions[session_id]["status"] = status_name.lower() or "running"
        if status_name == "COMPLETE":
            return "complete", last_status, time.monotonic() - started
        if status_name in ("ERROR", "FAILED", "CANCELLED"):
            return "failed", last_status, time.monotonic() - started
        await asyncio.sleep(KAGGLE_POLL_INTERVAL_SECONDS)
    return "timeout", last_status, time.monotonic() - started


async def _download_kaggle_results(
    client: KaggleClient, kernel_ref: str
) -> list[dict]:
    output_dir = Path(tempfile.gettempdir()) / "preview3d_output"
    output_dir.mkdir(exist_ok=True)
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
    if not output_path:
        for name in files:
            path = output_dir / name
            if path.suffix.lower() == ".json":
                output_path = path
                break
    if not output_path or not output_path.exists():
        raise RuntimeError("output.json not found in Kaggle output")
    output_data = json.loads(output_path.read_text(encoding="utf-8"))
    results = output_data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Invalid output.json format: missing results")
    return results


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
    month_name = MONTHS_RU.get(int(month.split("-")[1]), month)
    event_count = session.get("event_count", len(payload.get("events", [])))
    try:
        session["status"] = "dataset"
        dataset_id = await _create_preview3d_dataset(payload, session_id)
        session["kaggle_dataset"] = dataset_id
        session["status"] = "kernel_push"
        client = KaggleClient()
        kernel_ref = await _push_preview3d_kernel(client, dataset_id)
        session["kaggle_kernel_ref"] = kernel_ref
        session["status"] = "rendering"
        final_status, status_data, duration = await _poll_kaggle_kernel(
            client, kernel_ref, session_id
        )
        if final_status != "complete":
            failure = ""
            if status_data:
                failure = status_data.get("failureMessage") or ""
            raise RuntimeError(f"Kaggle kernel failed ({final_status}) {failure}".strip())
        session["status"] = "download"
        results = await _download_kaggle_results(client, kernel_ref)
        session["status"] = "apply_results"
        updated, errors, skipped = await update_previews_from_results(db, results)
        session["status"] = "done"
        if message_id:
            text = (
                f"🎨 <b>3D-превью: {month_name}</b>\n\n"
                f"📊 Событий: {event_count}\n"
                f"✅ Обновлено: {updated}\n"
                f"⚠️ Ошибок: {errors}\n"
                f"⏭ Пропущено: {skipped}\n"
                f"⏱ Длительность: {duration:.1f}с"
            )
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
    except Exception as exc:
        session["status"] = "error"
        session["error"] = str(exc)
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
    """Handle /3di command - show main menu."""
    if not await _is_authorized(db, message.from_user.id):
        await bot.send_message(message.chat.id, "❌ Недостаточно прав")
        return
    
    text = (
        "🎨 <b>3D-превью генератор</b>\n\n"
        "Генерация 3D-превью для событий с помощью Blender на Kaggle.\n\n"
        "Выберите действие:"
    )
    
    await bot.send_message(
        message.chat.id,
        text,
        reply_markup=_build_main_menu(),
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
    
    if data == "3di:close":
        await bot.delete_message(chat_id, message_id)
        await callback.answer()
        return
    
    if data == "3di:back":
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=(
                "🎨 <b>3D-превью генератор</b>\n\n"
                "Генерация 3D-превью для событий с помощью Blender на Kaggle.\n\n"
                "Выберите действие:"
            ),
            reply_markup=_build_main_menu(),
            parse_mode="HTML"
        )
        await callback.answer()
        return
    
    if data == "3di:month_select":
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text="📅 <b>Выберите месяц для генерации:</b>",
            reply_markup=_build_month_menu(),
            parse_mode="HTML"
        )
        await callback.answer()
        return
    
    if data == "3di:new":
        # Generate for all months - events without preview
        today = datetime.now(timezone.utc).date()
        month_key = today.strftime("%Y-%m")
        events = await _get_events_without_preview(db, month_key)
        
        if not events:
            await callback.answer("Нет событий без превью", show_alert=True)
            return
        
        await _start_generation(
            db, bot, callback, events, month_key, "new", start_kaggle_render
        )
        return
    
    if data == "3di:all":
        # Regenerate all for current month
        today = datetime.now(timezone.utc).date()
        month_key = today.strftime("%Y-%m")
        events = await _get_events_for_month(db, month_key)
        
        if not events:
            await callback.answer("Нет событий с изображениями", show_alert=True)
            return
        
        await _start_generation(
            db, bot, callback, events, month_key, "all", start_kaggle_render
        )
        return
    
    if data.startswith("3di:gen:"):
        month_key = data.split(":")[2]
        events = await _get_events_for_month(db, month_key)
        
        if not events:
            await callback.answer("Нет событий с изображениями в этом месяце", show_alert=True)
            return
        
        await _start_generation(
            db, bot, callback, events, month_key, "month", start_kaggle_render
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
    
    month_name = MONTHS_RU.get(int(month.split("-")[1]), month)
    
    status_text = (
        f"🎨 <b>3D-превью: {month_name}</b>\n\n"
        f"📊 Событий к обработке: {len(events)}\n"
        f"🔄 Статус: подготовка...\n\n"
        f"Режим: {mode}"
    )
    
    status_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔄 Обновить статус", callback_data=f"3di:status:{session_id}")],
        [InlineKeyboardButton(text="❌ Закрыть", callback_data="3di:close")],
    ])
    
    await bot.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text=status_text,
        reply_markup=status_keyboard,
        parse_mode="HTML"
    )
    await callback.answer("Генерация запущена!")
    
    _active_sessions[session_id]["status"] = "rendering"
    
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
