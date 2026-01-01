"""Handlers for 3D preview generation command /3di."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from aiogram import types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from db import Database
from models import Event, User
from sqlmodel import select

logger = logging.getLogger(__name__)

# Constants
MONTHS_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь"
}

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
    }
    
    # Build payload
    payload = {
        "events": [
            {
                "event_id": e.id,
                "title": e.title,
                "images": (e.photo_urls or [])[:57]  # Max 57 images per event
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
    
    # If we have a Kaggle render function, use it
    if start_kaggle_render:
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
                text=f"❌ Ошибка: {e}",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                    [InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")]
                ]),
                parse_mode="HTML"
            )
    else:
        # No render function - just show payload info
        logger.info("3di: No Kaggle render function, showing payload info")
        
        status_text = (
            f"🎨 <b>3D-превью: {month_name}</b>\n\n"
            f"📊 Событий: {len(events)}\n"
            f"📷 Всего изображений: {sum(len(e.photo_urls or []) for e in events)}\n\n"
            f"⚠️ Kaggle рендер не настроен.\n"
            f"Payload готов к отправке."
        )
        
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=status_text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="3di:back")]
            ]),
            parse_mode="HTML"
        )
        _active_sessions[session_id]["status"] = "done"


async def update_previews_from_results(
    db: Database,
    results: list[dict],
) -> tuple[int, int]:
    """Update Event.preview_3d_url from Kaggle results.
    
    Returns: (updated_count, error_count)
    """
    updated = 0
    errors = 0
    
    async with db.get_session() as session:
        for result in results:
            event_id = result.get("event_id")
            preview_url = result.get("preview_url")
            status = result.get("status", "")
            
            if not event_id:
                continue
            
            if status == "ok" and preview_url:
                event = await session.get(Event, event_id)
                if event:
                    event.preview_3d_url = preview_url
                    updated += 1
                    logger.info("3di: Updated preview for event %d: %s", event_id, preview_url)
            else:
                errors += 1
                error_msg = result.get("error", "unknown")
                logger.warning("3di: Failed for event %d: %s", event_id, error_msg)
        
        await session.commit()
    
    return updated, errors
