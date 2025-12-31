"""
/special command handler for generating holiday Telegraph pages.

FSM-based dialog flow:
1. User enters start date
2. User enters number of days
3. User uploads cover image
4. User enters page title
5. System generates Telegraph page
"""
from __future__ import annotations

import logging
from datetime import date, timezone
from typing import TYPE_CHECKING

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from runtime import require_main_attr

if TYPE_CHECKING:
    from aiogram import Bot

logger = logging.getLogger(__name__)

# Maximum days allowed for special pages
MAX_DAYS = 14


class SpecialStates(StatesGroup):
    """FSM states for /special command."""
    waiting_start_date = State()
    waiting_days = State()
    waiting_cover = State()
    waiting_title = State()


special_router = Router(name="special")


@special_router.message(Command("special"))
async def cmd_special(message: Message, state: FSMContext) -> None:
    """Start the /special command flow."""
    # Get db from running main module (avoids __main__ vs main split)
    get_db = require_main_attr("get_db")
    db = get_db()
    
    if db is None:
        logger.error("special_cmd: db is None, bot not fully initialized")
        await message.answer("❌ Бот ещё не инициализирован. Попробуйте позже.")
        return
    
    from models import User
    
    # Check superadmin access
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await message.answer("❌ Команда доступна только администраторам.")
            return
    
    await state.set_state(SpecialStates.waiting_start_date)
    
    await message.answer(
        "📅 **Генерация праздничной страницы**\n\n"
        "Введите дату начала периода.\n"
        "Формат: `2 января`, `02.01.2026` или `2026-01-02`\n\n"
        "Для отмены введите /cancel",
        parse_mode="Markdown"
    )


@special_router.message(Command("cancel"), SpecialStates.waiting_start_date)
@special_router.message(Command("cancel"), SpecialStates.waiting_days)
@special_router.message(Command("cancel"), SpecialStates.waiting_cover)
@special_router.message(Command("cancel"), SpecialStates.waiting_title)
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    """Cancel the /special command flow."""
    await state.clear()
    await message.answer("❌ Генерация праздничной страницы отменена.")


@special_router.message(SpecialStates.waiting_start_date)
async def handle_start_date(message: Message, state: FSMContext) -> None:
    """Handle start date input."""
    parse_events_date = require_main_attr("parse_events_date")
    
    text = message.text
    if not text:
        await message.answer(
            "❌ Пожалуйста, введите дату текстом.\n"
            "Формат: `2 января`, `02.01.2026` или `2026-01-02`",
            parse_mode="Markdown"
        )
        return
    
    parsed_date = parse_events_date(text.strip(), timezone.utc)
    if not parsed_date:
        await message.answer(
            "❌ Не удалось распознать дату.\n"
            "Формат: `2 января`, `02.01.2026` или `2026-01-02`\n\n"
            "Попробуйте ещё раз:",
            parse_mode="Markdown"
        )
        return
    
    await state.update_data(start_date=parsed_date.isoformat())
    await state.set_state(SpecialStates.waiting_days)
    
    format_day_pretty = require_main_attr("format_day_pretty")
    date_str = format_day_pretty(parsed_date)
    
    await message.answer(
        f"✅ Дата начала: **{date_str}**\n\n"
        f"Введите количество дней (1–{MAX_DAYS}):",
        parse_mode="Markdown"
    )


@special_router.message(SpecialStates.waiting_days)
async def handle_days(message: Message, state: FSMContext) -> None:
    """Handle number of days input."""
    text = message.text
    if not text:
        await message.answer(f"❌ Введите число от 1 до {MAX_DAYS}.")
        return
    
    try:
        days = int(text.strip())
    except ValueError:
        await message.answer(f"❌ Введите число от 1 до {MAX_DAYS}.")
        return
    
    if days < 1 or days > MAX_DAYS:
        await message.answer(f"❌ Количество дней должно быть от 1 до {MAX_DAYS}.")
        return
    
    await state.update_data(days=days)
    await state.set_state(SpecialStates.waiting_cover)
    
    await message.answer(
        f"✅ Количество дней: **{days}**\n\n"
        "Загрузите обложку страницы (фото или файл).\n"
        "Или отправьте `-` чтобы пропустить.",
        parse_mode="Markdown"
    )


@special_router.message(SpecialStates.waiting_cover)
async def handle_cover(message: Message, state: FSMContext) -> None:
    """Handle cover image upload."""
    extract_images = require_main_attr("extract_images")
    upload_images = require_main_attr("upload_images")
    
    # Check if user wants to skip cover
    if message.text and message.text.strip() == "-":
        await state.update_data(cover_url=None)
        await state.set_state(SpecialStates.waiting_title)
        await message.answer(
            "✅ Обложка пропущена.\n\n"
            "Введите заголовок страницы:"
        )
        return
    
    # Try to extract image
    images = await extract_images(message, message.bot)
    if not images:
        await message.answer(
            "❌ Не вижу изображения.\n"
            "Пришлите фото/файл или `-` чтобы пропустить."
        )
        return
    
    # Upload to Catbox
    images = images[:1]  # Only first image
    urls, _ = await upload_images(images, limit=1, force=True)
    
    if not urls:
        await message.answer(
            "❌ Не удалось загрузить изображение.\n"
            "Попробуйте другое фото или `-` чтобы пропустить."
        )
        return
    
    cover_url = urls[0]
    await state.update_data(cover_url=cover_url)
    await state.set_state(SpecialStates.waiting_title)
    
    await message.answer(
        "✅ Обложка загружена!\n\n"
        "Введите заголовок страницы:\n"
        "Например: `Новогодние праздники в Калининграде`",
        parse_mode="Markdown"
    )


@special_router.message(SpecialStates.waiting_title)
async def handle_title(message: Message, state: FSMContext) -> None:
    """Handle page title and generate the page."""
    from special_pages import create_special_telegraph_page
    
    text = message.text
    if not text or not text.strip():
        await message.answer("❌ Введите заголовок страницы.")
        return
    
    title = text.strip()
    data = await state.get_data()
    
    get_db = require_main_attr("get_db")
    db = get_db()
    start_date_str = data.get("start_date")
    days = data.get("days", 1)
    cover_url = data.get("cover_url")
    
    if not start_date_str or not db:
        await message.answer("❌ Ошибка: данные сессии потеряны. Начните заново с /special")
        await state.clear()
        return
    
    start_date = date.fromisoformat(start_date_str)
    
    # Notify user that generation is starting
    format_day_pretty = require_main_attr("format_day_pretty")
    from datetime import timedelta
    
    end_date = start_date + timedelta(days=days - 1)
    period_str = f"{format_day_pretty(start_date)} – {format_day_pretty(end_date)}"
    
    progress_msg = await message.answer(
        f"⏳ Генерация страницы...\n\n"
        f"📅 Период: {period_str}\n"
        f"📝 Заголовок: {title}"
    )
    
    try:
        url, used_days = await create_special_telegraph_page(
            db=db,
            start_date=start_date,
            days=days,
            cover_url=cover_url,
            title=title,
        )
        
        # Notify about day reduction if needed
        reduction_note = ""
        if used_days < days:
            reduction_note = (
                f"\n\n⚠️ Период сокращён до {used_days} дн. "
                "из-за лимита размера страницы."
            )
        
        await progress_msg.edit_text(
            f"✅ Страница создана!\n\n"
            f"📅 Период: {period_str}\n"
            f"📝 Заголовок: {title}\n"
            f"🔗 {url}"
            f"{reduction_note}"
        )
        
        logger.info(
            "special_page generated: url=%s days=%d/%d user=%d",
            url, used_days, days, message.from_user.id
        )
        
    except Exception as e:
        logger.exception("Failed to generate special page")
        await progress_msg.edit_text(
            f"❌ Ошибка при генерации страницы:\n{e}\n\n"
            "Попробуйте ещё раз с /special"
        )
    finally:
        await state.clear()
