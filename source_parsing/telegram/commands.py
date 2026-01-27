import logging
import asyncio
import shlex
from aiogram import Router, types, Bot, F
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db import Database
from models import TelegramSource
from .service import run_telegram_monitor

tg_router = Router()
logger = logging.getLogger(__name__)
tg_monitor_router = tg_router

# State management
# user_id -> mode
adding_source_sessions: dict[int, str] = {}
_monitor_lock = asyncio.Lock()

def get_tg_keyboard():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚀 Запустить мониторинг", callback_data="tg:run")],
            [InlineKeyboardButton(text="➕ Добавить источник", callback_data="tg:add")],
            [InlineKeyboardButton(text="📋 Список источников", callback_data="tg:list")],
        ]
    )

@tg_router.message(Command("tg"))
async def cmd_tg(message: types.Message, command: CommandObject):
    """
    Handle /tg command with UI.
    """
    import main
    db = main.get_db()
    if not db:
        await message.answer("❌ Database not initialized")
        return

    # Check admin
    if not main.has_admin_access(await _get_user(db, message.from_user.id)):
        return

    await message.answer(
        "🤖 <b>Telegram Monitor Control</b>",
        reply_markup=get_tg_keyboard(),
        parse_mode="HTML",
    )

@tg_router.callback_query(F.data.startswith("tg:"))
async def handle_tg_callback(callback: CallbackQuery):
    import main
    db = main.get_db()
    user_id = callback.from_user.id
    
    # Check admin again for safety
    if not main.has_admin_access(await _get_user(db, user_id)):
        await callback.answer("⛔ Access denied", show_alert=True)
        return

    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    
    if action == "run":
        await callback.message.answer("🚀 Starting Telegram Monitor...")
        await callback.answer()
        # Run async
        asyncio.create_task(run_monitor_task(callback.bot, callback.message.chat.id))
        
    elif action == "add":
        adding_source_sessions[user_id] = "add"
        await callback.message.answer(
            "✍️ Отправьте юзернейм или ссылку.\n"
            "Можно добавить опции: trust=high|medium|low location=... ticket=...\n"
            "Пример: @artkoder_events trust=high location='Научная библиотека' ticket='https://...'"
        )
        await callback.answer()
        
    elif action == "list":
        await list_sources(db, callback.message)
        await callback.answer()
    elif action == "toggle" and len(parts) >= 3:
        await toggle_source(db, callback.message, int(parts[2]))
        await callback.answer()
    elif action == "trust" and len(parts) >= 3:
        await cycle_trust(db, callback.message, int(parts[2]))
        await callback.answer()

@tg_router.message(lambda m: m.from_user.id in adding_source_sessions)
async def handle_source_input(message: types.Message):
    """Handle source username input."""
    import main
    db = main.get_db()
    
    user_id = message.from_user.id
    text = message.text.strip()
    
    if text.lower() == "/cancel":
        adding_source_sessions.pop(user_id, None)
        await message.answer("❌ Cancelled.")
        return

    try:
        username, options = parse_source_input(text)
        await add_source(db, message, username, options)
        adding_source_sessions.pop(user_id, None)
    except Exception as e:
        logger.error(f"Error adding source: {e}")
        await message.answer(f"❌ Error: {e}")
        adding_source_sessions.pop(user_id, None)

async def run_monitor_task(bot: Bot, chat_id: int):
    import main
    db = main.get_db()
    
    try:
        if _monitor_lock.locked():
            await bot.send_message(chat_id, "⏳ Мониторинг уже запущен, ждём завершения.")
            return
        async with _monitor_lock:
            await run_telegram_monitor(db, bot=bot, chat_id=chat_id)
    except Exception as e:
        logger.exception("Manual monitor run failed")
        await bot.send_message(chat_id, f"❌ Monitor run failed: {e}")

def normalize_source(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("@"):
        raw = raw[1:]
    if "t.me/" in raw:
        parts = raw.split("t.me/", 1)[-1].split("/")
        raw = parts[0]
    return raw.strip()


def parse_source_input(text: str) -> tuple[str, dict[str, str]]:
    parts = shlex.split(text)
    if not parts:
        raise ValueError("empty input")
    username = normalize_source(parts[0])
    if not username:
        raise ValueError("invalid source")
    options: dict[str, str] = {}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        options[key] = value
    return username, options


async def add_source(
    db: Database,
    message: types.Message,
    username: str,
    options: dict[str, str] | None = None,
) -> None:
    options = options or {}
    trust = options.get("trust")
    if trust and trust.lower() not in {"high", "medium", "low"}:
        await message.answer("⚠️ trust должен быть high|medium|low")
        trust = None
    default_location = options.get("location")
    default_ticket = options.get("ticket")

    async with db.get_session() as session:
        result = await session.execute(
            select(TelegramSource).where(TelegramSource.username == username)
        )
        source = result.scalar_one_or_none()
        if source:
            source.enabled = True
            if trust:
                source.trust_level = trust.lower()
            if default_location:
                source.default_location = default_location
            if default_ticket:
                source.default_ticket_link = default_ticket
            await session.commit()
            await message.answer(f"✅ Источник @{username} обновлён и включён.")
            return
        try:
            source = TelegramSource(
                username=username,
                enabled=True,
                trust_level=trust.lower() if trust else None,
                default_location=default_location,
                default_ticket_link=default_ticket,
            )
            session.add(source)
            await session.commit()
            await message.answer(f"✅ Источник @{username} добавлен.")
        except IntegrityError:
            await session.rollback()
            await message.answer("❌ Ошибка добавления источника.")


async def list_sources(db: Database, message: types.Message):
    async with db.get_session() as session:
        result = await session.execute(select(TelegramSource))
        sources = result.scalars().all()
    if not sources:
        await message.answer("Источники не настроены.")
        return
    lines = ["<b>Telegram Sources:</b>"]
    keyboard = []
    for src in sources:
        status = "✅" if src.enabled else "⛔"
        trust = src.trust_level or "low"
        lines.append(f"{status} @{src.username} (trust={trust})")
        keyboard.append(
            [
                InlineKeyboardButton(
                    text=f"{'Disable' if src.enabled else 'Enable'} @{src.username}",
                    callback_data=f"tg:toggle:{src.id}",
                ),
                InlineKeyboardButton(
                    text=f"Trust → {trust}",
                    callback_data=f"tg:trust:{src.id}",
                ),
            ]
        )
    await message.answer(
        "\n".join(lines),
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        parse_mode="HTML",
    )


async def toggle_source(db: Database, message: types.Message, source_id: int) -> None:
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        src.enabled = not src.enabled
        await session.commit()
    await list_sources(db, message)


async def cycle_trust(db: Database, message: types.Message, source_id: int) -> None:
    order = ["low", "medium", "high"]
    async with db.get_session() as session:
        src = await session.get(TelegramSource, source_id)
        if not src:
            await message.answer("Источник не найден")
            return
        current = (src.trust_level or "low").lower()
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        src.trust_level = order[(idx + 1) % len(order)]
        await session.commit()
    await list_sources(db, message)

async def _get_user(db: Database, user_id: int):
    from models import User
    async with db.get_session() as session:
        return await session.get(User, user_id)
