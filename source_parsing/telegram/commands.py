import logging
import asyncio
from aiogram import Router, types, Bot, F
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db import Database
from models import Channel
from .service import run_telegram_monitor

tg_router = Router()
logger = logging.getLogger(__name__)

# State management
# user_id -> True/False
adding_channel_sessions: set[int] = set()

def get_tg_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Запустить мониторинг", callback_data="tg:run")],
        [InlineKeyboardButton(text="➕ Добавить канал", callback_data="tg:add")],
        [InlineKeyboardButton(text="📋 Список каналов", callback_data="tg:list")]
    ])

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

    await message.answer("🤖 <b>Telegram Monitor Control</b>", reply_markup=get_tg_keyboard(), parse_mode="HTML")

@tg_router.callback_query(F.data.startswith("tg:"))
async def handle_tg_callback(callback: CallbackQuery):
    import main
    db = main.get_db()
    user_id = callback.from_user.id
    
    # Check admin again for safety
    if not main.has_admin_access(await _get_user(db, user_id)):
        await callback.answer("⛔ Access denied", show_alert=True)
        return

    action = callback.data.split(":")[1]
    
    if action == "run":
        await callback.message.answer("🚀 Starting Telegram Monitor...")
        await callback.answer()
        # Run async
        asyncio.create_task(run_monitor_task(callback.bot, callback.message.chat.id))
        
    elif action == "add":
        adding_channel_sessions.add(user_id)
        await callback.message.answer(
            "✍️ Отправьте юзернейм канала (например @artkoder_events) или ссылку."
        )
        await callback.answer()
        
    elif action == "list":
        await list_channels(db, callback.message)
        await callback.answer()

@tg_router.message(lambda m: m.from_user.id in adding_channel_sessions)
async def handle_channel_input(message: types.Message):
    """Handle channel username input."""
    import main
    db = main.get_db()
    
    user_id = message.from_user.id
    text = message.text.strip()
    
    if text.lower() == "/cancel":
        adding_channel_sessions.discard(user_id)
        await message.answer("❌ Cancelled.")
        return

    # Normalize username
    username = text.split("/")[-1].replace("@", "").strip()
    
    try:
        await add_channel(db, message, username)
        adding_channel_sessions.discard(user_id)
    except Exception as e:
        logger.error(f"Error adding channel: {e}")
        await message.answer(f"❌ Error: {e}")
        # Keep session open for retry? or close? Let's close to avoid stuck state.
        adding_channel_sessions.discard(user_id)

async def run_monitor_task(bot: Bot, chat_id: int):
    import main
    db = main.get_db()
    
    # Need to fetch channels logic?
    # Service expects list of strings.
    # Logic: Read from DB channels where is_asset=True
    async with db.get_session() as session:
        stmt = select(Channel).where(Channel.is_asset == True)
        result = await session.execute(stmt)
        channels_db = result.scalars().all()
        channel_usernames = [f"@{ch.username}" for ch in channels_db if ch.username]
        
    # Get session from env
    import os
    tg_session = os.environ.get("TG_SESSION")
    if not tg_session:
        await bot.send_message(chat_id, "❌ TG_SESSION not found in environment!")
        return
        
    try:
        await run_telegram_monitor(db, tg_session, channel_usernames)
        await bot.send_message(chat_id, "✅ Monitor job submitted to Kaggle.")
    except Exception as e:
        logger.exception("Manual monitor run failed")
        await bot.send_message(chat_id, f"❌ Monitor run failed: {e}")

async def add_channel(db: Database, message: types.Message, username: str):
    async with db.get_session() as session:
        stmt = select(Channel).where(Channel.username == username)
        result = await session.execute(stmt)
        channel = result.scalar_one_or_none()
        
        if channel:
            if channel.is_asset:
                await message.answer(f"ℹ️ Channel @{username} is already monitored.")
            else:
                channel.is_asset = True
                await session.commit()
                await message.answer(f"✅ Channel @{username} enabled for monitoring.")
        else:
            try:
                # Resolve via Bot API
                chat = await message.bot.get_chat(f"@{username}")
                # Note: Bot must likely be added to channel or have read access if public
                # Public channels can be resolved by username usually.
                
                channel = Channel(
                    channel_id=chat.id,
                    title=chat.title,
                    username=username,
                    is_asset=True,
                    is_registered=True # Assuming if we adding it, we consider it "registered" in our system? 
                    # Actually is_registered usually means bot is present?
                    # Let's set is_asset=True (monitored source)
                )
                session.add(channel)
                await session.commit()
                await message.answer(f"✅ Added new channel @{username} (ID: {chat.id})")
            except Exception as e:
                await message.answer(f"⚠️ Could not resolve @{username}: {e}\nSaving anyway as target...")
                # Fallback: Save with dummy negative ID based on hash if real ID unknown? 
                # Or just error out? User wants to add channels.
                # Kaggle script uses usernames.
                # We can insert a dummy ID.
                dummy_id = -abs(hash(username)) % 10000000000 # Semi-unique dummy
                try:
                    channel = Channel(
                        channel_id=dummy_id,
                        title=username,
                        username=username,
                        is_asset=True
                    )
                    session.add(channel)
                    await session.commit()
                    await message.answer(f"✅ Added channel @{username} (Unresolved ID: {dummy_id})")
                except IntegrityError:
                    await session.rollback()
                    await message.answer("❌ Error adding channel to DB.")

async def list_channels(db: Database, message: types.Message):
    async with db.get_session() as session:
        stmt = select(Channel).where(Channel.is_asset == True)
        result = await session.execute(stmt)
        channels = result.scalars().all()
        
        if not channels:
            await message.answer("No channels monitored.")
            return
            
        lines = ["<b>Monitored Channels:</b>"]
        for ch in channels:
            lines.append(f"- @{ch.username} ({ch.title})")
        
        await message.answer("\n".join(lines), parse_mode="HTML")

async def _get_user(db: Database, user_id: int):
    from models import User
    async with db.get_session() as session:
        return await session.get(User, user_id)
