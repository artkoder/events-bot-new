import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web, ClientSession
from difflib import SequenceMatcher
import json
import re
from telegraph import Telegraph
from functools import partial
import asyncio
import html
from io import BytesIO
import markdown
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import Field, SQLModel, select

logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
TELEGRAPH_TOKEN_FILE = os.getenv("TELEGRAPH_TOKEN_FILE", "/data/telegraph_token.txt")

# separator inserted between versions on Telegraph source pages
CONTENT_SEPARATOR = "🟧" * 10

# user_id -> (event_id, field?) for editing session
editing_sessions: dict[int, tuple[int, str | None]] = {}


class User(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    is_superadmin: bool = False


class PendingUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)


class RejectedUser(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    rejected_at: datetime = Field(default_factory=datetime.utcnow)


class Channel(SQLModel, table=True):
    channel_id: int = Field(primary_key=True)
    title: Optional[str] = None
    username: Optional[str] = None
    is_admin: bool = False
    is_registered: bool = False


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str


class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    festival: Optional[str] = None
    date: str
    time: str
    location_name: str
    location_address: Optional[str] = None
    city: Optional[str] = None
    ticket_price_min: Optional[int] = None
    ticket_price_max: Optional[int] = None
    ticket_link: Optional[str] = None
    event_type: Optional[str] = None
    emoji: Optional[str] = None
    end_date: Optional[str] = None
    is_free: bool = False
    telegraph_path: Optional[str] = None
    source_text: str
    telegraph_url: Optional[str] = None
    source_post_url: Optional[str] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)


class MonthPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    month: str = Field(primary_key=True)
    url: str
    path: str


class Database:
    def __init__(self, path: str):
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{path}")

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            result = await conn.exec_driver_sql("PRAGMA table_info(event)")
            cols = [r[1] for r in result.fetchall()]
            if "telegraph_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN telegraph_url VARCHAR"
                )
            if "ticket_price_min" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_price_min INTEGER"
                )
            if "ticket_price_max" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_price_max INTEGER"
                )
            if "ticket_link" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ticket_link VARCHAR"
                )
            if "source_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_post_url VARCHAR"
                )
            if "is_free" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN is_free BOOLEAN DEFAULT 0"
                )
            if "telegraph_path" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN telegraph_path VARCHAR"
                )
            if "event_type" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN event_type VARCHAR"
                )
            if "emoji" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN emoji VARCHAR"
                )
            if "end_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN end_date VARCHAR"
                )
            if "added_at" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN added_at VARCHAR"
                )

    def get_session(self) -> AsyncSession:
        """Create a new session with attributes kept after commit."""
        return AsyncSession(self.engine, expire_on_commit=False)


async def get_tz_offset(db: Database) -> str:
    async with db.get_session() as session:
        result = await session.get(Setting, "tz_offset")
        return result.value if result else "+00:00"


async def set_tz_offset(db: Database, value: str):
    async with db.get_session() as session:
        setting = await session.get(Setting, "tz_offset")
        if setting:
            setting.value = value
        else:
            setting = Setting(key="tz_offset", value=value)
            session.add(setting)
        await session.commit()


def validate_offset(value: str) -> bool:
    if len(value) != 6 or value[0] not in "+-" or value[3] != ":":
        return False
    try:
        h = int(value[1:3])
        m = int(value[4:6])
        return 0 <= h <= 14 and 0 <= m < 60
    except ValueError:
        return False


def offset_to_timezone(value: str) -> timezone:
    sign = 1 if value[0] == "+" else -1
    hours = int(value[1:3])
    minutes = int(value[4:6])
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


async def parse_event_via_4o(text: str) -> list[dict]:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    prompt_path = os.path.join("docs", "PROMPTS.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    loc_path = os.path.join("docs", "LOCATIONS.md")
    if os.path.exists(loc_path):
        with open(loc_path, "r", encoding="utf-8") as f:
            locations = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if locations:
            prompt += "\nKnown venues:\n" + "\n".join(locations)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    today = date.today().isoformat()
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Today is {today}. {text}"},
        ],
        "temperature": 0,
    }
    logging.info("Sending 4o parse request to %s", url)
    async with ClientSession() as session:
        resp = await session.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = await resp.json()
    logging.debug("4o response: %s", data)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
        .strip()
    )
    if content.startswith("```"):
        content = content.strip("`\n")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logging.error("Invalid JSON from 4o: %s", content)
        raise
    if isinstance(data, dict):
        if "events" in data and isinstance(data["events"], list):
            return data["events"]
        return [data]
    if isinstance(data, list):
        return data
    logging.error("Unexpected 4o format: %s", data)
    raise RuntimeError("bad 4o response")


async def ask_4o(text: str) -> str:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
    }
    logging.info("Sending 4o ask request to %s", url)
    async with ClientSession() as session:
        resp = await session.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = await resp.json()
    logging.debug("4o response: %s", data)
    return (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


async def check_duplicate_via_4o(ev: Event, new: Event) -> Tuple[bool, str, str]:
    """Ask the LLM whether two events are duplicates."""
    prompt = (
        "Existing event:\n"
        f"Title: {ev.title}\nDescription: {ev.description}\nLocation: {ev.location_name} {ev.location_address}\n"
        "New event:\n"
        f"Title: {new.title}\nDescription: {new.description}\nLocation: {new.location_name} {new.location_address}\n"
        "Are these the same event? Respond with JSON {\"duplicate\": true|false, \"title\": \"\", \"short_description\": \"\"}."
    )
    try:
        ans = await ask_4o(prompt)
        data = json.loads(ans)
        return (
            bool(data.get("duplicate")),
            data.get("title", ""),
            data.get("short_description", ""),
        )
    except Exception as e:
        logging.error("Duplicate check failed: %s", e)
        return False, "", ""


def get_telegraph_token() -> str | None:
    token = os.getenv("TELEGRAPH_TOKEN")
    if token:
        return token
    if os.path.exists(TELEGRAPH_TOKEN_FILE):
        with open(TELEGRAPH_TOKEN_FILE, "r", encoding="utf-8") as f:
            saved = f.read().strip()
            if saved:
                return saved
    try:
        tg = Telegraph()
        data = tg.create_account(short_name="eventsbot")
        token = data["access_token"]
        os.makedirs(os.path.dirname(TELEGRAPH_TOKEN_FILE), exist_ok=True)
        with open(TELEGRAPH_TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token)
        logging.info("Created Telegraph account; token stored at %s", TELEGRAPH_TOKEN_FILE)
        return token
    except Exception as e:
        logging.error("Failed to create Telegraph token: %s", e)
        return None


async def handle_start(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        result = await session.execute(select(User))
        user_count = len(result.scalars().all())
        user = await session.get(User, message.from_user.id)
        if user:
            await bot.send_message(message.chat.id, "Bot is running")
            return
        if user_count == 0:
            session.add(
                User(
                    user_id=message.from_user.id,
                    username=message.from_user.username,
                    is_superadmin=True,
                )
            )
            await session.commit()
            await bot.send_message(message.chat.id, "You are superadmin")
        else:
            await bot.send_message(message.chat.id, "Use /register to apply")


async def handle_register(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        if await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Already registered")
            return
        if await session.get(RejectedUser, message.from_user.id):
            await bot.send_message(message.chat.id, "Access denied by administrator")
            return
        if await session.get(PendingUser, message.from_user.id):
            await bot.send_message(message.chat.id, "Awaiting approval")
            return
        result = await session.execute(select(PendingUser))
        if len(result.scalars().all()) >= 10:
            await bot.send_message(
                message.chat.id, "Registration queue full, try later"
            )
            return
        session.add(
            PendingUser(
                user_id=message.from_user.id, username=message.from_user.username
            )
        )
        await session.commit()
        await bot.send_message(message.chat.id, "Registration pending approval")


async def handle_requests(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            return
        result = await session.execute(select(PendingUser))
        pending = result.scalars().all()
        if not pending:
            await bot.send_message(message.chat.id, "No pending users")
            return
        buttons = [
            [
                types.InlineKeyboardButton(
                    text="Approve", callback_data=f"approve:{p.user_id}"
                ),
                types.InlineKeyboardButton(
                    text="Reject", callback_data=f"reject:{p.user_id}"
                ),
            ]
            for p in pending
        ]
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        lines = [f"{p.user_id} {p.username or ''}" for p in pending]
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=keyboard)


async def process_request(callback: types.CallbackQuery, db: Database, bot: Bot):
    data = callback.data
    if data.startswith("approve") or data.startswith("reject"):
        uid = int(data.split(":", 1)[1])
        async with db.get_session() as session:
            p = await session.get(PendingUser, uid)
            if not p:
                await callback.answer("Not found", show_alert=True)
                return
            if data.startswith("approve"):
                session.add(User(user_id=uid, username=p.username, is_superadmin=False))
                await bot.send_message(uid, "You are approved")
            else:
                session.add(RejectedUser(user_id=uid, username=p.username))
                await bot.send_message(uid, "Your registration was rejected")
            await session.delete(p)
            await session.commit()
            await callback.answer("Done")
    elif data.startswith("del:"):
        _, eid, marker = data.split(":")
        month = None
        async with db.get_session() as session:
            event = await session.get(Event, int(eid))
            if event:
                month = event.date.split("..", 1)[0][:7]
                await session.delete(event)
                await session.commit()
        if month:
            await sync_month_page(db, month)
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        if marker == "exh":
            text, markup = await build_exhibitions_message(db, tz)
        else:
            target = datetime.strptime(marker, "%Y-%m-%d").date()
            text, markup = await build_events_message(db, target, tz)
        await callback.message.edit_text(text, reply_markup=markup)
        await callback.answer("Deleted")
    elif data.startswith("edit:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            event = await session.get(Event, eid)
        if event:
            editing_sessions[callback.from_user.id] = (eid, None)
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer()
    elif data.startswith("editfield:"):
        _, eid, field = data.split(":")
        editing_sessions[callback.from_user.id] = (int(eid), field)
        await callback.message.answer(f"Send new value for {field}")
        await callback.answer()
    elif data.startswith("editdone:"):
        if callback.from_user.id in editing_sessions:
            del editing_sessions[callback.from_user.id]
        await callback.message.answer("Editing finished")
        await callback.answer()
    elif data.startswith("togglefree:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            event = await session.get(Event, eid)
            if event:
                event.is_free = not event.is_free
                await session.commit()
                logging.info("togglefree: event %s set to %s", eid, event.is_free)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
        async with db.get_session() as session:
            event = await session.get(Event, eid)
        if event:
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer()
    elif data.startswith("markfree:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            event = await session.get(Event, eid)
            if event:
                event.is_free = True
                await session.commit()
                logging.info("markfree: event %s marked free", eid)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="\u2705 Бесплатное мероприятие",
                        callback_data=f"togglefree:{eid}",
                    )
                ]
            ]
        )
        try:
            await bot.edit_message_reply_markup(
                chat_id=callback.message.chat.id,
                message_id=callback.message.message_id,
                reply_markup=markup,
            )
        except Exception as e:
            logging.error("failed to update free button: %s", e)
        await callback.answer("Marked")
    elif data.startswith("nav:"):
        _, day = data.split(":")
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        target = datetime.strptime(day, "%Y-%m-%d").date()
        text, markup = await build_events_message(db, target, tz)
        await callback.message.edit_text(text, reply_markup=markup)
        await callback.answer()
    elif data.startswith("unset:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            ch = await session.get(Channel, cid)
            if ch:
                ch.is_registered = False
                logging.info("channel %s unset", cid)
                await session.commit()
        await send_channels_list(callback.message, db, bot, edit=True)
        await callback.answer("Removed")
    elif data.startswith("set:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            ch = await session.get(Channel, cid)
            if ch and ch.is_admin:
                ch.is_registered = True
                logging.info("channel %s registered", cid)
                await session.commit()
        await send_setchannel_list(callback.message, db, bot, edit=True)
        await callback.answer("Registered")


async def handle_tz(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2 or not validate_offset(parts[1]):
        await bot.send_message(message.chat.id, "Usage: /tz +02:00")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    await set_tz_offset(db, parts[1])
    await bot.send_message(message.chat.id, f"Timezone set to {parts[1]}")


async def handle_my_chat_member(update: types.ChatMemberUpdated, db: Database):
    if update.chat.type != "channel":
        return
    status = update.new_chat_member.status
    is_admin = status in {"administrator", "creator"}
    logging.info(
        "my_chat_member: %s -> %s (admin=%s)",
        update.chat.id,
        status,
        is_admin,
    )
    async with db.get_session() as session:
        channel = await session.get(Channel, update.chat.id)
        if not channel:
            channel = Channel(
                channel_id=update.chat.id,
                title=update.chat.title,
                username=getattr(update.chat, "username", None),
                is_admin=is_admin,
            )
            session.add(channel)
        else:
            channel.title = update.chat.title
            channel.username = getattr(update.chat, "username", None)
            channel.is_admin = is_admin
        await session.commit()


async def send_channels_list(message: types.Message, db: Database, bot: Bot, edit: bool = False):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(
            select(Channel).where(Channel.is_admin.is_(True))
        )
        channels = result.scalars().all()
    logging.info("channels list: %s", [c.channel_id for c in channels])
    lines = []
    keyboard = []
    for ch in channels:
        name = ch.title or ch.username or str(ch.channel_id)
        if ch.is_registered:
            lines.append(f"{name} ✅")
            keyboard.append([
                types.InlineKeyboardButton(text="Cancel", callback_data=f"unset:{ch.channel_id}")
            ])
        else:
            lines.append(name)
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_setchannel_list(message: types.Message, db: Database, bot: Bot, edit: bool = False):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(
            select(Channel).where(
                Channel.is_admin.is_(True), Channel.is_registered.is_(False)
            )
        )
        channels = result.scalars().all()
    logging.info("setchannel list: %s", [c.channel_id for c in channels])
    lines = []
    keyboard = []
    for ch in channels:
        name = ch.title or ch.username or str(ch.channel_id)
        lines.append(name)
        keyboard.append([
            types.InlineKeyboardButton(text=name, callback_data=f"set:{ch.channel_id}")
        ])
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)

async def handle_set_channel(message: types.Message, db: Database, bot: Bot):
    await send_setchannel_list(message, db, bot, edit=False)


async def handle_channels(message: types.Message, db: Database, bot: Bot):
    await send_channels_list(message, db, bot, edit=False)


async def upsert_event(session: AsyncSession, new: Event) -> Tuple[Event, bool]:
    """Insert or update an event if a similar one exists.

    Returns (event, added_flag)."""

    stmt = select(Event).where(
        Event.date == new.date,
        Event.time == new.time,
    )
    candidates = (await session.execute(stmt)).scalars().all()
    for ev in candidates:
        if (
            ev.location_name.strip().lower() == new.location_name.strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False

        title_ratio = SequenceMatcher(None, ev.title.lower(), new.title.lower()).ratio()
        if title_ratio >= 0.9:
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False

        if (
            ev.location_name.strip().lower() == new.location_name.strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False

        title_ratio = SequenceMatcher(None, ev.title.lower(), new.title.lower()).ratio()
        if title_ratio >= 0.9:
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False

        if (
            ev.location_name.strip().lower() == new.location_name.strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False

        title_ratio = SequenceMatcher(None, ev.title.lower(), new.title.lower()).ratio()
        loc_ratio = SequenceMatcher(None, ev.location_name.lower(), new.location_name.lower()).ratio()
        if title_ratio >= 0.6 and loc_ratio >= 0.6:
            ev.title = new.title
            ev.description = new.description
            ev.festival = new.festival
            ev.source_text = new.source_text
            ev.location_name = new.location_name
            ev.location_address = new.location_address
            ev.ticket_price_min = new.ticket_price_min
            ev.ticket_price_max = new.ticket_price_max
            ev.ticket_link = new.ticket_link
            ev.event_type = new.event_type
            ev.emoji = new.emoji
            ev.end_date = new.end_date
            ev.is_free = new.is_free
            await session.commit()
            return ev, False
        should_check = False
        if loc_ratio >= 0.4 or (ev.location_address or "") == (new.location_address or ""):
            should_check = True
        elif title_ratio >= 0.5:
            should_check = True
        if should_check:
            # uncertain, ask LLM
            try:
                dup, title, desc = await check_duplicate_via_4o(ev, new)
            except Exception:
                logging.exception("duplicate check failed")
                dup = False
            if dup:
                ev.title = title or new.title
                ev.description = desc or new.description
                ev.festival = new.festival
                ev.source_text = new.source_text
                ev.location_name = new.location_name
                ev.location_address = new.location_address
                ev.ticket_price_min = new.ticket_price_min
                ev.ticket_price_max = new.ticket_price_max
                ev.ticket_link = new.ticket_link
                ev.event_type = new.event_type
                ev.emoji = new.emoji
                ev.end_date = new.end_date
                ev.is_free = new.is_free
                await session.commit()
                return ev, False
    new.added_at = datetime.utcnow()
    session.add(new)
    await session.commit()
    return new, True


async def add_events_from_text(
    db: Database,
    text: str,
    source_link: str | None,
    html_text: str | None = None,
    media: tuple[bytes, str] | None = None,
) -> list[tuple[Event, bool, list[str], str]]:
    try:
        parsed = await parse_event_via_4o(text)
    except Exception as e:
        logging.error("LLM error: %s", e)
        return []

    results: list[tuple[Event, bool, list[str], str]] = []
    first = True
    for data in parsed:
        date_str = data.get("date", "") or ""
        end_date = data.get("end_date") or None
        if end_date and ".." in end_date:
            end_date = end_date.split("..", 1)[-1].strip()
        if ".." in date_str:
            start, maybe_end = [p.strip() for p in date_str.split("..", 1)]
            date_str = start
            if not end_date:
                end_date = maybe_end

        event = Event(
            title=data.get("title", ""),
            description=data.get("short_description", ""),
            festival=data.get("festival") or None,
            date=date_str,
            time=data.get("time", ""),
            location_name=data.get("location_name", ""),
            location_address=data.get("location_address"),
            city=data.get("city"),
            ticket_price_min=data.get("ticket_price_min"),
            ticket_price_max=data.get("ticket_price_max"),
            ticket_link=data.get("ticket_link"),
            event_type=data.get("event_type"),
            emoji=data.get("emoji"),
            end_date=end_date,
            is_free=bool(data.get("is_free")),
            source_text=text,
            source_post_url=source_link,
        )

        if not event.ticket_link and html_text:
            extracted = extract_link_from_html(html_text)
            if extracted:
                event.ticket_link = extracted

        # skip events that have already finished
        try:
            start = date.fromisoformat(event.date)
        except ValueError:
            logging.error("Invalid date from LLM: %s", event.date)
            continue
        final = date.fromisoformat(event.end_date) if event.end_date else start
        if final < date.today():
            logging.info("Ignoring past event %s on %s", event.title, event.date)
            continue

        async with db.get_session() as session:
            saved, added = await upsert_event(session, event)

        media_arg = media if first else None
        if saved.telegraph_url and saved.telegraph_path:
            await update_source_page(saved.telegraph_path, saved.title or "Event", html_text or text)
        else:
            res = await create_source_page(
                saved.title or "Event",
                saved.source_text,
                source_link,
                html_text,
                media_arg,
            )
            if res:
                url, path = res
                async with db.get_session() as session:
                    saved.telegraph_url = url
                    saved.telegraph_path = path
                    session.add(saved)
                    await session.commit()
        await sync_month_page(db, saved.date[:7])

        lines = [
            f"title: {saved.title}",
            f"date: {saved.date}",
            f"time: {saved.time}",
            f"location_name: {saved.location_name}",
        ]
        if saved.location_address:
            lines.append(f"location_address: {saved.location_address}")
        if saved.city:
            lines.append(f"city: {saved.city}")
        if saved.festival:
            lines.append(f"festival: {saved.festival}")
        if saved.description:
            lines.append(f"description: {saved.description}")
        if saved.event_type:
            lines.append(f"type: {saved.event_type}")
        if saved.ticket_price_min is not None:
            lines.append(f"price_min: {saved.ticket_price_min}")
        if saved.ticket_price_max is not None:
            lines.append(f"price_max: {saved.ticket_price_max}")
        if saved.ticket_link:
            lines.append(f"ticket_link: {saved.ticket_link}")
        if saved.telegraph_url:
            lines.append(f"telegraph: {saved.telegraph_url}")
        status = "added" if added else "updated"
        results.append((saved, added, lines, status))
        first = False
    return results


async def handle_add_event(message: types.Message, db: Database, bot: Bot):
    text = message.text.split(maxsplit=1)
    if len(text) != 2:
        await bot.send_message(message.chat.id, "Usage: /addevent <text>")
        return
    media = None
    if message.photo:
        bio = BytesIO()
        await bot.download(message.photo[-1].file_id, destination=bio)
        media = (bio.getvalue(), "photo.jpg")
    elif message.document and message.document.mime_type.startswith("image/"):
        bio = BytesIO()
        await bot.download(message.document.file_id, destination=bio)
        media = (bio.getvalue(), "image.jpg")
    elif message.video:
        bio = BytesIO()
        await bot.download(message.video.file_id, destination=bio)
        media = (bio.getvalue(), "video.mp4")

    results = await add_events_from_text(db, text[1], None, message.html_text, media)
    if not results:
        await bot.send_message(message.chat.id, "LLM error")
        return
    for saved, added, lines, status in results:
        markup = None
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[[
                    types.InlineKeyboardButton(
                        text="\u2753 Это бесплатное мероприятие", callback_data=f"markfree:{saved.id}"
                    )
                ]]
            )
        await bot.send_message(
            message.chat.id,
            f"Event {status}\n" + "\n".join(lines),
            reply_markup=markup,
        )


async def handle_add_event_raw(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2 or '|' not in parts[1]:
        await bot.send_message(message.chat.id, "Usage: /addevent_raw title|date|time|location")
        return
    title, date, time, location = (p.strip() for p in parts[1].split('|', 3))
    media = None
    if message.photo:
        bio = BytesIO()
        await bot.download(message.photo[-1].file_id, destination=bio)
        media = (bio.getvalue(), "photo.jpg")
    elif message.document and message.document.mime_type.startswith("image/"):
        bio = BytesIO()
        await bot.download(message.document.file_id, destination=bio)
        media = (bio.getvalue(), "image.jpg")
    elif message.video:
        bio = BytesIO()
        await bot.download(message.video.file_id, destination=bio)
        media = (bio.getvalue(), "video.mp4")

    event = Event(
        title=title,
        description="",
        festival=None,
        date=date,
        time=time,
        location_name=location,
        source_text=parts[1],
    )
    async with db.get_session() as session:
        event, added = await upsert_event(session, event)

    res = await create_source_page(
        event.title or "Event",
        event.source_text,
        None,
        event.source_text,
        media,
    )
    if res:
        url, path = res
        async with db.get_session() as session:
            event.telegraph_url = url
            event.telegraph_path = path
            session.add(event)
            await session.commit()
    await sync_month_page(db, event.date[:7])
    lines = [
        f"title: {event.title}",
        f"date: {event.date}",
        f"time: {event.time}",
        f"location_name: {event.location_name}",
    ]
    if event.telegraph_url:
        lines.append(f"telegraph: {event.telegraph_url}")
    status = "added" if added else "updated"
    markup = None
    if not event.is_free and event.ticket_price_min is None and event.ticket_price_max is None:
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[[
                types.InlineKeyboardButton(text="\u2753 Это бесплатное мероприятие", callback_data=f"markfree:{event.id}")
            ]]
        )
    await bot.send_message(
        message.chat.id,
        f"Event {status}\n" + "\n".join(lines),
        reply_markup=markup,
    )


def format_day(day: date, tz: timezone) -> str:
    if day == datetime.now(tz).date():
        return "Сегодня"
    return day.strftime("%d.%m.%Y")


MONTHS = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def format_day_pretty(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]}"


def month_name(month: str) -> str:
    y, m = month.split("-")
    return f"{MONTHS[int(m) - 1]} {y}"


MONTHS_PREP = [
    "январе",
    "феврале",
    "марте",
    "апреле",
    "мае",
    "июне",
    "июле",
    "августе",
    "сентябре",
    "октябре",
    "ноябре",
    "декабре",
]


def month_name_prepositional(month: str) -> str:
    y, m = month.split("-")
    return f"{MONTHS_PREP[int(m) - 1]} {y}"


def next_month(month: str) -> str:
    d = datetime.fromisoformat(month + "-01")
    n = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return n.strftime("%Y-%m")


def md_to_html(text: str) -> str:
    html_text = markdown.markdown(
        text,
        extensions=["markdown.extensions.fenced_code", "markdown.extensions.nl2br"],
    )
    # Telegraph API does not allow h1/h2 or Telegram-specific emoji tags
    html_text = re.sub(r"<(\/?)h[12]>", r"<\1h3>", html_text)
    html_text = re.sub(r"</?tg-emoji[^>]*>", "", html_text)
    return html_text


def extract_link_from_html(html_text: str) -> str | None:
    """Return a registration or ticket link from HTML if present."""
    pattern = re.compile(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(html_text))

    # prefer anchors whose text mentions registration or tickets
    for m in matches:
        href, label = m.group(1), m.group(2)
        text = label.lower()
        if any(word in text for word in ["регистра", "ticket", "билет"]):
            return href

    # otherwise look for anchors located near the word "регистрация"
    lower_html = html_text.lower()
    for m in matches:
        href = m.group(1)
        start, end = m.span()
        context_before = lower_html[max(0, start - 60) : start]
        context_after = lower_html[end : end + 60]
        if "регистра" in context_before or "регистра" in context_after:
            return href

    if matches:
        return matches[0].group(1)
    return None


def is_recent(e: Event) -> bool:
    if e.added_at is None:
        return False
    now = datetime.utcnow()
    start = datetime.combine(now.date() - timedelta(days=1), datetime.min.time())
    return e.added_at >= start


def format_event_md(e: Event) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001F6A9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "
    lines = [f"{prefix}{emoji_part}{e.title}".strip(), e.description.strip()]
    if e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link:
            txt += f" [по регистрации]({e.ticket_link})"
        lines.append(txt)
    elif e.ticket_link and (e.ticket_price_min is not None or e.ticket_price_max is not None):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        else:
            price = str(e.ticket_price_min or e.ticket_price_max or "")
        lines.append(f"[Билеты в источнике]({e.ticket_link}) {price}".strip())
    elif e.ticket_link:
        lines.append(f"[по регистрации]({e.ticket_link})")
    else:
        if e.ticket_price_min is not None and e.ticket_price_max is not None and e.ticket_price_min != e.ticket_price_max:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        elif e.ticket_price_min is not None:
            price = str(e.ticket_price_min)
        elif e.ticket_price_max is not None:
            price = str(e.ticket_price_max)
        else:
            price = ""
        if price:
            lines.append(f"Билеты {price}")
    if e.telegraph_url:
        lines.append(f"[подробнее]({e.telegraph_url})")
    loc = e.location_name
    if e.location_address:
        loc += f", {e.location_address}"
    if e.city:
        loc += f", #{e.city}"
    date_part = e.date.split("..", 1)[0]
    try:
        day = format_day_pretty(datetime.fromisoformat(date_part).date())
    except ValueError:
        logging.error("Invalid event date: %s", e.date)
        day = e.date
    lines.append(f"_{day} {e.time} {loc}_")
    return "\n".join(lines)


def format_exhibition_md(e: Event) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001F6A9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "
    lines = [f"{prefix}{emoji_part}{e.title}".strip(), e.description.strip()]
    if e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link:
            txt += f" [по регистрации]({e.ticket_link})"
        lines.append(txt)
    elif e.ticket_link:
        lines.append(f"[Билеты в источнике]({e.ticket_link})")
    elif e.ticket_price_min is not None and e.ticket_price_max is not None and e.ticket_price_min != e.ticket_price_max:
        lines.append(f"Билеты от {e.ticket_price_min} до {e.ticket_price_max}")
    elif e.ticket_price_min is not None:
        lines.append(f"Билеты {e.ticket_price_min}")
    elif e.ticket_price_max is not None:
        lines.append(f"Билеты {e.ticket_price_max}")
    if e.telegraph_url:
        lines.append(f"[подробнее]({e.telegraph_url})")
    loc = e.location_name
    if e.location_address:
        loc += f", {e.location_address}"
    if e.city:
        loc += f", #{e.city}"
    if e.end_date:
        end_part = e.end_date.split("..", 1)[0]
        try:
            end = format_day_pretty(datetime.fromisoformat(end_part).date())
        except ValueError:
            logging.error("Invalid end date: %s", e.end_date)
            end = e.end_date
        lines.append(f"_по {end}, {loc}_")
    return "\n".join(lines)


def event_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001F6A9 ")
    if e.emoji and not e.title.strip().startswith(e.emoji):
        nodes.append(f"{e.emoji} ")
    title_text = e.title
    if e.source_post_url:
        nodes.append({"tag": "a", "attrs": {"href": e.source_post_url}, "children": [title_text]})
    else:
        nodes.append(title_text)
    return nodes


def event_to_nodes(e: Event) -> list[dict]:
    md = format_event_md(e)
    lines = md.split("\n")
    body_md = "\n".join(lines[1:]) if len(lines) > 1 else ""
    from telegraph.utils import html_to_nodes
    nodes = [{"tag": "h4", "children": event_title_nodes(e)}]
    if body_md:
        html_text = md_to_html(body_md)
        nodes.extend(html_to_nodes(html_text))
    nodes.append({"tag": "p", "children": ["\u00A0"]})
    return nodes


def exhibition_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001F6A9 ")
    if e.emoji and not e.title.strip().startswith(e.emoji):
        nodes.append(f"{e.emoji} ")
    title_text = e.title
    if e.source_post_url:
        nodes.append({"tag": "a", "attrs": {"href": e.source_post_url}, "children": [title_text]})
    else:
        nodes.append(title_text)
    return nodes


def exhibition_to_nodes(e: Event) -> list[dict]:
    md = format_exhibition_md(e)
    lines = md.split("\n")
    body_md = "\n".join(lines[1:]) if len(lines) > 1 else ""
    from telegraph.utils import html_to_nodes
    nodes = [{"tag": "h4", "children": exhibition_title_nodes(e)}]
    if body_md:
        html_text = md_to_html(body_md)
        nodes.extend(html_to_nodes(html_text))
    nodes.append({"tag": "p", "children": ["\u00A0"]})
    return nodes


async def build_month_page_content(db: Database, month: str) -> tuple[str, list]:
    start = date.fromisoformat(month + "-01")
    next_start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date >= start.isoformat(), Event.date < next_start.isoformat())
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()

        ex_result = await session.execute(
            select(Event)
            .where(
                Event.end_date.is_not(None),
                Event.end_date >= start.isoformat(),
                Event.date <= next_start.isoformat(),
            )
            .order_by(Event.date)
        )
        exhibitions = ex_result.scalars().all()

        next_page = await session.get(MonthPage, next_month(month))
        next_url = next_page.url if next_page else None

    today = date.today()
    events = [
        e
        for e in events
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
    ]
    exhibitions = [
        e for e in exhibitions if e.end_date and e.end_date >= today.isoformat()
    ]

    by_day: dict[date, list[Event]] = {}
    for e in events:
        date_part = e.date.split("..", 1)[0]
        try:
            d = datetime.fromisoformat(date_part).date()
        except ValueError:
            logging.error("Invalid date for event %s: %s", e.id, e.date)
            continue
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    intro = (
        f"Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_name_prepositional(month)} — от лекций и концертов до культурных шоу. "
    )
    intro_nodes = [intro, {"tag": "a", "attrs": {"href": "https://t.me/kenigevents"}, "children": ["Полюбить Калининград Анонсы"]}]
    content.append({"tag": "p", "children": intro_nodes})

    for day in sorted(by_day):
        if day.weekday() == 5:
            content.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
        elif day.weekday() == 6:
            content.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
        content.append({"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(day)} 🟥🟥🟥"]})
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00A0"]})
        for ev in by_day[day]:
            content.extend(event_to_nodes(ev))

    if next_url:
        content.append({"tag": "a", "attrs": {"href": next_url}, "children": ["Страница следующего месяца"]})

    if exhibitions:
        content.append({"tag": "h3", "children": ["Постоянные выставки"]})
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00A0"]})
        for ev in exhibitions:
            content.extend(exhibition_to_nodes(ev))

    title = (
        f"События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
    )
    return title, content


async def sync_month_page(db: Database, month: str):
    title, content = await build_month_page_content(db, month)
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    async with db.get_session() as session:
        page = await session.get(MonthPage, month)
        try:
            if page:
                await asyncio.to_thread(
                    tg.edit_page, page.path, title=title, content=content
                )
                logging.info("Edited month page %s", month)
            else:
                data = await asyncio.to_thread(
                    tg.create_page, title, content=content
                )
                page = MonthPage(
                    month=month, url=data.get("url"), path=data.get("path")
                )
                session.add(page)
                logging.info("Created month page %s", month)
            await session.commit()
        except Exception as e:
            logging.error("Failed to sync month page %s: %s", month, e)


async def build_events_message(db: Database, target_date: date, tz: timezone):
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(
                (Event.date == target_date.isoformat())
                | (Event.end_date == target_date.isoformat())
            ).order_by(Event.time)
        )
        events = result.scalars().all()

    lines = []
    for e in events:
        prefix = ""
        if e.end_date and e.date == target_date.isoformat():
            prefix = "(Открытие) "
        elif e.end_date and e.end_date == target_date.isoformat() and e.end_date != e.date:
            prefix = "(Закрытие) "
        title = f"{e.emoji} {e.title}" if e.emoji else e.title
        lines.append(f"{e.id}. {prefix}{title}")
        loc = f"{e.time} {e.location_name}"
        if e.city:
            loc += f", #{e.city}"
        lines.append(loc)
        if e.is_free:
            lines.append("Бесплатно")
        else:
            price_parts = []
            if e.ticket_price_min is not None:
                price_parts.append(str(e.ticket_price_min))
            if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
                price_parts.append(str(e.ticket_price_max))
            if price_parts:
                lines.append("-".join(price_parts))
        if e.telegraph_url:
            lines.append(f"исходное: {e.telegraph_url}")
        lines.append("")
    if not lines:
        lines.append("No events")

    keyboard = [
        [
            types.InlineKeyboardButton(
                text="\u274C", callback_data=f"del:{e.id}:{target_date.isoformat()}"
            ),
            types.InlineKeyboardButton(
                text="\u270E", callback_data=f"edit:{e.id}"
            ),
        ]
        for e in events
    ]

    today = datetime.now(tz).date()
    prev_day = target_date - timedelta(days=1)
    next_day = target_date + timedelta(days=1)
    row = []
    if target_date > today:
        row.append(
            types.InlineKeyboardButton(text="\u25C0", callback_data=f"nav:{prev_day.isoformat()}")
        )
    row.append(
        types.InlineKeyboardButton(text="\u25B6", callback_data=f"nav:{next_day.isoformat()}")
    )
    keyboard.append(row)

    text = f"Events on {format_day(target_date, tz)}\n" + "\n".join(lines)
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    return text, markup


async def build_exhibitions_message(db: Database, tz: timezone):
    today = datetime.now(tz).date()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.end_date.is_not(None),
                Event.end_date >= today.isoformat(),
            )
            .order_by(Event.date)
        )
        events = result.scalars().all()

    lines = []
    for e in events:
        try:
            start = datetime.fromisoformat(e.date).date()
        except ValueError:
            if ".." in e.date:
                start = datetime.fromisoformat(e.date.split("..", 1)[0]).date()
            else:
                logging.error("Bad start date %s for event %s", e.date, e.id)
                continue
        end = None
        if e.end_date:
            try:
                end = datetime.fromisoformat(e.end_date).date()
            except ValueError:
                end = None

        period = ""
        if end:
            if start <= today:
                period = f"по {format_day_pretty(end)}"
            else:
                period = f"c {format_day_pretty(start)} по {format_day_pretty(end)}"
        title = f"{e.emoji} {e.title}" if e.emoji else e.title
        if period:
            lines.append(f"{e.id}. {title} ({period})")
        else:
            lines.append(f"{e.id}. {title}")
        loc = f"{e.time} {e.location_name}"
        if e.city:
            loc += f", #{e.city}"
        lines.append(loc)
        if e.is_free:
            lines.append("Бесплатно")
        else:
            price_parts = []
            if e.ticket_price_min is not None:
                price_parts.append(str(e.ticket_price_min))
            if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
                price_parts.append(str(e.ticket_price_max))
            if price_parts:
                lines.append("-".join(price_parts))
        if e.telegraph_url:
            lines.append(f"исходное: {e.telegraph_url}")
        lines.append("")

    if not lines:
        lines.append("No exhibitions")

    keyboard = [
        [
            types.InlineKeyboardButton(text="\u274C", callback_data=f"del:{e.id}:exh"),
            types.InlineKeyboardButton(text="\u270E", callback_data=f"edit:{e.id}"),
        ]
        for e in events
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if events else None
    text = "Exhibitions\n" + "\n".join(lines)
    return text, markup


async def show_edit_menu(user_id: int, event: Event, bot: Bot):
    lines = [
        f"title: {event.title}",
        f"description: {event.description}",
        f"festival: {event.festival or ''}",
        f"date: {event.date}",
        f"end_date: {event.end_date or ''}",
        f"time: {event.time}",
        f"location_name: {event.location_name}",
        f"location_address: {event.location_address or ''}",
        f"city: {event.city or ''}",
        f"event_type: {event.event_type or ''}",
        f"emoji: {event.emoji or ''}",
        f"ticket_price_min: {event.ticket_price_min}",
        f"ticket_price_max: {event.ticket_price_max}",
        f"ticket_link: {event.ticket_link or ''}",
        f"is_free: {event.is_free}",
    ]
    fields = [
        "title",
        "description",
        "festival",
        "date",
        "end_date",
        "time",
        "location_name",
        "location_address",
        "city",
        "event_type",
        "emoji",
        "ticket_price_min",
        "ticket_price_max",
        "ticket_link",
        "is_free",
    ]
    keyboard = []
    row = []
    for idx, field in enumerate(fields, 1):
        row.append(
            types.InlineKeyboardButton(
                text=field, callback_data=f"editfield:{event.id}:{field}"
            )
        )
        if idx % 3 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([
        types.InlineKeyboardButton(
            text=("\u2705 Бесплатно" if event.is_free else "\u274C Бесплатно"),
            callback_data=f"togglefree:{event.id}",
        )
    ])
    keyboard.append(
        [types.InlineKeyboardButton(text="Done", callback_data=f"editdone:{event.id}")]
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    await bot.send_message(user_id, "\n".join(lines), reply_markup=markup)


async def handle_events(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)

    if len(parts) == 2:
        text = parts[1]
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                day = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                day = None
        if day is None:
            await bot.send_message(message.chat.id, "Usage: /events YYYY-MM-DD")
            return
    else:
        day = datetime.now(tz).date()

    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return

    text, markup = await build_events_message(db, day, tz)
    await bot.send_message(message.chat.id, text, reply_markup=markup)


async def handle_ask_4o(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2:
        await bot.send_message(message.chat.id, "Usage: /ask4o <text>")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    try:
        answer = await ask_4o(parts[1])
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    await bot.send_message(message.chat.id, answer)


async def handle_exhibitions(message: types.Message, db: Database, bot: Bot):
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)

    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return

    text, markup = await build_exhibitions_message(db, tz)
    await bot.send_message(message.chat.id, text, reply_markup=markup)


async def handle_months(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(MonthPage).order_by(MonthPage.month))
        pages = result.scalars().all()
    lines = ["Months:"]
    for p in pages:
        lines.append(f"{p.month}: {p.url}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_edit_message(message: types.Message, db: Database, bot: Bot):
    state = editing_sessions.get(message.from_user.id)
    if not state:
        return
    eid, field = state
    if field is None:
        return
    value = message.text.strip()
    async with db.get_session() as session:
        event = await session.get(Event, eid)
        if not event:
            await bot.send_message(message.chat.id, "Event not found")
            del editing_sessions[message.from_user.id]
            return
        old_month = event.date.split("..", 1)[0][:7]
        if field in {"ticket_price_min", "ticket_price_max"}:
            try:
                setattr(event, field, int(value))
            except ValueError:
                await bot.send_message(message.chat.id, "Invalid number")
                return
        else:
            setattr(event, field, value)
        await session.commit()
        new_month = event.date.split("..", 1)[0][:7]
    await sync_month_page(db, old_month)
    if new_month != old_month:
        await sync_month_page(db, new_month)
    editing_sessions[message.from_user.id] = (eid, None)
    await show_edit_menu(message.from_user.id, event, bot)


processed_media_groups: set[str] = set()


async def handle_forwarded(message: types.Message, db: Database, bot: Bot):
    text = message.text or message.caption
    if message.media_group_id:
        if message.media_group_id in processed_media_groups:
            return
        if not text:
            # wait for the part of the album that contains the caption
            return
        processed_media_groups.add(message.media_group_id)
    if not text:
        return
    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            return
    link = None
    if message.forward_from_chat and message.forward_from_message_id:
        chat = message.forward_from_chat
        msg_id = message.forward_from_message_id
        async with db.get_session() as session:
            ch = await session.get(Channel, chat.id)
            allowed = ch.is_registered if ch else False
        if allowed:
            if chat.username:
                link = f"https://t.me/{chat.username}/{msg_id}"
            else:
                cid = str(chat.id)
                if cid.startswith("-100"):
                    cid = cid[4:]
                else:
                    cid = cid.lstrip("-")
                link = f"https://t.me/c/{cid}/{msg_id}"
    media = None
    # Skip downloading attachments to avoid large file transfers

    results = await add_events_from_text(
        db,
        text,
        link,
        message.html_text or message.caption_html,
        media,
    )
    for saved, added, lines, status in results:
        markup = None
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[[
                    types.InlineKeyboardButton(
                        text="\u2753 Это бесплатное мероприятие", callback_data=f"markfree:{saved.id}"
                    )
                ]]
            )
        await bot.send_message(
            message.chat.id,
            f"Event {status}\n" + "\n".join(lines),
            reply_markup=markup,
        )


async def telegraph_test():
    token = get_telegraph_token()
    if not token:
        print("Unable to obtain Telegraph token")
        return
    tg = Telegraph(access_token=token)
    page = await asyncio.to_thread(
        tg.create_page, "Test Page", html_content="<p>test</p>"
    )
    logging.info("Created %s", page["url"])
    print("Created", page["url"])
    await asyncio.to_thread(
        tg.edit_page, page["path"], title="Test Page", html_content="<p>updated</p>"
    )
    logging.info("Edited %s", page["url"])
    print("Edited", page["url"])


async def update_source_page(path: str, title: str, new_html: str):
    """Append text to an existing Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    try:
        logging.info("Fetching telegraph page %s", path)
        page = await asyncio.to_thread(
            tg.get_page, path, return_html=True
        )
        html_content = page.get("content") or page.get("content_html") or ""
        cleaned = re.sub(r"</?tg-emoji[^>]*>", "", new_html)
        cleaned = cleaned.replace("\U0001F193\U0001F193\U0001F193\U0001F193", "Бесплатно")
        html_content += f"<p>{CONTENT_SEPARATOR}</p><p>" + cleaned.replace("\n", "<br/>") + "</p>"
        logging.info("Editing telegraph page %s", path)
        await asyncio.to_thread(
            tg.edit_page, path, title=title, html_content=html_content
        )
        logging.info("Updated telegraph page %s", path)
    except Exception as e:
        logging.error("Failed to update telegraph page: %s", e)


async def create_source_page(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: tuple[bytes, str] | None = None,
) -> tuple[str, str] | None:
    """Create a Telegraph page with the original event text."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return None
    tg = Telegraph(access_token=token)
    html_content = ""

    def strip_title(line_text: str) -> str:
        lines = line_text.splitlines()
        if lines and lines[0].strip() == title.strip():
            return "\n".join(lines[1:]).lstrip()
        return line_text
    # Media uploads to Telegraph are flaky and consume bandwidth.
    # Skip uploading files for now to keep requests lightweight.
    if media:
        logging.info("Media upload skipped for telegraph page")

    if source_url:
        html_content += (
            f'<p><a href="{html.escape(source_url)}"><strong>'
            f"{html.escape(title)}</strong></a></p>"
        )
    else:
        html_content += f"<p><strong>{html.escape(title)}</strong></p>"

    if html_text:
        html_text = strip_title(html_text)
        cleaned = re.sub(r"</?tg-emoji[^>]*>", "", html_text)
        cleaned = cleaned.replace("\U0001F193\U0001F193\U0001F193\U0001F193", "Бесплатно")
        html_content += f"<p>{cleaned.replace('\n', '<br/>')}</p>"
    else:
        clean_text = strip_title(text)
        clean_text = clean_text.replace("\U0001F193\U0001F193\U0001F193\U0001F193", "Бесплатно")
        paragraphs = [f"<p>{html.escape(line)}</p>" for line in clean_text.splitlines()]
        html_content += "".join(paragraphs)
    try:
        page = await asyncio.to_thread(
            tg.create_page, title, html_content=html_content
        )
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    logging.info("Created telegraph page %s", page.get("url"))
    return page.get("url"), page.get("path")


def create_app() -> web.Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    webhook = os.getenv("WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("WEBHOOK_URL is missing")

    bot = Bot(token)
    logging.info("DB_PATH=%s", DB_PATH)
    logging.info("FOUR_O_TOKEN found: %s", bool(os.getenv("FOUR_O_TOKEN")))
    dp = Dispatcher()
    db = Database(DB_PATH)

    async def start_wrapper(message: types.Message):
        await handle_start(message, db, bot)

    async def register_wrapper(message: types.Message):
        await handle_register(message, db, bot)

    async def requests_wrapper(message: types.Message):
        await handle_requests(message, db, bot)

    async def tz_wrapper(message: types.Message):
        await handle_tz(message, db, bot)

    async def callback_wrapper(callback: types.CallbackQuery):
        await process_request(callback, db, bot)

    async def add_event_wrapper(message: types.Message):
        await handle_add_event(message, db, bot)

    async def add_event_raw_wrapper(message: types.Message):
        await handle_add_event_raw(message, db, bot)

    async def ask_4o_wrapper(message: types.Message):
        await handle_ask_4o(message, db, bot)

    async def list_events_wrapper(message: types.Message):
        await handle_events(message, db, bot)

    async def set_channel_wrapper(message: types.Message):
        await handle_set_channel(message, db, bot)

    async def channels_wrapper(message: types.Message):
        await handle_channels(message, db, bot)

    async def exhibitions_wrapper(message: types.Message):
        await handle_exhibitions(message, db, bot)

    async def months_wrapper(message: types.Message):
        await handle_months(message, db, bot)

    async def edit_message_wrapper(message: types.Message):
        await handle_edit_message(message, db, bot)

    async def forward_wrapper(message: types.Message):
        await handle_forwarded(message, db, bot)

    dp.message.register(start_wrapper, Command("start"))
    dp.message.register(register_wrapper, Command("register"))
    dp.message.register(requests_wrapper, Command("requests"))
    dp.callback_query.register(
        callback_wrapper,
        lambda c: c.data.startswith("approve")
        or c.data.startswith("reject")
        or c.data.startswith("del:")
        or c.data.startswith("nav:")
        or c.data.startswith("edit:")
        or c.data.startswith("editfield:")
        or c.data.startswith("editdone:")
        or c.data.startswith("unset:")
        or c.data.startswith("set:")
        or c.data.startswith("togglefree:")
        or c.data.startswith("markfree:"),
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(add_event_wrapper, Command("addevent"))
    dp.message.register(add_event_raw_wrapper, Command("addevent_raw"))
    dp.message.register(ask_4o_wrapper, Command("ask4o"))
    dp.message.register(list_events_wrapper, Command("events"))
    dp.message.register(set_channel_wrapper, Command("setchannel"))
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(months_wrapper, Command("months"))
    dp.message.register(edit_message_wrapper, lambda m: m.from_user.id in editing_sessions)
    dp.message.register(forward_wrapper, lambda m: bool(m.forward_date))
    dp.my_chat_member.register(partial(handle_my_chat_member, db=db))

    app = web.Application()
    SimpleRequestHandler(dp, bot).register(app, path="/webhook")
    setup_application(app, dp, bot=bot)

    async def on_startup(app: web.Application):
        logging.info("Initializing database")
        await db.init()
        hook = webhook.rstrip("/") + "/webhook"
        logging.info("Setting webhook to %s", hook)
        await bot.set_webhook(
            hook,
            allowed_updates=["message", "callback_query", "my_chat_member"],
        )

    async def on_shutdown(app: web.Application):
        await bot.session.close()

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
