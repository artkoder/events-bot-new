import logging
import os
from datetime import date, datetime, timedelta, timezone, time
from typing import Optional, Tuple, Iterable

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web, ClientSession, FormData
import imghdr
from difflib import SequenceMatcher
import json
import re
from telegraph import Telegraph
from functools import partial
import asyncio
import contextlib
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
# user_id -> channel_id for daily time editing
daily_time_sessions: dict[int, int] = {}

# toggle for uploading images to catbox
CATBOX_ENABLED: bool = False


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
    daily_time: Optional[str] = None
    last_daily: Optional[str] = None


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
    silent: bool = False
    telegraph_path: Optional[str] = None
    source_text: str
    telegraph_url: Optional[str] = None
    source_post_url: Optional[str] = None
    photo_count: int = 0
    added_at: datetime = Field(default_factory=datetime.utcnow)


class MonthPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    month: str = Field(primary_key=True)
    url: str
    path: str


class WeekendPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    start: str = Field(primary_key=True)
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
            if "silent" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN silent BOOLEAN DEFAULT 0"
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
                await conn.exec_driver_sql("ALTER TABLE event ADD COLUMN emoji VARCHAR")
            if "end_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN end_date VARCHAR"
                )
            if "added_at" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN added_at VARCHAR"
                )
            if "photo_count" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN photo_count INTEGER DEFAULT 0"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(channel)")
            cols = [r[1] for r in result.fetchall()]
            if "daily_time" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN daily_time VARCHAR"
                )
            if "last_daily" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN last_daily VARCHAR"
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


async def get_catbox_enabled(db: Database) -> bool:
    async with db.get_session() as session:
        setting = await session.get(Setting, "catbox_enabled")
        return setting.value == "1" if setting else False


async def set_catbox_enabled(db: Database, value: bool):
    async with db.get_session() as session:
        setting = await session.get(Setting, "catbox_enabled")
        if setting:
            setting.value = "1" if value else "0"
        else:
            setting = Setting(key="catbox_enabled", value="1" if value else "0")
            session.add(setting)
        await session.commit()
    global CATBOX_ENABLED
    CATBOX_ENABLED = value


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


async def extract_images(message: types.Message, bot: Bot) -> list[tuple[bytes, str]]:
    """Download up to three images from the message."""
    images: list[tuple[bytes, str]] = []
    if message.photo:
        bio = BytesIO()
        await bot.download(message.photo[-1].file_id, destination=bio)
        images.append((bio.getvalue(), "photo.jpg"))
    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        bio = BytesIO()
        await bot.download(message.document.file_id, destination=bio)
        name = message.document.file_name or "image.jpg"
        images.append((bio.getvalue(), name))
    return images[:3]


def normalize_hashtag_dates(text: str) -> str:
    """Replace hashtags like '#1_августа' with '1 августа'."""
    pattern = re.compile(
        r"#(\d{1,2})_(%s)" % "|".join(MONTHS)
    )
    return re.sub(pattern, lambda m: f"{m.group(1)} {m.group(2)}", text)


def strip_city_from_address(address: str | None, city: str | None) -> str | None:
    """Remove the city name from the end of the address if duplicated."""
    if not address or not city:
        return address
    city_clean = city.lstrip("#").strip().lower()
    addr = address.strip()
    if addr.lower().endswith(city_clean):
        addr = re.sub(r",?\s*#?%s$" % re.escape(city_clean), "", addr, flags=re.IGNORECASE)
        addr = addr.rstrip(", ")
    return addr


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
            locations = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
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
        data.get("choices", [{}])[0].get("message", {}).get("content", "{}").strip()
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
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


async def check_duplicate_via_4o(ev: Event, new: Event) -> Tuple[bool, str, str]:
    """Ask the LLM whether two events are duplicates."""
    prompt = (
        "Existing event:\n"
        f"Title: {ev.title}\nDescription: {ev.description}\nLocation: {ev.location_name} {ev.location_address}\n"
        "New event:\n"
        f"Title: {new.title}\nDescription: {new.description}\nLocation: {new.location_name} {new.location_address}\n"
        'Are these the same event? Respond with JSON {"duplicate": true|false, "title": "", "short_description": ""}.'
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
        logging.info(
            "Created Telegraph account; token stored at %s", TELEGRAPH_TOKEN_FILE
        )
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
            w_start = (
                weekend_start_for_date(datetime.fromisoformat(event.date).date())
                if event
                else None
            )
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())
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
            w_start = weekend_start_for_date(datetime.fromisoformat(event.date).date())
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())
        async with db.get_session() as session:
            event = await session.get(Event, eid)
        if event:
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer()
    elif data.startswith("togglesilent:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            event = await session.get(Event, eid)
            if event:
                event.silent = not event.silent
                await session.commit()
                logging.info("togglesilent: event %s set to %s", eid, event.silent)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
            w_start = weekend_start_for_date(datetime.fromisoformat(event.date).date())
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text=(
                            "\U0001f910 Тихий режим"
                            if event and event.silent
                            else "\U0001f6a9 Переключить на тихий режим"
                        ),
                        callback_data=f"togglesilent:{eid}",
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
            logging.error("failed to update silent button: %s", e)
        await callback.answer("Toggled")
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
            w_start = weekend_start_for_date(datetime.fromisoformat(event.date).date())
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="\u2705 Бесплатное мероприятие",
                        callback_data=f"togglefree:{eid}",
                    ),
                    types.InlineKeyboardButton(
                        text="\U0001f6a9 Переключить на тихий режим",
                        callback_data=f"togglesilent:{eid}",
                    ),
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
    elif data.startswith("dailyset:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            ch = await session.get(Channel, cid)
            if ch and ch.is_admin:
                ch.daily_time = "08:00"
                await session.commit()
        await send_regdaily_list(callback.message, db, bot, edit=True)
        await callback.answer("Registered")
    elif data.startswith("dailyunset:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            ch = await session.get(Channel, cid)
            if ch:
                ch.daily_time = None
                await session.commit()
        await send_daily_list(callback.message, db, bot, edit=True)
        await callback.answer("Removed")
    elif data.startswith("dailytime:"):
        cid = int(data.split(":")[1])
        daily_time_sessions[callback.from_user.id] = cid
        await callback.message.answer("Send new time HH:MM")
        await callback.answer()
    elif data.startswith("dailysend:"):
        cid = int(data.split(":")[1])
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        await send_daily_announcement(db, bot, cid, tz, record=False)
        await callback.answer("Sent")


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


async def handle_images(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    new_value = not CATBOX_ENABLED
    await set_catbox_enabled(db, new_value)
    status = "enabled" if new_value else "disabled"
    await bot.send_message(message.chat.id, f"Image uploads {status}")


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


async def send_channels_list(
    message: types.Message, db: Database, bot: Bot, edit: bool = False
):
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
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text="Cancel", callback_data=f"unset:{ch.channel_id}"
                    )
                ]
            )
        else:
            lines.append(name)
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_setchannel_list(
    message: types.Message, db: Database, bot: Bot, edit: bool = False
):
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
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text=name, callback_data=f"set:{ch.channel_id}"
                )
            ]
        )
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_regdaily_list(
    message: types.Message, db: Database, bot: Bot, edit: bool = False
):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(
            select(Channel).where(
                Channel.is_admin.is_(True), Channel.daily_time.is_(None)
            )
        )
        channels = result.scalars().all()
    lines = []
    keyboard = []
    for ch in channels:
        name = ch.title or ch.username or str(ch.channel_id)
        lines.append(name)
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text=name, callback_data=f"dailyset:{ch.channel_id}"
                )
            ]
        )
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_daily_list(
    message: types.Message, db: Database, bot: Bot, edit: bool = False
):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(
            select(Channel).where(Channel.daily_time.is_not(None))
        )
        channels = result.scalars().all()
    lines = []
    keyboard = []
    for ch in channels:
        name = ch.title or ch.username or str(ch.channel_id)
        t = ch.daily_time or "?"
        lines.append(f"{name} {t}")
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Cancel", callback_data=f"dailyunset:{ch.channel_id}"
                ),
                types.InlineKeyboardButton(
                    text="Time", callback_data=f"dailytime:{ch.channel_id}"
                ),
                types.InlineKeyboardButton(
                    text="Test", callback_data=f"dailysend:{ch.channel_id}"
                ),
            ]
        )
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


async def handle_regdailychannels(message: types.Message, db: Database, bot: Bot):
    await send_regdaily_list(message, db, bot, edit=False)


async def handle_daily(message: types.Message, db: Database, bot: Bot):
    await send_daily_list(message, db, bot, edit=False)


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
        loc_ratio = SequenceMatcher(
            None, ev.location_name.lower(), new.location_name.lower()
        ).ratio()
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
        if loc_ratio >= 0.4 or (ev.location_address or "") == (
            new.location_address or ""
        ):
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
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
) -> list[tuple[Event, bool, list[str], str]]:
    try:
        parsed = await parse_event_via_4o(text)
    except Exception as e:
        logging.error("LLM error: %s", e)
        return []

    results: list[tuple[Event, bool, list[str], str]] = []
    first = True
    links_iter = iter(extract_links_from_html(html_text) if html_text else [])
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

        addr = data.get("location_address")
        city = data.get("city")
        addr = strip_city_from_address(addr, city)

        base_event = Event(
            title=data.get("title", ""),
            description=data.get("short_description", ""),
            festival=data.get("festival") or None,
            date=date_str,
            time=data.get("time", ""),
            location_name=data.get("location_name", ""),
            location_address=addr,
            city=city,
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

        events_to_add = [base_event]
        if (
            base_event.event_type != "выставка"
            and base_event.end_date
            and base_event.end_date != base_event.date
        ):
            try:
                start_dt = date.fromisoformat(base_event.date)
                end_dt = date.fromisoformat(base_event.end_date)
            except ValueError:
                start_dt = end_dt = None
            if start_dt and end_dt and end_dt > start_dt:
                events_to_add = []
                for i in range((end_dt - start_dt).days + 1):
                    day = start_dt + timedelta(days=i)
                    copy_e = Event(**base_event.model_dump(exclude={"id", "added_at"}))
                    copy_e.date = day.isoformat()
                    copy_e.end_date = None
                    events_to_add.append(copy_e)

        for event in events_to_add:
            if not is_valid_url(event.ticket_link):
                try:
                    extracted = next(links_iter)
                except StopIteration:
                    extracted = None
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
            upload_info = ""
            photo_count = saved.photo_count
            if saved.telegraph_url and saved.telegraph_path:
                upload_info, added_count = await update_source_page(
                    saved.telegraph_path,
                    saved.title or "Event",
                    html_text or text,
                    media_arg,
                )
                if added_count:
                    photo_count += added_count
                    async with db.get_session() as session:
                        saved.photo_count = photo_count
                        session.add(saved)
                        await session.commit()
            else:
                res = await create_source_page(
                    saved.title or "Event",
                    saved.source_text,
                    source_link,
                    html_text,
                    media_arg,
                )
                if res:
                    if len(res) == 4:
                        url, path, upload_info, photo_count = res
                    elif len(res) == 3:
                        url, path, upload_info = res
                        photo_count = 0
                    else:
                        url, path = res
                        upload_info = ""
                        photo_count = 0
                    async with db.get_session() as session:
                        saved.telegraph_url = url
                        saved.telegraph_path = path
                        saved.photo_count = photo_count
                        session.add(saved)
                        await session.commit()
            await sync_month_page(db, saved.date[:7])
            w_start = weekend_start_for_date(datetime.fromisoformat(saved.date).date())
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())

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
            if upload_info:
                lines.append(f"catbox: {upload_info}")
            status = "added" if added else "updated"
            results.append((saved, added, lines, status))
            first = False
    return results


async def handle_add_event(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or message.caption or "").split(maxsplit=1)
    if len(parts) != 2:
        await bot.send_message(message.chat.id, "Usage: /addevent <text>")
        return
    images = await extract_images(message, bot)
    media = images if images else None
    html_text = message.html_text or message.caption_html
    if html_text and html_text.startswith("/addevent"):
        html_text = html_text[len("/addevent") :].lstrip()
    results = await add_events_from_text(
        db,
        parts[1],
        None,
        html_text,
        media,
    )
    if not results:
        await bot.send_message(message.chat.id, "LLM error")
        return
    for saved, added, lines, status in results:
        btns = []
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            btns.append(
                types.InlineKeyboardButton(
                    text="\u2753 Это бесплатное мероприятие",
                    callback_data=f"markfree:{saved.id}",
                )
            )
        btns.append(
            types.InlineKeyboardButton(
                text="\U0001f6a9 Переключить на тихий режим",
                callback_data=f"togglesilent:{saved.id}",
            )
        )
        markup = types.InlineKeyboardMarkup(inline_keyboard=[btns])
        await bot.send_message(
            message.chat.id,
            f"Event {status}\n" + "\n".join(lines),
            reply_markup=markup,
        )


async def handle_add_event_raw(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or message.caption or "").split(maxsplit=1)
    if len(parts) != 2 or "|" not in parts[1]:
        await bot.send_message(
            message.chat.id, "Usage: /addevent_raw title|date|time|location"
        )
        return
    title, date, time, location = (p.strip() for p in parts[1].split("|", 3))
    images = await extract_images(message, bot)
    media = images if images else None

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

    html_text = message.html_text or message.caption_html
    if html_text and html_text.startswith("/addevent_raw"):
        html_text = html_text[len("/addevent_raw") :].lstrip()
    res = await create_source_page(
        event.title or "Event",
        event.source_text,
        None,
        html_text or event.source_text,
        media,
    )
    upload_info = ""
    photo_count = 0
    if res:
        if len(res) == 4:
            url, path, upload_info, photo_count = res
        elif len(res) == 3:
            url, path, upload_info = res
            photo_count = 0
        else:
            url, path = res
            upload_info = ""
            photo_count = 0
        async with db.get_session() as session:
            event.telegraph_url = url
            event.telegraph_path = path
            event.photo_count = photo_count
            session.add(event)
            await session.commit()
    await sync_month_page(db, event.date[:7])
    w_start = weekend_start_for_date(datetime.fromisoformat(event.date).date())
    if w_start:
        await sync_weekend_page(db, w_start.isoformat())
    lines = [
        f"title: {event.title}",
        f"date: {event.date}",
        f"time: {event.time}",
        f"location_name: {event.location_name}",
    ]
    if event.telegraph_url:
        lines.append(f"telegraph: {event.telegraph_url}")
    if upload_info:
        lines.append(f"catbox: {upload_info}")
    status = "added" if added else "updated"
    btns = []
    if (
        not event.is_free
        and event.ticket_price_min is None
        and event.ticket_price_max is None
    ):
        btns.append(
            types.InlineKeyboardButton(
                text="\u2753 Это бесплатное мероприятие",
                callback_data=f"markfree:{event.id}",
            )
        )
    btns.append(
        types.InlineKeyboardButton(
            text="\U0001f6a9 Переключить на тихий режим",
            callback_data=f"togglesilent:{event.id}",
        )
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=[btns])
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

DAYS_OF_WEEK = [
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
]


def format_day_pretty(day: date) -> str:
    return f"{day.day} {MONTHS[day.month - 1]}"


def format_weekend_range(saturday: date) -> str:
    """Return human-friendly weekend range like '12–13 июля'."""
    sunday = saturday + timedelta(days=1)
    if saturday.month == sunday.month:
        return f"{saturday.day}\u2013{sunday.day} {MONTHS[saturday.month - 1]}"
    return (
        f"{saturday.day} {MONTHS[saturday.month - 1]} \u2013 "
        f"{sunday.day} {MONTHS[sunday.month - 1]}"
    )


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

# month names in nominative case for navigation links
MONTHS_NOM = [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]


def month_name_prepositional(month: str) -> str:
    y, m = month.split("-")
    return f"{MONTHS_PREP[int(m) - 1]} {y}"


def month_name_nominative(month: str) -> str:
    """Return month name in nominative case, add year if different from current."""
    y, m = month.split("-")
    name = MONTHS_NOM[int(m) - 1]
    if int(y) != date.today().year:
        return f"{name} {y}"
    return name


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


def extract_links_from_html(html_text: str) -> list[str]:
    """Return all registration or ticket links in order of appearance."""
    pattern = re.compile(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(html_text))
    lower_html = html_text.lower()

    def qualifies(label: str, start: int, end: int) -> bool:
        text = label.lower()
        if any(word in text for word in ["регистра", "ticket", "билет"]):
            return True
        context_before = lower_html[max(0, start - 60) : start]
        context_after = lower_html[end : end + 60]
        return "регистра" in context_before or "регистра" in context_after or "билет" in context_before or "билет" in context_after

    prioritized: list[tuple[int, str]] = []
    others: list[tuple[int, str]] = []
    for m in matches:
        href, label = m.group(1), m.group(2)
        if qualifies(label, *m.span()):
            prioritized.append((m.start(), href))
        else:
            others.append((m.start(), href))

    prioritized.sort(key=lambda x: x[0])
    others.sort(key=lambda x: x[0])
    links = [h for _, h in prioritized]
    links.extend(h for _, h in others)
    return links


def is_valid_url(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.match(r"https?://", text))


def is_recent(e: Event) -> bool:
    if e.added_at is None:
        return False
    now = datetime.utcnow()
    start = datetime.combine(now.date() - timedelta(days=1), datetime.min.time())
    return e.added_at >= start and not e.silent


def format_event_md(e: Event) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "
    lines = [f"{prefix}{emoji_part}{e.title}".strip(), e.description.strip()]
    if e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link:
            txt += f" [по регистрации]({e.ticket_link})"
        lines.append(txt)
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        else:
            price = str(e.ticket_price_min or e.ticket_price_max or "")
        lines.append(f"[Билеты в источнике]({e.ticket_link}) {price}".strip())
    elif e.ticket_link:
        lines.append(f"[по регистрации]({e.ticket_link})")
    else:
        if (
            e.ticket_price_min is not None
            and e.ticket_price_max is not None
            and e.ticket_price_min != e.ticket_price_max
        ):
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
        cam = "\U0001f4f8" * max(0, e.photo_count)
        prefix = f"{cam} " if cam else ""
        lines.append(f"{prefix}[подробнее]({e.telegraph_url})")
    loc = e.location_name
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {addr}"
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


def format_event_daily(e: Event, highlight: bool = False) -> str:
    """Return HTML-formatted text for a daily announcement item."""
    prefix = ""
    if highlight:
        prefix += "\U0001f449 "
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "

    title = html.escape(e.title)
    if e.source_post_url:
        title = f'<a href="{html.escape(e.source_post_url)}">{title}</a>'
    title = f"<b>{prefix}{emoji_part}{title}</b>".strip()
    lines = [title, html.escape(e.description.strip())]

    if e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link:
            txt += f' <a href="{html.escape(e.ticket_link)}">по регистрации</a>'
        lines.append(txt)
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        else:
            price = str(e.ticket_price_min or e.ticket_price_max or "")
        lines.append(
            f'<a href="{html.escape(e.ticket_link)}">Билеты в источнике</a> {price}'.strip()
        )
    elif e.ticket_link:
        lines.append(f'<a href="{html.escape(e.ticket_link)}">по регистрации</a>')
    else:
        price = ""
        if (
            e.ticket_price_min is not None
            and e.ticket_price_max is not None
            and e.ticket_price_min != e.ticket_price_max
        ):
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        elif e.ticket_price_min is not None:
            price = str(e.ticket_price_min)
        elif e.ticket_price_max is not None:
            price = str(e.ticket_price_max)
        if price:
            lines.append(f"Билеты {price}")

    loc = html.escape(e.location_name)
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {html.escape(addr)}"
    if e.city:
        loc += f", #{html.escape(e.city)}"
    date_part = e.date.split("..", 1)[0]
    try:
        day = format_day_pretty(datetime.fromisoformat(date_part).date())
    except ValueError:
        logging.error("Invalid event date: %s", e.date)
        day = e.date
    lines.append(f"<i>{day} {e.time} {loc}</i>")

    return "\n".join(lines)


def format_exhibition_md(e: Event) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001f6a9 "
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
    elif (
        e.ticket_price_min is not None
        and e.ticket_price_max is not None
        and e.ticket_price_min != e.ticket_price_max
    ):
        lines.append(f"Билеты от {e.ticket_price_min} до {e.ticket_price_max}")
    elif e.ticket_price_min is not None:
        lines.append(f"Билеты {e.ticket_price_min}")
    elif e.ticket_price_max is not None:
        lines.append(f"Билеты {e.ticket_price_max}")
    if e.telegraph_url:
        cam = "\U0001f4f8" * max(0, e.photo_count)
        prefix = f"{cam} " if cam else ""
        lines.append(f"{prefix}[подробнее]({e.telegraph_url})")
    loc = e.location_name
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {addr}"
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
        nodes.append("\U0001f6a9 ")
    if e.emoji and not e.title.strip().startswith(e.emoji):
        nodes.append(f"{e.emoji} ")
    title_text = e.title
    if e.source_post_url:
        nodes.append(
            {"tag": "a", "attrs": {"href": e.source_post_url}, "children": [title_text]}
        )
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
    nodes.append({"tag": "p", "children": ["\u00a0"]})
    return nodes


def exhibition_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001f6a9 ")
    if e.emoji and not e.title.strip().startswith(e.emoji):
        nodes.append(f"{e.emoji} ")
    title_text = e.title
    if e.source_post_url:
        nodes.append(
            {"tag": "a", "attrs": {"href": e.source_post_url}, "children": [title_text]}
        )
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
    nodes.append({"tag": "p", "children": ["\u00a0"]})
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
                Event.event_type == "выставка",
            )
            .order_by(Event.date)
        )
        exhibitions = ex_result.scalars().all()

        result_nav = await session.execute(select(MonthPage).order_by(MonthPage.month))
        nav_pages = result_nav.scalars().all()

    today = date.today()
    events = [
        e
        for e in events
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
        and not (e.event_type == "выставка" and e.date < today.isoformat())
    ]
    exhibitions = [
        e
        for e in exhibitions
        if e.end_date
        and e.end_date >= today.isoformat()
        and e.date <= today.isoformat()
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
    intro = f"Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_name_prepositional(month)} — от лекций и концертов до культурных шоу. "
    intro_nodes = [
        intro,
        {
            "tag": "a",
            "attrs": {"href": "https://t.me/kenigevents"},
            "children": ["Полюбить Калининград Анонсы"],
        },
    ]
    content.append({"tag": "p", "children": intro_nodes})

    for day in sorted(by_day):
        if day.weekday() == 5:
            content.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
        elif day.weekday() == 6:
            content.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
        content.append(
            {"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(day)} 🟥🟥🟥"]}
        )
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        for ev in by_day[day]:
            content.extend(event_to_nodes(ev))

    today_month = date.today().strftime("%Y-%m")
    future_pages = [p for p in nav_pages if p.month >= today_month]
    if future_pages:
        nav_children = []
        for idx, p in enumerate(future_pages):
            name = month_name_nominative(p.month)
            if p.month == month:
                nav_children.append(name)
            else:
                nav_children.append(
                    {"tag": "a", "attrs": {"href": p.url}, "children": [name]}
                )
            if idx < len(future_pages) - 1:
                nav_children.append(" ")
        content.append({"tag": "br"})
        content.append({"tag": "h4", "children": nav_children})

    if exhibitions:
        content.append({"tag": "h3", "children": ["Постоянные выставки"]})
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        for ev in exhibitions:
            content.extend(exhibition_to_nodes(ev))

    title = f"События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
    return title, content


async def sync_month_page(db: Database, month: str, update_links: bool = True):
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    async with db.get_session() as session:
        page = await session.get(MonthPage, month)
        try:
            created = False
            if not page:
                title, content = await build_month_page_content(db, month)
                data = await asyncio.to_thread(tg.create_page, title, content=content)
                page = MonthPage(
                    month=month, url=data.get("url"), path=data.get("path")
                )
                session.add(page)
                await session.commit()
                created = True

            title, content = await build_month_page_content(db, month)
            await asyncio.to_thread(
                tg.edit_page, page.path, title=title, content=content
            )
            logging.info("%s month page %s", "Created" if created else "Edited", month)
            await session.commit()
        except Exception as e:
            logging.error("Failed to sync month page %s: %s", month, e)

    if update_links:
        async with db.get_session() as session:
            result = await session.execute(select(MonthPage).order_by(MonthPage.month))
            months = result.scalars().all()
        for p in months:
            if p.month != month:
                await sync_month_page(db, p.month, update_links=False)


def weekend_start_for_date(d: date) -> date | None:
    if d.weekday() == 5:
        return d
    if d.weekday() == 6:
        return d - timedelta(days=1)
    return None


def next_weekend_start(d: date) -> date:
    w = weekend_start_for_date(d)
    if w and d <= w:
        return w
    days_ahead = (5 - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)


async def build_weekend_page_content(db: Database, start: str) -> tuple[str, list]:
    saturday = date.fromisoformat(start)
    sunday = saturday + timedelta(days=1)
    days = [saturday, sunday]
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(Event.date.in_([d.isoformat() for d in days]))
            .order_by(Event.date, Event.time)
        )
        events = result.scalars().all()

        ex_res = await session.execute(
            select(Event)
            .where(
                Event.event_type == "выставка",
                Event.end_date.is_not(None),
                Event.date <= sunday.isoformat(),
                Event.end_date >= saturday.isoformat(),
            )
            .order_by(Event.date)
        )
        exhibitions = ex_res.scalars().all()

        res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
        weekend_pages = res_w.scalars().all()
        res_m = await session.execute(select(MonthPage).order_by(MonthPage.month))
        month_pages = res_m.scalars().all()

    today = date.today()
    events = [
        e
        for e in events
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
    ]

    by_day: dict[date, list[Event]] = {}
    for e in events:
        d = date.fromisoformat(e.date)
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    content.append(
        {
            "tag": "p",
            "children": [
                "Проведите выходные ярко: все события Калининградской области и 39 региона — концерты, кино под открытым небом и гастрономические фестивали."
            ],
        }
    )

    for d in days:
        if d not in by_day:
            continue
        if d.weekday() == 5:
            content.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
        elif d.weekday() == 6:
            content.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
        content.append(
            {"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]}
        )
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        for ev in by_day[d]:
            content.extend(event_to_nodes(ev))

    weekend_nav: list[dict] = []
    future_weekends = [w for w in weekend_pages if w.start >= start]
    if future_weekends:
        nav_children = []
        for idx, w in enumerate(future_weekends):
            s = date.fromisoformat(w.start)
            label = format_weekend_range(s)
            if w.start == start:
                nav_children.append(label)
            else:
                nav_children.append(
                    {"tag": "a", "attrs": {"href": w.url}, "children": [label]}
                )
            if idx < len(future_weekends) - 1:
                nav_children.append(" ")
        weekend_nav = [{"tag": "br"}, {"tag": "h4", "children": nav_children}]
        content.extend(weekend_nav)

    month_nav: list[dict] = []
    cur_month = start[:7]
    today_month = date.today().strftime("%Y-%m")
    future_months = [m for m in month_pages if m.month >= today_month]
    if future_months:
        nav_children = []
        for idx, p in enumerate(future_months):
            name = month_name_nominative(p.month)
            nav_children.append({"tag": "a", "attrs": {"href": p.url}, "children": [name]})
            if idx < len(future_months) - 1:
                nav_children.append(" ")
        month_nav = [{"tag": "br"}, {"tag": "h4", "children": nav_children}]
        content.extend(month_nav)

    if exhibitions:
        if weekend_nav or month_nav:
            content.append({"tag": "br"})
            content.append({"tag": "p", "children": ["\u00a0"]})
        content.append({"tag": "h3", "children": ["Постоянные выставки"]})
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        for ev in exhibitions:
            content.extend(exhibition_to_nodes(ev))

    if weekend_nav:
        content.extend(weekend_nav)
    if month_nav:
        content.extend(month_nav)

    title = (
        "Чем заняться на выходных в Калининградской области "
        f"{format_weekend_range(saturday)}"
    )
    return title, content


async def sync_weekend_page(db: Database, start: str, update_links: bool = True):
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    async with db.get_session() as session:
        page = await session.get(WeekendPage, start)
        try:
            created = False
            if not page:
                # Create a placeholder page to obtain path and URL
                title, content = await build_weekend_page_content(db, start)
                data = await asyncio.to_thread(tg.create_page, title, content=content)
                page = WeekendPage(
                    start=start, url=data.get("url"), path=data.get("path")
                )
                session.add(page)
                await session.commit()
                created = True

            # Rebuild content including this page in navigation
            title, content = await build_weekend_page_content(db, start)
            await asyncio.to_thread(
                tg.edit_page, page.path, title=title, content=content
            )
            logging.info(
                "%s weekend page %s", "Created" if created else "Edited", start
            )
            await session.commit()
        except Exception as e:
            logging.error("Failed to sync weekend page %s: %s", start, e)

    if update_links:
        async with db.get_session() as session:
            result = await session.execute(
                select(WeekendPage).order_by(WeekendPage.start)
            )
            weekends = result.scalars().all()
        for w in weekends:
            if w.start != start:
                await sync_weekend_page(db, w.start, update_links=False)


async def build_daily_posts(
    db: Database, tz: timezone
) -> list[tuple[str, types.InlineKeyboardMarkup | None]]:
    today = datetime.now(tz).date()
    yesterday_start_local = datetime.combine(
        today - timedelta(days=1), time(0, 0), tz
    )
    yesterday_utc = yesterday_start_local.astimezone(timezone.utc)
    async with db.get_session() as session:
        res_today = await session.execute(
            select(Event).where(Event.date == today.isoformat()).order_by(Event.time)
        )
        events_today = res_today.scalars().all()
        res_new = await session.execute(
            select(Event)
            .where(
                Event.date > today.isoformat(),
                Event.added_at.is_not(None),
                Event.added_at >= yesterday_utc,
                Event.silent.is_(False),
            )
            .order_by(Event.date, Event.time)
        )
        events_new = res_new.scalars().all()

        w_start = next_weekend_start(today)
        wpage = await session.get(WeekendPage, w_start.isoformat())
        cur_month = today.strftime("%Y-%m")
        mp_cur = await session.get(MonthPage, cur_month)
        mp_next = await session.get(MonthPage, next_month(cur_month))

        new_events = (
            await session.execute(
                select(Event).where(
                    Event.added_at.is_not(None),
                    Event.added_at >= yesterday_utc,
                )
            )
        ).scalars().all()

        weekend_count = 0
        if wpage:
            sat = w_start
            sun = w_start + timedelta(days=1)
            for e in new_events:
                if e.date in {sat.isoformat(), sun.isoformat()} or (
                    e.event_type == "выставка"
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                ):
                    weekend_count += 1

        cur_count = 0
        next_count = 0
        for e in new_events:
            m = e.date[:7]
            if m == cur_month:
                cur_count += 1
            elif m == next_month(cur_month):
                next_count += 1

    lines1 = [
        f"<b>АНОНС на {format_day_pretty(today)} {today.year} #ежедневныйанонс</b>",
        DAYS_OF_WEEK[today.weekday()],
        "",
        "<b><i>НЕ ПРОПУСТИТЕ СЕГОДНЯ</i></b>",
    ]
    for e in events_today:
        lines1.append("")
        lines1.append(format_event_daily(e, highlight=True))
    section1 = "\n".join(lines1)

    lines2 = ["<b><i>ДОБАВИЛИ В АНОНС</i></b>"]
    for e in events_new:
        lines2.append("")
        lines2.append(format_event_daily(e))
    section2 = "\n".join(lines2)

    buttons = []
    if wpage:
        sunday = w_start + timedelta(days=1)
        prefix = f"(+{weekend_count}) " if weekend_count else ""
        text = (
            f"{prefix}Мероприятия на выходные {w_start.day} {sunday.day} {MONTHS[w_start.month - 1]}"
        )
        buttons.append(types.InlineKeyboardButton(text=text, url=wpage.url))
    if mp_cur:
        prefix = f"(+{cur_count}) " if cur_count else ""
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{prefix}Мероприятия на {month_name_nominative(cur_month)}",
                url=mp_cur.url,
            )
        )
    if mp_next:
        prefix = f"(+{next_count}) " if next_count else ""
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{prefix}Мероприятия на {month_name_nominative(next_month(cur_month))}",
                url=mp_next.url,
            )
        )
    markup = None
    if buttons:
        markup = types.InlineKeyboardMarkup(inline_keyboard=[[b] for b in buttons])

    combined = section1 + "\n\n\n" + section2
    if len(combined) <= 4096:
        return [(combined, markup)]
    return [(section1, None), (section2, markup)]


async def send_daily_announcement(
    db: Database,
    bot: Bot,
    channel_id: int,
    tz: timezone,
    *,
    record: bool = True,
):
    posts = await build_daily_posts(db, tz)
    for text, markup in posts:
        await bot.send_message(
            channel_id,
            text,
            reply_markup=markup,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    if record:
        async with db.get_session() as session:
            ch = await session.get(Channel, channel_id)
            if ch:
                ch.last_daily = datetime.now(tz).date().isoformat()
                await session.commit()


async def daily_scheduler(db: Database, bot: Bot):
    while True:
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        now_time = now.time().replace(second=0, microsecond=0)
        async with db.get_session() as session:
            result = await session.execute(
                select(Channel).where(Channel.daily_time.is_not(None))
            )
            channels = result.scalars().all()
        for ch in channels:
            if not ch.daily_time:
                continue
            try:
                target_time = datetime.strptime(ch.daily_time, "%H:%M").time()
            except ValueError:
                continue
            if (
                ch.last_daily or ""
            ) != now.date().isoformat() and now_time >= target_time:
                try:
                    await send_daily_announcement(db, bot, ch.channel_id, tz)
                except Exception as e:
                    logging.error("daily send failed for %s: %s", ch.channel_id, e)
        await asyncio.sleep(60)


async def build_events_message(db: Database, target_date: date, tz: timezone):
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                (Event.date == target_date.isoformat())
                | (Event.end_date == target_date.isoformat())
            )
            .order_by(Event.time)
        )
        events = result.scalars().all()

    lines = []
    for e in events:
        prefix = ""
        if e.end_date and e.date == target_date.isoformat():
            prefix = "(Открытие) "
        elif (
            e.end_date
            and e.end_date == target_date.isoformat()
            and e.end_date != e.date
        ):
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
            if (
                e.ticket_price_max is not None
                and e.ticket_price_max != e.ticket_price_min
            ):
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
                text="\u274c", callback_data=f"del:{e.id}:{target_date.isoformat()}"
            ),
            types.InlineKeyboardButton(text="\u270e", callback_data=f"edit:{e.id}"),
        ]
        for e in events
    ]

    today = datetime.now(tz).date()
    prev_day = target_date - timedelta(days=1)
    next_day = target_date + timedelta(days=1)
    row = []
    if target_date > today:
        row.append(
            types.InlineKeyboardButton(
                text="\u25c0", callback_data=f"nav:{prev_day.isoformat()}"
            )
        )
    row.append(
        types.InlineKeyboardButton(
            text="\u25b6", callback_data=f"nav:{next_day.isoformat()}"
        )
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
            if (
                e.ticket_price_max is not None
                and e.ticket_price_max != e.ticket_price_min
            ):
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
            types.InlineKeyboardButton(text="\u274c", callback_data=f"del:{e.id}:exh"),
            types.InlineKeyboardButton(text="\u270e", callback_data=f"edit:{e.id}"),
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
    keyboard.append(
        [
            types.InlineKeyboardButton(
                text=(
                    "\U0001f6a9 Переключить на тихий режим"
                    if not event.silent
                    else "\U0001f910 Тихий режим"
                ),
                callback_data=f"togglesilent:{event.id}",
            )
        ]
    )
    keyboard.append(
        [
            types.InlineKeyboardButton(
                text=("\u2705 Бесплатно" if event.is_free else "\u274c Бесплатно"),
                callback_data=f"togglefree:{event.id}",
            )
        ]
    )
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


async def handle_pages(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(MonthPage).order_by(MonthPage.month))
        months = result.scalars().all()
        res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
        weekends = res_w.scalars().all()
    lines = ["Months:"]
    for p in months:
        lines.append(f"{p.month}: {p.url}")
    if weekends:
        lines.append("")
        lines.append("Weekends:")
        for w in weekends:
            lines.append(f"{w.start}: {w.url}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_edit_message(message: types.Message, db: Database, bot: Bot):
    state = editing_sessions.get(message.from_user.id)
    if not state:
        return
    eid, field = state
    if field is None:
        return
    value = (message.text or message.caption or "").strip()
    if field == "ticket_link" and value in {"", "-"}:
        value = ""
    if not value and field != "ticket_link":
        await bot.send_message(message.chat.id, "No text supplied")
        return
    async with db.get_session() as session:
        event = await session.get(Event, eid)
        if not event:
            await bot.send_message(message.chat.id, "Event not found")
            del editing_sessions[message.from_user.id]
            return
        old_date = event.date.split("..", 1)[0]
        old_month = old_date[:7]
        if field in {"ticket_price_min", "ticket_price_max"}:
            try:
                setattr(event, field, int(value))
            except ValueError:
                await bot.send_message(message.chat.id, "Invalid number")
                return
        else:
            if field == "ticket_link" and value == "":
                setattr(event, field, None)
            else:
                setattr(event, field, value)
        await session.commit()
        new_date = event.date.split("..", 1)[0]
        new_month = new_date[:7]
    await sync_month_page(db, old_month)
    old_w = weekend_start_for_date(datetime.fromisoformat(old_date).date())
    if old_w:
        await sync_weekend_page(db, old_w.isoformat())
    if new_month != old_month:
        await sync_month_page(db, new_month)
    new_w = weekend_start_for_date(datetime.fromisoformat(new_date).date())
    if new_w and new_w != old_w:
        await sync_weekend_page(db, new_w.isoformat())
    editing_sessions[message.from_user.id] = (eid, None)
    await show_edit_menu(message.from_user.id, event, bot)


async def handle_daily_time_message(message: types.Message, db: Database, bot: Bot):
    cid = daily_time_sessions.get(message.from_user.id)
    if not cid:
        return
    value = (message.text or "").strip()
    if not re.match(r"^\d{1,2}:\d{2}$", value):
        await bot.send_message(message.chat.id, "Invalid time")
        return
    if len(value.split(":")[0]) == 1:
        value = f"0{value}"
    async with db.get_session() as session:
        ch = await session.get(Channel, cid)
        if ch:
            ch.daily_time = value
            await session.commit()
    del daily_time_sessions[message.from_user.id]
    await bot.send_message(message.chat.id, f"Time set to {value}")


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
    images = await extract_images(message, bot)
    media = images if images else None

    results = await add_events_from_text(
        db,
        text,
        link,
        message.html_text or message.caption_html,
        media,
    )
    for saved, added, lines, status in results:
        buttons = []
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            buttons.append(
                types.InlineKeyboardButton(
                    text="\u2753 Это бесплатное мероприятие",
                    callback_data=f"markfree:{saved.id}",
                )
            )
        buttons.append(
            types.InlineKeyboardButton(
                text="\U0001f6a9 Переключить на тихий режим",
                callback_data=f"togglesilent:{saved.id}",
            )
        )
        markup = (
            types.InlineKeyboardMarkup(inline_keyboard=[buttons]) if buttons else None
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


async def update_source_page(
    path: str,
    title: str,
    new_html: str,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
) -> tuple[str, int]:
    """Append text to an existing Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return "token missing"
    tg = Telegraph(access_token=token)
    try:
        logging.info("Fetching telegraph page %s", path)
        page = await asyncio.to_thread(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        catbox_msg = ""
        images: list[tuple[bytes, str]] = []
        if media:
            images = [media] if isinstance(media, tuple) else list(media)
        catbox_urls: list[str] = []
        if CATBOX_ENABLED and images:
            async with ClientSession() as session:
                for data, name in images[:3]:
                    if len(data) > 5 * 1024 * 1024:
                        logging.warning("catbox skip %s: too large", name)
                        catbox_msg += f"{name}: too large; "
                        continue
                    if not imghdr.what(None, data):
                        logging.warning("catbox skip %s: not image", name)
                        catbox_msg += f"{name}: not image; "
                        continue
                    try:
                        form = FormData()
                        form.add_field("reqtype", "fileupload")
                        form.add_field("fileToUpload", data, filename=name)
                        async with session.post(
                            "https://catbox.moe/user/api.php", data=form
                        ) as resp:
                            text = await resp.text()
                            if resp.status == 200 and text.startswith("http"):
                                url = text.strip()
                                catbox_urls.append(url)
                                catbox_msg += "ok; "
                                logging.info("catbox uploaded %s", url)
                            else:
                                catbox_msg += f"{name}: err {resp.status}; "
                                logging.error(
                                    "catbox upload failed %s: %s %s",
                                    name,
                                    resp.status,
                                    text,
                                )
                    except Exception as e:
                        catbox_msg += f"{name}: {e}; "
                        logging.error("catbox error %s: %s", name, e)
            catbox_msg = catbox_msg.strip("; ")
        elif images:
            catbox_msg = "disabled"
        for url in catbox_urls:
            html_content += f'<img src="{html.escape(url)}"/><p></p>'
        new_html = normalize_hashtag_dates(new_html)
        cleaned = re.sub(r"</?tg-emoji[^>]*>", "", new_html)
        cleaned = cleaned.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        html_content += (
            f"<p>{CONTENT_SEPARATOR}</p><p>" + cleaned.replace("\n", "<br/>") + "</p>"
        )
        logging.info("Editing telegraph page %s", path)
        await asyncio.to_thread(
            tg.edit_page, path, title=title, html_content=html_content
        )
        logging.info("Updated telegraph page %s", path)
        return catbox_msg, len(catbox_urls)
    except Exception as e:
        logging.error("Failed to update telegraph page: %s", e)
        return f"error: {e}", 0


async def create_source_page(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
) -> tuple[str, str, str, int] | None:
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

    images: list[tuple[bytes, str]] = []
    if media:
        images = [media] if isinstance(media, tuple) else list(media)
    catbox_urls: list[str] = []
    catbox_msg = ""
    if CATBOX_ENABLED and images:
        async with ClientSession() as session:
            for data, name in images[:3]:
                if len(data) > 5 * 1024 * 1024:
                    logging.warning("catbox skip %s: too large", name)
                    catbox_msg += f"{name}: too large; "
                    continue
                if not imghdr.what(None, data):
                    logging.warning("catbox skip %s: not image", name)
                    catbox_msg += f"{name}: not image; "
                    continue
                try:
                    form = FormData()
                    form.add_field("reqtype", "fileupload")
                    form.add_field("fileToUpload", data, filename=name)
                    async with session.post(
                        "https://catbox.moe/user/api.php", data=form
                    ) as resp:
                        text_r = await resp.text()
                        if resp.status == 200 and text_r.startswith("http"):
                            url = text_r.strip()
                            catbox_urls.append(url)
                            catbox_msg += "ok; "
                            logging.info("catbox uploaded %s", url)
                        else:
                            catbox_msg += f"{name}: err {resp.status}; "
                            logging.error(
                                "catbox upload failed %s: %s %s",
                                name,
                                resp.status,
                                text_r,
                            )
                except Exception as e:
                    catbox_msg += f"{name}: {e}; "
                    logging.error("catbox error %s: %s", name, e)
        catbox_msg = catbox_msg.strip("; ")
    elif images:
        catbox_msg = "disabled"

    if source_url:
        html_content += (
            f'<p><a href="{html.escape(source_url)}"><strong>'
            f"{html.escape(title)}</strong></a></p>"
        )
    else:
        html_content += f"<p><strong>{html.escape(title)}</strong></p>"

    for url in catbox_urls:
        html_content += f'<img src="{html.escape(url)}"/><p></p>'

    if html_text:
        html_text = strip_title(html_text)
        html_text = normalize_hashtag_dates(html_text)
        cleaned = re.sub(r"</?tg-emoji[^>]*>", "", html_text)
        cleaned = cleaned.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        html_content += f"<p>{cleaned.replace('\n', '<br/>')}</p>"
    else:
        clean_text = strip_title(text)
        clean_text = normalize_hashtag_dates(clean_text)
        clean_text = clean_text.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        paragraphs = [f"<p>{html.escape(line)}</p>" for line in clean_text.splitlines()]
        html_content += "".join(paragraphs)
    try:
        page = await asyncio.to_thread(tg.create_page, title, html_content=html_content)
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    logging.info("Created telegraph page %s", page.get("url"))
    return page.get("url"), page.get("path"), catbox_msg, len(catbox_urls)


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

    async def pages_wrapper(message: types.Message):
        await handle_pages(message, db, bot)

    async def edit_message_wrapper(message: types.Message):
        await handle_edit_message(message, db, bot)

    async def daily_time_wrapper(message: types.Message):
        await handle_daily_time_message(message, db, bot)

    async def forward_wrapper(message: types.Message):
        await handle_forwarded(message, db, bot)

    async def reg_daily_wrapper(message: types.Message):
        await handle_regdailychannels(message, db, bot)

    async def daily_wrapper(message: types.Message):
        await handle_daily(message, db, bot)

    async def images_wrapper(message: types.Message):
        await handle_images(message, db, bot)

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
        or c.data.startswith("dailyset:")
        or c.data.startswith("dailyunset:")
        or c.data.startswith("dailytime:")
        or c.data.startswith("dailysend:")
        or c.data.startswith("togglefree:")
        or c.data.startswith("markfree:")
        or c.data.startswith("togglesilent:"),
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(add_event_wrapper, Command("addevent"))
    dp.message.register(add_event_raw_wrapper, Command("addevent_raw"))
    dp.message.register(ask_4o_wrapper, Command("ask4o"))
    dp.message.register(list_events_wrapper, Command("events"))
    dp.message.register(set_channel_wrapper, Command("setchannel"))
    dp.message.register(images_wrapper, Command("images"))
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(reg_daily_wrapper, Command("regdailychannels"))
    dp.message.register(daily_wrapper, Command("daily"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(pages_wrapper, Command("pages"))
    dp.message.register(
        edit_message_wrapper, lambda m: m.from_user.id in editing_sessions
    )
    dp.message.register(
        daily_time_wrapper, lambda m: m.from_user.id in daily_time_sessions
    )
    dp.message.register(forward_wrapper, lambda m: bool(m.forward_date))
    dp.my_chat_member.register(partial(handle_my_chat_member, db=db))

    app = web.Application()
    SimpleRequestHandler(dp, bot).register(app, path="/webhook")
    setup_application(app, dp, bot=bot)

    async def on_startup(app: web.Application):
        logging.info("Initializing database")
        await db.init()
        global CATBOX_ENABLED
        CATBOX_ENABLED = await get_catbox_enabled(db)
        hook = webhook.rstrip("/") + "/webhook"
        logging.info("Setting webhook to %s", hook)
        await bot.set_webhook(
            hook,
            allowed_updates=["message", "callback_query", "my_chat_member"],
        )
        app["daily_task"] = asyncio.create_task(daily_scheduler(db, bot))

    async def on_shutdown(app: web.Application):
        await bot.session.close()
        if "daily_task" in app:
            app["daily_task"].cancel()
            with contextlib.suppress(Exception):
                await app["daily_task"]

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
