import logging
import os
from datetime import date, datetime, timedelta, timezone, time
from typing import Optional, Tuple, Iterable, Any
from urllib.parse import urlparse, parse_qs
import uuid
import textwrap
from supabase import create_client, Client
from icalendar import Calendar, Event as IcsEvent

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web, FormData, ClientSession, TCPConnector
from aiogram.client.session.aiohttp import AiohttpSession
import socket
import imghdr
from difflib import SequenceMatcher
import json
import re

from telegraph import Telegraph, TelegraphException

from telegraph.api import json_dumps
from functools import partial
import asyncio
import contextlib
import html
from io import BytesIO
import markdown
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import update
from sqlmodel import Field, SQLModel, select
import aiosqlite
import gc

logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("DB_PATH", "/data/db.sqlite")
TELEGRAPH_TOKEN_FILE = os.getenv("TELEGRAPH_TOKEN_FILE", "/data/telegraph_token.txt")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "events-ics")
VK_TOKEN = os.getenv("VK_TOKEN")
VK_USER_TOKEN = os.getenv("VK_USER_TOKEN")
VK_AFISHA_GROUP_ID = os.getenv("VK_AFISHA_GROUP_ID")
ICS_CONTENT_TYPE = "text/calendar; charset=utf-8"
ICS_CONTENT_DISP_TEMPLATE = 'inline; filename="{name}"'
ICS_CALNAME = "kenigevents"




def fold_unicode_line(line: str, limit: int = 74) -> str:
    """Return a folded iCalendar line without splitting UTF-8 code points."""
    encoded = line.encode("utf-8")
    parts: list[str] = []
    while len(encoded) > limit:
        cut = limit
        while cut > 0 and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        parts.append(encoded[:cut].decode("utf-8"))
        encoded = encoded[cut:]
    parts.append(encoded.decode("utf-8"))
    return "\r\n ".join(parts)

# currently active timezone offset for date calculations
LOCAL_TZ = timezone.utc

# separator inserted between versions on Telegraph source pages
CONTENT_SEPARATOR = "🟧" * 10
# separator line between events in VK posts

VK_EVENT_SEPARATOR = "\u2800\n\u2800"
# single blank line for VK posts
VK_BLANK_LINE = "\u2800"
# footer appended to VK source posts
VK_SOURCE_FOOTER = (
    f"{VK_BLANK_LINE}\n[https://vk.com/club231828790|Полюбить Калининград Анонсы]"
)
# default options for VK polls
VK_POLL_OPTIONS = ["Пойду", "Подумаю", "Нет"]


# user_id -> (event_id, field?) for editing session
editing_sessions: dict[int, tuple[int, str | None]] = {}
# user_id -> channel_id for daily time editing
daily_time_sessions: dict[int, int] = {}
# waiting for VK group ID input
vk_group_sessions: set[int] = set()
# user_id -> section (today/added) for VK time update
vk_time_sessions: dict[int, str] = {}

# superadmin user_id -> pending partner user_id
partner_info_sessions: dict[int, int] = {}
# user_id -> (festival_id, field?) for festival editing
festival_edit_sessions: dict[int, tuple[int, str | None]] = {}

# pending event text/photo input
add_event_sessions: set[int] = set()
# user_id -> event_id waiting for VK link
vk_link_sessions: dict[int, int] = {}
# waiting for a date for events listing
events_date_sessions: set[int] = set()

# toggle for uploading images to catbox
CATBOX_ENABLED: bool = False
# toggle for sending photos to VK
VK_PHOTOS_ENABLED: bool = False
_supabase_client: Client | None = None
_vk_user_token_bad: str | None = None

# Telegraph API rejects pages over ~64&nbsp;kB. Use a slightly lower limit
# to decide when month pages should be split into two parts.
TELEGRAPH_PAGE_LIMIT = 60000

# Timeout for Telegraph API operations (in seconds)
TELEGRAPH_TIMEOUT = float(os.getenv("TELEGRAPH_TIMEOUT", "30"))

# Timeout for posting ICS files to Telegram (in seconds)
ICS_POST_TIMEOUT = float(os.getenv("ICS_POST_TIMEOUT", "30"))


# Run blocking Telegraph API calls with a timeout
async def telegraph_call(func, /, *args, **kwargs):
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs), TELEGRAPH_TIMEOUT
        )
    except asyncio.TimeoutError as e:
        raise TelegraphException("Telegraph request timed out") from e


# main menu buttons
MENU_ADD_EVENT = "\u2795 Добавить событие"
MENU_EVENTS = "\U0001f4c5 События"


class IPv4AiohttpSession(AiohttpSession):
    """Aiohttp session that forces IPv4 connections."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connector_init["family"] = socket.AF_INET


def create_ipv4_session(session_cls: type[ClientSession] = ClientSession) -> ClientSession:
    """Return ClientSession that forces IPv4 connections."""
    connector = TCPConnector(family=socket.AF_INET)
    try:
        return session_cls(connector=connector)
    except TypeError:
        return session_cls()



class User(SQLModel, table=True):
    user_id: int = Field(primary_key=True)
    username: Optional[str] = None
    is_superadmin: bool = False
    is_partner: bool = False
    organization: Optional[str] = None
    location: Optional[str] = None
    blocked: bool = False
    last_partner_reminder: Optional[datetime] = None


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
    is_asset: bool = False
    daily_time: Optional[str] = None
    last_daily: Optional[str] = None


def build_channel_post_url(ch: Channel, message_id: int) -> str:
    """Return https://t.me/... link for a channel message."""
    if ch.username:
        return f"https://t.me/{ch.username}/{message_id}"
    cid = str(ch.channel_id)
    if cid.startswith("-100"):
        cid = cid[4:]
    else:
        cid = cid.lstrip("-")
    return f"https://t.me/c/{cid}/{message_id}"


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
    pushkin_card: bool = False
    silent: bool = False
    telegraph_path: Optional[str] = None
    source_text: str
    telegraph_url: Optional[str] = None
    ics_url: Optional[str] = None
    source_post_url: Optional[str] = None
    source_vk_post_url: Optional[str] = None
    ics_post_url: Optional[str] = None
    ics_post_id: Optional[int] = None
    source_chat_id: Optional[int] = None
    source_message_id: Optional[int] = None
    creator_id: Optional[int] = None
    photo_count: int = 0
    added_at: datetime = Field(default_factory=datetime.utcnow)


class MonthPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    month: str = Field(primary_key=True)
    url: str
    path: str
    url2: Optional[str] = None
    path2: Optional[str] = None


class WeekendPage(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    start: str = Field(primary_key=True)
    url: str
    path: str
    vk_post_url: Optional[str] = None


class Festival(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    full_name: Optional[str] = None
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    telegraph_url: Optional[str] = None
    telegraph_path: Optional[str] = None
    vk_post_url: Optional[str] = None
    vk_poll_url: Optional[str] = None
    photo_url: Optional[str] = None
    website_url: Optional[str] = None
    vk_url: Optional[str] = None
    tg_url: Optional[str] = None


class Database:
    def __init__(self, path: str):
        """Initialize async engine with increased busy timeout."""
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{path}",
            connect_args={"timeout": 30},
        )

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
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
            if "source_vk_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_vk_post_url VARCHAR"
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
            if "pushkin_card" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN pushkin_card BOOLEAN DEFAULT 0"
                )
            if "ics_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_url VARCHAR"
                )
            if "ics_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_post_url VARCHAR"
                )
            if "ics_post_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN ics_post_id INTEGER"
                )
            if "source_chat_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_chat_id INTEGER"
                )
            if "source_message_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN source_message_id INTEGER"
                )
            if "creator_id" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE event ADD COLUMN creator_id INTEGER"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(user)")
            cols = [r[1] for r in result.fetchall()]
            if "is_partner" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN is_partner BOOLEAN DEFAULT 0"
                )
            if "organization" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN organization VARCHAR"
                )
            if "location" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN location VARCHAR"
                )
            if "blocked" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN blocked BOOLEAN DEFAULT 0"
                )
            if "last_partner_reminder" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE user ADD COLUMN last_partner_reminder VARCHAR"
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
            if "is_asset" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE channel ADD COLUMN is_asset BOOLEAN DEFAULT 0"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(monthpage)")
            cols = [r[1] for r in result.fetchall()]
            if "url2" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN url2 VARCHAR"
                )
            if "path2" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE monthpage ADD COLUMN path2 VARCHAR"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(weekendpage)")
            cols = [r[1] for r in result.fetchall()]
            if "vk_post_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE weekendpage ADD COLUMN vk_post_url VARCHAR"
                )

            result = await conn.exec_driver_sql("PRAGMA table_info(festival)")
            cols = [r[1] for r in result.fetchall()]
            if "full_name" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN full_name VARCHAR"
                )
                await conn.exec_driver_sql(
                    "UPDATE festival SET full_name = name"
                )
            if "photo_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN photo_url VARCHAR"
                )
            if "website_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN website_url VARCHAR"
                )
            if "vk_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN vk_url VARCHAR"
                )
            if "tg_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN tg_url VARCHAR"
                )
            if "start_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN start_date VARCHAR"
                )
            if "end_date" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN end_date VARCHAR"
                )

            if "vk_poll_url" not in cols:
                await conn.exec_driver_sql(
                    "ALTER TABLE festival ADD COLUMN vk_poll_url VARCHAR"
                )

    def get_session(self) -> AsyncSession:
        """Create a new session with attributes kept after commit."""
        return AsyncSession(self.engine, expire_on_commit=False)


async def get_tz_offset(db: Database) -> str:
    async with db.get_session() as session:
        result = await session.get(Setting, "tz_offset")
        offset = result.value if result else "+00:00"
    global LOCAL_TZ
    LOCAL_TZ = offset_to_timezone(offset)
    return offset


async def set_tz_offset(db: Database, value: str):
    async with db.get_session() as session:
        setting = await session.get(Setting, "tz_offset")
        if setting:
            setting.value = value
        else:
            setting = Setting(key="tz_offset", value=value)
            session.add(setting)
        await session.commit()
    global LOCAL_TZ
    LOCAL_TZ = offset_to_timezone(value)


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


async def get_vk_photos_enabled(db: Database) -> bool:
    async with db.get_session() as session:
        setting = await session.get(Setting, "vk_photos_enabled")
        return setting.value == "1" if setting else False


async def set_vk_photos_enabled(db: Database, value: bool):
    async with db.get_session() as session:
        setting = await session.get(Setting, "vk_photos_enabled")
        if setting:
            setting.value = "1" if value else "0"
        else:
            setting = Setting(key="vk_photos_enabled", value="1" if value else "0")
            session.add(setting)
        await session.commit()
    global VK_PHOTOS_ENABLED
    VK_PHOTOS_ENABLED = value


async def get_setting_value(db: Database, key: str) -> str | None:
    async with db.get_session() as session:
        setting = await session.get(Setting, key)
        return setting.value if setting else None


async def set_setting_value(db: Database, key: str, value: str | None):
    async with db.get_session() as session:
        setting = await session.get(Setting, key)
        if value is None:
            if setting:
                await session.delete(setting)
        elif setting:
            setting.value = value
        else:
            setting = Setting(key=key, value=value)
            session.add(setting)
        await session.commit()


async def get_vk_group_id(db: Database) -> str | None:
    return await get_setting_value(db, "vk_group_id")


async def set_vk_group_id(db: Database, group_id: str | None):
    await set_setting_value(db, "vk_group_id", group_id)


async def get_vk_time_today(db: Database) -> str:
    return await get_setting_value(db, "vk_time_today") or "08:00"


async def set_vk_time_today(db: Database, value: str):
    await set_setting_value(db, "vk_time_today", value)


async def get_vk_time_added(db: Database) -> str:
    return await get_setting_value(db, "vk_time_added") or "20:00"


async def set_vk_time_added(db: Database, value: str):
    await set_setting_value(db, "vk_time_added", value)


async def get_vk_last_today(db: Database) -> str | None:
    return await get_setting_value(db, "vk_last_today")


async def set_vk_last_today(db: Database, value: str):
    await set_setting_value(db, "vk_last_today", value)


async def get_vk_last_added(db: Database) -> str | None:
    return await get_setting_value(db, "vk_last_added")


async def set_vk_last_added(db: Database, value: str):
    await set_setting_value(db, "vk_last_added", value)


def _vk_user_token() -> str | None:
    """Return user token unless it was previously marked invalid."""
    token = os.getenv("VK_USER_TOKEN")
    global _vk_user_token_bad
    if token and _vk_user_token_bad and token != _vk_user_token_bad:
        _vk_user_token_bad = None
    if token and token != _vk_user_token_bad:
        return token
    return None


async def _vk_api(
    method: str,
    params: dict,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
) -> dict:
    """Call VK API with token fallback."""
    tokens: list[tuple[str, str]] = []
    if token:
        tokens.append(("group", token))
    else:
        user_token = _vk_user_token()
        if user_token:
            tokens.append(("user", user_token))
        if VK_TOKEN:
            tokens.append(("group", VK_TOKEN))
    last_err: dict | None = None
    for kind, token in tokens:
        params["access_token"] = token
        params["v"] = "5.131"
        logging.info("calling VK API %s using %s token %s", method, kind, token)
        async with create_ipv4_session(ClientSession) as session:
            async with session.post(
                f"https://api.vk.com/method/{method}", data=params
            ) as resp:
                data = await resp.json()
        if "error" not in data:
            return data
        last_err = data["error"]
        if kind == "user" and last_err.get("error_code") in {5, 27}:
            global _vk_user_token_bad
            if _vk_user_token_bad != token:
                _vk_user_token_bad = token
                if db and bot:
                    await notify_superadmin(db, bot, "VK_USER_TOKEN expired")
            continue
        raise RuntimeError(f"VK error: {last_err}")
    if last_err:
        raise RuntimeError(f"VK error: {last_err}")
    raise RuntimeError("VK token missing")


async def upload_vk_photo(
    group_id: str,
    url: str,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
) -> str | None:
    """Upload an image to VK and return attachment id."""
    if not url:
        return None
    try:
        data = await _vk_api(
            "photos.getWallUploadServer",
            {"group_id": group_id.lstrip("-")},
            db,
            bot,
            token=token,
        )
        upload_url = data["response"]["upload_url"]
        async with create_ipv4_session(ClientSession) as session:

            async with session.get(url) as resp:
                img_bytes = await resp.read()
            form = FormData()
            ctype = "image/jpeg"
            kind = imghdr.what(None, img_bytes)
            if kind:
                ctype = f"image/{kind}"
            form.add_field(
                "photo",
                img_bytes,
                filename="image.jpg",
                content_type=ctype,
            )
            async with session.post(upload_url, data=form) as up:
                upload_result = await up.json()
        save = await _vk_api(
            "photos.saveWallPhoto",
            {
                "group_id": group_id.lstrip("-"),
                "photo": upload_result.get("photo"),
                "server": upload_result.get("server"),
                "hash": upload_result.get("hash"),
            },
            db,
            bot,
            token=token,
        )
        info = save["response"][0]
        return f"photo{info['owner_id']}_{info['id']}"
    except Exception as e:
        logging.error("VK photo upload failed: %s", e)
        return None


def get_supabase_client() -> Client | None:
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


async def get_festival(db: Database, name: str) -> Festival | None:
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.name == name)
        )
        return result.scalar_one_or_none()


async def ensure_festival(
    db: Database, name: str, description: str | None = None
) -> Festival:
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.name == name)
        )
        fest = result.scalar_one_or_none()
        if fest:
            return fest
        fest = Festival(name=name, description=description)
        session.add(fest)
        await session.commit()
        logging.info("festival %s created", name)
        return fest


async def get_asset_channel(db: Database) -> Channel | None:
    async with db.get_session() as session:
        result = await session.execute(
            select(Channel).where(Channel.is_asset.is_(True))
        )
        return result.scalars().first()


async def ensure_festival(
    db: Database,
    name: str,
    full_name: str | None = None,
    photo_url: str | None = None,
) -> Festival:
    """Return existing festival by name or create a new record."""
    async with db.get_session() as session:
        res = await session.execute(select(Festival).where(Festival.name == name))
        fest = res.scalar_one_or_none()
        if fest:
            updated = False
            if photo_url and not fest.photo_url:
                fest.photo_url = photo_url
                updated = True
            if full_name and not fest.full_name:
                fest.full_name = full_name
                updated = True
            if updated:
                session.add(fest)
                await session.commit()
            return fest
        fest = Festival(name=name, full_name=full_name, photo_url=photo_url)
        session.add(fest)
        await session.commit()
        logging.info("created festival %s", name)
        return fest


async def get_superadmin_id(db: Database) -> int | None:
    """Return the Telegram ID of the superadmin if present."""
    async with db.get_session() as session:
        result = await session.execute(
            select(User.user_id).where(User.is_superadmin.is_(True))
        )
        return result.scalars().first()


async def notify_superadmin(db: Database, bot: Bot, text: str):
    """Send a message to the superadmin, ignoring failures."""
    admin_id = await get_superadmin_id(db)
    if not admin_id:
        return
    try:
        await bot.send_message(admin_id, text)
    except Exception as e:
        logging.error("failed to notify superadmin: %s", e)


async def notify_event_added(
    db: Database, bot: Bot, user: User | None, event: Event, added: bool
) -> None:
    """Notify superadmin when a user or partner adds an event."""
    if not added or not user or user.is_superadmin:
        return
    role = "partner" if user.is_partner else "user"
    name = f"@{user.username}" if user.username else str(user.user_id)
    text = f"{name} ({role}) added event {event.title} {event.date}"
    await notify_superadmin(db, bot, text)


async def notify_inactive_partners(
    db: Database, bot: Bot, tz: timezone
) -> list[User]:
    """Send reminders to partners without events in the last week."""
    cutoff = week_cutoff(tz)
    now = datetime.utcnow()
    notified: list[User] = []
    async with db.get_session() as session:
        res = await session.execute(
            select(User).where(User.is_partner.is_(True), User.blocked.is_(False))
        )
        partners = res.scalars().all()
        for p in partners:
            last = (
                await session.execute(
                    select(Event.added_at)
                    .where(Event.creator_id == p.user_id)
                    .order_by(Event.added_at.desc())
                    .limit(1)
                )
            ).scalars().first()
            last_reminder = p.last_partner_reminder
            if (not last or last < cutoff) and (
                not last_reminder or last_reminder < cutoff
            ):
                await bot.send_message(
                    p.user_id,
                    "\u26a0\ufe0f Вы не добавляли мероприятия на прошлой неделе",
                )
                p.last_partner_reminder = now
                notified.append(p)
        await session.commit()
    return notified


async def dump_database(path: str = DB_PATH) -> bytes:
    """Return a SQL dump of the specified database."""
    async with aiosqlite.connect(path, timeout=30) as conn:
        lines: list[str] = []
        async for line in conn.iterdump():
            lines.append(line)
    return "\n".join(lines).encode("utf-8")


async def restore_database(data: bytes, db: Database, path: str = DB_PATH):
    """Replace current database with the provided dump."""
    if os.path.exists(path):
        os.remove(path)
    async with aiosqlite.connect(path, timeout=30) as conn:
        await conn.executescript(data.decode("utf-8"))
        await conn.commit()
    await db.init()


def build_asset_caption(event: Event, day: date) -> str:
    """Return HTML caption for a calendar asset post."""
    loc = html.escape(event.location_name or "")
    addr = event.location_address
    if addr and event.city:
        addr = strip_city_from_address(addr, event.city)
    if addr:
        loc += f", {html.escape(addr)}"
    if event.city:
        loc += f", #{html.escape(event.city)}"
    return (
        f"<b>{html.escape(event.title)}</b>\n"
        f"<i>{format_day_pretty(day)} {event.time} {loc}</i>"
    )


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


async def upload_to_catbox(images: list[tuple[bytes, str]]) -> tuple[list[str], str]:
    """Upload images to Catbox and return URLs with status message."""
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
    return catbox_urls, catbox_msg


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


def canonicalize_date(value: str) -> str | None:
    """Return ISO date string if value parses as date or ``None``."""
    value = value.split("..", 1)[0].strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        parsed = parse_events_date(value, timezone.utc)
        return parsed.isoformat() if parsed else None


def parse_iso_date(value: str) -> date | None:
    """Return ``date`` parsed from ISO string or ``None``."""
    try:
        return date.fromisoformat(value.split("..", 1)[0])
    except Exception:
        return None


def festival_date_range(events: Iterable[Event]) -> tuple[date | None, date | None]:
    """Return start and end dates for a festival based on its events."""
    starts: list[date] = []
    ends: list[date] = []
    for e in events:
        s = parse_iso_date(e.date)
        if not s:
            continue
        starts.append(s)
        if e.end_date:
            end = parse_iso_date(e.end_date)
        elif ".." in e.date:
            _, end_part = e.date.split("..", 1)
            end = parse_iso_date(end_part)
        else:
            end = s
        if end:
            ends.append(end)
    if not starts:
        return None, None
    return min(starts), max(ends) if ends else min(starts)


def festival_dates_from_text(text: str) -> tuple[date | None, date | None]:
    """Extract start and end dates for a festival from free-form text."""
    text = text.lower()
    m = RE_FEST_RANGE.search(text)
    if m:
        start_str, end_str = m.group(1), m.group(2)
        year = None
        m_year = re.search(r"\d{4}", end_str)
        if m_year:
            year = m_year.group(0)
        if year and not re.search(r"\d{4}", start_str):
            start_str = f"{start_str} {year}"
        if year and not re.search(r"\d{4}", end_str):
            end_str = f"{end_str} {year}"
        start = parse_events_date(start_str.replace("года", "").replace("г.", "").strip(), timezone.utc)
        end = parse_events_date(end_str.replace("года", "").replace("г.", "").strip(), timezone.utc)
        return start, end
    m = RE_FEST_SINGLE.search(text)
    if m:
        d = parse_events_date(m.group(1).replace("года", "").replace("г.", "").strip(), timezone.utc)
        return d, d
    return None, None


def festival_dates(fest: Festival, events: Iterable[Event]) -> tuple[date | None, date | None]:
    """Return start and end dates for a festival."""
    if fest.start_date or fest.end_date:
        s = parse_iso_date(fest.start_date) if fest.start_date else None
        e = parse_iso_date(fest.end_date) if fest.end_date else s
        return s, e
    start, end = festival_date_range(events)
    if start or end:
        return start, end
    if fest.description:
        s, e = festival_dates_from_text(fest.description)
        if s or e:
            return s, e or s
    return None, None


def festival_location(events: Iterable[Event]) -> str | None:
    """Return display string for festival venue(s)."""
    pairs = {(e.location_name, e.city) for e in events if e.location_name}
    if not pairs:
        return None
    names = sorted({name for name, _ in pairs})
    cities = {c for _, c in pairs if c}
    city_text = ""
    if len(cities) == 1:
        city_text = f", #{next(iter(cities))}"
    return ", ".join(names) + city_text


async def upcoming_festivals(
    db: Database,
    *,
    today: date | None = None,
    exclude: str | None = None,
) -> list[tuple[date | None, date | None, Festival]]:
    """Return festivals that are current or upcoming."""
    if today is None:
        today = date.today()
    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fests = res_f.scalars().all()
        res_e = await session.execute(select(Event))
        events = res_e.scalars().all()

    ev_map: dict[str, list[Event]] = {}
    for e in events:
        if e.festival:
            ev_map.setdefault(e.festival, []).append(e)

    data: list[tuple[date | None, date | None, Festival]] = []
    for fest in fests:
        if exclude and fest.name == exclude:
            continue
        evs = ev_map.get(fest.name, [])
        start, end = festival_dates(fest, evs)
        if end and end < today:
            continue
        if not start and not end:
            continue
        data.append((start, end, fest))

    data.sort(key=lambda t: t[0] or date.max)
    return data


async def build_festivals_list_nodes(
    db: Database, *, exclude: str | None = None, today: date | None = None
) -> list[dict]:
    """Return Telegraph nodes listing upcoming festivals."""
    items = await upcoming_festivals(db, today=today, exclude=exclude)
    if not items:
        return []
    if today is None:
        today = date.today()
    groups: dict[str, list[Festival]] = {}
    for start, end, fest in items:
        if start and start <= today <= (end or start):
            month = today.strftime("%Y-%m")
        else:
            month = (start or today).strftime("%Y-%m")
        groups.setdefault(month, []).append(fest)

    nodes: list[dict] = []
    nodes.append({"tag": "h3", "children": ["Ближайшие фестивали"]})
    for month in sorted(groups.keys()):
        nodes.append({"tag": "h4", "children": [month_name_nominative(month)]})
        for fest in groups[month]:
            if fest.telegraph_url:
                nodes.append(
                    {
                        "tag": "p",
                        "children": [
                            {
                                "tag": "a",
                                "attrs": {"href": fest.telegraph_url},
                                "children": [fest.name],
                            }
                        ],
                    }
                )
            else:
                nodes.append({"tag": "p", "children": [fest.name]})
    return nodes


async def build_festivals_list_lines_vk(
    db: Database, *, exclude: str | None = None, today: date | None = None
) -> list[str]:
    """Return lines listing upcoming festivals for VK posts."""
    items = await upcoming_festivals(db, today=today, exclude=exclude)
    if not items:
        return []
    if today is None:
        today = date.today()
    groups: dict[str, list[Festival]] = {}
    for start, end, fest in items:
        if start and start <= today <= (end or start):
            month = today.strftime("%Y-%m")
        else:
            month = (start or today).strftime("%Y-%m")
        groups.setdefault(month, []).append(fest)

    lines: list[str] = ["Ближайшие фестивали"]
    for month in sorted(groups.keys()):
        lines.append(month_name_nominative(month))
        for fest in groups[month]:
            if fest.vk_post_url:
                lines.append(f"[{fest.vk_post_url}|{fest.name}]")

            else:
                lines.append(fest.name)
    return lines


ICS_LABEL = "Добавить в календарь на телефоне (ICS)"
MONTH_NAV_START = "<!--month-nav-start-->"
MONTH_NAV_END = "<!--month-nav-end-->"

FOOTER_LINK_HTML = (
    '<p>&nbsp;</p>'
    '<p><a href="https://t.me/kenigevents">Полюбить Калининград Анонсы</a></p>'
    '<p>&nbsp;</p>'
)


def parse_time_range(value: str) -> tuple[time, time | None] | None:
    """Return start and optional end time from text like ``10:00`` or ``10:00-12:00``.

    Accepts ``-`` as well as ``..`` or ``—``/``–`` between times.
    """
    value = value.strip()
    parts = re.split(r"\s*(?:-|–|—|\.\.\.?|…)+\s*", value, maxsplit=1)
    try:
        start = datetime.strptime(parts[0], "%H:%M").time()
    except ValueError:
        return None
    end: time | None = None
    if len(parts) == 2:
        try:
            end = datetime.strptime(parts[1], "%H:%M").time()
        except ValueError:
            end = None
    return start, end


def apply_ics_link(html_content: str, url: str | None) -> str:
    """Insert or remove the ICS link block in Telegraph HTML."""
    idx = html_content.find(ICS_LABEL)
    if idx != -1:
        start = html_content.rfind("<p", 0, idx)
        end = html_content.find("</p>", idx)
        if start != -1 and end != -1:
            html_content = html_content[:start] + html_content[end + 4 :]
    if not url:
        return html_content
    link_html = (
        f'<p>\U0001f4c5 <a href="{html.escape(url)}">{ICS_LABEL}</a></p>'
    )
    idx = html_content.find("</p>")
    if idx == -1:
        return link_html + html_content
    pos = idx + 4
    img_pattern = re.compile(r"<img[^>]+><p></p>")
    for m in img_pattern.finditer(html_content, pos):
        pos = m.end()
    return html_content[:pos] + link_html + html_content[pos:]


def apply_month_nav(html_content: str, html_block: str | None) -> str:
    """Insert or replace the month navigation block."""
    start = html_content.find(MONTH_NAV_START)
    if start != -1:
        end = html_content.find(MONTH_NAV_END, start)
        if end != -1:
            html_content = html_content[:start] + html_content[end + len(MONTH_NAV_END) :]
    if html_block:
        html_content += f"{MONTH_NAV_START}{html_block}{MONTH_NAV_END}"
    return html_content


def apply_footer_link(html_content: str) -> str:
    """Ensure the Telegram channel link footer is present once."""
    pattern = re.compile(
        r'<p><a href="https://t\.me/kenigevents">[^<]+</a></p><p>&nbsp;</p>'
    )
    html_content = pattern.sub("", html_content).rstrip()
    return html_content + FOOTER_LINK_HTML


async def build_month_nav_html(db: Database) -> str:
    async with db.get_session() as session:
        result = await session.execute(select(MonthPage).order_by(MonthPage.month))
        months = result.scalars().all()
    if not months:
        return ""
    links: list[str] = []
    for idx, p in enumerate(months):
        name = month_name_nominative(p.month)
        links.append(f'<a href="{html.escape(p.url)}">{name}</a>')
        if idx < len(months) - 1:
            links.append(" ")
    return "<br/><h4>" + "".join(links) + "</h4>"

async def build_month_buttons(db: Database, limit: int = 3) -> list[types.InlineKeyboardButton]:
    """Return buttons linking to upcoming month pages."""
    cur_month = datetime.now(LOCAL_TZ).strftime("%Y-%m")
    async with db.get_session() as session:
        result = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= cur_month)
            .order_by(MonthPage.month)
        )
        months = result.scalars().all()
    buttons: list[types.InlineKeyboardButton] = []
    for p in months[:limit]:
        if p.url:
            label = f"\U0001f4c5 {month_name_nominative(p.month)}"
            buttons.append(types.InlineKeyboardButton(text=label, url=p.url))
    return buttons


def parse_bool_text(value: str) -> bool | None:
    """Convert text to boolean if possible."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "да", "д", "ok", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "нет", "off"}:
        return False
    return None


def parse_events_date(text: str, tz: timezone) -> date | None:
    """Parse a date argument for /events allowing '2 августа [2025]'."""
    text = text.strip().lower()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    m = re.match(r"(\d{1,2})\s+([а-яё]+)(?:\s+(\d{4}))?", text)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2)
    year_part = m.group(3)
    month = {name: i + 1 for i, name in enumerate(MONTHS)}.get(month_name)
    if not month:
        return None
    if year_part:
        year = int(year_part)
    else:
        today = datetime.now(tz).date()
        year = today.year
        if month < today.month or (month == today.month and day < today.day):
            year += 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


async def build_ics_content(db: Database, event: Event) -> str:
    """Build an RFC 5545 compliant ICS string for an event."""
    time_range = parse_time_range(event.time)
    if not time_range:
        raise ValueError("bad time")
    start_t, end_t = time_range
    date_obj = parse_iso_date(event.date)
    if not date_obj:
        raise ValueError("bad date")
    start_dt = datetime.combine(date_obj, start_t)
    if end_t:
        end_dt = datetime.combine(date_obj, end_t)
    else:
        end_dt = start_dt + timedelta(hours=1)

    title = event.title
    if event.location_name:
        title = f"{title} в {event.location_name}"

    desc = event.description or ""
    link = event.source_post_url or event.telegraph_url
    if link:
        desc = f"{desc}\n\n{link}" if desc else link

    loc_parts = []
    if event.location_address:
        loc_parts.append(event.location_address)
    if event.city:
        loc_parts.append(event.city)
    # Join without a space after comma to avoid iOS parsing issues
    location = ",".join(loc_parts)

    cal = Calendar()
    cal.add("VERSION", "2.0")
    cal.add("PRODID", "-//events-bot//RU")
    cal.add("CALSCALE", "GREGORIAN")
    cal.add("METHOD", "PUBLISH")
    cal.add("X-WR-CALNAME", ICS_CALNAME)

    vevent = IcsEvent()
    vevent.add("UID", f"{uuid.uuid4()}@{event.id}")
    vevent.add("DTSTAMP", datetime.now(timezone.utc))
    vevent.add("DTSTART", start_dt)
    vevent.add("DTEND", end_dt)
    vevent.add("SUMMARY", title)
    vevent.add("DESCRIPTION", desc)
    if location:
        vevent.add("LOCATION", location)
    if link:
        vevent.add("URL", link)
    cal.add_component(vevent)

    raw = cal.to_ical().decode("utf-8")
    lines = raw.split("\r\n")
    if lines and lines[-1] == "":
        lines.pop()

    # unfold lines first
    unfolded: list[str] = []
    for line in lines:
        if line.startswith(" ") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)

    for i, line in enumerate(unfolded):
        if line.startswith("LOCATION:") or line.startswith("LOCATION;"):
            unfolded[i] = line.replace("\\, ", "\\,\\ ")

    idx = unfolded.index("BEGIN:VEVENT")
    vbody = unfolded[idx + 1 : -2]  # between BEGIN:VEVENT and END:VEVENT
    order = ["UID", "DTSTAMP", "DTSTART", "DTEND"]
    props: list[str] = []
    for key in order:
        for l in list(vbody):
            if l.startswith(key + ":") or l.startswith(key + ";"):
                props.append(l)
                vbody.remove(l)
    props.extend(vbody)

    body = ["BEGIN:VEVENT"] + props + ["END:VEVENT"]
    headers = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//events-bot//RU",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{ICS_CALNAME}",
    ]
    final_lines = headers + body + ["END:VCALENDAR"]
    folded = [fold_unicode_line(l) for l in final_lines]
    return "\r\n".join(folded) + "\r\n"


async def upload_ics(event: Event, db: Database) -> str | None:
    client = get_supabase_client()
    if not client:
        logging.error("Supabase client not configured")
        return None
    if event.end_date:
        logging.info("skip ics for multi-day event %s", event.id)
        return None
    if not parse_time_range(event.time):
        logging.info("skip ics for unclear time %s", event.id)
        return None
    content = await build_ics_content(db, event)
    d = parse_iso_date(event.date)
    if d:
        path = f"Event-{event.id}-{d.day:02d}-{d.month:02d}-{d.year}.ics"
    else:
        path = f"Event-{event.id}.ics"
    try:
        logging.info("Uploading ICS to %s/%s", SUPABASE_BUCKET, path)
        client.storage.from_(SUPABASE_BUCKET).upload(
            path,
            content.encode("utf-8"),
            {
                "content-type": ICS_CONTENT_TYPE,
                "content-disposition": ICS_CONTENT_DISP_TEMPLATE.format(name=path),
                "upsert": "true",
            },
        )
        url = client.storage.from_(SUPABASE_BUCKET).get_public_url(path)
        logging.info("ICS uploaded: %s", url)
    except Exception as e:
        logging.error("Failed to upload ics: %s", e)
        return None
    return url


async def post_ics_asset(event: Event, db: Database, bot: Bot) -> tuple[str, int] | None:
    channel = await get_asset_channel(db)
    if not channel:
        logging.info("no asset channel configured")
        return None
    try:
        content = await build_ics_content(db, event)
    except Exception as e:
        logging.error("failed to build ics content: %s", e)
        return None

    d = parse_iso_date(event.date)
    if d:
        name = f"Event-{event.id}-{d.day:02d}-{d.month:02d}-{d.year}.ics"
    else:

        d = date.today()
        name = f"Event-{event.id}.ics"
    file = types.BufferedInputFile(content.encode("utf-8"), filename=name)
    caption = build_asset_caption(event, d)

    logging.info("posting ics asset to channel %s with caption %s", channel.channel_id, caption.replace('\n', ' | '))

    try:
        msg = await bot.send_document(
            channel.channel_id,
            file,
            caption=caption,
            parse_mode="HTML",
        )
        url = build_channel_post_url(channel, msg.message_id)
        logging.info("posted ics to asset channel: %s", url)
        return url, msg.message_id
    except Exception as e:
        logging.error("failed to post ics to asset channel: %s", e)
        return None

async def add_calendar_button(event: Event, db: Database, bot: Bot):
    """Attach calendar link button to the original channel post."""
    if not (
        event.source_chat_id
        and event.source_message_id
        and event.ics_post_url
    ):
        return
    month_buttons = await build_month_buttons(db)
    rows = [[types.InlineKeyboardButton(text="Добавить в календарь", url=event.ics_post_url)]]
    if month_buttons:
        rows.append(month_buttons)
    markup = types.InlineKeyboardMarkup(inline_keyboard=rows)
    try:
        await bot.edit_message_reply_markup(
            chat_id=event.source_chat_id,
            message_id=event.source_message_id,
            reply_markup=markup,
        )

        logging.info(
            "calendar button set for event %s post %s", event.id, event.source_post_url
        )

    except Exception as e:
        logging.error("failed to set calendar button: %s", e)


async def delete_ics(event: Event):
    client = get_supabase_client()
    if not client or not event.ics_url:
        return
    path = event.ics_url.split("/")[-1]
    try:
        logging.info("Deleting ICS %s from %s", path, SUPABASE_BUCKET)
        client.storage.from_(SUPABASE_BUCKET).remove([path])
    except Exception as e:
        logging.error("Failed to delete ics: %s", e)


async def delete_asset_post(event: Event, db: Database, bot: Bot):
    if not event.ics_post_id:
        return
    channel = await get_asset_channel(db)
    if not channel:
        return
    try:
        await bot.delete_message(channel.channel_id, event.ics_post_id)
    except Exception as e:
        logging.error("failed to delete asset message: %s", e)


async def remove_calendar_button(event: Event, bot: Bot):
    """Remove calendar button from the original channel post."""
    if not (event.source_chat_id and event.source_message_id):
        return
    try:
        await bot.edit_message_reply_markup(
            chat_id=event.source_chat_id,
            message_id=event.source_message_id,
            reply_markup=None,
        )

        logging.info(
            "calendar button removed for event %s post %s",
            event.id,
            event.source_post_url,
        )

    except Exception as e:
        logging.error("failed to remove calendar button: %s", e)


async def parse_event_via_4o(
    text: str,
    source_channel: str | None = None,
    *,
    festival_names: list[str] | None = None,
    **extra: str | None,
) -> list[dict]:
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
    if festival_names:
        prompt += "\nKnown festivals:\n" + "\n".join(festival_names)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if not source_channel:
        source_channel = extra.get("channel_title")
    today = datetime.now(LOCAL_TZ).date().isoformat()
    user_msg = f"Today is {today}. "
    if source_channel:
        user_msg += f"Channel: {source_channel}. "
    user_msg += text
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
    }
    logging.info("Sending 4o parse request to %s", url)
    async with create_ipv4_session(ClientSession) as session:
        resp = await session.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data_raw = await resp.json()
    content = (
        data_raw.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
        .strip()
    )
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("4o content snippet: %s", content[:1000])
    del data_raw
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
    async with create_ipv4_session(ClientSession) as session:
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


async def send_main_menu(bot: Bot, user: User | None, chat_id: int) -> None:
    """Show main menu buttons depending on user role."""
    buttons = [
        [types.KeyboardButton(text=MENU_ADD_EVENT)],
        [types.KeyboardButton(text=MENU_EVENTS)],
    ]
    markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
    await bot.send_message(chat_id, "Choose action", reply_markup=markup)


async def handle_start(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        result = await session.execute(select(User))
        user_count = len(result.scalars().all())
        user = await session.get(User, message.from_user.id)
        if user:
            if user.blocked:
                await bot.send_message(message.chat.id, "Access denied")
                return
            if user.is_partner:
                org = f" ({user.organization})" if user.organization else ""
                await bot.send_message(message.chat.id, f"You are partner{org}")
            else:
                await bot.send_message(message.chat.id, "Bot is running")
            await send_main_menu(bot, user, message.chat.id)
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
            new_user = await session.get(User, message.from_user.id)
            await send_main_menu(bot, new_user, message.chat.id)
        else:
            await bot.send_message(message.chat.id, "Use /register to apply")


async def handle_menu(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if user and not user.blocked:
        await send_main_menu(bot, user, message.chat.id)


async def handle_events_menu(message: types.Message, db: Database, bot: Bot):
    """Show options for events listing."""
    buttons = [
        [types.InlineKeyboardButton(text="Сегодня", callback_data="menuevt:today")],
        [types.InlineKeyboardButton(text="Дата", callback_data="menuevt:date")],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    await bot.send_message(message.chat.id, "Выберите день", reply_markup=markup)


async def handle_events_date_message(message: types.Message, db: Database, bot: Bot):
    if message.from_user.id not in events_date_sessions:
        return
    value = (message.text or "").strip()
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)
    day = parse_events_date(value, tz)
    if not day:
        await bot.send_message(message.chat.id, "Неверная дата")
        return
    events_date_sessions.discard(message.from_user.id)
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
        creator_filter = user.user_id if user.is_partner else None
    text, markup = await build_events_message(db, day, tz, creator_filter)
    await bot.send_message(message.chat.id, text, reply_markup=markup)


async def handle_register(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        existing = await session.get(User, message.from_user.id)
        if existing:
            if existing.blocked:
                await bot.send_message(message.chat.id, "Access denied")
            else:
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
                    text="Partner", callback_data=f"partner:{p.user_id}"
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
    if data.startswith("approve") or data.startswith("reject") or data.startswith("partner"):
        uid = int(data.split(":", 1)[1])
        async with db.get_session() as session:
            p = await session.get(PendingUser, uid)
            if not p:
                await callback.answer("Not found", show_alert=True)
                return
            if data.startswith("approve"):
                session.add(User(user_id=uid, username=p.username, is_superadmin=False))
                await bot.send_message(uid, "You are approved")
            elif data.startswith("partner"):
                partner_info_sessions[callback.from_user.id] = uid
                await callback.message.answer(
                    "Send organization and location, e.g. 'Дом Китобоя, Мира 9, Калининград'"
                )
                await callback.answer()
                return
            else:
                session.add(RejectedUser(user_id=uid, username=p.username))
                await bot.send_message(uid, "Your registration was rejected")
            await session.delete(p)
            await session.commit()
            await callback.answer("Done")
    elif data.startswith("block:") or data.startswith("unblock:"):
        uid = int(data.split(":", 1)[1])
        async with db.get_session() as session:
            user = await session.get(User, uid)
            if not user or user.is_superadmin:
                await callback.answer("Not allowed", show_alert=True)
                return
            user.blocked = data.startswith("block:")
            await session.commit()
        await send_users_list(callback.message, db, bot, edit=True)
        await callback.answer("Updated")
    elif data.startswith("del:"):
        _, eid, marker = data.split(":")
        month = None
        w_start: date | None = None
        vk_post: str | None = None
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, int(eid))
            if (user and user.blocked) or (
                user and user.is_partner and event and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event:
                month = event.date.split("..", 1)[0][:7]
                d = parse_iso_date(event.date)
                w_start = weekend_start_for_date(d) if d else None
                vk_post = event.source_vk_post_url
                await session.delete(event)
                await session.commit()
        if month:
            await sync_month_page(db, month)
            if w_start:
                await sync_weekend_page(db, w_start.isoformat(), post_vk=False)
                await sync_vk_weekend_post(db, w_start.isoformat())
        if vk_post:
            await delete_vk_post(vk_post, db, bot)
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        if marker == "exh":
            text, markup = await build_exhibitions_message(db, tz)
        else:
            target = datetime.strptime(marker, "%Y-%m-%d").date()
            filter_id = user.user_id if user and user.is_partner else None
            text, markup = await build_events_message(db, target, tz, filter_id)
        await callback.message.edit_text(text, reply_markup=markup)
        await callback.answer("Deleted")
    elif data.startswith("edit:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
        if event and ((user and user.blocked) or (user and user.is_partner and event.creator_id != user.user_id)):
            await callback.answer("Not authorized", show_alert=True)
            return
        if event:
            editing_sessions[callback.from_user.id] = (eid, None)
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer()
    elif data.startswith("editfield:"):
        _, eid, field = data.split(":")
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, int(eid))
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
        if field == "festival":
            async with db.get_session() as session:
                fests = (await session.execute(select(Festival))).scalars().all()
            keyboard = [
                [
                    types.InlineKeyboardButton(text=f.name, callback_data=f"setfest:{eid}:{f.id}")
                ]
                for f in fests
            ]
            keyboard.append([
                types.InlineKeyboardButton(text="None", callback_data=f"setfest:{eid}:0")
            ])
            markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
            await callback.message.answer("Choose festival", reply_markup=markup)
            await callback.answer()
            return
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
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event:
                event.is_free = not event.is_free
                await session.commit()
                logging.info("togglefree: event %s set to %s", eid, event.is_free)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
            d = parse_iso_date(event.date)
            w_start = weekend_start_for_date(d) if d else None
            if w_start:
                await sync_weekend_page(db, w_start.isoformat())
        async with db.get_session() as session:
            event = await session.get(Event, eid)
        if event:
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer()
    elif data.startswith("setfest:"):
        _, eid, fid = data.split(":")
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, int(eid))
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if fid == "0":
                event.festival = None
            else:
                fest = await session.get(Festival, int(fid))
                if fest:
                    event.festival = fest.name
            await session.commit()
            fest_name = event.festival
            logging.info(
                "event %s festival set to %s",
                eid,
                fest_name or "None",
            )
        if fest_name:
            await sync_festival_page(db, fest_name)

            await sync_festival_vk_post(db, fest_name, bot)

        await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer("Updated")
    elif data.startswith("festedit:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            if not await session.get(User, callback.from_user.id):
                await callback.answer("Not authorized", show_alert=True)
                return
            fest = await session.get(Festival, fid)
            if not fest:
                await callback.answer("Festival not found", show_alert=True)
                return
        festival_edit_sessions[callback.from_user.id] = (fid, None)
        await show_festival_edit_menu(callback.from_user.id, fest, bot)
        await callback.answer()
    elif data.startswith("festeditfield:"):
        _, fid, field = data.split(":")
        async with db.get_session() as session:
            if not await session.get(User, callback.from_user.id):
                await callback.answer("Not authorized", show_alert=True)
                return
            fest = await session.get(Festival, int(fid))
            if not fest:
                await callback.answer("Festival not found", show_alert=True)
                return
        festival_edit_sessions[callback.from_user.id] = (int(fid), field)
        if field == "description":
            prompt = "Send new description"
        elif field == "name":
            prompt = "Send short name"
        elif field == "full":
            prompt = "Send full name or '-' to delete"
        elif field == "start":
            prompt = "Send start date (YYYY-MM-DD) or '-' to delete"
        elif field == "end":
            prompt = "Send end date (YYYY-MM-DD) or '-' to delete"
        else:
            prompt = "Send URL or '-' to delete"
        await callback.message.answer(prompt)
        await callback.answer()
    elif data == "festeditdone":
        if callback.from_user.id in festival_edit_sessions:
            del festival_edit_sessions[callback.from_user.id]
        await callback.message.answer("Festival editing finished")
        await callback.answer()
    elif data.startswith("festdel:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            if not await session.get(User, callback.from_user.id):
                await callback.answer("Not authorized", show_alert=True)
                return
            fest = await session.get(Festival, fid)
            if not fest:
                await callback.answer("Festival not found", show_alert=True)
                return
            await session.execute(
                update(Event).where(Event.festival == fest.name).values(festival=None)
            )
            await session.delete(fest)
            await session.commit()
            logging.info("festival %s deleted", fest.name)
        await send_festivals_list(callback.message, db, bot, edit=True)
        await callback.answer("Deleted")

    elif data.startswith("togglesilent:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event:
                event.silent = not event.silent
                await session.commit()
                logging.info("togglesilent: event %s set to %s", eid, event.silent)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
            d = parse_iso_date(event.date)
            w_start = weekend_start_for_date(d) if d else None
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
    elif data.startswith("createics:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event:
                url = await upload_ics(event, db)
                if url:
                    event.ics_url = url
                    await session.commit()
                    logging.info("ICS saved for event %s: %s", eid, url)
                    posted = await post_ics_asset(event, db, bot)
                    if posted:
                        url_p, msg_id = posted
                        event.ics_post_url = url_p
                        event.ics_post_id = msg_id
                        await add_calendar_button(event, db, bot)
                    if event.telegraph_path:
                        await update_source_page_ics(
                            event.telegraph_path, event.title or "Event", url
                        )
                        if not is_vk_wall_url(event.source_post_url):
                            await sync_vk_source_post(
                                event,
                                event.source_text,
                                db,
                                bot,
                                ics_url=url,
                            )
                    month = event.date.split("..", 1)[0][:7]
                    await sync_month_page(db, month)

                    d = parse_iso_date(event.date)
                    w_start = weekend_start_for_date(d) if d else None

                    if w_start:
                        await sync_weekend_page(db, w_start.isoformat())
                else:
                    logging.warning("ICS creation failed for event %s", eid)
        if event:
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer("Created")
    elif data.startswith("delics:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event and event.ics_url:
                await delete_ics(event)
                event.ics_url = None
                await session.commit()
                logging.info("ICS removed for event %s", eid)
                await delete_asset_post(event, db, bot)
                event.ics_post_url = None
                event.ics_post_id = None
                await session.commit()
                await remove_calendar_button(event, bot)
                if event.telegraph_path:
                    await update_source_page_ics(
                        event.telegraph_path, event.title or "Event", None
                    )
                    if not is_vk_wall_url(event.source_post_url):
                        await sync_vk_source_post(
                            event,
                            event.source_text,
                            db,
                            bot,
                            ics_url=None,
                        )
                month = event.date.split("..", 1)[0][:7]
                await sync_month_page(db, month)

                d = parse_iso_date(event.date)
                w_start = weekend_start_for_date(d) if d else None

                if w_start:
                    await sync_weekend_page(db, w_start.isoformat())
            elif event:
                logging.debug("deleteics: no file for event %s", eid)
        if event:
            await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer("Deleted")
    elif data.startswith("markfree:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            if not event or (user and user.blocked) or (
                user and user.is_partner and event.creator_id != user.user_id
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
            if event:
                event.is_free = True
                await session.commit()
                logging.info("markfree: event %s marked free", eid)
                month = event.date.split("..", 1)[0][:7]
        if event:
            await sync_month_page(db, month)
            d = parse_iso_date(event.date)
            w_start = weekend_start_for_date(d) if d else None
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
    elif data.startswith("festedit:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        festival_edit_sessions[callback.from_user.id] = (fid, None)
        await show_festival_edit_menu(callback.from_user.id, fest, bot)
        await callback.answer()
    elif data.startswith("festeditfield:"):
        _, fid, field = data.split(":")
        async with db.get_session() as session:
            fest = await session.get(Festival, int(fid))
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        festival_edit_sessions[callback.from_user.id] = (int(fid), field)
        prompt = (
            "Send new description"
            if field == "description"
            else "Send URL or '-' to delete"
        )
        await callback.message.answer(prompt)
        await callback.answer()
    elif data == "festeditdone":
        if callback.from_user.id in festival_edit_sessions:
            del festival_edit_sessions[callback.from_user.id]
        await callback.message.answer("Festival editing finished")
        await callback.answer()
    elif data.startswith("festdel:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
            if fest:
                await session.delete(fest)
                await session.commit()
                logging.info("festival %s deleted", fest.name)
        await send_festivals_list(callback.message, db, bot, edit=True)
        await callback.answer("Deleted")
    elif data.startswith("nav:"):
        _, day = data.split(":")
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        target = datetime.strptime(day, "%Y-%m-%d").date()
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
        filter_id = user.user_id if user and user.is_partner else None
        text, markup = await build_events_message(db, target, tz, filter_id)
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
    elif data.startswith("assetunset:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            ch = await session.get(Channel, cid)
            if ch and ch.is_asset:
                ch.is_asset = False
                logging.info("asset channel unset %s", cid)
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
    elif data.startswith("assetset:"):
        cid = int(data.split(":")[1])
        async with db.get_session() as session:
            current = await session.execute(
                select(Channel).where(Channel.is_asset.is_(True))
            )
            cur = current.scalars().first()
            if cur and cur.channel_id != cid:
                cur.is_asset = False
            ch = await session.get(Channel, cid)
            if ch and ch.is_admin:
                ch.is_asset = True
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
    elif data.startswith("dailysendtom:"):
        cid = int(data.split(":")[1])
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz) + timedelta(days=1)
        await send_daily_announcement(db, bot, cid, tz, record=False, now=now)
        await callback.answer("Sent")
    elif data == "vkset":
        vk_group_sessions.add(callback.from_user.id)
        await callback.message.answer("Send VK group id or 'off'")
        await callback.answer()
    elif data == "vkunset":
        await set_vk_group_id(db, None)
        await send_daily_list(callback.message, db, bot, edit=True)
        await callback.answer("Disabled")
    elif data.startswith("vktime:"):
        typ = data.split(":", 1)[1]
        vk_time_sessions[callback.from_user.id] = typ
        await callback.message.answer("Send new time HH:MM")
        await callback.answer()
    elif data.startswith("vkdailysend:"):
        section = data.split(":", 1)[1]
        group_id = await get_vk_group_id(db)
        if group_id:
            offset = await get_tz_offset(db)
            tz = offset_to_timezone(offset)
            await send_daily_announcement_vk(
                db, group_id, tz, section=section, bot=bot
            )
        await callback.answer("Sent")
    elif data.startswith("vklink:"):
        eid = int(data.split(":", 1)[1])
        vk_link_sessions[callback.from_user.id] = eid
        await callback.message.answer("Send VK post link")
        await callback.answer()
    elif data == "vklinkskip":
        await callback.answer("Skipped")
    elif data == "menuevt:today":
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            if not user or user.blocked:
                await callback.message.answer("Not authorized")
                await callback.answer()
                return
            creator_filter = user.user_id if user.is_partner else None
        day = datetime.now(tz).date()
        text, markup = await build_events_message(db, day, tz, creator_filter)
        await callback.message.answer(text, reply_markup=markup)
        await callback.answer()
    elif data == "menuevt:date":
        events_date_sessions.add(callback.from_user.id)
        await callback.message.answer("Введите дату")
        await callback.answer()


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


async def handle_vkgroup(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    if len(parts) != 2:
        await bot.send_message(message.chat.id, "Usage: /vkgroup <id|off>")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    if parts[1].lower() == "off":
        await set_vk_group_id(db, None)
        await bot.send_message(message.chat.id, "VK posting disabled")
    else:
        await set_vk_group_id(db, parts[1])
        await bot.send_message(message.chat.id, f"VK group set to {parts[1]}")


async def handle_vktime(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split()
    if len(parts) != 3 or parts[1] not in {"today", "added"}:
        await bot.send_message(message.chat.id, "Usage: /vktime today|added HH:MM")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    if not re.match(r"^\d{2}:\d{2}$", parts[2]):
        await bot.send_message(message.chat.id, "Invalid time format")
        return
    if parts[1] == "today":
        await set_vk_time_today(db, parts[2])
    else:
        await set_vk_time_added(db, parts[2])
    await bot.send_message(message.chat.id, "VK time updated")


async def handle_vkphotos(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    new_value = not VK_PHOTOS_ENABLED
    await set_vk_photos_enabled(db, new_value)
    status = "enabled" if new_value else "disabled"
    await bot.send_message(message.chat.id, f"VK photo posting {status}")


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
        status = []
        row: list[types.InlineKeyboardButton] = []
        if ch.is_registered:
            status.append("✅")
            row.append(
                types.InlineKeyboardButton(
                    text="Cancel", callback_data=f"unset:{ch.channel_id}"
                )
            )
        if ch.is_asset:
            status.append("📅")
            row.append(
                types.InlineKeyboardButton(
                    text="Asset off", callback_data=f"assetunset:{ch.channel_id}"
                )
            )
        lines.append(f"{name} {' '.join(status)}".strip())
        if row:
            keyboard.append(row)
    if not lines:
        lines.append("No channels")
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_users_list(message: types.Message, db: Database, bot: Bot, edit: bool = False):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(User))
        users = result.scalars().all()
    lines = []
    keyboard = []
    for u in users:
        role = "superadmin" if u.is_superadmin else ("partner" if u.is_partner else "user")
        org = f" ({u.organization})" if u.is_partner and u.organization else ""
        status = " 🚫" if u.blocked else ""
        lines.append(f"{u.user_id} {u.username or ''} {role}{org}{status}".strip())
        if not u.is_superadmin:
            if u.blocked:
                keyboard.append([types.InlineKeyboardButton(text="Unblock", callback_data=f"unblock:{u.user_id}")])
            else:
                keyboard.append([types.InlineKeyboardButton(text="Block", callback_data=f"block:{u.user_id}")])
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None
    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def send_festivals_list(message: types.Message, db: Database, bot: Bot, edit: bool = False):
    async with db.get_session() as session:
        if not await session.get(User, message.from_user.id):
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(Festival))
        fests = result.scalars().all()
    lines = []
    for f in fests:
        parts = [f"{f.id} {f.name}"]
        if f.telegraph_url:
            parts.append(f.telegraph_url)
        if f.website_url:
            parts.append(f"site: {f.website_url}")
        if f.vk_url:
            parts.append(f"vk: {f.vk_url}")
        if f.tg_url:
            parts.append(f"tg: {f.tg_url}")
        lines.append(" ".join(parts))
    keyboard = [
        [
            types.InlineKeyboardButton(text="Edit", callback_data=f"festedit:{f.id}"),
            types.InlineKeyboardButton(text="Delete", callback_data=f"festdel:{f.id}"),
        ]
        for f in fests
    ]
    if not lines:
        lines.append("No festivals")
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
            select(Channel).where(Channel.is_admin.is_(True))
        )
        channels = result.scalars().all()
    logging.info("setchannel list: %s", [c.channel_id for c in channels])
    lines = []
    keyboard = []
    for ch in channels:
        name = ch.title or ch.username or str(ch.channel_id)
        lines.append(name)
        row = []
        if ch.daily_time is None:
            row.append(
                types.InlineKeyboardButton(
                    text="Announce", callback_data=f"set:{ch.channel_id}"
                )
            )
        if not ch.is_asset:
            row.append(
                types.InlineKeyboardButton(
                    text="Asset", callback_data=f"assetset:{ch.channel_id}"
                )
            )
        if row:
            keyboard.append(row)
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
    group_id = await get_vk_group_id(db)
    if group_id:
        lines.append(f"VK group {group_id}")
        keyboard.append([
            types.InlineKeyboardButton(text="Change", callback_data="vkset"),
            types.InlineKeyboardButton(text="Disable", callback_data="vkunset"),
        ])
    else:
        lines.append("VK group disabled")
        keyboard.append([
            types.InlineKeyboardButton(text="Set VK group", callback_data="vkset")
        ])
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
    group_id = await get_vk_group_id(db)
    if group_id:
        t_today = await get_vk_time_today(db)
        t_added = await get_vk_time_added(db)
        lines.append(f"VK group {group_id} {t_today}/{t_added}")
        keyboard.append([
            types.InlineKeyboardButton(text="Disable", callback_data="vkunset"),
            types.InlineKeyboardButton(text="Today", callback_data="vktime:today"),
            types.InlineKeyboardButton(text="Added", callback_data="vktime:added"),
            types.InlineKeyboardButton(text="Test today", callback_data="vkdailysend:today"),
            types.InlineKeyboardButton(text="Test added", callback_data="vkdailysend:added"),
        ])
    else:
        lines.append("VK group disabled")
        keyboard.append([
            types.InlineKeyboardButton(text="Set VK group", callback_data="vkset")
        ])
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
                types.InlineKeyboardButton(
                    text="Test tomorrow",
                    callback_data=f"dailysendtom:{ch.channel_id}",
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


async def handle_users(message: types.Message, db: Database, bot: Bot):
    await send_users_list(message, db, bot, edit=False)


async def handle_regdailychannels(message: types.Message, db: Database, bot: Bot):
    await send_regdaily_list(message, db, bot, edit=False)


async def handle_daily(message: types.Message, db: Database, bot: Bot):
    await send_daily_list(message, db, bot, edit=False)


async def upsert_event(session: AsyncSession, new: Event) -> Tuple[Event, bool]:
    """Insert or update an event if a similar one exists.

    Returns (event, added_flag)."""
    logging.info(
        "upsert_event: checking '%s' on %s %s",
        new.title,
        new.date,
        new.time,
    )

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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
            ev.pushkin_card = new.pushkin_card
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
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
                ev.pushkin_card = new.pushkin_card
                await session.commit()
                logging.info("upsert_event: updated event id=%s", ev.id)
                return ev, False
    new.added_at = datetime.utcnow()
    session.add(new)
    await session.commit()
    logging.info("upsert_event: inserted new event id=%s", new.id)
    return new, True


async def add_events_from_text(
    db: Database,
    text: str,
    source_link: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    *,
    raise_exc: bool = False,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
    creator_id: int | None = None,
    display_source: bool = True,
    source_channel: str | None = None,
    channel_title: str | None = None,


    bot: Bot | None = None,

) -> list[tuple[Event, bool, list[str], str]]:
    logging.info(
        "add_events_from_text start: len=%d source=%s", len(text), source_link
    )
    try:
        logging.info("LLM parse start (%d chars)", len(text))
        llm_text = text
        if channel_title:
            llm_text = f"{channel_title}\n{llm_text}"
        async with db.get_session() as session:
            res_f = await session.execute(select(Festival.name))
            fest_names = [r[0] for r in res_f.fetchall()]
        try:
            if source_channel:
                parsed = await parse_event_via_4o(
                    llm_text, source_channel, festival_names=fest_names
                )
            else:
                parsed = await parse_event_via_4o(llm_text, festival_names=fest_names)
        except TypeError:
            if source_channel:
                parsed = await parse_event_via_4o(llm_text, source_channel)
            else:
                parsed = await parse_event_via_4o(llm_text)

        logging.info("LLM returned %d events", len(parsed))
    except Exception as e:
        logging.error("LLM error: %s", e)
        if raise_exc:
            raise
        return []

    results: list[tuple[Event, bool, list[str], str]] = []
    first = True
    images: list[tuple[bytes, str]] = []
    if media:
        images = [media] if isinstance(media, tuple) else list(media)
    catbox_urls, catbox_msg_global = await upload_to_catbox(images)
    links_iter = iter(extract_links_from_html(html_text) if html_text else [])
    source_text_clean = re.sub(
        r"<[^>]+>", "", _vk_expose_links(html_text or text)
    )
    for data in parsed:
        logging.info(
            "processing event candidate: %s on %s %s",
            data.get("title"),
            data.get("date"),
            data.get("time"),
        )
        if data.get("festival"):
            logging.info(
                "4o recognized festival %s for event %s",
                data.get("festival"),
                data.get("title"),
            )

        date_raw = data.get("date", "") or ""
        end_date_raw = data.get("end_date") or None
        if end_date_raw and ".." in end_date_raw:
            end_date_raw = end_date_raw.split("..", 1)[-1].strip()
        if ".." in date_raw:
            start, maybe_end = [p.strip() for p in date_raw.split("..", 1)]
            date_raw = start
            if not end_date_raw:
                end_date_raw = maybe_end
        date_str = canonicalize_date(date_raw)
        end_date = canonicalize_date(end_date_raw) if end_date_raw else None


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
            pushkin_card=bool(data.get("pushkin_card")),
            source_text=source_text_clean,
            source_post_url=source_link,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            creator_id=creator_id,
        )

        if base_event.festival:
            photo_u = catbox_urls[0] if catbox_urls else None
            await ensure_festival(
                db,
                base_event.festival,
                full_name=data.get("festival_full"),
                photo_url=photo_u,
            )

        if base_event.event_type == "выставка" and not base_event.end_date:
            start_dt = parse_iso_date(base_event.date) or datetime.now(LOCAL_TZ).date()
            base_event.date = start_dt.isoformat()
            base_event.end_date = date(start_dt.year, 12, 31).isoformat()

        events_to_add = [base_event]
        if (
            base_event.event_type != "выставка"
            and base_event.end_date
            and base_event.end_date != base_event.date
        ):
            start_dt = parse_iso_date(base_event.date)
            end_dt = parse_iso_date(base_event.end_date) if base_event.end_date else None
            if start_dt and end_dt and end_dt > start_dt:
                events_to_add = []
                for i in range((end_dt - start_dt).days + 1):
                    day = start_dt + timedelta(days=i)
                    copy_e = Event(
                        **base_event.model_dump(
                            exclude={
                                "id",
                                "added_at",
                                "ics_post_url",
                                "ics_post_id",
                            }
                        )
                    )
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

            # skip events that have already finished - disabled for consistency in tests

            async with db.get_session() as session:
                saved, added = await upsert_event(session, event)
            logging.info(
                "event %s with id %s", "added" if added else "updated", saved.id
            )

            media_arg = None
            upload_info = catbox_msg_global if first else ""
            photo_count = saved.photo_count
            if saved.telegraph_url and saved.telegraph_path:
                upload_info, added_count = await update_source_page(
                    saved.telegraph_path,
                    saved.title or "Event",
                    html_text or text,
                    media_arg,
                    db,
                    catbox_urls=catbox_urls,
                )
                if added_count:
                    photo_count += added_count
                    async with db.get_session() as session:
                        saved.photo_count = photo_count
                        session.add(saved)
                        await session.commit()
                try:
                    await update_event_description(saved, db)
                except Exception as e:
                    logging.error("failed to update event %s description: %s", saved.id, e)
                if is_vk_wall_url(source_link):
                    vk_url = source_link
                else:
                    vk_url = await sync_vk_source_post(
                        saved,
                        saved.source_text,
                        db,
                        bot,
                        ics_url=saved.ics_url,
                    )
                if vk_url:
                    async with db.get_session() as session:
                        saved.source_vk_post_url = vk_url
                        session.add(saved)
                        await session.commit()
            else:
                if not saved.ics_url:
                    ics = await upload_ics(saved, db)
                    if ics:
                        logging.info("ICS saved for event %s: %s", saved.id, ics)
                        async with db.get_session() as session:
                            obj = await session.get(Event, saved.id)
                            if obj:
                                obj.ics_url = ics
                                await session.commit()
                                saved.ics_url = ics

                if bot and saved.ics_url and not saved.ics_post_url:
                    try:
                        posted = await asyncio.wait_for(
                            post_ics_asset(saved, db, bot), ICS_POST_TIMEOUT
                        )
                    except Exception as e:
                        logging.error("failed to post ics asset: %s", e)
                        posted = None
                    if posted:
                        url_p, msg_id = posted
                        logging.info(
                            "asset post %s for event %s", url_p, saved.id
                        )
                        async with db.get_session() as session:
                            obj = await session.get(Event, saved.id)
                            if obj:
                                obj.ics_post_url = url_p
                                obj.ics_post_id = msg_id
                                await session.commit()
                                saved.ics_post_url = url_p
                                saved.ics_post_id = msg_id
                        await add_calendar_button(saved, db, bot)
                        logging.info(
                            "calendar button added for event %s", saved.id
                        )
                        if saved.telegraph_path:
                            await update_source_page_ics(
                                saved.telegraph_path,
                                saved.title or "Event",
                                saved.ics_url,
                            )
                            if not is_vk_wall_url(source_link):
                                await sync_vk_source_post(
                                    saved,
                                    saved.source_text,
                                    db,
                                    bot,
                                    ics_url=saved.ics_url,
                                )
                extra_kwargs = {"display_link": False} if not display_source else {}
                res = await create_source_page(
                    saved.title or "Event",
                    saved.source_text,
                    source_link,
                    html_text,
                    None,
                    saved.ics_url,
                    db,
                    catbox_urls=catbox_urls,
                    **extra_kwargs,
                )
                if res:
                    if len(res) == 4:
                        url, path, _, photo_count = res
                    elif len(res) == 3:
                        url, path, _ = res
                        photo_count = 0
                    else:
                        url, path = res
                        photo_count = 0
                    upload_info = catbox_msg_global if first else ""
                    logging.info("telegraph page %s", url)
                    async with db.get_session() as session:
                        saved.telegraph_url = url
                        saved.telegraph_path = path
                        saved.photo_count = photo_count
                        session.add(saved)
                        await session.commit()
                    if is_vk_wall_url(source_link):
                        vk_url = source_link
                    else:
                        vk_url = await sync_vk_source_post(
                            saved,
                            saved.source_text,
                            db,
                            bot,
                            ics_url=saved.ics_url,
                        )
                    if vk_url:
                        async with db.get_session() as session:
                            saved.source_vk_post_url = vk_url
                            session.add(saved)
                            await session.commit()
            if saved.telegraph_path:
                try:
                    await update_event_description(saved, db)
                except Exception as e:
                    logging.error("failed to update event %s description: %s", saved.id, e)
            logging.info("syncing month page %s", saved.date[:7])
            asyncio.create_task(sync_month_page(db, saved.date[:7]))
            d_saved = parse_iso_date(saved.date)
            w_start = weekend_start_for_date(d_saved) if d_saved else None
            if w_start:
                logging.info("syncing weekend page %s", w_start.isoformat())
                asyncio.create_task(sync_weekend_page(db, w_start.isoformat()))
            fest_obj = None
            if saved.festival:
                logging.info("syncing festival %s", saved.festival)
                await sync_festival_page(db, saved.festival)
                asyncio.create_task(sync_festival_vk_post(db, saved.festival, bot))
                async with db.get_session() as session:
                    res = await session.execute(
                        select(Festival).where(Festival.name == saved.festival)
                    )
                    fest_obj = res.scalar_one_or_none()
            await asyncio.sleep(0)


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
                if fest_obj and fest_obj.telegraph_url:
                    lines.append(f"festival_page: {fest_obj.telegraph_url}")

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
            if saved.source_vk_post_url:
                if is_vk_wall_url(saved.source_post_url):
                    lines.append(f"vk_weekend_post: {saved.source_vk_post_url}")
                else:
                    lines.append(f"vk_source: {saved.source_vk_post_url}")
            elif is_vk_wall_url(saved.source_post_url):
                lines.append(f"Vk: {saved.source_post_url}")
            if upload_info:
                lines.append(f"catbox: {upload_info}")
            status = "added" if added else "updated"
            results.append((saved, added, lines, status))
            first = False
    logging.info("add_events_from_text finished with %d results", len(results))
    del parsed
    gc.collect()
    return results


async def handle_add_event(message: types.Message, db: Database, bot: Bot):
    using_session = False
    text_raw = message.text or message.caption or ""
    logging.info(
        "handle_add_event start: user=%s len=%d", message.from_user.id, len(text_raw)
    )
    if message.from_user.id in add_event_sessions:
        using_session = True
        add_event_sessions.discard(message.from_user.id)
        text_content = text_raw
    else:
        parts = text_raw.split(maxsplit=1)
        if len(parts) != 2:
            await bot.send_message(message.chat.id, "Usage: /addevent <text>")
            return
        text_content = parts[1]
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if user and user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    creator_id = user.user_id if user else message.from_user.id
    images = await extract_images(message, bot)
    media = images if images else None
    catbox_urls: list[str] | None = None
    catbox_msg = ""
    if images:
        catbox_urls, catbox_msg = await upload_to_catbox(images)
    catbox_urls: list[str] | None = None
    catbox_msg = ""
    if images:
        catbox_urls, catbox_msg = await upload_to_catbox(images)
    html_text = message.html_text or message.caption_html
    if not using_session and html_text and html_text.startswith("/addevent"):
        html_text = html_text[len("/addevent") :].lstrip()
    source_link = None
    lines = text_content.splitlines()
    if lines and is_vk_wall_url(lines[0].strip()):
        source_link = lines[0].strip()
        text_content = "\n".join(lines[1:]).lstrip()
        if html_text:
            html_lines = html_text.splitlines()
            if html_lines and is_vk_wall_url(html_lines[0].strip()):
                html_text = "\n".join(html_lines[1:]).lstrip()
    try:
        results = await add_events_from_text(
            db,
            text_content,
            source_link,
            html_text,
            media,
            raise_exc=True,
            creator_id=creator_id,
            display_source=False if source_link else True,
            source_channel=None,

            bot=bot,
        )
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    if not results:
        await bot.send_message(message.chat.id, "LLM error")
        return
    logging.info("handle_add_event parsed %d results", len(results))
    for saved, added, lines, status in results:
        logging.info(
            "handle_add_event %s event id=%s", status, saved.id
        )
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
        await notify_event_added(db, bot, user, saved, added)
        link_markup = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="Добавить ссылку на Вк этого мероприятия",
                        callback_data=f"vklink:{saved.id}",
                    ),
                    types.InlineKeyboardButton(text="Нет", callback_data="vklinkskip"),
                ]
            ]
        )
        await bot.send_message(
            message.chat.id,
            "Добавить ссылку на Вк этого мероприятия?",
            reply_markup=link_markup,
        )
    logging.info("handle_add_event finished for user %s", message.from_user.id)


async def handle_add_event_raw(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or message.caption or "").split(maxsplit=1)
    logging.info(
        "handle_add_event_raw start: user=%s text=%s",
        message.from_user.id,
        parts[1] if len(parts) > 1 else "",
    )
    if len(parts) != 2 or "|" not in parts[1]:
        await bot.send_message(
            message.chat.id, "Usage: /addevent_raw title|date|time|location"
        )
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if user and user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    creator_id = user.user_id if user else message.from_user.id
    title, date_raw, time, location = (p.strip() for p in parts[1].split("|", 3))
    date_iso = canonicalize_date(date_raw)
    if not date_iso:
        await bot.send_message(message.chat.id, "Invalid date")
        return
    images = await extract_images(message, bot)
    media = images if images else None
    catbox_urls: list[str] | None = None
    catbox_msg = ""
    if images:
        catbox_urls, catbox_msg = await upload_to_catbox(images)
    html_text = message.html_text or message.caption_html
    if html_text and html_text.startswith("/addevent_raw"):
        html_text = html_text[len("/addevent_raw") :].lstrip()
    source_clean = re.sub(r"<[^>]+>", "", _vk_expose_links(html_text or parts[1]))

    event = Event(
        title=title,
        description="",
        festival=None,
        date=date_iso,
        time=time,
        location_name=location,
        source_text=source_clean,
        creator_id=creator_id,
    )
    async with db.get_session() as session:
        event, added = await upsert_event(session, event)

    if not event.ics_url:
        ics = await upload_ics(event, db)
        if ics:
            async with db.get_session() as session:
                obj = await session.get(Event, event.id)
                if obj:
                    obj.ics_url = ics
                    await session.commit()
                    event.ics_url = ics

    res = await create_source_page(
        event.title or "Event",
        event.source_text,
        None,
        html_text or event.source_text,
        None,
        event.ics_url,
        db,
        catbox_urls=catbox_urls,
    )
    upload_info = catbox_msg
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
        if is_vk_wall_url(event.source_post_url):
            vk_url = event.source_post_url
        else:
            vk_url = await sync_vk_source_post(
                event,
                event.source_text,
                db,
                bot,
                ics_url=event.ics_url,
            )
        if vk_url:
            async with db.get_session() as session:
                event.source_vk_post_url = vk_url
                session.add(event)
                await session.commit()
    await sync_month_page(db, event.date[:7])
    d = parse_iso_date(event.date)
    w_start = weekend_start_for_date(d) if d else None
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
    if event.source_vk_post_url:
        if is_vk_wall_url(event.source_post_url):
            lines.append(f"vk_weekend_post: {event.source_vk_post_url}")
        else:
            lines.append(f"vk_source: {event.source_vk_post_url}")
    elif is_vk_wall_url(event.source_post_url):
        lines.append(f"Vk: {event.source_post_url}")
    if upload_info:
        lines.append(f"catbox: {upload_info}")
    status = "added" if added else "updated"
    logging.info("handle_add_event_raw %s event id=%s", status, event.id)
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
    await notify_event_added(db, bot, user, event, added)
    link_markup = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="Добавить ссылку на Вк этого мероприятия",
                    callback_data=f"vklink:{event.id}",
                ),
                types.InlineKeyboardButton(text="Нет", callback_data="vklinkskip"),
            ]
        ]
    )
    await bot.send_message(
        message.chat.id,
        "Добавить ссылку на Вк этого мероприятия?",
        reply_markup=link_markup,
    )
    logging.info("handle_add_event_raw finished for user %s", message.from_user.id)


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


DATE_WORDS = "|".join(MONTHS)
RE_FEST_RANGE = re.compile(
    rf"(?:\bс\s*)?(\d{{1,2}}\s+(?:{DATE_WORDS})(?:\s+\d{{4}})?)"
    rf"\s*(?:по|\-|–|—)\s*"
    rf"(\d{{1,2}}\s+(?:{DATE_WORDS})(?:\s+\d{{4}})?)",
    re.IGNORECASE,
)
RE_FEST_SINGLE = re.compile(
    rf"(\d{{1,2}}\s+(?:{DATE_WORDS})(?:\s+\d{{4}})?)",
    re.IGNORECASE,
)


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
    if int(y) != datetime.now(LOCAL_TZ).year:
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


def is_vk_wall_url(url: str | None) -> bool:
    """Return True if the URL points to a VK wall post."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = parsed.netloc.lower()
    if host in {"vk.cc", "vk.link", "go.vk.com", "l.vk.com"}:
        return False
    if not host.endswith("vk.com"):
        return False
    if "/wall" in parsed.path:
        return True
    query = parse_qs(parsed.query)
    if "w" in query and any(v.startswith("wall") for v in query["w"]):
        return True
    if "z" in query and any("wall" in v for v in query["z"]):
        return True
    return False


def recent_cutoff(tz: timezone, now: datetime | None = None) -> datetime:
    """Return UTC datetime for the start of the previous day in the given tz."""
    if now is None:
        now = datetime.now(tz)
    start_local = datetime.combine(
        now.date() - timedelta(days=1),
        time(0, 0),
        tz,
    )
    return start_local.astimezone(timezone.utc).replace(tzinfo=None)


def week_cutoff(tz: timezone, now: datetime | None = None) -> datetime:
    """Return UTC datetime for 7 days ago."""
    if now is None:
        now = datetime.now(tz)
    return (
        now.astimezone(timezone.utc).replace(tzinfo=None) - timedelta(days=7)
    )


def split_text(text: str, limit: int = 4096) -> list[str]:
    """Split text into chunks without breaking lines."""
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        parts.append(text)
    return parts


def is_recent(e: Event, tz: timezone | None = None, now: datetime | None = None) -> bool:
    if e.added_at is None or e.silent:
        return False
    if tz is None:
        tz = LOCAL_TZ
    start = recent_cutoff(tz, now)
    return e.added_at >= start


def format_event_md(e: Event, festival: Festival | None = None) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "
    lines = [f"{prefix}{emoji_part}{e.title}".strip()]
    if festival:
        link = festival.telegraph_url
        if link:
            lines.append(f"[{festival.name}]({link})")
        else:
            lines.append(festival.name)
    lines.append(e.description.strip())
    if e.pushkin_card:
        lines.append("\u2705 Пушкинская карта")
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
        more_line = f"{prefix}[подробнее]({e.telegraph_url})"
        if e.ics_post_url:
            more_line += f" \U0001f4c5 [добавить в календарь]({e.ics_post_url})"
        lines.append(more_line)
    loc = e.location_name
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {addr}"
    if e.city:
        loc += f", #{e.city}"
    date_part = e.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", e.date)
        day = e.date
    lines.append(f"_{day} {e.time} {loc}_")
    return "\n".join(lines)



def format_event_vk(
    e: Event,
    highlight: bool = False,
    weekend_url: str | None = None,
    festival: Festival | None = None,
) -> str:

    prefix = ""
    if highlight:
        prefix += "\U0001f449 "
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "

    vk_link = e.source_post_url if is_vk_wall_url(e.source_post_url) else None
    if not vk_link and is_vk_wall_url(e.source_vk_post_url):
        vk_link = e.source_vk_post_url

    title_text = f"{emoji_part}{e.title.upper()}".strip()
    if vk_link:
        title = f"{prefix}[{vk_link}|{title_text}]".strip()
    else:
        title = f"{prefix}{title_text}".strip()

    desc = re.sub(
        r",?\s*подробнее\s*\([^\n]*\)$",
        "",
        e.description.strip(),
        flags=re.I,
    )
    details_link = None
    if vk_link:
        details_link = vk_link
    elif e.telegraph_url:
        details_link = e.telegraph_url
    if details_link:

        desc = f"{desc}, [подробнее|{details_link}]"

    lines = [title]
    if festival:
        link = festival.vk_url or festival.vk_post_url
        prefix = "✨ "
        if link:
            lines.append(f"{prefix}[{link}|{festival.name}]")
        else:
            lines.append(f"{prefix}{festival.name}")
    lines.append(desc)

    if e.pushkin_card:
        lines.append("\u2705 Пушкинская карта")

    if e.is_free:
        lines.append("🟡 Бесплатно")
        if e.ticket_link:
            lines.append("по регистрации")
            lines.append(f"\U0001f39f {e.ticket_link}")
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max} руб."
        else:
            val = e.ticket_price_min if e.ticket_price_min is not None else e.ticket_price_max
            price = f"{val} руб." if val is not None else ""
        lines.append(f"Билеты в источнике {price}".strip())
        lines.append(f"\U0001f39f {e.ticket_link}")
    elif e.ticket_link:
        lines.append("по регистрации")
        lines.append(f"\U0001f39f {e.ticket_link}")
    else:
        price = ""
        if (
            e.ticket_price_min is not None
            and e.ticket_price_max is not None
            and e.ticket_price_min != e.ticket_price_max
        ):
            price = f"от {e.ticket_price_min} до {e.ticket_price_max} руб."
        elif e.ticket_price_min is not None:
            price = f"{e.ticket_price_min} руб."
        elif e.ticket_price_max is not None:
            price = f"{e.ticket_price_max} руб."
        if price:
            lines.append(f"Билеты {price}")

    # details link already appended to description above

    loc = e.location_name
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {addr}"
    if e.city:
        loc += f", #{e.city}"
    date_part = e.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", e.date)
        day = e.date
    if weekend_url and d and d.weekday() == 5:
        day_fmt = f"{day}"
    else:
        day_fmt = day
    lines.append(f"\U0001f4c5 {day_fmt} {e.time}")
    lines.append(loc)

    return "\n".join(lines)


def format_event_daily(
    e: Event,
    highlight: bool = False,
    weekend_url: str | None = None,
    festival: Festival | None = None,
) -> str:
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

    desc = e.description.strip()
    desc = re.sub(r",?\s*подробнее\s*\([^\n]*\)$", "", desc, flags=re.I)
    lines = [title]
    if festival:
        link = festival.telegraph_url
        if link:
            lines.append(f'<a href="{html.escape(link)}">{html.escape(festival.name)}</a>')
        else:
            lines.append(html.escape(festival.name))
    lines.append(html.escape(desc))

    if e.pushkin_card:
        lines.append("\u2705 Пушкинская карта")

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
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", e.date)
        day = e.date
    if weekend_url and d and d.weekday() == 5:
        day_fmt = f'<a href="{html.escape(weekend_url)}">{day}</a>'
    else:
        day_fmt = day
    lines.append(f"<i>{day_fmt} {e.time} {loc}</i>")

    return "\n".join(lines)


def format_exhibition_md(e: Event) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "
    lines = [f"{prefix}{emoji_part}{e.title}".strip(), e.description.strip()]
    if e.pushkin_card:
        lines.append("\u2705 Пушкинская карта")
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
        d_end = parse_iso_date(end_part)
        if d_end:
            end = format_day_pretty(d_end)
        else:
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


def event_to_nodes(
    e: Event, festival: Festival | None = None, fest_icon: bool = False
) -> list[dict]:
    md = format_event_md(e, festival)

    lines = md.split("\n")
    body_lines = lines[1:]
    if festival and body_lines:
        body_lines = body_lines[1:]
    body_md = "\n".join(body_lines) if body_lines else ""
    from telegraph.utils import html_to_nodes

    nodes = [{"tag": "h4", "children": event_title_nodes(e)}]
    if festival or e.festival:
        fest = festival
        if fest is None and e.festival:
            # caller typically provides the object
            pass
        if fest:
            prefix = "✨ " if fest_icon else ""
            if fest.telegraph_url:
                children = []
                if prefix:
                    children.append(prefix)
                children.append(
                    {
                        "tag": "a",
                        "attrs": {"href": fest.telegraph_url},
                        "children": [fest.name],
                    }
                )
                nodes.append({"tag": "p", "children": children})
            else:
                text = f"{prefix}{fest.name}" if prefix else fest.name
                nodes.append({"tag": "p", "children": [text]})
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

async def get_month_data(db: Database, month: str, *, fallback: bool = True):
    """Return events, exhibitions and nav pages for the given month."""
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


    today = datetime.now(LOCAL_TZ).date()

    if month == today.strftime("%Y-%m"):
        today_str = today.isoformat()
        cutoff = (today - timedelta(days=30)).isoformat()
        events = [e for e in events if e.date.split("..", 1)[0] >= today_str]
        exhibitions = [
            e for e in exhibitions if e.end_date and e.end_date >= cutoff
        ]

    if not exhibitions and fallback:
        prev_month = (start - timedelta(days=1)).strftime("%Y-%m")
        if prev_month != month:
            prev_events, prev_exh, _ = await get_month_data(db, prev_month, fallback=False)
            if not events:
                events.extend(prev_events)
            exhibitions.extend(prev_exh)

    return events, exhibitions, nav_pages


async def build_month_page_content(
    db: Database,
    month: str,
    events: list[Event] | None = None,
    exhibitions: list[Event] | None = None,
    nav_pages: list[MonthPage] | None = None,
    continuation_url: str | None = None,
) -> tuple[str, list]:
    if events is None or exhibitions is None or nav_pages is None:
        events, exhibitions, nav_pages = await get_month_data(db, month)

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fest_map = {f.name: f for f in res_f.scalars().all()}


    today = datetime.now(LOCAL_TZ).date()
    cutoff = (today - timedelta(days=30)).isoformat()

    if month == today.strftime("%Y-%m"):
        events = [e for e in events if e.date.split("..", 1)[0] >= cutoff]
        exhibitions = [e for e in exhibitions if e.end_date and e.end_date >= cutoff]

    today_str = today.isoformat()
    events = [
        e
        for e in events
        if not (e.event_type == "выставка" and e.date < today_str)
    ]
    exhibitions = [
        e for e in exhibitions if e.end_date and e.date <= today_str
    ]

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fest_map = {f.name: f for f in res_f.scalars().all()}

    by_day: dict[date, list[Event]] = {}
    for e in events:
        date_part = e.date.split("..", 1)[0]
        d = parse_iso_date(date_part)
        if not d:
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
            fest = fest_map.get(ev.festival or "")
            content.extend(event_to_nodes(ev, fest, fest_icon=True))


    future_pages = [p for p in nav_pages if p.month >= month]
    month_nav: list[dict] = []
    nav_children = []
    if future_pages:
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
    else:
        nav_children = [month_name_nominative(month)]

    if nav_children:
        month_nav = [{"tag": "br"}, {"tag": "h4", "children": nav_children}]
        content.extend(month_nav)

    if exhibitions:
        if month_nav:
            content.append({"tag": "br"})
            content.append({"tag": "p", "children": ["\u00a0"]})
        content.append({"tag": "h3", "children": ["Постоянные выставки"]})
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        for ev in exhibitions:
            content.extend(exhibition_to_nodes(ev))

    if month_nav:
        content.extend(month_nav)

    if continuation_url:
        content.append({"tag": "br"})
        content.append({"tag": "p", "children": ["\u00a0"]})
        content.append(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": continuation_url},
                        "children": [f"{month_name_nominative(month)} продолжение"],
                    }
                ],
            }
        )

    title = f"События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
    return title, content


async def sync_month_page(db: Database, month: str, update_links: bool = False):
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
                page = MonthPage(month=month, url="", path="")
                session.add(page)
                await session.commit()
                created = True

            events, exhibitions, nav_pages = await get_month_data(db, month)


            async def split_and_update():
                """Split the month into two pages keeping the first path."""
                # Find maximum number of events that fit on the first page
                low, high, best = 1, len(events) - 1, 1
                while low <= high:
                    mid = (low + high) // 2
                    _, c = await build_month_page_content(
                        db, month, events[:mid], exhibitions, nav_pages
                    )
                    if len(json_dumps(c).encode("utf-8")) <= TELEGRAPH_PAGE_LIMIT:
                        best = mid
                        low = mid + 1
                    else:
                        high = mid - 1

                first = events[:best]
                second = events[best:]

                title2, content2 = await build_month_page_content(
                    db, month, second, exhibitions, nav_pages
                )
                if not page.path2:
                    data2 = await telegraph_call(tg.create_page, title2, content=content2)
                    page.url2 = data2.get("url")
                    page.path2 = data2.get("path")
                else:
                    await telegraph_call(
                        tg.edit_page, page.path2, title=title2, content=content2
                    )

                title1, content1 = await build_month_page_content(
                    db, month, first, [], nav_pages, continuation_url=page.url2
                )
                if not page.path:
                    data1 = await telegraph_call(tg.create_page, title1, content=content1)
                    page.url = data1.get("url")
                    page.path = data1.get("path")
                else:
                    await telegraph_call(
                        tg.edit_page, page.path, title=title1, content=content1
                    )
                logging.info(
                    "%s month page %s split into two", "Created" if created else "Edited", month
                )
                await session.commit()


            title, content = await build_month_page_content(
                db, month, events, exhibitions, nav_pages
            )
            size = len(json_dumps(content).encode("utf-8"))


            try:
                if size <= TELEGRAPH_PAGE_LIMIT:
                    if not page.path:
                        data = await telegraph_call(tg.create_page, title, content=content)
                        page.url = data.get("url")
                        page.path = data.get("path")
                    else:
                        await telegraph_call(
                            tg.edit_page, page.path, title=title, content=content
                        )
                    page.url2 = None
                    page.path2 = None
                    logging.info(
                        "%s month page %s", "Created" if created else "Edited", month
                    )
                    await session.commit()
                else:
                    await split_and_update()
            except TelegraphException as e:
                if "CONTENT_TOO_BIG" in str(e):
                    logging.warning("Month page %s too big, splitting", month)
                    await split_and_update()
                else:
                    raise

        except Exception as e:
            logging.error("Failed to sync month page %s: %s", month, e)

    if update_links or created:
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
        res_f = await session.execute(select(Festival))
        fest_map = {f.name: f for f in res_f.scalars().all()}

    today = datetime.now(LOCAL_TZ).date()
    events = [
        e
        for e in events
        if (
            (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        )
    ]

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fest_map = {f.name: f for f in res_f.scalars().all()}

    by_day: dict[date, list[Event]] = {}
    for e in events:
        d = parse_iso_date(e.date)
        if not d:
            continue
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    content.append(
        {
            "tag": "p",
            "children": [
                "Вот что рекомендуют ",
                {
                    "tag": "a",
                    "attrs": {"href": "https://t.me/kenigevents"},
                    "children": ["Полюбить Калининград Анонсы"],
                },
                " чтобы провести выходные ярко: события Калининградской области и 39 региона — концерты, спектакли, фестивали.",
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
            fest = fest_map.get(ev.festival or "")
            content.extend(event_to_nodes(ev, fest, fest_icon=True))


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
    future_months = [m for m in month_pages if m.month >= cur_month]
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


async def sync_weekend_page(
    db: Database, start: str, update_links: bool = False, post_vk: bool = True
):
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
                data = await telegraph_call(tg.create_page, title, content=content)
                page = WeekendPage(
                    start=start, url=data.get("url"), path=data.get("path")
                )
                session.add(page)
                await session.commit()
                created = True

            # Rebuild content including this page in navigation
            title, content = await build_weekend_page_content(db, start)
            await telegraph_call(
                tg.edit_page, page.path, title=title, content=content
            )
            logging.info(
                "%s weekend page %s", "Created" if created else "Edited", start
            )
            await session.commit()
        except Exception as e:
            logging.error("Failed to sync weekend page %s: %s", start, e)

    if post_vk:
        await sync_vk_weekend_post(db, start)

    if update_links or created:
        async with db.get_session() as session:
            result = await session.execute(
                select(WeekendPage).order_by(WeekendPage.start)
            )
            weekends = result.scalars().all()
        for w in weekends:
            if w.start != start:
                await sync_weekend_page(
                    db, w.start, update_links=False, post_vk=False
                )


async def build_weekend_vk_message(db: Database, start: str) -> str:
    logging.info("build_weekend_vk_message start for %s", start)
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
        res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
        weekend_pages = res_w.scalars().all()

    by_day: dict[date, list[Event]] = {}
    for e in events:
        if not e.source_vk_post_url:
            continue
        d = parse_iso_date(e.date)
        if not d:
            continue
        by_day.setdefault(d, []).append(e)

    lines = [f"{format_weekend_range(saturday)} Афиша выходных"]
    for d in days:
        evs = by_day.get(d)
        if not evs:
            continue
        lines.append(VK_BLANK_LINE)
        lines.append(f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥")
        for ev in evs:
            line = f"[{ev.source_vk_post_url}|{ev.title}]"
            if ev.time:
                line = f"{ev.time} | {line}"
            lines.append(line)

            location_parts = [p for p in [ev.location_name, ev.city] if p]
            if location_parts:
                lines.append(", ".join(location_parts))

    nav_pages = [w for w in weekend_pages if w.vk_post_url or w.start == start]
    if nav_pages:
        parts = []
        for w in nav_pages:
            label = format_weekend_range(date.fromisoformat(w.start))
            if w.start == start or not w.vk_post_url:
                parts.append(label)
            else:
                parts.append(f"[{w.vk_post_url}|{label}]")
        lines.append(VK_BLANK_LINE)
        lines.append(VK_BLANK_LINE)
        lines.append(" ".join(parts))

    message = "\n".join(lines)
    logging.info(
        "build_weekend_vk_message built %d lines", len(lines)
    )
    return message


async def sync_vk_weekend_post(db: Database, start: str, bot: Bot | None = None) -> None:
    logging.info("sync_vk_weekend_post start for %s", start)
    group_id = VK_AFISHA_GROUP_ID
    if not group_id:
        logging.info("sync_vk_weekend_post: VK group not configured")
        return
    async with db.get_session() as session:
        page = await session.get(WeekendPage, start)
    if not page:
        logging.info("sync_vk_weekend_post: weekend page %s not found", start)
        return

    message = await build_weekend_vk_message(db, start)
    logging.info("sync_vk_weekend_post message len=%d", len(message))
    needs_new_post = not page.vk_post_url
    if page.vk_post_url:
        try:
            updated = await edit_vk_post(page.vk_post_url, message, db, bot)
            if updated:
                logging.info("sync_vk_weekend_post updated %s", page.vk_post_url)
            else:
                logging.info(
                    "sync_vk_weekend_post: no changes for %s", page.vk_post_url
                )
        except Exception as e:
            if "post or comment deleted" in str(e) or "Пост удалён" in str(e):
                logging.warning(
                    "sync_vk_weekend_post: original VK post missing, creating new"
                )
                needs_new_post = True
            else:
                logging.error("VK post error for weekend %s: %s", start, e)
                return
    if needs_new_post:
        url = await post_to_vk(group_id, message, db, bot)
        if url:
            async with db.get_session() as session:
                obj = await session.get(WeekendPage, start)
                if obj:
                    obj.vk_post_url = url
                    await session.commit()
            logging.info("sync_vk_weekend_post created %s", url)


async def generate_festival_description(fest: Festival, events: list[Event]) -> str:
    """Use LLM to craft a short festival blurb."""
    texts = [e.source_text for e in events[:5]]
    if fest.description:
        texts.insert(0, fest.description)
    prompt = (
        f"Напиши краткое описание фестиваля {fest.name}. "
        "Стиль профессионального журналиста в сфере мероприятий и культуры. "
        "Не используй типовые штампы и не придумывай факты. "
        "Описание должно состоять из трёх предложений, если сведений мало — из одного. "
        "Используй только информацию из этих текстов:\n\n" + "\n\n".join(texts)
    )
    try:
        text = await ask_4o(prompt)
        logging.info("generated description for festival %s", fest.name)
        return text.strip()
    except Exception as e:
        logging.error("failed to generate festival description %s: %s", fest.name, e)
        return ""


async def generate_festival_poll_text(fest: Festival) -> str:
    """Use LLM to craft poll question for VK."""
    base = (
        "Придумай короткий вопрос, приглашая читателей поделиться,"\
        f" пойдут ли они на фестиваль {fest.name}. "
        "Не повторяй дословно 'Друзья, а вы пойдёте на фестиваль'."
    )
    try:
        text = await ask_4o(base)
        text = text.strip()
    except Exception as e:
        logging.error("failed to generate poll text %s: %s", fest.name, e)
        text = f"Пойдёте ли вы на фестиваль {fest.name}?"
    if fest.vk_post_url:
        text += f"\n{fest.vk_post_url}"
    return text



async def build_festival_page_content(db: Database, fest: Festival) -> tuple[str, list]:
    logging.info("building festival page content for %s", fest.name)

    async with db.get_session() as session:
        res = await session.execute(
            select(Event).where(Event.festival == fest.name).order_by(Event.date, Event.time)
        )
        events = res.scalars().all()

        logging.info("festival %s has %d events", fest.name, len(events))

        desc = await generate_festival_description(fest, events)
        if desc:
            fest.description = desc
            await session.commit()

    nodes: list[dict] = []
    if fest.photo_url:
        nodes.append({"tag": "img", "attrs": {"src": fest.photo_url}})
        nodes.append({"tag": "p", "children": ["\u00a0"]})
    if events:
        start, end = festival_dates(fest, events)
        if start:
            date_text = format_day_pretty(start)
            if end and end != start:
                date_text += f" - {format_day_pretty(end)}"
            nodes.append({"tag": "p", "children": [f"\U0001f4c5 {date_text}"]})
        loc_text = festival_location(events)
        if loc_text:
            nodes.append({"tag": "p", "children": [f"\U0001f4cd {loc_text}"]})
    if fest.description:
        nodes.append({"tag": "p", "children": [fest.description]})

    if fest.website_url or fest.vk_url or fest.tg_url:
        nodes.append({"tag": "br"})
        nodes.append({"tag": "p", "children": ["\u00a0"]})

        nodes.append({"tag": "h3", "children": ["Контакты фестиваля"]})
        if fest.website_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "сайт: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.website_url},
                            "children": [fest.website_url],
                        },
                    ],
                }
            )
        if fest.vk_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "вк: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.vk_url},
                            "children": [fest.vk_url],
                        },
                    ],
                }
            )
        if fest.tg_url:
            nodes.append(
                {
                    "tag": "p",
                    "children": [
                        "телеграм: ",
                        {
                            "tag": "a",
                            "attrs": {"href": fest.tg_url},
                            "children": [fest.tg_url],
                        },
                    ],
                }
            )

    if events:
        nodes.append({"tag": "br"})
        nodes.append({"tag": "p", "children": ["\u00a0"]})
        nodes.append({"tag": "h3", "children": ["Мероприятия фестиваля"]})
        for e in events:
            nodes.extend(event_to_nodes(e))
    fest_list = await build_festivals_list_nodes(db, exclude=fest.name)
    if fest_list:
        nodes.append({"tag": "br"})
        nodes.append({"tag": "p", "children": ["\u00a0"]})
        nodes.extend(fest_list)
    title = fest.full_name or fest.name
    return title, nodes



async def sync_festival_page(db: Database, name: str):
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.name == name)
        )
        fest = result.scalar_one_or_none()

        if not fest:
            return
        try:
            title, content = await build_festival_page_content(db, fest)

            created = False
            if fest.telegraph_path:
                await telegraph_call(
                    tg.edit_page, fest.telegraph_path, title=title, content=content
                )
                logging.info("updated festival page %s in Telegraph", name)
            else:
                data = await telegraph_call(tg.create_page, title, content=content)
                fest.telegraph_url = data.get("url")
                fest.telegraph_path = data.get("path")
                created = True
                logging.info("created festival page %s: %s", name, fest.telegraph_url)
            await session.commit()
            logging.info("synced festival page %s", name)

        except Exception as e:
            logging.error("Failed to sync festival %s: %s", name, e)



async def build_festival_vk_message(db: Database, fest: Festival) -> str:
    async with db.get_session() as session:
        res = await session.execute(
            select(Event).where(Event.festival == fest.name).order_by(Event.date, Event.time)
        )
        events = res.scalars().all()
    lines = [fest.full_name or fest.name]
    if events:
        start, end = festival_date_range(events)

        if start:
            date_text = format_day_pretty(start)
            if end and end != start:
                date_text += f" - {format_day_pretty(end)}"
            lines.append(f"\U0001f4c5 {date_text}")
        loc_text = festival_location(events)
        if loc_text:
            lines.append(f"\U0001f4cd {loc_text}")
    if fest.description:
        lines.append(fest.description)
    if fest.website_url or fest.vk_url or fest.tg_url:
        lines.append(VK_BLANK_LINE)
        lines.append("Контакты фестиваля")
        if fest.website_url:
            lines.append(f"сайт: {fest.website_url}")
        if fest.vk_url:
            lines.append(f"вк: {fest.vk_url}")
        if fest.tg_url:
            lines.append(f"телеграм: {fest.tg_url}")
    for ev in events:
        lines.append(VK_BLANK_LINE)
        lines.append(format_event_vk(ev))
    fest_lines = await build_festivals_list_lines_vk(db, exclude=fest.name)
    if fest_lines:
        lines.append(VK_BLANK_LINE)
        lines.extend(fest_lines)
    return "\n".join(lines)


async def sync_festival_vk_post(db: Database, name: str, bot: Bot | None = None):
    group_id = await get_vk_group_id(db)
    if not group_id:
        return
    async with db.get_session() as session:
        res = await session.execute(select(Festival).where(Festival.name == name))
        fest = res.scalar_one_or_none()
        if not fest:
            return
    message = await build_festival_vk_message(db, fest)
    attachments: list[str] | None = None
    if fest.photo_url:
        if VK_PHOTOS_ENABLED:
            photo_id = await upload_vk_photo(group_id, fest.photo_url, db, bot)
            if photo_id:
                attachments = [photo_id]
        else:
            logging.info("VK photo posting disabled")
    try:
        if fest.vk_post_url:
            await edit_vk_post(fest.vk_post_url, message, db, bot, attachments)
            logging.info("updated festival post %s on VK", name)
        else:
            url = await post_to_vk(group_id, message, db, bot, attachments)
            if url:
                async with db.get_session() as session:
                    fest = (await session.execute(select(Festival).where(Festival.name == name))).scalar_one()
                    fest.vk_post_url = url
                    await session.commit()
            logging.info("created festival post %s: %s", name, url)
    except Exception as e:
        logging.error("VK post error for festival %s: %s", name, e)


async def send_festival_poll(
    db: Database,
    fest: Festival,
    group_id: str,
    bot: Bot | None = None,
) -> None:
    question = await generate_festival_poll_text(fest)
    url = await post_vk_poll(group_id, question, VK_POLL_OPTIONS, db, bot)
    if url:

        async with db.get_session() as session:
            obj = await session.get(Festival, fest.id)
            if obj:
                obj.vk_poll_url = url
                await session.commit()
        if bot:
            await notify_superadmin(db, bot, f"poll created {url}")




async def build_daily_posts(
    db: Database,
    tz: timezone,
    now: datetime | None = None,
) -> list[tuple[str, types.InlineKeyboardMarkup | None]]:
    if now is None:
        now = datetime.now(tz)
    today = now.date()
    yesterday_utc = recent_cutoff(tz, now)
    fest_map: dict[str, Festival] = {}
    async with db.get_session() as session:
        res_today = await session.execute(
            select(Event)
            .where(Event.date == today.isoformat())
            .order_by(Event.time)
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
        res_w_all = await session.execute(select(WeekendPage))
        weekend_map = {w.start: w for w in res_w_all.scalars().all()}
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

        res_fests = await session.execute(select(Festival))
        fest_map = {f.name: f for f in res_fests.scalars().all()}

        weekend_count = 0
        if wpage:
            sat = w_start
            sun = w_start + timedelta(days=1)
            weekend_new = [
                e
                for e in new_events
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    e.event_type == "выставка"
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_today = [
                e
                for e in events_today
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    e.event_type == "выставка"
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_count = max(0, len(weekend_new) - len(weekend_today))

        cur_count = 0
        next_count = 0
        for e in new_events:
            m = e.date[:7]
            if m == cur_month:
                cur_count += 1
            elif m == next_month(cur_month):
                next_count += 1

    tag = f"{today.day}{MONTHS[today.month - 1]}"
    lines1 = [
        f"<b>АНОНС на {format_day_pretty(today)} {today.year} #ежедневныйанонс</b>",
        DAYS_OF_WEEK[today.weekday()],
        "",
        "<b><i>НЕ ПРОПУСТИТЕ СЕГОДНЯ</i></b>",
    ]
    for e in events_today:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.url
        lines1.append("")
        lines1.append(
            format_event_daily(
                e,
                highlight=True,
                weekend_url=w_url,
                festival=fest_map.get(e.festival or ""),
            )
        )
    lines1.append("")
    lines1.append(
        f"#Афиша_Калининград #Калининград #концерт #{tag} #{today.day}_{MONTHS[today.month - 1]}"
    )
    section1 = "\n".join(lines1)

    lines2 = [f"<b><i>+{len(events_new)} ДОБАВИЛИ В АНОНС</i></b>"]
    for e in events_new:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.url
        lines2.append("")
        lines2.append(
            format_event_daily(
                e,
                weekend_url=w_url,
                festival=fest_map.get(e.festival or ""),
            )
        )
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

    posts: list[tuple[str, types.InlineKeyboardMarkup | None]] = []
    for part in split_text(section1):
        posts.append((part, None))
    section2_parts = split_text(section2)
    for part in section2_parts[:-1]:
        posts.append((part, None))
    posts.append((section2_parts[-1], markup))
    return posts


async def build_daily_sections_vk(
    db: Database,
    tz: timezone,
    now: datetime | None = None,
) -> tuple[str, str]:
    if now is None:
        now = datetime.now(tz)
    today = now.date()
    yesterday_utc = recent_cutoff(tz, now)
    fest_map: dict[str, Festival] = {}
    async with db.get_session() as session:
        res_today = await session.execute(
            select(Event)
            .where(Event.date == today.isoformat())
            .order_by(Event.time)
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
        res_w_all = await session.execute(select(WeekendPage))
        weekend_map = {w.start: w for w in res_w_all.scalars().all()}
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
            weekend_new = [
                e
                for e in new_events
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    e.event_type == "выставка"
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_today = [
                e
                for e in events_today
                if e.date in {sat.isoformat(), sun.isoformat()}
                or (
                    e.event_type == "выставка"
                    and e.end_date
                    and e.end_date >= sat.isoformat()
                    and e.date <= sun.isoformat()
                )
            ]
            weekend_count = max(0, len(weekend_new) - len(weekend_today))

        cur_count = 0
        next_count = 0
        for e in new_events:
            m = e.date[:7]
            if m == cur_month:
                cur_count += 1
            elif m == next_month(cur_month):
                next_count += 1

    lines1 = [
        f"\U0001f4c5 АНОНС на {format_day_pretty(today)} {today.year}",
        DAYS_OF_WEEK[today.weekday()],
        "",
        "НЕ ПРОПУСТИТЕ СЕГОДНЯ",
    ]
    for e in events_today:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.url
        lines1.append(
            format_event_vk(
                e,
                highlight=True,
                weekend_url=w_url,
                festival=fest_map.get(e.festival or ""),
            )
        )
        lines1.append(VK_EVENT_SEPARATOR)
    if events_today:
        lines1.pop()
    link_lines: list[str] = []
    if wpage:
        sunday = w_start + timedelta(days=1)
        prefix = f"(+{weekend_count}) " if weekend_count else ""
        link_lines.append(
            f"{prefix}выходные {w_start.day} {sunday.day} {MONTHS[w_start.month - 1]}: {wpage.url}"

        )
    if mp_cur:
        prefix = f"(+{cur_count}) " if cur_count else ""
        link_lines.append(
            f"{prefix}{month_name_nominative(cur_month)}: {mp_cur.url}"

        )
    if mp_next:
        prefix = f"(+{next_count}) " if next_count else ""
        link_lines.append(
            f"{prefix}{month_name_nominative(next_month(cur_month))}: {mp_next.url}"
        )
    if link_lines:
        lines1.append(VK_EVENT_SEPARATOR)
        lines1.extend(link_lines)
    lines1.append(VK_EVENT_SEPARATOR)
    lines1.append(
        f"#Афиша_Калининград #кудапойти_Калининград #Калининград #39region #концерт #{today.day}{MONTHS[today.month - 1]}"
    )
    section1 = "\n".join(lines1)

    lines2 = [f"+{len(events_new)} ДОБАВИЛИ В АНОНС", VK_BLANK_LINE]
    for e in events_new:
        w_url = None
        d = parse_iso_date(e.date)
        if d and d.weekday() == 5:
            w = weekend_map.get(d.isoformat())
            if w:
                w_url = w.url
        lines2.append(
            format_event_vk(
                e,
                weekend_url=w_url,
                festival=fest_map.get(e.festival or ""),
            )
        )
        lines2.append(VK_EVENT_SEPARATOR)
    if events_new:
        lines2.pop()
    if link_lines:
        lines2.append(VK_EVENT_SEPARATOR)
        lines2.extend(link_lines)
    lines2.append(VK_EVENT_SEPARATOR)
    lines2.append(
        f"#события_Калининград #Калининград #39region #новое #фестиваль #{today.day}{MONTHS[today.month - 1]}"
    )
    section2 = "\n".join(lines2)

    return section1, section2


async def post_to_vk(
    group_id: str,
    message: str,
    db: Database | None = None,
    bot: Bot | None = None,
    attachments: list[str] | None = None,
    token: str | None = None,
) -> str | None:
    if not group_id:
        return None
    logging.info(
        "post_to_vk start: group=%s len=%d attachments=%d",
        group_id,
        len(message),
        len(attachments or []),
    )
    params = {
        "owner_id": f"-{group_id.lstrip('-')}",
        "from_group": 1,
        "message": message,
    }
    if attachments:
        params["attachments"] = ",".join(attachments)
    data = await _vk_api("wall.post", params, db, bot, token=token)
    post_id = data.get("response", {}).get("post_id")
    if post_id:
        url = f"https://vk.com/wall-{group_id.lstrip('-')}_{post_id}"
        logging.info("post_to_vk success: %s", url)
        return url
    logging.error("post_to_vk failed for group %s", group_id)
    return None


async def create_vk_poll(
    group_id: str,
    question: str,
    options: list[str],
    db: Database | None = None,
    bot: Bot | None = None,
) -> str | None:
    """Create poll and return attachment id."""
    logging.info(
        "create_vk_poll start: group=%s question=%s", group_id, question
    )
    params = {
        "owner_id": f"-{group_id.lstrip('-')}",
        "question": question,
        "is_anonymous": 0,
        "add_answers": json.dumps(options, ensure_ascii=False),
    }
    data = await _vk_api("polls.create", params, db, bot)
    poll = data.get("response") or {}
    p_id = poll.get("id")
    owner = poll.get("owner_id", f"-{group_id.lstrip('-')}")
    if p_id is not None:
        attachment = f"poll{owner}_{p_id}"
        logging.info("create_vk_poll success: %s", attachment)
        return attachment
    logging.error("create_vk_poll failed for group %s", group_id)
    return None


async def post_vk_poll(
    group_id: str,
    question: str,
    options: list[str],
    db: Database | None = None,
    bot: Bot | None = None,
) -> str | None:
    """Create poll and post it to group wall."""
    logging.info("post_vk_poll start for group %s", group_id)
    attachment = await create_vk_poll(group_id, question, options, db, bot)
    if not attachment:
        logging.error("post_vk_poll: poll creation failed for group %s", group_id)
        return None
    return await post_to_vk(group_id, "", db, bot, [attachment])



def _vk_owner_and_post_id(url: str) -> tuple[str, str] | None:
    m = re.search(r"wall(-?\d+)_(\d+)", url)
    if not m:
        return None
    return m.group(1), m.group(2)


def _vk_expose_links(text: str) -> str:
    def repl_html(m: re.Match) -> str:
        href, label = m.group(1), m.group(2)
        return f"{label} ({href})"

    def repl_md(m: re.Match) -> str:
        label, href = m.group(1), m.group(2)
        return f"{label} ({href})"

    text = re.sub(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        repl_html,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", repl_md, text)
    return text


def build_vk_source_message(
    event: Event,
    text: str,
    festival: Festival | None = None,
    *,
    ics_url: str | None = None,
) -> str:
    """Build detailed VK post for an event including original source text."""

    text = _vk_expose_links(text)
    lines: list[str] = [event.title]

    if festival:
        link = festival.vk_url or festival.vk_post_url
        prefix = "✨ "
        if link:
            lines.append(f"{prefix}[{link}|{festival.name}]")
        else:
            lines.append(f"{prefix}{festival.name}")

    lines.append(VK_BLANK_LINE)

    if event.pushkin_card:
        lines.append("\u2705 Пушкинская карта")

    if event.is_free:
        lines.append("🟡 Бесплатно")
        if event.ticket_link:
            lines.append("по регистрации")
            lines.append(f"\U0001f39f {event.ticket_link}")
    elif event.ticket_link and (
        event.ticket_price_min is not None or event.ticket_price_max is not None
    ):
        if event.ticket_price_max is not None and event.ticket_price_max != event.ticket_price_min:
            price = f"от {event.ticket_price_min} до {event.ticket_price_max} руб."
        else:
            val = (
                event.ticket_price_min
                if event.ticket_price_min is not None
                else event.ticket_price_max
            )
            price = f"{val} руб." if val is not None else ""
        lines.append(f"Билеты в источнике {price}".strip())
        lines.append(f"\U0001f39f {event.ticket_link}")
    elif event.ticket_link:
        lines.append("по регистрации")
        lines.append(f"\U0001f39f {event.ticket_link}")
    else:
        price = ""
        if (
            event.ticket_price_min is not None
            and event.ticket_price_max is not None
            and event.ticket_price_min != event.ticket_price_max
        ):
            price = f"от {event.ticket_price_min} до {event.ticket_price_max} руб."
        elif event.ticket_price_min is not None:
            price = f"{event.ticket_price_min} руб."
        elif event.ticket_price_max is not None:
            price = f"{event.ticket_price_max} руб."
        if price:
            lines.append(f"Билеты {price}")

    date_part = event.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", event.date)
        day = event.date
    lines.append(f"{day} {event.time}")

    loc = event.location_name
    addr = event.location_address
    if addr and event.city:
        addr = strip_city_from_address(addr, event.city)
    if addr:
        loc += f", {addr}"
    if event.city:
        loc += f", #{event.city}"
    lines.append(loc)

    lines.append(VK_BLANK_LINE)
    lines.extend(text.strip().splitlines())
    lines.append(VK_BLANK_LINE)
    if ics_url:
        lines.append(f"Добавить в календарь {ics_url}")
    lines.append(VK_SOURCE_FOOTER)
    return "\n".join(lines)


async def sync_vk_source_post(
    event: Event,
    text: str,

    db: Database | None,
    bot: Bot | None,
    *,
    ics_url: str | None = None,
) -> str | None:
    """Create or update VK source post for an event."""
    if not VK_AFISHA_GROUP_ID:
        return None
    logging.info("sync_vk_source_post start for event %s", event.id)
    festival = None
    if event.festival and db:
        async with db.get_session() as session:
            res = await session.execute(
                select(Festival).where(Festival.name == event.festival)
            )
            festival = res.scalars().first()
    message = build_vk_source_message(event, text, festival=festival, ics_url=ics_url)
    if event.source_vk_post_url:
        existing = ""
        try:
            ids = _vk_owner_and_post_id(event.source_vk_post_url)
            if ids:
                data = await _vk_api(
                    "wall.getById",
                    {"posts": f"{ids[0]}_{ids[1]}"},
                    db,
                    bot,
                )
                items = data.get("response") or []
                if items:
                    existing = items[0].get("text", "")
        except Exception as e:
            logging.error("failed to fetch existing VK post: %s", e)
        base = existing.split(VK_SOURCE_FOOTER)[0].rstrip()
        new_message = f"{base}\n{CONTENT_SEPARATOR}\n{message}"
        await edit_vk_post(
            event.source_vk_post_url,
            new_message,
            db,
            bot,
        )
        url = event.source_vk_post_url
        logging.info("sync_vk_source_post updated %s", url)
    else:
        url = await post_to_vk(
            VK_AFISHA_GROUP_ID,
            message,
            db,
            bot,
        )
        if url:
            logging.info("sync_vk_source_post created %s", url)
    return url


async def edit_vk_post(
    post_url: str,
    message: str,
    db: Database | None = None,
    bot: Bot | None = None,
    attachments: list[str] | None = None,
    token: str | None = None,
) -> bool:
    """Edit an existing VK post.

    Returns ``True`` if the post was changed and ``False`` if the current
    content already matches ``message`` and ``attachments``.
    """
    logging.info("edit_vk_post start: %s", post_url)
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return
    owner_id, post_id = ids
    params = {
        "owner_id": owner_id,
        "post_id": post_id,
        "message": message,
        "from_group": 1,
    }
    current: list[str] = []
    post_text = ""
    old_attachments: list[str] = []
    try:
        data = await _vk_api(
            "wall.getById",
            {"posts": f"{owner_id}_{post_id}"},
            db,
            bot,
            token=token,
        )
        items = data.get("response") or []
        if items:
            post = items[0]
            post_text = post.get("text") or ""
            for att in post.get("attachments", []):
                if att.get("type") == "photo":
                    p = att.get("photo") or {}
                    o_id = p.get("owner_id")
                    p_id = p.get("id")
                    if o_id is not None and p_id is not None:
                        current.append(f"photo{o_id}_{p_id}")
            old_attachments = current.copy()
    except Exception as e:
        logging.error("failed to fetch VK post attachments: %s", e)
    if attachments:
        for a in attachments:
            if a not in current:
                current.append(a)
    if post_text == message and current == old_attachments:
        logging.info("edit_vk_post: no changes for %s", post_url)
        return False
    if current:
        params["attachments"] = ",".join(current)
    await _vk_api("wall.edit", params, db, bot, token=token)
    logging.info("edit_vk_post done: %s", post_url)
    return True


async def delete_vk_post(
    post_url: str,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
) -> None:
    """Delete a VK post given its URL."""
    logging.info("delete_vk_post start: %s", post_url)
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return
    owner_id, post_id = ids
    params = {"owner_id": owner_id, "post_id": post_id}
    try:
        await _vk_api("wall.delete", params, db, bot, token=token)
    except Exception as e:
        logging.error("failed to delete VK post %s: %s", post_url, e)
        return
    logging.info("delete_vk_post done: %s", post_url)


async def send_daily_announcement_vk(
    db: Database,
    group_id: str,
    tz: timezone,
    *,
    section: str,
    now: datetime | None = None,
    bot: Bot | None = None,
):
    section1, section2 = await build_daily_sections_vk(db, tz, now)
    if section == "today":
        await post_to_vk(group_id, section1, db, bot)
    elif section == "added":
        await post_to_vk(group_id, section2, db, bot)
    else:
        await post_to_vk(group_id, section1, db, bot)
        await post_to_vk(group_id, section2, db, bot)


async def send_daily_announcement(
    db: Database,
    bot: Bot,
    channel_id: int,
    tz: timezone,
    *,
    record: bool = True,
    now: datetime | None = None,
):
    posts = await build_daily_posts(db, tz, now)
    for text, markup in posts:
        try:
            await bot.send_message(
                channel_id,
                text,
                reply_markup=markup,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except Exception as e:
            logging.error("daily send failed for %s: %s", channel_id, e)
            if "message is too long" in str(e):
                continue
            raise
    if record and now is None:
        async with db.get_session() as session:
            ch = await session.get(Channel, channel_id)
            if ch:
                ch.last_daily = (now or datetime.now(tz)).date().isoformat()
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


async def vk_scheduler(db: Database, bot: Bot):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    while True:
        group_id = await get_vk_group_id(db)
        if not group_id:
            await asyncio.sleep(60)
            continue
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        now_time = now.time().replace(second=0, microsecond=0)
        today_time = datetime.strptime(await get_vk_time_today(db), "%H:%M").time()
        added_time = datetime.strptime(await get_vk_time_added(db), "%H:%M").time()

        last_today = await get_vk_last_today(db)
        if (last_today or "") != now.date().isoformat() and now_time >= today_time:
            try:
                await send_daily_announcement_vk(db, group_id, tz, section="today", bot=bot)
                await set_vk_last_today(db, now.date().isoformat())
            except Exception as e:
                logging.error("vk daily today failed: %s", e)

        last_added = await get_vk_last_added(db)
        if (last_added or "") != now.date().isoformat() and now_time >= added_time:
            try:
                await send_daily_announcement_vk(db, group_id, tz, section="added", bot=bot)
                await set_vk_last_added(db, now.date().isoformat())
            except Exception as e:
                logging.error("vk daily added failed: %s", e)

        await asyncio.sleep(60)


async def cleanup_old_events(db: Database, bot: Bot | None = None) -> int:
    """Remove events that finished over a week ago.

    Returns the number of deleted events."""
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)
    threshold = (datetime.now(tz) - timedelta(days=7)).date().isoformat()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(
                (
                    Event.end_date.is_not(None)
                    & (Event.end_date < threshold)
                )
                | (
                    Event.end_date.is_(None)
                    & (Event.date < threshold)
                )
            )
        )
        events = result.scalars().all()
        count = len(events)
        for event in events:
            await delete_ics(event)
            if bot:
                await delete_asset_post(event, db, bot)
                await remove_calendar_button(event, bot)
            await session.delete(event)
        if events:
            await session.commit()
    return count


async def cleanup_scheduler(db: Database, bot: Bot):
    last_run: date | None = None
    while True:
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        if now.time() >= time(3, 0) and now.date() != last_run:
            try:
                count = await cleanup_old_events(db, bot)
                await notify_superadmin(
                    db,
                    bot,
                    f"Cleanup completed: removed {count} events",
                )
            except Exception as e:
                logging.error("cleanup failed: %s", e)
                await notify_superadmin(db, bot, f"Cleanup failed: {e}")
            last_run = now.date()
        await asyncio.sleep(60)


async def page_update_scheduler(db: Database):
    """Refresh month and weekend Telegraph pages after midnight.

    To avoid unnecessary API calls after a manual restart during the day the
    first iteration skips syncing if the current time is well past the
    scheduled run window (01:00 local time).
    """
    last_run: date | None = None
    first = True
    while True:
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        if first:
            first = False
            # If the bot starts long after the scheduled time, assume today's
            # pages are already up to date.
            if now.time() >= time(2, 0):
                last_run = now.date()
        if now.time() >= time(1, 0) and now.date() != last_run:
            try:
                await sync_month_page(db, now.strftime("%Y-%m"))
                w_start = weekend_start_for_date(now.date())
                if w_start:
                    await sync_weekend_page(db, w_start.isoformat())
            except Exception as e:
                logging.error("page update failed: %s", e)
            last_run = now.date()
        await asyncio.sleep(60)


async def partner_notification_scheduler(db: Database, bot: Bot):
    """Remind partners who haven't added events for a week."""
    last_run: date | None = None
    while True:
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        if now.time() >= time(9, 0) and now.date() != last_run:
            try:
                notified = await notify_inactive_partners(db, bot, tz)
                if notified:
                    names = ", ".join(
                        f"@{u.username}" if u.username else str(u.user_id)
                        for u in notified
                    )
                    await notify_superadmin(
                        db, bot, f"Partner reminders sent to: {names}"
                    )
                else:
                    await notify_superadmin(db, bot, "Partner reminders: none")
            except Exception as e:
                logging.error("partner reminder failed: %s", e)
                await notify_superadmin(db, bot, f"Partner reminder failed: {e}")
            last_run = now.date()
        await asyncio.sleep(60)


async def vk_poll_scheduler(db: Database, bot: Bot):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    while True:
        group_id = await get_vk_group_id(db)
        if not group_id:
            await asyncio.sleep(60)
            continue
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        async with db.get_session() as session:
            res_f = await session.execute(select(Festival))
            festivals = res_f.scalars().all()
            res_e = await session.execute(select(Event))
            events = res_e.scalars().all()
        ev_map: dict[str, list[Event]] = {}
        for e in events:
            if e.festival:
                ev_map.setdefault(e.festival, []).append(e)
        for fest in festivals:
            if fest.vk_poll_url:
                continue
            evs = ev_map.get(fest.name, [])
            start, end = festival_dates(fest, evs)
            if not start:

                continue
            first_time: time | None = None
            for ev in evs:
                if ev.date != start.isoformat():
                    continue
                tr = parse_time_range(ev.time)
                if tr:
                    if first_time is None or tr[0] < first_time:
                        first_time = tr[0]
            if first_time is None:
                first_time = time(0, 0)
            if first_time >= time(17, 0):
                sched = datetime.combine(start, time(13, 0), tz)
            else:
                sched = datetime.combine(start - timedelta(days=1), time(21, 0), tz)
            if now >= sched and now.date() <= (end or start):

                try:
                    await send_festival_poll(db, fest, group_id, bot)
                except Exception as e:
                    logging.error("VK poll send failed for %s: %s", fest.name, e)
        await asyncio.sleep(60)


async def build_events_message(db: Database, target_date: date, tz: timezone, creator_id: int | None = None):
    async with db.get_session() as session:
        stmt = select(Event).where(
            (Event.date == target_date.isoformat())
            | (Event.end_date == target_date.isoformat())
        )
        if creator_id is not None:
            stmt = stmt.where(Event.creator_id == creator_id)
        result = await session.execute(stmt.order_by(Event.time))
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
                text=f"\u274c {e.id}", callback_data=f"del:{e.id}:{target_date.isoformat()}"
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
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
    cutoff = (today - timedelta(days=30)).isoformat()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.end_date.is_not(None),
                Event.end_date >= cutoff,
            )
            .order_by(Event.date)
        )
        events = result.scalars().all()

    lines = []
    for e in events:
        start = parse_iso_date(e.date)
        if not start:
            if ".." in e.date:
                start = parse_iso_date(e.date.split("..", 1)[0])
        if not start:
            logging.error("Bad start date %s for event %s", e.date, e.id)
            continue
        end = None
        if e.end_date:
            end = parse_iso_date(e.end_date)

        period = ""
        if end:
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
            types.InlineKeyboardButton(
                text=f"\u274c {e.id}", callback_data=f"del:{e.id}:exh"
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
            ),
        ]
        for e in events
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if events else None
    text = "Exhibitions\n" + "\n".join(lines)
    return text, markup


async def show_edit_menu(user_id: int, event: Event, bot: Bot):
    data: dict[str, Any]
    try:
        data = event.model_dump()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        data = event.dict()

    lines = []
    for key, value in data.items():
        if value is None:
            val = ""
        elif isinstance(value, str):
            val = value if len(value) <= 1000 else value[:1000] + "..."
        else:
            val = str(value)
        lines.append(f"{key}: {val}")

    fields = [k for k in data.keys() if k not in {"id", "added_at", "silent"}]
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
    if event.ics_url:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Delete ICS",
                    callback_data=f"delics:{event.id}",
                )
            ]
        )
    else:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="Create ICS",
                    callback_data=f"createics:{event.id}",
                )
            ]
        )
    keyboard.append(
        [types.InlineKeyboardButton(text="Done", callback_data=f"editdone:{event.id}")]
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    await bot.send_message(user_id, "\n".join(lines), reply_markup=markup)


async def show_festival_edit_menu(user_id: int, fest: Festival, bot: Bot):
    """Send festival fields with edit options."""
    lines = [
        f"name: {fest.name}",
        f"full: {fest.full_name or ''}",
        f"description: {fest.description or ''}",
        f"start: {fest.start_date or ''}",
        f"end: {fest.end_date or ''}",
        f"site: {fest.website_url or ''}",
        f"vk: {fest.vk_url or ''}",
        f"tg: {fest.tg_url or ''}",
    ]
    keyboard = [
        [
            types.InlineKeyboardButton(
                text="Edit short name",
                callback_data=f"festeditfield:{fest.id}:name",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Edit full name",
                callback_data=f"festeditfield:{fest.id}:full",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Edit description",
                callback_data=f"festeditfield:{fest.id}:description",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete start" if fest.start_date else "Add start"),
                callback_data=f"festeditfield:{fest.id}:start",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete end" if fest.end_date else "Add end"),
                callback_data=f"festeditfield:{fest.id}:end",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete site" if fest.website_url else "Add site"),
                callback_data=f"festeditfield:{fest.id}:site",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete VK" if fest.vk_url else "Add VK"),
                callback_data=f"festeditfield:{fest.id}:vk",
            )
        ],
        [
            types.InlineKeyboardButton(
                text=("Delete TG" if fest.tg_url else "Add TG"),
                callback_data=f"festeditfield:{fest.id}:tg",
            )
        ],
        [types.InlineKeyboardButton(text="Done", callback_data="festeditdone")],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    await bot.send_message(user_id, "\n".join(lines), reply_markup=markup)


async def handle_events(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split(maxsplit=1)
    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)

    if len(parts) == 2:
        day = parse_events_date(parts[1], tz)
        if not day:
            await bot.send_message(message.chat.id, "Usage: /events <date>")
            return
    else:
        day = datetime.now(tz).date()

    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
        creator_filter = user.user_id if user.is_partner else None

    text, markup = await build_events_message(db, day, tz, creator_filter)
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


async def handle_fest(message: types.Message, db: Database, bot: Bot):
    await send_festivals_list(message, db, bot, edit=False)





async def fetch_views(path: str, url: str | None = None) -> int | None:
    token = get_telegraph_token()
    if not token:
        return None
    domain = "telegra.ph"
    if url:
        try:
            domain = url.split("//", 1)[1].split("/", 1)[0]
        except Exception:
            pass
    tg = Telegraph(access_token=token, domain=domain)

    try:
        data = await telegraph_call(tg.get_views, path)
        return int(data.get("views", 0))
    except Exception as e:
        logging.error("Failed to fetch views for %s: %s", path, e)
        return None


async def collect_page_stats(db: Database) -> list[str]:
    today = datetime.now(LOCAL_TZ).date()
    prev_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    prev_month = prev_month_start.strftime("%Y-%m")

    prev_weekend = next_weekend_start(today - timedelta(days=7))
    cur_month = today.strftime("%Y-%m")
    cur_weekend = next_weekend_start(today)

    async with db.get_session() as session:
        mp_prev = await session.get(MonthPage, prev_month)
        wp_prev = await session.get(WeekendPage, prev_weekend.isoformat())

        res_months = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= cur_month)
            .order_by(MonthPage.month)
        )
        future_months = res_months.scalars().all()

        res_weekends = await session.execute(
            select(WeekendPage)
            .where(WeekendPage.start >= cur_weekend.isoformat())
            .order_by(WeekendPage.start)
        )
        future_weekends = res_weekends.scalars().all()

    lines: list[str] = []

    if mp_prev and mp_prev.path:

        views = await fetch_views(mp_prev.path, mp_prev.url)

        if views is not None:
            month_dt = date.fromisoformat(mp_prev.month + "-01")
            lines.append(f"{MONTHS_NOM[month_dt.month - 1]}: {views} просмотров")

    if wp_prev and wp_prev.path:

        views = await fetch_views(wp_prev.path, wp_prev.url)

        if views is not None:
            label = format_weekend_range(prev_weekend)
            lines.append(f"{label}: {views} просмотров")

    for wp in future_weekends:
        if not wp.path:
            continue

        views = await fetch_views(wp.path, wp.url)

        if views is not None:
            label = format_weekend_range(date.fromisoformat(wp.start))
            lines.append(f"{label}: {views} просмотров")

    for mp in future_months:
        if not mp.path:
            continue

        views = await fetch_views(mp.path, mp.url)

        if views is not None:
            month_dt = date.fromisoformat(mp.month + "-01")
            lines.append(f"{MONTHS_NOM[month_dt.month - 1]}: {views} просмотров")


    return lines


async def collect_event_stats(db: Database) -> list[str]:
    today = datetime.now(LOCAL_TZ).date()
    prev_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    async with db.get_session() as session:
        result = await session.execute(
            select(Event).where(
                Event.telegraph_path.is_not(None),
                Event.date >= prev_month_start.isoformat(),
            )
        )
        events = result.scalars().all()
    stats = []
    for e in events:
        if not e.telegraph_path:
            continue

        views = await fetch_views(e.telegraph_path, e.telegraph_url)

        if views is not None:
            stats.append((e.telegraph_url or e.telegraph_path, views))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{url}: {v}" for url, v in stats]


async def collect_festival_telegraph_stats(db: Database) -> list[str]:
    """Return Telegraph view counts for all festivals."""
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.telegraph_path.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int]] = []
    for f in fests:
        views = await fetch_views(f.telegraph_path, f.telegraph_url)
        if views is not None:
            stats.append((f.name, views))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}: {views}" for name, views in stats]


async def fetch_vk_post_stats(
    post_url: str, db: Database | None = None, bot: Bot | None = None
) -> tuple[int | None, int | None]:
    """Return view and reach counts for a VK post."""
    ids = _vk_owner_and_post_id(post_url)
    if not ids:
        logging.error("invalid VK post url %s", post_url)
        return None, None
    owner_id, post_id = ids
    views: int | None = None
    reach: int | None = None
    try:
        data = await _vk_api(
            "wall.getById", {"posts": f"{owner_id}_{post_id}"}, db, bot
        )
        items = data.get("response") or []
        if items:
            views = (items[0].get("views") or {}).get("count")
    except Exception as e:
        logging.error("VK views fetch error for %s: %s", post_url, e)
    try:
        data = await _vk_api(
            "stats.getPostReach",
            {"owner_id": owner_id, "post_id": post_id},
            db,
            bot,
        )
        items = data.get("response") or []
        if items:
            reach = items[0].get("reach_total")
    except Exception as e:
        logging.error("VK reach fetch error for %s: %s", post_url, e)
    return views, reach


async def collect_festival_vk_stats(db: Database) -> list[str]:
    """Return VK view and reach counts for all festivals."""
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.vk_post_url.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int, int]] = []
    for f in fests:
        views, reach = await fetch_vk_post_stats(f.vk_post_url, db)
        if views is not None or reach is not None:
            stats.append((f.name, views or 0, reach or 0))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}: {views}, {reach}" for name, views, reach in stats]


async def handle_stats(message: types.Message, db: Database, bot: Bot):
    parts = message.text.split()
    mode = parts[1] if len(parts) > 1 else ""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    if mode == "events":
        lines = await collect_event_stats(db)
    else:
        lines = await collect_page_stats(db)
        fest_tg = await collect_festival_telegraph_stats(db)
        fest_vk = await collect_festival_vk_stats(db)
        if fest_tg:
            lines.append("")
            lines.append("Фестивали (телеграм)")
            lines.extend(fest_tg)
        if fest_vk:
            lines.append("")
            lines.append("Фестивали (Вк) (просмотров, пользователи)")
            lines.extend(fest_vk)
    await bot.send_message(message.chat.id, "\n".join(lines) if lines else "No data")


async def handle_dumpdb(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
        result = await session.execute(select(Channel))
        channels = result.scalars().all()
        tz_setting = await session.get(Setting, "tz_offset")
        catbox_setting = await session.get(Setting, "catbox_enabled")

    data = await dump_database(db.engine.url.database)
    file = types.BufferedInputFile(data, filename="dump.sql")
    await bot.send_document(message.chat.id, file)
    token_exists = os.path.exists(TELEGRAPH_TOKEN_FILE)
    if token_exists:
        with open(TELEGRAPH_TOKEN_FILE, "rb") as f:
            token_file = types.BufferedInputFile(
                f.read(), filename="telegraph_token.txt"
            )
        await bot.send_document(message.chat.id, token_file)

    lines = ["Channels:"]
    for ch in channels:
        roles: list[str] = []
        if ch.is_registered:
            roles.append("announcement")
        if ch.is_asset:
            roles.append("asset")
        if ch.daily_time:
            roles.append(f"daily {ch.daily_time}")
        title = ch.title or ch.username or str(ch.channel_id)
        lines.append(f"- {title}: {', '.join(roles) if roles else 'admin'}")

    lines.append("")
    lines.append("To restore on another server:")
    step = 1
    lines.append(f"{step}. Start the bot and send /restore with the dump file.")
    step += 1
    if tz_setting:
        lines.append(f"{step}. Current timezone: {tz_setting.value}")
        step += 1
    lines.append(f"{step}. Add the bot as admin to the channels listed above.")
    step += 1
    if token_exists:
        lines.append(
            f"{step}. Copy telegraph_token.txt to {TELEGRAPH_TOKEN_FILE} before first run."
        )
        step += 1
    if catbox_setting and catbox_setting.value == "1":
        lines.append(f"{step}. Run /images to enable photo uploads.")

    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_restore(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    document = message.document
    if not document and message.reply_to_message:
        document = message.reply_to_message.document
    if not document:
        await bot.send_message(message.chat.id, "Attach dump file")
        return
    bio = BytesIO()
    await bot.download(document.file_id, destination=bio)
    await restore_database(bio.getvalue(), db, db.engine.url.database)
    await bot.send_message(message.chat.id, "Database restored")
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
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, eid)
        if not event or (user and user.blocked) or (
            user and user.is_partner and event.creator_id != user.user_id
        ):
            await bot.send_message(message.chat.id, "Event not found" if not event else "Not authorized")
            del editing_sessions[message.from_user.id]
            return
        old_date = event.date.split("..", 1)[0]
        old_month = old_date[:7]
        old_fest = event.festival
        if field in {"ticket_price_min", "ticket_price_max"}:
            try:
                setattr(event, field, int(value))
            except ValueError:
                await bot.send_message(message.chat.id, "Invalid number")
                return
        else:
            if field in {"is_free", "pushkin_card", "silent"}:
                bool_val = parse_bool_text(value)
                if bool_val is None:
                    await bot.send_message(message.chat.id, "Invalid boolean")
                    return
                setattr(event, field, bool_val)
            elif field == "ticket_link" and value == "":
                setattr(event, field, None)
            else:
                setattr(event, field, value)
        await session.commit()
        new_date = event.date.split("..", 1)[0]
        new_month = new_date[:7]
        new_fest = event.festival
    await sync_month_page(db, old_month)
    old_dt = parse_iso_date(old_date)
    old_w = weekend_start_for_date(old_dt) if old_dt else None
    if old_w:
        await sync_weekend_page(db, old_w.isoformat())
    if new_month != old_month:
        await sync_month_page(db, new_month)
    new_dt = parse_iso_date(new_date)
    new_w = weekend_start_for_date(new_dt) if new_dt else None
    if new_w and new_w != old_w:
        await sync_weekend_page(db, new_w.isoformat())
    if old_fest:
        await sync_festival_page(db, old_fest)

        await sync_festival_vk_post(db, old_fest, bot)
    if new_fest and new_fest != old_fest:
        await sync_festival_page(db, new_fest)
        await sync_festival_vk_post(db, new_fest, bot)

    editing_sessions[message.from_user.id] = (eid, None)
    await show_edit_menu(message.from_user.id, event, bot)


async def handle_add_event_start(message: types.Message, db: Database, bot: Bot):
    """Initiate event creation via the menu."""
    logging.info("handle_add_event_start from user %s", message.from_user.id)
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    add_event_sessions.add(message.from_user.id)
    logging.info(
        "handle_add_event_start session opened for user %s", message.from_user.id
    )
    await bot.send_message(message.chat.id, "Send event text and optional photo")


async def handle_vk_link_message(message: types.Message, db: Database, bot: Bot):
    eid = vk_link_sessions.get(message.from_user.id)
    if not eid:
        return
    link = (message.text or "").strip()
    if not is_vk_wall_url(link):
        await bot.send_message(message.chat.id, "Invalid link")
        return
    async with db.get_session() as session:
        event = await session.get(Event, eid)
        if event:
            event.source_post_url = link
            await session.commit()
    vk_link_sessions.pop(message.from_user.id, None)
    await bot.send_message(message.chat.id, "Link saved")


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


async def handle_vk_group_message(message: types.Message, db: Database, bot: Bot):
    if message.from_user.id not in vk_group_sessions:
        return
    value = (message.text or "").strip()
    if value.lower() == "off":
        await set_vk_group_id(db, None)
        await bot.send_message(message.chat.id, "VK posting disabled")
    else:
        await set_vk_group_id(db, value)
        await bot.send_message(message.chat.id, f"VK group set to {value}")
    vk_group_sessions.discard(message.from_user.id)


async def handle_vk_time_message(message: types.Message, db: Database, bot: Bot):
    typ = vk_time_sessions.get(message.from_user.id)
    if not typ:
        return
    value = (message.text or "").strip()
    if not re.match(r"^\d{1,2}:\d{2}$", value):
        await bot.send_message(message.chat.id, "Invalid time")
        return
    if len(value.split(":")[0]) == 1:
        value = f"0{value}"
    if typ == "today":
        await set_vk_time_today(db, value)
    else:
        await set_vk_time_added(db, value)
    vk_time_sessions.pop(message.from_user.id, None)
    await bot.send_message(message.chat.id, f"Time set to {value}")


async def handle_partner_info_message(message: types.Message, db: Database, bot: Bot):
    uid = partner_info_sessions.get(message.from_user.id)
    if not uid:
        return
    text = (message.text or "").strip()
    if "," not in text:
        await bot.send_message(message.chat.id, "Please send 'Organization, location'")
        return
    org, loc = [p.strip() for p in text.split(",", 1)]
    async with db.get_session() as session:
        pending = await session.get(PendingUser, uid)
        if not pending:
            await bot.send_message(message.chat.id, "Pending user not found")
            partner_info_sessions.pop(message.from_user.id, None)
            return
        session.add(
            User(
                user_id=uid,
                username=pending.username,
                is_partner=True,
                organization=org,
                location=loc,
            )
        )
        await session.delete(pending)
        await session.commit()
    partner_info_sessions.pop(message.from_user.id, None)
    await bot.send_message(uid, "You are approved as partner")
    await bot.send_message(
        message.chat.id,
        f"User {uid} approved as partner at {org}, {loc}",
    )
    logging.info("approved user %s as partner %s, %s", uid, org, loc)


async def handle_festival_edit_message(message: types.Message, db: Database, bot: Bot):
    state = festival_edit_sessions.get(message.from_user.id)
    if not state:
        return
    fid, field = state
    if field is None:
        return
    text = (message.text or "").strip()
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        if not fest:
            await bot.send_message(message.chat.id, "Festival not found")
            festival_edit_sessions.pop(message.from_user.id, None)
            return
        if field == "description":
            fest.description = None if text in {"", "-"} else text
        elif field == "name":
            if text:
                fest.name = text
        elif field == "full":
            fest.full_name = None if text in {"", "-"} else text
        elif field == "start":
            if text in {"", "-"}:
                fest.start_date = None
            else:
                d = parse_events_date(text, timezone.utc)
                if not d:
                    await bot.send_message(message.chat.id, "Invalid date")
                    return
                fest.start_date = d.isoformat()
        elif field == "end":
            if text in {"", "-"}:
                fest.end_date = None
            else:
                d = parse_events_date(text, timezone.utc)
                if not d:
                    await bot.send_message(message.chat.id, "Invalid date")
                    return
                fest.end_date = d.isoformat()
        elif field == "site":
            fest.website_url = None if text in {"", "-"} else text
        elif field == "vk":
            fest.vk_url = None if text in {"", "-"} else text
        elif field == "tg":
            fest.tg_url = None if text in {"", "-"} else text
        await session.commit()
        logging.info("festival %s updated", fest.name)
    festival_edit_sessions[message.from_user.id] = (fid, None)
    await show_festival_edit_menu(message.from_user.id, fest, bot)

    await sync_festival_page(db, fest.name)
    await sync_festival_vk_post(db, fest.name, bot)



processed_media_groups: set[str] = set()

# store up to three images for albums until the caption arrives

pending_media_groups: dict[str, list[tuple[bytes, str]]] = {}


async def handle_forwarded(message: types.Message, db: Database, bot: Bot):
    logging.info(
        "received forwarded message %s from %s",
        message.message_id,
        message.from_user.id,
    )
    text = message.text or message.caption
    images = await extract_images(message, bot)
    logging.info(
        "forward text len=%d photos=%d",
        len(text or ""),
        len(images or []),
    )
    media: list[tuple[bytes, str]] | None = None
    if message.media_group_id:
        gid = message.media_group_id
        if gid in processed_media_groups:
            logging.info("skip already processed album %s", gid)
            return
        if not text:
            if images:

                buf = pending_media_groups.setdefault(gid, [])
                if len(buf) < 3:
                    buf.extend(images[: 3 - len(buf)])
            logging.info("waiting for caption in album %s", gid)
            return
        stored = pending_media_groups.pop(gid, [])
        if len(stored) < 3 and images:
            stored.extend(images[: 3 - len(stored)])
        media = stored

        processed_media_groups.add(gid)
    else:
        if not text:
            logging.info("forwarded message has no text")
            return
        media = images[:3] if images else None
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            logging.info("user %s not registered or blocked", message.from_user.id)
            return
    link = None
    msg_id = None
    chat_id: int | None = None
    channel_name: str | None = None

    if message.forward_from_chat and message.forward_from_message_id:
        chat = message.forward_from_chat
        msg_id = message.forward_from_message_id
        chat_id = chat.id
        channel_name = chat.title or getattr(chat, "username", None)

        async with db.get_session() as session:
            ch = await session.get(Channel, chat_id)
            allowed = ch.is_registered if ch else False
        logging.info("forward from chat %s allowed=%s", chat_id, allowed)
        if allowed:
            if chat.username:
                link = f"https://t.me/{chat.username}/{msg_id}"
            else:
                cid = str(chat_id)
                if cid.startswith("-100"):
                    cid = cid[4:]
                else:
                    cid = cid.lstrip("-")
                link = f"https://t.me/c/{cid}/{msg_id}"
    else:
        fo = getattr(message, "forward_origin", None)
        if isinstance(fo, dict):
            fo_type = fo.get("type")
        else:
            fo_type = getattr(fo, "type", None)
        if fo_type in {"messageOriginChannel", "channel"}:
            chat_data = fo.get("chat") if isinstance(fo, dict) else getattr(fo, "chat", {})
            chat_id = chat_data.get("id") if isinstance(chat_data, dict) else getattr(chat_data, "id", None)
            msg_id = fo.get("message_id") if isinstance(fo, dict) else getattr(fo, "message_id", None)
            channel_name = (
                chat_data.get("title") if isinstance(chat_data, dict) else getattr(chat_data, "title", None)
            )
            if not channel_name:
                channel_name = (
                    chat_data.get("username") if isinstance(chat_data, dict) else getattr(chat_data, "username", None)
                )

            async with db.get_session() as session:
                ch = await session.get(Channel, chat_id)
                allowed = ch.is_registered if ch else False
            logging.info("forward from origin chat %s allowed=%s", chat_id, allowed)
            if allowed:
                username = chat_data.get("username") if isinstance(chat_data, dict) else getattr(chat_data, "username", None)
                if username:
                    link = f"https://t.me/{username}/{msg_id}"
                else:
                    cid = str(chat_id)
                    if cid.startswith("-100"):
                        cid = cid[4:]
                    else:
                        cid = cid.lstrip("-")
                    link = f"https://t.me/c/{cid}/{msg_id}"
    if link:
        logging.info("source post url %s", link)
    logging.info("parsing forwarded text via LLM")
    results = await add_events_from_text(
        db,
        text,
        link,
        message.html_text or message.caption_html,
        media,
        source_chat_id=chat_id if link else None,
        source_message_id=msg_id if link else None,
        creator_id=user.user_id,
        source_channel=channel_name,

        bot=bot,

    )
    logging.info("forward parsed %d events", len(results))
    if not results:
        logging.info("no events parsed from forwarded text")
        return
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
        text_out = f"Event {status}\n" + "\n".join(lines)
        logging.info("sending response for event %s", saved.id)
        try:
            await bot.send_message(
                message.chat.id,
                text_out,
                reply_markup=markup,
            )
            await notify_event_added(db, bot, user, saved, added)
        except Exception as e:
            logging.error("failed to send event response: %s", e)


async def telegraph_test():
    token = get_telegraph_token()
    if not token:
        print("Unable to obtain Telegraph token")
        return
    tg = Telegraph(access_token=token)
    page = await telegraph_call(
        tg.create_page, "Test Page", html_content="<p>test</p>"
    )
    logging.info("Created %s", page["url"])
    print("Created", page["url"])
    await telegraph_call(
        tg.edit_page, page["path"], title="Test Page", html_content="<p>updated</p>"
    )
    logging.info("Edited %s", page["url"])
    print("Edited", page["url"])


async def update_source_page(
    path: str,
    title: str,
    new_html: str,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    db: Database | None = None,
    *,
    catbox_urls: list[str] | None = None,
) -> tuple[str, int]:
    """Append text to an existing Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return "token missing"
    tg = Telegraph(access_token=token)
    try:
        logging.info("Fetching telegraph page %s", path)
        page = await telegraph_call(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        catbox_msg = ""
        images: list[tuple[bytes, str]] = []
        if media:
            images = [media] if isinstance(media, tuple) else list(media)
        if catbox_urls is not None:
            urls = list(catbox_urls)
        else:
            urls, catbox_msg = await upload_to_catbox(images)
        for url in urls:
            html_content += f'<img src="{html.escape(url)}"/><p></p>'
        new_html = normalize_hashtag_dates(new_html)
        cleaned = re.sub(r"</?tg-emoji[^>]*>", "", new_html)
        cleaned = cleaned.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        html_content += (
            f"<p>{CONTENT_SEPARATOR}</p><p>" + cleaned.replace("\n", "<br/>") + "</p>"
        )
        if db:
            nav_html = await build_month_nav_html(db)
            html_content = apply_month_nav(html_content, nav_html)
        html_content = apply_footer_link(html_content)
        logging.info("Editing telegraph page %s", path)
        await telegraph_call(
            tg.edit_page, path, title=title, html_content=html_content
        )
        logging.info("Updated telegraph page %s", path)
        return catbox_msg, len(urls)
    except Exception as e:
        logging.error("Failed to update telegraph page: %s", e)
        return f"error: {e}", 0


async def update_source_page_ics(path: str, title: str, url: str | None):
    """Insert or remove the ICS link in a Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    try:
        logging.info("Editing telegraph ICS for %s", path)
        page = await telegraph_call(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        html_content = apply_ics_link(html_content, url)
        html_content = apply_footer_link(html_content)
        await telegraph_call(
            tg.edit_page, path, title=title, html_content=html_content
        )
    except Exception as e:
        logging.error("Failed to update ICS link: %s", e)


async def get_source_page_text(path: str) -> str:
    """Return plain text from a Telegraph page."""
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return ""
    tg = Telegraph(access_token=token)
    try:
        page = await telegraph_call(tg.get_page, path, return_html=True)
    except Exception as e:
        logging.error("Failed to fetch telegraph page: %s", e)
        return ""
    html_content = page.get("content") or page.get("content_html") or ""
    html_content = apply_ics_link(html_content, None)
    html_content = apply_month_nav(html_content, None)
    html_content = html_content.replace(FOOTER_LINK_HTML, "")
    html_content = html_content.replace(f"<p>{CONTENT_SEPARATOR}</p>", f"\n{CONTENT_SEPARATOR}\n")
    html_content = html_content.replace("<br/>", "\n").replace("<br>", "\n")
    html_content = re.sub(r"</p>\s*<p>", "\n", html_content)
    html_content = re.sub(r"<[^>]+>", "", html_content)
    text = html.unescape(html_content)
    text = text.replace(CONTENT_SEPARATOR, "").replace("\xa0", " ")
    return text.strip()


async def update_event_description(event: Event, db: Database) -> None:
    """Rebuild event.description from the Telegraph source page."""
    if not event.telegraph_path:
        return
    text = await get_source_page_text(event.telegraph_path)
    if not text:
        return
    try:
        parsed = await parse_event_via_4o(text)
    except Exception as e:
        logging.error("Failed to parse source text for description: %s", e)
        return
    if not parsed:
        return
    desc = parsed[0].get("short_description", "").strip()
    if not desc:
        return
    async with db.get_session() as session:
        obj = await session.get(Event, event.id)
        if obj:
            obj.description = desc
            await session.commit()
            event.description = desc


async def create_source_page(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    ics_url: str | None = None,
    db: Database | None = None,
    *,
    display_link: bool = True,
    catbox_urls: list[str] | None = None,
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
    if catbox_urls is not None:
        urls = list(catbox_urls)
        catbox_msg = ""
    else:
        urls, catbox_msg = await upload_to_catbox(images)

    if source_url and display_link:
        html_content += (
            f'<p><a href="{html.escape(source_url)}"><strong>'
            f"{html.escape(title)}</strong></a></p>"
        )
    else:
        html_content += f"<p><strong>{html.escape(title)}</strong></p>"

    for url in urls:
        html_content += f'<img src="{html.escape(url)}"/><p></p>'

    html_content = apply_ics_link(html_content, ics_url)

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

    if db:
        nav_html = await build_month_nav_html(db)
        html_content = apply_month_nav(html_content, nav_html)
    html_content = apply_footer_link(html_content)
    try:
        page = await telegraph_call(tg.create_page, title, html_content=html_content)
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    logging.info("Created telegraph page %s", page.get("url"))
    return page.get("url"), page.get("path"), catbox_msg, len(urls)


def create_app() -> web.Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    webhook = os.getenv("WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("WEBHOOK_URL is missing")

    session = IPv4AiohttpSession()
    bot = Bot(token, session=session)
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

    async def stats_wrapper(message: types.Message):
        await handle_stats(message, db, bot)

    async def fest_wrapper(message: types.Message):
        await handle_fest(message, db, bot)

    async def festival_edit_wrapper(message: types.Message):
        await handle_festival_edit_message(message, db, bot)

    async def users_wrapper(message: types.Message):
        await handle_users(message, db, bot)

    async def dumpdb_wrapper(message: types.Message):
        await handle_dumpdb(message, db, bot)

    async def restore_wrapper(message: types.Message):
        await handle_restore(message, db, bot)

    async def edit_message_wrapper(message: types.Message):
        await handle_edit_message(message, db, bot)

    async def daily_time_wrapper(message: types.Message):
        await handle_daily_time_message(message, db, bot)

    async def vk_group_msg_wrapper(message: types.Message):
        await handle_vk_group_message(message, db, bot)

    async def vk_time_msg_wrapper(message: types.Message):
        await handle_vk_time_message(message, db, bot)

    async def partner_info_wrapper(message: types.Message):
        await handle_partner_info_message(message, db, bot)

    async def forward_wrapper(message: types.Message):
        await handle_forwarded(message, db, bot)

    async def reg_daily_wrapper(message: types.Message):
        await handle_regdailychannels(message, db, bot)

    async def daily_wrapper(message: types.Message):
        await handle_daily(message, db, bot)

    async def images_wrapper(message: types.Message):
        await handle_images(message, db, bot)

    async def vkgroup_wrapper(message: types.Message):
        await handle_vkgroup(message, db, bot)

    async def vktime_wrapper(message: types.Message):
        await handle_vktime(message, db, bot)

    async def vkphotos_wrapper(message: types.Message):
        await handle_vkphotos(message, db, bot)

    async def menu_wrapper(message: types.Message):
        await handle_menu(message, db, bot)

    async def events_menu_wrapper(message: types.Message):
        await handle_events_menu(message, db, bot)

    async def events_date_wrapper(message: types.Message):
        await handle_events_date_message(message, db, bot)

    async def add_event_start_wrapper(message: types.Message):
        await handle_add_event_start(message, db, bot)

    async def add_event_session_wrapper(message: types.Message):
        await handle_add_event(message, db, bot)

    async def vk_link_msg_wrapper(message: types.Message):
        await handle_vk_link_message(message, db, bot)

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
        or c.data.startswith("assetunset:")
        or c.data.startswith("set:")
        or c.data.startswith("assetset:")
        or c.data.startswith("dailyset:")
        or c.data.startswith("dailyunset:")
        or c.data.startswith("dailytime:")
        or c.data.startswith("dailysend:")
        or c.data.startswith("dailysendtom:")
        or c.data == "vkset"
        or c.data == "vkunset"
        or c.data.startswith("vktime:")
        or c.data.startswith("vkdailysend:")
        or c.data.startswith("vklink:")
        or c.data == "vklinkskip"
        or c.data.startswith("menuevt:")
        or c.data.startswith("togglefree:")
        or c.data.startswith("markfree:")
        or c.data.startswith("togglesilent:")
        or c.data.startswith("createics:")
        or c.data.startswith("delics:")
        or c.data.startswith("partner:")
        or c.data.startswith("block:")
        or c.data.startswith("unblock:")
        or c.data.startswith("festedit:")
        or c.data.startswith("festeditfield:")
        or c.data == "festeditdone"
        or c.data.startswith("festdel:")
        or c.data.startswith("setfest:")
    ,
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(add_event_wrapper, Command("addevent"))
    dp.message.register(add_event_raw_wrapper, Command("addevent_raw"))
    dp.message.register(ask_4o_wrapper, Command("ask4o"))
    dp.message.register(list_events_wrapper, Command("events"))
    dp.message.register(set_channel_wrapper, Command("setchannel"))
    dp.message.register(images_wrapper, Command("images"))
    dp.message.register(vkgroup_wrapper, Command("vkgroup"))
    dp.message.register(vktime_wrapper, Command("vktime"))
    dp.message.register(vkphotos_wrapper, Command("vkphotos"))
    dp.message.register(menu_wrapper, Command("menu"))
    dp.message.register(events_menu_wrapper, lambda m: m.text == MENU_EVENTS)
    dp.message.register(events_date_wrapper, lambda m: m.from_user.id in events_date_sessions)
    dp.message.register(add_event_start_wrapper, lambda m: m.text == MENU_ADD_EVENT)
    dp.message.register(add_event_session_wrapper, lambda m: m.from_user.id in add_event_sessions)
    dp.message.register(vk_link_msg_wrapper, lambda m: m.from_user.id in vk_link_sessions)
    dp.message.register(partner_info_wrapper, lambda m: m.from_user.id in partner_info_sessions)
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(reg_daily_wrapper, Command("regdailychannels"))
    dp.message.register(daily_wrapper, Command("daily"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(fest_wrapper, Command("fest"))

    dp.message.register(pages_wrapper, Command("pages"))
    dp.message.register(stats_wrapper, Command("stats"))
    dp.message.register(users_wrapper, Command("users"))
    dp.message.register(dumpdb_wrapper, Command("dumpdb"))
    dp.message.register(restore_wrapper, Command("restore"))
    dp.message.register(
        edit_message_wrapper, lambda m: m.from_user.id in editing_sessions
    )
    dp.message.register(
        daily_time_wrapper, lambda m: m.from_user.id in daily_time_sessions
    )
    dp.message.register(
        vk_group_msg_wrapper, lambda m: m.from_user.id in vk_group_sessions
    )
    dp.message.register(
        vk_time_msg_wrapper, lambda m: m.from_user.id in vk_time_sessions
    )
    dp.message.register(

        festival_edit_wrapper, lambda m: m.from_user.id in festival_edit_sessions

    )
    dp.message.register(
        forward_wrapper,
        lambda m: bool(m.forward_date)
        or "forward_origin" in getattr(m, "model_extra", {}),
    )
    dp.my_chat_member.register(partial(handle_my_chat_member, db=db))

    app = web.Application()
    SimpleRequestHandler(dp, bot).register(app, path="/webhook")
    setup_application(app, dp, bot=bot)

    async def on_startup(app: web.Application):
        logging.info("Initializing database")
        await db.init()
        await get_tz_offset(db)
        global CATBOX_ENABLED
        CATBOX_ENABLED = await get_catbox_enabled(db)
        global VK_PHOTOS_ENABLED
        VK_PHOTOS_ENABLED = await get_vk_photos_enabled(db)
        hook = webhook.rstrip("/") + "/webhook"
        logging.info("Setting webhook to %s", hook)
        try:
            await bot.set_webhook(
                hook,
                allowed_updates=["message", "callback_query", "my_chat_member"],
            )
        except Exception as e:
            logging.error("Failed to set webhook: %s", e)
        app["daily_task"] = asyncio.create_task(daily_scheduler(db, bot))
        app["vk_task"] = asyncio.create_task(vk_scheduler(db, bot))
        app["vk_poll_task"] = asyncio.create_task(vk_poll_scheduler(db, bot))
        app["cleanup_task"] = asyncio.create_task(cleanup_scheduler(db, bot))
        app["page_task"] = asyncio.create_task(page_update_scheduler(db))
        app["partner_task"] = asyncio.create_task(
            partner_notification_scheduler(db, bot)
        )

    async def on_shutdown(app: web.Application):
        await bot.session.close()
        if "daily_task" in app:
            app["daily_task"].cancel()
            with contextlib.suppress(Exception):
                await app["daily_task"]
        if "vk_task" in app:
            app["vk_task"].cancel()
            with contextlib.suppress(Exception):
                await app["vk_task"]
        if "vk_poll_task" in app:
            app["vk_poll_task"].cancel()
            with contextlib.suppress(Exception):
                await app["vk_poll_task"]
        if "cleanup_task" in app:
            app["cleanup_task"].cancel()
            with contextlib.suppress(Exception):
                await app["cleanup_task"]
        if "page_task" in app:
            app["page_task"].cancel()
            with contextlib.suppress(Exception):
                await app["page_task"]
        if "partner_task" in app:
            app["partner_task"].cancel()
            with contextlib.suppress(Exception):
                await app["partner_task"]

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
