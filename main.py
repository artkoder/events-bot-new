"""
Debugging:
    EVBOT_DEBUG=1 fly deploy ...
    Logs will include ▶/■ markers with RSS & duration.
"""

from __future__ import annotations

import logging
import os
import time
import time as _time


class DeduplicateFilter(logging.Filter):
    """Limit repeating DEBUG messages to avoid log spam."""

    def __init__(self, interval: float = 60.0) -> None:
        super().__init__()
        self.interval = interval
        self.last_seen: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
        if record.levelno > logging.DEBUG:
            return True
        msg = record.getMessage()
        now = _time.monotonic()
        last = self.last_seen.get(msg, 0.0)
        if now - last >= self.interval:
            self.last_seen[msg] = now
            return True
        return False


def configure_logging() -> None:
    debug = os.getenv("EVBOT_DEBUG") == "1"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    if os.getenv("LOG_SQL", "0") == "0":
        for noisy in ("aiosqlite", "sqlalchemy.engine"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    for noisy in ("aiogram", "httpx"):
        logging.getLogger(noisy).setLevel(logging.INFO if debug else logging.WARNING)
    logging.getLogger().addFilter(DeduplicateFilter())


configure_logging()

from datetime import date, datetime, timedelta, timezone, time
from typing import Optional, Tuple, Iterable, Any, Callable, Awaitable
from urllib.parse import urlparse, parse_qs
import uuid
import textwrap
# тяжёлый стек подтягиваем только если понадобится
Calendar = None
IcsEvent = None


def _load_icalendar() -> None:
    global Calendar, IcsEvent
    if Calendar is None or IcsEvent is None:  # pragma: no cover - simple
        from icalendar import Calendar as _Calendar, Event as _IcsEvent

        Calendar = _Calendar
        IcsEvent = _IcsEvent

from aiogram import Bot, Dispatcher, types
from safe_bot import SafeBot, BACKOFF_DELAYS
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import (
    web,
    FormData,
    ClientSession,
    TCPConnector,
    ClientTimeout,
    ClientOSError,
    ServerDisconnectedError,
)
from aiogram.client.session.aiohttp import AiohttpSession
import socket
from difflib import SequenceMatcher
import json
import re
import httpx
import hashlib
import unicodedata

from telegraph import Telegraph, TelegraphException
from net import http_call, telegraph_upload

from functools import partial, lru_cache
from collections import defaultdict, deque
import asyncio
import contextlib
import random
import html
import sqlite3
from io import BytesIO
import aiosqlite
import gc
import atexit
from cachetools import TTLCache
from markup import simple_md_to_html, telegraph_br, DAY_START, DAY_END, PERM_START, PERM_END
from sections import replace_between_markers, content_hash
from db import Database
from scheduler import startup as scheduler_startup, cleanup as scheduler_cleanup
from sqlalchemy import select, update, delete, text, func

from models import (
    User,
    PendingUser,
    RejectedUser,
    Channel,
    Setting,
    Event,
    MonthPage,
    WeekendPage,
    WeekPage,
    Festival,
    JobOutbox,
    JobTask,
    JobStatus,
)



from span import span

span.configure(
    {
        "db-query": 50,
        "vk-call": 1000,
        "telegraph-call": 1000,
        "event_update_job": 5000,
    }
)

DEBUG = os.getenv("EVBOT_DEBUG") == "1"


_page_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_month_next_run: dict[str, float] = defaultdict(float)
_partner_last_run: date | None = None

_startup_handler_registered = False

# in-memory diagnostic buffers
START_TIME = _time.time()
LOG_BUFFER: deque[tuple[datetime, str, str]] = deque(maxlen=200)
ERROR_BUFFER: deque[dict[str, Any]] = deque(maxlen=50)
JOB_HISTORY: deque[dict[str, Any]] = deque(maxlen=20)
LAST_RUN_ID: str | None = None


class MemoryLogHandler(logging.Handler):
    """Store recent log records in memory for diagnostics."""

    _job_id_re = re.compile(r"job_id=(\S+)")
    _run_id_re = re.compile(r"run_id=(\S+)")
    _took_re = re.compile(r"took_ms=(\d+)")

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        msg = record.getMessage()
        ts = datetime.utcnow()
        LOG_BUFFER.append((ts, record.levelname, msg))
        if record.levelno >= logging.ERROR:
            err_type = record.exc_info[0].__name__ if record.exc_info else record.levelname
            ERROR_BUFFER.append(
                {
                    "time": ts,
                    "type": err_type,
                    "where": f"{record.pathname}:{record.lineno}",
                    "message": msg,
                }
            )
        if msg.startswith("JOB_EXECUTED") or msg.startswith("JOB_ERROR"):
            job_id = self._job_id_re.search(msg)
            run_id = self._run_id_re.search(msg)
            took = self._took_re.search(msg)
            status = "ok" if msg.startswith("JOB_EXECUTED") else "err"
            JOB_HISTORY.append(
                {
                    "id": job_id.group(1) if job_id else "?",
                    "when": ts,
                    "status": status,
                    "took_ms": int(took.group(1)) if took else 0,
                }
            )
            if run_id:
                global LAST_RUN_ID
                LAST_RUN_ID = run_id.group(1)


logging.getLogger().addHandler(MemoryLogHandler())


_last_rss: int | None = None


def mem_info(label: str = "", update: bool = True) -> tuple[int, int]:
    try:
        import psutil  # type: ignore

        rss = psutil.Process().memory_info().rss
    except Exception:
        rss = 0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
        except FileNotFoundError:
            pass
    global _last_rss
    prev = _last_rss or rss
    delta = rss - prev
    if update:
        _last_rss = rss
    if DEBUG:
        logging.info(
            "MEM rss=%.1f MB (Δ%.1f MB)%s",
            rss / (1024**2),
            delta / (1024**2),
            f" {label}" if label else "",
        )
    return rss, delta


def normalize_telegraph_url(url: str | None) -> str | None:
    if url and url.startswith("https://t.me/"):
        return url.replace("https://t.me/", "https://telegra.ph/")
    return url



@lru_cache(maxsize=20)
def _weekend_vk_lock(start: str) -> asyncio.Lock:
    return asyncio.Lock()


@lru_cache(maxsize=20)
def _week_vk_lock(start: str) -> asyncio.Lock:
    return asyncio.Lock()

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
editing_sessions: TTLCache[int, tuple[int, str | None]] = TTLCache(maxsize=64, ttl=3600)
# user_id -> channel_id for daily time editing
daily_time_sessions: TTLCache[int, int] = TTLCache(maxsize=64, ttl=3600)
# waiting for VK group ID input
vk_group_sessions: set[int] = set()
# user_id -> section (today/added) for VK time update
vk_time_sessions: TTLCache[int, str] = TTLCache(maxsize=64, ttl=3600)

# superadmin user_id -> pending partner user_id
partner_info_sessions: TTLCache[int, int] = TTLCache(maxsize=64, ttl=3600)
# user_id -> (festival_id, field?) for festival editing
festival_edit_sessions: TTLCache[int, tuple[int, str | None]] = TTLCache(maxsize=64, ttl=3600)

# pending event text/photo input
add_event_sessions: TTLCache[int, bool] = TTLCache(maxsize=64, ttl=3600)
# waiting for a date for events listing
events_date_sessions: TTLCache[int, bool] = TTLCache(maxsize=64, ttl=3600)

# remove leading command like /addevent or /addevent@bot
def strip_leading_cmd(text: str, cmds: tuple[str, ...] = ("addevent",)) -> str:
    """Strip a leading command and following whitespace from *text*.

    Handles optional ``@username`` after the command and any whitespace,
    including newlines, that follows it.  Matching is case-insensitive and
    spans across lines (``re.S``).
    """

    if not text:
        return text
    cmds_re = "|".join(re.escape(c) for c in cmds)
    # allow NBSP after the command as whitespace
    return re.sub(
        rf"^/({cmds_re})(@\w+)?[\s\u00A0]*",
        "",
        text,
        flags=re.I | re.S,
    )


def _strip_leading_arrows(text: str) -> str:
    """Remove leading arrow emojis and related joiners."""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (" ", "\t", "\n", "\r", "\u00A0"):
            i += 1
            continue
        name = unicodedata.name(ch, "")
        if "ARROW" in name:
            i += 1
            # skip variation selectors and ZWJ
            while i < len(text) and ord(text[i]) in (0xFE0F, 0x200D):
                i += 1
            continue
        break
    return text[i:]


def normalize_addevent_text(text: str) -> str:
    """Normalize user-provided event text.

    Replaces NBSP with regular spaces, normalizes newlines, strips leading
    arrow emojis, joins lines and trims whitespace.
    """

    if not text:
        return ""
    text = text.replace("\u00A0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_leading_arrows(text)
    lines = [ln.strip() for ln in text.splitlines()]
    return " ".join(lines).strip()


async def send_usage_fast(bot: Bot, chat_id: int) -> None:
    """Send Usage help with quick retry and background fallback."""

    usage = "Usage: /addevent <text>"

    async def _direct_send():
        if isinstance(bot, SafeBot):
            # bypass SafeBot retry logic
            return await Bot.send_message(bot, chat_id, usage)
        return await bot.send_message(chat_id, usage)

    for attempt in range(2):
        try:
            await asyncio.wait_for(_direct_send(), timeout=1.0)
            return
        except Exception:
            if attempt == 0:
                continue

            async def _bg() -> None:
                with contextlib.suppress(Exception):
                    await bot.send_message(chat_id, usage)

            asyncio.create_task(_bg())

# cache for settings values to reduce DB hits
settings_cache: TTLCache[str, str | None] = TTLCache(maxsize=64, ttl=300)

# queue for background event processing
# limit the queue to avoid unbounded growth if parsing slows down
add_event_queue: asyncio.Queue[tuple[str, types.Message, bool, int]] = asyncio.Queue(
    maxsize=200
)
ADD_EVENT_TIMEOUT = int(os.getenv("ADD_EVENT_TIMEOUT", "180"))
ADD_EVENT_RETRY_DELAYS = [30, 60, 120]  # сек
ADD_EVENT_MAX_ATTEMPTS = len(ADD_EVENT_RETRY_DELAYS) + 1
_ADD_EVENT_LAST_DEQUEUE_TS: float = 0.0

# queue for post-commit event update jobs


async def _watch_add_event_worker(app: web.Application, db: Database, bot: Bot):
    worker: asyncio.Task = app["add_event_worker"]
    while True:
        alive = not worker.done()
        if DEBUG:
            logging.debug(
                "QSTAT add_event=%d worker_alive=%s",
                add_event_queue.qsize(),
                alive,
            )
        # воркер умер — перезапускаем
        if not alive and not worker.cancelled():
            try:
                exc = worker.exception()
            except Exception as e:  # pragma: no cover - defensive
                exc = e
            logging.error("add_event_queue_worker crashed: %s", exc)
            worker = asyncio.create_task(add_event_queue_worker(db, bot))
            app["add_event_worker"] = worker
            logging.info("add_event_queue_worker restarted")
        # воркер жив, но очередь не разгребается слишком долго -> «stalled»
        else:
            try:
                stalled_for = _time.monotonic() - _ADD_EVENT_LAST_DEQUEUE_TS
            except Exception:
                stalled_for = 0
            if add_event_queue.qsize() > 0 and stalled_for > 120:
                logging.error(
                    "add_event_queue stalled for %.0fs; restarting worker",
                    stalled_for,
                )
                worker.cancel()
                with contextlib.suppress(Exception):
                    await worker
                worker = asyncio.create_task(add_event_queue_worker(db, bot))
                app["add_event_worker"] = worker
                _ADD_EVENT_LAST_DEQUEUE_TS = _time.monotonic()
        await asyncio.sleep(10)

# toggle for uploading images to catbox
CATBOX_ENABLED: bool = False
# toggle for sending photos to VK
VK_PHOTOS_ENABLED: bool = False
_supabase_client: "Client | None" = None  # type: ignore[name-defined]
_vk_user_token_bad: str | None = None
_vk_captcha_needed: bool = False
_shared_session: ClientSession | None = None
# backward-compatible aliases used in tests
_http_session: ClientSession | None = None
_vk_session: ClientSession | None = None

# Telegraph API rejects pages over ~64&nbsp;kB. Use a slightly lower limit
# to decide when month pages should be split into two parts.
TELEGRAPH_LIMIT = 60000

def rough_size(nodes: Iterable[dict], limit: int | None = None) -> int:
    """Return an approximate size of Telegraph nodes in bytes.

    The calculation serializes each node individually and sums the byte lengths,
    which avoids materialising the whole JSON representation at once.  If
    ``limit`` is provided the iteration stops once the accumulated size exceeds
    it.
    """
    total = 0
    for n in nodes:
        total += len(json.dumps(n, ensure_ascii=False).encode())
        if limit is not None and total > limit:
            break
    return total


def slugify(text: str) -> str:
    """Return a simple ASCII slug for the given ``text``.

    Non-alphanumeric characters are replaced with ``-`` and the result is
    lower‑cased.  If the slug becomes empty, ``"page"`` is returned to avoid
    invalid Telegraph paths.
    """

    text_norm = unicodedata.normalize("NFKD", text)
    text_ascii = text_norm.encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text_ascii).strip("-").lower()
    return slug or "page"

# Timeout for Telegraph API operations (in seconds)
TELEGRAPH_TIMEOUT = float(os.getenv("TELEGRAPH_TIMEOUT", "30"))

# Timeout for posting ICS files to Telegram (in seconds)
ICS_POST_TIMEOUT = float(os.getenv("ICS_POST_TIMEOUT", "30"))

# Timeout for general HTTP requests (in seconds)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

# Limit concurrent HTTP requests
HTTP_SEMAPHORE = asyncio.Semaphore(2)

# Глобальный «тяжёлый» семафор оставляем для редких CPU-тяжёлых секций,
# но сетевые вызовы ограничиваем узкими шлюзами:
HEAVY_SEMAPHORE = asyncio.Semaphore(1)
TG_SEND_SEMAPHORE = asyncio.Semaphore(int(os.getenv("TG_SEND_CONCURRENCY", "2")))
VK_SEMAPHORE = asyncio.Semaphore(int(os.getenv("VK_CONCURRENCY", "1")))
TELEGRAPH_SEMAPHORE = asyncio.Semaphore(int(os.getenv("TELEGRAPH_CONCURRENCY", "1")))

# Maximum size (in bytes) for downloaded files
MAX_DOWNLOAD_SIZE = int(os.getenv("MAX_DOWNLOAD_SIZE", str(5 * 1024 * 1024)))


def detect_image_type(data: bytes) -> str | None:
    """Return image subtype based on magic numbers."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return None

# Timeout for OpenAI 4o requests (in seconds)
FOUR_O_TIMEOUT = float(os.getenv("FOUR_O_TIMEOUT", "60"))

# Limit prompt/response sizes for LLM calls (characters)
FOUR_O_PROMPT_LIMIT = int(os.getenv("FOUR_O_PROMPT_LIMIT", "4000"))
FOUR_O_RESPONSE_LIMIT = int(os.getenv("FOUR_O_RESPONSE_LIMIT", "1000"))


# Run blocking Telegraph API calls with a timeout and simple retries
async def telegraph_call(func, /, *args, retries: int = 3, **kwargs):
    """Execute a Telegraph API call in a thread with timeout and retries.

    Telegraph can occasionally respond very slowly causing timeouts.  In that
    case we retry the operation a few times before giving up.  This makes
    synchronization of month/weekend pages more reliable and helps ensure that
    events are not missed due to transient network issues.
    """

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            if DEBUG:
                mem_info(f"{func.__name__} before")
            async with TELEGRAPH_SEMAPHORE:
                async with span("telegraph"):
                    res = await asyncio.wait_for(
                        asyncio.to_thread(func, *args, **kwargs), TELEGRAPH_TIMEOUT
                    )
            if DEBUG:
                mem_info(f"{func.__name__} after")
            return res
        except asyncio.TimeoutError as e:
            last_exc = e
            if attempt < retries - 1:
                logging.warning("telegraph_call retry=%d", attempt + 1)
                await asyncio.sleep(2**attempt)
                continue
            raise TelegraphException("Telegraph request timed out") from e

    # If we exit the loop without returning or raising, raise the last exception
    if last_exc:
        raise TelegraphException("Telegraph request failed") from last_exc


async def telegraph_create_page(tg: Telegraph, *args, **kwargs):
    return await telegraph_call(tg.create_page, *args, **kwargs)


def seconds_to_next_minute(now: datetime) -> float:
    next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    return (next_minute - now).total_seconds()


# main menu buttons
MENU_ADD_EVENT = "\u2795 Добавить событие"
MENU_EVENTS = "\U0001f4c5 События"


class IPv4AiohttpSession(AiohttpSession):
    """Aiohttp session that forces IPv4 and reuses the shared ClientSession."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connector_init.update(
            family=socket.AF_INET,
            limit=6,
            limit_per_host=3,
            ttl_dns_cache=300,
            keepalive_timeout=30,
        )

    async def create_session(self) -> ClientSession:
        self._session = get_shared_session()
        return self._session

    async def close(self) -> None:  # pragma: no cover - cleanup
        await close_shared_session()



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



async def get_tz_offset(db: Database) -> str:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT value FROM setting WHERE key='tz_offset'"
        )
        row = await cursor.fetchone()
    offset = row[0] if row else "+00:00"
    global LOCAL_TZ
    LOCAL_TZ = offset_to_timezone(offset)
    return offset


async def set_tz_offset(db: Database, value: str):
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO setting(key, value) VALUES('tz_offset', ?)",
            (value,),
        )
        await conn.commit()
    global LOCAL_TZ
    LOCAL_TZ = offset_to_timezone(value)


async def get_catbox_enabled(db: Database) -> bool:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT value FROM setting WHERE key='catbox_enabled'"
        )
        row = await cursor.fetchone()
    return bool(row and row[0] == "1")


async def set_catbox_enabled(db: Database, value: bool):
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO setting(key, value) VALUES('catbox_enabled', ?)",
            ("1" if value else "0",),
        )
        await conn.commit()
    global CATBOX_ENABLED
    CATBOX_ENABLED = value


async def get_vk_photos_enabled(db: Database) -> bool:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT value FROM setting WHERE key='vk_photos_enabled'"
        )
        row = await cursor.fetchone()
    return bool(row and row[0] == "1")


async def set_vk_photos_enabled(db: Database, value: bool):
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO setting(key, value) VALUES('vk_photos_enabled', ?)",
            ("1" if value else "0",),
        )
        await conn.commit()
    global VK_PHOTOS_ENABLED
    VK_PHOTOS_ENABLED = value


async def get_setting_value(db: Database, key: str) -> str | None:
    cached = settings_cache.get(key)
    if cached is not None or key in settings_cache:
        return cached
    async with db.get_session() as session:
        setting = await session.get(Setting, key)
        value = setting.value if setting else None
    settings_cache[key] = value
    return value


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
    if value is None:
        settings_cache.pop(key, None)
    else:
        settings_cache[key] = value


async def get_partner_last_run(db: Database) -> date | None:
    val = await get_setting_value(db, "partner_last_run")
    return date.fromisoformat(val) if val else None


async def set_partner_last_run(db: Database, d: date) -> None:
    await set_setting_value(db, "partner_last_run", d.isoformat())


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


async def get_section_hash(db: Database, page_key: str, section_key: str) -> str | None:
    rows = await db.exec_driver_sql(
        "SELECT hash FROM page_section_cache WHERE page_key=? AND section_key=?",
        (page_key, section_key),
    )
    return rows[0][0] if rows else None


async def set_section_hash(db: Database, page_key: str, section_key: str, h: str) -> None:
    await db.exec_driver_sql(
        """
        INSERT INTO page_section_cache(page_key, section_key, hash)
        VALUES(?, ?, ?)
        ON CONFLICT(page_key, section_key)
        DO UPDATE SET hash=excluded.hash, updated_at=CURRENT_TIMESTAMP
        """,
        (page_key, section_key, h),
    )


async def close_shared_session() -> None:
    global _shared_session, _http_session, _vk_session
    if _shared_session is not None and not _shared_session.closed:
        with contextlib.suppress(Exception):
            await _shared_session.close()
    _shared_session = _http_session = _vk_session = None


def _close_shared_session_sync() -> None:
    try:
        asyncio.run(close_shared_session())
    except Exception:
        pass


def _create_session() -> ClientSession:
    connector = TCPConnector(
        family=socket.AF_INET,
        limit=6,
        ttl_dns_cache=300,
        limit_per_host=3,
        keepalive_timeout=30,
    )
    timeout = ClientTimeout(total=HTTP_TIMEOUT)
    try:
        return ClientSession(connector=connector, timeout=timeout)
    except TypeError:
        try:
            return ClientSession(timeout=timeout)
        except TypeError:
            return ClientSession()


def get_shared_session() -> ClientSession:
    global _shared_session, _http_session, _vk_session
    if (
        _shared_session is None
        or getattr(_shared_session, "closed", False)
        or _http_session is None
        or _vk_session is None
    ):
        _shared_session = _create_session()
        _http_session = _vk_session = _shared_session
        atexit.register(_close_shared_session_sync)
    return _shared_session


def get_vk_session() -> ClientSession:
    return get_shared_session()


def get_http_session() -> ClientSession:
    return get_shared_session()


async def close_vk_session() -> None:
    await close_shared_session()


async def close_http_session() -> None:
    await close_shared_session()


def redact_token(tok: str) -> str:
    return tok[:6] + "…" + tok[-4:] if tok and len(tok) > 10 else "<redacted>"


def _vk_user_token() -> str | None:
    """Return user token unless it was previously marked invalid."""
    token = os.getenv("VK_USER_TOKEN")
    global _vk_user_token_bad
    if token and _vk_user_token_bad and token != _vk_user_token_bad:
        _vk_user_token_bad = None
    if token and token != _vk_user_token_bad:
        return token
    return None


class VKAPIError(Exception):
    """Exception raised for VK API errors."""

    def __init__(self, code: int | None, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


async def _vk_api(
    method: str,
    params: dict,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
    token_kind: str = "group",
) -> dict:
    """Call VK API with token fallback."""
    global _vk_captcha_needed
    if _vk_captcha_needed:
        raise VKAPIError(14, "Captcha needed")
    tokens: list[tuple[str, str]] = []
    if token:
        tokens.append((token_kind, token))
    else:
        user_token = _vk_user_token()
        if user_token:
            tokens.append(("user", user_token))
        if VK_TOKEN:
            tokens.append(("group", VK_TOKEN))
    last_err: dict | None = None
    session = get_vk_session()
    for kind, token in tokens:
        params["access_token"] = token
        params["v"] = "5.131"
        logging.info(
            "calling VK API %s using %s token %s", method, kind, redact_token(token)
        )
        async def _call():
            resp = await http_call(
                f"vk.{method}",
                "POST",
                f"https://api.vk.com/method/{method}",
                timeout=HTTP_TIMEOUT,
                data=params,
            )
            return resp.json()

        err: dict | None = None
        last_msg: str | None = None
        for attempt, delay in enumerate(BACKOFF_DELAYS, start=1):
            async with VK_SEMAPHORE:
                async with span("vk-send"):
                    data = await asyncio.wait_for(_call(), HTTP_TIMEOUT)
            if "error" not in data:
                if attempt > 1 and last_msg:
                    logging.warning(
                        "vk api %s retried=%d last_error=%s",
                        method,
                        attempt - 1,
                        last_msg,
                    )
                return data
            err = data["error"]
            msg = err.get("error_msg", "")
            code = err.get("error_code")
            if code == 14:
                _vk_captcha_needed = True
                if db and bot:
                    await notify_superadmin(db, bot, "VK captcha needed")
                raise VKAPIError(code, msg)
            if kind == "user" and code in {5, 27}:
                global _vk_user_token_bad
                if _vk_user_token_bad != token:
                    _vk_user_token_bad = token
                    if db and bot:
                        await notify_superadmin(db, bot, "VK_USER_TOKEN expired")
                break
            if any(x in msg.lower() for x in ("already deleted", "already exists")):
                logging.info("vk no-retry error: %s", msg)
                return data
            last_msg = msg
            if attempt == len(BACKOFF_DELAYS):
                logging.warning(
                    "vk api %s failed after %d attempts: %s",
                    method,
                    attempt,
                    msg,
                )
                break
            await asyncio.sleep(delay)
        last_err = err
        continue
    if last_err:
        raise VKAPIError(last_err.get("error_code"), last_err.get("error_msg", ""))
    raise VKAPIError(None, "VK token missing")


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
        if DEBUG:
            mem_info("VK upload before")
        data = await _vk_api(
            "photos.getWallUploadServer",
            {"group_id": group_id.lstrip("-")},
            db,
            bot,
            token=token,
        )
        upload_url = data["response"]["upload_url"]
        session = get_http_session()
        async def _download():
            async with span("http"):
                async with HTTP_SEMAPHORE:
                    async with session.get(url) as resp:
                        if resp.content_length and resp.content_length > MAX_DOWNLOAD_SIZE:
                            raise ValueError("file too large")
                        data = await resp.content.read(MAX_DOWNLOAD_SIZE + 1)
                        if len(data) > MAX_DOWNLOAD_SIZE:
                            raise ValueError("file too large")
                        return data

        img_bytes = await asyncio.wait_for(_download(), HTTP_TIMEOUT)
        form = FormData()
        ctype = "image/jpeg"
        kind = detect_image_type(img_bytes)
        if kind:
            ctype = f"image/{kind}"
        form.add_field(
            "photo",
            img_bytes,
            filename="image.jpg",
            content_type=ctype,
        )
        async def _upload():
            async with span("http"):
                async with HTTP_SEMAPHORE:
                    async with session.post(upload_url, data=form) as up:
                        return await up.json()

        upload_result = await asyncio.wait_for(_upload(), HTTP_TIMEOUT)
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
        if DEBUG:
            mem_info("VK upload after")
        return f"photo{info['owner_id']}_{info['id']}"
    except Exception as e:
        logging.error("VK photo upload failed: %s", e)
        return None


def get_supabase_client() -> "Client | None":  # type: ignore[name-defined]
    if os.getenv("SUPABASE_DISABLED") == "1" or not (
        SUPABASE_URL and SUPABASE_KEY
    ):
        return None
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client, Client  # локальный импорт
        from supabase.client import ClientOptions

        options = ClientOptions()
        options.httpx_client = httpx.Client(timeout=HTTP_TIMEOUT)
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
        atexit.register(close_supabase_client)
    return _supabase_client


def close_supabase_client() -> None:
    global _supabase_client
    if _supabase_client is not None:
        with contextlib.suppress(Exception):
            _supabase_client.postgrest.session.close()
        _supabase_client = None


async def get_festival(db: Database, name: str) -> Festival | None:
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.name == name)
        )
        return result.scalar_one_or_none()


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
    start_date: str | None = None,
    end_date: str | None = None,
    location_name: str | None = None,
    location_address: str | None = None,
    city: str | None = None,
    source_text: str | None = None,
) -> tuple[Festival, bool]:
    """Return festival and creation flag."""
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
            if start_date and not fest.start_date:
                fest.start_date = start_date
                updated = True
            if end_date and not fest.end_date:
                fest.end_date = end_date
                updated = True
            if location_name and not fest.location_name:
                fest.location_name = location_name
                updated = True
            if location_address and not fest.location_address:
                fest.location_address = location_address
                updated = True
            if city and not fest.city:
                fest.city = city
                updated = True
            if source_text and not fest.source_text:
                fest.source_text = source_text
                updated = True
            if updated:
                session.add(fest)
                await session.commit()
            return fest, False
        fest = Festival(
            name=name,
            full_name=full_name,
            photo_url=photo_url,
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            location_address=location_address,
            city=city,
            source_text=source_text,
        )
        session.add(fest)
        await session.commit()
        logging.info("created festival %s", name)
        return fest, True


async def get_superadmin_id(db: Database) -> int | None:
    """Return the Telegram ID of the superadmin if present."""
    async with db.get_session() as session:
        result = await session.execute(
            select(User.user_id).where(User.is_superadmin.is_(True))
        )
        return result.scalars().first()


async def notify_superadmin(db: Database, bot: Bot, text: str):
    """Send a message to the superadmin with retry on network errors."""
    admin_id = await get_superadmin_id(db)
    if not admin_id:
        return
    try:
        async with span("tg-send"):
            await bot.send_message(admin_id, text)
        return
    except (ClientOSError, ServerDisconnectedError, asyncio.TimeoutError) as e:
        logging.warning("notify_superadmin failed: %s; retry with fresh session", e)
        timeout = ClientTimeout(total=HTTP_TIMEOUT)
        async with IPv4AiohttpSession(timeout=timeout) as session:
            fresh_bot = SafeBot(bot.token, session=session)
            try:
                async with span("tg-send"):
                    await fresh_bot.send_message(admin_id, text)
            except Exception as e2:
                logging.error("failed to notify superadmin: %s", e2)
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
        stream = await session.stream_scalars(
            select(User)
            .where(User.is_partner.is_(True), User.blocked.is_(False))
            .execution_options(yield_per=100)
        )
        count = 0
        async for p in stream:
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
                async with span("tg-send"):
                    await bot.send_message(
                        p.user_id,
                        "\u26a0\ufe0f Вы не добавляли мероприятия на прошлой неделе",
                    )
                p.last_partner_reminder = now
                notified.append(p)
            count += 1
            if count % 100 == 0:
                await asyncio.sleep(0)
        await session.commit()
    await asyncio.sleep(0)
    return notified


async def dump_database(db: Database) -> bytes:
    """Return a SQL dump using the shared connection."""
    async with db.raw_conn() as conn:
        lines: list[str] = []
        async for line in conn.iterdump():
            lines.append(line)
    return "\n".join(lines).encode("utf-8")


async def restore_database(data: bytes, db: Database):
    """Replace current database with the provided dump."""
    path = db.path
    if os.path.exists(path):
        os.remove(path)
    conn = await db.raw_conn()
    await conn.executescript(data.decode("utf-8"))
    await conn.commit()
    await conn.close()
    db._conn = None  # type: ignore[attr-defined]
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
        async with span("tg-send"):
            await bot.download(message.photo[-1].file_id, destination=bio)
        images.append((bio.getvalue(), "photo.jpg"))
    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        bio = BytesIO()
        async with span("tg-send"):
            await bot.download(message.document.file_id, destination=bio)
        name = message.document.file_name or "image.jpg"
        images.append((bio.getvalue(), name))
    return images[:3]


async def upload_images(
    images: list[tuple[bytes, str]]
) -> tuple[list[str], list[str], str]:
    """Upload images to Catbox with Telegraph fallback."""
    catbox_urls: list[str] = []
    tg_urls: list[str] = []
    catbox_msg = ""
    if CATBOX_ENABLED and images:
        session = get_http_session()
        for data, name in images[:3]:
            if len(data) > 5 * 1024 * 1024:
                logging.warning("catbox skip %s: too large", name)
                catbox_msg += f"{name}: too large; "
                continue
            if not detect_image_type(data):
                logging.warning("catbox skip %s: not image", name)
                catbox_msg += f"{name}: not image; "
                continue
            uploaded = False
            delays = [0.7, 1.5, 3.0]
            for attempt, delay in enumerate(delays, 1):
                try:
                    form = FormData()
                    form.add_field("reqtype", "fileupload")
                    form.add_field("fileToUpload", data, filename=name)
                    async with span("http"):
                        async with HTTP_SEMAPHORE:
                            async with session.post(
                                "https://catbox.moe/user/api.php", data=form
                            ) as resp:
                                text_r = await resp.text()
                                if resp.status == 200 and text_r.startswith("http"):
                                    url = text_r.strip()
                                    catbox_urls.append(url)
                                    catbox_msg += "ok; "
                                    logging.info("catbox uploaded %s", url)
                                    uploaded = True
                                    break
                                logging.warning(
                                    "catbox upload failed %s: %s %s", name, resp.status, text_r
                                )
                                if resp.status in {412, 429} or resp.status >= 500:
                                    if attempt < len(delays):
                                        await asyncio.sleep(delay)
                                        continue
                except Exception as e:
                    logging.error("catbox error %s: %s", name, e)
                break
            if not uploaded:
                url = await telegraph_upload(data, name)
                if url:
                    tg_urls.append(url)
                    catbox_msg += "tg ok; "
                    logging.info("telegraph uploaded %s", url)
                else:
                    catbox_msg += f"{name}: both failed; "
                    logging.error("both uploads failed for %s", name)
        catbox_msg = catbox_msg.strip("; ")
    elif images:
        catbox_msg = "disabled"
    return catbox_urls, tg_urls, catbox_msg


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


def normalize_event_type(
    title: str, description: str, event_type: str | None
) -> str | None:
    """Return corrected event type, marking film screenings as ``кинопоказ``."""
    text = f"{title} {description}".lower()
    if event_type in (None, "", "спектакль"):
        if any(word in text for word in ("кино", "фильм", "кинопоказ", "киносеанс")):
            return "кинопоказ"
    return event_type


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


def festival_location(fest: Festival, events: Iterable[Event]) -> str | None:
    """Return display string for festival venue(s)."""
    pairs = {(e.location_name, e.city) for e in events if e.location_name}
    if not pairs:
        parts: list[str] = []
        if fest.location_name:
            parts.append(fest.location_name)
        elif fest.location_address:
            parts.append(fest.location_address)
        if fest.city:
            parts.append(f"#{fest.city}")
        return ", ".join(parts) if parts else None
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
    limit: int | None = None,
) -> list[tuple[date | None, date | None, Festival]]:
    """Return festivals that are current or upcoming."""
    if today is None:
        today = date.today()
    today_str = today.isoformat()
    async with db.get_session() as session:
        ev_dates = (
            select(
                Event.festival,
                func.min(Event.date).label("start"),
                func.max(func.coalesce(Event.end_date, Event.date)).label("end"),
            )
            .group_by(Event.festival)
            .subquery()
        )

        stmt = (
            select(
                Festival.id,
                Festival.name,
                Festival.telegraph_path,
                Festival.vk_post_url,
                func.coalesce(Festival.start_date, ev_dates.c.start).label("start"),
                func.coalesce(Festival.end_date, ev_dates.c.end).label("end"),
            )
            .outerjoin(ev_dates, ev_dates.c.festival == Festival.name)
            .where(func.coalesce(Festival.end_date, ev_dates.c.end) >= today_str)
        )
        if exclude:
            stmt = stmt.where(Festival.name != exclude)
        stmt = stmt.order_by(func.coalesce(Festival.start_date, ev_dates.c.start))
        if limit:
            stmt = stmt.limit(limit)
        start_t = _time.perf_counter()
        rows = (await session.execute(stmt)).all()
        dur = (_time.perf_counter() - start_t) * 1000
        logging.debug("db upcoming_festivals took %.1f ms", dur)

    data: list[tuple[date | None, date | None, Festival]] = []
    for fid, name, path, vk_url, start_s, end_s in rows:
        start = parse_iso_date(start_s) if start_s else None
        end = parse_iso_date(end_s) if end_s else None
        if not start and not end:
            continue
        fest = Festival(id=fid, name=name, telegraph_path=path, vk_post_url=vk_url)
        data.append((start, end, fest))
    return data


async def build_festivals_list_nodes(
    db: Database, *, exclude: str | None = None, today: date | None = None
) -> list[dict]:
    """Return Telegraph nodes listing upcoming festivals."""
    items = await upcoming_festivals(db, today=today, exclude=exclude, limit=10)
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
            url = fest.telegraph_url
            if not url and fest.telegraph_path:
                url = f"https://telegra.ph/{fest.telegraph_path}"
            if url:
                nodes.append(
                    {
                        "tag": "p",
                        "children": [
                            {
                                "tag": "a",
                                "attrs": {"href": url},
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
    items = await upcoming_festivals(db, today=today, exclude=exclude, limit=10)
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


async def _build_festival_nav_block(
    db: Database,
    *,
    exclude: str | None = None,
    today: date | None = None,
    items: list[tuple[date | None, date | None, Festival]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Return navigation blocks for festival pages and VK posts."""
    if today is None:
        today = date.today()
    if items is None:
        items = await upcoming_festivals(db, today=today, exclude=exclude)
    else:
        if exclude:
            items = [t for t in items if t[2].name != exclude]
    if not items:
        return [], []
    groups: dict[str, list[Festival]] = {}
    for start, end, fest in items:
        if start and start <= today <= (end or start):
            month = today.strftime("%Y-%m")
        else:
            month = (start or today).strftime("%Y-%m")
        groups.setdefault(month, []).append(fest)

    nodes: list[dict] = []
    nodes.extend(telegraph_br())
    nodes.append({"tag": "h3", "children": ["Ближайшие фестивали"]})
    lines: list[str] = [VK_BLANK_LINE, "Ближайшие фестивали"]
    for month in sorted(groups.keys()):
        month_name = month_name_nominative(month)
        nodes.append({"tag": "h4", "children": [month_name]})
        lines.append(month_name)
        for fest in groups[month]:
            url = fest.telegraph_url
            if not url and fest.telegraph_path:
                url = f"https://telegra.ph/{fest.telegraph_path}"
            if url:
                nodes.append(
                    {
                        "tag": "p",
                        "children": [
                            {
                                "tag": "a",
                                "attrs": {"href": url},
                                "children": [fest.name],
                            }
                        ],
                    }
                )
            else:
                nodes.append({"tag": "p", "children": [fest.name]})
            if fest.vk_post_url:
                lines.append(f"[{fest.vk_post_url}|{fest.name}]")
            else:
                lines.append(fest.name)
    return nodes, lines


async def build_festivals_nav_block(
    db: Database,
) -> tuple[str, list[str], bool]:
    """Return cached navigation HTML and lines for all festivals.

    Stores HTML fragment and its hash in the ``setting`` table.
    Returns ``html``, ``lines`` and a boolean flag indicating whether
    the cached fragment changed.
    """
    nodes, lines = await _build_festival_nav_block(db)
    from telegraph.utils import nodes_to_html

    html = nodes_to_html(nodes) if nodes else ""
    new_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
    old_hash = await get_setting_value(db, "fest_nav_hash")
    if old_hash != new_hash:
        await set_setting_value(db, "fest_nav_hash", new_hash)
        await set_setting_value(db, "fest_nav_html", html)
        changed = True
    else:
        cached_html = await get_setting_value(db, "fest_nav_html")
        if cached_html is not None:
            html = cached_html
        changed = False
    return html, lines, changed


ICS_LABEL = "Добавить в календарь на телефоне (ICS)"
MONTH_NAV_START = "<!--month-nav-start-->"
MONTH_NAV_END = "<!--month-nav-end-->"
FEST_NAV_START = "<!-- FEST_NAV_START -->"
FEST_NAV_END = "<!-- FEST_NAV_END -->"

FOOTER_LINK_HTML = (
    '<p>&#8203;</p>'
    '<p><a href="https://t.me/kenigevents">Полюбить Калининград Анонсы</a></p>'
    '<p>&#8203;</p>'
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


def apply_festival_nav(html_content: str, html_block: str) -> tuple[str, str]:
    """Insert or replace festival navigation block and Telegram footer.

    Returns updated HTML and the strategy used: ``"markers"`` when existing
    HTML comment markers were found (or when simply appending) and
    ``"fallback_h3"`` when we had to fall back to the legacy
    ``<h3>Ближайшие фестивали</h3>`` heading.
    """

    strategy = "markers"
    start = html_content.find(FEST_NAV_START)
    if start != -1:
        end = html_content.find(FEST_NAV_END, start)
        if end != -1:
            html_content = html_content[:start] + html_content[end + len(FEST_NAV_END) :]
    else:
        heading = "<h3>Ближайшие фестивали</h3>"
        idx = html_content.rfind(heading)
        if idx != -1:
            strategy = "fallback_h3"
            html_content = html_content[:idx]

    html_content = f"{html_content}{FEST_NAV_START}{html_block}{FEST_NAV_END}"
    html_content = apply_footer_link(html_content)
    return html_content, strategy


def apply_footer_link(html_content: str) -> str:
    """Ensure the Telegram channel link footer is present once."""
    pattern = re.compile(
        r'<p><a href="https://t\.me/kenigevents">[^<]+</a></p><p>(?:&nbsp;|&#8203;)</p>'
    )
    html_content = pattern.sub("", html_content).rstrip()
    return html_content + FOOTER_LINK_HTML


async def build_month_nav_html(db: Database) -> str:
    cur_month = datetime.now(LOCAL_TZ).strftime("%Y-%m")
    async with db.get_session() as session:
        result = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= cur_month)
            .order_by(MonthPage.month)
        )
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
    _load_icalendar()
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
    async with span("http"):
        client = get_supabase_client()
        if not client:
            logging.debug("Supabase disabled")
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
        if DEBUG:
            logging.debug("ICS upload %s size=%d", path, len(content.encode("utf-8")))
        try:
            logging.info("Uploading ICS to %s/%s", SUPABASE_BUCKET, path)
            storage = client.storage.from_(SUPABASE_BUCKET)
            await asyncio.to_thread(
                storage.upload,
                path,
                content.encode("utf-8"),
                {
                    "content-type": ICS_CONTENT_TYPE,
                    "content-disposition": ICS_CONTENT_DISP_TEMPLATE.format(name=path),
                    "upsert": "true",
                },
            )
            url = await asyncio.to_thread(storage.get_public_url, path)
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

    logging.info(
        "posting ics asset to channel %s with caption %s",
        channel.channel_id,
        caption.replace("\n", " | "),
    )

    try:
        if event.ics_post_id:
            media = types.InputMediaDocument(
                media=file, caption=caption, parse_mode="HTML"
            )
            await bot.edit_message_media(
                chat_id=channel.channel_id,
                message_id=event.ics_post_id,
                media=media,
            )
            url = build_channel_post_url(channel, event.ics_post_id)
            logging.info("updated ics in asset channel: %s", url)
            return url, event.ics_post_id
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
        storage = client.storage.from_(SUPABASE_BUCKET)
        async with span("http"):
            await asyncio.to_thread(storage.remove, [path])
    except Exception as e:
        logging.error("Failed to delete ics: %s", e)


async def delete_asset_post(event: Event, db: Database, bot: Bot):
    if not event.ics_post_id:
        return
    try:
        channel = await get_asset_channel(db)
        if not channel:
            return
        async with span("tg-send"):
            await bot.delete_message(channel.channel_id, event.ics_post_id)
    except Exception as e:
        logging.error("failed to delete asset message: %s", e)


async def remove_calendar_button(event: Event, bot: Bot):
    """Remove calendar button from the original channel post."""
    if not (event.source_chat_id and event.source_message_id):
        return
    try:
        async with span("tg-send"):
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


@lru_cache(maxsize=1)
def _read_base_prompt() -> str:
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
    return prompt


@lru_cache(maxsize=2)
def _prompt_cache(festival_key: tuple[str, ...] | None) -> str:
    txt = _read_base_prompt()
    if festival_key:
        txt += "\nKnown festivals:\n" + "\n".join(festival_key)
    return txt


def _build_prompt(festival_names: list[str] | None) -> str:
    key = tuple(sorted(festival_names)) if festival_names else None
    return _prompt_cache(key)


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
    prompt = _build_prompt(festival_names)
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
    # ensure we start the network request with as little memory as possible
    gc.collect()
    logging.info("Sending 4o parse request to %s", url)
    session = get_http_session()
    async def _call():
        async with span("http"):
            async with HTTP_SEMAPHORE:
                resp = await session.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return await resp.json()
    data_raw = await asyncio.wait_for(_call(), FOUR_O_TIMEOUT)
    content = (
        data_raw.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{}")
        .strip()
    )
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug("4o content snippet: %s", content[:1000])
    del data_raw
    gc.collect()
    if content.startswith("```"):
        content = content.strip("`\n")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logging.error("Invalid JSON from 4o: %s", content)
        raise
    festival = None
    if isinstance(data, dict):
        festival = data.get("festival")
        if "events" in data and isinstance(data["events"], list):
            parse_event_via_4o._festival = festival
            return data["events"]
        parse_event_via_4o._festival = festival
        return [data]
    if isinstance(data, list):
        parse_event_via_4o._festival = None
        return data
    logging.error("Unexpected 4o format: %s", data)
    parse_event_via_4o._festival = None
    raise RuntimeError("bad 4o response")


async def ask_4o(
    text: str,
    *,
    system_prompt: str | None = None,
    response_format: dict | None = None,
    max_tokens: int = 1000,
) -> str:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if len(text) > FOUR_O_PROMPT_LIMIT:
        text = text[:FOUR_O_PROMPT_LIMIT]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})
    payload: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    logging.info("Sending 4o ask request to %s", url)
    session = get_http_session()

    async def _call():
        async with span("http"):
            async with HTTP_SEMAPHORE:
                resp = await session.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return await resp.json()

    data = await asyncio.wait_for(_call(), FOUR_O_TIMEOUT)
    logging.debug("4o response: %s", data)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if len(content) > FOUR_O_RESPONSE_LIMIT:
        content = content[:FOUR_O_RESPONSE_LIMIT]
    del data
    gc.collect()
    return content


async def check_duplicate_via_4o(ev: Event, new: Event) -> Tuple[bool, str, str]:
    """Ask the LLM whether two events are duplicates."""
    prompt = (
        "Existing event:\n"
        f"Title: {ev.title}\nDescription: {ev.description}\nLocation: {ev.location_name} {ev.location_address}\n"
        "New event:\n"
        f"Title: {new.title}\nDescription: {new.description}\nLocation: {new.location_name} {new.location_address}\n"
        'Are these the same event? Respond with JSON {"duplicate": true|false, "title": "", "short_description": ""}.'
    )
    start = _time.perf_counter()
    try:
        ans = await ask_4o(
            prompt,
            system_prompt=(
                'Return a JSON object {"duplicate": true|false, "title": "", '
                '"short_description": ""} and nothing else.'
            ),
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        ans = ans.strip()
        if ans.startswith("```"):
            ans = re.sub(r'^```[a-zA-Z]*\n', '', ans)
            if ans.endswith("```"):
                ans = ans[:-3]
            ans = ans.strip()
        data = json.loads(ans)
        dup = bool(data.get("duplicate"))
        title = data.get("title", "")
        desc = data.get("short_description", "")
    except Exception as e:  # pragma: no cover - simple
        logging.warning("duplicate check invalid JSON: %s", e)
        dup, title, desc = False, "", ""
    latency = _time.perf_counter() - start
    logging.info("duplicate check: %s, %.3f", dup, latency)
    return dup, title, desc


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
    async with span("render"):
        buttons = [
            [types.KeyboardButton(text=MENU_ADD_EVENT)],
            [types.KeyboardButton(text=MENU_EVENTS)],
        ]
        markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
    async with span("tg-send"):
        await bot.send_message(chat_id, "Choose action", reply_markup=markup)


async def handle_start(message: types.Message, db: Database, bot: Bot):
    async with span("db-query"):
        async with db.get_session() as session:
            result = await session.execute(select(User))
            user_count = len(result.scalars().all())
            user = await session.get(User, message.from_user.id)
            if user:
                if user.blocked:
                    msg = "Access denied"
                    menu_user = None
                else:
                    if user.is_partner:
                        org = f" ({user.organization})" if user.organization else ""
                        msg = f"You are partner{org}"
                    else:
                        msg = "Bot is running"
                    menu_user = user
            elif user_count == 0:
                session.add(
                    User(
                        user_id=message.from_user.id,
                        username=message.from_user.username,
                        is_superadmin=True,
                    )
                )
                await session.commit()
                msg = "You are superadmin"
                menu_user = await session.get(User, message.from_user.id)
            else:
                msg = "Use /register to apply"
                menu_user = None

    await bot.send_message(message.chat.id, msg)
    if menu_user:
        await send_main_menu(bot, menu_user, message.chat.id)


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
    events_date_sessions.pop(message.from_user.id, None)
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
    if data.startswith("requeue:"):
        eid = int(data.split(":", 1)[1])
        now = datetime.utcnow()
        async with db.get_session() as session:
            res = await session.execute(
                select(JobOutbox).where(
                    JobOutbox.event_id == eid, JobOutbox.status == JobStatus.error
                )
            )
            jobs = res.scalars().all()
            for j in jobs:
                j.status = JobStatus.pending
                j.attempts += 1
                j.updated_at = now
                j.next_run_at = now
                session.add(j)
            await session.commit()
        await callback.answer("Перезапущено")
        await run_event_update_jobs(
            db, bot, notify_chat_id=callback.message.chat.id, event_id=eid
        )
    elif data.startswith("approve") or data.startswith("reject") or data.startswith("partner"):
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
    elif data.startswith("festdays:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            fest = await session.get(Festival, fid)
            if not fest or (user and user.blocked):
                await callback.answer("Not authorized", show_alert=True)
                return
            start = parse_iso_date(fest.start_date or "")
            end = parse_iso_date(fest.end_date or "")
            if not start or not end:
                await callback.answer(
                    "Не задан период фестиваля. Сначала отредактируйте даты.",
                    show_alert=True,
                )
                return
            logging.info("festdays start fid=%s name=%s", fid, fest.name)
            events: list[tuple[Event, bool]] = []
            for i in range((end - start).days + 1):
                day = start + timedelta(days=i)
                event = Event(
                    title=f"{fest.full_name or fest.name} - день {i+1}",
                    description="",
                    festival=fest.name,
                    date=day.isoformat(),
                    time="",
                    location_name=fest.location_name or "",
                    location_address=fest.location_address,
                    city=fest.city,
                    source_text="",
                    creator_id=user.user_id if user else None,
                )
                saved, added = await upsert_event(session, event)
                await schedule_event_update_tasks(db, saved)
                events.append((saved, added))
            await session.commit()
        async with db.get_session() as session:
            notify_user = await session.get(User, callback.from_user.id)
        for saved, added in events:
            lines = [
                f"title: {saved.title}",
                f"date: {saved.date}",
                f"festival: {saved.festival}",
            ]
            if saved.location_name:
                lines.append(f"location_name: {saved.location_name}")
            if saved.city:
                lines.append(f"city: {saved.city}")
            await callback.message.answer("Event added\n" + "\n".join(lines))
            await notify_event_added(db, bot, notify_user, saved, added)

        asyncio.create_task(sync_festival_page(db, fest.name))
        asyncio.create_task(sync_festival_vk_post(db, fest.name, bot))
        summary = [
            f"Создано {len(events)} событий для {fest.name}.",
        ]
        if fest.telegraph_url:
            summary.append(f"Страница фестиваля: {fest.telegraph_url}")
        summary.append("Что дальше?")
        await callback.message.answer("\n".join(summary))
        await show_festival_edit_menu(callback.from_user.id, fest, bot)
        logging.info(
            "festdays created %d events for %s", len(events), fest.name
        )
        await callback.answer()
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
                                append_text=False,
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
                            append_text=False,
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
        now = None
        logging.info("manual daily send: channel=%s now=%s", cid, (now or datetime.now(tz)))
        await send_daily_announcement(db, bot, cid, tz, record=False, now=now)
        await callback.answer("Sent")
    elif data.startswith("dailysendtom:"):
        cid = int(data.split(":")[1])
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz) + timedelta(days=1)
        logging.info("manual daily send: channel=%s now=%s", cid, (now or datetime.now(tz)))
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
        events_date_sessions[callback.from_user.id] = True
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
        if f.ticket_url:
            parts.append(f"ticket: {f.ticket_url}")
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


def _copy_fields(dst: Event, src: Event) -> None:
    for f in (
        "title",
        "description",
        "festival",
        "source_text",
        "location_name",
        "location_address",
        "ticket_price_min",
        "ticket_price_max",
        "ticket_link",
        "event_type",
        "emoji",
        "end_date",
        "is_free",
        "pushkin_card",
        "photo_urls",
        "photo_count",
    ):
        setattr(dst, f, getattr(src, f))


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
            (ev.location_name or "").strip().lower()
            == (new.location_name or "").strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        title_ratio = SequenceMatcher(
            None, (ev.title or "").lower(), (new.title or "").lower()
        ).ratio()
        if title_ratio >= 0.9:
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        if (
            (ev.location_name or "").strip().lower()
            == (new.location_name or "").strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        title_ratio = SequenceMatcher(
            None, (ev.title or "").lower(), (new.title or "").lower()
        ).ratio()
        if title_ratio >= 0.9:
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        if (
            (ev.location_name or "").strip().lower()
            == (new.location_name or "").strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        title_ratio = SequenceMatcher(
            None, (ev.title or "").lower(), (new.title or "").lower()
        ).ratio()
        if title_ratio >= 0.9:
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        if (
            (ev.location_name or "").strip().lower()
            == (new.location_name or "").strip().lower()
            and (ev.location_address or "").strip().lower()
            == (new.location_address or "").strip().lower()
        ):
            _copy_fields(ev, new)
            await session.commit()
            logging.info("upsert_event: updated event id=%s", ev.id)
            return ev, False

        title_ratio = SequenceMatcher(
            None, (ev.title or "").lower(), (new.title or "").lower()
        ).ratio()
        loc_ratio = SequenceMatcher(
            None,
            (ev.location_name or "").lower(),
            (new.location_name or "").lower(),
        ).ratio()
        if title_ratio >= 0.6 and loc_ratio >= 0.6:
            _copy_fields(ev, new)
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
                _copy_fields(ev, new)
                ev.title = title or ev.title
                ev.description = desc or ev.description
                await session.commit()
                logging.info("upsert_event: updated event id=%s", ev.id)
                return ev, False
    new.added_at = datetime.utcnow()
    session.add(new)
    await session.commit()
    logging.info("upsert_event: inserted new event id=%s", new.id)
    return new, True


async def enqueue_job(
    db: Database, event_id: int, task: JobTask, payload: dict | None = None
) -> None:
    async with db.get_session() as session:
        stmt = select(JobOutbox).where(
            JobOutbox.event_id == event_id, JobOutbox.task == task
        )
        res = await session.execute(stmt)
        job = res.scalar_one_or_none()
        now = datetime.utcnow()
        if job and job.status in {
            JobStatus.pending,
            JobStatus.running,
            JobStatus.done,
        }:
            return
        if job:
            job.status = JobStatus.pending
            job.payload = payload
            job.attempts = 0
            job.last_error = None
            job.updated_at = now
            job.next_run_at = now
            session.add(job)
        else:
            session.add(
                JobOutbox(
                    event_id=event_id,
                    task=task,
                    payload=payload,
                    status=JobStatus.pending,
                    updated_at=now,
                    next_run_at=now,
                )
            )
        await session.commit()


async def schedule_event_update_tasks(db: Database, ev: Event) -> None:
    eid = ev.id
    await enqueue_job(db, eid, JobTask.telegraph_build)
    if not is_vk_wall_url(ev.source_post_url):
        await enqueue_job(db, eid, JobTask.vk_sync)
    await enqueue_job(db, eid, JobTask.month_pages)
    d = parse_iso_date(ev.date)
    if d:
        await enqueue_job(db, eid, JobTask.week_pages)
        w_start = weekend_start_for_date(d)
        if w_start:
            await enqueue_job(db, eid, JobTask.weekend_pages)
    if ev.festival:
        await enqueue_job(db, eid, JobTask.festival_pages)
    logging.info("scheduled event tasks for %s", eid)


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
        # Free any lingering objects before heavy LLM call to reduce peak memory
        gc.collect()
        if DEBUG:
            mem_info("LLM before")
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

        if DEBUG:
            mem_info("LLM after")
        festival_info = getattr(parse_event_via_4o, "_festival", None)
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
    catbox_urls, tg_urls, catbox_msg_global = await upload_images(images)
    links_iter = iter(extract_links_from_html(html_text) if html_text else [])
    source_text_clean = re.sub(
        r"<[^>]+>", "", _vk_expose_links(html_text or text)
    )

    if festival_info:
        fest_name = festival_info.get("name") or festival_info.get("festival")
        start = canonicalize_date(festival_info.get("start_date") or festival_info.get("date"))
        end = canonicalize_date(festival_info.get("end_date"))
        loc_name = festival_info.get("location_name")
        loc_addr = festival_info.get("location_address")
        city = festival_info.get("city")
        loc_addr = strip_city_from_address(loc_addr, city)
        photo_u = (catbox_urls or tg_urls)[0] if (catbox_urls or tg_urls) else None
        fest_obj, created = await ensure_festival(
            db,
            fest_name,
            full_name=festival_info.get("full_name"),
            photo_url=photo_u,
            start_date=start,
            end_date=end,
            location_name=loc_name,
            location_address=loc_addr,
            city=city,
            source_text=source_text_clean,
        )
        if not parsed and created:
            await sync_festival_page(db, fest_obj.name)
            await sync_festival_vk_post(db, fest_obj.name, bot)
            async with db.get_session() as session:
                res = await session.execute(
                    select(Festival).where(Festival.name == fest_obj.name)
                )
                fest_obj = res.scalar_one_or_none()
            lines = [f"festival: {fest_obj.name}"]
            if fest_obj.telegraph_url:
                lines.append(f"telegraph: {fest_obj.telegraph_url}")
            if fest_obj.vk_post_url:
                lines.append(f"vk_post: {fest_obj.vk_post_url}")
            if fest_obj.start_date:
                lines.append(f"start: {fest_obj.start_date}")
            if fest_obj.end_date:
                lines.append(f"end: {fest_obj.end_date}")
            if fest_obj.location_name:
                lines.append(f"location_name: {fest_obj.location_name}")
            if fest_obj.city:
                lines.append(f"city: {fest_obj.city}")
            results.append((fest_obj, True, lines, "festival"))
            logging.info("festival %s created without events", fest_obj.name)
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
            photo_count=len(catbox_urls or tg_urls),
            photo_urls=catbox_urls or tg_urls,
        )

        base_event.event_type = normalize_event_type(
            base_event.title, base_event.description, base_event.event_type
        )

        if base_event.festival:
            photo_u = (catbox_urls or tg_urls)[0] if (catbox_urls or tg_urls) else None
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
            await schedule_event_update_tasks(db, saved)
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
            status = "added" if added else "updated"
            results.append((saved, added, lines, status))
            first = False
    logging.info("add_events_from_text finished with %d results", len(results))
    del parsed
    gc.collect()
    return results


def _event_lines(ev: Event) -> list[str]:
    lines = [
        f"title: {ev.title}",
        f"date: {ev.date}",
        f"time: {ev.time}",
        f"location_name: {ev.location_name}",
    ]
    if ev.location_address:
        lines.append(f"location_address: {ev.location_address}")
    if ev.city:
        lines.append(f"city: {ev.city}")
    if ev.festival:
        lines.append(f"festival: {ev.festival}")
    if ev.description:
        lines.append(f"description: {ev.description}")
    if ev.event_type:
        lines.append(f"type: {ev.event_type}")
    if ev.ticket_price_min is not None:
        lines.append(f"price_min: {ev.ticket_price_min}")
    if ev.ticket_price_max is not None:
        lines.append(f"price_max: {ev.ticket_price_max}")
    if ev.ticket_link:
        lines.append(f"ticket_link: {ev.ticket_link}")
    return lines


async def handle_add_event(
    message: types.Message, db: Database, bot: Bot, *, using_session: bool = False
):
    text_raw = message.text or message.caption or ""
    logging.info(
        "handle_add_event start: user=%s len=%d", message.from_user.id, len(text_raw)
    )
    if using_session:
        text_raw = strip_leading_cmd(text_raw)
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
    html_text = message.html_text or message.caption_html
    if html_text:
        html_text = strip_leading_cmd(html_text)
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

            bot=None,
        )
    except Exception as e:
        await bot.send_message(message.chat.id, f"LLM error: {e}")
        return
    if not results:
        await bot.send_message(
            message.chat.id,
            "Не удалось распознать событие. Пример:\n"
            "Название | 21.08.2025 | 19:00 | Город, Адрес",
        )
        return
    logging.info("handle_add_event parsed %d results", len(results))
    grouped: dict[int, tuple[Event, bool]] = {}
    fest_msgs: list[tuple[Festival, list[str]]] = []
    for saved, added, lines, status in results:
        if isinstance(saved, Festival):
            fest_msgs.append((saved, lines))
            continue
        info = grouped.get(saved.id)
        if info:
            grouped[saved.id] = (saved, info[1] or added)
        else:
            grouped[saved.id] = (saved, added)

    for fest, lines in fest_msgs:
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[[types.InlineKeyboardButton(
                text="Создать события по дням",
                callback_data=f"festdays:{fest.id}")]]
        )
        await bot.send_message(
            message.chat.id,
            "Festival added\n" + "\n".join(lines),
            reply_markup=markup,
        )

    for saved, added in grouped.values():
        status = "added" if added else "updated"
        logging.info("handle_add_event %s event id=%s", status, saved.id)
        lines = _event_lines(saved)
        buttons_first: list[types.InlineKeyboardButton] = []
        if (
            not saved.is_free
            and saved.ticket_price_min is None
            and saved.ticket_price_max is None
        ):
            buttons_first.append(
                types.InlineKeyboardButton(
                    text="\u2753 Это бесплатное мероприятие",
                    callback_data=f"markfree:{saved.id}",
                )
            )
        buttons_first.append(
            types.InlineKeyboardButton(
                text="\U0001f6a9 Переключить на тихий режим",
                callback_data=f"togglesilent:{saved.id}",
            )
        )
        buttons_second = [
            types.InlineKeyboardButton(
                text="Добавить ссылку на Вк",
                switch_inline_query_current_chat=f"/vklink {saved.id} ",
            )
        ]
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[buttons_first, buttons_second]
        )
        await bot.send_message(
            message.chat.id,
            f"Event {status}\n" + "\n".join(lines),
            reply_markup=markup,
        )
        await notify_event_added(db, bot, user, saved, added)
        await publish_event_progress(saved, db, bot, message.chat.id)
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
    await schedule_event_update_tasks(db, event)
    lines = [
        f"title: {event.title}",
        f"date: {event.date}",
        f"time: {event.time}",
        f"location_name: {event.location_name}",
    ]
    status = "added" if added else "updated"
    logging.info("handle_add_event_raw %s event id=%s", status, event.id)
    buttons_first: list[types.InlineKeyboardButton] = []
    if (
        not event.is_free
        and event.ticket_price_min is None
        and event.ticket_price_max is None
    ):
        buttons_first.append(
            types.InlineKeyboardButton(
                text="\u2753 Это бесплатное мероприятие",
                callback_data=f"markfree:{event.id}",
            )
        )
    buttons_first.append(
        types.InlineKeyboardButton(
            text="\U0001f6a9 Переключить на тихий режим",
            callback_data=f"togglesilent:{event.id}",
        )
    )
    buttons_second = [
        types.InlineKeyboardButton(
            text="Добавить ссылку на Вк",
            switch_inline_query_current_chat=f"/vklink {event.id} ",
        )
    ]
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[buttons_first, buttons_second]
    )
    await bot.send_message(
        message.chat.id,
        f"Event {status}\n" + "\n".join(lines),
        reply_markup=markup,
    )
    await notify_event_added(db, bot, user, event, added)
    await publish_event_progress(event, db, bot, message.chat.id)
    logging.info("handle_add_event_raw finished for user %s", message.from_user.id)


async def enqueue_add_event(message: types.Message, db: Database, bot: Bot):
    """Queue an event addition for background processing."""
    using_session = message.from_user.id in add_event_sessions
    if using_session:
        add_event_sessions.pop(message.from_user.id, None)
    try:
        add_event_queue.put_nowait(("regular", message, using_session, 0))
    except asyncio.QueueFull:
        logging.warning(
            "enqueue_add_event queue full for user=%s", message.from_user.id
        )
        await bot.send_message(
            message.chat.id,
            "Очередь обработки переполнена, попробуйте позже",
        )
        return
    preview = (message.text or message.caption or "").strip().replace("\n", " ")[:80]
    logging.info(
        "enqueue_add_event user=%s chat=%s kind=%s preview=%r queue=%d",
        message.from_user.id,
        message.chat.id,
        "regular",
        preview,
        add_event_queue.qsize(),
    )
    await bot.send_message(message.chat.id, "Пост принят на обработку")


async def enqueue_add_event_raw(message: types.Message, db: Database, bot: Bot):
    """Queue a raw event addition for background processing."""
    try:
        add_event_queue.put_nowait(("raw", message, False, 0))
    except asyncio.QueueFull:
        logging.warning(
            "enqueue_add_event_raw queue full for user=%s", message.from_user.id
        )
        await bot.send_message(
            message.chat.id,
            "Очередь обработки переполнена, попробуйте позже",
        )
        return
    logging.info(
        "enqueue_add_event_raw user=%s queue=%d",
        message.from_user.id,
        add_event_queue.qsize(),
    )
    await bot.send_message(message.chat.id, "Пост принят на обработку")


async def add_event_queue_worker(db: Database, bot: Bot, limit: int = 2):
    """Background worker to process queued events with timeout & retries."""

    global _ADD_EVENT_LAST_DEQUEUE_TS
    while True:
        kind, msg, using_session, attempts = await add_event_queue.get()
        _ADD_EVENT_LAST_DEQUEUE_TS = _time.monotonic()
        start = _time.perf_counter()
        timed_out = False
        try:
            async def _run():
                if kind == "regular":
                    await handle_add_event(msg, db, bot, using_session=using_session)
                else:
                    await handle_add_event_raw(msg, db, bot)
            await asyncio.wait_for(_run(), timeout=ADD_EVENT_TIMEOUT)
        except asyncio.TimeoutError:
            timed_out = True
            logging.error(
                "add_event timeout user=%s attempt=%d",
                getattr(msg.from_user, "id", None),
                attempts + 1,
            )
            try:
                await bot.send_message(
                    msg.chat.id,
                    f"Задача зависла (таймаут {ADD_EVENT_TIMEOUT} c). "
                    f"Снята и будет повторена. Попытка {attempts+1}/{ADD_EVENT_MAX_ATTEMPTS}.",
                )
            except Exception:
                logging.warning("notify timeout failed")
        except Exception:  # pragma: no cover - log unexpected errors
            logging.exception("add_event_queue_worker error")
            try:
                await bot.send_message(
                    msg.chat.id,
                    "❌ Ошибка при обработке... Попробуйте ещё раз...",
                )
                if using_session:
                    add_event_sessions[msg.from_user.id] = True
            except Exception:  # pragma: no cover - notify fail
                logging.exception("add_event_queue_worker notify failed")
        finally:
            dur = (_time.perf_counter() - start) * 1000.0
            logging.info("add_event_queue item done in %.0f ms", dur)
            add_event_queue.task_done()

        if timed_out and attempts + 1 < ADD_EVENT_MAX_ATTEMPTS:
            delay = ADD_EVENT_RETRY_DELAYS[min(attempts, len(ADD_EVENT_RETRY_DELAYS) - 1)]
            async def _requeue() -> None:
                await asyncio.sleep(delay)
                try:
                    add_event_queue.put_nowait((kind, msg, using_session, attempts + 1))
                    logging.info(
                        "add_event requeued user=%s attempt=%d delay=%s",
                        getattr(msg.from_user, "id", None),
                        attempts + 2,
                        delay,
                    )
                except asyncio.QueueFull:
                    logging.error("requeue failed: queue full")
            asyncio.create_task(_requeue())
        elif timed_out:
            try:
                await bot.send_message(
                    msg.chat.id,
                    "Задача по добавлению события не выполнена после повторов. "
                    "Попробуйте ещё раз позже.",
                )
            except Exception:
                pass


BACKOFF_SCHEDULE = [30, 120, 600, 3600]


TASK_LABELS = {
    "telegraph_build": "Telegraph (событие)",
    "vk_sync": "VK",
    "month_pages": "Страница месяца",
    "week_pages": "Неделя",
    "weekend_pages": "Выходные",
    "festival_pages": "Фестиваль",
}


async def _job_result_link(task: JobTask, event_id: int, db: Database) -> str | None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            return None
        if task == JobTask.telegraph_build:
            return ev.telegraph_url
        if task == JobTask.vk_sync:
            return ev.source_vk_post_url
        if task == JobTask.month_pages:
            d = parse_iso_date(ev.date.split("..", 1)[0])
            month_key = d.strftime("%Y-%m") if d else None
            if month_key:
                page = await session.get(MonthPage, month_key)
                return page.url if page else None
            return None
        if task == JobTask.week_pages:
            d = parse_iso_date(ev.date.split("..", 1)[0])
            if d:
                w_start = week_start_for_date(d)
                page = await session.get(WeekPage, w_start.isoformat())
                return page.vk_post_url if page else None
            return None
        if task == JobTask.weekend_pages:
            d = parse_iso_date(ev.date.split("..", 1)[0])
            w_start = weekend_start_for_date(d) if d else None
            if w_start:
                page = await session.get(WeekendPage, w_start.isoformat())
                return page.url if page else None
            return None
    return None


async def reconcile_job_outbox(db: Database) -> None:
    now = datetime.utcnow()
    async with db.get_session() as session:
        await session.execute(
            update(JobOutbox)
            .where(JobOutbox.status == JobStatus.running)
            .values(status=JobStatus.error, next_run_at=now, updated_at=now)
        )
        await session.commit()


async def _run_due_jobs_once(
    db: Database,
    bot: Bot,
    notify: Callable[[JobTask, JobStatus, bool, str | None, str | None], Awaitable[None]] | None = None,
    only_event: int | None = None,
) -> int:
    now = datetime.utcnow()
    async with db.get_session() as session:
        stmt = (
            select(JobOutbox)
            .where(
                JobOutbox.status.in_([JobStatus.pending, JobStatus.error]),
                JobOutbox.next_run_at <= now,
            )
            .order_by(JobOutbox.id)
        )
        if only_event is not None:
            stmt = stmt.where(JobOutbox.event_id == only_event)
        jobs = (await session.execute(stmt)).scalars().all()
    processed = 0
    for job in jobs:
        async with db.get_session() as session:
            exists_stmt = (
                select(JobOutbox.id)
                .where(
                    JobOutbox.event_id == job.event_id,
                    JobOutbox.id < job.id,
                    JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                )
                .limit(1)
            )
            if (await session.execute(exists_stmt)).first():
                continue
            obj = await session.get(JobOutbox, job.id)
            if not obj or obj.status not in (JobStatus.pending, JobStatus.error):
                continue
            obj.status = JobStatus.running
            obj.updated_at = datetime.utcnow()
            session.add(obj)
            await session.commit()
        run_id = uuid.uuid4().hex
        attempt = job.attempts + 1
        logging.info(
            "TASK_START event=%s task=%s run_id=%s attempt=%d",
            job.event_id,
            job.task.value,
            run_id,
            attempt,
        )
        start = _time.perf_counter()
        changed = True
        try:
            handler = JOB_HANDLERS[job.task.value]
            async with span(
                "event_pipeline", step=job.task.value, event_id=job.event_id
            ):
                res = await handler(job.event_id, db, bot)
            rebuild = isinstance(res, str) and res == "rebuild"
            changed = res if isinstance(res, bool) else True
            link = (
                await _job_result_link(job.task, job.event_id, db)
                if changed
                else None
            )
            if rebuild and link:
                link += " (forced rebuild)"
            status = JobStatus.done
            err = None
            took_ms = (_time.perf_counter() - start) * 1000
            short = link or ("ok" if changed else "nochange")
            logging.info(
                "TASK_DONE event=%s task=%s run_id=%s attempt=%d took_ms=%.0f result=%s",
                job.event_id,
                job.task.value,
                run_id,
                attempt,
                took_ms,
                short,
            )
        except Exception as exc:  # pragma: no cover - log and backoff
            took_ms = (_time.perf_counter() - start) * 1000
            err = str(exc)
            logging.error(
                "TASK_FAIL event=%s task=%s run_id=%s attempt=%d took_ms=%.0f err=\"%s\"",
                job.event_id,
                job.task.value,
                run_id,
                attempt,
                took_ms,
                err.splitlines()[0],
            )
            logging.exception("job %s failed", job.id)
            status = JobStatus.error
            link = None
        text = None
        async with db.get_session() as session:
            obj = await session.get(JobOutbox, job.id)
            send = True
            if obj:
                prev = obj.last_result
                obj.status = status
                obj.last_error = err
                obj.updated_at = datetime.utcnow()
                if status == JobStatus.done:
                    cur_res = link if (changed and link) else ("ok" if changed else "nochange")
                    if cur_res == prev:
                        send = False
                    obj.last_result = cur_res
                    obj.next_run_at = datetime.utcnow()
                else:
                    obj.attempts += 1
                    delay = BACKOFF_SCHEDULE[min(obj.attempts - 1, len(BACKOFF_SCHEDULE) - 1)]
                    obj.next_run_at = datetime.utcnow() + timedelta(seconds=delay)
                session.add(obj)
                await session.commit()
            if notify and send:
                await notify(job.task, status, changed, link, err)
        processed += 1
    return processed


async def _log_job_outbox_stats(db: Database) -> None:
    now = datetime.utcnow()
    async with db.get_session() as session:
        cnt_rows = await session.execute(
            select(JobOutbox.status, func.count()).group_by(JobOutbox.status)
        )
        counts = {s: c for s, c in cnt_rows.all()}
        avg_age_res = await session.execute(
            select(
                func.avg(
                    func.strftime('%s', 'now') - func.strftime('%s', JobOutbox.updated_at)
                )
            ).where(JobOutbox.status == JobStatus.pending)
        )
        avg_age = avg_age_res.scalar() or 0
        lag_res = await session.execute(
            select(func.min(JobOutbox.next_run_at)).where(
                JobOutbox.status == JobStatus.pending
            )
        )
        next_run = lag_res.scalar()
    lag = (now - next_run).total_seconds() if next_run else 0
    if lag < 0:
        lag = 0
    logging.info(
        "WORKER_STATE pending=%d running=%d error=%d avg_age_s=%.1f lag_s=%.1f",
        counts.get(JobStatus.pending, 0),
        counts.get(JobStatus.running, 0),
        counts.get(JobStatus.error, 0),
        avg_age,
        lag,
    )


async def job_outbox_worker(db: Database, bot: Bot, interval: float = 2.0):
    last_log = 0.0
    while True:
        try:
            await _run_due_jobs_once(db, bot)
        except Exception:  # pragma: no cover - log unexpected errors
            logging.exception("job_outbox_worker cycle failed")
        if _time.monotonic() - last_log >= 30.0:
            await _log_job_outbox_stats(db)
            last_log = _time.monotonic()
        await asyncio.sleep(interval)


async def run_event_update_jobs(
    db: Database,
    bot: Bot,
    *,
    notify_chat_id: int | None = None,
    event_id: int | None = None,
) -> None:
    async def notifier(
        task: JobTask,
        status: JobStatus,
        changed: bool,
        link: str | None,
        err: str | None,
    ) -> None:
        label = TASK_LABELS[task.value]
        text = None
        if status == JobStatus.done:
            if changed:
                text = f"{label}: OK"
                if link:
                    text += f" — {link}"
            else:
                text = f"{label}: без изменений"
        elif status == JobStatus.error:
            err_short = err.splitlines()[0] if err else ""
            text = f"{label}: ERROR: {err_short}"
        if notify_chat_id is not None and text:
            await bot.send_message(notify_chat_id, text)

    while await _run_due_jobs_once(
        db, bot, notifier if notify_chat_id is not None else None, event_id
    ):
        await asyncio.sleep(0)


async def update_telegraph_event_page(
    event_id: int, db: Database, bot: Bot | None
) -> str | None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            return None
        if not ev.ics_url:
            ics = await upload_ics(ev, db)
            if ics:
                ev.ics_url = ics
        display_link = False if ev.source_post_url else True
        html_content, _, _ = await build_source_page_content(
            ev.title or "Event",
            ev.source_text,
            ev.source_post_url,
            ev.source_text,
            None,
            ev.ics_url,
            db,
            display_link=display_link,
            catbox_urls=ev.photo_urls,
        )
        from telegraph.utils import html_to_nodes

        nodes = html_to_nodes(html_content)
        new_hash = content_hash(html_content)
        if ev.content_hash == new_hash and ev.telegraph_url:
            await session.commit()
            return ev.telegraph_url
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            await session.commit()
            return ev.telegraph_url
        tg = Telegraph(access_token=token)
        title = ev.title or "Event"
        if not ev.telegraph_path:
            data = await telegraph_create_page(
                tg,
                title=title,
                author_name="Полюбить Калининград Анонсы",
                content=nodes,
                return_content=False,
            )
            ev.telegraph_url = normalize_telegraph_url(data.get("url"))
            ev.telegraph_path = data.get("path")
        else:
            await telegraph_call(
                tg.edit_page,
                path=ev.telegraph_path,
                title=title,
                content=nodes,
                return_content=False,
            )
        ev.content_hash = new_hash
        session.add(ev)
        await session.commit()
        return ev.telegraph_url


async def patch_month_page_for_date(db: Database, telegraph: Telegraph, month_key: str, d: date) -> bool:
    """Patch a single day's section on a month page if it changed."""
    page_key = f"telegraph:month:{month_key}"
    section_key = f"day:{d.isoformat()}"
    start = _time.perf_counter()

    async with db.get_session() as session:
        page = await session.get(MonthPage, month_key)
        if not page or not page.path:
            return False
        result = await session.execute(
            select(Event)
            .where(Event.date.like(f"{d.isoformat()}%"))
            .order_by(Event.time)
        )
        events = result.scalars().all()
        fest_names = [e.festival for e in events if e.festival]
        fest_map: dict[str, Festival] = {}
        if fest_names:
            res_f = await session.execute(
                select(Festival).where(Festival.name.in_(fest_names))
            )
            fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

    for ev in events:
        if ev.festival:
            setattr(ev, "_festival", fest_map.get(ev.festival.casefold()))

    html_section = render_month_day_section(d, events)
    new_hash = content_hash(html_section)
    old_hash = await get_section_hash(db, page_key, section_key)
    if new_hash == old_hash:
        dur = (_time.perf_counter() - start) * 1000
        logging.info(
            "month_patch page_key=%s day=%s changed=False dur=%.0fms",
            page_key,
            d.isoformat(),
            dur,
        )
        return False

    async def tg_call(func, /, *args, **kwargs):
        last: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs), 7
                )
            except Exception as e:
                last = e
                if attempt == 0:
                    await asyncio.sleep(random.uniform(0, 1))
                    continue
                raise

    page_data = await tg_call(telegraph.get_page, page.path, return_html=True)
    html_content = page_data.get("content") or page_data.get("content_html") or ""
    html_content = unescape_html_comments(html_content)
    title = page_data.get("title") or month_key
    start_marker = DAY_START(d)
    end_marker = DAY_END(d)
    legacy_start = f"<!-- DAY:{d.isoformat()} START -->"
    legacy_end = f"<!-- DAY:{d.isoformat()} END -->"
    marker_type = "new" if start_marker in html_content and end_marker in html_content else (
        "legacy" if legacy_start in html_content and legacy_end in html_content else "none"
    )
    logging.info(
        "month_patch page_key=%s day=%s markers=%s",
        page_key,
        d.isoformat(),
        marker_type,
    )
    branch = "replace"
    if marker_type == "new":
        updated_html = replace_between_markers(
            html_content, start_marker, end_marker, html_section
        )
    elif marker_type == "legacy":
        branch = "migrate"
        updated_html = replace_between_markers(
            html_content, legacy_start, legacy_end, html_section
        )
        updated_html = updated_html.replace(legacy_start, start_marker).replace(
            legacy_end, end_marker
        )
    else:
        branch = "rebuild"
        logging.warning(
            "month_patch page_key=%s day=%s markers_missing", page_key, d.isoformat()
        )
        await sync_month_page(db, month_key)
        await set_section_hash(db, page_key, section_key, new_hash)
        dur = (_time.perf_counter() - start) * 1000
        logging.info(
            "month_patch page_key=%s day=%s branch=%s changed=True dur=%.0fms",
            page_key,
            d.isoformat(),
            branch,
            dur,
        )
        return "rebuild"
    updated_html = lint_telegraph_html(updated_html)
    try:
        await tg_call(
            telegraph.edit_page, page.path, title=title, html_content=updated_html
        )
    except TelegraphException as e:
        if "CONTENT_TOO_BIG" in str(e):
            logging.warning(
                "month_patch page_key=%s day=%s content too big, rebuilding",
                page_key,
                d.isoformat(),
            )
            await sync_month_page(db, month_key)
        else:
            raise
    await set_section_hash(db, page_key, section_key, new_hash)
    dur = (_time.perf_counter() - start) * 1000
    logging.info(
        "month_patch page_key=%s day=%s branch=%s changed=True dur=%.0fms",
        page_key,
        d.isoformat(),
        branch,
        dur,
    )
    return True


async def update_month_pages_for(event_id: int, db: Database, bot: Bot | None) -> bool:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        return

    start_date = parse_iso_date(ev.date.split("..", 1)[0])
    end_date = None
    if ".." in ev.date:
        end_part = ev.date.split("..", 1)[1]
        end_date = parse_iso_date(end_part)
    elif ev.end_date:
        end_date = parse_iso_date(ev.end_date.split("..", 1)[0])
    dates: list[date] = []
    if start_date:
        dates.append(start_date)
        if end_date and end_date >= start_date:
            span_days = min((end_date - start_date).days, 30)
            for i in range(1, span_days + 1):
                dates.append(start_date + timedelta(days=i))

    # group all affected days by month to ensure month pages exist
    months: dict[str, list[date]] = {}
    for d in dates:
        months.setdefault(d.strftime("%Y-%m"), []).append(d)

    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        for month in months:
            await sync_month_page(db, month)
        return True

    tg = Telegraph(access_token=token, domain="telegra.ph")

    changed_any = False
    rebuild_any = False
    for month, month_dates in months.items():
        # ensure the month page is created before attempting a patch
        await sync_month_page(db, month)
        for d in month_dates:
            changed = await patch_month_page_for_date(db, tg, month, d)
            if changed:
                changed_any = True
                if changed == "rebuild":
                    rebuild_any = True
    return "rebuild" if rebuild_any else changed_any


async def update_weekend_pages_for(event_id: int, db: Database, bot: Bot | None) -> None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        return
    d = parse_iso_date(ev.date)
    w_start = weekend_start_for_date(d) if d else None
    if w_start:
        await sync_weekend_page(db, w_start.isoformat())


async def update_week_pages_for(event_id: int, db: Database, bot: Bot | None) -> None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        return
    d = parse_iso_date(ev.date)
    if d:
        w_start = week_start_for_date(d)
        await sync_vk_week_post(db, w_start.isoformat(), bot)


async def update_festival_pages_for_event(event_id: int, db: Database, bot: Bot | None) -> None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev or not ev.festival:
        return
    await sync_festival_page(db, ev.festival)
    await sync_festival_vk_post(db, ev.festival, bot)
    nav_html, nav_lines, changed = await build_festivals_nav_block(db)
    if changed:
        asyncio.create_task(
            refresh_nav_on_all_festivals(
                db, bot, nav_html=nav_html, nav_lines=nav_lines
            )
        )


async def publish_event_progress(event: Event, db: Database, bot: Bot, chat_id: int) -> None:
    async with db.get_session() as session:
        jobs = (
            await session.execute(
                select(JobOutbox.task).where(JobOutbox.event_id == event.id)
            )
        ).scalars().all()
    tasks = [job for job in jobs if job.value in TASK_LABELS]
    progress: dict[JobTask, dict[str, str]] = {
        t: {"icon": "\U0001f504", "suffix": ""} for t in tasks
    }
    lines = [f"\U0001f504 {TASK_LABELS[t.value]}" for t in tasks]
    head = "Идёт процесс публикации, ждите"
    text = head if not lines else head + "\n" + "\n".join(lines)
    msg = await bot.send_message(chat_id, text)

    async def updater(
        task: JobTask,
        status: JobStatus,
        changed: bool,
        link: str | None,
        err: str | None,
    ) -> None:
        if task not in progress:
            return
        if status == JobStatus.done:
            icon = "✅"
            if changed:
                suffix = f" — {link}" if link else ""
            else:
                suffix = " — без изменений"
        else:
            icon = "❌"
            suffix = f" — {err.splitlines()[0] if err else ''}"
        progress[task] = {"icon": icon, "suffix": suffix}
        all_done = all(info["icon"] != "\U0001f504" for info in progress.values())
        head = "Готово" if all_done else "Идёт процесс публикации, ждите"
        lines = [
            f"{info['icon']} {TASK_LABELS[t.value]}{info['suffix']}" for t, info in progress.items()
        ]
        text = head if not lines else head + "\n" + "\n".join(lines)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=text,
        )

    while await _run_due_jobs_once(db, bot, updater, event.id):
        await asyncio.sleep(0)

    if progress:
        all_done = all(info["icon"] != "\U0001f504" for info in progress.values())
        head = "Готово" if all_done else "Идёт процесс публикации, ждите"
        lines = [
            f"{info['icon']} {TASK_LABELS[t.value]}{info['suffix']}" for t, info in progress.items()
        ]
        text = head if not lines else head + "\n" + "\n".join(lines)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=text,
        )
    else:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text="Готово",
        )


async def job_sync_vk_source_post(event_id: int, db: Database, bot: Bot | None) -> None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev or is_vk_wall_url(ev.source_post_url):
        return
    new_hash = content_hash(ev.source_text or "")
    if ev.content_hash == new_hash and ev.source_vk_post_url:
        return
    vk_url = await sync_vk_source_post(ev, ev.source_text, db, bot, ics_url=ev.ics_url)
    async with db.get_session() as session:
        obj = await session.get(Event, event_id)
        if obj:
            if vk_url:
                obj.source_vk_post_url = vk_url
            obj.content_hash = new_hash
            session.add(obj)
            await session.commit()


JOB_HANDLERS = {
    "telegraph_build": update_telegraph_event_page,
    "vk_sync": job_sync_vk_source_post,
    "month_pages": update_month_pages_for,
    "week_pages": update_week_pages_for,
    "weekend_pages": update_weekend_pages_for,
    "festival_pages": update_festival_pages_for_event,
}


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


def format_week_range(monday: date) -> str:
    sunday = monday + timedelta(days=6)
    if monday.month == sunday.month:
        return f"{monday.day}\u2013{sunday.day} {MONTHS[monday.month - 1]}"
    return (
        f"{monday.day} {MONTHS[monday.month - 1]} \u2013 "
        f"{sunday.day} {MONTHS[sunday.month - 1]}"
    )


def format_weekend_range(saturday: date) -> str:
    """Return human-friendly weekend range like '12–13 июля'."""
    sunday = saturday + timedelta(days=1)
    if saturday.month == sunday.month:
        return f"{saturday.day}\u2013{sunday.day}\u00A0{MONTHS[saturday.month - 1]}"
    return (
        f"{saturday.day}\u00A0{MONTHS[saturday.month - 1]} \u2013 "
        f"{sunday.day}\u00A0{MONTHS[sunday.month - 1]}"
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


@lru_cache(maxsize=8)
def md_to_html(text: str) -> str:
    html_text = simple_md_to_html(text)
    html_text = re.sub(r"&lt;/?tg-emoji.*?&gt;", "", html_text)
    if not re.match(r"^<(?:h\d|p|ul|ol|blockquote|pre|table)", html_text):
        html_text = f"<p>{html_text}</p>"
    # Telegraph API does not allow h1/h2 or Telegram-specific emoji tags
    html_text = re.sub(r"<(\/?)h[12]>", r"<\1h3>", html_text)
    html_text = re.sub(r"</?tg-emoji[^>]*>", "", html_text)
    return html_text

_DISALLOWED_TAGS_RE = re.compile(r"</?(?:span|div|style|script)[^>]*>", re.IGNORECASE)


def lint_telegraph_html(html: str) -> str:
    """Strip tags that Telegraph does not allow."""
    return _DISALLOWED_TAGS_RE.sub("", html)


def unescape_html_comments(html: str) -> str:
    """Convert escaped HTML comments back to real comments."""
    return html.replace("&lt;!--", "<!--").replace("--&gt;", "-->")


async def check_month_page_markers(tg, path: str) -> None:
    """Fetch a month page and warn if DAY markers are missing."""
    try:
        page = await telegraph_call(tg.get_page, path, return_html=True)
    except Exception as e:
        logging.error("check_month_page_markers failed: %s", e)
        return
    html = page.get("content") or page.get("content_html") or ""
    html = unescape_html_comments(html)
    if "<!--DAY" in html:
        logging.info("month_rebuild_markers_present")
    else:
        logging.warning("month_rebuild_markers_missing")


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

    show_ticket_link = not vk_link
    if e.is_free:
        lines.append("🟡 Бесплатно")
        if e.ticket_link:
            lines.append("по регистрации")
            if show_ticket_link:
                lines.append(f"\U0001f39f {e.ticket_link}")
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max} руб."
        else:
            val = e.ticket_price_min if e.ticket_price_min is not None else e.ticket_price_max
            price = f"{val} руб." if val is not None else ""
        if show_ticket_link:
            lines.append(f"Билеты в источнике {price}".strip())
            lines.append(f"\U0001f39f {e.ticket_link}")
        else:
            lines.append(f"Билеты {price}".strip())
    elif e.ticket_link:
        lines.append("по регистрации")
        if show_ticket_link:
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
            fest = getattr(e, "_festival", None)
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
    nodes.extend(telegraph_br())
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
    nodes.extend(telegraph_br())
    return nodes


def add_day_sections(
    days: Iterable[date],
    by_day: dict[date, list[Event]],
    fest_map: dict[str, Festival],
    add_many: Callable[[Iterable[dict]], None],
    *,
    use_markers: bool = False,
):
    """Append event sections grouped by day to Telegraph content."""
    for d in days:
        events = by_day.get(d)
        if not events:
            continue
        if use_markers:
            add_many([DAY_START(d)])
        add_many(telegraph_br())
        if d.weekday() == 5:
            add_many([{ "tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"] }])
            add_many(telegraph_br())
        elif d.weekday() == 6:
            add_many([{ "tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"] }])
            add_many(telegraph_br())
        add_many([{ "tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"] }])
        add_many(telegraph_br())
        for ev in events:
            fest = fest_map.get((ev.festival or "").casefold())
            add_many(event_to_nodes(ev, fest, fest_icon=True))
        if use_markers:
            add_many([DAY_END(d)])


def render_month_day_section(d: date, events: list[Event]) -> str:
    """Return HTML snippet for a single day on a month page."""
    from telegraph.utils import nodes_to_html

    nodes: list[dict] = []
    nodes.extend(telegraph_br())
    if d.weekday() == 5:
        nodes.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
        nodes.extend(telegraph_br())
    elif d.weekday() == 6:
        nodes.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
        nodes.extend(telegraph_br())
    nodes.append({"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]})
    nodes.extend(telegraph_br())
    for ev in events:
        fest = getattr(ev, "_festival", None)
        nodes.extend(event_to_nodes(ev, fest, fest_icon=True))
    return nodes_to_html(nodes)

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
            e for e in exhibitions if e.end_date and e.end_date >= today_str
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
    size_limit: int | None = None,
) -> tuple[str, list, int]:
    if events is None or exhibitions is None or nav_pages is None:
        events, exhibitions, nav_pages = await get_month_data(db, month)

    async with span("db"):
        async with db.get_session() as session:
            res_f = await session.execute(select(Festival))
            fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

    async with span("render"):
        title, content, size = await asyncio.to_thread(
            _build_month_page_content_sync,
            month,
            events,
            exhibitions,
            nav_pages,
            fest_map,
            continuation_url,
            size_limit,
        )
    logging.info("build_month_page_content size=%d", size)
    return title, content, size


def _build_month_page_content_sync(
    month: str,
    events: list[Event],
    exhibitions: list[Event],
    nav_pages: list[MonthPage],
    fest_map: dict[str, Festival],
    continuation_url: str | None,
    size_limit: int | None,
) -> tuple[str, list, int]:
    today = datetime.now(LOCAL_TZ).date()
    today_str = today.isoformat()
    cutoff = (today - timedelta(days=30)).isoformat()

    if month == today.strftime("%Y-%m"):
        events = [e for e in events if e.date.split("..", 1)[0] >= today_str]
        exhibitions = [e for e in exhibitions if e.end_date and e.end_date >= today_str]
    events = [
        e for e in events if not (e.event_type == "выставка" and e.date < today_str)
    ]
    exhibitions = [
        e for e in exhibitions if e.end_date and e.date <= today_str and e.end_date >= today_str
    ]

    by_day: dict[date, list[Event]] = {}
    for e in events:
        date_part = e.date.split("..", 1)[0]
        d = parse_iso_date(date_part)
        if not d:
            logging.error("Invalid date for event %s: %s", e.id, e.date)
            continue
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    size = 0
    exceeded = False

    def add(node: dict):
        nonlocal size, exceeded
        size += rough_size((node,))
        if size_limit is not None and size > size_limit:
            exceeded = True
            return
        content.append(node)

    def add_many(nodes: Iterable[dict]):
        for n in nodes:
            if exceeded:
                break
            add(n)
    intro = (
        f"Планируйте свой месяц заранее: интересные мероприятия Калининграда и 39 региона в {month_name_prepositional(month)} — от лекций и концертов до культурных шоу. "
    )
    intro_nodes = [
        intro,
        {
            "tag": "a",
            "attrs": {"href": "https://t.me/kenigevents"},
            "children": ["Полюбить Калининград Анонсы"],
        },
    ]
    add({"tag": "p", "children": intro_nodes})

    add_day_sections(sorted(by_day), by_day, fest_map, add_many, use_markers=True)

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
        month_nav = [{"tag": "h4", "children": nav_children}]

    if exhibitions and not exceeded:
        add_many([PERM_START])
        add({"tag": "h3", "children": ["Постоянные выставки"]})
        add_many(telegraph_br())
        for ev in exhibitions:
            if exceeded:
                break
            add_many(exhibition_to_nodes(ev))
        add_many([PERM_END])

    if month_nav and not exceeded:
        add_many(telegraph_br())
        add_many(month_nav)
        add_many(telegraph_br())

    if continuation_url and not exceeded:
        add_many(telegraph_br())
        add(
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

    title = (
        f"События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
    )
    return title, content, size


async def _sync_month_page_inner(db: Database, month: str, update_links: bool = False):
    async with HEAVY_SEMAPHORE:
        now = _time.time()
        if "PYTEST_CURRENT_TEST" not in os.environ and now < _month_next_run[month]:
            logging.debug("sync_month_page skipped, debounced")
            return
        _month_next_run[month] = now + 60
        logging.info(
            "sync_month_page start: month=%s update_links=%s", month, update_links
        )
        if DEBUG:
            mem_info("month page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return
        tg = Telegraph(access_token=token, domain="telegra.ph")
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
            created = False
            if not page:
                page = MonthPage(month=month, url="", path="")
                session.add(page)
                await session.commit()
                created = True

            events, exhibitions, nav_pages = await get_month_data(db, month)

            async def commit_page() -> None:
                async with db.get_session() as s:
                    db_page = await s.get(MonthPage, month)
                    db_page.url = page.url
                    db_page.path = page.path
                    db_page.url2 = page.url2
                    db_page.path2 = page.path2
                    db_page.content_hash = page.content_hash
                    db_page.content_hash2 = page.content_hash2
                    await s.commit()
            from telegraph.utils import nodes_to_html

            def split_events(total_size: int) -> tuple[list[Event], list[Event]]:
                avg = total_size / len(events) if events else total_size
                split_idx = max(1, int(TELEGRAPH_LIMIT // avg)) if events else 0
                return events[:split_idx], events[split_idx:]

            async def update_split(first: list[Event], second: list[Event]) -> None:
                nonlocal created
                title2, content2, _ = await build_month_page_content(
                    db, month, second, exhibitions, nav_pages
                )
                html2 = unescape_html_comments(nodes_to_html(content2))
                hash2 = content_hash(html2)
                if page.path2 and page.content_hash2 == hash2:
                    logging.debug("telegraph_update skipped (no changes)")
                else:
                    rough2 = rough_size(content2)
                    if not page.path2:
                        logging.info("creating second page for %s", month)
                        data2 = await telegraph_create_page(tg, title2, html_content=html2)
                        page.url2 = normalize_telegraph_url(data2.get("url"))
                        page.path2 = data2.get("path")
                    else:
                        logging.info("updating second page for %s", month)
                        start = _time.perf_counter()
                        await telegraph_call(
                            tg.edit_page, page.path2, title=title2, html_content=html2
                        )
                        dur = (_time.perf_counter() - start) * 1000
                        logging.info("editPage %s done in %.0f ms", page.path2, dur)
                    logging.debug(
                        "telegraph_update page=%s nodes=%d bytes≈%d",
                        page.path2,
                        len(content2),
                        rough2,
                    )
                    page.content_hash2 = hash2
                    await asyncio.sleep(0)

                title1, content1, _ = await build_month_page_content(
                    db,
                    month,
                    first,
                    [],
                    nav_pages,
                    continuation_url=page.url2,
                )
                html1 = unescape_html_comments(nodes_to_html(content1))
                hash1 = content_hash(html1)
                if page.path and page.content_hash == hash1:
                    logging.debug("telegraph_update skipped (no changes)")
                else:
                    rough1 = rough_size(content1)
                    if not page.path:
                        logging.info("creating first page for %s", month)
                        data1 = await telegraph_create_page(tg, title1, html_content=html1)
                        page.url = normalize_telegraph_url(data1.get("url"))
                        page.path = data1.get("path")
                        created = True
                    else:
                        logging.info("updating first page for %s", month)
                        start = _time.perf_counter()
                        await telegraph_call(
                            tg.edit_page, page.path, title=title1, html_content=html1
                        )
                        dur = (_time.perf_counter() - start) * 1000
                        logging.info("editPage %s done in %.0f ms", page.path, dur)
                    logging.debug(
                        "telegraph_update page=%s nodes=%d bytes≈%d",
                        page.path,
                        len(content1),
                        rough1,
                    )
                    page.content_hash = hash1
                    await asyncio.sleep(0)

                logging.info(
                    "%s month page %s split into two",
                    "Created" if created else "Edited",
                    month,
                )
                await commit_page()

        title, content, _ = await build_month_page_content(
            db, month, events, exhibitions, nav_pages
        )
        html_full = unescape_html_comments(nodes_to_html(content))
        hash_full = content_hash(html_full)
        size = len(html_full.encode())

        try:
            if size <= TELEGRAPH_LIMIT:
                if page.path and page.content_hash == hash_full:
                    logging.debug("telegraph_update skipped (no changes)")
                else:
                    if not page.path:
                        logging.info("creating month page %s", month)
                        data = await telegraph_create_page(
                            tg, title, html_content=html_full
                        )
                        page.url = normalize_telegraph_url(data.get("url"))
                        page.path = data.get("path")
                        created = True
                    else:
                        logging.info("updating month page %s", month)
                        start = _time.perf_counter()
                        await telegraph_call(
                            tg.edit_page, page.path, title=title, html_content=html_full
                        )
                        dur = (_time.perf_counter() - start) * 1000
                        logging.info("editPage %s done in %.0f ms", page.path, dur)
                    rough = rough_size(content)
                    logging.debug(
                        "telegraph_update page=%s nodes=%d bytes≈%d",
                        page.path,
                        len(content),
                        rough,
                    )
                    page.content_hash = hash_full
                    page.content_hash2 = None
                page.url2 = None
                page.path2 = None
                logging.info(
                    "%s month page %s", "Created" if created else "Edited", month
                )
                await commit_page()
            else:
                first, second = split_events(size)
                logging.info(
                    "sync_month_page: splitting %s (events=%d)",
                    month,
                    len(events),
                )
                await update_split(first, second)
        except TelegraphException as e:
            if "CONTENT_TOO_BIG" in str(e):
                first, second = split_events(size)
                logging.warning("Month page %s too big, splitting", month)
                await update_split(first, second)
            else:
                logging.error("Failed to sync month page %s: %s", month, e)
                raise
        except Exception as e:
            logging.error("Failed to sync month page %s: %s", month, e)

            if update_links or created:
                async with db.get_session() as session:
                    result = await session.execute(
                        select(MonthPage).order_by(MonthPage.month)
                    )
                    months = result.scalars().all()
                for p in months:
                    if p.month != month:
                        await sync_month_page(db, p.month, update_links=False)
                        await asyncio.sleep(0)
            raise
        if page.path:
            await check_month_page_markers(tg, page.path)
        if page.path2:
            await check_month_page_markers(tg, page.path2)
        if DEBUG:
            mem_info("month page after")


async def sync_month_page(db: Database, month: str, update_links: bool = False):
    async with _page_locks[f"month:{month}"]:
        await _sync_month_page_inner(db, month, update_links)


def week_start_for_date(d: date) -> date:
    return d - timedelta(days=d.weekday())


def next_week_start(d: date) -> date:
    w = week_start_for_date(d)
    if d <= w:
        return w
    return w + timedelta(days=7)


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


async def build_weekend_page_content(
    db: Database, start: str, size_limit: int | None = None
) -> tuple[str, list, int]:
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
        fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

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
        fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}

    by_day: dict[date, list[Event]] = {}
    for e in events:
        d = parse_iso_date(e.date)
        if not d:
            continue
        by_day.setdefault(d, []).append(e)

    content: list[dict] = []
    size = 0
    exceeded = False

    def add(node: dict):
        nonlocal size, exceeded
        size += rough_size((node,))
        if size_limit is not None and size > size_limit:
            exceeded = True
            return
        content.append(node)

    def add_many(nodes: Iterable[dict]):
        for n in nodes:
            if exceeded:
                break
            add(n)

    add(
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

    add_day_sections(days, by_day, fest_map, add_many)

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
        weekend_nav = [{"tag": "h4", "children": nav_children}]

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
        month_nav = [{"tag": "h4", "children": nav_children}]

    if exhibitions and not exceeded:
        add({"tag": "h3", "children": ["Постоянные выставки"]})
        add_many(telegraph_br())
        for ev in exhibitions:
            if exceeded:
                break
            add_many(exhibition_to_nodes(ev))

    if weekend_nav and not exceeded:
        add_many(telegraph_br())
        add_many(weekend_nav)
        add_many(telegraph_br())
    if month_nav and not exceeded:
        add_many(telegraph_br())
        add_many(month_nav)
        add_many(telegraph_br())

    title = (
        "Чем заняться на выходных в Калининградской области "
        f"{format_weekend_range(saturday)}"
    )
    if DEBUG:
        from telegraph.utils import nodes_to_html
        html = nodes_to_html(content)
        logging.debug(
            "weekend_html sizes: html=%d json=%d",
            len(html),
            len(json.dumps(content, ensure_ascii=False)),
        )
    return title, content, size


async def _sync_weekend_page_inner(
    db: Database, start: str, update_links: bool = True, post_vk: bool = True
):
    async with HEAVY_SEMAPHORE:
        if DEBUG:
            mem_info("weekend page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return
        tg = Telegraph(access_token=token, domain="telegra.ph")
        from telegraph.utils import nodes_to_html

        async with db.get_session() as session:
            page = await session.get(WeekendPage, start)
            if not page:
                page = WeekendPage(start=start, url="", path="")
                session.add(page)
                await session.commit()
                created = True
            else:
                created = False
            path = page.path

        try:
            title, content, _ = await build_weekend_page_content(db, start)
            html = nodes_to_html(content)
            hash_new = content_hash(html)
            if not path:
                data = await telegraph_create_page(tg, title, content=content)
                page.url = normalize_telegraph_url(data.get("url"))
                page.path = data.get("path")
                created = True
                rough = rough_size(content)
                logging.debug(
                    "telegraph_update page=%s nodes=%d bytes≈%d",
                    page.path,
                    len(content),
                    rough,
                )
                page.content_hash = hash_new
            elif page.content_hash == hash_new and not update_links:
                logging.debug("telegraph_update skipped (no changes)")
            else:
                start_t = _time.perf_counter()
                await telegraph_call(tg.edit_page, path, title=title, content=content)
                dur = (_time.perf_counter() - start_t) * 1000
                logging.info("editPage %s done in %.0f ms", path, dur)
                rough = rough_size(content)
                logging.debug(
                    "telegraph_update page=%s nodes=%d bytes≈%d",
                    path,
                    len(content),
                    rough,
                )
                page.content_hash = hash_new
            logging.info("%s weekend page %s", "Created" if created else "Edited", start)
        except Exception as e:
            logging.error("Failed to sync weekend page %s: %s", start, e)
            return

        async with db.get_session() as session:
            db_page = await session.get(WeekendPage, start)
            if db_page:
                db_page.url = page.url
                db_page.path = page.path
                db_page.content_hash = page.content_hash
                await session.commit()

        if post_vk:
            await sync_vk_weekend_post(db, start)
        if DEBUG:
            mem_info("weekend page after")


async def sync_weekend_page(
    db: Database, start: str, update_links: bool = True, post_vk: bool = True
):
    async with _page_locks[f"week:{start}"]:
        await _sync_weekend_page_inner(db, start, update_links, post_vk)


def _build_month_vk_nav_lines(week_pages: list[WeekPage], cur_month: str) -> list[str]:
    first_by_month: dict[str, WeekPage] = {}
    for w in week_pages:
        m = w.start[:7]
        if m not in first_by_month or w.start < first_by_month[m].start:
            first_by_month[m] = w
    parts: list[str] = []
    for m in sorted(first_by_month):
        if m < cur_month:
            continue
        w = first_by_month[m]
        label = month_name_nominative(m)
        if m == cur_month or not w.vk_post_url:
            parts.append(label)
        else:
            parts.append(f"[{w.vk_post_url}|{label}]")
    return parts


async def build_week_vk_message(db: Database, start: str) -> str:
    logging.info("build_week_vk_message start for %s", start)
    monday = date.fromisoformat(start)
    days = [monday + timedelta(days=i) for i in range(7)]
    async with span("db"):
        async with db.get_session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.date.in_([d.isoformat() for d in days]))
                .order_by(Event.date, Event.time)
            )
            events = result.scalars().all()
            res_w = await session.execute(select(WeekPage).order_by(WeekPage.start))
            week_pages = res_w.scalars().all()

    async with span("render"):
        by_day: dict[date, list[Event]] = {}
        for e in events:
            if not e.source_vk_post_url:
                continue
            d = parse_iso_date(e.date)
            if not d:
                continue
            by_day.setdefault(d, []).append(e)

        lines = [f"{format_week_range(monday)} Афиша недели"]
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

        nav_weeks = [
            w
            for w in week_pages
            if w.start[:7] == start[:7] and (w.vk_post_url or w.start == start)
        ]
        if nav_weeks:
            parts = []
            for w in nav_weeks:
                label = format_week_range(date.fromisoformat(w.start))
                if w.start == start or not w.vk_post_url:
                    parts.append(label)
                else:
                    parts.append(f"[{w.vk_post_url}|{label}]")
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(parts))

        month_parts = _build_month_vk_nav_lines(week_pages, start[:7])
        if month_parts:
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(month_parts))

        message = "\n".join(lines)
    logging.info("build_week_vk_message built %d lines", len(lines))
    return message


async def build_weekend_vk_message(db: Database, start: str) -> str:
    logging.info("build_weekend_vk_message start for %s", start)
    saturday = date.fromisoformat(start)
    sunday = saturday + timedelta(days=1)
    days = [saturday, sunday]
    async with span("db"):
        async with db.get_session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.date.in_([d.isoformat() for d in days]))
                .order_by(Event.date, Event.time)
            )
            events = result.scalars().all()
            res_w = await session.execute(select(WeekendPage).order_by(WeekendPage.start))
            weekend_pages = res_w.scalars().all()
            res_week = await session.execute(select(WeekPage).order_by(WeekPage.start))
            week_pages = res_week.scalars().all()

    async with span("render"):
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

        nav_pages = [
            w
            for w in weekend_pages
            if w.start >= start and (w.vk_post_url or w.start == start)
        ]
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

        month_parts = _build_month_vk_nav_lines(week_pages, start[:7])
        if month_parts:
            lines.append(VK_BLANK_LINE)
            lines.append(VK_BLANK_LINE)
            lines.append(" ".join(month_parts))

        message = "\n".join(lines)
    logging.info(
        "build_weekend_vk_message built %d lines", len(lines)
    )
    return message


async def sync_vk_weekend_post(db: Database, start: str, bot: Bot | None = None) -> None:
    lock = _weekend_vk_lock(start)
    async with lock:
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


async def sync_vk_week_post(db: Database, start: str, bot: Bot | None = None) -> None:
    lock = _week_vk_lock(start)
    async with lock:
        logging.info("sync_vk_week_post start for %s", start)
        group_id = VK_AFISHA_GROUP_ID
        if not group_id:
            logging.info("sync_vk_week_post: VK group not configured")
            return
        async with db.get_session() as session:
            page = await session.get(WeekPage, start)

        message = await build_week_vk_message(db, start)
        logging.info("sync_vk_week_post message len=%d", len(message))
        needs_new_post = not page or not page.vk_post_url
        if page and page.vk_post_url:
            try:
                updated = await edit_vk_post(page.vk_post_url, message, db, bot)
                if updated:
                    logging.info("sync_vk_week_post updated %s", page.vk_post_url)
                else:
                    logging.info(
                        "sync_vk_week_post: no changes for %s", page.vk_post_url
                    )
            except Exception as e:
                if "post or comment deleted" in str(e) or "Пост удалён" in str(e):
                    logging.warning(
                        "sync_vk_week_post: original VK post missing, creating new"
                    )
                    needs_new_post = True
                else:
                    logging.error("VK post error for week %s: %s", start, e)
                    return
        if needs_new_post:
            url = await post_to_vk(group_id, message, db, bot)
            if url:
                async with db.get_session() as session:
                    obj = await session.get(WeekPage, start)
                    if obj:
                        obj.vk_post_url = url
                    else:
                        session.add(WeekPage(start=start, vk_post_url=url))
                    await session.commit()
                logging.info("sync_vk_week_post created %s", url)


async def generate_festival_description(fest: Festival, events: list[Event]) -> str:
    """Use LLM to craft a short festival blurb."""
    texts: list[str] = []
    if fest.source_text:
        texts.append(fest.source_text)
    elif fest.description:
        texts.append(fest.description)
    texts.extend(e.source_text for e in events[:5])
    if not texts:
        return ""
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
            await session.execute(
                update(Festival)
                .where(Festival.id == fest.id)
                .values(description=desc)
            )
            await session.commit()

    nodes: list[dict] = []
    if fest.photo_url:
        nodes.append({"tag": "img", "attrs": {"src": fest.photo_url}})
        nodes.append({"tag": "p", "children": ["\u00a0"]})
    start, end = festival_dates(fest, events)
    if start:
        date_text = format_day_pretty(start)
        if end and end != start:
            date_text += f" - {format_day_pretty(end)}"
        nodes.append({"tag": "p", "children": [f"\U0001f4c5 {date_text}"]})
    loc_text = festival_location(fest, events)
    if loc_text:
        nodes.append({"tag": "p", "children": [f"\U0001f4cd {loc_text}"]})
    if fest.ticket_url:
        nodes.append({"tag": "p", "children": [f"\U0001f39f {fest.ticket_url}"]})
    if fest.description:
        nodes.append({"tag": "p", "children": [fest.description]})

    if fest.website_url or fest.vk_url or fest.tg_url:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
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
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "h3", "children": ["Мероприятия фестиваля"]})
        for e in events:
            nodes.extend(event_to_nodes(e))
    nav_nodes, _ = await _build_festival_nav_block(db, exclude=fest.name)
    if nav_nodes:
        from telegraph.utils import nodes_to_html, html_to_nodes
        nav_html = nodes_to_html(nav_nodes)
        nav_block = f"{FEST_NAV_START}{nav_html}{FEST_NAV_END}"
        nodes.extend(html_to_nodes(nav_block))
    title = fest.full_name or fest.name
    return title, nodes


async def sync_festival_page(
    db: Database,
    name: str,
    *,
    refresh_nav_only: bool = False,
    items: list[tuple[date | None, date | None, Festival]] | None = None,
):
    async with HEAVY_SEMAPHORE:
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return
        tg = Telegraph(access_token=token, domain="telegra.ph")
        async with db.get_session() as session:
            result = await session.execute(
                select(Festival).where(Festival.name == name)
            )
            fest = result.scalar_one_or_none()
            if not fest:
                return
            title = fest.full_name or fest.name
            path = fest.telegraph_path
            url = fest.telegraph_url

        try:
            created = False
            if refresh_nav_only and path:
                nav_html, _, _ = await build_festivals_nav_block(db)
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                new_html, strategy = apply_festival_nav(html_content, nav_html)
                if new_html != html_content:
                    await telegraph_call(
                        tg.edit_page, path, title=title, html_content=new_html
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"marker_strategy": strategy, "action": "edited", "target": "tg", "path": path},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"marker_strategy": strategy, "action": "skipped_nochange", "target": "tg", "path": path},
                    )
            else:
                title, content = await build_festival_page_content(db, fest)
                path = fest.telegraph_path
                url = fest.telegraph_url
                if path:
                    await telegraph_call(tg.edit_page, path, title=title, content=content)
                    logging.info("updated festival page %s in Telegraph", name)
                else:
                    data = await telegraph_create_page(tg, title, content=content)
                    url = normalize_telegraph_url(data.get("url"))
                    path = data.get("path")
                    created = True
                    logging.info("created festival page %s: %s", name, url)
        except Exception as e:
            logging.error("Failed to sync festival %s: %s", name, e)
            return

        async with db.get_session() as session:
            result = await session.execute(
                select(Festival).where(Festival.name == name)
            )
            fest_db = result.scalar_one_or_none()
            if fest_db:
                fest_db.telegraph_url = url
                fest_db.telegraph_path = path
                await session.commit()
                logging.info("synced festival page %s", name)


async def refresh_nav_on_all_festivals(
    db: Database,
    bot: Bot | None = None,
    *,
    nav_html: str | None = None,
    nav_lines: list[str] | None = None,
) -> None:
    """Refresh navigation on all festival pages and VK posts."""
    async with db.get_session() as session:
        res = await session.execute(
            select(
                Festival.id,
                Festival.name,
                Festival.telegraph_path,
                Festival.vk_post_url,
            )
        )
        fests = res.all()

    if nav_html is None or nav_lines is None:
        nav_html, nav_lines, changed = await build_festivals_nav_block(db)
        if not changed:
            return
    token = get_telegraph_token()
    tg = Telegraph(access_token=token, domain="telegra.ph") if token else None
    for fid, name, path, vk_url in fests:
        if tg and path:
            try:
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                title = page.get("title") or name
                new_html, strategy = apply_festival_nav(html_content, nav_html)
                if new_html != html_content:
                    await telegraph_call(
                        tg.edit_page, path, title=title, html_content=new_html
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"marker_strategy": strategy, "action": "edited", "target": "tg", "path": path},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"marker_strategy": strategy, "action": "skipped_nochange", "target": "tg", "path": path},
                    )
            except Exception as e:
                logging.error("Failed to update festival page %s: %s", name, e)
        if vk_url:
            await sync_festival_vk_post(
                db, name, bot, nav_only=True, nav_lines=nav_lines
            )


async def build_festival_vk_message(db: Database, fest: Festival) -> str:
    async with db.get_session() as session:
        res = await session.execute(
            select(Event).where(Event.festival == fest.name).order_by(Event.date, Event.time)
        )
        events = res.scalars().all()
        if not fest.description:
            desc = await generate_festival_description(fest, events)
            if desc:
                fest.description = desc
                await session.execute(
                    update(Festival)
                    .where(Festival.id == fest.id)
                    .values(description=desc)
                )
                await session.commit()
    lines = [fest.full_name or fest.name]
    start, end = festival_dates(fest, events)

    if start:
        date_text = format_day_pretty(start)
        if end and end != start:
            date_text += f" - {format_day_pretty(end)}"
        lines.append(f"\U0001f4c5 {date_text}")
    loc_text = festival_location(fest, events)
    if loc_text:
        lines.append(f"\U0001f4cd {loc_text}")
    if fest.ticket_url:
        lines.append(f"\U0001f39f {fest.ticket_url}")
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
    _, nav_lines = await _build_festival_nav_block(db, exclude=fest.name)
    if nav_lines:
        lines.extend(nav_lines)
    return "\n".join(lines)


async def sync_festival_vk_post(
    db: Database,
    name: str,
    bot: Bot | None = None,
    *,
    nav_only: bool = False,
    nav_lines: list[str] | None = None,
):
    group_id = await get_vk_group_id(db)
    if not group_id:
        return
    async with db.get_session() as session:
        res = await session.execute(select(Festival).where(Festival.name == name))
        fest = res.scalar_one_or_none()
        if not fest:
            return

    async def _try_edit(message: str, attachments: list[str] | None) -> bool | None:
        if not fest.vk_post_url:
            return False
        for attempt in range(1, 4):
            try:
                await edit_vk_post(fest.vk_post_url, message, db, bot, attachments)
                return True
            except VKAPIError as e:
                logging.warning(
                    "Ошибка VK при редактировании (попытка %d из 3, код %s): %s",
                    attempt,
                    e.code,
                    e.message,
                )
                if e.code in {213, 214} or "edit time expired" in e.message.lower():
                    return False
                if attempt == 3:
                    return None
                await asyncio.sleep(2 ** (attempt - 1))
        return None

    async def _try_post(message: str, attachments: list[str] | None) -> str | None:
        for attempt in range(1, 4):
            try:
                url = await post_to_vk(group_id, message, db, bot, attachments)
                if url:
                    return url
            except VKAPIError as e:
                logging.warning(
                    "Ошибка VK при публикации (попытка %d из 3, код %s): %s",
                    attempt,
                    e.code,
                    e.message,
                )
            if attempt == 3:
                return None
            await asyncio.sleep(2 ** (attempt - 1))
        return None

    can_edit = True
    if nav_only and fest.vk_post_url:
        nav_lines_local = nav_lines
        if nav_lines_local is None:
            _, nav_lines_local = await _build_festival_nav_block(db, exclude=fest.name)
        if nav_lines_local:
            ids = _vk_owner_and_post_id(fest.vk_post_url)
            if not ids:
                logging.error("invalid VK post url %s", fest.vk_post_url)
                return
            owner_id, post_id = ids
            user_token = _vk_user_token()
            if not user_token:
                logging.error("VK_USER_TOKEN missing")
                return
            try:
                data = await _vk_api(
                    "wall.getById",
                    {"posts": f"{owner_id}_{post_id}"},
                    db,
                    bot,
                    token=user_token,
                    token_kind="user",
                )
                text = data.get("response", [{}])[0].get("text", "")
            except VKAPIError as e:
                logging.error(
                    "Не удалось получить пост VK для %s: код %s %s",
                    name,
                    e.code,
                    e.message,
                )
                return
            lines = text.split("\n")
            idx = None
            for i, line in enumerate(lines):
                if line == "Ближайшие фестивали":
                    idx = i
                    if i > 0 and lines[i - 1] == VK_BLANK_LINE:
                        idx -= 1
                    break
            base = lines[:idx] if idx is not None else lines
            message = "\n".join(base + nav_lines_local)
            res_edit = await _try_edit(message, None)
            if res_edit is True:
                logging.info("updated festival post %s on VK", name)
                return
            if res_edit is None:
                logging.error("VK post error for festival %s", name)
                return
            can_edit = False  # editing not possible, create new post

    message = await build_festival_vk_message(db, fest)
    attachments: list[str] | None = None
    if fest.photo_url:
        if VK_PHOTOS_ENABLED:
            photo_id = await upload_vk_photo(group_id, fest.photo_url, db, bot)
            if photo_id:
                attachments = [photo_id]
        else:
            logging.info("VK photo posting disabled")

    if fest.vk_post_url and can_edit:
        res_edit = await _try_edit(message, attachments)
        if res_edit is True:
            logging.info("updated festival post %s on VK", name)
            return
        if res_edit is None:
            logging.error("VK post error for festival %s", name)
            return

    url = await _try_post(message, attachments)
    if url:
        async with db.get_session() as session:
            fest_db = (
                await session.execute(select(Festival).where(Festival.name == name))
            ).scalar_one()
            fest_db.vk_post_url = url
            await session.commit()
        logging.info("created festival post %s: %s", name, url)
    else:
        logging.error("VK post error for festival %s", name)


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
    from models import Event, WeekendPage, MonthPage, Festival

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
                festival=fest_map.get((e.festival or "").casefold()),
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
                festival=fest_map.get((e.festival or "").casefold()),
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
        res_week_all = await session.execute(select(WeekPage))
        week_pages = res_week_all.scalars().all()
        cur_month = today.strftime("%Y-%m")

        def closest_week_page(month: str, ref: date) -> WeekPage | None:
            candidates = [w for w in week_pages if w.start[:7] == month and w.vk_post_url]
            if not candidates:
                return None
            return min(candidates, key=lambda w: abs(date.fromisoformat(w.start) - ref))

        week_cur = closest_week_page(cur_month, today)
        next_month_str = next_month(cur_month)
        week_next = closest_week_page(next_month_str, date.fromisoformat(f"{next_month_str}-01"))

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
                w_url = w.vk_post_url
        lines1.append(
            format_event_vk(
                e,
                highlight=True,
                weekend_url=w_url,
                festival=fest_map.get((e.festival or "").casefold()),
            )
        )
        lines1.append(VK_EVENT_SEPARATOR)
    if events_today:
        lines1.pop()
    link_lines: list[str] = []
    if wpage and wpage.vk_post_url:
        label = f"выходные {format_weekend_range(w_start)}"
        prefix = f"(+{weekend_count}) " if weekend_count else ""
        link_lines.append(f"{prefix}[{wpage.vk_post_url}|{label}]")
    if week_cur:
        label = month_name_nominative(cur_month)
        prefix = f"(+{cur_count}) " if cur_count else ""
        link_lines.append(f"{prefix}[{week_cur.vk_post_url}|{label}]")
    if week_next:
        label = month_name_nominative(next_month_str)
        prefix = f"(+{next_count}) " if next_count else ""
        link_lines.append(f"{prefix}[{week_next.vk_post_url}|{label}]")
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
                w_url = w.vk_post_url
        lines2.append(
            format_event_vk(
                e,
                weekend_url=w_url,
                festival=fest_map.get((e.festival or "").casefold()),
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
    token_kind: str = "group",
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
    if DEBUG:
        mem_info("VK post before")
    data = await _vk_api("wall.post", params, db, bot, token=token, token_kind=token_kind)
    if DEBUG:
        mem_info("VK post after")
    post_id = data.get("response", {}).get("post_id")
    if post_id:
        url = f"https://vk.com/wall-{group_id.lstrip('-')}_{post_id}"
        logging.info(
            "post_to_vk ok group=%s post_id=%s len=%d attachments=%d",
            group_id,
            post_id,
            len(message),
            len(attachments or []),
        )
        return url
    err_code = data.get("error", {}).get("error_code") if isinstance(data, dict) else None
    logging.error(
        "post_to_vk fail group=%s code=%s len=%d attachments=%d",
        group_id,
        err_code,
        len(message),
        len(attachments or []),
    )
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


def build_vk_source_header(event: Event, festival: Festival | None = None) -> list[str]:
    """Build header lines for VK source post with general event info."""

    lines: list[str] = [event.title]

    if festival:
        link = festival.vk_url or festival.vk_post_url
        prefix = "✨ "
        if link:
            lines.append(f"{prefix}[{link}|{festival.name}]")
        else:
            lines.append(f"{prefix}{festival.name}")

    lines.append(VK_BLANK_LINE)

    date_part = event.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        logging.error("Invalid event date: %s", event.date)
        day = event.date
    lines.append(f"\U0001f4c5 {day} {event.time}")

    loc = event.location_name
    addr = event.location_address
    if addr and event.city:
        addr = strip_city_from_address(addr, event.city)
    if addr:
        loc += f", {addr}"
    if event.city:
        loc += f", #{event.city}"
    lines.append(f"\U0001f4cd {loc}")

    if event.pushkin_card:
        lines.append("\u2705 Пушкинская карта")

    if event.is_free:
        lines.append("🟡 Бесплатно")
        if event.ticket_link:
            lines.append(f"\U0001f39f по регистрации {event.ticket_link}")
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
        info = f"Билеты в источнике {price}".strip()
        lines.append(f"\U0001f39f {info} {event.ticket_link}".strip())
    elif event.ticket_link:
        lines.append(f"\U0001f39f по регистрации {event.ticket_link}")
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
            lines.append(f"\U0001f39f Билеты {price}")

    lines.append(VK_BLANK_LINE)
    return lines


def build_vk_source_message(
    event: Event,
    text: str,
    festival: Festival | None = None,
    *,
    ics_url: str | None = None,
) -> str:
    """Build detailed VK post for an event including original source text."""

    text = _vk_expose_links(text)
    lines = build_vk_source_header(event, festival)
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
    append_text: bool = True,
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

        # Extract previous text versions
        existing_main = existing.split(VK_SOURCE_FOOTER)[0].rstrip()
        segments = (
            existing_main.split(f"\n{CONTENT_SEPARATOR}\n") if existing_main else []
        )
        texts: list[str] = []
        for seg in segments:
            lines = seg.splitlines()
            blanks = 0
            i = 0
            while i < len(lines):
                if lines[i] == VK_BLANK_LINE:
                    blanks += 1
                    if blanks == 2:
                        i += 1
                        break
                i += 1
            lines = lines[i:]
            if lines and lines[-1].startswith("Добавить в календарь"):
                lines.pop()
            while lines and lines[-1] == VK_BLANK_LINE:
                lines.pop()
            texts.append("\n".join(lines).strip())

        text_clean = _vk_expose_links(text).strip()
        if texts:
            if append_text:
                texts.append(text_clean)
            else:
                texts[-1] = text_clean
        else:
            texts = [text_clean]

        header_lines = build_vk_source_header(event, festival)
        new_lines = header_lines[:]
        for idx, t in enumerate(texts):
            if t:
                new_lines.extend(t.splitlines())
            new_lines.append(VK_BLANK_LINE)
            if idx < len(texts) - 1:
                new_lines.append(CONTENT_SEPARATOR)
        if ics_url:
            new_lines.append(f"Добавить в календарь {ics_url}")
        new_lines.append(VK_SOURCE_FOOTER)
        new_message = "\n".join(new_lines)
        await edit_vk_post(
            event.source_vk_post_url,
            new_message,
            db,
            bot,
        )
        url = event.source_vk_post_url
        logging.info("sync_vk_source_post updated %s", url)
    else:
        message = build_vk_source_message(
            event, text, festival=festival, ics_url=ics_url
        )
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
    user_token = _vk_user_token()
    if not user_token:
        raise VKAPIError(None, "VK_USER_TOKEN missing")
    try:
        data = await _vk_api(
            "wall.getById",
            {"posts": f"{owner_id}_{post_id}"},
            db,
            bot,
            token=user_token,
            token_kind="user",
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
    await _vk_api("wall.edit", params, db, bot, token=user_token, token_kind="user")
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
    # сборка постов/текста — вне семафоров
    async with span("db"):
        section1, section2 = await build_daily_sections_vk(db, tz, now)
    if section == "today":
        async with span("vk-send"):
            await post_to_vk(group_id, section1, db, bot)
    elif section == "added":
        async with span("vk-send"):
            await post_to_vk(group_id, section2, db, bot)
    else:
        async with span("vk-send"):
            await post_to_vk(group_id, section1, db, bot)
        async with span("vk-send"):
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
    # 1) Собираем контент вне любых семафоров
    posts = await build_daily_posts(db, tz, now)
    if not posts:
        logging.info("daily: no posts for channel=%s; skip last_daily", channel_id)
        return
    # 2) Отправляем с «узким» шлюзом TG, чтобы не блокировать систему целиком
    sent = 0
    for text, markup in posts:
        try:
            async with TG_SEND_SEMAPHORE:
                async with span("tg-send"):
                    await bot.send_message(
                        channel_id,
                        text,
                        reply_markup=markup,
                        parse_mode="HTML",
                        disable_web_page_preview=True,
                    )
            sent += 1
        except Exception as e:
            logging.error("daily send failed for %s: %s", channel_id, e)
            if "message is too long" in str(e):
                continue
            raise
    # 3) Отмечаем только если что-то реально ушло
    if record and now is None and sent > 0:
        async with db.raw_conn() as conn:
            await conn.execute(
                "UPDATE channel SET last_daily=? WHERE channel_id=?",
                ((now or datetime.now(tz)).date().isoformat(), channel_id),
            )
            await conn.commit()


async def daily_scheduler(db: Database, bot: Bot):
    import asyncio, logging, datetime as dt

    log = logging.getLogger(__name__)
    while True:
        log.info("daily_scheduler: start")
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = dt.datetime.now(tz)
        now_time = now.time().replace(second=0, microsecond=0)
        async with db.raw_conn() as conn:
            conn.row_factory = __import__("sqlite3").Row
            rows = await conn.execute_fetchall(
                """
                SELECT channel_id, daily_time, last_daily
                FROM channel
                WHERE daily_time IS NOT NULL
                """
            )
        for r in rows:
            if not r["daily_time"]:
                continue
            try:
                target_time = dt.datetime.strptime(r["daily_time"], "%H:%M").time()
            except ValueError:
                continue
            due = (r["last_daily"] or "") != now.date().isoformat() and now_time >= target_time
            logging.info(
                "daily_scheduler: channel=%s due=%s last_daily=%s now=%s target=%s",
                r["channel_id"], due, r["last_daily"], now_time, target_time
            )
            if due:
                try:
                    # не блокируем цикл планировщика — отправляем в фоне
                    asyncio.create_task(send_daily_announcement(db, bot, r["channel_id"], tz))
                except Exception as e:
                    log.exception(
                        "daily_scheduler: channel %s failed: %s",
                        r["channel_id"],
                        e,
                    )
        log.info("daily_scheduler: done")
        await asyncio.sleep(seconds_to_next_minute(dt.datetime.now(tz)))


async def vk_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        group_id = await get_vk_group_id(db)
        if not group_id:
            return
        now_time = now.time().replace(second=0, microsecond=0)
        today_time = datetime.strptime(await get_vk_time_today(db), "%H:%M").time()
        added_time = datetime.strptime(await get_vk_time_added(db), "%H:%M").time()
        last_today = await get_vk_last_today(db)
        last_added = await get_vk_last_added(db)

    if (last_today or "") != now.date().isoformat() and now_time >= today_time:
        try:
            await send_daily_announcement_vk(db, group_id, tz, section="today", bot=bot)
            async with span("db"):
                await set_vk_last_today(db, now.date().isoformat())
        except Exception as e:
            logging.error("vk daily today failed: %s", e)

    if (last_added or "") != now.date().isoformat() and now_time >= added_time:
        try:
            await send_daily_announcement_vk(db, group_id, tz, section="added", bot=bot)
            async with span("db"):
                await set_vk_last_added(db, now.date().isoformat())
        except Exception as e:
            logging.error("vk daily added failed: %s", e)


async def cleanup_old_events(db: Database, now_utc: datetime | None = None) -> int:
    """Delete events that finished more than a week ago."""
    cutoff = (now_utc or datetime.utcnow()) - timedelta(days=7)
    cutoff_str = cutoff.date().isoformat()
    async with db.get_session() as session:
        async with session.begin():
            start_t = _time.perf_counter()
            res1 = await session.execute(
                delete(Event).where(
                    Event.end_date.is_not(None), Event.end_date < cutoff_str
                )
            )
            res2 = await session.execute(
                delete(Event).where(
                    Event.end_date.is_(None), Event.date < cutoff_str
                )
            )
        dur = (_time.perf_counter() - start_t) * 1000
        logging.debug("db cleanup_old_events took %.1f ms", dur)
    deleted = (res1.rowcount or 0) + (res2.rowcount or 0)
    return deleted


async def cleanup_scheduler(
    db: Database, bot: Bot, run_id: str | None = None
) -> None:
    retries = [0.8, 2.0]
    attempt = 0
    while True:
        try:
            start = _time.perf_counter()
            async with db.ensure_connection():
                deleted = await cleanup_old_events(db)
            db_took_ms = (_time.perf_counter() - start) * 1000
            logging.info(
                "cleanup_ok run_id=%s deleted_count=%s scanned=%s db_took_ms=%.0f commit_ms=0",
                run_id,
                deleted,
                0,
                db_took_ms,
            )
            try:
                await notify_superadmin(
                    db, bot, f"Cleanup finished, deleted={deleted}"
                )
            except Exception as e:
                logging.warning("cleanup notify failed: %s", e)
            break
        except (sqlite3.ProgrammingError, sqlite3.OperationalError) as e:
            msg = str(e)
            if "Connection closed" in msg or "database is locked" in msg:
                if attempt < len(retries):
                    delay = retries[attempt]
                    attempt += 1
                    logging.warning(
                        "cleanup_retry run_id=%s delay=%.1fs error=%s",
                        run_id,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                    continue
            logging.error("Cleanup failed: %s", e)
            break
        except Exception as e:
            logging.error("Cleanup failed: %s", e)
            break


async def nightly_page_sync(db: Database, run_id: str | None = None) -> None:
    """Rebuild all stored month and weekend pages once per night.

    The job is optional and disabled by default. It is intended for manual
    activation via an environment variable when a full resync of pages is
    required.
    """
    async with span("db"):
        async with db.get_session() as session:
            months = (
                await session.exec(select(MonthPage.month))
            ).scalars().all()
            weekends = (
                await session.exec(select(WeekendPage.start))
            ).scalars().all()
    for month in months:
        try:
            await sync_month_page(db, month, update_links=False)
        except Exception as e:  # pragma: no cover - log and continue
            logging.error("nightly_page_sync month %s failed: %s", month, e)
    for start in weekends:
        try:
            await sync_weekend_page(db, start, update_links=False, post_vk=False)
        except Exception as e:  # pragma: no cover - log and continue
            logging.error("nightly_page_sync weekend %s failed: %s", start, e)


async def partner_notification_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    """Remind partners who haven't added events for a week."""
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        last_run = await get_partner_last_run(db)
    if now.time() >= time(9, 0) and (last_run is None or last_run != now.date()):
        try:
            async with span("db-query"):
                async with db.get_session() as session:
                    stream = await session.stream(
                        text(
                            "SELECT id, title FROM event "
                            "WHERE festival IS NOT NULL "
                            "AND date BETWEEN :start AND :end "
                            "ORDER BY date"
                        ),
                        {
                            "start": now.date().isoformat(),
                            "end": (now.date() + timedelta(days=30)).isoformat(),
                        },
                    )
                    async for _ in stream:
                        pass
            await asyncio.sleep(0)
            notified = await notify_inactive_partners(db, bot, tz)
            if notified:
                names = ", ".join(
                    f"@{u.username}" if u.username else str(u.user_id)
                    for u in notified
                )
                async with span("tg-send"):
                    await notify_superadmin(
                        db, bot, f"Partner reminders sent to: {names}"
                    )
            else:
                logging.info("Partner reminders: none")
            await set_partner_last_run(db, now.date())
        except Exception as e:
            logging.error("partner reminder failed: %s", e)
            await notify_superadmin(db, bot, f"Partner reminder failed: {e}")


async def vk_poll_scheduler(db: Database, bot: Bot, run_id: str | None = None):
    if not (VK_TOKEN or os.getenv("VK_USER_TOKEN")):
        return
    async with span("db"):
        offset = await get_tz_offset(db)
        tz = offset_to_timezone(offset)
        now = datetime.now(tz)
        group_id = await get_vk_group_id(db)
        if not group_id:
            return
        ev_map: dict[str, list[Event]] = {}
        async with db.get_session() as session:
            stream = await session.stream_scalars(
                select(Event).execution_options(yield_per=500)
            )
            async for e in stream:
                if e.festival:
                    ev_map.setdefault(e.festival, []).append(e)

            LIMIT = 1000
            off = 0
            while True:
                result = await session.execute(
                    select(
                        Festival.id,
                        Festival.name,
                        Festival.start_date,
                        Festival.end_date,
                        Festival.vk_poll_url,
                        Festival.vk_post_url,
                    )
                    .order_by(Festival.id)
                    .limit(LIMIT)
                    .offset(off)
                )
                rows = result.all()
                if not rows:
                    break
                for (
                    fest_id,
                    name,
                    start_date,
                    end_date,
                    vk_poll_url,
                    vk_post_url,
                ) in rows:
                    if vk_poll_url:
                        continue
                    fest = Festival(
                        id=fest_id,
                        name=name,
                        start_date=start_date,
                        end_date=end_date,
                        vk_poll_url=vk_poll_url,
                        vk_post_url=vk_post_url,
                    )
                    evs = ev_map.get(name, [])
                    start_dt, end_dt = festival_dates(fest, evs)
                    if not start_dt:
                        continue
                    first_time: time | None = None
                    for ev in evs:
                        if ev.date != start_dt.isoformat():
                            continue
                        tr = parse_time_range(ev.time)
                        if tr and (first_time is None or tr[0] < first_time):
                            first_time = tr[0]
                    if first_time is None:
                        first_time = time(0, 0)
                    if first_time >= time(17, 0):
                        sched = datetime.combine(start_dt, time(13, 0), tz)
                    else:
                        sched = datetime.combine(start_dt - timedelta(days=1), time(21, 0), tz)
                    if now >= sched and now.date() <= (end_dt or start_dt):
                        try:
                            await send_festival_poll(db, fest, group_id, bot)
                        except Exception as e:
                            logging.error("VK poll send failed for %s: %s", name, e)
                del rows
                await asyncio.sleep(0)
                off += LIMIT


async def init_db_and_scheduler(
    app: web.Application, db: Database, bot: Bot, webhook: str
) -> None:
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
    scheduler_startup(db, bot)
    app["daily_scheduler"] = asyncio.create_task(daily_scheduler(db, bot))
    app["add_event_worker"] = asyncio.create_task(add_event_queue_worker(db, bot))
    app["add_event_watch"] = asyncio.create_task(_watch_add_event_worker(app, db, bot))
    app["job_outbox_worker"] = asyncio.create_task(job_outbox_worker(db, bot))
    gc.collect()
    logging.info("BOOT_OK pid=%s", os.getpid())


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
        f"ticket: {fest.ticket_url or ''}",
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
        [
            types.InlineKeyboardButton(
                text=("Delete ticket" if fest.ticket_url else "Add ticket"),
                callback_data=f"festeditfield:{fest.id}:ticket",
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


async def send_job_status(chat_id: int, event_id: int, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        stmt = select(JobOutbox).where(JobOutbox.event_id == event_id).order_by(JobOutbox.id)
        jobs = (await session.execute(stmt)).scalars().all()
    if not jobs:
        await bot.send_message(chat_id, "Нет задач")
        return
    lines = ["task | status | attempts | updated | result"]
    for j in jobs:
        link = await _job_result_link(j.task, event_id, db)
        result = link if j.status == JobStatus.done else (j.last_error or "")
        lines.append(
            f"{TASK_LABELS[j.task.value]} | {j.status.value} | {j.attempts} | {j.updated_at:%H:%M:%S} | {result}"
        )
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="🔁 Перезапустить невыполненные",
                    callback_data=f"requeue:{event_id}",
                )
            ]
        ]
    )
    await bot.send_message(chat_id, "\n".join(lines), reply_markup=markup)


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


async def handle_status(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or "").split()
    if len(parts) > 1 and parts[1].isdigit():
        await send_job_status(message.chat.id, int(parts[1]), db, bot)
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    uptime = time.time() - START_TIME
    qlen = add_event_queue.qsize()
    jobs = list(JOB_HISTORY)[-5:]
    lines = [
        f"uptime: {int(uptime)}s",
        f"queue_len: {qlen}",
    ]
    if jobs:
        lines.append("last_jobs:")
        for j in reversed(jobs):
            when = j["when"].strftime("%H:%M:%S")
            lines.append(f"- {j['id']} {when} {j['status']} {j['took_ms']}ms")
    if LAST_RUN_ID:
        lines.append(f"last_run_id: {LAST_RUN_ID}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_trace(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await bot.send_message(message.chat.id, "Usage: /trace <run_id>")
        return
    run_id = parts[1].strip()
    lines = []
    for ts, level, msg in LOG_BUFFER:
        if run_id in msg:
            lines.append(f"{ts.strftime('%H:%M:%S')} {level[0]} {msg}")
    await bot.send_message(message.chat.id, "\n".join(lines) or "No trace")


async def handle_last_errors(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = message.text.split()
    count = 5
    if len(parts) > 1:
        try:
            count = min(int(parts[1]), len(ERROR_BUFFER))
        except Exception:
            pass
    errs = list(ERROR_BUFFER)[-count:]
    lines = []
    for e in reversed(errs):
        lines.append(
            f"{e['time'].strftime('%H:%M:%S')} {e['type']} {e['where']}"
        )
        await bot.send_message(message.chat.id, "\n".join(lines) or "No errors")


async def handle_debug(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = (message.text or "").split()
    if len(parts) < 2 or parts[1] != "queue":
        await bot.send_message(message.chat.id, "Usage: /debug queue")
        return
    async with db.get_session() as session:
        task_rows = await session.execute(
            select(JobOutbox.task, func.count()).group_by(JobOutbox.task)
        )
        event_rows = await session.execute(
            select(JobOutbox.event_id, func.count()).group_by(JobOutbox.event_id)
        )
    lines = ["tasks:"]
    for task, cnt in task_rows.all():
        lines.append(f"{task.value}: {cnt}")
    lines.append("events:")
    for eid, cnt in event_rows.all():
        lines.append(f"{eid}: {cnt}")
    await bot.send_message(message.chat.id, "\n".join(lines))


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


async def handle_mem(message: types.Message, db: Database, bot: Bot):
    rss, _ = mem_info(update=False)
    await bot.send_message(message.chat.id, f"RSS: {rss / (1024**2):.1f} MB")


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

    data = await dump_database(db)
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
    await restore_database(bio.getvalue(), db)
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

    if event.source_vk_post_url:
        try:
            await sync_vk_source_post(
                event,
                event.source_text,
                db,
                bot,
                ics_url=event.ics_url,
                append_text=False,
            )
        except Exception as e:
            logging.error("failed to sync VK source post: %s", e)

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
    add_event_sessions[message.from_user.id] = True
    logging.info(
        "handle_add_event_start session opened for user %s", message.from_user.id
    )
    await bot.send_message(message.chat.id, "Send event text and optional photo")


async def handle_vk_link_command(message: types.Message, db: Database, bot: Bot):
    parts = (message.text or "").split(maxsplit=2)
    logging.info("handle_vk_link_command start: user=%s", message.from_user.id)
    if len(parts) < 3:
        await bot.send_message(
            message.chat.id, "Usage: /vklink <event_id> <VK post link>"
        )
        return
    try:
        eid = int(parts[1])
    except ValueError:
        await bot.send_message(message.chat.id, "Invalid event id")
        return
    link = parts[2].strip()
    if not is_vk_wall_url(link):
        await bot.send_message(message.chat.id, "Invalid link")
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, eid)
        if not event:
            await bot.send_message(message.chat.id, "Event not found")
            return
        if event.creator_id != message.from_user.id and not (user and user.is_superadmin):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        event.source_post_url = link
        await session.commit()
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
        elif field == "ticket":
            fest.ticket_url = None if text in {"", "-"} else text
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

        bot=None,

    )
    logging.info("forward parsed %d events", len(results))
    if not results:
        logging.info("no events parsed from forwarded text")
        return
    for saved, added, lines, status in results:
        if isinstance(saved, Festival):
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text="Создать события по дням",
                            callback_data=f"festdays:{saved.id}",
                        )
                    ]
                ]
            )
            await bot.send_message(
                message.chat.id,
                "Festival added\n" + "\n".join(lines),
                reply_markup=markup,
            )
            continue
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
            await publish_event_progress(saved, db, bot, message.chat.id)
        except Exception as e:
            logging.error("failed to send event response: %s", e)


async def telegraph_test():
    token = get_telegraph_token()
    if not token:
        print("Unable to obtain Telegraph token")
        return
    tg = Telegraph(access_token=token, domain="telegra.ph")
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
    tg = Telegraph(access_token=token, domain="telegra.ph")
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
            catbox_msg = ""
            tg_urls: list[str] = []
        else:
            catbox_urls, tg_urls, catbox_msg = await upload_images(images)
            urls = catbox_urls or tg_urls
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
        html_content = lint_telegraph_html(html_content)
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
    tg = Telegraph(access_token=token, domain="telegra.ph")
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
    tg = Telegraph(access_token=token, domain="telegra.ph")
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
    """Populate event.description from the Telegraph source page if missing."""
    if event.description:
        logging.info(
            "skip description update for event %s: already present", event.id
        )
        return
    if not event.telegraph_path:
        logging.info(
            "skip description update for event %s: no telegraph page", event.id
        )
        return
    logging.info(
        "updating description for event %s from %s",
        event.id,
        event.telegraph_path,
    )
    text = await get_source_page_text(event.telegraph_path)
    if not text:
        logging.info("no source text for event %s", event.id)
        return
    try:
        parsed = await parse_event_via_4o(text)
    except Exception as e:
        logging.error("Failed to parse source text for description: %s", e)
        return
    if not parsed:
        logging.info("4o returned no data for event %s", event.id)
        return
    desc = parsed[0].get("short_description", "").strip()
    if not desc:
        logging.info("no short description parsed for event %s", event.id)
        return
    async with db.get_session() as session:
        obj = await session.get(Event, event.id)
        if obj:
            obj.description = desc
            await session.commit()
            event.description = desc
            logging.info("stored description for event %s", event.id)


async def build_source_page_content(
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
) -> tuple[str, str, int]:
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
        catbox_urls, tg_urls, catbox_msg = await upload_images(images)
        urls = catbox_urls or tg_urls
    if source_url and display_link:
        html_content += (
            f'<p><a href="{html.escape(source_url)}"><strong>{html.escape(title)}</strong></a></p>'
        )
    else:
        html_content += f"<p><strong>{html.escape(title)}</strong></p>"
    for url in urls:
        html_content += f'<figure><img src="{html.escape(url)}"/></figure>'
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
    html_content = lint_telegraph_html(html_content)
    return html_content, catbox_msg, len(urls)


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
    if db:
        from models import Event
        from sqlalchemy import select

        async with db.get_session() as session:
            res = await session.execute(
                select(Event.telegraph_url, Event.telegraph_path).where(
                    Event.source_text == text
                )
            )
            existing = res.first()
            if existing and existing.telegraph_path:
                return existing.telegraph_url, existing.telegraph_path, "", 0
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return None
    tg = Telegraph(access_token=token)
    html_content, catbox_msg, uploaded = await build_source_page_content(
        title,
        text,
        source_url,
        html_text,
        media,
        ics_url,
        db,
        display_link=display_link,
        catbox_urls=catbox_urls,
    )
    from telegraph.utils import html_to_nodes

    nodes = html_to_nodes(html_content)
    try:
        page = await telegraph_create_page(
            tg,
            title=title,
            author_name="Полюбить Калининград Анонсы",
            content=nodes,
            return_content=False,
        )
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    url = normalize_telegraph_url(page.get("url"))
    logging.info("Created telegraph page %s", url)
    return url, page.get("path"), catbox_msg, uploaded


def create_app() -> web.Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    webhook = os.getenv("WEBHOOK_URL")
    if not webhook:
        raise RuntimeError("WEBHOOK_URL is missing")

    session = IPv4AiohttpSession(timeout=ClientTimeout(total=HTTP_TIMEOUT))
    bot = SafeBot(token, session=session)
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
        logging.info("add_event_wrapper start: user=%s", message.from_user.id)
        if message.from_user.id in add_event_sessions:
            return

        text = normalize_addevent_text(strip_leading_cmd(message.text or ""))
        if not text:
            text = normalize_addevent_text(strip_leading_cmd(message.caption or ""))
        if not text:
            logging.info("add_event_wrapper usage: empty input")
            await send_usage_fast(bot, message.chat.id)
            return
        message.text = f"/addevent {text}"
        await enqueue_add_event(message, db, bot)

    async def add_event_raw_wrapper(message: types.Message):
        logging.info("add_event_raw_wrapper start: user=%s", message.from_user.id)
        await enqueue_add_event_raw(message, db, bot)

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
        logging.info("add_event_start_wrapper start: user=%s", message.from_user.id)
        await handle_add_event_start(message, db, bot)

    async def add_event_session_wrapper(message: types.Message):
        logging.info("add_event_session_wrapper start: user=%s", message.from_user.id)
        await enqueue_add_event(message, db, bot)

    async def vk_link_cmd_wrapper(message: types.Message):
        logging.info("vk_link_cmd_wrapper start: user=%s", message.from_user.id)
        await handle_vk_link_command(message, db, bot)

    async def status_wrapper(message: types.Message):
        await handle_status(message, db, bot)

    async def trace_wrapper(message: types.Message):
        await handle_trace(message, db, bot)

    async def last_errors_wrapper(message: types.Message):
        await handle_last_errors(message, db, bot)

    async def debug_wrapper(message: types.Message):
        await handle_debug(message, db, bot)

    async def mem_wrapper(message: types.Message):
        await handle_mem(message, db, bot)

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
        or c.data.startswith("festdays:")
        or c.data.startswith("requeue:")
    ,
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(
        add_event_session_wrapper, lambda m: m.from_user.id in add_event_sessions
    )
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
    dp.message.register(vk_link_cmd_wrapper, Command("vklink"))
    dp.message.register(partner_info_wrapper, lambda m: m.from_user.id in partner_info_sessions)
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(reg_daily_wrapper, Command("regdailychannels"))
    dp.message.register(daily_wrapper, Command("daily"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(fest_wrapper, Command("fest"))

    dp.message.register(pages_wrapper, Command("pages"))
    dp.message.register(stats_wrapper, Command("stats"))
    dp.message.register(status_wrapper, Command("status"))
    dp.message.register(trace_wrapper, Command("trace"))
    dp.message.register(last_errors_wrapper, Command("last_errors"))
    dp.message.register(debug_wrapper, Command("debug"))
    dp.message.register(mem_wrapper, Command("mem"))
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

    async def health_handler(request: web.Request) -> web.Response:
        async with span("healthz"):
            return web.Response(text="ok")

    app.router.add_get("/healthz", health_handler)

    async def on_startup(app: web.Application):
        loop = asyncio.get_running_loop()
        loop.create_task(init_db_and_scheduler(app, db, bot, webhook))

    async def on_shutdown(app: web.Application):
        await bot.session.close()
        if "add_event_watch" in app:
            app["add_event_watch"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_watch"]
        if "add_event_worker" in app:
            app["add_event_worker"].cancel()
            with contextlib.suppress(Exception):
                await app["add_event_worker"]
        if "daily_scheduler" in app:
            app["daily_scheduler"].cancel()
            with contextlib.suppress(Exception):
                await app["daily_scheduler"]
        scheduler_cleanup()
        await close_vk_session()
        close_supabase_client()

    global _startup_handler_registered
    if not _startup_handler_registered:
        app.on_startup.append(on_startup)
        _startup_handler_registered = True
    app.on_shutdown.append(on_shutdown)
    return app


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
