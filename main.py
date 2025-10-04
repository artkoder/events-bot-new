"""
Debugging:
    EVBOT_DEBUG=1 fly deploy ...
    Logs will include ▶/■ markers with RSS & duration.
"""

from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary
import logging
import os
import time as unixtime
import time as _time
import tempfile
import calendar
import math
from collections import Counter


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
logger = logging.getLogger(__name__)

def logline(tag: str, eid: int | None, msg: str, **kw) -> None:
    kv = " ".join(f"{k}={v}" for k, v in kw.items() if v is not None)
    logging.info(
        "%s %s %s%s",
        tag,
        f"[E{eid}]" if eid else "",
        msg,
        (f" | {kv}" if kv else ""),
    )


def log_festcover(level: int, festival_id: int | None, action: str, **kw: object) -> None:
    details = " ".join(f"{key}={value}" for key, value in kw.items() if value is not None)
    parts = [f"festcover.{action}"]
    if festival_id is not None:
        parts.append(f"festival_id={festival_id}")
    if details:
        parts.append(details)
    logging.log(level, " ".join(parts))


_QUOTE_CHARS = "'\"«»“”„‹›‚‘’`"
_START_WORDS = ("фестиваль", "международный", "областной", "городской")

def normalize_alias(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.casefold().strip()
    if not normalized:
        return ""
    normalized = normalized.translate(str.maketrans("", "", _QUOTE_CHARS))
    while True:
        for word in _START_WORDS:
            if normalized.startswith(word + " "):
                normalized = normalized[len(word) :].lstrip()
                break
            if normalized == word:
                normalized = ""
                break
        else:
            break
        if not normalized:
            break
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

from datetime import date, datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo
from typing import (
    Optional,
    Tuple,
    Iterable,
    Any,
    Callable,
    Awaitable,
    List,
    Literal,
    Collection,
    Sequence,
    Mapping,
)
from urllib.parse import urlparse, parse_qs, ParseResult
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

from aiogram import Bot, Dispatcher, types, F
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
from aiogram.exceptions import TelegramBadRequest
import socket
from difflib import SequenceMatcher
import json
import re
import httpx
import hashlib
import unicodedata
import vk_intake
import vk_review
import poster_ocr
from poster_media import (
    PosterMedia,
    apply_ocr_results_to_media,
    build_poster_summary,
    collect_poster_texts,
    process_media,
)
import argparse
import shlex

from telegraph import Telegraph, TelegraphException
from net import http_call, VK_FALLBACK_CODES
from digests import (
    build_lectures_digest_preview,
    build_masterclasses_digest_preview,
    build_exhibitions_digest_preview,
    build_psychology_digest_preview,
    build_science_pop_digest_preview,
    build_kraevedenie_digest_preview,
    build_networking_digest_preview,
    build_entertainment_digest_preview,
    build_markets_digest_preview,
    build_theatre_classic_digest_preview,
    build_theatre_modern_digest_preview,
    build_meetups_digest_preview,
    build_movies_digest_preview,
    format_event_line_html,
    pick_display_link,
    extract_catbox_covers_from_telegraph,
    compose_digest_caption,
    compose_digest_intro_via_4o,
    compose_masterclasses_intro_via_4o,
    compose_exhibitions_intro_via_4o,
    compose_psychology_intro_via_4o,
    normalize_topics,
    visible_caption_len,
    attach_caption_if_fits,
)

from functools import partial, lru_cache
from collections import defaultdict, deque
from bisect import bisect_left
from cachetools import TTLCache
import asyncio
import contextlib
import random
import html
from types import SimpleNamespace
from dataclasses import dataclass, field
import sqlite3
from io import BytesIO
import aiosqlite
import gc
import atexit
import vision_test
from markup import (
    simple_md_to_html,
    telegraph_br,
    DAY_START,
    DAY_END,
    PERM_START,
    PERM_END,
    FEST_NAV_START,
    FEST_NAV_END,
    FEST_INDEX_INTRO_START,
    FEST_INDEX_INTRO_END,
    linkify_for_telegraph,
    sanitize_for_vk,
)
from aiogram.utils.text_decorations import html_decoration
from sections import (
    replace_between_markers,
    content_hash,
    parse_month_sections,
    ensure_footer_nav_with_hr,
    dedup_same_date,
)
from db import Database
from shortlinks import (
    ensure_vk_short_ics_link,
    ensure_vk_short_ticket_link,
    format_vk_short_url,
)
from scheduling import startup as scheduler_startup, cleanup as scheduler_cleanup
from sqlalchemy import select, update, delete, text, func, or_, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from models import (
    TOPIC_LABELS,
    TOPIC_IDENTIFIERS,
    normalize_topic_identifier,
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
    EventPoster,
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
# public alias for external modules/handlers
page_lock = _page_locks
_month_next_run: dict[str, float] = defaultdict(float)
_week_next_run: dict[str, float] = defaultdict(float)
_weekend_next_run: dict[str, float] = defaultdict(float)
_vk_week_next_run: dict[str, float] = defaultdict(float)
_vk_weekend_next_run: dict[str, float] = defaultdict(float)
_partner_last_run: date | None = None

_startup_handler_registered = False

# in-memory diagnostic buffers
START_TIME = _time.time()
LOG_BUFFER: deque[tuple[datetime, str, str]] = deque(maxlen=200)
ERROR_BUFFER: deque[dict[str, Any]] = deque(maxlen=50)
JOB_HISTORY: deque[dict[str, Any]] = deque(maxlen=20)
LAST_RUN_ID: str | None = None


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_job(job: "JobOutbox" | None) -> "JobOutbox" | None:
    if job is None:
        return None
    job.updated_at = _ensure_utc(job.updated_at)
    job.next_run_at = _ensure_utc(job.next_run_at)
    return job


class MemoryLogHandler(logging.Handler):
    """Store recent log records in memory for diagnostics."""

    _job_id_re = re.compile(r"job_id=(\S+)")
    _run_id_re = re.compile(r"run_id=(\S+)")
    _took_re = re.compile(r"took_ms=(\d+)")

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - simple
        msg = record.getMessage()
        ts = datetime.now(timezone.utc)
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
db: Database | None = None
TELEGRAPH_TOKEN_FILE = os.getenv("TELEGRAPH_TOKEN_FILE", "/data/telegraph_token.txt")
TELEGRAPH_AUTHOR_NAME = os.getenv(
    "TELEGRAPH_AUTHOR_NAME", "Полюбить Калининград Анонсы"
)
TELEGRAPH_AUTHOR_URL = os.getenv(
    "TELEGRAPH_AUTHOR_URL", "https://t.me/kenigevents"
)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "events-ics")
VK_TOKEN = os.getenv("VK_TOKEN")
VK_TOKEN_AFISHA = os.getenv("VK_TOKEN_AFISHA")  # NEW
VK_USER_TOKEN = os.getenv("VK_USER_TOKEN")
VK_SERVICE_TOKEN = os.getenv("VK_SERVICE_TOKEN")
VK_READ_VIA_SERVICE = os.getenv("VK_READ_VIA_SERVICE", "true").lower() == "true"
VK_MIN_INTERVAL_MS = int(os.getenv("VK_MIN_INTERVAL_MS", "350"))
_last_vk_call = 0.0


async def _vk_throttle() -> None:
    global _last_vk_call
    now = _time.monotonic()
    wait = (_last_vk_call + VK_MIN_INTERVAL_MS / 1000) - now
    if wait > 0:
        await asyncio.sleep(wait)
    _last_vk_call = _time.monotonic()


VK_SERVICE_READ_METHODS = {
    "utils.resolveScreenName",
    "groups.getById",
    "wall.get",
    "wall.getById",
    "photos.getById",
}
VK_SERVICE_READ_PREFIXES = ("video.get",)

VK_MAIN_GROUP_ID = os.getenv("VK_MAIN_GROUP_ID")
VK_AFISHA_GROUP_ID = os.getenv("VK_AFISHA_GROUP_ID")
VK_API_VERSION = os.getenv("VK_API_VERSION", "5.199")
try:
    VK_MAX_ATTACHMENTS = int(os.getenv("VK_MAX_ATTACHMENTS", "10"))
except ValueError:
    VK_MAX_ATTACHMENTS = 10

VK_ALLOW_TRUE_REPOST = (
    os.getenv("VK_ALLOW_TRUE_REPOST", "false").lower() == "true"
)
try:
    VK_SHORTPOST_MAX_PHOTOS = int(os.getenv("VK_SHORTPOST_MAX_PHOTOS", "4"))
except ValueError:
    VK_SHORTPOST_MAX_PHOTOS = 4

# VK allows editing community posts for 14 days.
VK_POST_MAX_EDIT_AGE = timedelta(days=14)

# which actor token to use for VK API calls
VK_ACTOR_MODE = os.getenv("VK_ACTOR_MODE", "auto")

# error codes triggering fallback from group to user token are baked in

# scheduling options for weekly VK post edits
WEEK_EDIT_MODE = os.getenv("WEEK_EDIT_MODE", "deferred")
WEEK_EDIT_CRON = os.getenv("WEEK_EDIT_CRON", "02:30")

# new scheduling and captcha parameters
VK_WEEK_EDIT_ENABLED = (
    os.getenv("VK_WEEK_EDIT_ENABLED", "false").lower() == "true"
)
# schedule for VK week post edits (HH:MM)
VK_WEEK_EDIT_SCHEDULE = os.getenv("VK_WEEK_EDIT_SCHEDULE", "02:10")
# timezone for schedule and captcha quiet hours
VK_WEEK_EDIT_TZ = os.getenv("VK_WEEK_EDIT_TZ", "Europe/Kaliningrad")

# captcha handling configuration
CAPTCHA_WAIT_S = int(os.getenv("CAPTCHA_WAIT_S", "600"))
CAPTCHA_MAX_ATTEMPTS = int(os.getenv("CAPTCHA_MAX_ATTEMPTS", "2"))
CAPTCHA_NIGHT_RANGE = os.getenv("CAPTCHA_NIGHT_RANGE", "00:00-07:00")
CAPTCHA_RETRY_AT = os.getenv("CAPTCHA_RETRY_AT", "08:10")
VK_CAPTCHA_TTL_MIN = int(os.getenv("VK_CAPTCHA_TTL_MIN", "60"))
# quiet hours for captcha notifications (HH:MM-HH:MM, empty = disabled)
VK_CAPTCHA_QUIET = os.getenv("VK_CAPTCHA_QUIET", "")
VK_CRAWL_JITTER_SEC = int(os.getenv("VK_CRAWL_JITTER_SEC", "600"))

logging.info(
    "vk.config groups: main=-%s, afisha=-%s; user_token=%s, token_main=%s, token_afisha=%s, service_token=%s",
    VK_MAIN_GROUP_ID,
    VK_AFISHA_GROUP_ID,
    "present" if VK_USER_TOKEN else "missing",
    "present" if VK_TOKEN else "missing",
    "present" if VK_TOKEN_AFISHA else "missing",
    "present" if VK_SERVICE_TOKEN else "missing",
)


@dataclass
class VkActor:
    kind: Literal["group", "user"]
    token: str | None
    label: str  # for logs: "group:main", "group:afisha", "user"


def choose_vk_actor(owner_id: int, intent: str) -> list[VkActor]:
    actors: list[VkActor] = []
    try:
        main_id = int(VK_MAIN_GROUP_ID) if VK_MAIN_GROUP_ID else None
    except ValueError:
        main_id = None
    try:
        afisha_id = int(VK_AFISHA_GROUP_ID) if VK_AFISHA_GROUP_ID else None
    except ValueError:
        afisha_id = None
    if owner_id == -(afisha_id or 0):
        if VK_TOKEN_AFISHA:
            actors.append(VkActor("group", VK_TOKEN_AFISHA, "group:afisha"))
    elif owner_id == -(main_id or 0):
        if VK_TOKEN:
            actors.append(VkActor("group", VK_TOKEN, "group:main"))
    elif VK_TOKEN:
        actors.append(VkActor("group", VK_TOKEN, "group:main"))
    if VK_USER_TOKEN:
        actors.append(VkActor("user", None, "user"))
    return actors

# metrics counters
vk_fallback_group_to_user_total: dict[str, int] = defaultdict(int)
vk_crawl_groups_total = 0
vk_crawl_posts_scanned_total = 0
vk_crawl_matched_total = 0
vk_crawl_duplicates_total = 0
vk_crawl_safety_cap_total = 0
vk_inbox_inserted_total = 0
vk_review_actions_total: dict[str, int] = defaultdict(int)
vk_repost_attempts_total = 0
vk_repost_errors_total = 0

# histogram buckets for VK import duration in seconds
vk_import_duration_buckets: dict[float, int] = {
    1.0: 0,
    2.5: 0,
    5.0: 0,
    10.0: 0,
}
vk_import_duration_sum = 0.0
vk_import_duration_count = 0


def format_metrics() -> str:
    lines: list[str] = []
    for method, count in vk_fallback_group_to_user_total.items():
        lines.append(
            f"vk_fallback_group_to_user_total{{method=\"{method}\"}} {count}"
        )
    lines.append(f"vk_crawl_groups_total {vk_crawl_groups_total}")
    lines.append(f"vk_crawl_posts_scanned_total {vk_crawl_posts_scanned_total}")
    lines.append(f"vk_crawl_matched_total {vk_crawl_matched_total}")
    lines.append(f"vk_crawl_duplicates_total {vk_crawl_duplicates_total}")
    lines.append(f"vk_crawl_safety_cap_total {vk_crawl_safety_cap_total}")
    lines.append(f"vk_inbox_inserted_total {vk_inbox_inserted_total}")
    for action, count in vk_review_actions_total.items():
        lines.append(
            f"vk_review_actions_total{{action=\"{action}\"}} {count}"
        )
    lines.append(f"vk_repost_attempts_total {vk_repost_attempts_total}")
    lines.append(f"vk_repost_errors_total {vk_repost_errors_total}")

    cumulative = 0
    for bound in sorted(vk_import_duration_buckets):
        cumulative += vk_import_duration_buckets[bound]
        lines.append(
            f"vk_import_duration_seconds_bucket{{le=\"{bound}\"}} {cumulative}"
        )
    lines.append(
        f"vk_import_duration_seconds_bucket{{le=\"+Inf\"}} {vk_import_duration_count}"
    )
    lines.append(f"vk_import_duration_seconds_sum {vk_import_duration_sum:.6f}")
    lines.append(f"vk_import_duration_seconds_count {vk_import_duration_count}")
    lines.append(
        "vk_intake_processing_time_seconds_total "
        f"{vk_intake.processing_time_seconds_total:.6f}"
    )
    return "\n".join(lines) + "\n"


async def metrics_handler(request: web.Request) -> web.Response:
    return web.Response(
        text=format_metrics(), content_type="text/plain; version=0.0.4"
    )
# circuit breaker for group-token permission errors
VK_CB_TTL = 12 * 3600
vk_group_blocked: dict[str, float] = {}
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
# user_id -> vk_source_id for default time update
@dataclass
class VkDefaultTimeSession:
    source_id: int
    page: int
    message: types.Message | None = None


@dataclass
class VkDefaultTicketLinkSession:
    source_id: int
    page: int
    message: types.Message | None = None


@dataclass
class VkDefaultLocationSession:
    source_id: int
    page: int
    message: types.Message | None = None


vk_default_time_sessions: TTLCache[int, VkDefaultTimeSession] = TTLCache(
    maxsize=64, ttl=3600
)
vk_default_ticket_link_sessions: TTLCache[
    int, VkDefaultTicketLinkSession
] = TTLCache(maxsize=64, ttl=3600)
vk_default_location_sessions: TTLCache[
    int, VkDefaultLocationSession
] = TTLCache(maxsize=64, ttl=3600)
# waiting for VK source add input
vk_add_source_sessions: set[int] = set()

# operator_id -> (inbox_id, batch_id) awaiting extra info during VK review
vk_review_extra_sessions: dict[int, tuple[int, str, bool]] = {}


@dataclass
class VkReviewStorySession:
    inbox_id: int
    batch_id: str | None = None


vk_review_story_sessions: dict[int, VkReviewStorySession] = {}

@dataclass
class VkShortpostOpState:
    chat_id: int
    preview_text: str | None = None
    preview_link_attachment: str | None = None


# event_id -> operator chat id awaiting shortpost publication and cached preview
vk_shortpost_ops: dict[int, VkShortpostOpState] = {}
# admin user_id -> (event_id, admin_chat_message_id) awaiting custom shortpost text
vk_shortpost_edit_sessions: dict[int, tuple[int, int]] = {}

# superadmin user_id -> pending partner user_id
partner_info_sessions: TTLCache[int, int] = TTLCache(maxsize=64, ttl=3600)
# user_id -> (festival_id, field?) for festival editing
festival_edit_sessions: TTLCache[int, tuple[int, str | None]] = TTLCache(maxsize=64, ttl=3600)

# user_id -> cached festival inference for makefest flow
makefest_sessions: TTLCache[int, dict[str, Any]] = TTLCache(maxsize=64, ttl=3600)

# cache for first image in Telegraph pages
telegraph_first_image: TTLCache[str, str] = TTLCache(maxsize=128, ttl=24 * 3600)

# pending event text/photo input
AddEventMode = Literal["event", "festival"]
add_event_sessions: TTLCache[int, AddEventMode] = TTLCache(maxsize=64, ttl=3600)


class FestivalRequiredError(RuntimeError):
    """Raised when festival mode requires an explicit festival but none was found."""
# waiting for a date for events listing
events_date_sessions: TTLCache[int, bool] = TTLCache(maxsize=64, ttl=3600)

@dataclass(frozen=True)
class TouristFactor:
    code: str
    emoji: str
    title: str


TOURIST_FACTORS: list[TouristFactor] = [
    TouristFactor("targeted_for_tourists", "🎯", "Нацелен на туристов"),
    TouristFactor("unique_to_region", "🧭", "Уникально для региона"),
    TouristFactor("festival_major", "🎪", "Фестиваль / масштаб"),
    TouristFactor("nature_or_landmark", "🌊", "Природа / море / лендмарк / замок"),
    TouristFactor("photogenic_blogger", "📸", "Фотогенично / есть что постить"),
    TouristFactor("local_flavor_crafts", "🍲", "Местный колорит / кухня / крафт"),
    TouristFactor("easy_logistics", "🚆", "Просто добраться"),
]

TOURIST_FACTOR_BY_CODE: dict[str, TouristFactor] = {
    factor.code: factor for factor in TOURIST_FACTORS
}
TOURIST_FACTOR_CODES: list[str] = [factor.code for factor in TOURIST_FACTORS]
TOURIST_FACTOR_ALIASES: dict[str, str] = {
    "history": "unique_to_region",
    "culture": "unique_to_region",
    "atmosphere": "local_flavor_crafts",
    "city": "local_flavor_crafts",
    "sea": "nature_or_landmark",
    "water": "nature_or_landmark",
    "nature": "nature_or_landmark",
    "scenic_nature": "nature_or_landmark",
    "iconic_location": "photogenic_blogger",
    "shows_local_life": "local_flavor_crafts",
    "local_cuisine": "local_flavor_crafts",
    "food": "local_flavor_crafts",
    "gastronomy": "local_flavor_crafts",
    "family": "easy_logistics",
    "family_friendly": "easy_logistics",
    "events": "festival_major",
    "event": "festival_major",
    "photogenic": "photogenic_blogger",
    "blogger": "photogenic_blogger",
}


@dataclass
class TouristReasonSession:
    event_id: int
    chat_id: int
    message_id: int
    source: str


@dataclass
class TouristNoteSession:
    event_id: int
    chat_id: int
    message_id: int
    source: str
    markup: types.InlineKeyboardMarkup | None
    message_text: str | None
    menu: bool


tourist_reason_sessions: TTLCache[int, TouristReasonSession] = TTLCache(
    maxsize=256, ttl=15 * 60
)
tourist_note_sessions: TTLCache[int, TouristNoteSession] = TTLCache(
    maxsize=256, ttl=10 * 60
)
tourist_message_sources: dict[tuple[int, int], str] = {}


def _tourist_label_display(event: Event) -> str:
    if event.tourist_label == 1:
        return "Да"
    if event.tourist_label == 0:
        return "Нет"
    return "—"


def _normalize_tourist_factors(factors: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for code in factors:
        mapped = TOURIST_FACTOR_ALIASES.get(code, code)
        if mapped in TOURIST_FACTOR_BY_CODE and mapped not in seen:
            seen.add(mapped)
    ordered = [code for code in TOURIST_FACTOR_CODES if code in seen]
    return ordered


def build_tourist_status_lines(event: Event) -> list[str]:
    lines = [f"🌍 Туристам: {_tourist_label_display(event)}"]
    factors = _normalize_tourist_factors(event.tourist_factors or [])
    if factors:
        lines.append(f"🧩 {len(factors)} причин")
    if event.tourist_note and event.tourist_note.strip():
        lines.append("📝 есть комментарий")
    return lines


def _determine_tourist_source(callback: types.CallbackQuery) -> str:
    message = callback.message
    if message:
        key = (message.chat.id, message.message_id)
        stored = tourist_message_sources.get(key)
        if stored:
            return stored
    return "tg"


def _is_tourist_menu_markup(
    markup: types.InlineKeyboardMarkup | None,
) -> bool:
    if not markup or not markup.inline_keyboard:
        return False
    has_reason_buttons = False
    has_done = False
    has_skip = False
    for row in markup.inline_keyboard:
        for btn in row:
            data = btn.callback_data
            if not data:
                continue
            if data.startswith("tourist:fx:"):
                has_reason_buttons = True
            elif data.startswith("tourist:fxdone"):
                has_done = True
            elif data.startswith("tourist:fxskip"):
                has_skip = True
    if has_reason_buttons:
        return True
    if has_done and has_skip:
        return True
    return False


def build_tourist_keyboard_block(
    event: Event, source: str
) -> list[list[types.InlineKeyboardButton]]:
    if not getattr(event, "id", None):
        return []
    _ = source
    yes_prefix = "✅ " if event.tourist_label == 1 else ""
    no_prefix = "✅ " if event.tourist_label == 0 else ""
    rows: list[list[types.InlineKeyboardButton]] = [
        [
            types.InlineKeyboardButton(
                text=f"{yes_prefix}Интересно туристам",
                callback_data=f"tourist:yes:{event.id}"
            ),
            types.InlineKeyboardButton(
                text=f"{no_prefix}Не интересно туристам",
                callback_data=f"tourist:no:{event.id}"
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="Причины",
                callback_data=f"tourist:fxdone:{event.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="✍️ Комментарий",
                callback_data=f"tourist:note:start:{event.id}",
            )
        ],
    ]
    if event.tourist_note and event.tourist_note.strip():
        rows[-1].append(
            types.InlineKeyboardButton(
                text="🧽 Очистить комментарий",
                callback_data=f"tourist:note:clear:{event.id}",
            )
        )
    return rows


def build_tourist_reason_rows(
    event: Event, source: str
) -> list[list[types.InlineKeyboardButton]]:
    if not getattr(event, "id", None):
        return []
    _ = source
    normalized = _normalize_tourist_factors(event.tourist_factors or [])
    selected = set(normalized)
    rows: list[list[types.InlineKeyboardButton]] = []
    for factor in TOURIST_FACTORS:
        prefix = "✅" if factor.code in selected else "➕"
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=f"{prefix} {factor.emoji} {factor.title}",
                    callback_data=f"tourist:fx:{factor.code}:{event.id}",
                )
            ]
        )
    comment_row = [
        types.InlineKeyboardButton(
            text="✍️ Комментарий",
            callback_data=f"tourist:note:start:{event.id}",
        )
    ]
    if event.tourist_note and event.tourist_note.strip():
        comment_row.append(
            types.InlineKeyboardButton(
                text="🧽 Очистить комментарий",
                callback_data=f"tourist:note:clear:{event.id}",
            )
        )
    rows.append(comment_row)
    rows.append(
        [
            types.InlineKeyboardButton(
                text="Готово", callback_data=f"tourist:fxdone:{event.id}"
            ),
            types.InlineKeyboardButton(
                text="Пропустить", callback_data=f"tourist:fxskip:{event.id}"
            ),
        ]
    )
    return rows


def append_tourist_block(
    base_rows: Sequence[Sequence[types.InlineKeyboardButton]],
    event: Event,
    source: str,
) -> list[list[types.InlineKeyboardButton]]:
    rows = [list(row) for row in base_rows]
    if getattr(event, "id", None):
        rows.extend(build_tourist_keyboard_block(event, source))
    return rows


def replace_tourist_block(
    markup: types.InlineKeyboardMarkup | None,
    event: Event,
    source: str,
    *,
    menu: bool = False,
) -> types.InlineKeyboardMarkup:
    base_rows: list[list[types.InlineKeyboardButton]] = []
    if markup and markup.inline_keyboard:
        for row in markup.inline_keyboard:
            if any(
                btn.callback_data and btn.callback_data.startswith("tourist:")
                for btn in row
            ):
                continue
            base_rows.append([btn for btn in row])
    if getattr(event, "id", None):
        if menu:
            base_rows.extend(build_tourist_reason_rows(event, source))
        else:
            base_rows.extend(build_tourist_keyboard_block(event, source))
    return types.InlineKeyboardMarkup(inline_keyboard=base_rows)


def apply_tourist_status_to_text(original_text: str | None, event: Event) -> str:
    status_lines = build_tourist_status_lines(event)
    if not original_text:
        return "\n".join(status_lines) if status_lines else ""
    lines = original_text.splitlines()
    if not lines:
        return original_text
    header = lines[0]
    rest = list(lines[1:])
    while rest and rest[0].startswith(("🌍", "🧩", "📝")):
        rest.pop(0)
    return "\n".join([header, *status_lines, *rest])


def build_event_card_message(
    header: str,
    event: Event,
    detail_lines: Sequence[str],
    extra_lines: Sequence[str] | None = None,
) -> str:
    body_lines = [*build_tourist_status_lines(event), *detail_lines]
    if extra_lines:
        for extra in extra_lines:
            if extra:
                body_lines.append(extra)
    if body_lines:
        return "\n".join([header, *body_lines])
    return header


def _user_can_label_event(user: User | None) -> bool:
    if not user or user.blocked:
        return False
    if user.is_superadmin:
        return True
    if user.is_partner:
        return False
    return True


async def update_tourist_message(
    callback: types.CallbackQuery,
    bot: Bot,
    event: Event,
    source: str,
    *,
    menu: bool = False,
    update_text: bool = True,
) -> None:
    message = callback.message
    if not message:
        return
    tourist_message_sources[(message.chat.id, message.message_id)] = source
    new_markup = replace_tourist_block(message.reply_markup, event, source, menu=menu)
    new_text: str | None = None
    if update_text:
        current = message.text if message.text is not None else message.caption
        if current is not None:
            new_text = apply_tourist_status_to_text(current, event)
    try:
        if new_text is not None:
            if message.text is not None:
                await bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    text=new_text,
                    reply_markup=new_markup,
                )
            elif message.caption is not None:
                await bot.edit_message_caption(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    caption=new_text,
                    reply_markup=new_markup,
                )
            else:
                await bot.edit_message_reply_markup(
                    chat_id=message.chat.id,
                    message_id=message.message_id,
                    reply_markup=new_markup,
                )
        else:
            await bot.edit_message_reply_markup(
                chat_id=message.chat.id,
                message_id=message.message_id,
                reply_markup=new_markup,
            )
    except TelegramBadRequest as exc:  # pragma: no cover - network quirks
        logging.warning(
            "tourist_update_failed",
            extra={"event_id": getattr(event, "id", None), "error": exc.message},
        )
        if new_text is None:
            return
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(
                chat_id=message.chat.id,
                message_id=message.message_id,
                reply_markup=new_markup,
            )
    if new_text is not None:
        try:
            session = tourist_note_sessions[callback.from_user.id]
        except KeyError:
            pass
        else:
            if (
                session.chat_id == message.chat.id
                and session.message_id == message.message_id
            ):
                session.markup = new_markup
                session.message_text = new_text
                tourist_note_sessions[callback.from_user.id] = session


async def _restore_tourist_reason_keyboard(
    callback: types.CallbackQuery,
    bot: Bot,
    db: Database,
    event_id: int,
    source: str,
) -> None:
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
    if event:
        await update_tourist_message(callback, bot, event, source, menu=False)

async def _build_makefest_session_state(
    event: Event, known_fests: Sequence[Festival]
) -> dict[str, Any]:
    fest_result = await infer_festival_for_event_via_4o(event, known_fests)

    telegraph_images: list[str] = []
    telegraph_source = event.telegraph_url or event.telegraph_path
    if telegraph_source:
        telegraph_images = await extract_telegraph_image_urls(telegraph_source)

    photo_candidates: list[str] = []
    for url in telegraph_images + (event.photo_urls or []):
        if url and url not in photo_candidates:
            photo_candidates.append(url)

    fest_data = fest_result["festival"]
    event_start: str | None = None
    event_end: str | None = None
    raw_date = getattr(event, "date", None)
    if isinstance(raw_date, str) and raw_date.strip():
        if ".." in raw_date:
            start_part, end_part = raw_date.split("..", 1)
            event_start = start_part.strip() or None
            event_end = end_part.strip() or None
        else:
            event_start = raw_date.strip()
    explicit_end = getattr(event, "end_date", None)
    if isinstance(explicit_end, str) and explicit_end.strip():
        event_end = explicit_end.strip()
    if event_start and not event_end:
        event_end = event_start
    if event_start and not fest_data.get("start_date"):
        fest_data["start_date"] = event_start
    if event_end and not fest_data.get("end_date"):
        fest_data["end_date"] = event_end

    duplicate_info_raw = fest_result.get("duplicate")
    duplicate_info: dict[str, Any] = {
        "match": False,
        "name": None,
        "normalized_name": None,
        "confidence": None,
        "dup_fid": None,
    }
    if isinstance(duplicate_info_raw, dict):
        match_flag = bool(duplicate_info_raw.get("match"))
        name = clean_optional_str(duplicate_info_raw.get("name"))
        normalized_name_raw = clean_optional_str(duplicate_info_raw.get("normalized_name"))
        confidence_raw = duplicate_info_raw.get("confidence")
        confidence_val: float | None = None
        if isinstance(confidence_raw, (int, float)):
            confidence_val = float(confidence_raw)
        elif isinstance(confidence_raw, str):
            try:
                confidence_val = float(confidence_raw.strip())
            except (TypeError, ValueError):
                confidence_val = None
        dup_fid_raw = duplicate_info_raw.get("dup_fid")
        dup_fid: int | None = None
        if dup_fid_raw not in (None, ""):
            try:
                dup_fid = int(dup_fid_raw)
            except (TypeError, ValueError):
                dup_fid = None
        duplicate_info = {
            "match": match_flag,
            "name": name,
            "normalized_name": normalized_name_raw or normalize_duplicate_name(name),
            "confidence": confidence_val,
            "dup_fid": dup_fid,
        }
    elif duplicate_info_raw is not None:
        logging.debug(
            "infer_festival_for_event_via_4o returned non-dict duplicate: %s",
            duplicate_info_raw,
        )

    if duplicate_info.get("dup_fid") is None and duplicate_info.get("normalized_name"):
        normalized_target = duplicate_info["normalized_name"]
        for fest in known_fests:
            if not getattr(fest, "id", None) or not getattr(fest, "name", None):
                continue
            if normalize_duplicate_name(fest.name) == normalized_target:
                duplicate_info["dup_fid"] = fest.id
                if not duplicate_info.get("name"):
                    duplicate_info["name"] = fest.name
                break

    existing_names: set[str] = set()
    if fest_data.get("name"):
        existing_names.add(fest_data["name"].lower())
    for name in fest_data.get("existing_candidates", []):
        if isinstance(name, str) and name.strip():
            existing_names.add(name.strip().lower())

    duplicate_fest: Festival | None = None
    if duplicate_info.get("dup_fid"):
        dup_id = duplicate_info["dup_fid"]
        duplicate_fest = next((fest for fest in known_fests if fest.id == dup_id), None)
        if duplicate_fest and not duplicate_info.get("name"):
            duplicate_info["name"] = duplicate_fest.name
        if duplicate_fest and not duplicate_info.get("normalized_name"):
            duplicate_info["normalized_name"] = normalize_duplicate_name(duplicate_fest.name)

    matched_fests: list[Festival] = []
    for fest in known_fests:
        if not fest.id:
            continue
        if fest.name and fest.name.lower() in existing_names:
            matched_fests.append(fest)

    seen_ids: set[int] = set()
    ordered_matches: list[Festival] = []
    if duplicate_fest and duplicate_fest.id:
        ordered_matches.append(duplicate_fest)
        seen_ids.add(duplicate_fest.id)
    for fest in matched_fests:
        if fest.id in seen_ids:
            continue
        ordered_matches.append(fest)
        seen_ids.add(fest.id)
    ordered_matches = ordered_matches[:5]

    return {
        "festival": fest_data,
        "photos": photo_candidates,
        "matches": [
            {"id": fest.id, "name": fest.name} for fest in ordered_matches if fest.id
        ],
        "duplicate": duplicate_info,
    }


# chat_id -> list of (message_id, text) for /exhibitions chunks
exhibitions_message_state: dict[int, list[tuple[int, str]]] = {}

# digest_id -> session data for digest preview
digest_preview_sessions: TTLCache[str, dict] = TTLCache(maxsize=64, ttl=30 * 60)

# ожидание фото после выбора выходных: user_id -> start(YYYY-MM-DD)
weekend_img_wait: TTLCache[int, str] = TTLCache(maxsize=100, ttl=900)
FESTIVALS_INDEX_MARKER = "festivals-index"

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
add_event_queue: asyncio.Queue[
    tuple[str, types.Message, AddEventMode | None, int]
] = asyncio.Queue(
    maxsize=200
)
# allow more time for handling slow background operations
ADD_EVENT_TIMEOUT = int(os.getenv("ADD_EVENT_TIMEOUT", "600"))
ADD_EVENT_RETRY_DELAYS = [30, 60, 120]  # сек
ADD_EVENT_MAX_ATTEMPTS = len(ADD_EVENT_RETRY_DELAYS) + 1
_ADD_EVENT_LAST_DEQUEUE_TS: float = 0.0

# queue for post-commit event update jobs


async def _watch_add_event_worker(app: web.Application, db: Database, bot: Bot):
    worker: asyncio.Task = app["add_event_worker"]
    STALL_GUARD_SECS = int(os.getenv("STALL_GUARD_SECS", str(ADD_EVENT_TIMEOUT + 30)))
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
            stalled_for = (
                _time.monotonic() - _ADD_EVENT_LAST_DEQUEUE_TS
                if _ADD_EVENT_LAST_DEQUEUE_TS
                else 0
            )
            if (
                add_event_queue.qsize() > 0
                and _ADD_EVENT_LAST_DEQUEUE_TS
                and stalled_for > STALL_GUARD_SECS
            ):
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
# toggle for Telegraph image uploads (disabled by default)
TELEGRAPH_IMAGE_UPLOAD: bool = os.getenv("TELEGRAPH_IMAGE_UPLOAD", "0") != "0"
_supabase_client: "Client | None" = None  # type: ignore[name-defined]
_vk_user_token_bad: str | None = None
_vk_captcha_needed: bool = False
_vk_captcha_sid: str | None = None
_vk_captcha_img: str | None = None
_vk_captcha_method: str | None = None
_vk_captcha_params: dict | None = None
_vk_captcha_resume: Callable[[], Awaitable[None]] | None = None
_vk_captcha_timeout: asyncio.Task | None = None
_vk_captcha_requested_at: datetime | None = None
_vk_captcha_awaiting_user: int | None = None
_vk_captcha_scheduler: Any | None = None
_vk_captcha_key: str | None = None
_shared_session: ClientSession | None = None
# backward-compatible aliases used in tests
_http_session: ClientSession | None = None

# tasks affected by VK captcha pause
VK_JOB_TASKS = {
    JobTask.vk_sync,
    JobTask.week_pages,
    JobTask.weekend_pages,
    JobTask.festival_pages,
}
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
ICS_SEMAPHORE: asyncio.Semaphore | None = None

# Skip creation/update of individual event Telegraph pages
DISABLE_EVENT_PAGE_UPDATES = False


def get_ics_semaphore() -> asyncio.Semaphore:
    global ICS_SEMAPHORE
    loop = asyncio.get_event_loop()
    if ICS_SEMAPHORE is None or ICS_SEMAPHORE._loop is not loop:  # type: ignore[attr-defined]
        ICS_SEMAPHORE = asyncio.Semaphore(1)
    return ICS_SEMAPHORE
FEST_JOB_MULT = 100_000

# Maximum number of images to accept in an album
# The default was previously 10 but the pipeline now supports up to 12 images
# per event source page.
MAX_ALBUM_IMAGES = int(os.getenv("MAX_ALBUM_IMAGES", "12"))

# Delay before finalizing a forwarded album (milliseconds)
ALBUM_FINALIZE_DELAY_MS = int(os.getenv("ALBUM_FINALIZE_DELAY_MS", "1500"))

# Time to keep album buffers without captions (seconds)
ALBUM_PENDING_TTL_S = int(os.getenv("ALBUM_PENDING_TTL_S", "60"))

# Maximum number of pending album timers
MAX_PENDING_ALBUMS = int(os.getenv("MAX_PENDING_ALBUMS", "50"))

LAST_CATBOX_MSG = ""
LAST_HTML_MODE = "native"
CUSTOM_EMOJI_MAP = {"\U0001f193" * 4: "Бесплатно"}

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
    if data[4:12] == b"ftypavif":
        return "avif"
    return None


def ensure_jpeg(data: bytes, name: str) -> tuple[bytes, str]:
    """Convert WEBP or AVIF images to JPEG."""
    kind = detect_image_type(data)
    if kind in {"webp", "avif"}:
        from PIL import Image

        bio_in = BytesIO(data)
        with Image.open(bio_in) as im:
            rgb = im.convert("RGB")
            bio_out = BytesIO()
            rgb.save(bio_out, format="JPEG")
            data = bio_out.getvalue()
        name = re.sub(r"\.[^.]+$", "", name) + ".jpg"
    return data, name


def validate_jpeg_markers(data: bytes) -> None:
    """Ensure JPEG payload contains SOS and EOI markers."""
    if b"\xff\xda" not in data or not data.endswith(b"\xff\xd9"):
        raise ValueError("incomplete jpeg payload")

# Timeout for OpenAI 4o requests (in seconds)
FOUR_O_TIMEOUT = float(os.getenv("FOUR_O_TIMEOUT", "60"))

# Limit prompt/response sizes for LLM calls
# Prompt limit is measured in characters because we clip raw text before sending it
# to the API, while response limits are expressed in tokens via the API parameters.
FOUR_O_PROMPT_LIMIT = int(os.getenv("FOUR_O_PROMPT_LIMIT", "4000"))
FOUR_O_RESPONSE_LIMIT = int(os.getenv("FOUR_O_RESPONSE_LIMIT", "1000"))
FOUR_O_EDITOR_MAX_TOKENS = int(os.getenv("FOUR_O_EDITOR_MAX_TOKENS", "2000"))
FOUR_O_PITCH_MAX_TOKENS = int(os.getenv("FOUR_O_PITCH_MAX_TOKENS", "200"))

# Track OpenAI usage against a daily budget.  OpenAI resets usage at midnight UTC.
FOUR_O_DAILY_TOKEN_LIMIT = int(os.getenv("FOUR_O_DAILY_TOKEN_LIMIT", "1000000"))

FOUR_O_TRACKED_MODELS: tuple[str, str] = ("gpt-4o", "gpt-4o-mini")


def _current_utc_date() -> date:
    return datetime.now(timezone.utc).date()


_four_o_usage_state = {
    "date": _current_utc_date(),
    "total": 0,
    "used": 0,
    "models": {model: 0 for model in FOUR_O_TRACKED_MODELS},
}


def _reset_four_o_usage_state(today: date) -> None:
    _four_o_usage_state["date"] = today
    _four_o_usage_state["total"] = 0
    _four_o_usage_state["used"] = 0
    _four_o_usage_state["models"] = {model: 0 for model in FOUR_O_TRACKED_MODELS}


def _ensure_four_o_usage_state(current_date: date | None = None) -> None:
    today = current_date or _current_utc_date()
    if _four_o_usage_state.get("date") != today:
        _reset_four_o_usage_state(today)


def _get_four_o_usage_snapshot() -> dict[str, Any]:
    _ensure_four_o_usage_state()
    models = dict(_four_o_usage_state.get("models", {}))
    for model in FOUR_O_TRACKED_MODELS:
        models.setdefault(model, 0)
    return {
        "date": _four_o_usage_state.get("date"),
        "total": _four_o_usage_state.get("total", 0),
        "used": _four_o_usage_state.get("used", 0),
        "models": models,
    }


def _record_four_o_usage(
    operation: str,
    model: str,
    usage: Mapping[str, Any] | None,
) -> int:
    limit = max(FOUR_O_DAILY_TOKEN_LIMIT, 0)
    today = _current_utc_date()
    _ensure_four_o_usage_state(today)
    usage_data: Mapping[str, Any] = usage or {}

    def _coerce_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    prompt_tokens = _coerce_int(usage_data.get("prompt_tokens"))
    completion_tokens = _coerce_int(usage_data.get("completion_tokens"))
    total_tokens = _coerce_int(usage_data.get("total_tokens"))

    if prompt_tokens is None:
        prompt_tokens = _coerce_int(usage_data.get("input_tokens"))
    if completion_tokens is None:
        completion_tokens = _coerce_int(usage_data.get("output_tokens"))

    extra_tokens = 0
    for key, value in usage_data.items():
        if key in {"total_tokens", "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}:
            continue
        if "tokens" not in key:
            continue
        value_int = _coerce_int(value)
        if value_int is None:
            continue
        extra_tokens += max(value_int, 0)

    if total_tokens is not None:
        spent = max(total_tokens, 0)
    else:
        spent = max(
            (prompt_tokens or 0)
            + (completion_tokens or 0)
            + extra_tokens,
            0,
        )
        if spent:
            total_tokens = spent
    models = _four_o_usage_state.setdefault("models", {})
    models.setdefault(model, 0)
    models[model] += spent
    new_total = _four_o_usage_state.get("total", 0) + spent
    _four_o_usage_state["total"] = new_total
    previous_used = _four_o_usage_state.get("used", 0)
    new_used = previous_used + spent
    if limit:
        new_used = min(new_used, limit)
        remaining = max(limit - new_used, 0)
    else:
        new_used = 0
        remaining = 0
    _four_o_usage_state["used"] = new_used
    logging.info(
        "four_o.usage op=%s model=%s spent=%d remaining=%d/%d day_total=%d model_total=%d prompt=%d completion=%d total=%d",
        operation,
        model,
        spent,
        remaining,
        limit,
        new_total,
        models[model],
        int(prompt_tokens or 0),
        int(completion_tokens or 0),
        int(total_tokens or 0),
    )
    return remaining


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
        except Exception as e:
            msg = str(e)
            m = re.search(r"Flood control exceeded.*Retry in (\d+) seconds", msg, re.I)
            if m and attempt < retries - 1:
                wait = int(m.group(1)) + 1
                logging.warning(
                    "telegraph_call flood wait=%ss attempt=%d", wait, attempt + 1
                )
                await asyncio.sleep(wait)
                continue
            raise

    # If we exit the loop without returning or raising, raise the last exception
    if last_exc:
        raise TelegraphException("Telegraph request failed") from last_exc


async def telegraph_create_page(
    tg: Telegraph, *args, caller: str = "event_pipeline", eid: int | None = None, **kwargs
):
    kwargs.setdefault("author_name", TELEGRAPH_AUTHOR_NAME)
    if TELEGRAPH_AUTHOR_URL:
        kwargs.setdefault("author_url", TELEGRAPH_AUTHOR_URL)
    if eid is not None and caller != "event_pipeline":
        logging.error(
            "AGGREGATE_SHOULD_NOT_TOUCH_EVENTS create caller=%s eid=%s", caller, eid
        )
        return {}
    res = await telegraph_call(tg.create_page, *args, **kwargs)
    path = res.get("path") if isinstance(res, dict) else ""
    logging.info(
        "telegraph_create_page author=%s url=%s path=%s caller=%s eid=%s",
        kwargs.get("author_name"),
        kwargs.get("author_url"),
        path,
        caller,
        eid,
    )
    return res


async def telegraph_edit_page(
    tg: Telegraph,
    path: str,
    *,
    caller: str = "event_pipeline",
    eid: int | None = None,
    **kwargs,
):
    kwargs.setdefault("author_name", TELEGRAPH_AUTHOR_NAME)
    if TELEGRAPH_AUTHOR_URL:
        kwargs.setdefault("author_url", TELEGRAPH_AUTHOR_URL)
    if eid is not None and caller != "event_pipeline":
        logging.error(
            "AGGREGATE_SHOULD_NOT_TOUCH_EVENTS edit caller=%s eid=%s", caller, eid
        )
        return {}
    logging.info(
        "telegraph_edit_page author=%s url=%s path=%s caller=%s eid=%s",
        kwargs.get("author_name"),
        kwargs.get("author_url"),
        path,
        caller,
        eid,
    )
    try:
        return await telegraph_call(tg.edit_page, path, **kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "author_name" in msg or "author_url" in msg:
            kwargs.pop("author_name", None)
            kwargs.pop("author_url", None)
            return await telegraph_call(tg.edit_page, path, **kwargs)
        raise


def seconds_to_next_minute(now: datetime) -> float:
    next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    return (next_minute - now).total_seconds()


# main menu buttons
MENU_ADD_EVENT = "\u2795 Добавить событие"
MENU_ADD_FESTIVAL = "\u2795 Добавить фестиваль"
MENU_EVENTS = "\U0001f4c5 События"
VK_BTN_ADD_SOURCE = "\u2795 Добавить сообщество"
VK_BTN_LIST_SOURCES = "\U0001f4cb Показать список сообществ"
VK_BTN_CHECK_EVENTS = "\U0001f50e Проверить события"
VK_BTN_QUEUE_SUMMARY = "\U0001f4ca Сводка очереди"

# command help descriptions by role
# roles: guest (not registered), user (registered), superadmin
HELP_COMMANDS = [
    {
        "usage": "/help",
        "desc": "Show available commands for your role",
        "roles": {"guest", "user", "superadmin"},
    },
    {
        "usage": "/start",
        "desc": "Register the first user as superadmin or display status",
        "roles": {"guest", "user", "superadmin"},
    },
    {
        "usage": "/register",
        "desc": "Request moderator access",
        "roles": {"guest"},
    },
    {
        "usage": "/menu",
        "desc": "Show main menu",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "🎪 Сделать фестиваль",
        "desc": "Кнопка в меню редактирования события предложит создать или привязать фестиваль",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/addevent <text>",
        "desc": "Parse text with model 4o and store events",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/addevent_raw <title>|<date>|<time>|<location>",
        "desc": "Add event without LLM",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/events [DATE]",
        "desc": "List events for the day",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/exhibitions",
        "desc": "List active exhibitions",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/digest",
        "desc": "Build digest preview for lectures and master-classes",
        "roles": {"superadmin"},
    },
    {
        "usage": "/pages",
        "desc": "Show Telegraph month and weekend pages",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/pages_rebuild [YYYY-MM[,YYYY-MM...]] [--past=0] [--future=2] [--force]",
        "desc": "Rebuild Telegraph pages manually",
        "roles": {"superadmin"},
    },
    {
        "usage": "/fest",
        "desc": "List festivals with edit options",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/vklink <event_id> <VK post link>",
        "desc": "Attach VK post link to an event",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/vk",
        "desc": "VK Intake: add/list sources, check/review events, and open queue summary",
        "roles": {"superadmin"},
    },
    {
        "usage": "/vk_queue",
        "desc": "Show VK inbox summary (pending/locked/skipped/imported/rejected) and a \"🔎 Проверить события\" button to start the review flow",
        "roles": {"superadmin"},
    },
    {
        "usage": "/vk_crawl_now",
        "desc": "Run VK crawling now (admin only); reports \"добавлено N, всего M\" to the admin chat",
        "roles": {"superadmin"},
    },
    {
        "usage": "↪️ Репостнуть в Vk",
        "desc": "Опубликовать пост с фото по ID",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "✂️ Сокращённый рерайт",
        "desc": "LLM-сжатый текст без фото, предпросмотр и правка перед публикацией",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/requests",
        "desc": "Review pending registrations",
        "roles": {"superadmin"},
    },
    {
        "usage": "/tourist_export [period]",
        "desc": "Export events with tourist_* fields to JSONL",
        "roles": {"user", "superadmin"},
    },
    {
        "usage": "/tz <±HH:MM>",
        "desc": "Set timezone offset",
        "roles": {"superadmin"},
    },
    {
        "usage": "/setchannel",
        "desc": "Register announcement or asset channel",
        "roles": {"superadmin"},
    },
    {
        "usage": "/channels",
        "desc": "List admin channels",
        "roles": {"superadmin"},
    },
    {
        "usage": "/regdailychannels",
        "desc": "Choose channels for daily announcements",
        "roles": {"superadmin"},
    },
    {
        "usage": "/daily",
        "desc": "Manage daily announcement channels",
        "roles": {"superadmin"},
    },
    {
        "usage": "/images",
        "desc": "Toggle uploading photos to Catbox",
        "roles": {"superadmin"},
    },
    {
        "usage": "/vkgroup <id|off>",
        "desc": "Set or disable VK group",
        "roles": {"superadmin"},
    },
    {
        "usage": "/vktime today|added <HH:MM>",
        "desc": "Change VK posting times",
        "roles": {"superadmin"},
    },
    {
        "usage": "/vkphotos",
        "desc": "Toggle VK photo posting",
        "roles": {"superadmin"},
    },
    {
        "usage": "/captcha <code>",
        "desc": "Submit VK captcha code",
        "roles": {"superadmin"},
    },
    {
        "usage": "/ask4o <text>",
        "desc": "Send query to model 4o",
        "roles": {"superadmin"},
    },
    {
        "usage": "/ocrtest",
        "desc": "сравнить распознавание афиш",
        "roles": {"superadmin"},
    },
    {
        "usage": "/stats [events]",
        "desc": "Show Telegraph view counts",
        "roles": {"superadmin"},
    },
    {
        "usage": "/status [job_id]",
        "desc": "Show uptime and job status",
        "roles": {"superadmin"},
    },
    {
        "usage": "/trace <run_id>",
        "desc": "Show recent log trace",
        "roles": {"superadmin"},
    },
    {
        "usage": "/last_errors [N]",
        "desc": "Show last N errors",
        "roles": {"superadmin"},
    },
    {
        "usage": "/debug queue",
        "desc": "Show background job counts",
        "roles": {"superadmin"},
    },
    {
        "usage": "/mem",
        "desc": "Show memory usage",
        "roles": {"superadmin"},
    },
    {
        "usage": "/festivals_fix_nav",
        "desc": "Fix festival navigation links",
        "roles": {"superadmin"},
    },
    {
        "usage": "/users",
        "desc": "List users and roles",
        "roles": {"superadmin"},
    },
    {
        "usage": "/dumpdb",
        "desc": "Download database dump",
        "roles": {"superadmin"},
    },
    {
        "usage": "/restore",
        "desc": "Restore database from dump",
        "roles": {"superadmin"},
    },
]

HELP_COMMANDS.insert(
    0,
    {
        "usage": "/weekendimg",
        "desc": (
            "Добавить обложку к странице выходных или лендинга фестивалей: "
            "выбрать дату/лендинг и загрузить фото в Catbox"
        ),
        "roles": {"superadmin"},
    },
)


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

        truthy_aliases = {"1", "true", "t", "on", "yes"}
        falsy_aliases = {"0", "false", "f", "off", "no"}

        desired_value: str | None = None
        should_update = False
        enabled = False

        if not row:
            desired_value = "1"
            enabled = True
            should_update = True
        else:
            raw_value = row[0]
            normalized = (raw_value or "").strip()
            lowered = normalized.lower()

            if not normalized or lowered in truthy_aliases:
                desired_value = "1"
                enabled = True
                should_update = normalized != desired_value
            elif lowered in falsy_aliases:
                desired_value = "0"
                enabled = False
                should_update = normalized != desired_value
            else:
                enabled = lowered in truthy_aliases

        if desired_value is not None and should_update:
            await conn.execute(
                "INSERT OR REPLACE INTO setting(key, value) VALUES('catbox_enabled', ?)",
                (desired_value,),
            )
            await conn.commit()

    return enabled


async def set_catbox_enabled(db: Database, value: bool):
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO setting(key, value) VALUES('catbox_enabled', ?)",
            ("1" if value else "0",),
        )
        await conn.commit()
    global CATBOX_ENABLED
    CATBOX_ENABLED = value
    logging.info("CATBOX_ENABLED set to %s", CATBOX_ENABLED)


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


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive parameters like access tokens."""
    redacted: dict[str, Any] = {}
    for k, v in params.items():
        redacted[k] = "<redacted>" if "token" in k else v
    return redacted


def _vk_user_token() -> str | None:
    """Return user token unless it was previously marked invalid."""
    token = os.getenv("VK_USER_TOKEN")
    global _vk_user_token_bad
    if token and _vk_user_token_bad and token != _vk_user_token_bad:
        _vk_user_token_bad = None
    if token and token != _vk_user_token_bad:
        return token
    return None


async def vk_api(method: str, **params: Any) -> Any:
    """Simple VK API GET request with token and version."""
    service_allowed = method in VK_SERVICE_READ_METHODS or any(
        method.startswith(prefix) for prefix in VK_SERVICE_READ_PREFIXES
    )
    token: str | None = None
    kind: str | None = None
    if VK_READ_VIA_SERVICE and VK_SERVICE_TOKEN and service_allowed:
        token = VK_SERVICE_TOKEN
        kind = "service"
    else:
        if VK_USER_TOKEN:
            token = VK_USER_TOKEN
            kind = "user"
        elif VK_TOKEN:
            token = VK_TOKEN
            kind = "group"
        elif VK_TOKEN_AFISHA:
            token = VK_TOKEN_AFISHA
            kind = "group"
    if not token:
        raise VKAPIError(None, "VK token not set", method=method)
    redacted_token = redact_token(token)
    call_params = params.copy()
    call_params["access_token"] = token
    call_params["v"] = VK_API_VERSION
    async with VK_SEMAPHORE:
        await _vk_throttle()
        resp = await http_call(
            f"vk.{method}",
            "GET",
            f"https://api.vk.com/method/{method}",
            timeout=HTTP_TIMEOUT,
            params=call_params,
        )
    logging.info("vk.actor=%s method=%s", kind or "unknown", method)
    data = resp.json()
    if "error" in data:
        err = data["error"]
        logging.error(
            "VK API error: method=%s code=%s msg=%s params=%s actor=%s token=%s",
            method,
            err.get("error_code"),
            err.get("error_msg"),
            redact_params(call_params),
            kind,
            redacted_token,
        )
        raise VKAPIError(
            err.get("error_code"),
            err.get("error_msg", ""),
            err.get("captcha_sid"),
            err.get("captcha_img"),
            method,
            actor=kind,
            token=redacted_token,
        )
    return data.get("response")


_VK_URL_RE = re.compile(r"(?:https?://)?(?:www\.)?vk\.com/([^/?#]+)")


async def vk_resolve_group(screen_or_url: str) -> tuple[int, str, str]:
    """Return (group_id, name, screen_name) for a VK community."""
    raw = (screen_or_url or "").strip()
    m = _VK_URL_RE.search(raw)
    screen = m.group(1) if m else raw.lstrip("@/")

    if screen.startswith(("club", "public")) and screen[len("club"):].isdigit():
        screen = screen.split("b", 1)[-1] if screen.startswith("club") else screen.split("c", 1)[-1]

    gid: int | None = None
    try:
        rs = await vk_api("utils.resolveScreenName", screen_name=screen)
        if rs and rs.get("type") == "group" and int(rs.get("object_id", 0)) > 0:
            gid = int(rs["object_id"])
    except Exception:
        pass

    try:
        arg = gid if gid is not None else screen
        gb = await vk_api("groups.getById", group_ids=arg, fields="screen_name")
        resp = gb if isinstance(gb, list) else (gb.get("groups") or [gb])
        if not isinstance(resp, list) or not resp:
            raise ValueError("Empty response from groups.getById")
        g = resp[0]
        group_id = int(g["id"])
        name = g.get("name") or str(group_id)
        screen_name = g.get("screen_name") or screen
        return group_id, name, screen_name
    except Exception as e:
        logging.error("vk_resolve_group failed: %s", e)
        raise


def _pick_biggest_photo(photo: dict) -> str | None:
    sizes = photo.get("sizes") or []
    if not sizes:
        return None
    best = max(sizes, key=lambda s: s.get("width", 0))
    return best.get("url")


def _extract_post_photos(post: dict) -> list[str]:
    photos: list[str] = []
    for att in post.get("attachments", []):
        if att.get("type") == "photo":
            url = _pick_biggest_photo(att["photo"])
            if url:
                photos.append(url)
    return photos


async def vk_wall_since(
    group_id: int, since_ts: int, *, count: int = 100, offset: int = 0
) -> list[dict]:
    """Return wall posts for a group since timestamp.

    ``count`` and ``offset`` are forwarded to :func:`wall.get` allowing
    pagination.
    """
    resp = await vk_api(
        "wall.get",
        owner_id=-group_id,
        count=count,
        offset=offset,
        filter="owner",
    )
    items = resp.get("items", []) if isinstance(resp, dict) else resp["items"]
    posts: list[dict] = []
    for item in items:
        if item.get("date", 0) < since_ts:
            continue
        src = item.get("copy_history", [item])[0]
        photos = _extract_post_photos(src)
        posts.append(
            {
                "group_id": group_id,
                "post_id": item["id"],
                "date": item["date"],
                "text": src.get("text", ""),
                "photos": photos,
                "url": f"https://vk.com/wall-{group_id}_{item['id']}",
            }
        )
    posts.sort(key=lambda p: (p["date"], p["post_id"]), reverse=True)
    return posts


class VKAPIError(Exception):
    """Exception raised for VK API errors."""

    def __init__(
        self,
        code: int | None,
        message: str,
        captcha_sid: str | None = None,
        captcha_img: str | None = None,
        method: str | None = None,
        actor: str | None = None,
        token: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.method = method
        # additional info for captcha challenge
        self.captcha_sid = captcha_sid
        self.captcha_img = captcha_img
        self.actor = actor
        self.token = token
        super().__init__(message)


class VKPermissionError(VKAPIError):
    """Raised when VK posting is blocked and no fallback token is available."""


async def _vk_api(
    method: str,
    params: dict,
    db: Database | None = None,
    bot: Bot | None = None,
    token: str | None = None,
    token_kind: str = "group",
    skip_captcha: bool = False,
) -> dict:
    """Call VK API with token fallback."""
    global _vk_captcha_needed, _vk_captcha_sid, _vk_captcha_img, _vk_captcha_method, _vk_captcha_params
    if _vk_captcha_needed and not skip_captcha:
        raise VKAPIError(14, "Captcha needed", _vk_captcha_sid, _vk_captcha_img, method)
    orig_params = dict(params)
    tokens: list[tuple[str, str]] = []
    if token:
        tokens.append((token_kind, token))
    else:
        user_token = _vk_user_token()
        group_token = VK_TOKEN
        mode = VK_ACTOR_MODE
        now = _time.time()
        if mode == "group":
            if group_token:
                tokens.append(("group", group_token))
        elif mode == "user":
            if user_token:
                tokens.append(("user", user_token))
        elif mode == "auto":
            auto_methods = ("wall.post", "wall.edit", "wall.getById")
            blocked_until = vk_group_blocked.get(method, 0.0)
            group_allowed = not group_token or now >= blocked_until
            if any(method.startswith(m) for m in auto_methods) or method.startswith("photos."):
                if group_token:
                    if group_allowed:
                        tokens.append(("group", group_token))
                    else:
                        logging.info(
                            "vk.actor=skip group reason=circuit method=%s", method
                        )
                if user_token:
                    tokens.append(("user", user_token))
            else:
                if user_token:
                    tokens.append(("user", user_token))
                if group_token:
                    if group_allowed:
                        tokens.append(("group", group_token))
                    else:
                        logging.info(
                            "vk.actor=skip group reason=circuit method=%s", method
                        )
        else:
            if user_token:
                tokens.append(("user", user_token))
            if group_token:
                tokens.append(("group", group_token))
        if not tokens and mode == "auto" and method == "wall.post" and blocked_until > now and not user_token:
            raise VKPermissionError(None, "permission error")
    last_err: dict | None = None
    last_actor: str | None = None
    last_token: str | None = None
    session = get_vk_session()
    fallback_next = False
    for idx, (kind, token) in enumerate(tokens):
        call_params = orig_params.copy()
        call_params["access_token"] = token
        call_params["v"] = "5.131"
        redacted_token = redact_token(token)
        actor_msg = f"vk.actor={kind}"
        if kind == "user" and fallback_next:
            actor_msg += " (fallback)"
        logging.info("%s method=%s", actor_msg, method)
        logging.info(
            "calling VK API %s using %s token %s", method, kind, redacted_token
        )
        async def _call():
            resp = await http_call(
                f"vk.{method}",
                "POST",
                f"https://api.vk.com/method/{method}",
                timeout=HTTP_TIMEOUT,
                data=call_params,
            )
            return resp.json()

        err: dict | None = None
        last_msg: str | None = None
        for attempt, delay in enumerate(BACKOFF_DELAYS, start=1):
            async with VK_SEMAPHORE:
                await _vk_throttle()
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
            logging.error(
                "VK API error: method=%s code=%s msg=%s params=%s actor=%s token=%s",
                method,
                err.get("error_code"),
                err.get("error_msg"),
                redact_params(call_params),
                kind,
                redacted_token,
            )
            msg = err.get("error_msg")
            if not isinstance(msg, str):
                msg = "" if msg is None else str(msg)
            msg_l = msg.lower()
            code = err.get("error_code")
            if code == 14:
                _vk_captcha_needed = True
                _vk_captcha_sid = err.get("captcha_sid")
                _vk_captcha_img = err.get("captcha_img")
                _vk_captcha_method = method
                _vk_captcha_params = orig_params.copy()
                logging.warning(
                    "vk captcha sid=%s method=%s params=%s",
                    _vk_captcha_sid,
                    method,
                    str(orig_params)[:200],
                )
                if db and bot:
                    await notify_vk_captcha(db, bot, _vk_captcha_img)
                # surface captcha details to caller
                raise VKAPIError(
                    code,
                    msg,
                    _vk_captcha_sid,
                    _vk_captcha_img,
                    method,
                    actor=kind,
                    token=redacted_token,
                )
            if code == 15 and "edit time expired" in msg_l:
                logging.info("vk no-retry error code=15: %s", msg)
                break
            if kind == "user" and code in {5, 27}:
                global _vk_user_token_bad
                if _vk_user_token_bad != token:
                    _vk_user_token_bad = token
                    if db and bot:
                        await notify_superadmin(db, bot, "VK_USER_TOKEN expired")
                break
            if any(x in msg_l for x in ("already deleted", "already exists")):
                logging.info("vk no-retry error: %s", msg)
                return data
            last_msg = msg
            perm_error = (
                idx == 0
                and len(tokens) > 1
                and kind == "group"
                and VK_ACTOR_MODE == "auto"
                and (
                    code in VK_FALLBACK_CODES
                    or "method is unavailable with group auth" in msg_l
                    or "access to adding post denied" in msg_l
                    or "access denied" in msg_l
                )
            )
            if perm_error:
                vk_fallback_group_to_user_total[method] += 1
                expires = _time.time() + VK_CB_TTL
                vk_group_blocked[method] = expires
                logging.info(
                    "vk.circuit[%s]=blocked, until=%s",
                    method,
                    datetime.fromtimestamp(expires, timezone.utc).isoformat(),
                )
                last_err = err
                last_actor = kind
                last_token = redacted_token
                fallback_next = True
                break
            if attempt == len(BACKOFF_DELAYS):
                logging.warning(
                    "vk api %s failed after %d attempts: %s",
                    method,
                    attempt,
                    msg,
                )
                break
            await asyncio.sleep(delay)
        if err:
            if fallback_next and kind == "group":
                continue
            code = err.get("error_code")
            raise VKAPIError(
                code,
                err.get("error_msg", ""),
                err.get("captcha_sid"),
                err.get("captcha_img"),
                method,
                actor=kind,
                token=redacted_token,
            )
        break
    if last_err:
        raise VKAPIError(
            last_err.get("error_code"),
            last_err.get("error_msg", ""),
            last_err.get("captcha_sid"),
            last_err.get("captcha_img"),
            method,
            actor=last_actor,
            token=last_token,
        )
    raise VKAPIError(None, "VK token missing", method=method)


async def upload_vk_photo(
    group_id: str,
    url: str,
    db: Database | None = None,
    bot: Bot | None = None,
    *,
    token: str | None = None,
    token_kind: str = "group",
) -> str | None:
    """Upload an image to VK and return attachment id."""
    if not url:
        return None
    try:
        owner_id = -int(group_id.lstrip("-"))
        if token:
            actors = [VkActor(token_kind, token, f"{token_kind}:explicit")]
        else:
            actors = choose_vk_actor(owner_id, "photos.getWallUploadServer")
        if not actors:
            raise VKAPIError(None, "VK token missing", method="photos.getWallUploadServer")
        if all(actor.kind == "group" for actor in actors):
            logging.info(
                "vk.upload skipped owner_id=%s reason=user_token_required",
                owner_id,
            )
            return None
        for idx, actor in enumerate(actors, start=1):
            logging.info(
                "vk.call method=photos.getWallUploadServer owner_id=%s try=%d/%d actor=%s",
                owner_id,
                idx,
                len(actors),
                actor.label,
            )
            token = actor.token if actor.kind == "group" else VK_USER_TOKEN
            try:
                if DEBUG:
                    mem_info("VK upload before")
                data = await _vk_api(
                    "photos.getWallUploadServer",
                    {"group_id": group_id.lstrip("-")},
                    db,
                    bot,
                    token=token,
                    token_kind=actor.kind,
                    skip_captcha=(actor.kind == "group"),
                )
                upload_url = data["response"]["upload_url"]
                session = get_http_session()

                async def _download():
                    async with span("http"):
                        async with HTTP_SEMAPHORE:
                            async with session.get(url) as resp:
                                resp.raise_for_status()
                                header_length = resp.headers.get("Content-Length")
                                if resp.content_length and resp.content_length > MAX_DOWNLOAD_SIZE:
                                    raise ValueError("file too large")
                                buf = bytearray()
                                async for chunk in resp.content.iter_chunked(64 * 1024):
                                    buf.extend(chunk)
                                    if len(buf) > MAX_DOWNLOAD_SIZE:
                                        raise ValueError("file too large")
                                data = bytes(buf)
                                if header_length:
                                    try:
                                        expected_size = int(header_length)
                                    except ValueError as exc:
                                        raise ValueError("invalid Content-Length header") from exc
                                    if expected_size != len(data):
                                        raise ValueError("content-length mismatch")
                                if detect_image_type(data) == "jpeg":
                                    validate_jpeg_markers(data)
                                return data

                img_bytes = await asyncio.wait_for(_download(), HTTP_TIMEOUT)
                img_bytes, _ = ensure_jpeg(img_bytes, "image.jpg")
                if detect_image_type(img_bytes) == "jpeg":
                    validate_jpeg_markers(img_bytes)
                form = FormData()
                form.add_field(
                    "photo",
                    img_bytes,
                    filename="image.jpg",
                    content_type="image/jpeg",
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
                    token_kind=actor.kind,
                    skip_captcha=(actor.kind == "group"),
                )
                info = save["response"][0]
                if DEBUG:
                    mem_info("VK upload after")
                return f"photo{info['owner_id']}_{info['id']}"
            except VKAPIError as e:
                logging.warning(
                    "vk.upload error actor=%s token=%s code=%s msg=%s",
                    e.actor,
                    e.token,
                    e.code,
                    e.message,
                )
                msg_l = (e.message or "").lower()
                perm = (
                    e.code in VK_FALLBACK_CODES
                    or "method is unavailable with group auth" in msg_l
                    or "access denied" in msg_l
                )
                if idx < len(actors) and perm:
                    logging.info(
                        "vk.retry reason=%s actor_next=%s",
                        e.code or e.message,
                        actors[idx].label,
                    )
                    continue
                raise
        return None
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
    photo_urls: list[str] | None = None,
    website_url: str | None = None,
    program_url: str | None = None,
    ticket_url: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    location_name: str | None = None,
    location_address: str | None = None,
    city: str | None = None,
    source_text: str | None = None,
    source_post_url: str | None = None,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
) -> tuple[Festival, bool, bool]:
    """Return festival and flags (created, updated)."""
    async with db.get_session() as session:
        res = await session.execute(select(Festival).where(Festival.name == name))
        fest = res.scalar_one_or_none()
        if fest:
            updated = False
            url_updates = {
                "website_url": website_url.strip() if website_url else None,
                "program_url": program_url.strip() if program_url else None,
                "ticket_url": ticket_url.strip() if ticket_url else None,
            }
            if photo_urls:
                merged = fest.photo_urls[:]
                for u in photo_urls:
                    if u not in merged:
                        merged.append(u)
                if merged != fest.photo_urls:
                    fest.photo_urls = merged
                    updated = True
            if not fest.photo_url:
                if photo_url:
                    fest.photo_url = photo_url
                    updated = True
                elif photo_urls:
                    fest.photo_url = photo_urls[0]
                    updated = True
            for field, value in url_updates.items():
                if value and value != getattr(fest, field):
                    setattr(fest, field, value)
                    updated = True
            if full_name and full_name != fest.full_name:
                fest.full_name = full_name
                updated = True
            if start_date and start_date != fest.start_date:
                fest.start_date = start_date
                updated = True
            if end_date and end_date != fest.end_date:
                fest.end_date = end_date
                updated = True
            if location_name and location_name != fest.location_name:
                fest.location_name = location_name
                updated = True
            if location_address and location_address != fest.location_address:
                fest.location_address = location_address
                updated = True
            if city and city.strip() and city.strip() != (fest.city or "").strip():
                fest.city = city.strip()
                updated = True
            if source_text and source_text != fest.source_text:
                fest.source_text = source_text
                updated = True
            if source_post_url and source_post_url != fest.source_post_url:
                fest.source_post_url = source_post_url
                updated = True
            if source_chat_id and source_chat_id != fest.source_chat_id:
                fest.source_chat_id = source_chat_id
                updated = True
            if source_message_id and source_message_id != fest.source_message_id:
                fest.source_message_id = source_message_id
                updated = True
            if updated:
                session.add(fest)
                await session.commit()
                await rebuild_fest_nav_if_changed(db)
            return fest, False, updated
        fest = Festival(
            name=name,
            full_name=full_name,
            photo_url=photo_url or (photo_urls[0] if photo_urls else None),
            photo_urls=photo_urls or ([photo_url] if photo_url else []),
            website_url=website_url.strip() if website_url else None,
            program_url=program_url.strip() if program_url else None,
            ticket_url=ticket_url.strip() if ticket_url else None,
            start_date=start_date,
            end_date=end_date,
            location_name=location_name,
            location_address=location_address,
            city=city,
            source_text=source_text,
            source_post_url=source_post_url,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
            created_at=datetime.now(timezone.utc),
        )
        session.add(fest)
        await session.commit()
        logging.info("created festival %s", name)
        await rebuild_fest_nav_if_changed(db)
        return fest, True, True


def _festival_admin_url(fest_id: int | None) -> str | None:
    """Return admin URL for the festival if environment is configured."""

    if not fest_id:
        return None
    template = os.getenv("FEST_ADMIN_URL_TEMPLATE")
    if template:
        try:
            return template.format(id=fest_id)
        except Exception:
            logging.exception("failed to format FEST_ADMIN_URL_TEMPLATE", extra={"id": fest_id})
            return None
    base = os.getenv("FEST_ADMIN_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/{fest_id}"
    return None


def _festival_location_text(fest: Festival) -> str:
    parts = []
    if fest.location_name:
        parts.append(fest.location_name)
    if fest.location_address:
        parts.append(fest.location_address)
    return " — ".join(parts) if parts else "—"


def _festival_period_text(fest: Festival) -> str:
    start = (fest.start_date or "").strip() if fest.start_date else ""
    end = (fest.end_date or "").strip() if fest.end_date else ""
    if start and end:
        if start == end:
            return start
        return f"{start} — {end}"
    return start or end or "—"


def _festival_photo_count(fest: Festival) -> int:
    urls = [u for u in (fest.photo_urls or []) if u]
    if not urls and fest.photo_url:
        return 1
    if fest.photo_url and fest.photo_url not in urls:
        urls.append(fest.photo_url)
    return len(urls)


def _festival_telegraph_url(fest: Festival) -> str | None:
    if fest.telegraph_url:
        return normalize_telegraph_url(fest.telegraph_url)
    if fest.telegraph_path:
        return normalize_telegraph_url(f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}")
    return None


async def _build_makefest_response(
    db: Database, fest: Festival, *, status: str, photo_count: int
) -> tuple[str, types.InlineKeyboardMarkup | None]:
    telegraph_url = _festival_telegraph_url(fest)
    lines = [
        f"✅ Фестиваль {status} и привязан",
        "",
        f"ID: {fest.id if fest.id is not None else '—'}",
        f"Название: {fest.name}",
        f"Полное название: {fest.full_name or '—'}",
        f"Период: {_festival_period_text(fest)}",
        f"Город: {(fest.city or '—').strip() or '—'}",
        f"Локация: {_festival_location_text(fest)}",
        f"Фото добавлено: {photo_count}",
        f"Telegraph: {telegraph_url or '—'}",
        "",
        "Событие привязано к фестивалю.",
    ]

    buttons: list[types.InlineKeyboardButton] = []
    admin_url = _festival_admin_url(fest.id)
    if admin_url:
        buttons.append(types.InlineKeyboardButton(text="Админка", url=admin_url))
    landing_url = await get_setting_value(db, "festivals_index_url") or await get_setting_value(
        db, "fest_index_url"
    )
    if landing_url:
        buttons.append(types.InlineKeyboardButton(text="Лендинг", url=landing_url))
    markup = (
        types.InlineKeyboardMarkup(inline_keyboard=[buttons]) if buttons else None
    )
    return "\n".join(lines), markup


async def extract_telegra_ph_cover_url(
    page_url: str, *, event_id: str | int | None = None
) -> str | None:
    """Return first image from a Telegraph page.

    Besides ``/file/...`` paths (which are rewritten to ``https://telegra.ph``),
    this helper now also accepts absolute ``https://`` links pointing to
    external hosts such as ``catbox``. Only typical image extensions are
    allowed; if the extension is unknown, a ``HEAD`` request is made to verify
    that the ``Content-Type`` starts with ``image/``.
    """
    url = page_url.split("#", 1)[0].split("?", 1)[0]
    cached = telegraph_first_image.get(url)
    if cached is not None:
        logging.info(
            "digest.cover.fetch event_id=%s result=found url=%s source=cache took_ms=0",
            event_id,
            cached,
        )
        return cached
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host not in {"telegra.ph", "te.legra.ph"}:
        return None
    path = parsed.path.lstrip("/")
    if not path:
        return None
    api_url = f"https://api.telegra.ph/getPage/{path}?return_content=true"
    timeout = httpx.Timeout(HTTP_TIMEOUT)
    start = _time.monotonic()
    for _ in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("result", {}).get("content") or []

            async def norm(src: str | None) -> str | None:
                if not src:
                    return None
                src = src.split("#", 1)[0].split("?", 1)[0]
                if src.startswith("/file/"):
                    return f"https://telegra.ph{src}"
                parsed_src = urlparse(src)
                if parsed_src.scheme != "https":
                    return None
                lower = parsed_src.path.lower()
                if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
                    return src
                # Unknown extension; try HEAD to check content-type
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        head = await client.head(src)
                    ctype = head.headers.get("content-type", "")
                    if ctype.startswith("image/"):
                        return src
                except Exception:
                    pass
                return None

            async def dfs(nodes) -> str | None:
                for node in nodes:
                    if isinstance(node, dict):
                        tag = node.get("tag")
                        attrs = node.get("attrs") or {}
                        if tag == "img":
                            u = await norm(attrs.get("src"))
                            if u:
                                return u
                        if tag == "a":
                            u = await norm(attrs.get("href"))
                            if u:
                                return u
                        children = node.get("children") or []
                        found = await dfs(children)
                        if found:
                            return found
                return None

            cover = await dfs(content)
            duration_ms = int((_time.monotonic() - start) * 1000)
            if cover:
                telegraph_first_image[url] = cover
                logging.info(
                    "digest.cover.fetch event_id=%s result=found url=%s source=telegraph_api took_ms=%s",
                    event_id,
                    cover,
                    duration_ms,
                )
                return cover
            logging.info(
                "digest.cover.fetch event_id=%s result=none url='' source=telegraph_api took_ms=%s",
                event_id,
                duration_ms,
            )
            return None
        except Exception:
            await asyncio.sleep(1)
    duration_ms = int((_time.monotonic() - start) * 1000)
    logging.info(
        "digest.cover.fetch event_id=%s result=none url='' source=telegraph_api took_ms=%s",
        event_id,
        duration_ms,
    )
    return None


async def try_set_fest_cover_from_program(
    db: Database, fest: Festival, force: bool = False
) -> bool:
    """Fetch Telegraph cover and set festival.photo_url if missing."""
    if not force and fest.photo_url:
        log_festcover(
            logging.DEBUG,
            fest.id,
            "skip_existing_photo",
            force=force,
            current=fest.photo_url,
        )
        return False
    target_url = fest.program_url or _festival_telegraph_url(fest)
    cover = None
    skip_reason: str | None = None
    if target_url:
        cover = await extract_telegra_ph_cover_url(target_url)
        if not cover:
            skip_reason = "skip_no_cover_found"
    else:
        skip_reason = "skip_no_program_url"
    async with db.get_session() as session:
        fresh = await session.get(Festival, fest.id)
        if not fresh:
            log_festcover(
                logging.INFO,
                fest.id,
                "skip_festival_missing",
            )
            return False
        existing_photos = list(fresh.photo_urls or [])
        existing_set = set(existing_photos)
        event_cover_urls: list[str] = []
        if fresh.name:
            result = await session.execute(
                select(Event.telegraph_url, Event.photo_urls)
                .where(Event.festival == fresh.name)
                .order_by(Event.id)
            )
            seen_event_urls: set[str] = set()
            for telegraph_url, event_photos in result:
                candidate = next((url for url in (event_photos or []) if url), None)
                if not candidate and telegraph_url:
                    candidate = await extract_telegra_ph_cover_url(telegraph_url)
                if candidate and candidate not in seen_event_urls:
                    seen_event_urls.add(candidate)
                    event_cover_urls.append(candidate)

        candidate_urls: list[str] = []
        if cover:
            candidate_urls.append(cover)
        candidate_urls.extend(event_cover_urls)

        new_urls: list[str] = []
        seen_candidates: set[str] = set()
        for url in candidate_urls:
            if not url or url in seen_candidates:
                continue
            seen_candidates.add(url)
            if url in existing_set:
                continue
            new_urls.append(url)

        updated_photos = existing_photos
        photos_changed = False
        if new_urls:
            updated_photos = new_urls + existing_photos
            fresh.photo_urls = updated_photos
            photos_changed = True
        if cover and cover in existing_set and cover not in new_urls:
            log_festcover(
                logging.DEBUG,
                fest.id,
                "cover_already_listed",
                cover=cover,
            )

        selected_cover: str | None = None
        if cover:
            selected_cover = cover
        elif new_urls and (
            force or not fresh.photo_url or fresh.photo_url not in updated_photos
        ):
            selected_cover = new_urls[0]

        cover_changed = False
        if selected_cover and fresh.photo_url != selected_cover:
            fresh.photo_url = selected_cover
            cover_changed = True

        if photos_changed or cover_changed:
            await session.commit()
        success = bool(cover) or bool(new_urls)
    if success:
        log_festcover(
            logging.INFO,
            fest.id,
            "set_ok",
            cover=cover,
            target_url=target_url,
            new_event_covers=len(new_urls),
        )
        return True
    if skip_reason == "skip_no_program_url":
        log_festcover(
            logging.INFO,
            fest.id,
            "skip_no_program_url",
            force=force,
        )
    elif skip_reason == "skip_no_cover_found":
        log_festcover(
            logging.INFO,
            fest.id,
            "skip_no_cover_found",
            target_url=target_url,
        )
    else:
        log_festcover(
            logging.DEBUG,
            fest.id,
            "skip_no_updates",
            force=force,
        )
    return False


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

def _vk_captcha_quiet_until() -> datetime | None:
    if not VK_CAPTCHA_QUIET:
        return None
    try:
        start_s, end_s = VK_CAPTCHA_QUIET.split('-', 1)
        tz = ZoneInfo(VK_WEEK_EDIT_TZ)
        now = datetime.now(tz)
        start_t = datetime.strptime(start_s, '%H:%M').time()
        end_t = datetime.strptime(end_s, '%H:%M').time()
        start_dt = datetime.combine(now.date(), start_t, tz)
        end_dt = datetime.combine(now.date(), end_t, tz)
        if start_dt <= end_dt:
            if start_dt <= now < end_dt:
                return end_dt
        else:
            if now >= start_dt:
                return end_dt + timedelta(days=1)
            if now < end_dt:
                return end_dt
    except Exception:
        logging.exception('captcha quiet parse failed')
    return None




async def notify_vk_captcha(db: Database, bot: Bot, img_url: str | None):
    global _vk_captcha_requested_at
    admin_id = await get_superadmin_id(db)
    if not admin_id:
        return
    ttl = VK_CAPTCHA_TTL_MIN
    caption = f"Нужна капча для ВК. Введите код ниже (действует {ttl} минут)."
    buttons = [[types.InlineKeyboardButton(text="Ввести код", callback_data="captcha_input")]]
    quiet_until = _vk_captcha_quiet_until()
    if quiet_until:
        buttons.append(
            [
                types.InlineKeyboardButton(
                    text=f"Отложить до {quiet_until.strftime('%H:%M')}",
                    callback_data="captcha_delay",
                )
            ]
        )
    markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    try:
        if img_url:
            try:
                session = get_http_session()
                async with HTTP_SEMAPHORE:
                    async with session.get(img_url) as resp:
                        data = await resp.read()
                try:
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(data))
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG")
                    data = buf.getvalue()
                except Exception:
                    pass
                photo = types.BufferedInputFile(data, filename="vk_captcha.jpg")
                async with span("tg-send"):
                    await bot.send_photo(admin_id, photo, caption=caption, reply_markup=markup)
                _vk_captcha_requested_at = datetime.now(ZoneInfo(VK_WEEK_EDIT_TZ))
                logging.info("vk_captcha requested %s", _vk_captcha_sid)
                return
            except Exception:
                logging.exception("failed to download captcha image")
        text = caption
        if img_url:
            text = f"VK captcha needed: {img_url}\nUse /captcha <code>"
        async with span("tg-send"):
            await bot.send_message(admin_id, text, reply_markup=markup)
        _vk_captcha_requested_at = datetime.now(ZoneInfo(VK_WEEK_EDIT_TZ))
        logging.info("vk_captcha requested %s", _vk_captcha_sid)
    except Exception as e:  # pragma: no cover - network issues
        logging.error("failed to send vk captcha: %s", e)

async def handle_vk_captcha_prompt(callback: types.CallbackQuery, db: Database, bot: Bot):
    global _vk_captcha_awaiting_user
    await callback.answer()
    _vk_captcha_awaiting_user = callback.from_user.id
    remaining = VK_CAPTCHA_TTL_MIN
    if _vk_captcha_requested_at:
        elapsed = (
            datetime.now(ZoneInfo(VK_WEEK_EDIT_TZ)) - _vk_captcha_requested_at
        ).total_seconds()
        remaining = max(0, int(VK_CAPTCHA_TTL_MIN - elapsed // 60))
    await bot.send_message(
        callback.message.chat.id,
        f"Введите код с картинки (осталось {remaining} мин.)",
        reply_markup=types.ForceReply(),
    )


async def handle_vk_captcha_delay(callback: types.CallbackQuery, db: Database, bot: Bot):
    await callback.answer()
    quiet_until = _vk_captcha_quiet_until()
    if not quiet_until:
        return
    await bot.send_message(
        callback.message.chat.id,
        f"Отложено до {quiet_until.strftime('%H:%M')}",
    )
    delay = (quiet_until - datetime.now(ZoneInfo(VK_WEEK_EDIT_TZ))).total_seconds()
    async def _remind():
        await asyncio.sleep(max(0, delay))
        if _vk_captcha_needed and _vk_captcha_img:
            await notify_vk_captcha(db, bot, _vk_captcha_img)
    asyncio.create_task(_remind())


async def handle_vk_captcha_refresh(callback: types.CallbackQuery, db: Database, bot: Bot):
    await callback.answer()
    if _vk_captcha_method and _vk_captcha_params is not None:
        try:
            global _vk_captcha_needed
            _vk_captcha_needed = False
            await _vk_api(_vk_captcha_method, _vk_captcha_params, db, bot)
        except VKAPIError as e:
            logging.info(
                "vk_captcha refresh failed actor=%s token=%s code=%s msg=%s",
                e.actor,
                e.token,
                e.code,
                e.message,
            )
            if _vk_captcha_scheduler and _vk_captcha_key:
                vk_captcha_paused(_vk_captcha_scheduler, _vk_captcha_key)




def vk_captcha_paused(scheduler, key: str) -> None:
    """Register callback to resume VK jobs after captcha."""
    global _vk_captcha_resume, _vk_captcha_timeout, _vk_captcha_scheduler, _vk_captcha_key
    _vk_captcha_scheduler = scheduler
    _vk_captcha_key = key
    async def _resume():
        try:
            if scheduler.progress:
                scheduler.progress.finish_job(key, "done")
            if getattr(scheduler, "_remaining", None):
                scheduler._remaining.discard(key)  # type: ignore[attr-defined]
            await scheduler.run()
        except Exception:
            logging.exception("VK resume failed")
    _vk_captcha_resume = _resume
    if _vk_captcha_timeout:
        _vk_captcha_timeout.cancel()

    async def _timeout():
        global _vk_captcha_needed, _vk_captcha_sid, _vk_captcha_img
        global _vk_captcha_timeout
        await asyncio.sleep(VK_CAPTCHA_TTL_MIN * 60)
        if scheduler.progress:
            for k in list(scheduler.remaining_jobs):
                scheduler.progress.finish_job(k, "captcha_expired")
        _vk_captcha_needed = False
        _vk_captcha_sid = None
        _vk_captcha_img = None
        _vk_captcha_requested_at = None
        _vk_captcha_timeout = None
        logging.info("vk_captcha invalid/expired")

    _vk_captcha_timeout = asyncio.create_task(_timeout())


async def vk_captcha_pause_outbox(db: Database) -> None:
    """Pause all VK jobs and register resume callback."""
    global _vk_captcha_resume
    far = datetime.now(timezone.utc) + timedelta(days=3650)
    async with db.get_session() as session:
        await session.execute(
            update(JobOutbox)
            .where(
                JobOutbox.task.in_(VK_JOB_TASKS),
                JobOutbox.status.in_(
                    [JobStatus.pending, JobStatus.error, JobStatus.running]
                ),
            )
            .values(status=JobStatus.paused, next_run_at=far)
        )
        await session.commit()

    async def _resume() -> None:
        async with db.get_session() as session:
            await session.execute(
                update(JobOutbox)
                .where(
                    JobOutbox.task.in_(VK_JOB_TASKS),
                    JobOutbox.status == JobStatus.paused,
                )
                .values(status=JobStatus.pending, next_run_at=datetime.now(timezone.utc))
            )
            await session.commit()

    _vk_captcha_resume = _resume


@dataclass
class PartnerAdminNotice:
    chat_id: int
    message_id: int
    is_photo: bool
    caption: str


_PARTNER_ADMIN_NOTICES: dict[int, PartnerAdminNotice] = {}


def _event_telegraph_link(event: Event) -> str | None:
    if event.telegraph_url:
        return event.telegraph_url
    if event.telegraph_path:
        return f"https://telegra.ph/{event.telegraph_path}"
    return None


def _partner_admin_caption(event: Event) -> str:
    parts = [event.title]
    telegraph_link = _event_telegraph_link(event)
    if telegraph_link:
        parts.append(f"Telegraph: {telegraph_link}")
    if event.source_vk_post_url:
        parts.append(f"VK: {event.source_vk_post_url}")
    return "\n".join(parts)


async def _send_or_update_partner_admin_notice(
    db: Database,
    bot: Bot,
    event: Event,
    user: User | None = None,
) -> None:
    if not bot or not event.id:
        return
    if user is None:
        creator_id = event.creator_id
        if not creator_id:
            return
        async with db.get_session() as session:
            user = await session.get(User, creator_id)
    if not user or not user.is_partner:
        return
    admin_id = await get_superadmin_id(db)
    if not admin_id:
        return
    caption = _partner_admin_caption(event)
    if not caption:
        return
    notice = _PARTNER_ADMIN_NOTICES.get(event.id)
    photo_url = event.photo_urls[0] if event.photo_urls else None
    if photo_url:
        if notice and notice.is_photo and notice.caption == caption:
            return
        if notice and notice.is_photo:
            async with span("tg-send"):
                await bot.edit_message_caption(
                    chat_id=notice.chat_id,
                    message_id=notice.message_id,
                    caption=caption,
                )
            _PARTNER_ADMIN_NOTICES[event.id] = PartnerAdminNotice(
                notice.chat_id, notice.message_id, True, caption
            )
        else:
            async with span("tg-send"):
                msg = await bot.send_photo(admin_id, photo_url, caption=caption)
            _PARTNER_ADMIN_NOTICES[event.id] = PartnerAdminNotice(
                admin_id, msg.message_id, True, caption
            )
    else:
        if notice and not notice.is_photo and notice.caption == caption:
            return
        if notice and not notice.is_photo:
            async with span("tg-send"):
                await bot.edit_message_text(
                    caption,
                    chat_id=notice.chat_id,
                    message_id=notice.message_id,
                )
            _PARTNER_ADMIN_NOTICES[event.id] = PartnerAdminNotice(
                notice.chat_id, notice.message_id, False, caption
            )
        else:
            async with span("tg-send"):
                msg = await bot.send_message(admin_id, caption)
            _PARTNER_ADMIN_NOTICES[event.id] = PartnerAdminNotice(
                admin_id, msg.message_id, False, caption
            )


async def notify_event_added(
    db: Database, bot: Bot, user: User | None, event: Event, added: bool
) -> None:
    """Notify superadmin when a user or partner adds an event."""
    if not added or not user or user.is_superadmin:
        return
    role = "partner" if user.is_partner else "user"
    name = f"@{user.username}" if user.username else str(user.user_id)
    link = _event_telegraph_link(event)
    text = f"{name} ({role}) added event {event.title}"
    if link:
        text += f" — {link}"
    await notify_superadmin(db, bot, text)
    if user.is_partner:
        await _send_or_update_partner_admin_notice(db, bot, event, user=user)


async def notify_inactive_partners(
    db: Database, bot: Bot, tz: timezone
) -> list[User]:
    """Send reminders to partners without events in the last week."""
    cutoff = week_cutoff(tz)
    now = datetime.now(timezone.utc)
    notified: list[User] = []
    async with db.get_session() as session:
        stream = await session.stream_scalars(
            select(User)
            .where(User.is_partner.is_(True), User.blocked.is_(False))
            .execution_options(yield_per=100)
        )
        count = 0
        async for p in stream:
            last = _ensure_utc(
                (
                    await session.execute(
                        select(Event.added_at)
                        .where(Event.creator_id == p.user_id)
                        .order_by(Event.added_at.desc())
                        .limit(1)
                    )
                ).scalars().first()
            )
            last_reminder = _ensure_utc(p.last_partner_reminder)
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
        data, name = ensure_jpeg(bio.getvalue(), "photo.jpg")
        images.append((data, name))
        logging.info("IMG download type=photo name=%s size=%d", name, len(data))
    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        bio = BytesIO()
        async with span("tg-send"):
            await bot.download(message.document.file_id, destination=bio)
        name = message.document.file_name or "image.jpg"
        data, name = ensure_jpeg(bio.getvalue(), name)
        images.append((data, name))
        logging.info("IMG download type=document name=%s size=%d", name, len(data))
    names = [n for _, n in images[:MAX_ALBUM_IMAGES]]
    logging.info(
        "IMG extract done count=%d names=%s limit=%d",
        len(names),
        names,
        MAX_ALBUM_IMAGES,
    )
    return images[:MAX_ALBUM_IMAGES]


def ensure_html_text(message: types.Message) -> tuple[str | None, str]:
    html = message.html_text or message.caption_html
    mode = "native"
    if not html:
        text = message.text or message.caption
        entities = message.entities or message.caption_entities or []
        if text and entities:
            html = html_decoration.unparse(text, entities)
        mode = "rebuilt_from_entities"
    global LAST_HTML_MODE
    LAST_HTML_MODE = mode
    logging.info("html_mode=%s", mode)
    return html, mode


async def upload_images(
    images: list[tuple[bytes, str]],
    limit: int = MAX_ALBUM_IMAGES,
    *,
    force: bool = False,
    event_hint: str | None = None,
) -> tuple[list[str], str]:
    """Upload images to Catbox with retries."""
    catbox_urls: list[str] = []
    catbox_msg = ""
    if not images:
        return [], ""
    logging.info("CATBOX start images=%d limit=%d", len(images or []), limit)
    if not CATBOX_ENABLED and not force:
        logging.info(
            "CATBOX disabled catbox_enabled=%s force=%s images=%d event_hint=%s",
            CATBOX_ENABLED,
            force,
            len(images or []),
            event_hint,
        )
        return [], "disabled"

    session = get_http_session()
    for data, name in images[:limit]:
        data, name = ensure_jpeg(data, name)
        logging.info("CATBOX candidate name=%s size=%d", name, len(data))
        if len(data) > 5 * 1024 * 1024:
            logging.warning("catbox skip %s: too large", name)
            catbox_msg += f"{name}: too large; "
            continue
        if not detect_image_type(data):
            logging.warning("catbox upload %s: not image", name)
            catbox_msg += f"{name}: not image; "
        success = False
        delays = [0.5, 1.0, 2.0]
        for attempt in range(1, 4):
            logging.info("catbox try %d/3", attempt)
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
                                logging.info("catbox ok %s", url)
                                success = True
                                break
                            reason = f"{resp.status} {text_r}".strip()
            except Exception as e:  # pragma: no cover - network errors
                reason = str(e)
            if success:
                break
            if attempt < 3:
                await asyncio.sleep(delays[attempt - 1])
        if not success:
            logging.warning("catbox failed %s", reason)
            catbox_msg += f"{name}: failed; "
        catbox_msg = catbox_msg.strip("; ")
    logging.info(
        "CATBOX done uploaded=%d skipped=%d msg=%s",
        len(catbox_urls),
        max(0, len(images[:limit]) - len(catbox_urls)),
        catbox_msg,
    )
    global LAST_CATBOX_MSG
    LAST_CATBOX_MSG = catbox_msg
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


def normalize_event_type(
    title: str, description: str, event_type: str | None
) -> str | None:
    """Return corrected event type, marking film screenings as ``кинопоказ``."""
    text = f"{title} {description}".lower()
    if event_type in (None, "", "спектакль"):
        if any(word in text for word in ("кино", "фильм", "кинопоказ", "киносеанс")):
            return "кинопоказ"
    return event_type


def canonicalize_date(value: str | None) -> str | None:
    """Return ISO date string if value parses as date or ``None``."""
    if not value:
        return None
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


def parse_period_range(value: str) -> tuple[date | None, date | None]:
    """Parse period string like ``YYYY-MM`` or ``YYYY-MM-DD..YYYY-MM-DD``."""
    if not value:
        return None, None
    raw = value.strip()
    if not raw:
        return None, None
    if raw.startswith("period="):
        raw = raw.split("=", 1)[1]
    if ".." in raw:
        start_raw, end_raw = raw.split("..", 1)
    else:
        start_raw, end_raw = raw, raw

    def _parse_endpoint(component: str, *, is_start: bool) -> date | None:
        comp = component.strip()
        if not comp:
            return None
        try:
            return date.fromisoformat(comp)
        except ValueError:
            if re.fullmatch(r"\d{4}-\d{2}", comp):
                year, month = map(int, comp.split("-"))
                day = 1 if is_start else calendar.monthrange(year, month)[1]
                return date(year, month, day)
            if re.fullmatch(r"\d{4}", comp):
                year = int(comp)
                month = 1 if is_start else 12
                day = 1 if is_start else 31
                return date(year, month, day)
        return None

    start = _parse_endpoint(start_raw, is_start=True)
    end = _parse_endpoint(end_raw, is_start=False)
    if start and end and end < start:
        start, end = end, start
    return start, end


def parse_city_from_fest_name(name: str) -> str | None:
    """Extract city name from festival name like 'День города <Город>'."""
    m = re.search(r"День города\s+([A-ЯЁа-яёA-Za-z\- ]+?)(?:\s+\d|$)", name)
    if not m:
        return None
    return m.group(1).strip()


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
        today = datetime.now(LOCAL_TZ).date()
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
                Festival.full_name,
                Festival.telegraph_url,
                Festival.telegraph_path,
                Festival.photo_url,
                Festival.vk_post_url,
                Festival.nav_hash,
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
    for (
        fid,
        name,
        full_name,
        tg_url,
        path,
        photo_url,
        vk_url,
        nav_hash,
        start_s,
        end_s,
    ) in rows:
        start = parse_iso_date(start_s) if start_s else None
        end = parse_iso_date(end_s) if end_s else None
        if not start and not end:
            continue
        fest = Festival(
            id=fid,
            name=name,
            full_name=full_name,
            telegraph_url=tg_url,
            telegraph_path=path,
            photo_url=photo_url,
            vk_post_url=vk_url,
            nav_hash=nav_hash,
        )
        data.append((start, end, fest))
    return data


async def all_festivals(db: Database) -> list[tuple[date | None, date | None, Festival]]:
    """Return all festivals with their inferred date ranges."""
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
                Festival.full_name,
                Festival.telegraph_url,
                Festival.telegraph_path,
                Festival.photo_url,
                Festival.vk_post_url,
                Festival.nav_hash,
                func.coalesce(Festival.start_date, ev_dates.c.start).label("start"),
                func.coalesce(Festival.end_date, ev_dates.c.end).label("end"),
            )
            .outerjoin(ev_dates, ev_dates.c.festival == Festival.name)
            .order_by(func.coalesce(Festival.start_date, ev_dates.c.start))
        )
        rows = (await session.execute(stmt)).all()

    data: list[tuple[date | None, date | None, Festival]] = []
    for (
        fid,
        name,
        full_name,
        tg_url,
        path,
        photo_url,
        vk_url,
        nav_hash,
        start_s,
        end_s,
    ) in rows:
        start = parse_iso_date(start_s) if start_s else None
        end = parse_iso_date(end_s) if end_s else None
        fest = Festival(
            id=fid,
            name=name,
            full_name=full_name,
            telegraph_url=tg_url,
            telegraph_path=path,
            photo_url=photo_url,
            vk_post_url=vk_url,
            nav_hash=nav_hash,
        )
        data.append((start, end, fest))
    return data


async def _build_festival_nav_block(
    db: Database,
    *,
    exclude: str | None = None,
    today: date | None = None,
    items: list[tuple[date | None, date | None, Festival]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Return navigation blocks for festival pages and VK posts."""
    if today is None:
        today = datetime.now(LOCAL_TZ).date()
    if items is None:
        items = await upcoming_festivals(db, today=today, exclude=exclude)
    else:
        if exclude:
            items = [t for t in items if t[2].name != exclude]
    if not items:
        return [], []
    groups: dict[str, list[tuple[date | None, Festival]]] = {}
    for start, end, fest in items:
        if start and start <= today <= (end or start):
            month = today.strftime("%Y-%m")
        else:
            month = (start or today).strftime("%Y-%m")
        groups.setdefault(month, []).append((start, fest))

    nodes: list[dict] = []
    nodes.extend(telegraph_br())
    nodes.append({"tag": "h3", "children": ["Ближайшие фестивали"]})
    lines: list[str] = [VK_BLANK_LINE, "Ближайшие фестивали"]
    for month in sorted(groups.keys()):
        month_name = month_name_nominative(month)
        nodes.append({"tag": "h4", "children": [month_name]})
        lines.append(month_name)
        for start, fest in sorted(
            groups[month], key=lambda t: t[0] or date.max
        ):
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


def _festival_period_str(start: date | None, end: date | None) -> str:
    if start and end:
        if start == end:
            return format_day_pretty(start)
        return f"{format_day_pretty(start)} - {format_day_pretty(end)}"
    if start:
        return format_day_pretty(start)
    if end:
        return format_day_pretty(end)
    return ""


def build_festival_card_nodes(
    fest: Festival,
    start: date | None,
    end: date | None,
    *,
    with_image: bool,
    add_spacer: bool,
) -> tuple[list[dict], bool, int]:
    """Return Telegraph nodes for a single festival card.

    Returns a tuple ``(nodes, used_img, spacer_count)`` where ``nodes`` is a list
    of Telegraph nodes representing the card, ``used_img`` indicates whether a
    figure with an image was rendered and ``spacer_count`` is ``1`` if a trailing
    spacer paragraph was added.
    """

    nodes: list[dict] = []
    url = fest.telegraph_url or (
        f"https://telegra.ph/{fest.telegraph_path}" if fest.telegraph_path else ""
    )
    title = fest.full_name or fest.name
    if url:
        title_node = {
            "tag": "h3",
            "children": [
                {"tag": "a", "attrs": {"href": url}, "children": [title]}
            ],
        }
    else:
        logging.debug("festival_card_missing_url", extra={"fest": title})
        title_node = {"tag": "h3", "children": [title]}

    period = _festival_period_str(start, end)
    used_img = False
    if with_image and fest.photo_url:
        fig_children: list[dict] = []
        img_node: dict = {"tag": "img", "attrs": {"src": fest.photo_url}}
        if url:
            fig_children.append({"tag": "a", "attrs": {"href": url}, "children": [img_node]})
        else:
            fig_children.append(img_node)
        if period:
            fig_children.append({"tag": "figcaption", "children": [f"📅 {period}"]})
        nodes.append({"tag": "figure", "children": fig_children})
        nodes.append(title_node)
        used_img = True
    else:
        nodes.append(title_node)
        if period:
            nodes.append({"tag": "p", "children": [f"📅 {period}"]})

    spacer_count = 0
    if add_spacer:
        nodes.append({"tag": "p", "attrs": {"dir": "auto"}, "children": ["\u200b"]})
        spacer_count = 1

    return nodes, used_img, spacer_count


def _build_festival_cards(
    items: list[tuple[date | None, date | None, Festival]]
) -> tuple[list[dict], int, int, int, bool]:
    nodes: list[dict] = []
    with_img = 0
    without_img = 0
    spacers = 0
    compact_tail = False
    for idx, (start, end, fest) in enumerate(items):
        add_spacer = idx < len(items) - 1
        use_img = not compact_tail
        card_nodes, used_img, spacer_count = build_festival_card_nodes(
            fest, start, end, with_image=use_img, add_spacer=add_spacer
        )
        candidate = nodes + card_nodes
        if not compact_tail and rough_size(candidate, TELEGRAPH_LIMIT + 1) > TELEGRAPH_LIMIT:
            compact_tail = True
            card_nodes, used_img, spacer_count = build_festival_card_nodes(
                fest, start, end, with_image=False, add_spacer=add_spacer
            )
            candidate = nodes + card_nodes
            if rough_size(candidate, TELEGRAPH_LIMIT + 1) > TELEGRAPH_LIMIT:
                break
        elif compact_tail and rough_size(candidate, TELEGRAPH_LIMIT + 1) > TELEGRAPH_LIMIT:
            break

        nodes.extend(card_nodes)
        spacers += spacer_count
        if used_img:
            with_img += 1
        else:
            without_img += 1
    return nodes, with_img, without_img, spacers, compact_tail


def _ensure_img_links(
    page_html: str,
    link_map: dict[str, tuple[str, int | None, str, str]],
) -> tuple[str, int, int]:
    """Ensure every image retains its link, add fallback if missing.

    ``link_map`` maps image ``src`` to a tuple ``(url, fest_id, slug, name)``.
    Returns updated HTML and counts of ok/fixed image links.
    """

    img_links_ok = 0
    img_links_fixed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal img_links_ok, img_links_fixed
        fig_html = match.group(0)
        m = re.search(r'<img[^>]+src="([^"]+)"', fig_html)
        if not m:
            return fig_html
        src = m.group(1)
        info = link_map.get(src)
        if not info:
            return fig_html
        url, fest_id, slug, name = info
        if "<a" in fig_html:
            img_links_ok += 1
            return fig_html
        img_links_fixed += 1
        logging.warning(
            "festivals_index img_link_missing",
            extra={"festival_id": fest_id, "slug": slug, "festival": name},
        )
        fallback = f'<p><a href="{html.escape(url)}">Открыть страницу фестиваля →</a></p>'
        return fig_html + fallback

    updated_html = re.sub(r"<figure>.*?</figure>", repl, page_html, flags=re.DOTALL)
    return updated_html, img_links_ok, img_links_fixed


async def sync_festivals_index_page(db: Database) -> None:
    """Create or update landing page listing all festivals."""
    token = get_telegraph_token()
    if not token:
        logging.error(
            "Telegraph token unavailable",
            extra={"action": "error", "target": "tg"},
        )
        return
    tg = Telegraph(access_token=token)

    items = await all_festivals(db)
    today = datetime.now(LOCAL_TZ).date()
    items = [t for t in items if t[1] is None or t[1] >= today]
    items.sort(key=lambda t: t[0] or date.max)
    link_map = {}
    for _, _, fest in items:
        url = fest.telegraph_url or (
            f"https://telegra.ph/{fest.telegraph_path}" if fest.telegraph_path else ""
        )
        if fest.photo_url and url:
            link_map[fest.photo_url] = (
                url,
                fest.id,
                fest.telegraph_path or "",
                fest.name,
            )
    nodes, with_img, without_img, spacers, compact_tail = _build_festival_cards(items)
    from telegraph.utils import nodes_to_html

    cover_url = await get_setting_value(db, "festivals_index_cover")
    cover_html = (
        f'<figure><img src="{html.escape(cover_url)}"/></figure>' if cover_url else ""
    )
    intro_html = (
        f"{FEST_INDEX_INTRO_START}<p><i>Вот какие фестивали нашёл для вас канал "
        f'<a href="https://t.me/kenigevents">Полюбить Калининград Анонсы</a>.</i></p>'
        f"{FEST_INDEX_INTRO_END}"
    )
    content_html = (
        cover_html + intro_html + (nodes_to_html(nodes) if nodes else "") + FOOTER_LINK_HTML
    )
    content_html = sanitize_telegraph_html(content_html)
    path = await get_setting_value(db, "fest_index_path")
    url = await get_setting_value(db, "fest_index_url")
    title = "Все фестивали Калининградской области"

    try:
        if path:
            await telegraph_edit_page(
                tg,
                path,
                title=title,
                html_content=content_html,
                caller="festival_build",
            )
        else:
            data = await telegraph_create_page(
                tg,
                title=title,
                html_content=content_html,
                caller="festival_build",
            )
            url = normalize_telegraph_url(data.get("url"))
            path = data.get("path")
        page = await telegraph_call(tg.get_page, path, return_html=True)
        page_html = page.get("content_html", "")
        page_html, img_ok, img_fix = _ensure_img_links(page_html, link_map)
        if img_fix:
            page_html = sanitize_telegraph_html(page_html)
            await telegraph_edit_page(
                tg, path, title=title, html_content=page_html, caller="festival_build"
            )
        logging.info(
            "updated festivals index page" if path else f"created festivals index page {url}",
            extra={
                "action": "edited" if path else "created",
                "target": "tg",
                "path": path,
                "url": url,
                "with_img": with_img,
                "without_img": without_img,
                "spacers": spacers,
                "compact_tail": compact_tail,
                "img_links_ok": img_ok,
                "img_links_fixed": img_fix,
            },
        )
    except Exception as e:
        logging.error(
            "Failed to sync festivals index page: %s",
            e,
            extra={
                "action": "error",
                "target": "tg",
                "path": path,
                "img_links_ok": 0,
                "img_links_fixed": 0,
            },
        )
        return

    if path:
        await set_setting_value(db, "fest_index_path", path)
    if url is None and path:
        url = f"https://telegra.ph/{path}"
    if url:
        await set_setting_value(db, "fest_index_url", url)


async def rebuild_festivals_index_if_needed(
    db: Database, telegraph: Telegraph | None = None, force: bool = False
) -> tuple[str, str]:
    """Rebuild the aggregated festivals landing page if content changed.

    Returns a tuple ``(status, url)`` where ``status`` is one of
    ``"built"``, ``"updated"`` or ``"nochange"``. The landing page lists all
    upcoming festivals grouped by month. The resulting HTML is hashed and the
    hash is compared with the previously stored one to avoid unnecessary
    Telegraph updates.
    """

    start_t = _time.perf_counter()
    items = await upcoming_festivals(db)
    link_map: dict[str, tuple[str, int | None, str, str]] = {}
    for _, _, fest in items:
        url = fest.telegraph_url or (
            f"https://telegra.ph/{fest.telegraph_path}" if fest.telegraph_path else ""
        )
        if fest.photo_url and url:
            link_map[fest.photo_url] = (
                url,
                fest.id,
                fest.telegraph_path or "",
                fest.name,
            )
    nodes, with_img, without_img, spacers, compact_tail = _build_festival_cards(items)
    from telegraph.utils import nodes_to_html

    cover_url = await get_setting_value(db, "festivals_index_cover")
    cover_html = (
        f'<figure><img src="{html.escape(cover_url)}"/></figure>' if cover_url else ""
    )
    intro_html = (
        f"{FEST_INDEX_INTRO_START}<p><i>Вот какие фестивали нашёл для вас канал "
        f'<a href="https://t.me/kenigevents">Полюбить Калининград Анонсы</a>.'
        f"</i></p>{FEST_INDEX_INTRO_END}"
    )
    nav_html = nodes_to_html(nodes) if nodes else "<p>Пока нет ближайших фестивалей</p>"
    content_html = cover_html + intro_html + nav_html + FOOTER_LINK_HTML
    content_html = sanitize_telegraph_html(content_html)
    new_hash = hashlib.sha256(content_html.encode("utf-8")).hexdigest()
    old_hash = await get_setting_value(db, "festivals_index_hash")
    url = await get_setting_value(db, "festivals_index_url") or await get_setting_value(
        db, "fest_index_url"
    )
    path = await get_setting_value(db, "festivals_index_path") or await get_setting_value(
        db, "fest_index_path"
    )

    if not force and old_hash == new_hash and url:
        dur = (_time.perf_counter() - start_t) * 1000
        logging.info(
            "festivals_index",
            extra={
                "action": "nochange",
                "page": "festivals_index",
                "title": "Все фестивали Калининградской области",
                "old_hash": (old_hash or "")[:6],
                "new_hash": new_hash[:6],
                "count": len(items),
                "size": len(content_html),
                "took_ms": dur,
                "with_img": with_img,
                "without_img": without_img,
                "spacers": spacers,
                "compact_tail": compact_tail,
                "img_links_ok": 0,
                "img_links_fixed": 0,
            },
        )
        return "nochange", url

    token = get_telegraph_token()
    if telegraph is None:
        if not token:
            logging.error(
                "Telegraph token unavailable",
                extra={"action": "error", "target": "tg"},
            )
            dur = (_time.perf_counter() - start_t) * 1000
            logging.info(
                "festivals_index",
                extra={
                    "action": "nochange",
                    "page": "festivals_index",
                    "title": "Все фестивали Калининградской области",
                    "old_hash": (old_hash or "")[:6],
                    "new_hash": new_hash[:6],
                    "count": len(items),
                    "size": len(content_html),
                    "took_ms": dur,
                    "with_img": with_img,
                    "without_img": without_img,
                    "spacers": spacers,
                    "compact_tail": compact_tail,
                    "img_links_ok": 0,
                    "img_links_fixed": 0,
                },
            )
            return "nochange", url or ""
        telegraph = Telegraph(access_token=token)

    title = "Все фестивали Калининградской области"
    try:
        if path:
            await telegraph_edit_page(
                telegraph,
                path,
                title=title,
                html_content=content_html,
                caller="festival_build",
            )
            status = "updated"
            if not url:
                url = f"https://telegra.ph/{path}"
        else:
            data = await telegraph_create_page(
                telegraph,
                title=title,
                html_content=content_html,
                caller="festival_build",
            )
            url = normalize_telegraph_url(data.get("url"))
            path = data.get("path")
            status = "built"
        page = await telegraph_call(telegraph.get_page, path, return_html=True)
        page_html = page.get("content_html", "")
        page_html, img_ok, img_fix = _ensure_img_links(page_html, link_map)
        if img_fix:
            page_html = sanitize_telegraph_html(page_html)
            await telegraph_edit_page(
                telegraph,
                path,
                title=title,
                html_content=page_html,
                caller="festival_build",
            )
    except Exception as e:
        dur = (_time.perf_counter() - start_t) * 1000
        logging.error(
            "Failed to rebuild festivals index page: %s",
            e,
            extra={
                "action": "error",
                "target": "tg",
                "path": path,
                "page": "festivals_index",
                "title": "Все фестивали Калининградской области",
                "old_hash": (old_hash or "")[:6],
                "new_hash": new_hash[:6],
                "count": len(items),
                "size": len(content_html),
                "took_ms": dur,
                "with_img": with_img,
                "without_img": without_img,
                "spacers": spacers,
                "compact_tail": compact_tail,
                "img_links_ok": 0,
                "img_links_fixed": 0,
            },
        )
        raise

    await set_setting_value(db, "festivals_index_url", url)
    await set_setting_value(db, "fest_index_url", url)
    if path:
        await set_setting_value(db, "festivals_index_path", path)
        await set_setting_value(db, "fest_index_path", path)
    await set_setting_value(db, "festivals_index_hash", new_hash)
    await set_setting_value(db, "festivals_index_built_at", datetime.now(timezone.utc).isoformat())

    dur = (_time.perf_counter() - start_t) * 1000
    logging.info(
        "festivals_index",
        extra={
            "action": status,
            "page": "festivals_index",
            "title": "Все фестивали Калининградской области",
            "old_hash": (old_hash or "")[:6],
            "new_hash": new_hash[:6],
            "count": len(items),
            "size": len(content_html),
            "took_ms": dur,
            "with_img": with_img,
            "without_img": without_img,
            "spacers": spacers,
            "compact_tail": compact_tail,
            "img_links_ok": img_ok,
            "img_links_fixed": img_fix,
        },
    )
    return status, url


async def rebuild_fest_nav_if_changed(db: Database) -> bool:
    """Rebuild festival navigation and enqueue update jobs if changed.

    Returns ``True`` if navigation hash changed and jobs were scheduled.
    """

    _, _, changed = await build_festivals_nav_block(db)
    if not changed:
        return False
    await rebuild_festivals_index_if_needed(db)
    nav_hash = await get_setting_value(db, "fest_nav_hash") or "0"
    suffix = int(nav_hash[:4], 16)
    eid = -suffix
    await enqueue_job(db, eid, JobTask.fest_nav_update_all)
    logging.info(
        "scheduled festival navigation update",
        extra={"action": "scheduled", "count": 1, "nav_hash": nav_hash[:6]},
    )
    return True


ICS_LABEL = "Добавить в календарь"

BODY_SPACER_HTML = '<p>&#8203;</p>'

FOOTER_LINK_HTML = (
    BODY_SPACER_HTML
    + '<p><a href="https://t.me/kenigevents">Полюбить Калининград Анонсы</a></p>'
    + BODY_SPACER_HTML
)

HISTORY_FOOTER_HTML = '<p><a href="https://t.me/kgdstories">Полюбить Калининград Истории</a></p>'


TELEGRAPH_ALLOWED_TAGS = {
    "p",
    "a",
    "img",
    "figure",
    "figcaption",
    "h3",
    "h4",
    "b",
    "strong",
    "i",
    "em",
    "u",
    "s",
    "del",
    "blockquote",
    "code",
    "pre",
    "ul",
    "ol",
    "li",
}

_TG_HEADER_RE = re.compile(r"<(/?)h([1-6])(\b[^>]*)>", re.IGNORECASE)
_TELEGRAPH_TAG_RE = re.compile(r"<\/?([a-z0-9]+)", re.IGNORECASE)


def sanitize_telegraph_html(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        slash, level, attrs = match.groups()
        level = level.lower()
        if level in {"1", "2", "5", "6"}:
            level = "3"
        return f"<{slash}h{level}{attrs}>"

    html = _TG_HEADER_RE.sub(repl, html)
    tags = {t.lower() for t in _TELEGRAPH_TAG_RE.findall(html)}
    disallowed = [t for t in tags if t not in TELEGRAPH_ALLOWED_TAGS]
    if disallowed:
        raise ValueError(f"Unsupported tag(s): {', '.join(disallowed)}")
    return html


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
    # Skip initial images: legacy pages may have ``<img><p></p>`` pairs while
    # new pages wrap the first image in ``<figure>``.  We advance the insertion
    # point past any such blocks so that the ICS link always appears under the
    # title but after all leading images.
    img_pattern = re.compile(
        r"(?:<img[^>]+><p></p>|<figure><img[^>]+/></figure>)"
    )
    for m in img_pattern.finditer(html_content, pos):
        pos = m.end()
    return html_content[:pos] + link_html + html_content[pos:]


def apply_month_nav(html_content: str, html_block: str | None) -> str:
    """Insert or remove the month navigation block anchored by ``<hr>``."""
    if html_block is None:
        pattern = re.compile(r"<hr\s*/?>", flags=re.I)
        matches = list(pattern.finditer(html_content))
        if matches:
            html_content = html_content[: matches[-1].end()]
        return html_content
    return ensure_footer_nav_with_hr(html_content, html_block)


def apply_festival_nav(
    html_content: str, nav_html: str | Iterable[str]
) -> tuple[str, bool, int, bool]:
    """Idempotently insert or replace the festival navigation block.

    ``nav_html`` may be a pre-rendered HTML fragment or an iterable of pieces
    which will be concatenated deterministically. The resulting block includes
    a ``NAV_HASH`` comment with a SHA256 hash of the normalized HTML so that
    the existing block can be compared cheaply.

    Returns a tuple ``(html, changed, removed_legacy_blocks, legacy_markers_replaced)``.
    """

    if not isinstance(nav_html, str):
        nav_html = "".join(nav_html)
    nav_hash = content_hash(nav_html)
    nav_block = f"<!--NAV_HASH:{nav_hash}-->{nav_html}"
    legacy_start_variants = [
        "<!--fest-nav-start-->",
        "<!-- fest-nav-start -->",
        "<!-- FEST_NAV_START -->",
        "<!--FEST_NAV_START-->",
    ]
    legacy_end_variants = [
        "<!--fest-nav-end-->",
        "<!-- fest-nav-end -->",
        "<!-- FEST_NAV_END -->",
        "<!--FEST_NAV_END-->",
    ]
    legacy_markers_replaced = False
    if any(v in html_content for v in legacy_start_variants + legacy_end_variants):
        for v in legacy_start_variants:
            if v in html_content:
                legacy_markers_replaced = True
                html_content = html_content.replace(v, FEST_NAV_START)
        for v in legacy_end_variants:
            if v in html_content:
                legacy_markers_replaced = True
                html_content = html_content.replace(v, FEST_NAV_END)

    block_pattern = re.compile(
        re.escape(FEST_NAV_START) + r"(.*?)" + re.escape(FEST_NAV_END), re.DOTALL
    )
    unmarked_block_pattern = re.compile(
        r"(?:<p>\s*)?(?:<h3[^>]*>\s*Ближайшие(?:\s|&nbsp;)+фестивали\s*</h3>|"
        r"<p>\s*<strong>\s*Ближайшие(?:\s|&nbsp;)+фестивали\s*</strong>\s*</p>)"
        r".*?(?=(?:<h[23][^>]*>|<!--\s*FEST_NAV_START\s*-->|$))",
        re.DOTALL | re.IGNORECASE,
    )

    blocks = block_pattern.findall(html_content)
    legacy_headings = unmarked_block_pattern.findall(html_content)

    if len(blocks) == 1 and not legacy_headings:
        current = blocks[0]
        if content_hash(current) == content_hash(nav_block):
            if legacy_markers_replaced:
                html_content = block_pattern.sub(nav_block, html_content, count=1)
                html_content = apply_footer_link(html_content)
                return html_content, True, 0, True
            html_content = apply_footer_link(html_content)
            return html_content, False, 0, False

    removed_legacy_blocks = 0
    html_content, n = block_pattern.subn("", html_content)
    removed_legacy_blocks += n
    html_content, n = unmarked_block_pattern.subn("", html_content)
    removed_legacy_blocks += n

    html_content = replace_between_markers(
        html_content, FEST_NAV_START, FEST_NAV_END, nav_block
    )
    html_content = apply_footer_link(html_content)
    return html_content, True, removed_legacy_blocks, legacy_markers_replaced


def apply_footer_link(html_content: str) -> str:
    """Ensure the Telegram channel link footer is present once."""
    pattern = re.compile(
        r'(?:<p>(?:&nbsp;|&#8203;)</p>)?<p><a href="https://t\.me/kenigevents">[^<]+</a></p><p>(?:&nbsp;|&#8203;)</p>'
    )
    html_content = pattern.sub("", html_content).rstrip()
    return html_content + FOOTER_LINK_HTML


async def build_month_nav_html(db: Database, current_month: str | None = None) -> str:
    today = datetime.now(LOCAL_TZ).date()
    start_nav = today.replace(day=1)
    end_nav = date(today.year + 1, 4, 1)
    async with db.get_session() as session:
        res_nav = await session.execute(
            select(func.substr(Event.date, 1, 7))
            .where(
                Event.date >= start_nav.isoformat(),
                Event.date < end_nav.isoformat(),
            )
            .group_by(func.substr(Event.date, 1, 7))
            .order_by(func.substr(Event.date, 1, 7))
        )
        months = [r[0] for r in res_nav]
        if not months:
            return ""
        res_pages = await session.execute(
            select(MonthPage).where(MonthPage.month.in_(months))
        )
        page_map = {p.month: p for p in res_pages.scalars().all()}
    links: list[str] = []
    for idx, m in enumerate(months):
        p = page_map.get(m)
        if not p or not p.url:
            continue
        name = month_name_nominative(m)
        if current_month and m == current_month:
            links.append(name)
        else:
            links.append(f'<a href="{html.escape(p.url)}">{name}</a>')
        if idx < len(months) - 1:
            links.append(" ")
    if not links:
        return ""
    result = "<br/><h4>" + "".join(links) + "</h4>"
    fest_index_url = await get_setting_value(db, "fest_index_url")
    if fest_index_url:
        result += (
            f'<br/><h3><a href="{html.escape(fest_index_url)}">'
            f"Фестивали</a></h3>"
        )
    return result


async def build_month_nav_block(
    db: Database, current_month: str | None = None
) -> str:
    """Return the Telegraph-ready month navigation block.

    ``current_month`` — month key (``YYYY-MM``) of the page being built. If
    provided, this month will be shown as plain text instead of a link.
    """
    nav_html = await build_month_nav_html(db, current_month)
    if not nav_html:
        return ""
    if nav_html.startswith("<br/>"):
        nav_html = nav_html[5:]
    return f"<p>\u200B</p>{nav_html}<p>\u200B</p>"


async def refresh_month_nav(db: Database) -> None:
    logging.info("refresh_month_nav start")
    today = datetime.now(LOCAL_TZ).date()
    start_nav = today.replace(day=1)
    end_nav = date(today.year + 1, 4, 1)
    async with db.get_session() as session:
        res_nav = await session.execute(
            select(func.substr(Event.date, 1, 7))
            .where(
                Event.date >= start_nav.isoformat(),
                Event.date < end_nav.isoformat(),
            )
            .group_by(func.substr(Event.date, 1, 7))
            .order_by(func.substr(Event.date, 1, 7))
        )
        months = [r[0] for r in res_nav]
        res_pages = await session.execute(
            select(MonthPage).where(MonthPage.month.in_(months))
        )
        page_map = {p.month: p for p in res_pages.scalars().all()}

    for m in months:
        page = page_map.get(m)
        update_links = bool(page and page.url)
        try:
            await asyncio.wait_for(
                sync_month_page(db, m, update_links=update_links, force=True),
                timeout=55,
            )
        except asyncio.TimeoutError:
            logging.error("refresh_month_nav timeout month=%s", m)
        await asyncio.sleep(0)
    logging.info("refresh_month_nav finish")

async def build_month_buttons(
    db: Database, limit: int = 3, debug: bool = False
) -> list[types.InlineKeyboardButton] | tuple[
    list[types.InlineKeyboardButton], str, list[str]
]:
    """Return buttons linking to upcoming month pages."""
    # Ensure LOCAL_TZ is initialised based on current DB setting.
    await get_tz_offset(db)
    cur_month = datetime.now(LOCAL_TZ).strftime("%Y-%m")
    async with db.get_session() as session:
        result = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= cur_month)
            .order_by(MonthPage.month)
        )
        months = result.scalars().all()
    buttons: list[types.InlineKeyboardButton] = []
    shown: list[str] = []
    for p in months:
        if not p.url:
            continue
        label = f"\U0001f4c5 {month_name_nominative(p.month)}"
        buttons.append(types.InlineKeyboardButton(text=label, url=p.url))
        shown.append(p.month)
        if len(buttons) >= limit:
            break
    if debug:
        return buttons, cur_month, shown
    return buttons


async def build_event_month_buttons(event: Event, db: Database) -> list[types.InlineKeyboardButton]:
    """Return navigation buttons for the event's month and the next month with events."""
    month = (event.date.split("..", 1)[0])[:7]
    async with db.get_session() as session:
        result = await session.execute(
            select(MonthPage)
            .where(MonthPage.month >= month)
            .order_by(MonthPage.month)
        )
        months = result.scalars().all()
    buttons: list[types.InlineKeyboardButton] = []
    cur_page = next((m for m in months if m.month == month and m.url), None)
    if cur_page:
        label = f"\U0001f4c5 {month_name_nominative(cur_page.month)}"
        buttons.append(types.InlineKeyboardButton(text=label, url=cur_page.url))
    next_page = None
    for m in months:
        if m.month > month and m.url:
            next_page = m
            break
    if next_page:
        label = f"\U0001f4c5 {month_name_nominative(next_page.month)}"
        buttons.append(types.InlineKeyboardButton(text=label, url=next_page.url))
    return buttons


async def update_source_post_keyboard(event_id: int, db: Database, bot: Bot) -> None:
    """Update reply markup on the source post with ICS and month navigation buttons."""
    logging.info("update_source_post_keyboard start for event %s", event_id)
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        logging.info("update_source_post_keyboard skip for event %s: no event", event_id)
        return

    def detect_chat_type(cid: int | None) -> str:
        if cid is None:
            return "unknown"
        if cid > 0:
            return "private"
        return "channel" if str(cid).startswith("-100") else "group"

    # attempt to restore correct source_chat_id from source_post_url if it looks like
    # it points to a channel message but event stores a private chat id
    if ev.source_post_url and ev.source_chat_id and ev.source_chat_id > 0:
        chat_match = None
        msg_id = None
        m = re.match(r"https://t.me/c/([0-9]+)/([0-9]+)", ev.source_post_url)
        if m:
            cid, msg_id = m.groups()
            chat_match = int("-100" + cid)
        else:
            m = re.match(r"https://t.me/([A-Za-z0-9_]+)/([0-9]+)", ev.source_post_url)
            if m:
                username, msg_id = m.group(1), int(m.group(2))
                try:
                    chat = await bot.get_chat("@" + username)
                    chat_match = chat.id
                except Exception:
                    pass
        if chat_match and msg_id:
            ev.source_chat_id = chat_match
            ev.source_message_id = int(msg_id)
            async with db.get_session() as session:
                obj = await session.get(Event, event_id)
                if obj:
                    obj.source_chat_id = ev.source_chat_id
                    obj.source_message_id = ev.source_message_id
                    await session.commit()

    rows: list[list[types.InlineKeyboardButton]] = []
    if ev.ics_post_url:
        rows.append(
            [
                types.InlineKeyboardButton(
                    text="Добавить в календарь", url=ev.ics_post_url
                )
            ]
        )
    month_result = await build_month_buttons(db, limit=2, debug=True)
    month_buttons, cur_month, months_shown = month_result
    logging.info(
        "month_buttons_source_post cur=%s -> %s", cur_month, months_shown
    )
    if month_buttons:
        rows.append(month_buttons)
    if not rows:
        logging.info("update_source_post_keyboard skip for event %s: no buttons", event_id)
        return
    markup = types.InlineKeyboardMarkup(inline_keyboard=rows)
    target = f"{ev.source_chat_id}/{ev.source_message_id}"
    chat_type = detect_chat_type(ev.source_chat_id)
    edit_failed_reason: str | None = None

    can_edit = True
    if ev.source_chat_id and ev.source_chat_id < 0:
        try:
            me = await bot.get_me()
            member = await bot.get_chat_member(ev.source_chat_id, me.id)
            can_edit = bool(getattr(member, "can_edit_messages", False)) or getattr(
                member, "status", ""
            ) == "creator"
        except Exception:
            pass

    if can_edit and ev.source_chat_id and ev.source_message_id:
        try:
            async with TG_SEND_SEMAPHORE:
                async with span("tg-send"):
                    await bot.edit_message_reply_markup(
                        ev.source_chat_id,
                        ev.source_message_id,
                        reply_markup=markup,
                    )
            logging.info(
                "update_source_post_keyboard done for event %s target=%s chat_type=%s",
                event_id,
                target,
                chat_type,
            )
            return
        except TelegramBadRequest as e:
            if "message is not modified" in str(e):
                logging.info(
                    "update_source_post_keyboard no change for event %s target=%s chat_type=%s",
                    event_id,
                    target,
                    chat_type,
                )
                return
            edit_failed_reason = str(e)
            if "message can't be edited" not in str(e):
                logging.error(
                    "update_source_post_keyboard failed for event %s target=%s chat_type=%s: %s",
                    event_id,
                    target,
                    chat_type,
                    e,
                )
                return
        except Exception as e:  # pragma: no cover - network failures
            logging.error(
                "update_source_post_keyboard failed for event %s target=%s chat_type=%s: %s",
                event_id,
                target,
                chat_type,
                e,
            )
            return
    elif ev.source_chat_id and ev.source_message_id and not can_edit:
        edit_failed_reason = "no edit rights"
    if not ev.source_chat_id:
        logging.info(
            "update_source_post_keyboard skip for event %s: no target chat", event_id
        )
        return

    fallback_chat = ev.creator_id if ev.source_chat_id < 0 else ev.source_chat_id
    if edit_failed_reason:
        logging.warning(
            "update_source_post_keyboard edit failed for event %s target=%s chat_type=%s reason=%s fallback=%s",
            event_id,
            target,
            chat_type,
            edit_failed_reason,
            fallback_chat,
        )

    if not fallback_chat:
        return

    try:
        async with TG_SEND_SEMAPHORE:
            async with span("tg-send"):
                msg = await bot.send_message(
                    fallback_chat,
                    "Добавить в календарь/Навигация по месяцам",
                    reply_markup=markup,
                )
        if ev.source_chat_id > 0:
            async with db.get_session() as session:
                obj = await session.get(Event, event_id)
                if obj:
                    obj.source_chat_id = fallback_chat
                    obj.source_message_id = msg.message_id
                    await session.commit()
        logging.info(
            "update_source_post_keyboard service message for event %s target=%s/%s chat_type=%s",
            event_id,
            fallback_chat,
            msg.message_id,
            detect_chat_type(fallback_chat),
        )
    except Exception as e:  # pragma: no cover - network failures
        logging.error(
            "update_source_post_keyboard service failed for event %s target=%s chat_type=%s: %s",
            event_id,
            fallback_chat,
            detect_chat_type(fallback_chat),
            e,
        )


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


def _ics_filename(event: Event) -> str:
    d = parse_iso_date(event.date.split("..", 1)[0])
    if d:
        return f"event-{event.id}-{d.isoformat()}.ics"
    return f"event-{event.id}.ics"


def message_link(chat_id: int, message_id: int) -> str:
    """Return a t.me link for a message."""
    if chat_id < 0:
        cid = str(chat_id)
        if cid.startswith("-100"):
            cid = cid[4:]
        else:
            cid = cid[1:]
        return f"https://t.me/c/{cid}/{message_id}"
    return f"https://t.me/{chat_id}/{message_id}"


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def format_event_caption(ev: Event, *, style: str = "ics") -> tuple[str, str | None]:
    emoji_part = ""
    if ev.emoji and not ev.title.strip().startswith(ev.emoji):
        emoji_part = f"{ev.emoji} "
    title = f"{emoji_part}{ev.title}".strip()

    date_part = ev.date.split("..", 1)[0]
    d = parse_iso_date(date_part)
    if d:
        day = format_day_pretty(d)
    else:
        day = ev.date

    parts: list[str] = [html.escape(day)]
    if ev.time:
        parts.append(html.escape(ev.time))

    loc_parts: list[str] = []
    loc = ev.location_name.strip()
    if loc:
        loc_parts.append(html.escape(loc))
    addr = ev.location_address
    if addr and ev.city:
        addr = strip_city_from_address(addr, ev.city)
    if addr:
        loc_parts.append(html.escape(addr))
    if ev.city:
        loc_parts.append(f"#{html.escape(ev.city)}")
    if loc_parts:
        parts.append(", ".join(loc_parts))

    details = " ".join(parts)

    lines = [html.escape(title)]
    if ev.telegraph_url:
        lines.append(f'<a href="{html.escape(ev.telegraph_url)}">Подробнее</a>')
    lines.append(f"<i>{details}</i>")
    return "\n".join(lines), "HTML"


async def ics_publish(event_id: int, db: Database, bot: Bot, progress=None) -> bool:
    async with get_ics_semaphore():
        async with db.get_session() as session:
            ev = await session.get(Event, event_id)
        if not ev:
            return False

        try:
            content = await build_ics_content(db, ev)
            ics_bytes = content.encode("utf-8")
            hash_source = "\r\n".join(
                l for l in content.split("\r\n") if not l.startswith(("UID:", "DTSTAMP:"))
            ).encode("utf-8")
        except Exception as e:  # pragma: no cover - build failure
            if progress:
                progress.mark("ics_supabase", "error", str(e))
            raise

        ics_hash = hashlib.sha256(hash_source).hexdigest()
        filename = _ics_filename(ev)
        supabase_url: str | None = None

        if ev.ics_hash == ics_hash:
            if progress:
                progress.mark("ics_supabase", "skipped_nochange", "no change")
            changed = False
        else:
            supabase_disabled = os.getenv("SUPABASE_DISABLED") == "1"
            if not supabase_disabled:
                try:
                    client = get_supabase_client()
                    if client:
                        storage = client.storage.from_(SUPABASE_BUCKET)
                        async with span("http"):
                            await asyncio.to_thread(
                                storage.upload,
                                filename,
                                ics_bytes,
                                {
                                    "content-type": ICS_CONTENT_TYPE,
                                    "content-disposition": ICS_CONTENT_DISP_TEMPLATE.format(
                                        name=filename
                                    ),
                                    "upsert": "true",
                                },
                            )
                            supabase_url = await asyncio.to_thread(
                                storage.get_public_url, filename
                            )
                        if progress:
                            progress.mark("ics_supabase", "done", supabase_url)
                        logging.info("ics_publish supabase_url=%s", supabase_url)
                        logline("ICS", event_id, "supabase done", url=supabase_url)
                except OSError:
                    if progress:
                        progress.mark(
                            "ics_supabase", "warn_net", "временная ошибка сети, будет повтор"
                        )
                    raise RuntimeError("temporary network error")
                except Exception as se:  # pragma: no cover - network failure
                    if progress:
                        progress.mark("ics_supabase", "error", str(se))
                    raise
            else:
                logging.info("ics_publish SUPABASE_DISABLED=1")
                if progress:
                    progress.mark("ics_supabase", "skipped_disabled", "disabled")
            changed = True

        async with db.get_session() as session:
            ev = await session.get(Event, event_id)
            if ev:
                ev.ics_hash = ics_hash
                ev.ics_updated_at = datetime.now(timezone.utc)
                if supabase_url is not None:
                    if ev.ics_url != supabase_url:
                        ev.vk_ics_short_url = None
                        ev.vk_ics_short_key = None
                    ev.ics_url = supabase_url
                else:
                    if ev.ics_url is not None or ev.vk_ics_short_url or ev.vk_ics_short_key:
                        ev.ics_url = None
                        ev.vk_ics_short_url = None
                        ev.vk_ics_short_key = None
                await session.commit()
        if supabase_url is not None:
            await update_source_page_ics(event_id, db, supabase_url)
        return changed


async def tg_ics_post(event_id: int, db: Database, bot: Bot, progress=None) -> bool:
    async with get_ics_semaphore():
        async with db.get_session() as session:
            ev = await session.get(Event, event_id)
        if not ev:
            return False

        try:
            content = await build_ics_content(db, ev)
            ics_bytes = content.encode("utf-8")
            hash_source = "\r\n".join(
                l for l in content.split("\r\n") if not l.startswith(("UID:", "DTSTAMP:"))
            ).encode("utf-8")
        except Exception as e:  # pragma: no cover - build failure
            if progress:
                progress.mark("ics_telegram", "error", str(e))
            raise

        ics_hash = hashlib.sha256(hash_source).hexdigest()
        if ev.ics_hash == ics_hash and ev.ics_file_id and ev.ics_post_url:
            if progress:
                progress.mark("ics_telegram", "skipped_nochange", "no change")
            try:
                await update_source_post_keyboard(event_id, db, bot)
            except Exception as e:  # pragma: no cover - logging inside
                logging.warning(
                    "update_source_post_keyboard failed for %s: %s", event_id, e
                )
            return False

        channel = await get_asset_channel(db)
        if not channel:
            logline("ICS", event_id, "telegram skipped", reason="no_channel")
            return False

        filename = _ics_filename(ev)
        file = types.BufferedInputFile(ics_bytes, filename=filename)
        caption, parse_mode = format_event_caption(ev)
        try:
            async with span("tg-send"):
                msg = await bot.send_document(
                    channel.channel_id,
                    file,
                    caption=caption,
                    parse_mode=parse_mode,
                )
        except TelegramBadRequest:
            async with span("tg-send"):
                msg = await bot.send_document(
                    channel.channel_id,
                    file,
                    caption=caption,
                )

        tg_file_id = msg.document.file_id
        tg_post_id = msg.message_id
        tg_post_url = message_link(msg.chat.id, msg.message_id)

        async with db.get_session() as session:
            obj = await session.get(Event, event_id)
            if obj:
                obj.ics_file_id = tg_file_id
                obj.ics_post_url = tg_post_url
                obj.ics_post_id = tg_post_id
                await session.commit()

        if progress:
            progress.mark("ics_telegram", "done", tg_post_url)
        logline("ICS", event_id, "telegram done", url=tg_post_url)

        try:
            await update_source_post_keyboard(event_id, db, bot)
        except Exception as e:  # pragma: no cover - logging inside
            logging.warning(
                "update_source_post_keyboard failed for %s: %s", event_id, e
            )

        return True

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


@lru_cache(maxsize=8)
def _prompt_cache(
    festival_key: tuple[str, ...] | None,
) -> str:
    txt = _read_base_prompt()
    if festival_key:
        txt += "\nUse the JSON below to normalise festival names and map aliases.\n"
    return txt


def _build_prompt(
    festival_names: Sequence[str] | None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None,
) -> str:
    festival_key = tuple(sorted(festival_names)) if festival_names else None
    prompt = _prompt_cache(festival_key)
    if festival_key:
        payload: dict[str, Any] = {"festival_names": list(festival_key)}
        alias_pairs = (
            tuple(sorted(festival_alias_pairs)) if festival_alias_pairs else None
        )
        if alias_pairs:
            prompt += (
                "\nFestival normalisation helper:\n"
                "- Compute norm(text) by casefolding, trimming, removing quotes,"
                " leading words (фестиваль/международный/областной/городской)"
                " and collapsing internal whitespace.\n"
                "- Each entry in festival_alias_pairs is [alias_norm,"
                " festival_index]; festival_index points to festival_names.\n"
                "- When norm(text) matches alias_norm, use"
                " festival_names[festival_index] as the canonical name.\n"
            )
            payload["festival_alias_pairs"] = [list(pair) for pair in alias_pairs]
        json_block = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt += json_block
    return prompt


async def parse_event_via_4o(
    text: str,
    source_channel: str | None = None,
    *,
    festival_names: Sequence[str] | None = None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None = None,
    poster_texts: Sequence[str] | None = None,
    poster_summary: str | None = None,
    **extra: str | None,
) -> list[dict]:
    token = os.getenv("FOUR_O_TOKEN")
    if not token:
        raise RuntimeError("FOUR_O_TOKEN is missing")
    url = os.getenv("FOUR_O_URL", "https://api.openai.com/v1/chat/completions")
    prompt = _build_prompt(festival_names, festival_alias_pairs)
    if poster_summary:
        prompt = f"{prompt}\nPoster summary:\n{poster_summary.strip()}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if not source_channel:
        source_channel = extra.get("channel_title")
    today = datetime.now(LOCAL_TZ).date().isoformat()
    user_msg_parts = [f"Today is {today}. "]
    if source_channel:
        user_msg_parts.append(f"Channel: {source_channel}. ")
    user_msg = "".join(user_msg_parts)
    poster_lines: list[str] = []
    if poster_texts:
        poster_lines.append(
            "Poster OCR may contain recognition mistakes; cross-check with the main text."
        )
        poster_lines.append("Poster OCR:")
        for idx, block in enumerate(poster_texts, start=1):
            poster_lines.append(f"[{idx}] {block.strip()}")
        poster_lines.append("")
    if poster_lines:
        user_msg += "\n" + "\n".join(poster_lines)
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
    call_started = _time.monotonic()
    semaphore_acquired = False
    semaphore_wait: float | None = None

    async def _call():
        nonlocal semaphore_acquired, semaphore_wait
        wait_started = _time.monotonic()
        async with span("http"):
            async with HTTP_SEMAPHORE:
                semaphore_acquired = True
                semaphore_wait = _time.monotonic() - wait_started
                resp = await session.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return await resp.json()

    try:
        data_raw = await asyncio.wait_for(_call(), FOUR_O_TIMEOUT)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        elapsed = _time.monotonic() - call_started
        setattr(
            exc,
            "_four_o_call_meta",
            {
                "elapsed": elapsed,
                "semaphore_acquired": semaphore_acquired,
                "semaphore_wait": semaphore_wait,
            },
        )
        raise
    usage = data_raw.get("usage") or {}
    _record_four_o_usage(
        "parse",
        str(payload.get("model", "unknown")),
        usage,
    )
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
        fest = None
        if isinstance(festival, str):
            fest = {"name": festival}
        elif isinstance(festival, dict):
            fest = festival.copy()
        for k in (
            "start_date",
            "end_date",
            "city",
            "location_name",
            "location_address",
            "full_name",
            "program_url",
            "website_url",
        ):
            if k in data and fest is not None and fest.get(k) in (None, ""):
                fest[k] = data[k]
        parse_event_via_4o._festival = fest
        if "events" in data and isinstance(data["events"], list):
            return data["events"]
        return [data]
    if isinstance(data, list):
        parse_event_via_4o._festival = None
        return data
    logging.error("Unexpected 4o format: %s", data)
    parse_event_via_4o._festival = None
    raise RuntimeError("bad 4o response")


FOUR_O_EDITOR_PROMPT = textwrap.dedent(
    """
    Ты — выпускающий редактор русскоязычного Telegram-канала о событиях.
    Переформатируй текст истории для публикации на Telegraph.
    Разбей материал на абзацы по 2–3 предложения и вставь понятные промежуточные подзаголовки.
    Делай только лёгкие правки: исправляй опечатки и очевидные неточности, не выдумывай новые детали.
    Добавь подходящие эмодзи к тексту.
    Сохраняй факты, даты, имена и ссылки, не добавляй новые данные.
    Используй только простой HTML или markdown, понятный Telegraph (<p>, <h3>, <ul>, <ol>, <b>, <i>, <a>, <blockquote>, <br/>).
    Не добавляй вводные комментарии, пояснения об обработке или служебные пометки — верни только готовый текст.
    """
)


FOUR_O_PITCH_PROMPT = textwrap.dedent(
    """
    Ты — редактор русскоязычного Telegram-канала о событиях.
    Твоя задача — придумать одно продающее предложение для анонса истории.
    Ориентируйся на триггеры любопытства или лёгкой интриги, когда это уместно.
    Допускай лёгкую, но ниже умеренной, гиперболизацию ради выразительности.
    Можешь использовать одно уместное эмодзи, но это необязательно.
    Излагай ярко и по делу, избегай клише и упоминаний про нейросети или сам процесс написания.
    Верни только готовое предложение без кавычек, комментариев и служебных пометок.
    """
)


async def compose_story_pitch_via_4o(text: str, *, title: str | None = None) -> str:
    """Return a single-sentence pitch using the 4o helper with graceful fallback."""

    raw = (text or "").strip()
    fallback = ""
    if raw:
        for line in raw.splitlines():
            candidate = line.strip()
            if candidate:
                fallback = candidate
                break
    if not raw:
        return fallback
    sections: list[str] = [
        "Сделай одно энергичное предложение, чтобы читатель захотел открыть историю на Telegraph.",
    ]
    if title:
        title_clean = title.strip()
        if title_clean:
            sections.append(f"Заголовок: {title_clean}")
    sections.append("Текст:\n" + raw)
    prompt_text = "\n\n".join(sections)
    try:
        response = await ask_4o(
            prompt_text,
            system_prompt=FOUR_O_PITCH_PROMPT,
            max_tokens=FOUR_O_PITCH_MAX_TOKENS,
        )
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning(
            "vk_review story pitch request failed",
            extra={"error": str(exc)},
        )
        return fallback
    candidate = (response or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        candidate = candidate.strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = candidate.strip(' "\'')
    if candidate:
        return candidate
    return fallback


async def compose_story_editorial_via_4o(text: str, *, title: str | None = None) -> str:
    """Return formatted HTML/markdown for Telegraph using the 4o editor prompt."""

    raw = (text or "").strip()
    if not raw:
        return ""
    sections: list[str] = []
    if title:
        title_clean = title.strip()
        if title_clean:
            sections.append(f"Заголовок: {title_clean}")
    sections.append("Текст:\n" + raw)
    prompt_text = "\n\n".join(sections)
    response = await ask_4o(
        prompt_text,
        system_prompt=FOUR_O_EDITOR_PROMPT,
        max_tokens=FOUR_O_EDITOR_MAX_TOKENS,
    )
    formatted = (response or "").strip()
    if formatted.startswith("```"):
        formatted = re.sub(r"^```[a-zA-Z]*\n?", "", formatted)
        if formatted.endswith("```"):
            formatted = formatted[:-3]
        formatted = formatted.strip()
    return formatted


async def ask_4o(
    text: str,
    *,
    system_prompt: str | None = None,
    response_format: dict | None = None,
    max_tokens: int = FOUR_O_RESPONSE_LIMIT,
    model: str | None = None,
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
        "model": model or "gpt-4o",
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
    usage = data.get("usage") or {}
    _record_four_o_usage(
        "ask",
        str(payload.get("model", "unknown")),
        usage,
    )
    logging.debug("4o response: %s", data)
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    del data
    gc.collect()
    return content


FESTIVAL_INFERENCE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "FestivalInference",
        "schema": {
            "type": "object",
            "properties": {
                "festival": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "full_name": {"type": ["string", "null"]},
                        "summary": {"type": ["string", "null"]},
                        "reason": {"type": ["string", "null"]},
                        "start_date": {"type": ["string", "null"]},
                        "end_date": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "location_name": {"type": ["string", "null"]},
                        "location_address": {"type": ["string", "null"]},
                        "website_url": {"type": ["string", "null"]},
                        "program_url": {"type": ["string", "null"]},
                        "ticket_url": {"type": ["string", "null"]},
                        "existing_candidates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                        },
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                "duplicate": {
                    "type": ["object", "null"],
                    "properties": {
                        "match": {"type": "boolean"},
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["match", "name", "confidence"],
                    "additionalProperties": False,
                },
            },
            "required": ["festival"],
            "additionalProperties": False,
        },
    },
}


def normalize_duplicate_name(value: str | None) -> str | None:
    """Normalize festival name returned by LLM for duplicate matching."""

    if not value:
        return None
    text = value.strip().lower()
    if not text:
        return None
    text = text.replace("\u00ab", " ").replace("\u00bb", " ")
    text = text.replace("\u201c", " ").replace("\u201d", " ")
    text = text.replace("\u201e", " ").replace("\u2019", " ")
    text = text.replace('"', " ").replace("'", " ").replace("`", " ")
    text = re.sub(r"\bфестиваль\b", " ", text)
    text = re.sub(r"\bfestival\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip() or None


async def infer_festival_for_event_via_4o(
    event: Event, known_fests: Sequence[Festival]
) -> dict[str, Any]:
    """Ask 4o to infer festival metadata for *event*."""

    def _clip(text: str | None, limit: int = 2500) -> str:
        if not text:
            return ""
        txt = text.strip()
        if len(txt) <= limit:
            return txt
        return txt[: limit - 3].rstrip() + "..."

    system_prompt = textwrap.dedent(
        """
        Ты помогаешь редактору определить фестиваль, к которому относится событие.
        Ответь JSON-объектом с полями:
        - festival: объект с ключами name (обязательное поле), full_name, summary, reason, start_date, end_date, city,
          location_name, location_address, website_url, program_url, ticket_url и existing_candidates (массив до пяти строк).
        - duplicate: объект или null. Укажи match (bool), name (строка) и confidence (доля от 0 до 1, float), если событие
          относится к одному из известных фестивалей. Если подходящих фестивалей нет, верни null.
        Используй null, если данных нет. Не добавляй других полей.
        """
    ).strip()

    parts: list[str] = [
        f"Title: {event.title}",
        f"Date: {event.date}",
    ]
    if getattr(event, "end_date", None):
        parts.append(f"End date: {event.end_date}")
    if getattr(event, "time", None):
        parts.append(f"Time: {event.time}")
    location_bits = [
        getattr(event, "location_name", "") or "",
        getattr(event, "location_address", "") or "",
        getattr(event, "city", "") or "",
    ]
    location_text = ", ".join(bit for bit in location_bits if bit)
    if location_text:
        parts.append(f"Location: {location_text}")
    description = _clip(getattr(event, "description", ""))
    if description:
        parts.append("Description:\n" + description)
    source = _clip(getattr(event, "source_text", ""), limit=4000)
    if source and source != description:
        parts.append("Original message:\n" + source)
    normalized_fest_lookup: dict[str, Festival] = {}
    known_payload = [
        {
            "id": fest.id,
            "name": fest.name,
            "full_name": fest.full_name,
            "start_date": fest.start_date,
            "end_date": fest.end_date,
            "city": fest.city,
        }
        for fest in known_fests
        if getattr(fest, "id", None)
    ]
    for fest in known_fests:
        if not getattr(fest, "id", None) or not getattr(fest, "name", None):
            continue
        normalized_name = normalize_duplicate_name(fest.name)
        if normalized_name and normalized_name not in normalized_fest_lookup:
            normalized_fest_lookup[normalized_name] = fest
    if known_payload:
        catalog = json.dumps(known_payload, ensure_ascii=False)
        parts.append("Известные фестивали (JSON):\n" + catalog)

    payload = "\n\n".join(parts)

    response = await ask_4o(
        payload,
        system_prompt=system_prompt,
        response_format=FESTIVAL_INFERENCE_RESPONSE_FORMAT,
        max_tokens=600,
    )
    try:
        data = json.loads(response or "{}")
    except json.JSONDecodeError:
        logging.error("infer_festival_for_event_via_4o invalid JSON: %s", response)
        raise
    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from festival inference")

    festival = data.get("festival")
    if not isinstance(festival, dict):
        raise ValueError("Festival block missing in inference result")

    def _clean(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
        else:
            text = str(value).strip()
        return text or None

    existing = festival.get("existing_candidates")
    if not isinstance(existing, list):
        existing = []
    else:
        existing = [str(item).strip() for item in existing if str(item).strip()]
    festival["existing_candidates"] = existing

    for field in (
        "name",
        "full_name",
        "summary",
        "reason",
        "start_date",
        "end_date",
        "city",
        "location_name",
        "location_address",
        "website_url",
        "program_url",
        "ticket_url",
    ):
        festival[field] = _clean(festival.get(field))

    if not festival.get("name"):
        raise ValueError("Festival name missing in inference result")

    def _safe_date(text: str | None) -> str | None:
        if not text:
            return None
        txt = text.strip()
        if not txt:
            return None
        return txt

    event_start: str | None = None
    event_end: str | None = None
    raw_date = getattr(event, "date", None) or ""
    if isinstance(raw_date, str) and raw_date.strip():
        if ".." in raw_date:
            start_part, end_part = raw_date.split("..", 1)
            event_start = _safe_date(start_part)
            event_end = _safe_date(end_part) or event_start
        else:
            event_start = _safe_date(raw_date)
    explicit_end = getattr(event, "end_date", None)
    if explicit_end:
        event_end = _safe_date(str(explicit_end)) or event_end
    if event_start and not event_end:
        event_end = event_start

    if not festival.get("start_date") and event_start:
        festival["start_date"] = event_start
    if not festival.get("end_date") and event_end:
        festival["end_date"] = event_end

    duplicate_raw = data.get("duplicate")
    duplicate: dict[str, Any] = {
        "match": False,
        "name": None,
        "normalized_name": None,
        "confidence": None,
        "dup_fid": None,
    }
    if isinstance(duplicate_raw, dict):
        match_flag = bool(duplicate_raw.get("match"))
        name = _clean(duplicate_raw.get("name"))
        confidence_value: float | None = None
        confidence_raw = duplicate_raw.get("confidence")
        if isinstance(confidence_raw, (int, float)):
            confidence_value = float(confidence_raw)
        elif isinstance(confidence_raw, str):
            try:
                confidence_value = float(confidence_raw.strip())
            except (TypeError, ValueError):
                confidence_value = None
        normalized_name = normalize_duplicate_name(name)
        dup_fid: int | None = None
        if match_flag and normalized_name:
            fest_obj = normalized_fest_lookup.get(normalized_name)
            if fest_obj and getattr(fest_obj, "id", None):
                dup_fid = fest_obj.id
        duplicate = {
            "match": match_flag,
            "name": name,
            "normalized_name": normalized_name,
            "confidence": confidence_value,
            "dup_fid": dup_fid,
        }
    elif duplicate_raw is not None:
        logging.debug("infer_festival_for_event_via_4o unexpected duplicate block: %s", duplicate_raw)

    return {"festival": festival, "duplicate": duplicate}


def clean_optional_str(value: Any) -> str | None:
    """Return stripped string or ``None`` for empty values."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def extract_telegraph_image_urls(page_url: str) -> list[str]:
    """Return ordered list of image URLs found on a Telegraph page."""

    def normalize(src: str | None) -> str | None:
        if not src:
            return None
        val = src.split("#", 1)[0].split("?", 1)[0]
        if val.startswith("/file/"):
            return f"https://telegra.ph{val}"
        parsed_src = urlparse(val)
        if parsed_src.scheme != "https":
            return None
        lower = parsed_src.path.lower()
        if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            return val
        return None

    raw = (page_url or "").strip()
    if not raw:
        return []
    cleaned = raw.split("#", 1)[0].split("?", 1)[0]
    parsed = urlparse(cleaned)
    path: str | None
    if parsed.scheme:
        host = parsed.netloc.lower()
        if host not in {"telegra.ph", "te.legra.ph"}:
            return []
        path = parsed.path.lstrip("/")
    else:
        trimmed = cleaned.lstrip("/")
        if trimmed.startswith("telegra.ph/"):
            trimmed = trimmed.split("/", 1)[1] if "/" in trimmed else ""
        elif trimmed.startswith("te.legra.ph/"):
            trimmed = trimmed.split("/", 1)[1] if "/" in trimmed else ""
        path = trimmed
    if not path:
        return []
    path = path.lstrip("/")
    if not path:
        return []
    api_url = f"https://api.telegra.ph/getPage/{path}?return_content=true"
    timeout = httpx.Timeout(HTTP_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(api_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.warning("telegraph image fetch failed: %s", exc)
        return []
    content = data.get("result", {}).get("content") or []
    results: list[str] = []

    async def dfs(nodes: list[Any]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            tag = node.get("tag")
            attrs = node.get("attrs") or {}
            if tag == "img":
                normalized = normalize(attrs.get("src"))
                if normalized and normalized not in results:
                    results.append(normalized)
            if tag == "a":
                normalized = normalize(attrs.get("href"))
                if normalized and normalized not in results:
                    results.append(normalized)
            children = node.get("children") or []
            if children:
                await dfs(children)

    await dfs(content)
    return results


_EVENT_TOPIC_LISTING = "\n".join(
    f"- {topic} — «{label}»" for topic, label in TOPIC_LABELS.items()
)

EVENT_TOPIC_SYSTEM_PROMPT = textwrap.dedent(
    f"""
    Ты — ассистент, который классифицирует культурные события по темам.
    Верни JSON с массивом `topics`: выбери от 0 до 3 подходящих идентификаторов тем.
    Используй только идентификаторы из списка ниже, записывай их ровно так, как показано, и не добавляй другие значения.
    Не отмечай темы про скидки, «Бесплатно» или бесплатное участие и игнорируй «Фестивали», сетевые программы и серии мероприятий.
    Не повторяй одинаковые идентификаторы.
    Если в названии, описании или хэштегах явно указан возрастной ценз (например, «18+», «18 +», «(16+)», «16-18», «12–14 лет», «от 14 лет», «18 лет и старше», «21+ only»), то не выбирай темы `FAMILY` и `KIDS_SCHOOL`.
    Возрастной ценз может записываться как число со знаком «+» (включая варианты с пробелами или скобками), как диапазон («12-16», «12–16 лет») или словами «от N лет», «N лет и старше», «для N+».
    Допустимые темы:
    {_EVENT_TOPIC_LISTING}
    Если ни одна тема не подходит, верни пустой массив.
    Для театральных событий уточняй подтипы: `THEATRE_CLASSIC` ставь за постановки по канону — пьесы классических авторов (например, Шекспир, Мольер, Пушкин, Гоголь), исторические или мифологические сюжеты, традиционная драматургия; `THEATRE_MODERN` применяй к новой драме, современным текстам, экспериментальным, иммерсивным или мультимедийным форматам.
    Если классический сюжет переосмыслен в современном или иммерсивном исполнении, ставь обе темы `THEATRE_CLASSIC` и `THEATRE_MODERN`.
    """
).strip()

EVENT_TOPIC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventTopics",
        "schema": {
            "type": "object",
            "properties": {
                "topics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(TOPIC_LABELS.keys()),
                    },
                    "maxItems": 3,
                    "uniqueItems": True,
                }
            },
            "required": ["topics"],
            "additionalProperties": False,
        },
    },
}


def _extract_available_hashtags(event: Event) -> list[str]:
    text_sources = [
        getattr(event, "description", "") or "",
        getattr(event, "source_text", "") or "",
    ]
    seen: dict[str, None] = {}
    for chunk in text_sources:
        if not chunk:
            continue
        for match in re.findall(r"#[\w\d_]+", chunk, flags=re.UNICODE):
            normalized = match.strip()
            if normalized and normalized not in seen:
                seen[normalized] = None
    return list(seen.keys())


async def classify_event_topics(event: Event) -> list[str]:
    allowed_topics = TOPIC_IDENTIFIERS
    title = (getattr(event, "title", "") or "").strip()
    descriptions: list[str] = []
    for attr in ("description", "source_text"):
        value = getattr(event, attr, "") or ""
        value = value.strip()
        if not value:
            continue
        descriptions.append(value)
    description_text = "\n\n".join(descriptions)
    if len(description_text) > FOUR_O_PROMPT_LIMIT:
        description_text = description_text[:FOUR_O_PROMPT_LIMIT]
    hashtags = _extract_available_hashtags(event)
    location_parts = [
        (getattr(event, "city", "") or "").strip(),
        (getattr(event, "location_name", "") or "").strip(),
        (getattr(event, "location_address", "") or "").strip(),
    ]
    location_text = ", ".join(part for part in location_parts if part)
    sections: list[str] = []
    if title:
        sections.append(f"Название: {title}")
    if description_text:
        sections.append(f"Описание:\n{description_text}")
    if hashtags:
        sections.append("Хэштеги: " + ", ".join(hashtags))
    if location_text:
        sections.append(f"Локация: {location_text}")
    prompt_text = "\n\n".join(sections).strip()
    logger.info(
        "Classify topics prompt lengths: title=%s desc=%s hashtags=%s location=%s total=%s",
        len(title),
        len(description_text),
        len(", ".join(hashtags)) if hashtags else 0,
        len(location_text),
        len(prompt_text),
    )
    model_name = "gpt-4o-mini" if os.getenv("FOUR_O_MINI") == "1" else None
    try:
        raw = await ask_4o(
            prompt_text,
            system_prompt=EVENT_TOPIC_SYSTEM_PROMPT,
            response_format=EVENT_TOPIC_RESPONSE_FORMAT,
            max_tokens=FOUR_O_RESPONSE_LIMIT,
            model=model_name,
        )
    except Exception as exc:
        logging.warning("Topic classification request failed: %s", exc)
        return []
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except Exception as exc:
        logging.warning("Topic classification JSON parse failed: %s", exc)
        return []
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        logging.warning("Topic classification response missing list: %s", raw)
        return []
    result: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        canonical = normalize_topic_identifier(topic)
        if canonical is None or canonical not in allowed_topics:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)
        if len(result) >= 3:
            break
    return result


def _event_topic_text_length(event: Event) -> int:
    parts = [
        getattr(event, "title", "") or "",
        getattr(event, "description", "") or "",
        getattr(event, "source_text", "") or "",
    ]
    return sum(len(part) for part in parts)


_KALININGRAD_TOPIC_ID = "KRAEVEDENIE_KALININGRAD_OBLAST"
_KALININGRAD_CITY_KEYWORDS: tuple[str, ...] = (
    "калининград",
    "светлогорск",
    "зеленоградск",
    "пионерский",
    "янтарный",
    "балтийск",
    "светлый",
    "гвардейск",
    "советск",
    "гусев",
    "черняховск",
    "полесск",
    "неман",
    "нестеров",
    "правдинск",
    "ладушкин",
    "багратионовск",
    "мамоново",
    "славск",
    "краснознаменск",
)
_KALININGRAD_REGION_KEYWORDS: tuple[str, ...] = (
    "калининград",
    "kaliningrad",
    "калининградская область",
    "калининградской области",
    "калининградскаяобласть",
    "калининградскойобласти",
    "кёнигсберг",
    "кенигсберг",
    "kenigsberg",
    "königsberg",
    "koenigsberg",
    "konigsberg",
    "kenig",
    "янтарный край",
    "янтарного края",
    "39 регион",
    "39регион",
    "39-й регион",
    "39й регион",
    "39йрегион",
    "#калининград",
    "#kaliningrad",
    "#kenig",
    "#kenigsberg",
    "#кёнигсберг",
    "#кенигсберг",
    "#39регион",
    "#39region",
    "#39йрегион",
    "39rus",
    "klgd.ru",
    "klgd",
)
_KALININGRAD_HASHTAGS: set[str] = {
    "#калининград",
    "#kaliningrad",
    "#kenig",
    "#kenigsberg",
    "#кёнигсберг",
    "#кенигсберг",
    "#39регион",
    "#39йрегион",
    "#39region",
    "#янтарныйкрай",
}
_KALININGRAD_URL_KEYWORDS: tuple[str, ...] = (
    "kaliningrad",
    "kenig",
    "kenigsberg",
    "koenigsberg",
    "konigsberg",
    "königsberg",
    "klgd",
)


def _contains_kaliningrad_city(value: str | None) -> bool:
    if not value or not isinstance(value, str):
        return False
    lowered = value.casefold()
    return any(keyword in lowered for keyword in _KALININGRAD_CITY_KEYWORDS)


def _contains_kaliningrad_keyword(value: str | None) -> bool:
    if not value or not isinstance(value, str):
        return False
    lowered = value.casefold()
    return any(keyword in lowered for keyword in _KALININGRAD_REGION_KEYWORDS)


def _should_add_kaliningrad_topic(event: Event, topics: Iterable[str]) -> bool:
    normalized_topics = {
        normalize_topic_identifier(topic) or topic for topic in topics
    }
    if _KALININGRAD_TOPIC_ID in normalized_topics:
        return False

    if _contains_kaliningrad_city(getattr(event, "city", None)):
        return True

    text_fields = [
        getattr(event, "location_address", None),
        getattr(event, "location_name", None),
        getattr(event, "description", None),
        getattr(event, "source_text", None),
        getattr(event, "title", None),
    ]
    for value in text_fields:
        if _contains_kaliningrad_keyword(value):
            return True

    for hashtag in _extract_available_hashtags(event):
        if hashtag.casefold() in _KALININGRAD_HASHTAGS:
            return True

    url_fields = [
        getattr(event, "source_post_url", None),
        getattr(event, "ticket_link", None),
        getattr(event, "vk_repost_url", None),
    ]
    for value in url_fields:
        if not value or not isinstance(value, str):
            continue
        lowered = value.casefold()
        if any(keyword in lowered for keyword in _KALININGRAD_URL_KEYWORDS):
            return True

    return False


def _apply_topic_postprocessors(event: Event, topics: list[str]) -> list[str]:
    processed = list(dict.fromkeys(topics))
    if _should_add_kaliningrad_topic(event, processed):
        processed.append(_KALININGRAD_TOPIC_ID)
    return processed


async def assign_event_topics(event: Event) -> tuple[list[str], int, str | None, bool]:
    """Populate ``event.topics`` using automatic classification."""

    text_length = _event_topic_text_length(event)
    if getattr(event, "topics_manual", False):
        current = list(getattr(event, "topics", []) or [])
        return current, text_length, None, True

    try:
        topics = await classify_event_topics(event)
        error_text: str | None = None
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Topic classification raised an exception: %s", exc)
        topics = []
        error_text = str(exc)

    topics = _apply_topic_postprocessors(event, topics)
    event.topics = topics
    event.topics_manual = False
    return topics, text_length, error_text, False


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
            [
                types.KeyboardButton(text=MENU_ADD_EVENT),
                types.KeyboardButton(text=MENU_ADD_FESTIVAL),
            ],
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


async def handle_help(message: types.Message, db: Database, bot: Bot) -> None:
    """Send command list according to user role."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    role = "guest"
    if user and not user.blocked:
        role = "superadmin" if user.is_superadmin else "user"
    lines = [
        f"{item['usage']} — {item['desc']}"
        for item in HELP_COMMANDS
        if role in item["roles"]
    ]
    await bot.send_message(message.chat.id, "\n".join(lines) or "No commands available")


async def handle_ocrtest(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return

    session = get_http_session()
    await vision_test.start(
        message,
        bot,
        http_session=session,
        http_semaphore=HTTP_SEMAPHORE,
    )


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
        now = datetime.now(timezone.utc)
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
        handled = False
        if marker == "exh":
            chunks, markup = await build_exhibitions_message(db, tz)
            if not chunks:
                chunks = [""]
            first_text = chunks[0]
            chat = callback.message.chat if callback.message else None
            chat_id = chat.id if chat else None
            stored = (
                exhibitions_message_state.get(chat_id, [])
                if chat_id is not None
                else []
            )
            first_id = callback.message.message_id if callback.message else None
            prev_followups: list[tuple[int, str]] = []
            if stored and first_id is not None:
                for mid, prev_text in stored:
                    if mid != first_id:
                        prev_followups.append((mid, prev_text))
            else:
                prev_followups = stored

            if callback.message:
                with contextlib.suppress(TelegramBadRequest):
                    await callback.message.edit_text(first_text, reply_markup=markup)

            new_followups: list[tuple[int, str]] = []
            if chat_id is not None:
                for idx, chunk in enumerate(chunks[1:]):
                    if idx < len(prev_followups):
                        mid, prev_text = prev_followups[idx]
                        if chunk != prev_text:
                            with contextlib.suppress(TelegramBadRequest):
                                await bot.edit_message_text(
                                    chunk, chat_id=chat_id, message_id=mid
                                )
                        new_followups.append((mid, chunk))
                    else:
                        msg = await bot.send_message(chat_id, chunk)
                        new_followups.append((msg.message_id, chunk))

                for mid, _ in prev_followups[len(new_followups) :]:
                    with contextlib.suppress(TelegramBadRequest):
                        await bot.delete_message(chat_id, mid)

                updated_state: list[tuple[int, str]] = []
                if first_id is not None:
                    updated_state.append((first_id, first_text))
                updated_state.extend(new_followups)
                exhibitions_message_state[chat_id] = updated_state

            handled = True
        else:
            target = datetime.strptime(marker, "%Y-%m-%d").date()
            filter_id = user.user_id if user and user.is_partner else None
            text, markup = await build_events_message(db, target, tz, filter_id)
        if not handled and callback.message:
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
    elif data.startswith("makefest:"):
        parts = data.split(":")
        if len(parts) < 2:
            await callback.answer("Некорректный запрос", show_alert=True)
            return
        eid = int(parts[1])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            known_fests = (await session.execute(select(Festival))).scalars().all()
        if not event:
            await callback.answer("Событие не найдено", show_alert=True)
            return
        if event.festival:
            await callback.answer("У события уже есть фестиваль", show_alert=True)
            return
        if user and (user.blocked or (user.is_partner and event.creator_id != user.user_id)):
            await callback.answer("Not authorized", show_alert=True)
            return
        try:
            state_payload = await _build_makefest_session_state(event, known_fests)
        except Exception as exc:  # pragma: no cover - network / LLM failures
            logging.exception("makefest inference failed for %s: %s", eid, exc)
            await callback.message.answer(
                "Не удалось получить подсказку от модели. Попробуйте позже."
            )
            await callback.answer()
            return

        makefest_sessions[callback.from_user.id] = {
            "event_id": eid,
            **state_payload,
        }

        fest_data = state_payload["festival"]
        duplicate_info = state_payload["duplicate"]
        photo_candidates = state_payload.get("photos", [])
        matches = state_payload.get("matches", [])

        def _short(text: str | None, limit: int = 400) -> str:
            if not text:
                return ""
            txt = text.strip()
            if len(txt) <= limit:
                return txt
            return txt[: limit - 3].rstrip() + "..."

        lines = ["\U0001f3aa Предпросмотр фестиваля", f"Событие: {event.title}"]
        if event.date:
            lines.append(f"Дата события: {event.date}")
        lines.append(f"Название: {fest_data['name']}")
        if fest_data.get("full_name"):
            lines.append(f"Полное название: {fest_data['full_name']}")
        if fest_data.get("summary"):
            lines.append(_short(fest_data.get("summary")))
        period_bits = [bit for bit in [fest_data.get("start_date"), fest_data.get("end_date")] if bit]
        if period_bits:
            if len(period_bits) == 2 and period_bits[0] != period_bits[1]:
                lines.append(f"Период: {period_bits[0]} — {period_bits[1]}")
            else:
                lines.append(f"Дата фестиваля: {period_bits[0]}")
        place_bits = [
            fest_data.get("location_name"),
            fest_data.get("location_address"),
            fest_data.get("city"),
        ]
        place_text = ", ".join(bit for bit in place_bits if bit)
        if place_text:
            lines.append(f"Локация: {place_text}")
        if fest_data.get("reason"):
            lines.append("Почему: " + _short(fest_data.get("reason")))
        def _format_confidence(value: float | None) -> str | None:
            if value is None:
                return None
            if 0 <= value <= 1:
                return f"{value * 100:.0f}%"
            return f"{value:.2f}"

        if duplicate_info.get("name"):
            dup_line = f"Похоже на: {duplicate_info['name']}"
            conf_text = _format_confidence(duplicate_info.get("confidence"))
            if conf_text:
                dup_line += f" (уверенность {conf_text})"
            lines.append(dup_line)
        if photo_candidates:
            lines.append(f"Фото для альбома: {len(photo_candidates)} шт.")
        if matches:
            lines.append("Возможные совпадения:")
            for match in matches:
                name = match.get("name")
                if name:
                    lines.append(f" • {name}")
        lines.append("\nВыберите действие ниже.")
        buttons = [
            [
                types.InlineKeyboardButton(
                    text="✅ Создать и привязать", callback_data=f"makefest_create:{eid}"
                )
            ]
        ]
        if duplicate_info.get("dup_fid"):
            label = duplicate_info.get("name") or "найденному фестивалю"
            conf_text = _format_confidence(duplicate_info.get("confidence"))
            if conf_text:
                label += f" ({conf_text})"
            label = f"… {label}" if label else "…"
            buttons.append(
                [
                    types.InlineKeyboardButton(
                        text=f"🔗 Привязать к {label}",
                        callback_data=f"makefest_bind:{eid}:{duplicate_info['dup_fid']}",
                    )
                ]
            )
        if matches:
            buttons.append(
                [
                    types.InlineKeyboardButton(
                        text="Выбрать другой фестиваль",
                        callback_data=f"makefest_bind:{eid}",
                    )
                ]
            )
        buttons.append(
            [types.InlineKeyboardButton(text="Отмена", callback_data=f"edit:{eid}")]
        )
        markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
        await callback.message.answer("\n".join(lines), reply_markup=markup)
        await callback.answer()
    elif data.startswith("makefest_create:"):
        parts = data.split(":")
        if len(parts) < 2:
            await callback.answer("Некорректный запрос", show_alert=True)
            return
        eid = int(parts[1])
        state = makefest_sessions.get(callback.from_user.id)
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            known_fests = (await session.execute(select(Festival))).scalars().all()
            if not event or (
                user
                and (user.blocked or (user.is_partner and event.creator_id != user.user_id))
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
        if not state or state.get("event_id") != eid:
            try:
                state_payload = await _build_makefest_session_state(event, known_fests)
            except Exception as exc:  # pragma: no cover - network / LLM failures
                logging.exception("makefest inference failed for %s: %s", eid, exc)
                await callback.message.answer(
                    "Не удалось получить подсказку от модели. Попробуйте позже."
                )
                await callback.answer()
                return
            state = {"event_id": eid, **state_payload}
            makefest_sessions[callback.from_user.id] = state
        fest_data = state["festival"]
        photos: list[str] = state.get("photos", [])

        fest_obj, created, _ = await ensure_festival(
            db,
            fest_data["name"],
            full_name=clean_optional_str(fest_data.get("full_name")),
            photo_url=photos[0] if photos else None,
            photo_urls=photos,
            website_url=clean_optional_str(fest_data.get("website_url")),
            program_url=clean_optional_str(fest_data.get("program_url")),
            ticket_url=clean_optional_str(fest_data.get("ticket_url")),
            start_date=clean_optional_str(fest_data.get("start_date")),
            end_date=clean_optional_str(fest_data.get("end_date")),
            location_name=clean_optional_str(fest_data.get("location_name")),
            location_address=clean_optional_str(fest_data.get("location_address")),
            city=clean_optional_str(fest_data.get("city")),
            source_text=event.source_text,
            source_post_url=event.source_post_url,
            source_chat_id=event.source_chat_id,
            source_message_id=event.source_message_id,
        )
        async with db.get_session() as session:
            event = await session.get(Event, eid)
            if not event:
                await callback.answer("Событие не найдено", show_alert=True)
                return
            event.festival = fest_obj.name
            session.add(event)
            await session.commit()
        makefest_sessions.pop(callback.from_user.id, None)
        await schedule_event_update_tasks(db, event, skip_vk_sync=True)
        asyncio.create_task(sync_festival_page(db, fest_obj.name))
        asyncio.create_task(sync_festivals_index_page(db))
        status = "создан" if created else "обновлён"
        text, markup = await _build_makefest_response(
            db, fest_obj, status=status, photo_count=len(photos)
        )
        await callback.message.answer(text, reply_markup=markup)
        await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer("Готово")
    elif data.startswith("makefest_bind:"):
        parts = data.split(":")
        if len(parts) < 2:
            await callback.answer("Некорректный запрос", show_alert=True)
            return
        eid = int(parts[1])
        state = makefest_sessions.get(callback.from_user.id)
        if not state or state.get("event_id") != eid:
            await callback.answer("Предпросмотр не найден", show_alert=True)
            return
        if len(parts) == 2:
            matches = state.get("matches", [])
            if not matches:
                await callback.answer("Подходящих фестивалей не нашли", show_alert=True)
                return
            keyboard = [
                [
                    types.InlineKeyboardButton(
                        text=match["name"],
                        callback_data=f"makefest_bind:{eid}:{match['id']}",
                    )
                ]
                for match in matches
            ]
            keyboard.append(
                [types.InlineKeyboardButton(text="Отмена", callback_data=f"edit:{eid}")]
            )
            await callback.message.answer(
                "Выберите фестиваль для привязки",
                reply_markup=types.InlineKeyboardMarkup(inline_keyboard=keyboard),
            )
            await callback.answer()
            return
        fest_id = int(parts[2])
        async with db.get_session() as session:
            user = await session.get(User, callback.from_user.id)
            event = await session.get(Event, eid)
            fest = await session.get(Festival, fest_id)
            if not event or not fest or (
                user
                and (user.blocked or (user.is_partner and event.creator_id != user.user_id))
            ):
                await callback.answer("Not authorized", show_alert=True)
                return
        fest_data = state["festival"]
        photos: list[str] = state.get("photos", [])
        fest_obj, _, _ = await ensure_festival(
            db,
            fest.name,
            full_name=clean_optional_str(fest_data.get("full_name")),
            photo_url=photos[0] if photos else None,
            photo_urls=photos,
            website_url=clean_optional_str(fest_data.get("website_url")),
            program_url=clean_optional_str(fest_data.get("program_url")),
            ticket_url=clean_optional_str(fest_data.get("ticket_url")),
            start_date=clean_optional_str(fest_data.get("start_date")),
            end_date=clean_optional_str(fest_data.get("end_date")),
            location_name=clean_optional_str(fest_data.get("location_name")),
            location_address=clean_optional_str(fest_data.get("location_address")),
            city=clean_optional_str(fest_data.get("city")),
            source_text=event.source_text,
            source_post_url=event.source_post_url,
            source_chat_id=event.source_chat_id,
            source_message_id=event.source_message_id,
        )
        async with db.get_session() as session:
            event = await session.get(Event, eid)
            fest = await session.get(Festival, fest_id)
            if not event or not fest:
                await callback.answer("Not authorized", show_alert=True)
                return
            event.festival = fest.name
            session.add(event)
            await session.commit()
        makefest_sessions.pop(callback.from_user.id, None)
        await schedule_event_update_tasks(db, event, skip_vk_sync=True)
        asyncio.create_task(sync_festival_page(db, fest.name))
        asyncio.create_task(sync_festivals_index_page(db))
        text, markup = await _build_makefest_response(
            db,
            fest_obj,
            status="привязан к существующему",
            photo_count=len(photos),
        )
        await callback.message.answer(text, reply_markup=markup)
        await show_edit_menu(callback.from_user.id, event, bot)
        await callback.answer("Готово")
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
            old_start = fest.start_date
            old_end = fest.end_date
            if not start or not end:
                await callback.answer(
                    "Не задан период фестиваля. Сначала отредактируйте даты.",
                    show_alert=True,
                )
                return
            logging.info("festdays start fid=%s name=%s", fid, fest.name)
            city_from_name = parse_city_from_fest_name(fest.name)
            city_for_days = (fest.city or city_from_name or "").strip()
            if not city_for_days:
                logging.warning(
                    "festdays: city unresolved for fest %s (id=%s)",
                    fest.name,
                    fest.id,
                )
            elif city_from_name and fest.city and city_from_name.strip() != fest.city.strip():
                logging.warning(
                    "festdays: city mismatch name=%s fest.city=%s using=%s",
                    city_from_name,
                    fest.city,
                    city_from_name.strip(),
                )
                city_for_days = city_from_name.strip()
            if not fest.city and city_for_days:
                fest.city = city_for_days
            logging.info(
                "festdays: use city=%s for fest id=%s name=%s",
                city_for_days,
                fest.id,
                fest.name,
            )
            add_source = (
                (end - start).days == 0
                and bool(fest.source_post_url)
                and bool(fest.source_chat_id)
                and bool(fest.source_message_id)
            )
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
                    city=city_for_days,
                    source_text=f"{fest.name} — {day.isoformat()}",
                    source_post_url=fest.source_post_url if add_source else None,
                    source_chat_id=fest.source_chat_id if add_source else None,
                    source_message_id=fest.source_message_id if add_source else None,
                    creator_id=user.user_id if user else None,
                )
                saved, added = await upsert_event(session, event)
                await schedule_event_update_tasks(db, saved)
                events.append((saved, added))
            await session.commit()
        async with db.get_session() as session:
            notify_user = await session.get(User, callback.from_user.id)
            fresh = await session.get(Festival, fest.id)
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
        if fresh and (fresh.start_date != old_start or fresh.end_date != old_end):
            await rebuild_fest_nav_if_changed(db)
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
    elif data.startswith("festpage:"):
        parts = data.split(":")
        page = 1
        mode = "active"
        if len(parts) > 1:
            try:
                page = int(parts[1])
            except ValueError:
                page = 1
        if len(parts) > 2 and parts[2] in {"active", "archive"}:
            mode = parts[2]
        await send_festivals_list(
            callback.message,
            db,
            bot,
            user_id=callback.from_user.id,
            edit=True,
            page=page,
            archive=(mode == "archive"),
        )
        await callback.answer()
    elif data.startswith("festdel:"):
        parts = data.split(":")
        fid = int(parts[1]) if len(parts) > 1 else 0
        page = 1
        mode = "active"
        if len(parts) > 2:
            try:
                page = int(parts[2])
            except ValueError:
                page = 1
        if len(parts) > 3 and parts[3] in {"active", "archive"}:
            mode = parts[3]
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
        await send_festivals_list(
            callback.message,
            db,
            bot,
            user_id=callback.from_user.id,
            edit=True,
            page=page,
            archive=(mode == "archive"),
        )
        await callback.answer("Deleted")

    elif data.startswith("festcover:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        log_festcover(
            logging.INFO,
            fest.id,
            "request",
            initiator=callback.from_user.id,
            force=True,
            program_url=fest.program_url,
        )
        ok = await try_set_fest_cover_from_program(db, fest, force=True)
        log_festcover(
            logging.INFO,
            fest.id,
            "result",
            initiator=callback.from_user.id,
            success=ok,
        )
        msg = "Обложка обновлена" if ok else "Картинка не найдена"
        await callback.message.answer(msg)
        await callback.answer()
    elif data.startswith("festimgs:"):
        fid = int(data.split(":")[1])
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        photo_urls = list(fest.photo_urls or [])
        total = len(photo_urls)
        current = (
            photo_urls.index(fest.photo_url) + 1
            if fest.photo_url in photo_urls
            else 0
        )
        text = (
            "Иллюстрации фестиваля\n"
            f"Всего: {total}\n"
            f"Текущая обложка: #{current}\n"
            "Выберите новое изображение обложки:"
        )
        buttons = [
            types.InlineKeyboardButton(
                text=f"#{i+1}", callback_data=f"festsetcover:{fid}:{i+1}"
            )
            for i in range(total)
        ]
        keyboard = [buttons[i : i + 5] for i in range(0, len(buttons), 5)]
        keyboard.append(
            [types.InlineKeyboardButton(text="Отмена", callback_data=f"festedit:{fid}")]
        )
        markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
        await callback.message.answer(text, reply_markup=markup)
        await callback.answer()
    elif data.startswith("festsetcover:"):
        _, fid, idx = data.split(":")
        fid_i = int(fid)
        idx_i = int(idx)
        async with db.get_session() as session:
            fest = await session.get(Festival, fid_i)
            photo_urls = list(fest.photo_urls or []) if fest else []
            if not fest or idx_i < 1 or idx_i > len(photo_urls):
                await callback.answer("Invalid selection", show_alert=True)
                return
            fest.photo_url = photo_urls[idx_i - 1]
            await session.commit()
            name = fest.name
        asyncio.create_task(sync_festival_page(db, name))
        asyncio.create_task(sync_festivals_index_page(db))
        await callback.message.answer(
            f"Обложка изменена на #{idx_i}.\nСтраницы фестиваля и лэндинг обновлены."
        )
        await callback.answer()

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
        await enqueue_job(db, eid, JobTask.ics_publish)
        await callback.answer("Enqueued")
    elif data.startswith("delics:"):
        eid = int(data.split(":")[1])
        async with db.get_session() as session:
            ev = await session.get(Event, eid)
            if ev:
                ev.ics_url = None
                ev.ics_file_id = None
                ev.ics_hash = None
                ev.ics_updated_at = None
                ev.vk_ics_short_url = None
                ev.vk_ics_short_key = None
                await session.commit()
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
    elif data.startswith("tourist:"):
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        source = _determine_tourist_source(callback)
        if action in {"yes", "no"}:
            try:
                event_id = int(parts[2])
            except (ValueError, IndexError):
                event_id = 0
            if not event_id:
                await callback.answer("Некорректное событие", show_alert=True)
                return
            async with db.get_session() as session:
                user = await session.get(User, callback.from_user.id)
                event = await session.get(Event, event_id)
                if not event or not _user_can_label_event(user):
                    await callback.answer("Not authorized", show_alert=True)
                    return
                event.tourist_label = 1 if action == "yes" else 0
                event.tourist_label_by = callback.from_user.id
                event.tourist_label_at = datetime.now(timezone.utc)
                event.tourist_label_source = "operator"
                session.add(event)
                await session.commit()
                await session.refresh(event)
            tourist_note_sessions.pop(callback.from_user.id, None)
            logging.info(
                "tourist_label_update",
                extra={
                    "event_id": event_id,
                    "user_id": callback.from_user.id,
                    "value": action,
                },
            )
            if action == "yes" and callback.message:
                session_state = TouristReasonSession(
                    event_id=event_id,
                    chat_id=callback.message.chat.id,
                    message_id=callback.message.message_id,
                    source=source,
                )
                tourist_reason_sessions[callback.from_user.id] = session_state
                await update_tourist_message(callback, bot, event, source, menu=True)
                await callback.answer("Отмечено")
            else:
                tourist_reason_sessions.pop(callback.from_user.id, None)
                await update_tourist_message(
                    callback,
                    bot,
                    event,
                    source,
                    menu=_is_tourist_menu_markup(callback.message.reply_markup),
                )
                await callback.answer("Отмечено")
        elif action == "fx":
            code = parts[2] if len(parts) > 2 else ""
            try:
                event_id = int(parts[3])
            except (ValueError, IndexError):
                event_id = 0
            if not event_id:
                await callback.answer("Некорректное событие", show_alert=True)
                return
            try:
                session_state = tourist_reason_sessions[callback.from_user.id]
            except KeyError:
                await _restore_tourist_reason_keyboard(
                    callback, bot, db, event_id, source
                )
                await callback.answer(
                    "Сессия истекла, откройте причины заново"
                )
                return
            if session_state.event_id != event_id:
                tourist_reason_sessions.pop(callback.from_user.id, None)
                await _restore_tourist_reason_keyboard(
                    callback, bot, db, event_id, source
                )
                await callback.answer(
                    "Сессия истекла, откройте причины заново"
                )
                return
            async with db.get_session() as session:
                user = await session.get(User, callback.from_user.id)
                event = await session.get(Event, event_id)
                if not event or not _user_can_label_event(user):
                    await callback.answer("Not authorized", show_alert=True)
                    return
                factor = TOURIST_FACTOR_BY_CODE.get(
                    TOURIST_FACTOR_ALIASES.get(code, code)
                )
                if not factor:
                    await callback.answer("Неизвестная причина", show_alert=True)
                    return
                effective_code = factor.code
                factors = _normalize_tourist_factors(event.tourist_factors or [])
                if effective_code in factors:
                    factors = [item for item in factors if item != effective_code]
                else:
                    factors.append(effective_code)
                ordered = _normalize_tourist_factors(factors)
                event.tourist_factors = ordered
                event.tourist_label_by = callback.from_user.id
                event.tourist_label_at = datetime.now(timezone.utc)
                event.tourist_label_source = "operator"
                session.add(event)
                await session.commit()
                await session.refresh(event)
            tourist_reason_sessions[callback.from_user.id] = TouristReasonSession(
                event_id=session_state.event_id,
                chat_id=session_state.chat_id,
                message_id=session_state.message_id,
                source=session_state.source,
            )
            logging.info(
                "tourist_factor_toggle",
                extra={
                    "event_id": event_id,
                    "user_id": callback.from_user.id,
                    "factor": effective_code,
                },
            )
            await update_tourist_message(
                callback, bot, event, session_state.source, menu=True
            )
            await callback.answer("Отмечено")
        elif action in {"fxdone", "fxskip"}:
            try:
                event_id = int(parts[2])
            except (ValueError, IndexError):
                event_id = 0
            if not event_id:
                await callback.answer("Некорректное событие", show_alert=True)
                return
            async with db.get_session() as session:
                user = await session.get(User, callback.from_user.id)
                event = await session.get(Event, event_id)
                if not event or not _user_can_label_event(user):
                    await callback.answer("Not authorized", show_alert=True)
                    return
            session_state = tourist_reason_sessions.get(callback.from_user.id)
            if action == "fxdone" and (
                not session_state
                or session_state.event_id != event_id
                or (callback.message and session_state.message_id != callback.message.message_id)
            ):
                if callback.message:
                    tourist_reason_sessions[callback.from_user.id] = TouristReasonSession(
                        event_id=event_id,
                        chat_id=callback.message.chat.id,
                        message_id=callback.message.message_id,
                        source=source,
                    )
                    await update_tourist_message(callback, bot, event, source, menu=True)
                await callback.answer("Выберите причины")
                return
            session_state = tourist_reason_sessions.pop(callback.from_user.id, None)
            session_source = session_state.source if session_state else source
            await update_tourist_message(
                callback, bot, event, session_source, menu=False
            )
            if action == "fxdone":
                await callback.answer("Причины сохранены")
            else:
                await callback.answer("Причины можно выбрать позже")
        elif action == "note":
            note_action = parts[2] if len(parts) > 2 else ""
            try:
                event_id = int(parts[3])
            except (ValueError, IndexError):
                await callback.answer("Некорректное событие", show_alert=True)
                return
            if note_action == "start":
                async with db.get_session() as session:
                    user = await session.get(User, callback.from_user.id)
                    event = await session.get(Event, event_id)
                    if not event or not _user_can_label_event(user):
                        await callback.answer("Not authorized", show_alert=True)
                        return
                tourist_note_sessions.pop(callback.from_user.id, None)
                tourist_note_sessions[callback.from_user.id] = TouristNoteSession(
                    event_id=event_id,
                    chat_id=callback.message.chat.id,
                    message_id=callback.message.message_id,
                    source=source,
                    markup=callback.message.reply_markup,
                    message_text=(
                        callback.message.text
                        if callback.message and callback.message.text is not None
                        else (
                            callback.message.caption
                            if callback.message
                            else None
                        )
                    ),
                    menu=_is_tourist_menu_markup(callback.message.reply_markup),
                )
                await bot.send_message(
                    callback.message.chat.id,
                    "Отправьте комментарий для туристов одним сообщением. Сессия длится 10 минут.",
                )
                await callback.answer("Ожидаю")
            elif note_action == "clear":
                async with db.get_session() as session:
                    user = await session.get(User, callback.from_user.id)
                    event = await session.get(Event, event_id)
                    if not event or not _user_can_label_event(user):
                        await callback.answer("Not authorized", show_alert=True)
                        return
                    event.tourist_note = None
                    event.tourist_label_by = callback.from_user.id
                    event.tourist_label_at = datetime.now(timezone.utc)
                    event.tourist_label_source = "operator"
                    session.add(event)
                    await session.commit()
                    await session.refresh(event)
                tourist_note_sessions.pop(callback.from_user.id, None)
                logging.info(
                    "tourist_note_cleared",
                    extra={"event_id": event_id, "user_id": callback.from_user.id},
                )
                await update_tourist_message(
                    callback,
                    bot,
                    event,
                    source,
                    menu=_is_tourist_menu_markup(callback.message.reply_markup),
                )
                await callback.answer("Отмечено")
            else:
                await callback.answer()
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
        parts = data.split(":")
        fid = int(parts[1]) if len(parts) > 1 else 0
        page = 1
        mode = "active"
        if len(parts) > 2:
            try:
                page = int(parts[2])
            except ValueError:
                page = 1
        if len(parts) > 3 and parts[3] in {"active", "archive"}:
            mode = parts[3]
        async with db.get_session() as session:
            fest = await session.get(Festival, fid)
            if fest:
                await session.delete(fest)
                await session.commit()
                logging.info("festival %s deleted", fest.name)
        await send_festivals_list(
            callback.message,
            db,
            bot,
            user_id=callback.from_user.id,
            edit=True,
            page=page,
            archive=(mode == "archive"),
        )
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


async def handle_vk_captcha(message: types.Message, db: Database, bot: Bot):
    global _vk_captcha_needed, _vk_captcha_sid, _vk_captcha_img, _vk_captcha_resume, _vk_captcha_timeout, _vk_captcha_method, _vk_captcha_params, _vk_captcha_awaiting_user
    text = message.text or ""
    code: str | None = None
    if text.startswith("/captcha"):
        parts = text.split(maxsplit=1)
        if len(parts) != 2:
            await bot.send_message(message.chat.id, "Usage: /captcha <code>")
            return
        code = parts[1].strip()
    elif message.reply_to_message and message.from_user.id == _vk_captcha_awaiting_user:
        code = text.strip()
    else:
        await bot.send_message(message.chat.id, "Usage: /captcha <code>")
        return
    _vk_captcha_awaiting_user = None
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            return
    invalid_markup = types.InlineKeyboardMarkup(
        inline_keyboard=[[types.InlineKeyboardButton(text="Отправить новый код", callback_data="captcha_refresh")]]
    )
    if not _vk_captcha_sid or not _vk_captcha_method or _vk_captcha_params is None:
        await bot.send_message(message.chat.id, "код не подошёл", reply_markup=invalid_markup)
        logging.info("vk_captcha invalid/expired")
        return
    if _vk_captcha_requested_at and (
        datetime.now(ZoneInfo(VK_WEEK_EDIT_TZ)) - _vk_captcha_requested_at
    ).total_seconds() > VK_CAPTCHA_TTL_MIN * 60:
        await bot.send_message(message.chat.id, "код не подошёл", reply_markup=invalid_markup)
        logging.info("vk_captcha invalid/expired")
        return
    params = dict(_vk_captcha_params)
    params.update({"captcha_sid": _vk_captcha_sid, "captcha_key": code})
    logging.info("vk_captcha code_received")
    try:
        await _vk_api(_vk_captcha_method, params, db, bot, skip_captcha=True)
        _vk_captcha_needed = False
        _vk_captcha_sid = None
        _vk_captcha_img = None
        _vk_captcha_method = None
        _vk_captcha_params = None
        if _vk_captcha_timeout:
            _vk_captcha_timeout.cancel()
            _vk_captcha_timeout = None
        resume = _vk_captcha_resume
        _vk_captcha_resume = None
        if resume:
            await resume()
            eid = None
            if _vk_captcha_key and ":" in _vk_captcha_key:
                try:
                    eid = int(_vk_captcha_key.split(":", 1)[1])
                except ValueError:
                    eid = None
            if eid:
                logline("VK", eid, "resumed after captcha")
        await bot.send_message(message.chat.id, "VK ✅")
        logging.info("vk_captcha ok")
    except VKAPIError as e:
        await bot.send_message(message.chat.id, "код не подошёл", reply_markup=invalid_markup)
        logging.info(
            "vk_captcha invalid/expired actor=%s token=%s code=%s msg=%s",
            e.actor,
            e.token,
            e.code,
            e.message,
        )


async def handle_askloc(callback: types.CallbackQuery, db: Database, bot: Bot):
    await callback.answer()
    await bot.send_message(callback.message.chat.id, "Пришлите сообщение с локацией и пересланным постом")


async def handle_askcity(callback: types.CallbackQuery, db: Database, bot: Bot):
    await callback.answer()
    await bot.send_message(callback.message.chat.id, "Пришлите сообщение с городом и пересланным постом")


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


async def send_festivals_list(
    message: types.Message,
    db: Database,
    bot: Bot,
    *,
    user_id: int | None = None,
    edit: bool = False,
    page: int = 1,
    archive: bool = False,
):
    PAGE_SIZE = 10
    today = datetime.now(LOCAL_TZ).date().isoformat()
    mode = "archive" if archive else "active"

    resolved_user_id = user_id
    if resolved_user_id is None:
        chat = getattr(message, "chat", None)
        if chat and getattr(chat, "type", None) == "private":
            resolved_user_id = chat.id
        elif getattr(message, "from_user", None):
            resolved_user_id = message.from_user.id

    if resolved_user_id is None:
        if not edit:
            await bot.send_message(message.chat.id, "Not authorized")
        return

    async with db.get_session() as session:
        if not await session.get(User, resolved_user_id):
            if not edit:
                await bot.send_message(message.chat.id, "Not authorized")
            return

        event_agg = (
            select(
                Event.festival.label("festival_name"),
                func.min(Event.date).label("first_date"),
                func.max(func.coalesce(Event.end_date, Event.date)).label("last_date"),
                func.count()
                .filter(func.coalesce(Event.end_date, Event.date) >= today)
                .label("future_count"),
            )
            .where(Event.festival.is_not(None))
            .group_by(Event.festival)
            .subquery()
        )

        last_date_expr = case(
            (Festival.end_date.is_(None), event_agg.c.last_date),
            (event_agg.c.last_date.is_(None), Festival.end_date),
            (
                event_agg.c.last_date >= Festival.end_date,
                event_agg.c.last_date,
            ),
            else_=Festival.end_date,
        )

        base_query = (
            select(
                Festival,
                event_agg.c.first_date,
                last_date_expr.label("last_date"),
                event_agg.c.future_count,
            )
            .outerjoin(event_agg, event_agg.c.festival_name == Festival.name)
            .order_by(Festival.name)
        )

        if archive:
            base_query = base_query.where(
                and_(
                    last_date_expr.is_not(None),
                    last_date_expr < today,
                )
            )
        else:
            base_query = base_query.where(
                or_(
                    last_date_expr.is_(None),
                    last_date_expr >= today,
                )
            )

        result = await session.execute(base_query)
        rows = result.all()

    total_count = len(rows)
    total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    visible_rows = rows[start : start + PAGE_SIZE]

    heading = f"Фестивали {'архив' if archive else 'активные'} (стр. {page}/{total_pages})"
    lines: list[str] = [heading]
    keyboard: list[list[types.InlineKeyboardButton]] = []

    for fest, first_date, last_date, future_count in visible_rows:
        parts = [f"{fest.id} {fest.name}"]
        if first_date and last_date:
            if first_date == last_date:
                parts.append(first_date)
            else:
                parts.append(f"{first_date}..{last_date}")
        elif first_date:
            parts.append(first_date)
        if future_count:
            parts.append(f"актуальных: {future_count}")
        if fest.telegraph_url:
            parts.append(fest.telegraph_url)
        if fest.website_url:
            parts.append(f"site: {fest.website_url}")
        if fest.program_url:
            parts.append(f"program: {fest.program_url}")
        if fest.vk_url:
            parts.append(f"vk: {fest.vk_url}")
        if fest.tg_url:
            parts.append(f"tg: {fest.tg_url}")
        if fest.ticket_url:
            parts.append(f"ticket: {fest.ticket_url}")
        lines.append(" ".join(parts))

        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text=f"Edit {fest.id}", callback_data=f"festedit:{fest.id}"
                ),
                types.InlineKeyboardButton(
                    text=f"Delete {fest.id}",
                    callback_data=f"festdel:{fest.id}:{page}:{mode}",
                ),
            ]
        )

    if not visible_rows:
        lines.append("Нет фестивалей")

    nav_row: list[types.InlineKeyboardButton] = []
    if total_pages > 1 and page > 1:
        nav_row.append(
            types.InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=f"festpage:{page-1}:{mode}",
            )
        )
    if total_pages > 1 and page < total_pages:
        nav_row.append(
            types.InlineKeyboardButton(
                text="Вперёд ➡️",
                callback_data=f"festpage:{page+1}:{mode}",
            )
        )
    if nav_row:
        keyboard.append(nav_row)

    toggle_mode = "archive" if not archive else "active"
    toggle_text = "Показать архив" if not archive else "Показать активные"
    keyboard.append(
        [
            types.InlineKeyboardButton(
                text=toggle_text,
                callback_data=f"festpage:1:{toggle_mode}",
            )
        ]
    )

    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)

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

    for f in ("source_chat_id", "source_message_id", "source_post_url"):
        val = getattr(src, f)
        if val is not None:
            setattr(dst, f, val)

    if not dst.topics_manual:
        dst.topics = list(src.topics or [])
        dst.topics_manual = src.topics_manual
    else:
        dst.topics_manual = bool(dst.topics_manual or src.topics_manual)


async def upsert_event_posters(
    session: AsyncSession,
    event_id: int,
    poster_items: Sequence[PosterMedia] | None,
) -> None:
    if not poster_items:
        return

    existing = (
        await session.execute(
            select(EventPoster).where(EventPoster.event_id == event_id)
        )
    ).scalars()
    existing_map = {row.poster_hash: row for row in existing}
    seen: set[str] = set()
    now = datetime.now(timezone.utc)

    for item in poster_items:
        digest = item.digest
        if not digest and item.data:
            digest = hashlib.sha256(item.data).hexdigest()
        if not digest or digest in seen:
            continue
        seen.add(digest)
        row = existing_map.get(digest)
        prompt_tokens = item.prompt_tokens
        completion_tokens = item.completion_tokens
        total_tokens = item.total_tokens
        if row:
            if item.catbox_url:
                row.catbox_url = item.catbox_url
            if item.ocr_text is not None:
                row.ocr_text = item.ocr_text
            if prompt_tokens is not None:
                row.prompt_tokens = int(prompt_tokens)
            if completion_tokens is not None:
                row.completion_tokens = int(completion_tokens)
            if total_tokens is not None:
                row.total_tokens = int(total_tokens)
            row.updated_at = now
        else:
            session.add(
                EventPoster(
                    event_id=event_id,
                    catbox_url=item.catbox_url,
                    poster_hash=digest,
                    ocr_text=item.ocr_text,
                    prompt_tokens=int(prompt_tokens or 0),
                    completion_tokens=int(completion_tokens or 0),
                    total_tokens=int(total_tokens or 0),
                    updated_at=now,
                )
            )

    stale_entries = [row for key, row in existing_map.items() if key not in seen]
    for entry in stale_entries:
        await session.delete(entry)

    await session.commit()


async def _fetch_event_posters(
    event_id: int | None, db_obj: Database | None
) -> list[EventPoster]:
    """Return saved poster rows for the given event ordered by recency."""

    if not event_id or db_obj is None:
        return []

    async with db_obj.get_session() as session:
        result = await session.execute(
            select(EventPoster)
            .where(EventPoster.event_id == event_id)
            .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
        )
        return list(result.scalars().all())


async def get_event_poster_texts(
    event_id: int | None,
    db_obj: Database | None,
    *,
    posters: Sequence[EventPoster] | None = None,
) -> list[str]:
    """Load stored OCR blocks for an event and return non-empty texts."""

    if posters is None:
        posters = await _fetch_event_posters(event_id, db_obj)

    texts: list[str] = []
    for poster in posters:
        raw = (poster.ocr_text or "").strip()
        if raw:
            texts.append(raw)
    return texts


def _summarize_event_posters(posters: Sequence[EventPoster]) -> str | None:
    """Build a short summary describing stored OCR usage."""

    if not posters:
        return None

    prompt_tokens = sum(p.prompt_tokens or 0 for p in posters)
    completion_tokens = sum(p.completion_tokens or 0 for p in posters)
    total_tokens = sum(p.total_tokens or 0 for p in posters)

    if prompt_tokens == completion_tokens == total_tokens == 0:
        return f"Posters processed: {len(posters)}."

    return (
        f"Posters processed: {len(posters)}. "
        f"Tokens — prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}."
    )


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
    new.added_at = datetime.now(timezone.utc)
    session.add(new)
    await session.commit()
    logging.info("upsert_event: inserted new event id=%s", new.id)
    return new, True


async def enqueue_job(
    db: Database,
    event_id: int,
    task: JobTask,
    payload: dict | None = None,
    *,
    coalesce_key: str | None = None,
    depends_on: list[str] | None = None,
) -> str:
    async with db.get_session() as session:
        now = datetime.now(timezone.utc)
        ev = None
        if coalesce_key is None or depends_on is None:
            ev = await session.get(Event, event_id)
        if coalesce_key is None and ev:
            if task == JobTask.month_pages:
                month = ev.date.split("..", 1)[0][:7]
                coalesce_key = f"month_pages:{month}"
            elif task == JobTask.week_pages:
                d = parse_iso_date(ev.date)
                if d:
                    week = d.isocalendar().week
                    coalesce_key = f"week_pages:{d.year}-{week:02d}"
            elif task == JobTask.weekend_pages:
                d = parse_iso_date(ev.date)
                w = weekend_start_for_date(d) if d else None
                if w:
                    coalesce_key = f"weekend_pages:{w.isoformat()}"
            elif task == JobTask.festival_pages and ev and ev.festival:
                fest = (
                    await session.execute(
                        select(Festival.id).where(Festival.name == ev.festival)
                    )
                ).scalar_one_or_none()
                if fest:
                    coalesce_key = f"festival_pages:{fest}"
        if depends_on is None and ev and ev.festival and task in {
            JobTask.month_pages,
            JobTask.week_pages,
            JobTask.weekend_pages,
        }:
            fest = (
                await session.execute(
                    select(Festival.id).where(Festival.name == ev.festival)
                )
            ).scalar_one_or_none()
            if fest:
                depends_on = [f"festival_pages:{fest}"]
        job_key = coalesce_key or f"{task.value}:{event_id}"
        if coalesce_key:
            stmt = (
                select(JobOutbox)
                .where(JobOutbox.coalesce_key == coalesce_key)
                .order_by(JobOutbox.id.desc())
                .limit(1)
            )
        else:
            stmt = (
                select(JobOutbox)
                .where(JobOutbox.event_id == event_id, JobOutbox.task == task)
                .order_by(JobOutbox.id.desc())
                .limit(1)
            )
        res = await session.execute(stmt)
        job = _normalize_job(res.scalar_one_or_none())
        dep_str = ",".join(depends_on) if depends_on else None
        if job:
            if job.status == JobStatus.done and task == JobTask.vk_sync:
                logline("ENQ", event_id, "skipped", job_key=job_key)
                return "skipped"
            if job.status == JobStatus.running:
                age = (now - job.updated_at).total_seconds()
                if age > 600:
                    job.status = JobStatus.error
                    job.last_error = "stale"
                    job.next_run_at = now
                    job.updated_at = now
                    session.add(job)
                    await session.commit()
                    logging.info(
                        "OUTBOX_STALE_FIXED key=%s prev_owner=%s", job_key, job.event_id
                    )
                    job = None
        if job:
            if job.status == JobStatus.pending:
                if payload is not None:
                    job.payload = payload
                if depends_on:
                    cur = set(filter(None, (job.depends_on or "").split(",")))
                    cur.update(depends_on)
                    job.depends_on = ",".join(sorted(cur))
                now = datetime.now(timezone.utc)
                job.next_run_at = now
                job.updated_at = now
                job.attempts = 0
                job.last_error = None
                session.add(job)
                await session.commit()
                logline(
                    "ENQ",
                    event_id,
                    "merged",
                    job_key=job_key,
                    status="pending",
                    owner_eid=job.event_id if job.event_id != event_id else None,
                    coalesce_key=job.coalesce_key,
                )
                return "merged-rearmed"
            if job.status == JobStatus.running:
                updated = False
                if payload is not None:
                    job.payload = payload
                    updated = True
                if depends_on:
                    cur = set(filter(None, (job.depends_on or "").split(",")))
                    before = cur.copy()
                    cur.update(depends_on)
                    if cur != before:
                        job.depends_on = ",".join(sorted(cur))
                        updated = True
                if updated:
                    session.add(job)
                    await session.commit()
                follow_key = None
                if (
                    task == JobTask.month_pages
                    and job.event_id != event_id
                    and job.coalesce_key
                ):
                    follow_key = f"{job.coalesce_key}:v2:{event_id}"
                    exists = (
                        await session.execute(
                            select(JobOutbox.id).where(
                                JobOutbox.coalesce_key == follow_key
                            )
                        )
                    ).scalar_one_or_none()
                    if not exists:
                        session.add(
                            JobOutbox(
                                event_id=event_id,
                                task=task,
                                payload=payload,
                                status=JobStatus.pending,
                                updated_at=now,
                                next_run_at=now,
                                coalesce_key=follow_key,
                                depends_on=job.coalesce_key,
                            )
                        )
                        await session.commit()
                        logging.info(
                            "ENQ nav followup key=%s reason=owner_running",
                            follow_key,
                        )
                if task in NAV_TASKS:
                    logging.info(
                        "ENQ nav merged key=%s into_owner_eid=%s owner_started_at=%s",
                        job_key,
                        job.event_id,
                        job.updated_at.isoformat(),
                    )
                logline(
                    "ENQ",
                    event_id,
                    "merged",
                    job_key=job_key,
                    status="running",
                    owner_eid=job.event_id if job.event_id != event_id else None,
                    coalesce_key=job.coalesce_key,
                )
                return "merged"

            # requeue for existing (possibly coalesced) task
            job.status = JobStatus.pending
            job.payload = payload
            job.attempts = 0
            job.last_error = None
            job.updated_at = now
            job.next_run_at = now
            if depends_on:
                cur = set(filter(None, (job.depends_on or "").split(",")))
                cur.update(depends_on)
                job.depends_on = ",".join(sorted(cur))
            session.add(job)
            await session.commit()
            logline(
                "ENQ",
                event_id,
                "requeued",
                job_key=job_key,
                coalesce_key=job.coalesce_key,
            )
            return "requeued"
        session.add(
            JobOutbox(
                event_id=event_id,
                task=task,
                payload=payload,
                status=JobStatus.pending,
                updated_at=now,
                next_run_at=now,
                coalesce_key=coalesce_key,
                depends_on=dep_str,
            )
        )
        await session.commit()
        if task in NAV_TASKS:
            logging.info("ENQ nav task key=%s eid=%s", job_key, event_id)
        logline("ENQ", event_id, "new", job_key=job_key)
        return "new"


async def schedule_event_update_tasks(
    db: Database, ev: Event, *, drain_nav: bool = True, skip_vk_sync: bool = False
) -> dict[JobTask, str]:
    eid = ev.id
    results: dict[JobTask, str] = {}
    ics_dep: str | None = None
    if ev.time and "ics_publish" in JOB_HANDLERS:
        ics_dep = await enqueue_job(db, eid, JobTask.ics_publish, depends_on=None)
        results[JobTask.ics_publish] = ics_dep
    telegraph_dep = [ics_dep] if ics_dep else None
    results[JobTask.telegraph_build] = await enqueue_job(
        db, eid, JobTask.telegraph_build, depends_on=telegraph_dep
    )
    if "tg_ics_post" in JOB_HANDLERS:
        tg_ics_deps = [results[JobTask.telegraph_build]]
        if ics_dep:
            tg_ics_deps.append(ics_dep)
        results[JobTask.tg_ics_post] = await enqueue_job(
            db, eid, JobTask.tg_ics_post, depends_on=tg_ics_deps
        )
    page_deps = [results[JobTask.telegraph_build]]
    if ics_dep:
        page_deps.append(ics_dep)
    results[JobTask.month_pages] = await enqueue_job(
        db, eid, JobTask.month_pages, depends_on=page_deps
    )
    d = parse_iso_date(ev.date)
    if d:
        results[JobTask.week_pages] = await enqueue_job(
            db, eid, JobTask.week_pages, depends_on=page_deps
        )
        w_start = weekend_start_for_date(d)
        if w_start:
            results[JobTask.weekend_pages] = await enqueue_job(
                db, eid, JobTask.weekend_pages, depends_on=page_deps
            )
    if ev.festival:
        results[JobTask.festival_pages] = await enqueue_job(
            db, eid, JobTask.festival_pages
        )
    if not skip_vk_sync:
        if not (is_vk_wall_url(ev.source_post_url) or ev.source_vk_post_url):
            results[JobTask.vk_sync] = await enqueue_job(db, eid, JobTask.vk_sync)
    logging.info("scheduled event tasks for %s", eid)
    if drain_nav:
        await _drain_nav_tasks(db, eid)
    return results


NAV_TASKS = {
    JobTask.month_pages,
    JobTask.week_pages,
    JobTask.weekend_pages,
    JobTask.festival_pages,
}


async def _drain_nav_tasks(db: Database, event_id: int, timeout: float = 90.0) -> None:
    deadline = _time.monotonic() + timeout

    keys: set[str] = set()
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if ev:
            month = ev.date.split("..", 1)[0][:7]
            keys.add(f"month_pages:{month}")
            d = parse_iso_date(ev.date)
            if d:
                week = d.isocalendar().week
                keys.add(f"week_pages:{d.year}-{week:02d}")
                w = weekend_start_for_date(d)
                if w:
                    keys.add(f"weekend_pages:{w.isoformat()}")
            if ev.festival:
                fest = (
                    await session.execute(
                        select(Festival.id).where(Festival.name == ev.festival)
                    )
                ).scalar_one_or_none()
                if fest:
                    keys.add(f"festival_pages:{fest}")

    logging.info(
        "NAV drain start eid=%s keys=%s",
        event_id,
        sorted(keys),
    )

    owners_limit = 3
    merged: dict[str, int] = {}

    while True:
        await _run_due_jobs_once(
            db,
            None,
            None,
            event_id,
            None,
            None,
            NAV_TASKS,
            True,
        )

        async with db.get_session() as session:
            rows = (
                await session.execute(
                    select(JobOutbox.event_id, JobOutbox.coalesce_key)
                    .where(
                        JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                        JobOutbox.task.in_(NAV_TASKS),
                        JobOutbox.coalesce_key.in_(keys),
                    )
                )
            ).all()

        owners: dict[int, set[str]] = {}
        self_keys = {key for owner, key in rows if owner == event_id}
        for owner, key in rows:
            if owner == event_id:
                continue
            owners.setdefault(owner, set()).add(key)
            if key not in self_keys and key not in merged:
                merged[key] = owner

        ran_any = False
        for idx, (owner, oks) in enumerate(owners.items()):
            if idx >= owners_limit:
                break
            count = await _run_due_jobs_once(
                db,
                None,
                notify=None,
                only_event=owner,
                ics_progress=None,
                fest_progress=None,
                allowed_tasks=NAV_TASKS,
                force_notify=True,
            )
            if count > 0:
                logging.info(
                    "nav_drain owner_event=%s key=%s ran=%d",
                    owner,
                    ",".join(sorted(oks)),
                    count,
                )
            ran_any = ran_any or (count > 0)

        async with db.get_session() as session:
            rows = (
                await session.execute(
                    select(JobOutbox.coalesce_key)
                    .where(
                        JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                        JobOutbox.task.in_(NAV_TASKS),
                        JobOutbox.coalesce_key.in_(keys),
                    )
                )
            ).all()
        current_keys = {key for (key,) in rows}
        for key, owner in list(merged.items()):
            if key not in current_keys:
                task_name = key.split(":", 1)[0]
                try:
                    task = JobTask(task_name)
                except Exception:
                    continue
                new_key = f"{key}:v2:{event_id}"
                logging.info(
                    "ENQ nav followup key=%s reason=owner_running",
                    new_key,
                )
                await enqueue_job(db, event_id, task, coalesce_key=new_key)
                keys.add(new_key)
                del merged[key]

        async with db.get_session() as session:
            remaining = (
                await session.execute(
                    select(func.count())
                    .where(
                        JobOutbox.task.in_(NAV_TASKS),
                        JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                        or_(
                            JobOutbox.event_id == event_id,
                            JobOutbox.coalesce_key.in_(keys),
                        ),
                    )
                )
            ).scalar_one()
        if not remaining:
            logging.info("NAV drain done")
            break
        if _time.monotonic() > deadline:
            logging.warning(
                "NAV drain timeout remaining=%s",
                sorted(current_keys),
            )
            break
        if not ran_any:
            ttl = int(max(0, deadline - _time.monotonic()))
            logging.info(
                "NAV drain wait remaining=%s ttl=%d",
                sorted(current_keys),
                ttl,
            )
            await asyncio.sleep(1.0)


def missing_fields(event: dict | Event) -> list[str]:
    """Return a list of required fields missing from ``event``.

    ``event`` can be either an ``Event`` instance or a mapping with string keys.
    The required fields are: ``title``, ``date``, ``location_name`` and ``city``.
    The ``time`` field is optional.
    """

    if isinstance(event, Event):
        data = {
            "title": event.title,
            "date": event.date,
            "location_name": event.location_name,
            "city": event.city,
        }
    else:
        data = {
            key: (event.get(key) or "").strip() for key in (
                "title",
                "date",
                "location_name",
                "city",
            )
        }

    return [field for field, value in data.items() if not value]


class AddEventsResult(list):
    """Container for parsed events along with poster OCR usage stats."""

    def __init__(
        self,
        entries: list[tuple[Event | Festival | None, bool, list[str], str]],
        tokens_spent: int,
        tokens_remaining: int | None,
        *,
        limit_notice: str | None = None,
    ) -> None:
        super().__init__(entries)
        self.ocr_tokens_spent = tokens_spent
        self.ocr_tokens_remaining = tokens_remaining
        self.ocr_limit_notice = limit_notice


async def add_events_from_text(
    db: Database,
    text: str,
    source_link: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    poster_media: Sequence[PosterMedia] | None = None,
    force_festival: bool = False,
    *,
    raise_exc: bool = False,
    source_chat_id: int | None = None,
    source_message_id: int | None = None,
    creator_id: int | None = None,
    display_source: bool = True,
    source_channel: str | None = None,
    channel_title: str | None = None,


    bot: Bot | None = None,

) -> AddEventsResult:
    logging.info(
        "add_events_from_text start: len=%d source=%s", len(text), source_link
    )
    poster_items: list[PosterMedia] = []
    ocr_tokens_spent = 0
    ocr_tokens_remaining: int | None = None
    ocr_limit_notice: str | None = None
    normalized_media: list[tuple[bytes, str]] = []
    if media:
        normalized_media = [media] if isinstance(media, tuple) else list(media)
    if poster_media:
        poster_items = list(poster_media)
    elif normalized_media:
        poster_items, _ = await process_media(
            normalized_media, need_catbox=True, need_ocr=False
        )
    source_marker = (
        source_link
        or (f"channel:{source_channel}" if source_channel else None)
        or (
            f"chat:{source_chat_id}/{source_message_id}"
            if source_chat_id and source_message_id
            else None
        )
        or (f"chat:{source_chat_id}" if source_chat_id else None)
        or (f"message:{source_message_id}" if source_message_id else None)
        or (f"creator:{creator_id}" if creator_id else None)
        or (f"channel_title:{channel_title}" if channel_title else None)
        or "add_events_from_text"
    )
    ocr_log_context = {"event_id": None, "source": source_marker}
    hash_to_indices: dict[str, list[int]] | None = None
    if normalized_media:
        hash_to_indices = {}
        for idx, (payload, _name) in enumerate(normalized_media):
            digest = hashlib.sha256(payload).hexdigest()
            hash_to_indices.setdefault(digest, []).append(idx)
    ocr_results: list[poster_ocr.PosterOcrCache] = []
    try:
        if normalized_media:
            (
                ocr_results,
                ocr_tokens_spent,
                ocr_tokens_remaining,
            ) = await poster_ocr.recognize_posters(
                db, normalized_media, log_context=ocr_log_context
            )
        elif poster_items:
            _, _, ocr_tokens_remaining = await poster_ocr.recognize_posters(
                db, [], log_context=ocr_log_context
            )
            ocr_tokens_spent = sum(item.total_tokens or 0 for item in poster_items)
        else:
            _, _, ocr_tokens_remaining = await poster_ocr.recognize_posters(
                db, [], log_context=ocr_log_context
            )
    except poster_ocr.PosterOcrLimitExceededError as exc:
        logging.warning("poster OCR skipped: %s", exc, extra=ocr_log_context)
        ocr_results = list(exc.results or [])
        ocr_tokens_spent = exc.spent_tokens
        ocr_tokens_remaining = exc.remaining
        ocr_limit_notice = (
            "OCR недоступен: дневной лимит токенов исчерпан, распознавание пропущено."
        )

    if ocr_results:
        apply_ocr_results_to_media(
            poster_items,
            ocr_results,
            hash_to_indices=hash_to_indices if hash_to_indices else None,
        )

    catbox_urls = [item.catbox_url for item in poster_items if item.catbox_url]
    poster_texts = collect_poster_texts(poster_items)
    poster_summary = build_poster_summary(poster_items)

    llm_call_started = _time.monotonic()
    try:
        # Free any lingering objects before heavy LLM call to reduce peak memory
        gc.collect()
        if DEBUG:
            mem_info("LLM before")
        logging.info("LLM parse start (%d chars)", len(text))
        llm_text = text
        if channel_title:
            llm_text = f"{channel_title}\n{llm_text}"
        if force_festival:
            llm_text = (
                f"{llm_text}\n"
                "Оператор подтверждает, что пост описывает фестиваль. "
                "Сопоставь с существующими фестивалями (JSON ниже) или создай новый."
            )
        today = datetime.now(LOCAL_TZ).date()
        cutoff_date = (today - timedelta(days=31)).isoformat()
        festival_names_set: set[str] = set()
        alias_map: dict[str, set[str]] = {}
        async with db.get_session() as session:
            stmt = select(Festival).where(
                or_(
                    Festival.end_date.is_(None),
                    Festival.end_date >= cutoff_date,
                    Festival.start_date.is_(None),
                )
            )
            res_f = await session.execute(stmt)
            for fest in res_f.scalars():
                name = (fest.name or "").strip()
                if name:
                    festival_names_set.add(name)
                base_norm = normalize_alias(name)
                aliases = getattr(fest, "aliases", None) or []
                if not aliases or not name:
                    continue
                for alias in aliases:
                    norm = normalize_alias(alias)
                    if not norm or norm == base_norm:
                        continue
                    alias_map.setdefault(name, set()).add(norm)
        fest_names = sorted(festival_names_set)
        fest_alias_pairs: list[tuple[str, int]] = []
        for idx, fest_name in enumerate(fest_names):
            for alias_norm in sorted(alias_map.get(fest_name, ())):
                fest_alias_pairs.append((alias_norm, idx))
        parse_kwargs: dict[str, Any] = {}
        if poster_texts:
            parse_kwargs["poster_texts"] = poster_texts
        if poster_summary:
            parse_kwargs["poster_summary"] = poster_summary
        try:
            if source_channel:
                parsed = await parse_event_via_4o(
                    llm_text,
                    source_channel,
                    festival_names=fest_names,
                    festival_alias_pairs=fest_alias_pairs,
                    **parse_kwargs,
                )
            else:
                parsed = await parse_event_via_4o(
                    llm_text,
                    festival_names=fest_names,
                    festival_alias_pairs=fest_alias_pairs,
                    **parse_kwargs,
                )
        except TypeError:
            if source_channel:
                parsed = await parse_event_via_4o(
                    llm_text, source_channel, **parse_kwargs
                )
            else:
                parsed = await parse_event_via_4o(llm_text, **parse_kwargs)

        if DEBUG:
            mem_info("LLM after")
        festival_info = getattr(parse_event_via_4o, "_festival", None)
        if isinstance(festival_info, str):
            festival_info = {"name": festival_info}
        logging.info("LLM returned %d events", len(parsed))
    except Exception as e:
        elapsed_total = _time.monotonic() - llm_call_started
        meta = getattr(e, "_four_o_call_meta", {}) or {}
        meta_elapsed = meta.get("elapsed")
        meta_wait = meta.get("semaphore_wait")

        def _fmt_duration(value: float | None) -> str:
            return f"{value:.2f}s" if isinstance(value, (int, float)) else str(value)

        logging.exception(
            "LLM error (%s) source=%s len=%d total_elapsed=%s call_elapsed=%s "
            "semaphore_acquired=%s semaphore_wait=%s",
            type(e).__name__,
            source_marker,
            len(text),
            _fmt_duration(elapsed_total),
            _fmt_duration(meta_elapsed),
            meta.get("semaphore_acquired"),
            _fmt_duration(meta_wait),
        )
        if raise_exc:
            raise
        return []

    results: list[tuple[Event | Festival | None, bool, list[str], str]] = []
    first = True
    links_iter = iter(extract_links_from_html(html_text) if html_text else [])
    source_text_clean = html_text or text
    program_url: str | None = None
    prog_links: list[str] = []
    if html_text:
        prog_links.extend(re.findall(r"href=['\"]([^'\"]+)['\"]", html_text))
    if text:
        prog_links.extend(re.findall(r"https?://\S+", text))
    for url in prog_links:
        if "telegra.ph" in url:
            program_url = url
            break
    if not program_url:
        for url in prog_links:
            u = url.lower()
            if any(x in u for x in ["program", "schedule", "расписан", "програм"]):
                program_url = url
                break

    festival_obj: Festival | None = None
    fest_created = False
    fest_updated = False
    if festival_info:
        fest_name = (
            festival_info.get("name")
            or festival_info.get("festival")
            or festival_info.get("full_name")
        )
        if force_festival and not (fest_name and fest_name.strip()):
            raise FestivalRequiredError("festival name missing")
        start = canonicalize_date(festival_info.get("start_date") or festival_info.get("date"))
        end = canonicalize_date(festival_info.get("end_date"))
        loc_name = festival_info.get("location_name")
        loc_addr = festival_info.get("location_address")
        city = festival_info.get("city")
        loc_addr = strip_city_from_address(loc_addr, city)
        photo_u = catbox_urls[0] if catbox_urls else None
        fest_obj, created, updated = await ensure_festival(
            db,
            fest_name,
            full_name=festival_info.get("full_name"),
            photo_url=photo_u,
            photo_urls=catbox_urls,
            program_url=program_url,
            start_date=start,
            end_date=end,
            location_name=loc_name,
            location_address=loc_addr,
            city=city,
            source_text=source_text_clean,
            source_post_url=source_link,
            source_chat_id=source_chat_id,
            source_message_id=source_message_id,
        )
        festival_obj = fest_obj
        fest_created = created
        fest_updated = updated
        async def _safe_sync_fest(name: str) -> None:
            try:
                await sync_festival_page(db, name)
            except Exception:
                logging.exception("festival page sync failed for %s", name)
            try:
                await sync_festivals_index_page(db)
            except Exception:
                logging.exception("festival index sync failed")
            try:
                await sync_festival_vk_post(db, name, bot, strict=True)
            except Exception:
                logging.exception("festival VK sync failed for %s", name)
                if bot:
                    try:
                        await notify_superadmin(
                            db, bot, f"festival VK sync failed for {name}"
                        )
                    except Exception:
                        logging.exception("notify_superadmin failed for %s", name)

        if created or fest_updated:
            await _safe_sync_fest(fest_obj.name)
            async with db.get_session() as session:
                res = await session.execute(
                    select(Festival).where(Festival.name == fest_obj.name)
                )
                festival_obj = res.scalar_one_or_none()
        if festival_obj:
            await try_set_fest_cover_from_program(db, festival_obj)
    elif force_festival:
        raise FestivalRequiredError("festival name missing")
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
        event_type_raw = data.get("event_type")
        event_type_name = (
            event_type_raw.casefold() if isinstance(event_type_raw, str) else ""
        )
        title = (data.get("title") or "").strip()
        time_str = (data.get("time") or "").strip()
        location_name = (data.get("location_name") or "").strip()
        if not location_name and addr:
            location_name, addr = addr, None
        loc_text = f"{location_name} {addr or ''}".lower()
        if city and city.lower() not in loc_text:
            city = None
        if not city:
            city = "Калининград"
        addr = strip_city_from_address(addr, city)
        allow_missing_date = bool(end_date and event_type_name == "выставка")
        if allow_missing_date and not date_str:
            date_str = datetime.now(LOCAL_TZ).date().isoformat()
        missing = missing_fields(
            {
                "title": title,
                "date": date_str,
                "location_name": location_name,
                "city": city or "",
            }
        )
        required_missing = [m for m in missing if m != "city"]
        if required_missing:
            logging.warning(
                "Skipping event due to missing fields: %s", ", ".join(missing)
            )
            results.append((None, False, missing, "missing"))
            continue

        base_event = Event(
            title=title,
            description=data.get("short_description", ""),
            festival=data.get("festival") or None,
            date=date_str,
            time=time_str,
            location_name=location_name,
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
            photo_count=len(catbox_urls),
            photo_urls=catbox_urls,
        )

        base_event.event_type = normalize_event_type(
            base_event.title, base_event.description, base_event.event_type
        )

        if base_event.festival:
            photo_u = catbox_urls[0] if catbox_urls else None
            await ensure_festival(
                db,
                base_event.festival,
                full_name=data.get("festival_full"),
                photo_url=photo_u,
                photo_urls=catbox_urls,
            )

        if base_event.event_type == "выставка" and not base_event.end_date:
            start_dt = parse_iso_date(base_event.date) or datetime.now(LOCAL_TZ).date()
            base_event.date = start_dt.isoformat()
            base_event.end_date = date(start_dt.year, 12, 31).isoformat()

        topics_meta_map: dict[int, tuple[list[str], int, str | None, bool]] = {}
        topics_meta_map[id(base_event)] = await assign_event_topics(base_event)

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
                            }
                        )
                    )
                    copy_e.date = day.isoformat()
                    copy_e.end_date = None
                    copy_e.topics = list(base_event.topics or [])
                    copy_e.topics_manual = base_event.topics_manual
                    topics_meta_map[id(copy_e)] = topics_meta_map[id(base_event)]
                    events_to_add.append(copy_e)
        for event in events_to_add:
            rejected_links: list[str] = []
            if event.ticket_link and is_tg_folder_link(event.ticket_link):
                rejected_links.append(event.ticket_link)
                event.ticket_link = None
            if not is_valid_url(event.ticket_link):
                while True:
                    try:
                        extracted = next(links_iter)
                    except StopIteration:
                        extracted = None
                    if extracted is None:
                        break
                    if is_tg_folder_link(extracted):
                        rejected_links.append(extracted)
                        continue
                    event.ticket_link = extracted
                    break

            # skip events that have already finished - disabled for consistency in tests

            _meta_topics, meta_text_len, meta_error, meta_manual = topics_meta_map.get(
                id(event),
                (
                    list(event.topics or []),
                    _event_topic_text_length(event),
                    None,
                    bool(event.topics_manual),
                ),
            )

            async with db.get_session() as session:
                saved, added = await upsert_event(session, event)
                await upsert_event_posters(session, saved.id, poster_items)
                if rejected_links:
                    for url in rejected_links:
                        pattern = (
                            "telegram_folder" if is_tg_folder_link(url) else "unknown"
                        )
                        logging.info(
                            "ticket_link_rejected pattern=%s url=%s eid=%s",
                            pattern,
                            url,
                            saved.id,
                        )
            if meta_manual:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s manual=True",
                    saved.id,
                    meta_text_len,
                    list(saved.topics or []),
                )
            elif meta_error:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s error=%s",
                    saved.id,
                    meta_text_len,
                    list(saved.topics or []),
                    meta_error,
                )
            else:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s",
                    saved.id,
                    meta_text_len,
                    list(saved.topics or []),
                )
            logline("FLOW", saved.id, "start add_event", user=creator_id)
            logline(
                "FLOW",
                saved.id,
                "parsed",
                title=f'"{saved.title}"',
                date=saved.date,
                time=saved.time,
            )
            await schedule_event_update_tasks(db, saved)
            d = parse_iso_date(saved.date)
            week = d.isocalendar().week if d else None
            w_start = weekend_start_for_date(d) if d else None
            logline(
                "FLOW",
                saved.id,
                "scheduled",
                month=saved.date[:7],
                week=f"{d.year}-{week:02d}" if week else None,
                weekend=w_start.isoformat() if w_start else None,
            )
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
    if festival_obj and (fest_created or fest_updated):
        lines = [f"festival: {festival_obj.name}"]
        if festival_obj.telegraph_url:
            lines.append(f"telegraph: {festival_obj.telegraph_url}")
        if festival_obj.vk_post_url:
            lines.append(f"vk_post: {festival_obj.vk_post_url}")
        if festival_obj.start_date:
            lines.append(f"start: {festival_obj.start_date}")
        if festival_obj.end_date:
            lines.append(f"end: {festival_obj.end_date}")
        if festival_obj.location_name:
            lines.append(f"location_name: {festival_obj.location_name}")
        if festival_obj.city:
            lines.append(f"city: {festival_obj.city}")
        results.insert(0, (festival_obj, fest_created, lines, "festival"))
        logging.info(
            "festival %s %s", festival_obj.name, "created" if fest_created else "updated"
        )
    logging.info("add_events_from_text finished with %d results", len(results))
    del parsed
    gc.collect()
    return AddEventsResult(
        results,
        ocr_tokens_spent,
        ocr_tokens_remaining,
        limit_notice=ocr_limit_notice,
    )


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
    message: types.Message,
    db: Database,
    bot: Bot,
    *,
    session_mode: AddEventMode | None = None,
    force_festival: bool = False,
    media: list[tuple[bytes, str]] | None = None,
    poster_media: Sequence[PosterMedia] | None = None,
    catbox_msg: str | None = None,
):
    text_raw = message.text or message.caption or ""
    logging.info(
        "handle_add_event start: user=%s len=%d", message.from_user.id, len(text_raw)
    )
    if session_mode:
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
    if media is None:
        images = await extract_images(message, bot)
        media = images if images else None
    normalized_media = []
    if media:
        normalized_media = [media] if isinstance(media, tuple) else list(media)
    poster_items: list[PosterMedia] = []
    catbox_msg_local = catbox_msg or ""
    if poster_media is not None:
        poster_items = list(poster_media)
    elif normalized_media:
        poster_items, catbox_msg_local = await process_media(
            normalized_media, need_catbox=True, need_ocr=False
        )
    global LAST_CATBOX_MSG
    LAST_CATBOX_MSG = catbox_msg_local
    html_text, _mode = ensure_html_text(message)
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
    effective_force_festival = force_festival or session_mode == "festival"
    try:
        results = await add_events_from_text(
            db,
            text_content,
            source_link,
            html_text,
            normalized_media,
            poster_media=poster_items,
            force_festival=effective_force_festival,
            raise_exc=True,
            creator_id=creator_id,
            display_source=False if source_link else True,
            source_channel=None,

            bot=None,
        )
    except FestivalRequiredError:
        await bot.send_message(
            message.chat.id,
            "Не удалось распознать фестиваль. Уточните название фестиваля и попробуйте снова.",
        )
        if session_mode == "festival":
            add_event_sessions[message.from_user.id] = "festival"
        return
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
    ocr_line = None
    if normalized_media and results.ocr_tokens_remaining is not None:
        base_line = (
            f"OCR: потрачено {results.ocr_tokens_spent}, осталось "
            f"{results.ocr_tokens_remaining}"
        )
        if results.ocr_limit_notice:
            ocr_line = f"{results.ocr_limit_notice}\n{base_line}"
        else:
            ocr_line = base_line
    grouped: dict[int, tuple[Event, bool]] = {}
    fest_msgs: list[tuple[Festival, bool, list[str]]] = []
    for saved, added, lines, status in results:
        if isinstance(saved, Festival):
            fest_msgs.append((saved, added, lines))
            continue
        info = grouped.get(saved.id)
        if info:
            grouped[saved.id] = (saved, info[1] or added)
        else:
            grouped[saved.id] = (saved, added)

    for fest, added, lines in fest_msgs:
        async with db.get_session() as session:
            res = await session.execute(
                select(func.count()).select_from(Event).where(Event.festival == fest.name)
            )
            count = res.scalar_one()
        markup = None
        if count == 0:
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[[types.InlineKeyboardButton(
                    text="Создать события по дням",
                    callback_data=f"festdays:{fest.id}")]]
            )
        status = "added" if added else "updated"
        text_out = f"Festival {status}\n" + "\n".join(lines)
        if ocr_line:
            text_out = f"{text_out}\n{ocr_line}"
            ocr_line = None
        await bot.send_message(message.chat.id, text_out, reply_markup=markup)

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
        buttons_second.append(
            types.InlineKeyboardButton(
                text="Редактировать",
                callback_data=f"edit:{saved.id}",
            )
        )
        inline_keyboard = append_tourist_block(
            [buttons_first, buttons_second], saved, "tg"
        )
        markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
        extra_lines = [ocr_line] if ocr_line else None
        text_out = build_event_card_message(
            f"Event {status}", saved, lines, extra_lines=extra_lines
        )
        if ocr_line:
            ocr_line = None
        await bot.send_message(message.chat.id, text_out, reply_markup=markup)
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
    source_clean = html_text or parts[1]

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
    results = await schedule_event_update_tasks(db, event)
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
    buttons_second.append(
        types.InlineKeyboardButton(
            text="Редактировать",
            callback_data=f"edit:{event.id}",
        )
    )
    inline_keyboard = append_tourist_block(
        [buttons_first, buttons_second], event, "tg"
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
    text_out = build_event_card_message(
        f"Event {status}", event, lines
    )
    await bot.send_message(
        message.chat.id,
        text_out,
        reply_markup=markup,
    )
    await notify_event_added(db, bot, user, event, added)
    await publish_event_progress(event, db, bot, message.chat.id, results)
    logging.info("handle_add_event_raw finished for user %s", message.from_user.id)


async def enqueue_add_event(
    message: types.Message,
    db: Database,
    bot: Bot,
    *,
    session_mode: AddEventMode | None = None,
):
    """Queue an event addition for background processing."""
    if session_mode is None:
        session_mode = add_event_sessions.get(message.from_user.id)
    if session_mode:
        add_event_sessions.pop(message.from_user.id, None)
    try:
        add_event_queue.put_nowait(("regular", message, session_mode, 0))
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
    await bot.send_message(message.chat.id, "⏳ Разбираю текст…")


async def enqueue_add_event_raw(message: types.Message, db: Database, bot: Bot):
    """Queue a raw event addition for background processing."""
    try:
        add_event_queue.put_nowait(("raw", message, None, 0))
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
    await bot.send_message(message.chat.id, "⏳ Разбираю текст…")


async def add_event_queue_worker(db: Database, bot: Bot, limit: int = 2):
    """Background worker to process queued events with timeout & retries."""

    global _ADD_EVENT_LAST_DEQUEUE_TS
    while True:
        kind, msg, session_mode, attempts = await add_event_queue.get()
        _ADD_EVENT_LAST_DEQUEUE_TS = _time.monotonic()
        logging.info(
            "add_event_queue dequeued user=%s attempts=%d qsize=%d",
            getattr(msg.from_user, "id", None),
            attempts,
            add_event_queue.qsize(),
        )
        start = _time.perf_counter()
        timed_out = False
        try:
            async def _run():
                if kind == "regular":
                    await handle_add_event(
                        msg,
                        db,
                        bot,
                        session_mode=session_mode,
                        force_festival=session_mode == "festival",
                    )
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
                    "Фоновая публикация ещё идёт, статус обновится в этом сообщении",
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
                if session_mode == "festival":
                    add_event_sessions[msg.from_user.id] = session_mode
            except Exception:  # pragma: no cover - notify fail
                logging.exception("add_event_queue_worker notify failed")
        finally:
            dur = (_time.perf_counter() - start) * 1000.0
            logging.info("add_event_queue item done in %.0f ms", dur)
            add_event_queue.task_done()

        if timed_out:
            pass


BACKOFF_SCHEDULE = [30, 120, 600, 3600]


TASK_LABELS = {
    "telegraph_build": "Telegraph (событие)",
    "vk_sync": "VK (событие)",
    "ics_publish": "Календарь (ICS)",
    "tg_ics_post": "ICS (Telegram)",
    "month_pages": "Страница месяца",
    "week_pages": "VK (неделя)",
    "weekend_pages": "VK (выходные)",
    "festival_pages": "VK (фестиваль)",
    "fest_nav:update_all": "Навигация",
}

JOB_TTL: dict[JobTask, int] = {
    JobTask.telegraph_build: 600,
    JobTask.ics_publish: 600,
    JobTask.tg_ics_post: 600,
    JobTask.month_pages: 600,
    JobTask.week_pages: 600,
    JobTask.weekend_pages: 600,
}

JOB_MAX_RUNTIME: dict[JobTask, int] = {
    JobTask.telegraph_build: 180,
    JobTask.ics_publish: 60,
    JobTask.tg_ics_post: 60,
    JobTask.month_pages: 180,
    JobTask.week_pages: 180,
    JobTask.weekend_pages: 180,
}

DEFAULT_JOB_TTL = 600
DEFAULT_JOB_MAX_RUNTIME = 900

# runtime storage for progress callbacks keyed by event id
_EVENT_PROGRESS: dict[int, SimpleNamespace] = {}
# mapping from coalesce key to events waiting for progress updates
_EVENT_PROGRESS_KEYS: dict[str, set[int]] = {}


async def _job_result_link(task: JobTask, event_id: int, db: Database) -> str | None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            return None
        if task == JobTask.telegraph_build:
            return ev.telegraph_url
        if task == JobTask.vk_sync:
            return ev.source_vk_post_url
        if task == JobTask.ics_publish:
            return ev.ics_url
        if task == JobTask.tg_ics_post:
            return ev.ics_post_url
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
                return page.vk_post_url if page else None
            return None
        if task == JobTask.festival_pages:
            if ev.festival:
                fest = (
                    await session.execute(
                        select(Festival).where(Festival.name == ev.festival)
                    )
                ).scalar_one_or_none()
                return fest.vk_post_url if fest else None
            return None
    return None


async def reconcile_job_outbox(db: Database) -> None:
    now = datetime.now(timezone.utc)
    async with db.get_session() as session:
        await session.execute(
            update(JobOutbox)
            .where(JobOutbox.status == JobStatus.running)
            .values(status=JobStatus.error, next_run_at=now, updated_at=now)
        )
        await session.commit()


_run_due_jobs_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = WeakKeyDictionary()


def _get_run_due_jobs_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _run_due_jobs_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _run_due_jobs_locks[loop] = lock
    return lock


def _reset_run_due_jobs_locks() -> None:
    _run_due_jobs_locks.clear()


async def _run_due_jobs_once(
    db: Database,
    bot: Bot,
    notify: Callable[[JobTask, int, JobStatus, bool, str | None, str | None], Awaitable[None]] | None = None,
    only_event: int | None = None,
    ics_progress: dict[int, Any] | Any | None = None,
    fest_progress: dict[int, Any] | Any | None = None,
    allowed_tasks: set[JobTask] | None = None,
    force_notify: bool = False,
) -> int:
    async with _get_run_due_jobs_lock():
        return await _run_due_jobs_once_locked(
            db,
            bot,
            notify=notify,
            only_event=only_event,
            ics_progress=ics_progress,
            fest_progress=fest_progress,
            allowed_tasks=allowed_tasks,
            force_notify=force_notify,
        )


async def _run_due_jobs_once_locked(
    db: Database,
    bot: Bot,
    notify: Callable[[JobTask, int, JobStatus, bool, str | None, str | None], Awaitable[None]] | None = None,
    only_event: int | None = None,
    ics_progress: dict[int, Any] | Any | None = None,
    fest_progress: dict[int, Any] | Any | None = None,
    allowed_tasks: set[JobTask] | None = None,
    force_notify: bool = False,
) -> int:
    now = datetime.now(timezone.utc)
    async with db.get_session() as session:
        running_rows = await session.execute(
            select(JobOutbox).where(JobOutbox.status == JobStatus.running)
        )
        running_jobs = [_normalize_job(job) for job in running_rows.scalars().all()]
        stale: list[str] = []
        for rjob in running_jobs:
            limit = JOB_MAX_RUNTIME.get(rjob.task, DEFAULT_JOB_MAX_RUNTIME)
            age = (now - rjob.updated_at).total_seconds()
            if age > limit:
                rjob.status = JobStatus.error
                rjob.last_error = "stale"
                rjob.updated_at = now
                rjob.next_run_at = now + timedelta(days=3650)
                session.add(rjob)
                stale.append(
                    rjob.coalesce_key or f"{rjob.task.value}:{rjob.event_id}"
                )
        if stale:
            logging.info("OUTBOX_STALE keys=%s", ",".join(stale))
        await session.commit()
    async with db.get_session() as session:
        stmt = (
            select(JobOutbox)
            .where(
                JobOutbox.status.in_([JobStatus.pending, JobStatus.error]),
                JobOutbox.next_run_at <= now,
            )
        )
        if only_event is not None:
            stmt = stmt.where(JobOutbox.event_id == only_event)
        if allowed_tasks:
            stmt = stmt.where(JobOutbox.task.in_(allowed_tasks))
        jobs = [
            _normalize_job(job) for job in (await session.execute(stmt)).scalars().all()
        ]
    priority = {
        JobTask.telegraph_build: 0,
        JobTask.ics_publish: 0,
        JobTask.tg_ics_post: 0,
        JobTask.month_pages: 1,
        JobTask.week_pages: 1,
        JobTask.weekend_pages: 1,
        JobTask.festival_pages: 1,
        JobTask.vk_sync: 2,
    }
    jobs.sort(key=lambda j: (priority.get(j.task, 99), j.id))
    processed = 0
    for job in jobs:
        async with db.get_session() as session:
            obj = _normalize_job(await session.get(JobOutbox, job.id))
            if not obj or obj.status not in (JobStatus.pending, JobStatus.error):
                continue
            ttl = JOB_TTL.get(obj.task, DEFAULT_JOB_TTL)
            age = (now - obj.updated_at).total_seconds()
            if age > ttl:
                obj.status = JobStatus.error
                obj.last_error = "expired"
                obj.updated_at = now
                obj.next_run_at = now + timedelta(days=3650)
                session.add(obj)
                await session.commit()
                logging.info(
                    "OUTBOX_EXPIRED key=%s",
                    obj.coalesce_key or f"{obj.task.value}:{obj.event_id}",
                )
                logline(
                    "RUN",
                    obj.event_id,
                    "skip",
                    job_id=obj.id,
                    task=obj.task.value,
                    reason="expired",
                )
                continue
            if obj.coalesce_key:
                later = await session.execute(
                    select(JobOutbox.id)
                        .where(
                            JobOutbox.coalesce_key == obj.coalesce_key,
                            JobOutbox.id > obj.id,
                            JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                        )
                        .limit(1)
                )
                if later.first():
                    obj.status = JobStatus.error
                    obj.last_error = "superseded"
                    obj.updated_at = now
                    obj.next_run_at = now + timedelta(days=3650)
                    session.add(obj)
                    await session.commit()
                    logging.info(
                        "OUTBOX_SUPERSEDED key=%s", obj.coalesce_key
                    )
                    logline(
                        "RUN",
                        obj.event_id,
                        "skip",
                        job_id=obj.id,
                        task=obj.task.value,
                        reason="superseded",
                    )
                    continue
            exists_stmt = (
                select(
                    JobOutbox.id,
                    JobOutbox.task,
                    JobOutbox.status,
                    JobOutbox.next_run_at,
                )
                .where(
                    JobOutbox.event_id == obj.event_id,
                    JobOutbox.id < obj.id,
                    JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                    JobOutbox.next_run_at <= now,
                )
                .limit(1)
            )
            if obj.task == JobTask.ics_publish:
                exists_stmt = exists_stmt.where(JobOutbox.task == JobTask.ics_publish)
            early = (await session.execute(exists_stmt)).first()
            if early:
                ejob = early[0]
                etask = early[1]
                estat = early[2]
                enext = early[3]
                logging.info(
                    "RUN skip eid=%s task=%s blocked_by id=%s task=%s status=%s next_run_at=%s",
                    obj.event_id,
                    obj.task.value,
                    ejob,
                    etask.value if isinstance(etask, JobTask) else etask,
                    estat.value if isinstance(estat, JobStatus) else estat,
                    enext.isoformat() if enext else None,
                )
                logline(
                    "RUN",
                    obj.event_id,
                    "skip",
                    job_id=obj.id,
                    task=obj.task.value,
                    blocking_id=ejob,
                    blocking_task=etask.value if isinstance(etask, JobTask) else etask,
                    blocking_status=estat.value if isinstance(estat, JobStatus) else estat,
                    blocking_run_at=enext.isoformat() if enext else None,
                )
                continue
            obj.status = JobStatus.running
            obj.updated_at = datetime.now(timezone.utc)
            session.add(obj)
            await session.commit()
        run_id = uuid.uuid4().hex
        attempt = job.attempts + 1
        job_key = obj.coalesce_key or f"{obj.task.value}:{obj.event_id}"
        logging.info(
            "RUN pick key=%s owner_eid=%s started_at=%s attempts=%d",
            job_key,
            obj.event_id,
            obj.updated_at.isoformat(),
            attempt,
        )
        logline(
            "RUN",
            obj.event_id,
            "start",
            job_id=obj.id,
            task=obj.task.value,
            key=job_key,
        )
        start = _time.perf_counter()
        changed = True
        handler = JOB_HANDLERS.get(obj.task.value)
        pause = False
        if not handler:
            status = JobStatus.done
            err = None
            changed = False
            link = None
            took_ms = (_time.perf_counter() - start) * 1000
            logline(
                "RUN",
                obj.event_id,
                "done",
                job_id=obj.id,
                task=obj.task.value,
                result="nochange",
            )
        else:
            try:
                async with span(
                    "event_pipeline", step=obj.task.value, event_id=obj.event_id
                ):
                    if obj.task == JobTask.ics_publish:
                        prog = (
                            ics_progress.get(job.event_id)
                            if isinstance(ics_progress, dict)
                            else ics_progress
                        )
                        res = await handler(obj.event_id, db, bot, prog)
                    elif obj.task == JobTask.festival_pages:
                        fest_prog = (
                            fest_progress.get(job.event_id)
                            if isinstance(fest_progress, dict)
                            else fest_progress
                        )
                        res = await handler(obj.event_id, db, bot, fest_prog)
                    else:
                        res = await handler(obj.event_id, db, bot)
                rebuild = isinstance(res, str) and res == "rebuild"
                changed = res if isinstance(res, bool) else True
                link = await _job_result_link(obj.task, obj.event_id, db)
                if rebuild and link:
                    link += " (forced rebuild)"
                status = JobStatus.done
                err = None
                took_ms = (_time.perf_counter() - start) * 1000
                short = link or ("ok" if changed else "nochange")
                logline(
                    "RUN",
                    obj.event_id,
                    "done",
                    job_id=obj.id,
                    task=obj.task.value,
                    result_url=link,
                    result="changed" if changed else "nochange",
                )
            except Exception as exc:  # pragma: no cover - log and backoff
                took_ms = (_time.perf_counter() - start) * 1000
                pause = False
                if isinstance(exc, VKAPIError):
                    if exc.code == 14:
                        err = "captcha"
                        status = JobStatus.paused
                        pause = True
                        retry = False
                        global _vk_captcha_key
                        _vk_captcha_key = job_key
                        logline(
                            "VK",
                            obj.event_id,
                            "paused captcha",
                            group=f"@{VK_AFISHA_GROUP_ID}" if VK_AFISHA_GROUP_ID else None,
                        )
                    else:
                        prefix = (
                            "ошибка публикации VK"
                            if exc.method and exc.method.startswith("wall.")
                            else "ошибка VK"
                        )
                        err = f"{prefix}: {exc.code}/{exc.message} ({exc.method})"
                        status = JobStatus.error
                        retry = not isinstance(exc, VKPermissionError)
                else:
                    err = str(exc)
                    status = JobStatus.error
                    retry = True
                logline(
                    "RUN",
                    obj.event_id,
                    "error",
                    job_id=obj.id,
                    task=obj.task.value,
                    exc=err.splitlines()[0],
                )
                logging.exception("job %s failed", job.id)
                link = None
        logging.info(
            "RUN done key=%s status=%s duration_ms=%.0f",
            job_key,
            "ok" if status == JobStatus.done else "fail",
            took_ms,
        )
        text = None
        async with db.get_session() as session:
            obj = await session.get(JobOutbox, obj.id)
            send = True
            if obj:
                prev = obj.last_result
                obj.status = status
                obj.last_error = err
                obj.updated_at = datetime.now(timezone.utc)
                if status == JobStatus.done:
                    cur_res = link if link else ("ok" if changed else "nochange")
                    if cur_res == prev and not force_notify:
                        send = False
                    obj.last_result = cur_res
                    obj.next_run_at = datetime.now(timezone.utc)
                else:
                    if retry:
                        obj.attempts += 1
                        delay = BACKOFF_SCHEDULE[
                            min(obj.attempts - 1, len(BACKOFF_SCHEDULE) - 1)
                        ]
                        obj.next_run_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                    else:
                        obj.next_run_at = datetime.now(timezone.utc) + timedelta(days=3650)
                session.add(obj)
                await session.commit()
            if notify and send:
                await notify(job.task, job.event_id, status, changed, link, err)
            if job.coalesce_key:
                for eid in _EVENT_PROGRESS_KEYS.get(job.coalesce_key, set()):
                    if eid == job.event_id:
                        continue
                    ctx = _EVENT_PROGRESS.get(eid)
                    if not ctx:
                        continue
                    try:
                        await ctx.updater(job.task, eid, status, changed, link, err)
                    except Exception:
                        logging.exception("progress callback error eid=%s", eid)
        processed += 1
        if pause:
            await vk_captcha_pause_outbox(db)
            continue
    return processed


async def _log_job_outbox_stats(db: Database) -> None:
    now = datetime.now(timezone.utc)
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
        next_run = _ensure_utc(lag_res.scalar())
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


_nav_watchdog_warned: set[str] = set()


async def _watch_nav_jobs(db: Database, bot: Bot) -> None:
    now = datetime.now(timezone.utc) - timedelta(seconds=60)
    async with db.get_session() as session:
        rows = await session.execute(
            select(JobOutbox)
            .where(
                JobOutbox.task.in_(
                    [JobTask.month_pages, JobTask.week_pages, JobTask.weekend_pages]
                ),
                JobOutbox.status == JobStatus.pending,
                JobOutbox.updated_at < now,
            )
        )
        jobs = [_normalize_job(job) for job in rows.scalars().all()]
    for job in jobs:
        if not job.coalesce_key or job.coalesce_key in _nav_watchdog_warned:
            continue
        blockers: list[str] = []
        async with db.get_session() as session:
            early = await session.execute(
                select(JobOutbox.task)
                .where(
                    JobOutbox.event_id == job.event_id,
                    JobOutbox.id < job.id,
                    JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                )
            )
            tasks = [t.value for t in early.scalars().all()]
            if tasks:
                blockers.append("prior:" + ",".join(tasks))
            deps = [d for d in (job.depends_on or "").split(",") if d]
            if deps:
                dep_rows = await session.execute(
                    select(JobOutbox.coalesce_key)
                    .where(
                        JobOutbox.coalesce_key.in_(deps),
                        JobOutbox.status.in_([JobStatus.pending, JobStatus.running]),
                    )
                )
                dep_keys = [c for c in dep_rows.scalars().all()]
                if dep_keys:
                    blockers.append("depends_on:" + ",".join(dep_keys))
        blockers.append(f"next_run_at:{job.next_run_at.isoformat()}")
        msg = f"NAV_WATCHDOG key={job.coalesce_key} blocked_by={' ; '.join(blockers)}"
        logging.warning(msg)
        await notify_superadmin(db, bot, msg)
        _nav_watchdog_warned.add(job.coalesce_key)


async def job_outbox_worker(db: Database, bot: Bot, interval: float = 2.0):
    last_log = 0.0
    while True:
        try:
            async def notifier(
                task: JobTask,
                eid: int,
                status: JobStatus,
                changed: bool,
                link: str | None,
                err: str | None,
            ) -> None:
                ctx = _EVENT_PROGRESS.get(eid)
                if ctx:
                    await ctx.updater(task, eid, status, changed, link, err)

            ics_map = {
                eid: ctx.ics_progress
                for eid, ctx in _EVENT_PROGRESS.items()
                if ctx.ics_progress
            }
            fest_map = {
                eid: ctx.fest_progress
                for eid, ctx in _EVENT_PROGRESS.items()
                if ctx.fest_progress
            }
            await _run_due_jobs_once(
                db,
                bot,
                notifier,
                None,
                ics_map if ics_map else None,
                fest_map if fest_map else None,
            )
            await _watch_nav_jobs(db, bot)
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
        eid: int,
        status: JobStatus,
        changed: bool,
        link: str | None,
        err: str | None,
    ) -> None:
        label = TASK_LABELS[task.value]
        text = None
        if status == JobStatus.done:
            if changed:
                if task == JobTask.month_pages and not link:
                    text = f"{label}: создано/обновлено"
                else:
                    text = f"{label}: OK"
                if link:
                    text += f" — {link}"
            else:
                text = f"{label}: без изменений"
        elif status == JobStatus.error:
            err_short = err.splitlines()[0] if err else ""
            if task == JobTask.ics_publish and "temporary network error" in err_short.lower():
                text = f"{label}: временная ошибка сети, будет повтор"
            else:
                text = f"{label}: ERROR: {err_short}"
        if notify_chat_id is not None and text:
            await bot.send_message(notify_chat_id, text)

    while await _run_due_jobs_once(
        db, bot, notifier if notify_chat_id is not None else None, event_id
    ):
        await asyncio.sleep(0)


def festival_event_slug(ev: Event, fest: Festival | None) -> str | None:
    """Return deterministic slug for festival day events."""
    if not fest or not fest.id:
        return None
    d = parse_iso_date(ev.date)
    start = parse_iso_date(fest.start_date) if fest.start_date else None
    if d and start:
        day_num = (d - start).days + 1
    else:
        day_num = 1
    base = f"fest-{fest.id}-day-{day_num}-{ev.date}-{ev.city or ''}"
    return slugify(base)


async def ensure_event_telegraph_link(e: Event, fest: Festival | None, db: Database) -> None:
    """Populate ``e.telegraph_url`` without creating/editing Telegraph pages."""
    global DISABLE_EVENT_PAGE_UPDATES
    if e.telegraph_url:
        return
    if DISABLE_EVENT_PAGE_UPDATES:
        e.telegraph_url = e.telegraph_url or e.source_post_url or ""
        return
    if e.telegraph_path:
        url = normalize_telegraph_url(f"https://telegra.ph/{e.telegraph_path.lstrip('/')}")
        e.telegraph_url = url
        async with db.get_session() as session:
            await session.execute(
                update(Event).where(Event.id == e.id).values(telegraph_url=url)
            )
            await session.commit()
        return
    e.telegraph_url = e.source_post_url or ""


async def update_telegraph_event_page(
    event_id: int, db: Database, bot: Bot | None
) -> str | None:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            return None
        display_link = False if ev.source_post_url else True
        summary = SourcePageEventSummary(
            date=getattr(ev, "date", None),
            end_date=getattr(ev, "end_date", None),
            time=getattr(ev, "time", None),
            event_type=getattr(ev, "event_type", None),
            location_name=(ev.location_name or None),
            location_address=(ev.location_address or None),
            city=(ev.city or None),
            ticket_price_min=getattr(ev, "ticket_price_min", None),
            ticket_price_max=getattr(ev, "ticket_price_max", None),
            ticket_link=(ev.ticket_link or None),
            is_free=bool(getattr(ev, "is_free", False)),
        )
        html_content, _, _ = await build_source_page_content(
            ev.title or "Event",
            ev.source_text,
            ev.source_post_url,
            ev.source_text,
            None,
            ev.ics_url,
            db,
            event_summary=summary,
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
                content=nodes,
                return_content=False,
                caller="event_pipeline",
                eid=ev.id,
            )
            ev.telegraph_url = normalize_telegraph_url(data.get("url"))
            ev.telegraph_path = data.get("path")
        else:
            await telegraph_edit_page(
                tg,
                ev.telegraph_path,
                title=title,
                content=nodes,
                return_content=False,
                caller="event_pipeline",
                eid=ev.id,
            )
        ev.content_hash = new_hash
        session.add(ev)
        await session.commit()
        url = ev.telegraph_url

    logline("TG-EVENT", event_id, "done", url=url)
    await update_month_pages_for(event_id, db, bot)
    return url


def ensure_day_markers(page_html: str, d: date) -> tuple[str, bool]:
    """Ensure that DAY markers for a date exist on a month page.

    Inserts an empty marker block ``<!--DAY:YYYY-MM-DD START-->`` … ``END``
    if missing and it's safe to do so. The block is placed before the
    ``PERM_START`` section when present, otherwise appended to the end of the
    document. Returns the possibly updated HTML and a boolean flag indicating
    whether any changes were made.

    If a header containing the rendered date already exists, the function
    assumes the page has legacy content and leaves it untouched.
    """

    start_marker = DAY_START(d)
    end_marker = DAY_END(d)
    if start_marker in page_html and end_marker in page_html:
        return page_html, False

    pretty = format_day_pretty(d)
    if pretty in page_html:
        return page_html, False

    insert = f"{start_marker}{end_marker}"
    for m in re.finditer(r"<!--DAY:(\d{4}-\d{2}-\d{2}) START-->", page_html):
        existing = date.fromisoformat(m.group(1))
        if existing > d:
            idx = m.start()
            page_html = page_html[:idx] + insert + page_html[idx:]
            return page_html, True
    idx = page_html.find(PERM_START)
    if idx != -1:
        page_html = page_html[:idx] + insert + page_html[idx:]
    else:
        page_html += insert
    return page_html, True


def _parse_pretty_date(text: str, year: int) -> date | None:
    m = re.match(r"(\d{1,2})\s+([а-яё]+)", text.strip(), re.IGNORECASE)
    if not m:
        return None
    day = int(m.group(1))
    month = {name: i + 1 for i, name in enumerate(MONTHS)}.get(m.group(2).lower())
    if not month:
        return None
    return date(year, month, day)


def locate_month_day_page(page_html_1: str, page_html_2: str | None, d: date) -> int:
    """Return which part of a split month page should contain ``d``.

    Returns ``1`` for the first part or ``2`` for the second. ``page_html_2``
    can be ``None`` for non-split months.
    """

    if not page_html_2:
        return 1

    start_marker = DAY_START(d)
    end_marker = DAY_END(d)
    legacy_start = f"<!-- DAY:{d.isoformat()} START -->"
    legacy_end = f"<!-- DAY:{d.isoformat()} END -->"
    header = f"<h3>🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥</h3>"

    markers1 = start_marker in page_html_1 and end_marker in page_html_1
    markers2 = start_marker in page_html_2 and end_marker in page_html_2
    if markers2:
        return 2
    if markers1:
        return 1

    legacy1 = legacy_start in page_html_1 and legacy_end in page_html_1
    legacy2 = legacy_start in page_html_2 and legacy_end in page_html_2
    if legacy2:
        return 2
    if legacy1:
        return 1

    header1 = header in page_html_1
    header2 = header in page_html_2
    if header2:
        return 2
    if header1:
        return 1

    def dates_from_html(html: str) -> list[date]:
        dates: list[date] = []
        for m in re.finditer(r"<!--DAY:(\d{4}-\d{2}-\d{2}) START-->", html):
            dates.append(date.fromisoformat(m.group(1)))
        for m in re.finditer(r"<h3>🟥🟥🟥 (\d{1,2} [^<]+) 🟥🟥🟥</h3>", html):
            parsed = _parse_pretty_date(m.group(1), d.year)
            if parsed:
                dates.append(parsed)
        return dates

    dates1 = dates_from_html(page_html_1)
    dates2 = dates_from_html(page_html_2)

    if dates1 and max(dates1) < d:
        return 2
    if not dates1 and page_html_2:
        if len(page_html_1.encode()) > TELEGRAPH_LIMIT * 0.9:
            return 2
        return 2
    return 1


async def split_month_until_ok(
    db: Database,
    tg: Telegraph,
    page: MonthPage,
    month: str,
    events: list[Event],
    exhibitions: list[Exhibition],
    nav_block: str,
) -> None:
    from telegraph.utils import nodes_to_html

    if len(events) < 2:
        raise RuntimeError(
            f"split_month_until_ok: cannot split {month} without at least two events"
        )

    def event_day_key(ev: Event) -> str | None:
        parsed = parse_iso_date(ev.date)
        if parsed:
            return parsed.isoformat()
        raw = (ev.date or "").split("..", 1)[0].strip()
        return raw or None

    day_boundaries: list[int] = []
    prev_key = event_day_key(events[0])
    for idx, ev in enumerate(events[1:], start=1):
        key = event_day_key(ev)
        if key is None:
            continue
        if prev_key is None:
            prev_key = key
            continue
        if key != prev_key:
            day_boundaries.append(idx)
        prev_key = key

    if not day_boundaries:
        raise RuntimeError(
            f"split_month_until_ok: no valid day boundary found for {month}"
        )

    def snap_index(idx: int, *, direction: int | None = None) -> int:
        if not day_boundaries:
            raise RuntimeError(
                f"split_month_until_ok: no valid day boundary found for {month}"
            )
        idx = max(1, min(len(events) - 1, idx))
        if direction is None:
            pos = bisect_left(day_boundaries, idx)
            candidates: list[int] = []
            if pos < len(day_boundaries):
                candidates.append(day_boundaries[pos])
            if pos > 0:
                candidates.append(day_boundaries[pos - 1])
            if not candidates:
                return day_boundaries[0]
            return min(candidates, key=lambda b: (abs(b - idx), b))
        if direction < 0:
            pos = bisect_left(day_boundaries, idx)
            if pos < len(day_boundaries) and day_boundaries[pos] == idx:
                return day_boundaries[pos]
            if pos > 0:
                return day_boundaries[pos - 1]
            return day_boundaries[0]
        pos = bisect_left(day_boundaries, idx)
        if pos < len(day_boundaries):
            return day_boundaries[pos]
        return day_boundaries[-1]

    async def attempt(include_ics: bool) -> None:
        title, content, _ = await build_month_page_content(
            db, month, events, exhibitions, include_ics=include_ics
        )
        html_full = unescape_html_comments(nodes_to_html(content))
        html_full = ensure_footer_nav_with_hr(html_full, nav_block, month=month, page=1)
        total_size = len(html_full.encode())
        avg = total_size / len(events) if events else total_size
        base_idx = max(1, min(len(events) - 1, int(TELEGRAPH_LIMIT // avg)))
        split_idx = snap_index(base_idx)
        logging.info(
            "month_split start month=%s events=%d total_bytes=%d nav_bytes=%d split_idx=%d include_ics=%s",
            month,
            len(events),
            total_size,
            len(nav_block),
            split_idx,
            include_ics,
        )
        attempts = 0
        fallback_reason = ""
        saw_both_too_big = False
        while attempts < 50:
            attempts += 1
            first, second = events[:split_idx], events[split_idx:]
            title2, content2, _ = await build_month_page_content(
                db, month, second, exhibitions, include_ics=include_ics
            )
            rough2 = rough_size(content2) + len(nav_block)
            title1, content1, _ = await build_month_page_content(
                db,
                month,
                first,
                [],
                continuation_url="x",
                include_ics=include_ics,
            )
            rough1 = rough_size(content1) + len(nav_block) + 200
            logging.info(
                "month_split try attempt=%d idx=%d first_events=%d second_events=%d rough1=%d rough2=%d include_ics=%s",
                attempts,
                split_idx,
                len(first),
                len(second),
                rough1,
                rough2,
                include_ics,
            )
            if rough1 > TELEGRAPH_LIMIT and rough2 > TELEGRAPH_LIMIT:
                logging.info(
                    "month_split forcing attempt idx=%d include_ics=%s",
                    split_idx,
                    include_ics,
                )
                saw_both_too_big = True
            if rough1 > TELEGRAPH_LIMIT:
                delta = max(1, split_idx // 6)
                new_idx = snap_index(split_idx - delta, direction=-1)
                if new_idx != split_idx:
                    split_idx = new_idx
                    logging.info(
                        "month_split adjust idx=%d reason=rough_size target=first include_ics=%s",
                        split_idx,
                        include_ics,
                    )
                    continue
            elif rough2 > TELEGRAPH_LIMIT:
                delta = max(1, (len(events) - split_idx) // 6)
                new_idx = snap_index(split_idx + delta, direction=1)
                if new_idx != split_idx:
                    split_idx = new_idx
                    logging.info(
                        "month_split adjust idx=%d reason=rough_size target=second include_ics=%s",
                        split_idx,
                        include_ics,
                    )
                    continue
            html2 = unescape_html_comments(nodes_to_html(content2))
            html2 = ensure_footer_nav_with_hr(html2, nav_block, month=month, page=2)
            hash2 = content_hash(html2)
            try:
                if not page.path2:
                    logging.info("creating second page for %s", month)
                    data2 = await telegraph_create_page(
                        tg,
                        title=title2,
                        html_content=html2,
                        caller="month_build",
                    )
                    page.url2 = normalize_telegraph_url(data2.get("url"))
                    page.path2 = data2.get("path")
                else:
                    logging.info("updating second page for %s", month)
                    start = _time.perf_counter()
                    await telegraph_edit_page(
                        tg,
                        page.path2,
                        title=title2,
                        html_content=html2,
                        caller="month_build",
                    )
                    dur = (_time.perf_counter() - start) * 1000
                    logging.info("editPage %s done in %.0f ms", page.path2, dur)
            except TelegraphException as e:
                msg = str(e).lower()
                if "content" in msg and "too" in msg and "big" in msg:
                    delta = max(1, (len(events) - split_idx) // 6)
                    new_idx = snap_index(split_idx + delta, direction=1)
                    if new_idx != split_idx:
                        split_idx = new_idx
                        logging.info(
                            "month_split adjust idx=%d reason=telegraph_too_big target=second include_ics=%s",
                            split_idx,
                            include_ics,
                        )
                        continue
                raise
            page.content_hash2 = hash2
            await asyncio.sleep(0)
            title1, content1, _ = await build_month_page_content(
                db,
                month,
                first,
                [],
                continuation_url=page.url2,
                include_ics=include_ics,
            )
            html1 = unescape_html_comments(nodes_to_html(content1))
            html1 = ensure_footer_nav_with_hr(html1, nav_block, month=month, page=1)
            hash1 = content_hash(html1)
            try:
                if not page.path:
                    logging.info("creating first page for %s", month)
                    data1 = await telegraph_create_page(
                        tg,
                        title=title1,
                        html_content=html1,
                        caller="month_build",
                    )
                    page.url = normalize_telegraph_url(data1.get("url"))
                    page.path = data1.get("path")
                else:
                    logging.info("updating first page for %s", month)
                    start = _time.perf_counter()
                    await telegraph_edit_page(
                        tg,
                        page.path,
                        title=title1,
                        html_content=html1,
                        caller="month_build",
                    )
                    dur = (_time.perf_counter() - start) * 1000
                    logging.info("editPage %s done in %.0f ms", page.path, dur)
            except TelegraphException as e:
                msg = str(e).lower()
                if "content" in msg and "too" in msg and "big" in msg:
                    delta = max(1, split_idx // 6)
                    new_idx = snap_index(split_idx - delta, direction=-1)
                    if new_idx != split_idx:
                        split_idx = new_idx
                        logging.info(
                            "month_split adjust idx=%d reason=telegraph_too_big target=first include_ics=%s",
                            split_idx,
                            include_ics,
                        )
                        continue
                raise
            page.content_hash = hash1
            async with db.get_session() as session:
                db_page = await session.get(MonthPage, month)
                db_page.url = page.url
                db_page.path = page.path
                db_page.url2 = page.url2
                db_page.path2 = page.path2
                db_page.content_hash = page.content_hash
                db_page.content_hash2 = page.content_hash2
                await session.commit()
            logging.info(
                "month_split done month=%s idx=%d first_bytes=%d second_bytes=%d include_ics=%s",
                month,
                split_idx,
                rough1,
                rough2,
                include_ics,
            )
            return
        if saw_both_too_big:
            fallback_reason = "both_too_big"
        if not fallback_reason:
            fallback_reason = "attempts_exhausted"
        logging.error(
            "month_split failed month=%s attempts=%d last_idx=%d include_ics=%s reason=%s",
            month,
            attempts,
            split_idx,
            include_ics,
            fallback_reason,
        )
        raise TelegraphException("CONTENT_TOO_BIG")

    try:
        await attempt(True)
        return
    except TelegraphException as exc:
        msg = str(exc).lower()
        if "content" not in msg or "too" not in msg or "big" not in msg:
            raise
        logging.info("month_split retry_without_ics month=%s", month)

    await attempt(False)


async def patch_month_page_for_date(
    db: Database, telegraph: Telegraph, month_key: str, d: date, _retried: bool = False
) -> bool:
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

    # обогащаем события ссылкой на телеграф-страницу, если она уже есть
    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}
    for ev in events:
        fest = fest_map.get((ev.festival or "").casefold())
        await ensure_event_telegraph_link(ev, fest, db)
        if fest:
            setattr(ev, "_festival", fest)

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
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(func, *args, **kwargs), 7
                )
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(random.uniform(0, 1))
                    continue
                raise

    split = bool(page.path2)
    if split:
        data1, data2 = await asyncio.gather(
            tg_call(telegraph.get_page, page.path, return_html=True),
            tg_call(telegraph.get_page, page.path2, return_html=True),
        )
        html1 = unescape_html_comments(
            data1.get("content") or data1.get("content_html") or ""
        )
        html2 = unescape_html_comments(
            data2.get("content") or data2.get("content_html") or ""
        )

        from telegraph.utils import html_to_nodes

        nodes1 = html_to_nodes(html1)
        sections1, need_rebuild1 = parse_month_sections(nodes1, page=1)
        if not sections1:
            sections1, need_rebuild1 = parse_month_sections(nodes1, page=1)
        nodes2 = html_to_nodes(html2)
        sections2, need_rebuild2 = parse_month_sections(nodes2, page=2)

        dates1 = [date(d.year, s.date.month, s.date.day) for s in sections1]
        dates2 = [date(d.year, s.date.month, s.date.day) for s in sections2]
        p1_min = min(dates1).isoformat() if dates1 else ""
        p1_max = max(dates1).isoformat() if dates1 else ""
        p2_min = min(dates2).isoformat() if dates2 else ""
        p2_max = max(dates2).isoformat() if dates2 else ""

        if not dates1:
            part = 1
        elif not dates2:
            part = 2 if p1_max and d > date.fromisoformat(p1_max) else 1
        elif d <= date.fromisoformat(p1_max):
            part = 1
        elif d >= date.fromisoformat(p2_min):
            part = 2
        else:
            if len(dates1) < len(dates2):
                part = 1
            elif len(dates2) < len(dates1):
                part = 2
            else:
                if len(html1) < len(html2):
                    part = 1
                elif len(html2) < len(html1):
                    part = 2
                else:
                    part = 1

        if part == 1:
            html_content = html1
            page_path = page.path
            title = data1.get("title") or month_key
            hash_attr = "content_hash"
            nodes = nodes1
            sections = sections1
            need_rebuild = need_rebuild1
        else:
            html_content = html2
            page_path = page.path2
            title = data2.get("title") or month_key
            hash_attr = "content_hash2"
            nodes = nodes2
            sections = sections2
            need_rebuild = need_rebuild2
    else:
        data1 = await tg_call(telegraph.get_page, page.path, return_html=True)
        html_content = unescape_html_comments(
            data1.get("content") or data1.get("content_html") or ""
        )
        page_path = page.path
        title = data1.get("title") or month_key
        hash_attr = "content_hash"
        part = 1
        from telegraph.utils import html_to_nodes

        nodes = html_to_nodes(html_content)
        sections, need_rebuild = parse_month_sections(nodes, page=1)
        dates1 = [date(d.year, s.date.month, s.date.day) for s in sections]
        p1_min = min(dates1).isoformat() if dates1 else ""
        p1_max = max(dates1).isoformat() if dates1 else ""
        p2_min = p2_max = ""

    logging.info(
        "TG-MONTH select: p1=[%s..%s], p2=[%s..%s], target=%s → page%s",
        p1_min,
        p1_max,
        p2_min,
        p2_max,
        d.isoformat(),
        "2" if part == 2 else "1",
    )

    logging.info(
        "patch_month_day update_links=True anchor=h3 part=%s",
        2 if hash_attr == "content_hash2" else 1,
    )
    from telegraph.utils import nodes_to_html

    if need_rebuild:
        logging.info(
            "month_patch inline rebuild month=%s day=%s", month_key, d.isoformat()
        )
        from telegraph.utils import html_to_nodes, nodes_to_html

        day_nodes = html_to_nodes(html_section)
        nav_block = await build_month_nav_block(db, month_key)
        nodes = ensure_footer_nav_with_hr(day_nodes, nav_block, month=month_key, page=part)
        nodes, removed = dedup_same_date(nodes, d)
        updated_html = nodes_to_html(nodes)
        updated_html, _ = ensure_day_markers(updated_html, d)
        updated_html = lint_telegraph_html(updated_html)
        await telegraph_edit_page(
            telegraph,
            page_path,
            title=title,
            html_content=updated_html,
            caller="month_build",
        )
        await set_section_hash(db, page_key, section_key, content_hash(updated_html))
        async with db.get_session() as session:
            db_page = await session.get(MonthPage, month_key)
            setattr(db_page, hash_attr, content_hash(updated_html))
            await session.commit()
        logging.info(
            "month_patch inline rebuild done month=%s day=%s", month_key, d.isoformat()
        )
        return True

    logging.info(
        "TG-MONTH dates=%s page=%s",
        [date(d.year, s.date.month, s.date.day).isoformat() for s in sections],
        part,
    )

    day_nodes = html_to_nodes(html_section)

    target_sec = next(
        (s for s in sections if s.date.month == d.month and s.date.day == d.day),
        None,
    )

    anchor: str
    if target_sec:
        nodes[target_sec.start_idx : target_sec.end_idx] = day_nodes
        anchor = "replace"
    else:
        after_sec = next(
            (
                s
                for s in sections
                if (s.date.month, s.date.day) > (d.month, d.day)
            ),
            None,
        )

        def _index_before_last_hr(nodes_list: List[Any]) -> int:
            for idx in range(len(nodes_list) - 1, -1, -1):
                n = nodes_list[idx]
                if isinstance(n, dict) and n.get("tag") == "hr":
                    return idx
                if (
                    isinstance(n, dict)
                    and n.get("tag") in {"p", "figure"}
                    and n.get("children")
                    and isinstance(n["children"][0], dict)
                    and n["children"][0].get("tag") == "hr"
                ):
                    return idx
            return len(nodes_list)

        if after_sec:
            insert_at = after_sec.start_idx
            anchor = "insert_before=" + date(d.year, after_sec.date.month, after_sec.date.day).isoformat()
        else:
            insert_at = _index_before_last_hr(nodes)
            anchor = "insert_before_hr"
        nodes[insert_at:insert_at] = day_nodes
    logging.info(
        "TG-MONTH anchor=%s page=%s target=%s",
        anchor,
        part,
        d.isoformat(),
    )
    before_footer = nodes_to_html(nodes)
    nav_block = await build_month_nav_block(db, month_key)
    nodes = ensure_footer_nav_with_hr(nodes, nav_block, month=month_key, page=part)
    after_footer = nodes_to_html(nodes)
    footer_fixed = before_footer != after_footer

    nodes, removed = dedup_same_date(nodes, d)

    logging.info(
        "anchor=%s footer_fixed=%s dedup_removed=%d",
        anchor,
        str(footer_fixed).lower(),
        removed,
    )

    updated_html = nodes_to_html(nodes)

    updated_html, _ = ensure_day_markers(updated_html, d)

    changed = content_hash(updated_html) != content_hash(html_content)
    updated_html = lint_telegraph_html(updated_html)

    try:
        edit_start = _time.perf_counter()
        await telegraph_edit_page(
            telegraph,
            page_path,
            title=title,
            html_content=updated_html,
            caller="month_build",
        )
        edit_dur = (_time.perf_counter() - edit_start) * 1000
        logging.info(
            "month_patch edit path=%s dur=%.0fms result=%s",
            page_path,
            edit_dur,
            "changed" if changed else "nochange",
        )
    except TelegraphException as e:
        msg = str(e)
        if ("CONTENT_TOO_BIG" in msg or "content too big" in msg.lower()) and not _retried:
            logging.warning(
                "month_patch split-inline month=%s day=%s",
                month_key,
                d.isoformat(),
            )
            async with _page_locks[f"month:{month_key}"]:
                events_m, exhibitions = await get_month_data(db, month_key)
                nav_block = await build_month_nav_block(db, month_key)
                await split_month_until_ok(
                    db, telegraph, page, month_key, events_m, exhibitions, nav_block
                )
            logging.info(
                "month_patch retry month=%s day=%s", month_key, d.isoformat()
            )
            return await patch_month_page_for_date(
                db, telegraph, month_key, d, _retried=True
            )
        raise

    await set_section_hash(db, page_key, section_key, new_hash)

    async with db.get_session() as session:
        db_page = await session.get(MonthPage, month_key)
        setattr(db_page, hash_attr, content_hash(updated_html))
        await session.commit()

    dur = (_time.perf_counter() - start) * 1000
    logging.info(
        "month_patch page_key=%s day=%s branch=replace changed=%s dur=%.0fms",
        page_key,
        d.isoformat(),
        "True" if changed else "False",
        dur,
    )
    url = db_page.url2 if part == 2 else db_page.url
    logging.info(
        "TG-MONTH patch ym=%s date=%s target=%s anchor=%s footer_fixed=%s dedup_removed=%d changed=%s url=%s",
        month_key,
        d.isoformat(),
        "page2" if part == 2 else "page1",
        anchor,
        str(footer_fixed).lower(),
        removed,
        str(changed).lower(),
        url,
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

    tg = Telegraph(access_token=token)

    changed_any = False
    rebuild_any = False
    rebuild_months: set[str] = set()
    for month, month_dates in months.items():
        # ensure the month page is created before attempting a patch
        await sync_month_page(db, month, update_links=True)
        for d in month_dates:
            logline("TG-MONTH", event_id, "patch start", month=month, day=d.isoformat())
            changed = await patch_month_page_for_date(db, tg, month, d)
            if changed == "rebuild":
                changed_any = True
                rebuild_any = True
                rebuild_months.add(month)
                async with db.get_session() as session:
                    page = await session.get(MonthPage, month)
                logline(
                    "TG-MONTH",
                    event_id,
                    "rebuild",
                    month=month,
                    url1=page.url,
                    url2=page.url2,
                )
            elif changed:
                changed_any = True
                async with db.get_session() as session:
                    page = await session.get(MonthPage, month)
                url = page.url2 or page.url
                logline("TG-MONTH", event_id, "patch changed", month=month, url=url)
            else:
                logline("TG-MONTH", event_id, "patch nochange", month=month)
    for month in rebuild_months:
        await sync_month_page(db, month, update_links=False, force=True)
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


async def ics_fix_nav(db: Database, month: str | None = None) -> int:
    """Enqueue rebuild jobs for events with ICS links."""
    if month:
        months = [month]
    else:
        today = date.today().replace(day=1)
        this_month = today.strftime("%Y-%m")
        next_month = (today + timedelta(days=32)).strftime("%Y-%m")
        months = [this_month, next_month]
    count = 0
    async with db.get_session() as session:
        for m in months:
            stmt = select(Event).where(
                Event.ics_url.is_not(None), Event.date.like(f"{m}%")
            )
            res = await session.execute(stmt)
            events = res.scalars().all()
            for ev in events:
                await enqueue_job(db, ev.id, JobTask.month_pages)
                await enqueue_job(db, ev.id, JobTask.weekend_pages)
                await enqueue_job(db, ev.id, JobTask.week_pages)
                count += 1
    logging.info("ics_fix_nav enqueued tasks for %s events", count)
    return count


async def update_festival_pages_for_event(
    event_id: int, db: Database, bot: Bot | None, progress: Any | None = None
) -> bool:
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev or not ev.festival:
        return False
    today = date.today().isoformat()
    end = ev.end_date or ev.date
    if end < today:
        return False
    try:
        url = await sync_festival_page(db, ev.festival)
        if progress:
            if url:
                progress.mark("tg", "done", url)
            else:
                progress.mark("tg", "skipped", "")
    except Exception as e:
        if progress:
            progress.mark("tg", "error", str(e))
        raise
    vk_changed = bool(await sync_festival_vk_post(db, ev.festival, bot))
    nav_changed = await rebuild_fest_nav_if_changed(db)
    if progress and nav_changed:
        url = await get_setting_value(db, "fest_index_url")
        progress.mark("index", "done", url or "")
    return vk_changed or nav_changed


async def publish_event_progress(
    event: Event,
    db: Database,
    bot: Bot,
    chat_id: int,
    initial_statuses: dict[JobTask, str] | None = None,
) -> None:
    d = parse_iso_date(event.date.split("..", 1)[0])
    coalesce_keys: list[str] = []
    if d:
        coalesce_keys.append(f"month_pages:{d.strftime('%Y-%m')}")
        week = d.isocalendar().week
        coalesce_keys.append(f"week_pages:{d.year}-{week:02d}")
        w_start = weekend_start_for_date(d)
        if w_start:
            coalesce_keys.append(f"weekend_pages:{w_start.isoformat()}")
    for key in coalesce_keys:
        _EVENT_PROGRESS_KEYS.setdefault(key, set()).add(event.id)
    async with db.get_session() as session:
        jobs = await session.execute(
            select(JobOutbox.task, JobOutbox.status, JobOutbox.last_result).where(
                (JobOutbox.event_id == event.id)
                | (JobOutbox.coalesce_key.in_(coalesce_keys))
            )
        )
        rows = jobs.all()
    tasks = []
    seen_tasks: set[JobTask] = set()
    for task, status, _ in rows:
        if task not in seen_tasks and task.value in TASK_LABELS:
            tasks.append(task)
            seen_tasks.add(task)
    progress: dict[JobTask, dict[str, str]] = {}
    for task, status, last_res in rows:
        if task.value not in TASK_LABELS:
            continue
        icon = "\U0001f504"
        suffix = ""
        action = initial_statuses.get(task) if initial_statuses else None
        link = last_res if last_res and last_res.startswith("http") else None
        if action == "skipped" or status == JobStatus.done:
            if link:
                icon = "✅"
                suffix = f" — {link}"
            else:
                icon = "⏭"
                suffix = " — актуально"
        elif action == "requeued":
            suffix = " — перезапущено"
        progress[task] = {"icon": icon, "suffix": suffix}
    vk_group = await get_vk_group_id(db)
    vk_scope = ""
    if vk_group:
        vk_scope = f"@{vk_group}" if not vk_group.startswith("-") else f"#{vk_group}"
    vk_tasks = VK_JOB_TASKS

    captcha_markup = None
    vk_present = any(t in vk_tasks for t in tasks)
    if _vk_captcha_needed and vk_present:
        captcha_markup = types.InlineKeyboardMarkup(
            inline_keyboard=[[types.InlineKeyboardButton(text="Ввести код", callback_data="captcha_input")]]
        )
        for t in tasks:
            if t in vk_tasks:
                progress[t] = {
                    "icon": "⏸",
                    "suffix": " — требуется капча; нажмите «Ввести код»",
                }

    def job_label(task: JobTask) -> str:
        if task == JobTask.month_pages:
            d = parse_iso_date(event.date.split("..", 1)[0])
            month_key = d.strftime("%Y-%m") if d else None
            if month_key:
                return f"Telegraph ({month_name_nominative(month_key)})"
        base = TASK_LABELS[task.value]
        if task in vk_tasks and vk_scope and base.endswith(")"):
            return base[:-1] + f" {vk_scope})"
        if task in vk_tasks and vk_scope:
            return f"{base} {vk_scope}"
        return base
    ics_sub: dict[str, dict[str, str]] = {}
    if JobTask.ics_publish in tasks:
        link = event.ics_url
        suffix = f" — {link}" if link else ""
        ics_sub["ics_supabase"] = {"icon": "\U0001f504", "suffix": suffix}
    if JobTask.tg_ics_post in tasks:
        link = event.ics_post_url
        suffix = f" — {link}" if link else ""
        ics_sub["ics_telegram"] = {"icon": "\U0001f504", "suffix": suffix}
    fest_sub: dict[str, dict[str, str]] = {}
    if JobTask.festival_pages in tasks:
        fest_sub["tg"] = {"icon": "\U0001f504", "suffix": ""}
    lines = []
    vk_line_added = False
    for t in tasks:
        info = progress[t]
        if _vk_captcha_needed and vk_present and t in vk_tasks:
            if vk_line_added:
                continue
            label = "VK"
            if vk_scope:
                label += f" {vk_scope}"
            lines.append(f"{info['icon']} {label}{info['suffix']}")
            vk_line_added = True
        elif t == JobTask.ics_publish:
            lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
            if "ics_supabase" in ics_sub:
                lines.append(
                    f"{ics_sub['ics_supabase']['icon']} ICS (Supabase){ics_sub['ics_supabase']['suffix']}"
                )
        elif t == JobTask.tg_ics_post:
            lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
            if "ics_telegram" in ics_sub:
                lines.append(
                    f"{ics_sub['ics_telegram']['icon']} ICS (Telegram){ics_sub['ics_telegram']['suffix']}"
                )
        elif t == JobTask.festival_pages:
            lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
            labels = {
                "tg": "Telegraph (фестиваль)",
                "index": "Все фестивали (Telegraph)",
            }
            for key, sub in fest_sub.items():
                label = labels.get(key, key)
                lines.append(f"{sub['icon']} {label}{sub['suffix']}")
        else:
            lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
    head = "Идёт процесс публикации, ждите"
    text = head if not lines else head + "\n" + "\n".join(lines)
    text += "\n<!-- v0 -->"
    progress_ready = asyncio.Event()
    msg = await bot.send_message(
        chat_id,
        text,
        disable_web_page_preview=True,
        reply_markup=captcha_markup,
    )
    progress_ready.set()

    version = 1

    async def render() -> None:
        nonlocal version
        all_done = all(info["icon"] != "\U0001f504" for info in progress.values())
        if fest_sub:
            all_done = all_done and all(
                info["icon"] != "\U0001f504" for info in fest_sub.values()
            )
        head = "Готово" if all_done else "Идёт процесс публикации, ждите"
        lines: list[str] = []
        vk_line_added = False
        for t, info in progress.items():
            if _vk_captcha_needed and vk_present and t in vk_tasks:
                if vk_line_added:
                    continue
                label = "VK"
                if vk_scope:
                    label += f" {vk_scope}"
                lines.append(f"{info['icon']} {label}{info['suffix']}")
                vk_line_added = True
            elif t == JobTask.ics_publish:
                lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
                sup = ics_sub.get("ics_supabase")
                if sup:
                    lines.append(f"{sup['icon']} ICS (Supabase){sup['suffix']}")
            elif t == JobTask.tg_ics_post:
                lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
                tg = ics_sub.get("ics_telegram")
                if tg:
                    lines.append(f"{tg['icon']} ICS (Telegram){tg['suffix']}")
            elif t == JobTask.festival_pages:
                lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
                labels = {
                    "tg": "Telegraph (фестиваль)",
                    "index": "Все фестивали (Telegraph)",
                }
                for key, sub in fest_sub.items():
                    label = labels.get(key, key)
                    lines.append(f"{sub['icon']} {label}{sub['suffix']}")
            else:
                lines.append(f"{info['icon']} {job_label(t)}{info['suffix']}")
        text = head if not lines else head + "\n" + "\n".join(lines)
        text += f"\n<!-- v{version} -->"
        version += 1
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=text,
            disable_web_page_preview=True,
            reply_markup=captcha_markup if _vk_captcha_needed else None,
        )

    def ics_mark(key: str, status: str, detail: str) -> None:
        if status == "skipped_disabled":
            icon = "⏸"
            suffix = " — отключено"
        elif status.startswith("warn"):
            icon = "⚠️"
            suffix = f" — {detail}" if detail else ""
        else:
            icon = "✅" if status.startswith("done") or status.startswith("skipped") else "❌"
            suffix = f" — {detail}" if detail else ""
        ics_sub[key] = {"icon": icon, "suffix": suffix}
        label = "ICS (Supabase)" if key == "ics_supabase" else "ICS (Telegram)"
        line = f"{icon} {label}{suffix}"
        logline("PROG", event.id, "set", line=f'"{line}"')
        asyncio.create_task(render())

    ics_progress = SimpleNamespace(mark=ics_mark) if ics_sub else None

    def fest_mark(key: str, status: str, detail: str) -> None:
        icon = "✅" if status in {"done", "skipped"} else "❌"
        if status == "done" and detail:
            suffix = f" — {detail}"
        elif status == "skipped":
            suffix = " — без изменений"
        elif detail:
            suffix = f" — {detail}"
        else:
            suffix = ""
        fest_sub[key] = {"icon": icon, "suffix": suffix}
        labels = {"tg": "Telegraph (фестиваль)", "index": "Все фестивали (Telegraph)"}
        label = labels.get(key, key)
        line = f"{icon} {label}{suffix}"
        logline("PROG", event.id, "set", line=f'"{line}"')
        asyncio.create_task(render())

    fest_progress = SimpleNamespace(mark=fest_mark) if fest_sub else None

    async def updater(
        task: JobTask,
        eid: int,
        status: JobStatus,
        changed: bool,
        link: str | None,
        err: str | None,
    ) -> None:
        await progress_ready.wait()
        if task not in progress:
            return
        if status == JobStatus.done:
            if link:
                icon = "✅"
                suffix = f" — {link}"
            elif not changed:
                icon = "⏭"
                suffix = ""
            else:
                icon = "✅"
                suffix = (
                    " — создано/обновлено" if task == JobTask.month_pages else ""
                )
        elif status == JobStatus.paused:
            icon = "⏸"
            suffix = " — требуется капча; нажмите «Ввести код»"
        elif err and "disabled" in err.lower():
            icon = "⏸"
            suffix = f" — {err}" if err else " — отключено"
        elif err and "temporary network error" in err.lower():
            icon = "⚠️"
            suffix = " — временная ошибка сети, будет повтор"
        else:
            icon = "❌"
            if err:
                suffix = f" — {err}"
            else:
                suffix = ""
        progress[task] = {"icon": icon, "suffix": suffix}
        line = f"{icon} {job_label(task)}{suffix}"
        logline("PROG", event.id, "set", line=f'"{line}"')
        await render()
        all_done = all(info["icon"] != "\U0001f504" for info in progress.values())
        if fest_sub:
            all_done = all_done and all(
                info["icon"] != "\U0001f504" for info in fest_sub.values()
            )
        if all_done:
            ctx = _EVENT_PROGRESS.pop(event.id, None)
            if ctx and getattr(ctx, "keys", None):
                for key in ctx.keys:
                    ids = _EVENT_PROGRESS_KEYS.get(key)
                    if ids:
                        ids.discard(event.id)
                        if not ids:
                            _EVENT_PROGRESS_KEYS.pop(key, None)

    _EVENT_PROGRESS[event.id] = SimpleNamespace(
        updater=updater,
        ics_progress=ics_progress,
        fest_progress=fest_progress,
        keys=coalesce_keys,
    )

    deadline = _time.monotonic() + 30
    while True:
        processed = await _run_due_jobs_once(
            db, bot, updater, event.id, ics_progress, fest_progress, None, True
        )
        if processed:
            await asyncio.sleep(0)
            continue
        async with db.get_session() as session:
            next_run = (
                await session.execute(
                    select(func.min(JobOutbox.next_run_at)).where(
                        JobOutbox.event_id == event.id,
                        JobOutbox.status.in_([JobStatus.pending, JobStatus.error]),
                    )
                )
            ).scalar()
        next_run = _ensure_utc(next_run)
        if not next_run:
            break
        wait = (next_run - datetime.now(timezone.utc)).total_seconds()
        if wait <= 0:
            continue
        if _time.monotonic() + wait > deadline:
            break
        await asyncio.sleep(min(wait, 1.0))

    async with db.get_session() as session:
        ev = await session.get(Event, event.id)
    fixed: list[str] = []
    if ev:
        if (
            JobTask.telegraph_build in progress
            and progress[JobTask.telegraph_build]["icon"] == "\U0001f504"
            and ev.telegraph_url
        ):
            progress[JobTask.telegraph_build] = {
                "icon": "✅",
                "suffix": f" — {ev.telegraph_url}",
            }
            line = f"✅ {job_label(JobTask.telegraph_build)} — {ev.telegraph_url}"
            logline("PROG", event.id, "set", line=f'"{line}"')
            fixed.append("telegraph_event")
        if (
            JobTask.vk_sync in progress
            and progress[JobTask.vk_sync]["icon"] == "\U0001f504"
            and ev.source_vk_post_url
        ):
            progress[JobTask.vk_sync] = {
                "icon": "✅",
                "suffix": f" — {ev.source_vk_post_url}",
            }
            line = f"✅ {job_label(JobTask.vk_sync)} — {ev.source_vk_post_url}"
            logline("PROG", event.id, "set", line=f'"{line}"')
            fixed.append("vk_event")
        if JobTask.ics_publish in progress:
            sup = ics_sub.get("ics_supabase")
            if ev.ics_url and sup and sup["icon"] == "\U0001f504":
                ics_sub["ics_supabase"] = {
                    "icon": "✅",
                    "suffix": f" — {ev.ics_url}",
                }
                line = f"✅ ICS (Supabase) — {ev.ics_url}"
                logline("PROG", event.id, "set", line=f'"{line}"')
                fixed.append("ics_supabase")
        if JobTask.tg_ics_post in progress:
            tg = ics_sub.get("ics_telegram")
            if ev.ics_post_url and tg and tg["icon"] == "\U0001f504":
                ics_sub["ics_telegram"] = {
                    "icon": "✅",
                    "suffix": f" — {ev.ics_post_url}",
                }
                line = f"✅ ICS (Telegram) — {ev.ics_post_url}"
                logline("PROG", event.id, "set", line=f'"{line}"')
                fixed.append("ics_telegram")
    if progress:
        await render()
    else:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"Готово\n<!-- v{version} -->",
        )
    if fixed:
        logline("PROG", event.id, "reconcile", fixed=",".join(fixed))
    ctx = _EVENT_PROGRESS.pop(event.id, None)
    if ctx and getattr(ctx, "keys", None):
        for key in ctx.keys:
            ids = _EVENT_PROGRESS_KEYS.get(key)
            if ids:
                ids.discard(event.id)
                if not ids:
                    _EVENT_PROGRESS_KEYS.pop(key, None)


async def job_sync_vk_source_post(event_id: int, db: Database, bot: Bot | None) -> None:
    if vk_group_blocked.get("wall.post", 0.0) > _time.time() and not _vk_user_token():
        raise VKPermissionError(None, "permission error")
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    logging.info(
        "job_sync_vk_source_post: event_id=%s source_post_url=%s is_wall=%s",
        event_id,
        ev.source_post_url if ev else None,
        is_vk_wall_url(ev.source_post_url) if ev else None,
    )
    if not ev or is_vk_wall_url(ev.source_post_url):
        return
    new_hash = content_hash(ev.source_text or "")
    if ev.content_hash == new_hash and ev.source_vk_post_url:
        return
    vk_url = await sync_vk_source_post(ev, ev.source_text, db, bot, ics_url=ev.ics_url)
    partner_user: User | None = None
    event_for_notice: Event | None = None
    async with db.get_session() as session:
        obj = await session.get(Event, event_id)
        if obj:
            if vk_url:
                obj.source_vk_post_url = vk_url
            obj.content_hash = new_hash
            session.add(obj)
            if bot and obj.creator_id:
                partner_user = await session.get(User, obj.creator_id)
            await session.commit()
            event_for_notice = obj
    if vk_url:
        logline("VK", event_id, "event done", url=vk_url)
        if bot and event_for_notice:
            await _send_or_update_partner_admin_notice(
                db, bot, event_for_notice, user=partner_user
            )


@dataclass
class NavUpdateResult:
    changed: bool
    removed_legacy: int
    replaced_markers: bool

    def __bool__(self) -> bool:  # pragma: no cover - simple
        return self.changed


async def update_festival_tg_nav(event_id: int, db: Database, bot: Bot | None) -> NavUpdateResult:
    fid = (-event_id) // FEST_JOB_MULT if event_id < 0 else event_id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
        if not fest or not fest.telegraph_path:
            return NavUpdateResult(False, 0, False)
        token = get_telegraph_token()
        if not token:
            logging.error(
                "Telegraph token unavailable",
                extra={"action": "error", "target": "tg", "fest": fest.name},
            )
            return NavUpdateResult(False, 0, False)
        tg = Telegraph(access_token=token)
        nav_html = await get_setting_value(db, "fest_nav_html")
        if nav_html is None:
            nav_html, _, _ = await build_festivals_nav_block(db)
        path = fest.telegraph_path
        try:
            page = await telegraph_call(tg.get_page, path, return_html=True)
            html_content = page.get("content") or page.get("content_html") or ""
            title = page.get("title") or fest.full_name or fest.name
            m = re.search(r"<!--NAV_HASH:([0-9a-f]+)-->", html_content)
            old_hash = m.group(1) if m else ""
            new_html, changed, removed_blocks, markers_replaced = apply_festival_nav(
                html_content, nav_html
            )
            m2 = re.search(r"<!--NAV_HASH:([0-9a-f]+)-->", new_html)
            new_hash = m2.group(1) if m2 else ""
            extra = {
                "target": "tg",
                "path": path,
                "nav_old": old_hash,
                "nav_new": new_hash,
                "fest": fest.name,
                "removed_legacy_blocks": removed_blocks,
                "legacy_markers_replaced": markers_replaced,
            }
            if changed:
                await telegraph_edit_page(
                    tg,
                    path,
                    title=title,
                    html_content=new_html,
                    caller="festival_build",
                )
                fest.nav_hash = await get_setting_value(db, "fest_nav_hash")
                session.add(fest)
                await session.commit()
                logging.info(
                    "updated festival page %s in Telegraph", fest.name,
                    extra={"action": "edited", **extra},
                )
            else:
                logging.info(
                    "festival page %s navigation unchanged", fest.name,
                    extra={"action": "skipped_nochange", **extra},
                )
            return NavUpdateResult(changed, removed_blocks, markers_replaced)
        except Exception as e:
            logging.error(
                "Failed to update festival page %s: %s", fest.name, e,
                extra={"action": "error", "target": "tg", "path": path, "fest": fest.name},
            )
            raise


async def update_festival_vk_nav(event_id: int, db: Database, bot: Bot | None) -> bool:
    fid = (-event_id) // FEST_JOB_MULT if event_id < 0 else event_id
    async with db.get_session() as session:
        fest = await session.get(Festival, fid)
    if not fest:
        return False
    try:
        res = await sync_festival_vk_post(db, fest.name, bot, nav_only=True)
        return bool(res)
    except Exception as e:
        logging.error(
            "Failed to update festival VK post %s: %s", fest.name, e,
            extra={"action": "error", "target": "vk", "fest": fest.name},
        )
        raise


async def update_all_festival_nav(event_id: int, db: Database, bot: Bot | None) -> bool:
    items = await upcoming_festivals(db)
    nav_hash = await get_setting_value(db, "fest_nav_hash")
    changed_any = False
    errors: list[Exception] = []
    for _, _, fest in items:
        if nav_hash and fest.nav_hash == nav_hash:
            logging.info(
                "festival page %s navigation hash matches, skipping",
                fest.name,
                extra={"action": "skipped_same_hash", "fest": fest.name},
            )
            continue
        eid = -(fest.id * FEST_JOB_MULT)
        try:
            res = await update_festival_tg_nav(eid, db, bot)
            if res.changed:
                changed_any = True
        except Exception as e:  # pragma: no cover - logged in callee
            errors.append(e)
        try:
            if await update_festival_vk_nav(eid, db, bot):
                changed_any = True
        except Exception as e:  # pragma: no cover - logged in callee
            errors.append(e)
    logging.info(
        "fest_nav_update_all finished",
        extra={"action": "done", "changed": changed_any},
    )
    if errors:
        raise errors[0]
    return changed_any


async def festivals_fix_nav(
    db: Database, bot: Bot | None = None
) -> tuple[int, int, int, int]:
    async with db.get_session() as session:
        res = await session.execute(select(Festival))
        fests = res.scalars().all()

    pages = 0
    changed = 0
    duplicates_removed = 0
    legacy_markers = 0

    for fest in fests:
        eid = -(fest.id * FEST_JOB_MULT)
        if fest.telegraph_path:
            pages += 1
            try:
                res = await update_festival_tg_nav(eid, db, bot)
                if res.changed:
                    changed += 1
                    duplicates_removed += res.removed_legacy
                    if res.replaced_markers:
                        legacy_markers += 1
                    logging.info(
                        "fest_nav page_updated",
                        extra={
                            "fest": fest.name,
                            "removed_legacy": res.removed_legacy,
                            "replaced_markers": res.replaced_markers,
                        },
                    )
            except Exception as e:
                logging.error(
                    "festivals_fix_nav telegraph_failed",
                    extra={"path": fest.telegraph_path, "err": str(e), "fest": fest.name},
                )
        try:
            await update_festival_vk_nav(eid, db, bot)
        except Exception:
            pass

    logging.info(
        "festivals_fix_nav nav_done",
        extra={
            "pages": pages,
            "changed": changed,
            "duplicates_removed": duplicates_removed,
            "legacy_markers": legacy_markers,
        },
    )
    return pages, changed, duplicates_removed, legacy_markers


festivals_nav_dedup = festivals_fix_nav
rebuild_festival_pages_nav = festivals_fix_nav


JOB_HANDLERS = {
    "telegraph_build": update_telegraph_event_page,
    "vk_sync": job_sync_vk_source_post,
    "ics_publish": ics_publish,
    "tg_ics_post": tg_ics_post,
    "month_pages": update_month_pages_for,
    "week_pages": update_week_pages_for,
    "weekend_pages": update_weekend_pages_for,
    "festival_pages": update_festival_pages_for_event,
    "fest_nav:update_all": update_all_festival_nav,
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


_TG_TAG_RE = re.compile(r"</?tg-(?:emoji|spoiler)[^>]*?>", re.IGNORECASE)
_ESCAPED_TG_TAG_RE = re.compile(r"&lt;/?tg-(?:emoji|spoiler).*?&gt;", re.IGNORECASE)


def sanitize_telegram_html(html: str) -> str:
    """Remove Telegram-specific HTML wrappers while keeping inner text.

    >>> sanitize_telegram_html("<tg-emoji e=1/>")
    ''
    >>> sanitize_telegram_html("<tg-emoji e=1></tg-emoji>")
    ''
    >>> sanitize_telegram_html("<tg-emoji e=1>➡</tg-emoji>")
    '➡'
    >>> sanitize_telegram_html("&lt;tg-emoji e=1/&gt;")
    ''
    >>> sanitize_telegram_html("&lt;tg-emoji e=1&gt;&lt;/tg-emoji&gt;")
    ''
    >>> sanitize_telegram_html("&lt;tg-emoji e=1&gt;➡&lt;/tg-emoji&gt;")
    '➡'
    """
    raw = len(_TG_TAG_RE.findall(html))
    escaped = len(_ESCAPED_TG_TAG_RE.findall(html))
    if raw or escaped:
        logging.info("telegraph:sanitize tg-tags raw=%d escaped=%d", raw, escaped)
    cleaned = _TG_TAG_RE.sub("", html)
    cleaned = _ESCAPED_TG_TAG_RE.sub("", cleaned)
    return cleaned


@lru_cache(maxsize=8)
def md_to_html(text: str) -> str:
    html_text = simple_md_to_html(text)
    html_text = linkify_for_telegraph(html_text)
    html_text = sanitize_telegram_html(html_text)
    if not re.match(r"^<(?:h\d|p|ul|ol|blockquote|pre|table)", html_text):
        html_text = f"<p>{html_text}</p>"
    # Telegraph API does not allow h1/h2 or Telegram-specific tags
    html_text = re.sub(r"<(\/?)h[12]>", r"<\1h3>", html_text)
    html_text = sanitize_telegram_html(html_text)
    return html_text

_DISALLOWED_TAGS_RE = re.compile(
    r"</?(?:span|div|style|script|tg-spoiler|tg-emoji)[^>]*>", re.IGNORECASE
)


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
    skip_phrases = ["полюбить 39"]

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
        context_before = lower_html[max(0, m.start() - 60) : m.start()]
        if any(phrase in context_before for phrase in skip_phrases):
            continue
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


TELEGRAM_FOLDER_RX = re.compile(r"^https?://t\.me/addlist/[A-Za-z0-9_-]+/?$")


def _strip_qf(u: str) -> str:
    return u.split('#', 1)[0].split('?', 1)[0]


def is_tg_folder_link(u: str) -> bool:
    return bool(TELEGRAM_FOLDER_RX.match(_strip_qf(u)))


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
        now_local = datetime.now(tz)
    else:
        now_local = _ensure_utc(now).astimezone(tz)
    start_local = datetime.combine(
        now_local.date() - timedelta(days=1),
        time(0, 0),
        tz,
    )
    return start_local.astimezone(timezone.utc)


def week_cutoff(tz: timezone, now: datetime | None = None) -> datetime:
    """Return UTC datetime for 7 days ago."""
    if now is None:
        now_utc = datetime.now(tz).astimezone(timezone.utc)
    else:
        now_utc = _ensure_utc(now)
    return now_utc - timedelta(days=7)


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
    added_at = _ensure_utc(e.added_at)
    return added_at >= start


def format_event_md(
    e: Event,
    festival: Festival | None = None,
    *,
    include_ics: bool = True,
) -> str:
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
        cam = "\U0001f4f8" * min(2, max(0, e.photo_count))
        prefix = f"{cam} " if cam else ""
        more_line = f"{prefix}[подробнее]({e.telegraph_url})"
        ics = e.ics_url or e.ics_post_url
        if include_ics and ics:
            more_line += f" \U0001f4c5 [добавить в календарь]({ics})"
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
    partner_creator_ids: Collection[int] | None = None,
    prefer_vk_repost: bool = False,
) -> str:

    prefix = ""
    if highlight:
        prefix += "\U0001f449 "
    if is_recent(e):
        prefix += "\U0001f6a9 "
    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "

    partner_creator_ids = partner_creator_ids or ()
    is_partner_creator = (
        e.creator_id in partner_creator_ids if e.creator_id is not None else False
    )

    vk_link = None
    if (
        prefer_vk_repost
        and not is_partner_creator
        and is_vk_wall_url(e.vk_repost_url)
    ):
        vk_link = e.vk_repost_url
    if not vk_link and is_vk_wall_url(e.source_post_url):
        vk_link = e.source_post_url
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
    formatted_short_ticket = (
        format_vk_short_url(e.vk_ticket_short_url)
        if e.vk_ticket_short_url
        else None
    )
    ticket_link_display = formatted_short_ticket or e.ticket_link
    if e.is_free:
        lines.append("🟡 Бесплатно")
        if e.ticket_link:
            lines.append("по регистрации")
            if show_ticket_link and ticket_link_display:
                lines.append(f"\U0001f39f {ticket_link_display}")
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max} руб."
        else:
            val = e.ticket_price_min if e.ticket_price_min is not None else e.ticket_price_max
            price = f"{val} руб." if val is not None else ""
        if show_ticket_link and ticket_link_display:
            lines.append(f"Билеты в источнике {price}".strip())
            lines.append(f"\U0001f39f {ticket_link_display}")
        else:
            lines.append(f"Билеты {price}".strip())
    elif e.ticket_link:
        lines.append("по регистрации")
        if show_ticket_link and ticket_link_display:
            lines.append(f"\U0001f39f {ticket_link_display}")
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
    partner_creator_ids: Collection[int] | None = None,
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

    partner_creator_ids = partner_creator_ids or ()
    is_partner_creator = e.creator_id in partner_creator_ids if e.creator_id is not None else False
    title = html.escape(e.title)
    link_href: str | None = None
    if is_partner_creator and is_vk_wall_url(e.source_post_url):
        link_href = e.source_post_url
    elif is_vk_wall_url(e.source_post_url):
        link_href = e.telegraph_url or e.source_post_url
    elif is_vk_wall_url(e.source_vk_post_url):
        link_href = e.telegraph_url or e.source_vk_post_url
    elif e.source_post_url:
        link_href = e.source_post_url
    if link_href:
        title = f'<a href="{html.escape(link_href)}">{title}</a>'
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

    ticket_link_display = e.vk_ticket_short_url or e.ticket_link
    if e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link and ticket_link_display:
            txt += f' <a href="{html.escape(ticket_link_display)}">по регистрации</a>'
        lines.append(txt)
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        else:
            price = str(e.ticket_price_min or e.ticket_price_max or "")
        if ticket_link_display:
            lines.append(
                f'<a href="{html.escape(ticket_link_display)}">Билеты в источнике</a> {price}'.strip()
            )
    elif e.ticket_link:
        if ticket_link_display:
            lines.append(
                f'<a href="{html.escape(ticket_link_display)}">по регистрации</a>'
            )
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


def format_event_daily_inline(
    e: Event,
    partner_creator_ids: Collection[int] | None = None,
) -> str:
    """Return a compact single-line HTML representation for daily lists."""

    date_part = ""
    if e.date:
        date_part = e.date.split("..", 1)[0]
    elif e.end_date:
        date_part = e.end_date.split("..", 1)[0]

    formatted_date = ""
    if date_part:
        d = parse_iso_date(date_part)
        if d:
            formatted_date = d.strftime("%d.%m")
        else:
            formatted_date = date_part

    markers: list[str] = []
    if is_recent(e):
        markers.append("\U0001f6a9")
    if e.is_free:
        markers.append("🟡")
    prefix = "".join(f"{m} " for m in markers)

    emoji_part = ""
    if e.emoji and not e.title.strip().startswith(e.emoji):
        emoji_part = f"{e.emoji} "

    partner_creator_ids = partner_creator_ids or ()
    is_partner_creator = (
        e.creator_id in partner_creator_ids if e.creator_id is not None else False
    )
    title = html.escape(e.title)
    link_href: str | None = None
    if is_partner_creator and is_vk_wall_url(e.source_post_url):
        link_href = e.source_post_url
    elif is_vk_wall_url(e.source_post_url):
        link_href = e.telegraph_url or e.source_post_url
    elif is_vk_wall_url(e.source_vk_post_url):
        link_href = e.telegraph_url or e.source_vk_post_url
    elif e.source_post_url:
        link_href = e.source_post_url
    if link_href:
        title = f'<a href="{html.escape(link_href)}">{title}</a>'
    body = f"{prefix}{emoji_part}{title}".strip()
    if formatted_date:
        return f"{formatted_date} {body}".strip()
    return body


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
        cam = "\U0001f4f8" * min(2, max(0, e.photo_count))
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
    url = e.telegraph_url
    if not url and e.telegraph_path:
        url = f"https://telegra.ph/{e.telegraph_path.lstrip('/')}"
    if not url and e.source_post_url:
        url = e.source_post_url
    if url:
        nodes.append({"tag": "a", "attrs": {"href": url}, "children": [title_text]})
    else:
        nodes.append(title_text)
    return nodes


def event_to_nodes(
    e: Event,
    festival: Festival | None = None,
    fest_icon: bool = False,
    log_fest_link: bool = False,
    *,
    show_festival: bool = True,
    include_ics: bool = True,
) -> list[dict]:
    md = format_event_md(
        e,
        festival if show_festival else None,
        include_ics=include_ics,
    )

    lines = md.split("\n")
    body_lines = lines[1:]
    if show_festival and festival and body_lines:
        body_lines = body_lines[1:]
    body_md = "\n".join(body_lines) if body_lines else ""
    from telegraph.utils import html_to_nodes

    nodes = [{"tag": "h4", "children": event_title_nodes(e)}]
    fest = festival if show_festival else None
    if fest is None and show_festival and e.festival:
        fest = getattr(e, "_festival", None)
    if log_fest_link and show_festival and e.festival:
        has_url = bool(getattr(fest, "telegraph_url", None))
        has_path = bool(getattr(fest, "telegraph_path", None))
        href = ""
        if has_url:
            href = fest.telegraph_url
        elif has_path:
            href = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"
        logging.info(
            "month_render_fest_link",
            extra={
                "event_id": e.id,
                "festival": e.festival,
                "has_url": has_url,
                "has_path": has_path,
                "href_used": href,
            },
        )
    if fest:
        prefix = "✨ " if fest_icon else ""
        url = fest.telegraph_url
        if not url and fest.telegraph_path:
            url = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"
        if url:
            children: list = []
            if prefix:
                children.append(prefix)
            children.append(
                {
                    "tag": "a",
                    "attrs": {"href": url},
                    "children": [fest.name],
                }
            )
            nodes.append({"tag": "p", "children": children})
        else:
            text = f"{prefix}{fest.name}" if prefix else fest.name
            nodes.append({"tag": "p", "children": [text]})
    if body_md:
        html_text = md_to_html(body_md)
        body_nodes = html_to_nodes(html_text)
        if (
            festival
            and not show_festival
            and not e.telegraph_url
            and not (e.description and e.description.strip())
            and body_nodes
        ):
            first = body_nodes[0]
            if first.get("tag") == "p":
                children = first.get("children") or []
                if children and isinstance(children[0], dict) and children[0].get("tag") == "br":
                    rest = children[1:]
                    if festival.program_url:
                        link_node = {
                            "tag": "p",
                            "children": [
                                {
                                    "tag": "a",
                                    "attrs": {"href": festival.program_url},
                                    "children": ["программа"],
                                }
                            ],
                        }
                        body_nodes = [link_node, {"tag": "p", "children": rest}]
                    else:
                        body_nodes = [{"tag": "p", "children": rest}]
        nodes.extend(body_nodes)
    nodes.extend(telegraph_br())
    return nodes


def exhibition_title_nodes(e: Event) -> list:
    nodes: list = []
    if is_recent(e):
        nodes.append("\U0001f6a9 ")
    if e.emoji and not e.title.strip().startswith(e.emoji):
        nodes.append(f"{e.emoji} ")
    title_text = e.title
    url = e.telegraph_url
    if not url and e.telegraph_path:
        url = f"https://telegra.ph/{e.telegraph_path.lstrip('/')}"
    if not url and e.source_post_url:
        url = e.source_post_url
    if url:
        nodes.append({"tag": "a", "attrs": {"href": url}, "children": [title_text]})
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
    include_ics: bool = True,
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
        elif d.weekday() == 6:
            add_many([{ "tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"] }])
        add_many([
            {"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]},
            {"tag": "br"},
        ])
        add_many(telegraph_br())
        for ev in events:
            fest = fest_map.get((ev.festival or "").casefold())
            add_many(
                event_to_nodes(
                    ev,
                    fest,
                    fest_icon=True,
                    log_fest_link=use_markers,
                    include_ics=include_ics,
                )
            )
        if use_markers:
            add_many([DAY_END(d)])


def render_month_day_section(d: date, events: list[Event]) -> str:
    """Return HTML snippet for a single day on a month page."""
    from telegraph.utils import nodes_to_html

    nodes: list[dict] = []
    nodes.extend(telegraph_br())
    if d.weekday() == 5:
        nodes.append({"tag": "h3", "children": ["🟥🟥🟥 суббота 🟥🟥🟥"]})
    elif d.weekday() == 6:
        nodes.append({"tag": "h3", "children": ["🟥🟥 воскресенье 🟥🟥"]})
    nodes.append({"tag": "h3", "children": [f"🟥🟥🟥 {format_day_pretty(d)} 🟥🟥🟥"]})
    nodes.extend(telegraph_br())
    for ev in events:
        fest = getattr(ev, "_festival", None)
        nodes.extend(
            event_to_nodes(ev, fest, fest_icon=True, log_fest_link=True)
        )
    return nodes_to_html(nodes)

async def get_month_data(db: Database, month: str, *, fallback: bool = True):
    """Return events and exhibitions for the given month."""
    start = date.fromisoformat(month + "-01")
    next_start = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    today = datetime.now(LOCAL_TZ).date()
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

    if month == today.strftime("%Y-%m"):
        today_str = today.isoformat()
        cutoff = (today - timedelta(days=30)).isoformat()
        events = [e for e in events if e.date.split("..", 1)[0] >= today_str]
        exhibitions = [
            e for e in exhibitions if e.end_date and e.end_date >= today_str
        ]

    if fallback and not events and not exhibitions:
        prev_month = (start - timedelta(days=1)).strftime("%Y-%m")
        if prev_month != month:
            prev_events, prev_exh = await get_month_data(
                db, prev_month, fallback=True
            )
            if not events:
                events.extend(prev_events)
            if not exhibitions:
                exhibitions.extend(prev_exh)

    return events, exhibitions


async def build_month_page_content(
    db: Database,
    month: str,
    events: list[Event] | None = None,
    exhibitions: list[Event] | None = None,
    continuation_url: str | None = None,
    size_limit: int | None = None,
    *,
    include_ics: bool = True,
) -> tuple[str, list, int]:
    if events is None or exhibitions is None:
        events, exhibitions = await get_month_data(db, month)

    async with span("db"):
        async with db.get_session() as session:
            res_f = await session.execute(select(Festival))
            fest_map = {f.name.casefold(): f for f in res_f.scalars().all()}
    for e in events:
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)
    for e in exhibitions:
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)
    fest_index_url = await get_setting_value(db, "fest_index_url")

    async with span("render"):
        title, content, size = await asyncio.to_thread(
            _build_month_page_content_sync,
            month,
            events,
            exhibitions,
            fest_map,
            continuation_url,
            size_limit,
            fest_index_url,
            include_ics,
        )
    logging.info("build_month_page_content size=%d", size)
    return title, content, size


def _build_month_page_content_sync(
    month: str,
    events: list[Event],
    exhibitions: list[Event],
    fest_map: dict[str, Festival],
    continuation_url: str | None,
    size_limit: int | None,
    fest_index_url: str | None,
    include_ics: bool,
) -> tuple[str, list, int]:
    # Ensure festivals have full Telegraph URLs for easy linking
    for fest in fest_map.values():
        if not fest.telegraph_url and fest.telegraph_path:
            fest.telegraph_url = f"https://telegra.ph/{fest.telegraph_path.lstrip('/')}"

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

    add_day_sections(
        sorted(by_day),
        by_day,
        fest_map,
        add_many,
        use_markers=True,
        include_ics=include_ics,
    )

    if exhibitions and not exceeded:
        add_many([PERM_START])
        add({"tag": "h3", "children": ["Постоянные выставки"]})
        add({"tag": "br"})
        add_many(telegraph_br())
        for ev in exhibitions:
            if exceeded:
                break
            add_many(exhibition_to_nodes(ev))
        add_many([PERM_END])


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
        add({"tag": "br"})
        add_many(telegraph_br())

    if fest_index_url and not exceeded:
        add_many(telegraph_br())
        add(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        add_many(telegraph_br())

    title = (
        f"События Калининграда в {month_name_prepositional(month)}: полный анонс от Полюбить Калининград Анонсы"
    )
    return title, content, size


async def _sync_month_page_inner(
    db: Database,
    month: str,
    update_links: bool = False,
    force: bool = False,
    progress: Any | None = None,
) -> bool:
    tasks: list[Awaitable[None]] = []
    async with HEAVY_SEMAPHORE:
        now = _time.time()
        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and not force
            and now < _month_next_run[month]
        ):
            logging.debug("sync_month_page skipped, debounced")
            return False
        _month_next_run[month] = now + 60
        logging.info(
            "sync_month_page start: month=%s update_links=%s force=%s",
            month,
            update_links,
            force,
        )
        if DEBUG:
            mem_info("month page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            raise RuntimeError("Telegraph token unavailable")
        tg = Telegraph(access_token=token)
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
            created = False
            if not page:
                page = MonthPage(month=month, url="", path="")
                session.add(page)
                await session.commit()
                created = True
            prev_hash = page.content_hash
            prev_hash2 = page.content_hash2

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

        if update_links:
            nav_block = await build_month_nav_block(db, month)
            nav_update_failed = False
            for path_attr, hash_attr in (("path", "content_hash"), ("path2", "content_hash2")):
                path = getattr(page, path_attr)
                if not path:
                    continue
                page_data = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page_data.get("content") or page_data.get("content_html") or ""
                html_content = unescape_html_comments(html_content)
                changed_any = False
                if "<!--DAY" not in html_content:
                    logging.warning("month_rebuild_markers_missing")
                    year = int(month.split("-")[0])
                    for m in re.finditer(
                        r"<h3>🟥🟥🟥 (\d{1,2} [^<]+) 🟥🟥🟥</h3>", html_content
                    ):
                        parsed = _parse_pretty_date(m.group(1), year)
                        if parsed:
                            html_content, changed = ensure_day_markers(
                                html_content, parsed
                            )
                            changed_any = changed_any or changed
                updated_html = ensure_footer_nav_with_hr(
                    html_content, nav_block, month=month, page=1 if path_attr == "path" else 2
                )
                if not changed_any and content_hash(updated_html) == content_hash(html_content):
                    continue
                if len(updated_html.encode()) > TELEGRAPH_LIMIT:
                    logging.warning(
                        "Updated navigation for %s (%s) exceeds limit, rebuilding",
                        month,
                        path,
                    )
                    nav_update_failed = True
                    break
                title = page_data.get("title") or month_name_prepositional(month)
                try:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=updated_html,
                        caller="month_build",
                    )
                except TelegraphException as e:
                    msg = str(e).lower()
                    if all(word in msg for word in ("content", "too", "big")):
                        logging.warning(
                            "Updated navigation for %s (%s) too big, rebuilding",
                            month,
                            path,
                        )
                        nav_update_failed = True
                        break
                    raise
                setattr(page, hash_attr, content_hash(updated_html))
            if not nav_update_failed:
                await commit_page()
                return False
            logging.info("Falling back to full rebuild for %s", month)

        events, exhibitions = await get_month_data(db, month)
        nav_block = await build_month_nav_block(db, month)

        from telegraph.utils import nodes_to_html

        title, content, _ = await build_month_page_content(
            db, month, events, exhibitions
        )
        html_full = unescape_html_comments(nodes_to_html(content))
        html_full = ensure_footer_nav_with_hr(html_full, nav_block, month=month, page=1)
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
                            tg,
                            title=title,
                            html_content=html_full,
                            caller="month_build",
                        )
                        page.url = normalize_telegraph_url(data.get("url"))
                        page.path = data.get("path")
                        created = True
                    else:
                        logging.info("updating month page %s", month)
                        start = _time.perf_counter()
                        await telegraph_edit_page(
                            tg,
                            page.path,
                            title=title,
                            html_content=html_full,
                            caller="month_build",
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
                logging.info(
                    "sync_month_page: splitting %s (events=%d)",
                    month,
                    len(events),
                )
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
        except TelegraphException as e:
            msg = str(e).lower()
            if all(word in msg for word in ("content", "too", "big")):
                logging.warning("Month page %s too big, splitting", month)
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
            else:
                logging.error("Failed to sync month page %s: %s", month, e)
                raise
        except Exception as e:
            msg = str(e).lower()
            if all(word in msg for word in ("content", "too", "big")):
                logging.info(
                    "sync_month_page: splitting %s (events=%d)",
                    month,
                    len(events),
                )
                logging.warning("Month page %s too big, splitting", month)
                had_path = bool(page.path)
                await split_month_until_ok(
                    db, tg, page, month, events, exhibitions, nav_block
                )
                if not had_path and page.path:
                    created = True
            else:
                logging.error("Failed to sync month page %s: %s", month, e)

                if progress:
                    progress.mark(f"month_pages:{month}", "error", str(e))
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
        changed = (
            created
            or page.content_hash != prev_hash
            or page.content_hash2 != prev_hash2
        )
        if progress:
            status = "done" if changed else "skipped_nochange"
            progress.mark(f"month_pages:{month}", status, page.url or "")
        paths = ", ".join(p for p in [page.path, page.path2] if p)
        if changed:
            logging.info("month page %s: edited path=%s size=%d", month, paths, size)
        else:
            logging.info("month page %s: nochange", month)
    return True

async def sync_month_page(
    db: Database,
    month: str,
    update_links: bool = False,
    force: bool = False,
    progress: Any | None = None,
):
    async with _page_locks[f"month:{month}"]:
        needs_nav = await _sync_month_page_inner(
            db, month, update_links, force, progress
        )
    if needs_nav:
        await refresh_month_nav(db)


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

    fest_index_url = await get_setting_value(db, "fest_index_url")

    for e in events:
        fest = fest_map.get((e.festival or "").casefold())
        await ensure_event_telegraph_link(e, fest, db)

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
    # NEW: weekend cover
    cover_url = await get_setting_value(db, f"weekend_cover:{start}")
    if cover_url and not exceeded:
        add(
            {
                "tag": "figure",
                "children": [
                    {"tag": "img", "attrs": {"src": cover_url}, "children": []}
                ],
            }
        )
        add_many(telegraph_br())

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

    if fest_index_url and not exceeded:
        add_many(telegraph_br())
        add(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        add_many(telegraph_br())

    label = format_weekend_range(saturday)
    if saturday.month == sunday.month:
        label = f"{saturday.day}-{sunday.day} {MONTHS[saturday.month - 1]}"
    title = (
        "Чем заняться на выходных в Калининградской области "
        f"{label}"
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
    db: Database,
    start: str,
    update_links: bool = True,
    post_vk: bool = True,
    force: bool = False,
    progress: Any | None = None,
):
    tasks: list[Awaitable[None]] = []
    async with HEAVY_SEMAPHORE:
        now = _time.time()
        if (
            "PYTEST_CURRENT_TEST" not in os.environ
            and not force
            and now < _weekend_next_run[start]
        ):
            logging.debug("sync_weekend_page skipped, debounced")
            return
        _weekend_next_run[start] = now + 60
        logging.info(
            "sync_weekend_page start: start=%s update_links=%s post_vk=%s",
            start,
            update_links,
            post_vk,
        )
        if DEBUG:
            mem_info("weekend page before")
        token = get_telegraph_token()
        if not token:
            logging.error("Telegraph token unavailable")
            return
        tg = Telegraph(access_token=token)
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
            prev_hash = page.content_hash

        try:
            title, content, _ = await build_weekend_page_content(db, start)
            html = nodes_to_html(content)
            hash_new = content_hash(html)
            if not path:
                title = re.sub(r"(\d+)-(\d+)", r"\1 - \2", title)
                data = await telegraph_create_page(
                    tg, title, content=content, caller="weekend_build"
                )
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
                if update_links:
                    await telegraph_edit_page(
                        tg,
                        page.path,
                        title=title,
                        content=content,
                        caller="weekend_build",
                    )
            elif page.content_hash == hash_new and not update_links:
                logging.debug("telegraph_update skipped (no changes)")
            else:
                start_t = _time.perf_counter()
                await telegraph_edit_page(
                    tg, path, title=title, content=content, caller="weekend_build"
                )
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
            if progress:
                progress.mark(f"weekend_pages:{start}", "error", str(e))
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

        changed = created or page.content_hash != prev_hash
        if progress:
            status = "done" if changed else "skipped_nochange"
            progress.mark(f"weekend_pages:{start}", status, page.url or "")
        size = len(html.encode())
        if changed:
            logging.info(
                "weekend page %s: edited path=%s size=%d", start, page.path, size
            )
        else:
            logging.info("weekend page %s: nochange", start)

        if update_links or created:
            d_start = date.fromisoformat(start)
            for d_adj in (d_start - timedelta(days=7), d_start + timedelta(days=7)):
                w_key = d_adj.isoformat()
                async with db.get_session() as session:
                    if await session.get(WeekendPage, w_key):
                        tasks.append(
                            sync_weekend_page(db, w_key, update_links=False, post_vk=False)
                        )
    for t in tasks:
        await t


async def sync_weekend_page(
    db: Database,
    start: str,
    update_links: bool = True,
    post_vk: bool = True,
    force: bool = False,
    progress: Any | None = None,
):
    async with _page_locks[f"week:{start}"]:
        await _sync_weekend_page_inner(db, start, update_links, post_vk, force, progress)


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
        now = _time.time()
        if "PYTEST_CURRENT_TEST" not in os.environ and now < _vk_weekend_next_run[start]:
            logging.debug("sync_vk_weekend_post skipped, debounced")
            return
        _vk_weekend_next_run[start] = now + 60
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
        now = _time.time()
        if "PYTEST_CURRENT_TEST" not in os.environ and now < _vk_week_next_run[start]:
            logging.debug("sync_vk_week_post skipped, debounced")
            return
        _vk_week_next_run[start] = now + 60
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


MAX_FEST_DESCRIPTION_LENGTH = 350
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


def _russian_plural(value: int, forms: tuple[str, str, str]) -> str:
    tail = value % 100
    if 10 < tail < 20:
        form = forms[2]
    else:
        tail = value % 10
        if tail == 1:
            form = forms[0]
        elif 1 < tail < 5:
            form = forms[1]
        else:
            form = forms[2]
    return f"{value} {form}"


async def generate_festival_description(
    fest: Festival, events: list[Event]
) -> str | None:
    """Use LLM to craft a short festival blurb."""

    name = fest.full_name or fest.name

    titles: list[str] = []
    seen_titles: set[str] = set()
    venues: list[str] = []
    seen_venues: set[str] = set()
    for event in events:
        title = (event.title or "").strip()
        if title:
            key = title.casefold()
            if key not in seen_titles and len(titles) < 10:
                seen_titles.add(key)
                titles.append(title)
        venue = (event.location_name or "").strip()
        if venue:
            key = venue.casefold()
            if key not in seen_venues and len(venues) < 5:
                seen_venues.add(key)
                venues.append(venue)

    start, end = festival_date_range(events)
    date_clause = ""
    if start and end:
        if start == end:
            date_clause = format_day_pretty(start)
        else:
            date_clause = (
                f"с {format_day_pretty(start)} по {format_day_pretty(end)}"
            )
    elif start:
        date_clause = format_day_pretty(start)

    city_values: list[str] = []
    if fest.city:
        city_values.append(fest.city)
    for event in events:
        if event.city:
            city_values.append(event.city)
    city_clause = ""
    city_counter = Counter(c.strip() for c in city_values if c and c.strip())
    if city_counter:
        most_common_city, _ = city_counter.most_common(1)[0]
        if most_common_city:
            city_clause = most_common_city

    duration_days = 0
    if start and end:
        duration_days = (end - start).days + 1

    event_count = len(events)

    fact_parts: list[str] = []
    if date_clause:
        fact_parts.append(f"период — {date_clause}")
    if city_clause:
        fact_parts.append(f"город — {city_clause}")
    if duration_days > 1:
        fact_parts.append(
            f"продолжительность — {_russian_plural(duration_days, ('день', 'дня', 'дней'))}"
        )
    if event_count:
        fact_parts.append(
            f"в программе {_russian_plural(event_count, ('событие', 'события', 'событий'))}"
        )
    if titles:
        fact_parts.append(f"сюжеты: {', '.join(titles)}")

    if not fact_parts:
        logging.warning(
            "generate_festival_description: insufficient data for %s", fest.name
        )
        return None

    context_sources: list[str] = []
    for candidate in (fest.source_text, fest.description):
        if candidate:
            context_sources.append(candidate)
    for event in events[:5]:
        if event.source_text:
            context_sources.append(event.source_text)

    context_snippet = ""
    for raw in context_sources:
        snippet = " ".join(raw.split()).strip()
        if snippet:
            context_snippet = snippet[:200]
            break

    facts_sentence = f"Исходные факты: {', '.join(fact_parts)}."
    third_segments: list[str] = []
    if venues:
        third_segments.append(f"площадки: {', '.join(venues)}")
    if context_snippet:
        third_segments.append(f"контекст: {context_snippet}")
    third_segments.append(
        "итоговый текст до 350 знаков, один абзац без списков и эмодзи"
    )
    third_sentence = "; ".join(third_segments) + "."

    prompt_sentences = [
        (
            "Ты — культурный журналист: напиши лаконичное описание фестиваля "
            f"{name} без выдуманных фактов."
        ),
        facts_sentence,
        third_sentence,
    ]
    prompt = " ".join(prompt_sentences)

    try:
        text = await ask_4o(prompt)
        logging.info("generated description for festival %s", fest.name)
    except Exception as e:
        logging.error("failed to generate festival description %s: %s", fest.name, e)
        return None

    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_FEST_DESCRIPTION_LENGTH:
        logging.warning(
            "festival description too long for %s: %d", fest.name, len(cleaned)
        )
        return None
    if _EMOJI_RE.search(cleaned):
        logging.warning(
            "festival description contains emoji for %s", fest.name
        )
        return None
    return cleaned


async def regenerate_festival_description(
    db: Database, fest_ref: Festival | int
) -> str | None:
    """Regenerate festival description using latest events."""

    async with db.get_session() as session:
        fest_obj: Festival | None = None
        if isinstance(fest_ref, Festival):
            if fest_ref.id is not None:
                fest_obj = await session.get(Festival, fest_ref.id)
            else:
                result = await session.execute(
                    select(Festival).where(Festival.name == fest_ref.name)
                )
                fest_obj = result.scalar_one_or_none()
        else:
            fest_obj = await session.get(Festival, fest_ref)

        if not fest_obj:
            return None

        events_query = (
            select(Event)
            .where(Event.festival == fest_obj.name)
            .order_by(Event.date, Event.time)
        )
        events_res = await session.execute(events_query)
        events = list(events_res.scalars().all())

        return await generate_festival_description(fest_obj, events)


async def merge_festivals(
    db: Database, src_id: int, dst_id: int, bot: Bot | None = None
) -> bool:
    """Merge two festivals moving events, media and metadata."""

    if src_id == dst_id:
        logging.warning("merge_festivals: identical ids src=%s dst=%s", src_id, dst_id)
        return False

    moved_events_count = 0
    aliases_added = 0
    photos_added = 0
    description_updated = False

    async with db.get_session() as session:
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

        if not src or not dst:
            logging.error(
                "merge_festivals: missing festivals src=%s dst=%s", src_id, dst_id
            )
            return False

        src_name = src.name
        dst_name = dst.name
        dst_pk = dst.id

        events_to_move_res = await session.execute(
            select(Event.id).where(Event.festival == src_name)
        )
        moved_events_count = len(list(events_to_move_res.scalars()))

        dst_photos_before = {url for url in list(dst.photo_urls or []) if url}
        dst_cover_before = dst.photo_url

        await session.execute(
            update(Event).where(Event.festival == src_name).values(festival=dst_name)
        )

        merged_photos: list[str] = []
        for url in list(dst.photo_urls or []) + list(src.photo_urls or []):
            if url and url not in merged_photos:
                merged_photos.append(url)
        if merged_photos:
            dst.photo_urls = merged_photos
            if not dst.photo_url or dst.photo_url not in merged_photos:
                dst.photo_url = merged_photos[0]
        elif src.photo_url and not dst.photo_url:
            dst.photo_url = src.photo_url

        if merged_photos:
            photos_added = sum(
                1 for url in merged_photos if url and url not in dst_photos_before
            )
        elif (
            dst.photo_url
            and dst.photo_url != dst_cover_before
            and dst.photo_url not in dst_photos_before
        ):
            photos_added = 1

        def _fill(field: str) -> None:
            dst_val = getattr(dst, field)
            src_val = getattr(src, field)
            if (dst_val is None or dst_val == "") and src_val:
                setattr(dst, field, src_val)

        existing_aliases = {
            normalized
            for alias in list(dst.aliases or [])
            if (normalized := normalize_alias(alias))
        }
        for field in (
            "full_name",
            "description",
            "website_url",
            "program_url",
            "vk_url",
            "tg_url",
            "ticket_url",
            "location_name",
            "location_address",
            "city",
            "source_text",
            "source_post_url",
            "source_chat_id",
            "source_message_id",
        ):
            _fill(field)

        skip_keys = {normalize_alias(dst_name)}
        if dst.full_name:
            dst_full_norm = normalize_alias(dst.full_name)
            if dst_full_norm:
                skip_keys.add(dst_full_norm)

        seen_aliases: set[str] = set()
        merged_aliases: list[str] = []

        def add_alias(raw: str | None) -> None:
            normalized = normalize_alias(raw)
            if not normalized or normalized in skip_keys or normalized in seen_aliases:
                return
            if len(merged_aliases) >= 8:
                return
            seen_aliases.add(normalized)
            merged_aliases.append(normalized)

        for alias in list(dst.aliases or []) + list(src.aliases or []):
            add_alias(alias)

        add_alias(src_name)
        add_alias(src.full_name)

        dst.aliases = merged_aliases
        aliases_added = sum(1 for alias in merged_aliases if alias not in existing_aliases)

        events_res = await session.execute(
            select(Event).where(Event.festival == dst_name)
        )
        all_events = list(events_res.scalars().all())
        start, end = festival_date_range(all_events)
        if not start and src.start_date:
            start = parse_iso_date(src.start_date)
        if not end and src.end_date:
            end = parse_iso_date(src.end_date)
        dst.start_date = start.isoformat() if start else None
        dst.end_date = end.isoformat() if end else None

        await session.delete(src)
        await session.commit()

        fest_ref: Festival | int
        if dst_pk is not None:
            fest_ref = dst_pk
        else:
            fest_ref = dst
        new_description = await regenerate_festival_description(db, fest_ref)
        if new_description and new_description != (dst.description or ""):
            dst.description = new_description
            await session.commit()
            description_updated = True

    await sync_festival_page(db, dst_name)
    await rebuild_fest_nav_if_changed(db)
    await sync_festival_vk_post(db, dst_name, bot)
    logging.info(
        "merge_festivals: merged src=%s dst=%s events_moved=%s aliases_added=%s photos_added=%s description_updated=%s",
        src_id,
        dst_id,
        moved_events_count,
        aliases_added,
        photos_added,
        description_updated,
    )
    return True


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
        today = date.today().isoformat()
        res = await session.execute(
            select(Event)
            .where(Event.festival == fest.name, Event.date >= today)
            .order_by(Event.date, Event.time)
        )
        events = res.scalars().all()

        logging.info("festival %s has %d events", fest.name, len(events))

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

    nodes: list[dict] = []
    photo_urls = list(fest.photo_urls or [])
    cover = fest.photo_url or (photo_urls[0] if photo_urls else None)
    gallery_photos = [url for url in photo_urls if url != cover]
    if cover:
        nodes.append(
            {
                "tag": "figure",
                "children": [{"tag": "img", "attrs": {"src": cover}}],
            }
        )
        nodes.append({"tag": "p", "children": ["\u00a0"]})
    if fest.program_url:
        nodes.append({"tag": "h2", "children": ["ПРОГРАММА"]})
        links = [
            {
                "tag": "p",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest.program_url},
                        "children": ["Смотреть программу"],
                    }
                ],
            }
        ]
        if fest.website_url:
            links.append(
                {
                    "tag": "p",
                    "children": [
                        {
                            "tag": "a",
                            "attrs": {"href": fest.website_url},
                            "children": ["Сайт"],
                        }
                    ],
                }
            )
        nodes.extend(links)
        nodes.extend(telegraph_br())
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
            nodes.extend(event_to_nodes(e, festival=fest, show_festival=False))
    else:
        nodes.extend(telegraph_br())
        nodes.extend(telegraph_br())
        nodes.append({"tag": "p", "children": ["Расписание скоро обновим"]})
    if gallery_photos:
        nodes.extend(telegraph_br())
        for url in gallery_photos:
            nodes.append({"tag": "img", "attrs": {"src": url}})
            nodes.append({"tag": "p", "children": ["\u00a0"]})
    nav_nodes, _ = await _build_festival_nav_block(db, exclude=fest.name)
    if nav_nodes:
        from telegraph.utils import nodes_to_html, html_to_nodes
        nav_html = nodes_to_html(nav_nodes)
        nav_block = f"{FEST_NAV_START}{nav_html}{FEST_NAV_END}"
        nodes.extend(html_to_nodes(nav_block))
    fest_index_url = await get_setting_value(db, "fest_index_url")
    if fest_index_url:
        logging.info(
            "festival_page_index_link",
            extra={"festival": fest.name, "fest_index_url": fest_index_url},
        )
        nodes.append(
            {
                "tag": "h3",
                "children": [
                    {
                        "tag": "a",
                        "attrs": {"href": fest_index_url},
                        "children": ["Фестивали"],
                    }
                ],
            }
        )
        nodes.extend(telegraph_br())
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
            return None
        tg = Telegraph(access_token=token)
        async with db.get_session() as session:
            result = await session.execute(
                select(Festival).where(Festival.name == name)
            )
            fest = result.scalar_one_or_none()
            if not fest:
                return None
            title = fest.full_name or fest.name
            path = fest.telegraph_path
            url = fest.telegraph_url

        changed = False
        try:
            created = False
            if refresh_nav_only and path:
                nav_html, _, _ = await build_festivals_nav_block(db)
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                new_html, changed, removed_blocks, markers_replaced = apply_festival_nav(
                    html_content, nav_html
                )
                extra = {
                    "target": "tg",
                    "path": path,
                    "removed_legacy_blocks": removed_blocks,
                    "legacy_markers_replaced": markers_replaced,
                }
                if changed:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=new_html,
                        caller="festival_build",
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"action": "edited", **extra},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"action": "skipped_nochange", **extra},
                    )
            else:
                title, content = await build_festival_page_content(db, fest)
                path = fest.telegraph_path
                url = fest.telegraph_url
                if path:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        content=content,
                        caller="festival_build",
                    )
                    changed = True
                    logging.info("updated festival page %s in Telegraph", name)
                else:
                    data = await telegraph_create_page(
                        tg, title, content=content, caller="festival_build"
                    )
                    url = normalize_telegraph_url(data.get("url"))
                    path = data.get("path")
                    created = True
                    changed = True
                    logging.info("created festival page %s: %s", name, url)
        except Exception as e:
            logging.error("Failed to sync festival %s: %s", name, e)
            return None

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
        return url if changed else None


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
    tg = Telegraph(access_token=token) if token else None
    for fid, name, path, vk_url in fests:
        if tg and path:
            try:
                page = await telegraph_call(tg.get_page, path, return_html=True)
                html_content = page.get("content") or page.get("content_html") or ""
                title = page.get("title") or name
                new_html, changed, removed_blocks, markers_replaced = apply_festival_nav(
                    html_content, nav_html
                )
                extra = {
                    "target": "tg",
                    "path": path,
                    "fest": name,
                    "removed_legacy_blocks": removed_blocks,
                    "legacy_markers_replaced": markers_replaced,
                }
                if changed:
                    await telegraph_edit_page(
                        tg,
                        path,
                        title=title,
                        html_content=new_html,
                        caller="festival_build",
                    )
                    logging.info(
                        "updated festival page %s in Telegraph", name,
                        extra={"action": "edited", **extra},
                    )
                else:
                    logging.info(
                        "festival page %s navigation unchanged", name,
                        extra={"action": "skipped_nochange", **extra},
                    )
            except Exception as e:
                logging.error(
                    "Failed to update festival page %s: %s", name, e,
                    extra={"action": "error", "target": "tg", "path": path, "fest": name},
                )
        if vk_url:
            await sync_festival_vk_post(
                db, name, bot, nav_only=True, nav_lines=nav_lines
            )


async def build_festival_vk_message(db: Database, fest: Festival) -> str:
    async with db.get_session() as session:
        today = date.today().isoformat()
        res = await session.execute(
            select(Event)
            .where(Event.festival == fest.name, Event.date >= today)
            .order_by(Event.date, Event.time)
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
    if fest.program_url:
        lines.append(f"программа: {fest.program_url}")
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
    if events:
        for ev in events:
            lines.append(VK_BLANK_LINE)
            lines.append(format_event_vk(ev))
    else:
        lines.append(VK_BLANK_LINE)
        lines.append("Расписание скоро обновим")
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
    strict: bool = False,
) -> bool | None:
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
        user_token = _vk_user_token()
        if not user_token:
            logging.error(
                "VK_USER_TOKEN missing",
                extra={
                    "action": "error",
                    "target": "vk",
                    "url": fest.vk_post_url,
                    "fest": name,
                },
            )
            return None
        for attempt in range(1, 4):
            try:
                await edit_vk_post(fest.vk_post_url, message, db, bot, attachments)
                return True
            except VKAPIError as e:
                logging.warning(
                    "Ошибка VK при редактировании (попытка %d из 3, код %s): %s actor=%s token=%s",
                    attempt,
                    e.code,
                    e.message,
                    e.actor,
                    e.token,
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
                    "Ошибка VK при публикации (попытка %d из 3, код %s): %s actor=%s token=%s",
                    attempt,
                    e.code,
                    e.message,
                    e.actor,
                    e.token,
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
                logging.error(
                    "invalid VK post url %s",
                    fest.vk_post_url,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
                )
                return
            owner_id, post_id = ids
            try:
                response = await vk_api(
                    "wall.getById", posts=f"{owner_id}_{post_id}"
                )
                if isinstance(response, dict):
                    items = response.get("response") or (
                        response["response"] if "response" in response else response
                    )
                else:
                    items = response or []
                if not isinstance(items, list):
                    items = [items] if items else []
                text = items[0].get("text", "") if items else ""
            except Exception as e:
                logging.error(
                    "Не удалось получить пост VK для %s: %s",
                    name,
                    e,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
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
            if message == text:
                logging.info(
                    "festival post %s navigation unchanged", name,
                    extra={
                        "action": "skipped_nochange",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return False
            res_edit = await _try_edit(message, None)
            if res_edit is True:
                logging.info(
                    "updated festival post %s on VK", name,
                    extra={
                        "action": "edited",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return True
            if res_edit is None:
                logging.error(
                    "VK post error for festival %s", name,
                    extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
                )
                if strict:
                    raise RuntimeError("vk edit failed")
                return False
            if os.getenv("VK_NAV_FALLBACK") == "skip":
                logging.info(
                    "festival post %s skipping VK edit", name,
                    extra={
                        "action": "vk_nav_skip_edit",
                        "target": "vk",
                        "url": fest.vk_post_url,
                        "fest": name,
                    },
                )
                return False
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
            logging.info(
                "updated festival post %s on VK", name,
                extra={
                    "action": "edited",
                    "target": "vk",
                    "url": fest.vk_post_url,
                    "fest": name,
                },
            )
            return True
        if res_edit is None:
            logging.error(
                "VK post error for festival %s", name,
                extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
            )
            if strict:
                raise RuntimeError("vk edit failed")
            return False

    url = await _try_post(message, attachments)
    if url:
        async with db.get_session() as session:
            fest_db = (
                await session.execute(select(Festival).where(Festival.name == name))
            ).scalar_one()
            fest_db.vk_post_url = url
            await session.commit()
        logging.info(
            "created festival post %s: %s", name, url,
            extra={"action": "created", "target": "vk", "url": url, "fest": name},
        )
        return True
    logging.error(
        "VK post error for festival %s", name,
        extra={"action": "error", "target": "vk", "url": fest.vk_post_url, "fest": name},
    )
    if strict:
        raise RuntimeError("vk post failed")
    return False


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
    recent_festival_entries: list[str] = []
    partner_creator_ids: set[int] = set()
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
        festivals = res_fests.scalars().all()
        fest_map = {f.name: f for f in festivals}
        recent_festivals: list[tuple[datetime, str]] = []
        for fest in festivals:
            url = _festival_telegraph_url(fest)
            if not url:
                continue
            created_at = _ensure_utc(getattr(fest, "created_at", None))
            if created_at and created_at >= yesterday_utc:
                recent_festivals.append(
                    (
                        created_at,
                        f'<a href="{url}">✨ {html.escape(fest.name)}</a>',
                    )
                )
        recent_festivals.sort(key=lambda item: item[0])
        recent_festival_entries = [entry for _, entry in recent_festivals]

        creator_ids = {
            e.creator_id
            for e in (*events_today, *events_new, *new_events)
            if e.creator_id is not None
        }
        if creator_ids:
            res_partners = await session.execute(
                select(User.user_id).where(
                    User.user_id.in_(creator_ids),
                    User.is_partner.is_(True),
                )
            )
            partner_creator_ids = set(res_partners.scalars().all())

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

    processed_short_ids: set[int] = set()
    for candidate in (*events_today, *events_new):
        if (
            candidate.ticket_link
            and candidate.id is not None
            and candidate.id not in processed_short_ids
            and not (candidate.vk_ticket_short_url and candidate.vk_ticket_short_key)
        ):
            await ensure_vk_short_ticket_link(
                candidate,
                db,
                vk_api_fn=_vk_api,
            )
            processed_short_ids.add(candidate.id)

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
                partner_creator_ids=partner_creator_ids,
            )
        )
    lines1.append("")
    lines1.append(
        f"#Афиша_Калининград #Калининград #концерт #{tag} #{today.day}_{MONTHS[today.month - 1]}"
    )
    section1 = "\n".join(lines1)

    lines2 = [f"<b><i>+{len(events_new)} ДОБАВИЛИ В АНОНС</i></b>"]
    if len(events_new) > 9:
        grouped: dict[str, list[Event]] = {}
        for e in events_new:
            raw_city = (e.city or "Калининград").strip()
            city = raw_city or "Калининград"
            grouped.setdefault(city, []).append(e)
        for city, events in grouped.items():
            lines2.append("")
            lines2.append(html.escape(city.upper()))
            for e in events:
                lines2.append(
                    format_event_daily_inline(
                        e,
                        partner_creator_ids=partner_creator_ids,
                    )
                )
        lines2.append("")
        lines2.append("ℹ️ Нажмите на название мероприятия, чтобы открыть подробности")
    else:
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
                    partner_creator_ids=partner_creator_ids,
                )
            )
    if recent_festival_entries:
        lines2.append("")
        lines2.append("ФЕСТИВАЛИ")
        lines2.append(" ".join(recent_festival_entries))
    section2 = "\n".join(lines2)

    fest_index_url = await get_setting_value(db, "fest_index_url")

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
    if fest_index_url:
        buttons.append(
            types.InlineKeyboardButton(text="Фестивали", url=fest_index_url)
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
    from models import User

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

        creator_ids = {
            e.creator_id
            for e in (*events_today, *events_new, *new_events)
            if e.creator_id is not None
        }
        partner_creator_ids: set[int] = set()
        if creator_ids:
            res_partners = await session.execute(
                select(User.user_id).where(
                    User.user_id.in_(creator_ids),
                    User.is_partner.is_(True),
                )
            )
            partner_creator_ids = set(res_partners.scalars().all())

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

    processed_short_ids: set[int] = set()
    for candidate in (*events_today, *events_new):
        if (
            candidate.ticket_link
            and candidate.id is not None
            and candidate.id not in processed_short_ids
            and not (
                candidate.vk_ticket_short_url and candidate.vk_ticket_short_key
            )
        ):
            await ensure_vk_short_ticket_link(
                candidate,
                db,
                vk_api_fn=_vk_api,
            )
            processed_short_ids.add(candidate.id)

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
                partner_creator_ids=partner_creator_ids,
                prefer_vk_repost=True,
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
                partner_creator_ids=partner_creator_ids,
                prefer_vk_repost=True,
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
) -> str | None:
    if not group_id:
        return None
    logging.info(
        "post_to_vk start: group=%s len=%d attachments=%d",
        group_id,
        len(message),
        len(attachments or []),
    )
    owner_id = -int(group_id.lstrip("-"))
    params_base = {"owner_id": f"-{group_id.lstrip('-')}", "message": message}
    if attachments:
        params_base["attachments"] = ",".join(attachments)
    actors = choose_vk_actor(owner_id, "wall.post")
    if not actors:
        raise VKAPIError(None, "VK token missing", method="wall.post")
    for idx, actor in enumerate(actors, start=1):
        params = params_base.copy()
        if actor.kind == "user" and owner_id < 0:
            params["from_group"] = 1
        logging.info(
            "vk.call method=wall.post owner_id=%s try=%d/%d actor=%s",
            owner_id,
            idx,
            len(actors),
            actor.label,
        )
        token = actor.token if actor.kind == "group" else VK_USER_TOKEN
        try:
            if DEBUG:
                mem_info("VK post before")
            data = await _vk_api(
                "wall.post",
                params,
                db,
                bot,
                token=token,
                token_kind=actor.kind,
                skip_captcha=(actor.kind == "group"),
            )
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
            err_code = (
                data.get("error", {}).get("error_code") if isinstance(data, dict) else None
            )
            logging.error(
                "post_to_vk fail group=%s code=%s len=%d attachments=%d",
                group_id,
                err_code,
                len(message),
                len(attachments or []),
            )
            return None
        except VKAPIError as e:
            logging.warning(
                "post_to_vk error code=%s msg=%s actor=%s token=%s",
                e.code,
                e.message,
                e.actor,
                e.token,
            )
            msg_l = (e.message or "").lower()
            perm = (
                e.code in VK_FALLBACK_CODES
                or "method is unavailable with group auth" in msg_l
                or "access denied" in msg_l
            )
            if idx < len(actors) and perm:
                logging.info(
                    "vk.retry reason=%s actor_next=%s",
                    e.code or e.message,
                    actors[idx].label,
                )
                continue
            raise
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
    # ensure non-empty message to satisfy VK API
    return await post_to_vk(group_id, question or "?", db, bot, [attachment])



def _vk_owner_and_post_id(url: str) -> tuple[str, str] | None:
    m = re.search(r"wall(-?\d+)_(\d+)", url)
    if not m:
        return None
    return m.group(1), m.group(2)




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

    ticket_link_display = (
        format_vk_short_url(event.vk_ticket_short_url)
        if event.vk_ticket_short_url
        else event.ticket_link
    )

    if event.is_free:
        lines.append("🟡 Бесплатно")
        if event.ticket_link:
            lines.append(f"\U0001f39f по регистрации {ticket_link_display}")
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
        lines.append(f"\U0001f39f {info} {ticket_link_display}".strip())
    elif event.ticket_link:
        lines.append(f"\U0001f39f по регистрации {ticket_link_display}")
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
    calendar_url: str | None = None,
) -> str:
    """Build detailed VK post for an event including original source text."""

    text = sanitize_for_vk(text)
    lines = build_vk_source_header(event, festival)
    lines.extend(text.strip().splitlines())
    lines.append(VK_BLANK_LINE)
    if calendar_url:
        lines.append(f"Добавить в календарь {calendar_url}")
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

    attachments: list[str] | None = None
    if VK_PHOTOS_ENABLED and event.photo_urls:
        token = VK_TOKEN_AFISHA or VK_TOKEN
        if token:
            ids: list[str] = []
            for url in event.photo_urls[:VK_MAX_ATTACHMENTS]:
                photo_id = await upload_vk_photo(
                    VK_AFISHA_GROUP_ID, url, db, bot, token=token, token_kind="group"
                )
                if photo_id:
                    ids.append(photo_id)
                elif not VK_USER_TOKEN:
                    logging.info(
                        "VK photo upload skipped: user token required",
                        extra={"eid": event.id},
                    )
                    break
            if ids:
                attachments = ids
        else:
            logging.info("VK photo upload skipped: no group token")

    calendar_line_value: str | None = None
    previous_ics_url = event.ics_url
    calendar_source_url = event.ics_url if ics_url is None else ics_url
    if calendar_source_url != previous_ics_url or calendar_source_url is None:
        if event.vk_ics_short_url or event.vk_ics_short_key:
            event.vk_ics_short_url = None
            event.vk_ics_short_key = None
    event.ics_url = calendar_source_url
    if calendar_source_url:
        short_ics = await ensure_vk_short_ics_link(
            event,
            db,
            bot=bot,
            vk_api_fn=_vk_api,
        )
        if short_ics:
            calendar_line_value = format_vk_short_url(short_ics[0])
        else:
            calendar_line_value = calendar_source_url

    if event.source_vk_post_url:
        await ensure_vk_short_ticket_link(
            event, db, bot=bot, vk_api_fn=_vk_api
        )
        existing = ""
        try:
            ids = _vk_owner_and_post_id(event.source_vk_post_url)
            if ids:
                response = await vk_api(
                    "wall.getById", posts=f"{ids[0]}_{ids[1]}"
                )
                if isinstance(response, dict):
                    items = response.get("response") or (
                        response["response"] if "response" in response else response
                    )
                else:
                    items = response or []
                if not isinstance(items, list):
                    items = [items] if items else []
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

        text_clean = sanitize_for_vk(text).strip()
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
        if calendar_line_value:
            new_lines.append(f"Добавить в календарь {calendar_line_value}")
        new_lines.append(VK_SOURCE_FOOTER)
        new_message = "\n".join(new_lines)
        await edit_vk_post(
            event.source_vk_post_url,
            new_message,
            db,
            bot,
            attachments,
        )
        url = event.source_vk_post_url
        logging.info("sync_vk_source_post updated %s", url)
    else:
        _short_link_result = await ensure_vk_short_ticket_link(
            event, db, vk_api_fn=_vk_api, bot=bot
        )
        message = build_vk_source_message(
            event, text, festival=festival, calendar_url=calendar_line_value
        )
        url = await post_to_vk(
            VK_AFISHA_GROUP_ID,
            message,
            db,
            bot,
            attachments,
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
    owner_id_num: int | None = None
    try:
        owner_id_num = int(owner_id)
    except (TypeError, ValueError):
        owner_id_num = None

    def normalize_group_id(group_id: str | None) -> int | None:
        if not group_id:
            return None
        try:
            value = int(group_id)
        except (TypeError, ValueError):
            return None
        return -abs(value) if value else value

    main_owner_id = normalize_group_id(VK_MAIN_GROUP_ID)
    afisha_owner_id = normalize_group_id(VK_AFISHA_GROUP_ID)

    use_internal_api = False
    edit_token: str | None = None
    edit_token_kind = "group"

    if owner_id_num is not None:
        if owner_id_num == main_owner_id:
            use_internal_api = True
            if VK_TOKEN:
                edit_token = VK_TOKEN
        elif owner_id_num == afisha_owner_id:
            use_internal_api = True
            if VK_TOKEN_AFISHA:
                edit_token = VK_TOKEN_AFISHA

    if edit_token is None:
        edit_token = _vk_user_token()
        edit_token_kind = "user"
    else:
        edit_token_kind = "group"
    current: list[str] = []
    post_text = ""
    old_attachments: list[str] = []
    edit_allowed = True
    edit_block_reason: str | None = None
    try:
        if use_internal_api:
            response = await _vk_api(
                "wall.getById",
                {"posts": f"{owner_id}_{post_id}"},
                db,
                bot,
                token=edit_token,
                token_kind=edit_token_kind,
                skip_captcha=True,
            )
        else:
            response = await vk_api("wall.getById", posts=f"{owner_id}_{post_id}")
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
        if items:
            post = items[0]
            post_text = post.get("text") or ""
            can_edit_flag = post.get("can_edit")
            if can_edit_flag is not None and not bool(can_edit_flag):
                edit_allowed = False
                edit_block_reason = "can_edit=0"
            else:
                ts = post.get("date")
                if ts is not None:
                    try:
                        post_dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                    except (ValueError, OSError, OverflowError):
                        pass
                    else:
                        if datetime.now(timezone.utc) - post_dt > VK_POST_MAX_EDIT_AGE:
                            edit_allowed = False
                            edit_block_reason = "post too old"
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
    if attachments is not None:
        current = attachments[:]
    else:
        current = old_attachments.copy()
    if not edit_allowed:
        logging.warning(
            "edit_vk_post: skipping %s, edit unavailable (%s)",
            post_url,
            edit_block_reason or "unknown reason",
        )
        if db is not None and bot is not None:
            try:
                await notify_superadmin(
                    db,
                    bot,
                    f"Не удалось отредактировать пост {post_url}: окно редактирования истекло",
                )
            except Exception:  # pragma: no cover - best effort
                logging.exception("edit_vk_post notify_superadmin failed")
        return False
    if post_text == message and current == old_attachments:
        logging.info("edit_vk_post: no changes for %s", post_url)
        return False
    if attachments is not None:
        params["attachments"] = ",".join(current) if current else ""
    elif current:
        params["attachments"] = ",".join(current)
    if not edit_token:
        raise VKAPIError(None, "VK_USER_TOKEN missing", method="wall.edit")
    await _vk_api(
        "wall.edit",
        params,
        db,
        bot,
        token=edit_token,
        token_kind=edit_token_kind,
    )
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


async def vk_crawl_cron(db: Database, bot: Bot, run_id: str | None = None) -> None:
    """Scheduled VK crawl according to ``VK_CRAWL_TIMES_LOCAL``."""
    now = datetime.now(LOCAL_TZ).strftime("%H:%M")
    logging.info("vk.crawl.cron.fire time=%s", now)
    delay = max(0, random.uniform(-VK_CRAWL_JITTER_SEC, VK_CRAWL_JITTER_SEC))
    if delay:
        await asyncio.sleep(delay)
    try:
        await vk_intake.crawl_once(db, broadcast=True, bot=bot)
    except Exception:
        logging.exception("vk.crawl.cron.error")


async def cleanup_old_events(db: Database, now_utc: datetime | None = None) -> int:
    """Delete events that finished more than a week ago."""
    cutoff = (now_utc or datetime.now(timezone.utc)) - timedelta(days=7)
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

async def rebuild_pages(
    db: Database,
    months: list[str],
    weekends: list[str],
    *,
    force: bool = False,
) -> dict[str, dict[str, dict[str, list[str] | str]]]:
    logging.info(
        "pages_rebuild start months=%s weekends=%s force=%s",
        months,
        weekends,
        force,
    )
    months_updated: dict[str, list[str]] = {}
    weekends_updated: dict[str, list[str]] = {}
    months_failed: dict[str, str] = {}
    weekends_failed: dict[str, str] = {}
    for month in months:
        logging.info("rebuild month start %s", month)
        async with db.get_session() as session:
            prev = await session.get(MonthPage, month)
            prev_hash = prev.content_hash if prev else None
            prev_hash2 = prev.content_hash2 if prev else None
        try:
            if force:
                await sync_month_page(db, month, update_links=False, force=True)
            else:
                await sync_month_page(db, month, update_links=False)
        except Exception as e:  # pragma: no cover
            logging.error("update month %s failed %s", month, e)
            months_failed[month] = str(e)
            continue
        async with db.get_session() as session:
            page = await session.get(MonthPage, month)
        if page and (force or prev is None or page.content_hash != prev_hash or page.content_hash2 != prev_hash2):
            urls = [u for u in [page.url, page.url2] if u]
            months_updated[month] = urls
            for idx, u in enumerate(urls, start=1):
                logging.info("update month %s part%d done %s", month, idx, u)
        else:
            logging.info("update month %s no changes", month)
        logging.info(
            "rebuild month finish %s updated=%s failed=%s",
            month,
            month in months_updated,
            month in months_failed,
        )
    for start in weekends:
        logging.info("rebuild weekend start %s", start)
        async with db.get_session() as session:
            prev = await session.get(WeekendPage, start)
            prev_hash = prev.content_hash if prev else None
        try:
            if force:
                await sync_weekend_page(
                    db, start, update_links=False, post_vk=False, force=True
                )
            else:
                await sync_weekend_page(db, start, update_links=False, post_vk=False)
        except Exception as e:  # pragma: no cover
            logging.error("update weekend %s failed %s", start, e)
            weekends_failed[start] = str(e)
            continue
        async with db.get_session() as session:
            page = await session.get(WeekendPage, start)
        if page and (force or prev is None or page.content_hash != prev_hash):
            urls = [page.url] if page.url else []
            weekends_updated[start] = urls
            for u in urls:
                logging.info("update weekend %s done %s", start, u)
        else:
            logging.info("update weekend %s no changes", start)
        logging.info(
            "rebuild weekend finish %s updated=%s failed=%s",
            start,
            start in weekends_updated,
            start in weekends_failed,
        )
    logging.info("rebuild finished")
    return {
        "months": {"updated": months_updated, "failed": months_failed},
        "weekends": {"updated": weekends_updated, "failed": weekends_failed},
    }


async def nightly_page_sync(db: Database, run_id: str | None = None) -> None:
    """Rebuild all stored month and weekend pages once per night."""
    async with span("db"):
        async with db.get_session() as session:
            res = await session.execute(select(Event.date))
            dates = [d for (d,) in res.all()]
    months: set[str] = set()
    weekends: set[str] = set()
    for dt_str in dates:
        start = dt_str.split("..", 1)[0]
        d = parse_iso_date(start)
        if not d:
            continue
        months.add(d.strftime("%Y-%m"))
        w = weekend_start_for_date(d)
        if w:
            weekends.add(w.isoformat())
    months_list = sorted(months)
    weekends_list = sorted(weekends)
    logging.info(
        "nightly_page_sync start months=%s weekends=%s",
        months_list,
        weekends_list,
    )
    await rebuild_pages(db, months_list, weekends_list, force=True)
    logging.info("nightly_page_sync finish")


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
    logging.info("CATBOX_ENABLED resolved to %s", CATBOX_ENABLED)
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
    try:
        scheduler_startup(db, bot)
    except Exception:
        logging.exception("scheduler_startup failed; continuing without scheduler")
    app["daily_scheduler"] = asyncio.create_task(daily_scheduler(db, bot))
    app["add_event_worker"] = asyncio.create_task(add_event_queue_worker(db, bot))
    app["add_event_watch"] = asyncio.create_task(_watch_add_event_worker(app, db, bot))
    app["job_outbox_worker"] = asyncio.create_task(job_outbox_worker(db, bot))
    gc.collect()
    logging.info("BOOT_OK pid=%s", os.getpid())


def _topic_labels_for_display(topics: Sequence[str] | None) -> list[str]:
    labels: list[str] = []
    if not topics:
        return labels

    seen: set[str] = set()
    for topic in topics:
        if not isinstance(topic, str):
            continue
        raw = topic.strip()
        if not raw:
            continue
        canonical = normalize_topic_identifier(raw)
        if canonical:
            if canonical in seen:
                continue
            seen.add(canonical)
            labels.append(TOPIC_LABELS.get(canonical, canonical))
        else:
            dedup_key = raw.casefold()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            labels.append(raw)
    return labels


def _format_topics_line(topics: Sequence[str] | None, manual: bool) -> str:
    labels = _topic_labels_for_display(topics)
    content = ", ".join(labels) if labels else "—"
    suffix = " (ручной режим)" if manual else ""
    return f"Темы: {content}{suffix}"


def _format_topic_badges(topics: Sequence[str] | None) -> str | None:
    labels = _topic_labels_for_display(topics)
    if not labels:
        return None
    return " ".join(f"[{label}]" for label in labels)


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
        badges = _format_topic_badges(getattr(e, "topics", None))
        if badges:
            lines.append(badges)
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
        if e.vk_ticket_short_key:
            lines.append(
                f"Статистика VK: https://vk.com/cc?act=stats&key={e.vk_ticket_short_key}"
            )
        lines.append("")
    if not lines:
        lines.append("No events")

    keyboard = []
    for e in events:
        icon = "✂️" if not e.vk_repost_url else "✅"
        row = [
            types.InlineKeyboardButton(
                text=f"\u274c {e.id}",
                callback_data=f"del:{e.id}:{target_date.isoformat()}",
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
            ),
            types.InlineKeyboardButton(
                text=f"{icon} Рерайт {e.id}",
                callback_data=f"vkrev:shortpost:{e.id}",
            ),
        ]
        keyboard.append(row)

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


async def build_exhibitions_message(
    db: Database, tz: timezone
) -> tuple[list[str], types.InlineKeyboardMarkup | None]:
    today = datetime.now(tz).date()
    today_iso = today.isoformat()
    async with db.get_session() as session:
        result = await session.execute(
            select(Event)
            .where(
                Event.event_type == "выставка",
                or_(
                    Event.end_date.is_not(None),
                    and_(Event.end_date.is_(None), Event.date >= today_iso),
                ),
                or_(Event.end_date >= today_iso, Event.end_date.is_(None)),
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
        badges = _format_topic_badges(getattr(e, "topics", None))
        if badges:
            lines.append(badges)
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
        if e.vk_ticket_short_key:
            lines.append(
                f"Статистика VK: https://vk.com/cc?act=stats&key={e.vk_ticket_short_key}"
            )
        lines.append("")

    if not lines:
        lines.append("No exhibitions")

    while lines and lines[-1] == "":
        lines.pop()

    keyboard = []
    for e in events:
        row = [
            types.InlineKeyboardButton(
                text=f"\u274c {e.id}", callback_data=f"del:{e.id}:exh"
            ),
            types.InlineKeyboardButton(
                text=f"\u270e {e.id}", callback_data=f"edit:{e.id}"
            ),
        ]
        keyboard.append(row)
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard) if events else None
    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_lines, current_len
        if current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_len = 0

    def add_line(line: str) -> None:
        nonlocal current_len
        while True:
            newline_cost = 1 if current_lines else 0
            projected = current_len + newline_cost + len(line)
            if projected <= TELEGRAM_MESSAGE_LIMIT:
                if newline_cost:
                    current_len += 1
                current_lines.append(line)
                current_len += len(line)
                return
            if current_lines:
                flush_current()
                continue
            truncated = _truncate_with_indicator(line, TELEGRAM_MESSAGE_LIMIT)
            if truncated:
                chunks.append(truncated)
            return

    for line in ["Exhibitions", *lines]:
        add_line(line)

    flush_current()

    if not chunks:
        chunks.append("")

    return chunks, markup


TELEGRAM_MESSAGE_LIMIT = 4096
POSTER_TRUNCATION_INDICATOR = "… (обрезано)"
POSTER_PREVIEW_UNAVAILABLE = "Poster OCR: превью недоступно — сообщение слишком длинное."


def _truncate_with_indicator(
    text: str, limit: int, indicator: str = POSTER_TRUNCATION_INDICATOR
) -> str:
    if limit <= 0:
        return ""
    if limit <= len(indicator):
        return indicator[:limit]
    return text[: limit - len(indicator)] + indicator


def _fit_poster_preview_lines(
    lines: Sequence[str], budget: int, indicator: str = POSTER_TRUNCATION_INDICATOR
) -> list[str]:
    if budget <= 0:
        return []

    fitted: list[str] = []
    for line in lines:
        candidate = fitted + [line]
        if len("\n".join(candidate)) <= budget:
            fitted.append(line)
            continue

        used_len = len("\n".join(fitted))
        newline_cost = 1 if fitted else 0
        remaining_for_content = budget - used_len - newline_cost
        if remaining_for_content <= 0:
            if not fitted:
                truncated = _truncate_with_indicator("", budget, indicator)
                return [truncated] if truncated else []

            prefix = fitted[:-1]
            last_line = fitted[-1]
            prefix_len = len("\n".join(prefix))
            if prefix:
                prefix_len += 1
            allowed_for_last = max(0, budget - prefix_len)
            fitted[-1] = _truncate_with_indicator(last_line, allowed_for_last, indicator)
            return fitted

        truncated_line = _truncate_with_indicator(line, remaining_for_content, indicator)
        if truncated_line:
            fitted.append(truncated_line)
        return fitted

    return fitted


async def show_edit_menu(
    user_id: int,
    event: Event,
    bot: Bot,
    db_obj: Database | None = None,
):
    data: dict[str, Any]
    try:
        data = event.model_dump()  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        data = event.dict()

    database = db_obj or globals().get("db")
    poster_lines: list[str] = []
    if database and event.id:
        async with database.get_session() as session:
            posters = (
                await session.execute(
                    select(EventPoster)
                    .where(EventPoster.event_id == event.id)
                    .order_by(EventPoster.updated_at.desc(), EventPoster.id.desc())
                )
            ).scalars().all()
        if posters:
            poster_lines.append("Poster OCR:")
            for idx, poster in enumerate(posters[:3], 1):
                token_parts: list[str] = []
                if poster.prompt_tokens:
                    token_parts.append(f"prompt={poster.prompt_tokens}")
                if poster.completion_tokens:
                    token_parts.append(f"completion={poster.completion_tokens}")
                if poster.total_tokens:
                    token_parts.append(f"total={poster.total_tokens}")
                token_info = f" ({', '.join(token_parts)})" if token_parts else ""
                hash_display = poster.poster_hash[:10]
                poster_lines.append(f"{idx}. hash={hash_display}{token_info}")
                raw_lines = (poster.ocr_text or "").splitlines()
                cleaned_lines = [line.strip() for line in raw_lines if line.strip()]
                if not cleaned_lines:
                    cleaned_lines = ["<пусто>"]
                for text_line in cleaned_lines:
                    poster_lines.append(f"    {text_line}")
                if poster.catbox_url:
                    url = poster.catbox_url
                    if len(url) > 120:
                        url = url[:117] + "..."
                    poster_lines.append(f"    {url}")
            poster_lines.append("---")

    lines = []
    topics_manual_flag = bool(data.get("topics_manual"))
    for key, value in data.items():
        if key == "topics":
            topics_value: Sequence[str] | None = None
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                topics_value = [str(item) for item in value]
            lines.append(_format_topics_line(topics_value, topics_manual_flag))
            continue
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
    if event.id and not event.festival:
        keyboard.append(
            [
                types.InlineKeyboardButton(
                    text="\U0001f3aa Сделать фестиваль",
                    callback_data=f"makefest:{event.id}",
                )
            ]
        )
    keyboard.append(
        [types.InlineKeyboardButton(text="Done", callback_data=f"editdone:{event.id}")]
    )
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)

    base_text = "\n".join(lines)
    base_len = len(base_text)
    poster_block: list[str] = []
    if poster_lines:
        newline_between_blocks = 1 if lines else 0
        poster_budget = TELEGRAM_MESSAGE_LIMIT - base_len - newline_between_blocks
        if poster_budget <= 0:
            notice_budget = TELEGRAM_MESSAGE_LIMIT - base_len - newline_between_blocks
            poster_block = _fit_poster_preview_lines(
                [POSTER_PREVIEW_UNAVAILABLE], notice_budget
            )
        else:
            poster_block = _fit_poster_preview_lines(poster_lines, poster_budget)
            if not poster_block and poster_budget > 0:
                poster_block = _fit_poster_preview_lines(
                    [POSTER_PREVIEW_UNAVAILABLE], poster_budget
                )

    message_lines = poster_block + lines if poster_block else lines
    message_text = "\n".join(message_lines)
    if len(message_text) > TELEGRAM_MESSAGE_LIMIT:
        message_text = _truncate_with_indicator(
            message_text, TELEGRAM_MESSAGE_LIMIT, POSTER_TRUNCATION_INDICATOR
        )

    await bot.send_message(user_id, message_text, reply_markup=markup)


async def show_festival_edit_menu(user_id: int, fest: Festival, bot: Bot):
    """Send festival fields with edit options."""
    lines = [
        f"name: {fest.name}",
        f"full: {fest.full_name or ''}",
        f"description: {fest.description or ''}",
        f"start: {fest.start_date or ''}",
        f"end: {fest.end_date or ''}",
        f"site: {fest.website_url or ''}",
        f"program: {fest.program_url or ''}",
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
                text=("Delete program" if fest.program_url else "Add program"),
                callback_data=f"festeditfield:{fest.id}:program",
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
        [
            types.InlineKeyboardButton(
                text="Обновить обложку из Telegraph",
                callback_data=f"festcover:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="Иллюстрации / обложка",
                callback_data=f"festimgs:{fest.id}",
            )
        ],
        [
            types.InlineKeyboardButton(
                text="🧩 Склеить с…",
                callback_data=f"festmerge:{fest.id}",
            )
        ],
        [types.InlineKeyboardButton(text="Done", callback_data="festeditdone")],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    await bot.send_message(user_id, "\n".join(lines), reply_markup=markup)


FEST_MERGE_PAGE_SIZE = 12


_FEST_MERGE_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _festival_merge_tokens(fest: Festival) -> set[str]:
    tokens: set[str] = set()
    if fest.name:
        tokens.update(token.lower() for token in _FEST_MERGE_TOKEN_RE.findall(fest.name))
    if fest.city:
        tokens.add(fest.city.lower())
    return tokens


def _sort_festival_merge_targets(
    source: Festival, targets: Sequence[Festival]
) -> list[Festival]:
    source_tokens = _festival_merge_tokens(source)
    scored: list[tuple[int, str, int, Festival]] = []
    for index, target in enumerate(targets):
        target_tokens = _festival_merge_tokens(target)
        overlap = len(source_tokens & target_tokens)
        name_key = (target.name or "").lower()
        scored.append((overlap, name_key, index, target))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored]


def _format_festival_merge_line(fest: Festival) -> str:
    parts = [f"{fest.id} {fest.name}"]
    if fest.start_date and fest.end_date:
        if fest.start_date == fest.end_date:
            parts.append(fest.start_date)
        else:
            parts.append(f"{fest.start_date}..{fest.end_date}")
    elif fest.start_date:
        parts.append(fest.start_date)
    elif fest.end_date:
        parts.append(fest.end_date)
    if fest.city:
        parts.append(f"#{fest.city}")
    return " · ".join(parts)


def build_festival_merge_selection(
    source: Festival, targets: Sequence[Festival], page: int
) -> tuple[str, types.InlineKeyboardMarkup]:
    sorted_targets = _sort_festival_merge_targets(source, targets)
    total = len(sorted_targets)
    total_pages = max(1, (total + FEST_MERGE_PAGE_SIZE - 1) // FEST_MERGE_PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * FEST_MERGE_PAGE_SIZE
    visible = sorted_targets[start : start + FEST_MERGE_PAGE_SIZE]

    heading = (
        f"🧩 Склеить фестиваль «{source.name}» (ID {source.id}).\n"
        f"Выберите фестиваль-цель (страница {page}/{total_pages}):"
    )
    lines = [heading]
    keyboard: list[list[types.InlineKeyboardButton]] = []

    if not visible:
        lines.append("Нет других фестивалей для объединения.")
    else:
        for target in visible:
            lines.append(_format_festival_merge_line(target))
            button_text = target.name
            if target.city:
                button_text = f"{target.name} · #{target.city}"
            keyboard.append(
                [
                    types.InlineKeyboardButton(
                        text=button_text,
                        callback_data=f"festmerge_to:{source.id}:{target.id}:{page}",
                    )
                ]
            )

    nav_row: list[types.InlineKeyboardButton] = []
    if total_pages > 1 and page > 1:
        nav_row.append(
            types.InlineKeyboardButton(
                text="⬅️ Назад",
                callback_data=f"festmergep:{source.id}:{page-1}",
            )
        )
    if total_pages > 1 and page < total_pages:
        nav_row.append(
            types.InlineKeyboardButton(
                text="Вперёд ➡️",
                callback_data=f"festmergep:{source.id}:{page+1}",
            )
        )
    if nav_row:
        keyboard.append(nav_row)

    keyboard.append(
        [types.InlineKeyboardButton(text="Отмена", callback_data=f"festedit:{source.id}")]
    )

    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    return "\n".join(lines), markup


async def handle_festmerge_callback(
    callback: types.CallbackQuery, db: Database, _bot: Bot
) -> None:
    """Show merge selection for a festival."""

    try:
        fid = int((callback.data or "").split(":")[1])
    except (IndexError, ValueError):
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        fest = await session.get(Festival, fid)
        if not fest:
            await callback.answer("Festival not found", show_alert=True)
            return
        res = await session.execute(
            select(Festival).where(Festival.id != fid).order_by(Festival.name)
        )
        targets = list(res.scalars().all())

    if not targets:
        await callback.answer("Нет других фестивалей", show_alert=True)
        return

    text, markup = build_festival_merge_selection(fest, targets, page=1)
    await callback.message.answer(text, reply_markup=markup)
    await callback.answer()


async def handle_festmerge_page_callback(
    callback: types.CallbackQuery, db: Database, _: Bot
) -> None:
    """Handle pagination inside merge selection."""

    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, page_raw = parts
    try:
        src_id = int(src_raw)
        page = int(page_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        if not src:
            await callback.answer("Festival not found", show_alert=True)
            return
        res = await session.execute(
            select(Festival).where(Festival.id != src_id).order_by(Festival.name)
        )
        targets = list(res.scalars().all())

    text, markup = build_festival_merge_selection(src, targets, page)
    await callback.message.edit_text(text, reply_markup=markup)
    await callback.answer()


async def handle_festmerge_to_callback(
    callback: types.CallbackQuery, db: Database, _: Bot
) -> None:
    """Confirm merge target selection."""

    parts = (callback.data or "").split(":")
    if len(parts) != 4:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, dst_raw, page_raw = parts
    try:
        src_id = int(src_raw)
        dst_id = int(dst_raw)
        page = int(page_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

    if not src or not dst:
        await callback.answer("Festival not found", show_alert=True)
        return

    confirm_lines = [
        "Вы уверены, что хотите склеить фестивали?",
        f"Источник: {src.id} {src.name}",
        f"Цель: {dst.id} {dst.name}",
        "Все события и данные источника будут перенесены, сам фестиваль будет удалён.",
    ]
    keyboard = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="✅ Склеить",
                    callback_data=f"festmerge_do:{src_id}:{dst_id}",
                )
            ],
            [
                types.InlineKeyboardButton(
                    text="⬅️ Назад",
                    callback_data=f"festmergep:{src_id}:{page}",
                ),
                types.InlineKeyboardButton(
                    text="Отмена",
                    callback_data=f"festedit:{src_id}",
                ),
            ],
        ]
    )
    await callback.message.edit_text("\n".join(confirm_lines), reply_markup=keyboard)
    await callback.answer()


async def handle_festmerge_do_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    """Execute merge and report result."""

    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        await callback.answer("Некорректный запрос", show_alert=True)
        return
    _, src_raw, dst_raw = parts
    try:
        src_id = int(src_raw)
        dst_id = int(dst_raw)
    except ValueError:
        await callback.answer("Некорректный запрос", show_alert=True)
        return

    async with db.get_session() as session:
        if not await session.get(User, callback.from_user.id):
            await callback.answer("Not authorized", show_alert=True)
            return
        src = await session.get(Festival, src_id)
        dst = await session.get(Festival, dst_id)

    if not src or not dst:
        await callback.answer("Festival not found", show_alert=True)
        return

    src_name = src.name
    dst_name = dst.name
    ok = await merge_festivals(db, src_id, dst_id, bot)
    if not ok:
        await callback.answer("Не удалось объединить", show_alert=True)
        return

    async with db.get_session() as session:
        dest = await session.get(Festival, dst_id)

    if dest:
        festival_edit_sessions[callback.from_user.id] = (dst_id, None)
        await show_festival_edit_menu(callback.from_user.id, dest, bot)

    await callback.message.edit_text(
        f"Фестиваль «{src_name}» объединён с «{dst_name}»."
    )
    await callback.answer("Склеено")


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


async def show_digest_menu(message: types.Message, db: Database, bot: Bot) -> None:
    if not (message.text or "").startswith("/digest"):
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return

    digest_id = uuid.uuid4().hex
    keyboard = [
        [
            types.InlineKeyboardButton(
                text="✅ Лекции",
                callback_data=f"digest:select:lectures:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Мастер-классы",
                callback_data=f"digest:select:masterclasses:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Выставки",
                callback_data=f"digest:select:exhibitions:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Психология",
                callback_data=f"digest:select:psychology:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Научпоп",
                callback_data=f"digest:select:science_pop:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Краеведение",
                callback_data=f"digest:select:kraevedenie:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Нетворкинг",
                callback_data=f"digest:select:networking:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Развлечения",
                callback_data=f"digest:select:entertainment:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Маркеты",
                callback_data=f"digest:select:markets:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Кинопоказы",
                callback_data=f"digest:select:movies:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Классический театр",
                callback_data=f"digest:select:theatre_classic:{digest_id}",
            ),
            types.InlineKeyboardButton(
                text="✅ Современный театр",
                callback_data=f"digest:select:theatre_modern:{digest_id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="✅ Встречи и клубы",
                callback_data=f"digest:select:meetups:{digest_id}",
            )
        ],
        [
            types.InlineKeyboardButton(text="⏳ Выходные", callback_data="digest:disabled"),
            types.InlineKeyboardButton(text="⏳ Популярное за неделю", callback_data="digest:disabled"),
        ],
    ]
    markup = types.InlineKeyboardMarkup(inline_keyboard=keyboard)
    sent = await bot.send_message(
        message.chat.id, "Выберите тип дайджеста:", reply_markup=markup
    )
    logging.info(
        "digest.menu.shown digest_id=%s chat_id=%s user_id=%s message_id=%s",
        digest_id,
        message.chat.id,
        message.from_user.id,
        getattr(sent, "message_id", None),
    )


DEFAULT_PANEL_TEXT = (
    "Управление дайджестом\nВыключите лишнее и нажмите «Обновить превью»."
)


def _build_digest_panel_markup(digest_id: str, session: dict) -> types.InlineKeyboardMarkup:
    buttons: List[types.InlineKeyboardButton] = []
    excluded: set[int] = session.get("excluded", set())
    for idx, item in enumerate(session["items"]):
        mark = "✅" if idx not in excluded else "❌"
        buttons.append(
            types.InlineKeyboardButton(
                text=f"{mark} {item['index']}",
                callback_data=f"dg:t:{digest_id}:{item['index']}",
            )
        )
    rows = [buttons[i : i + 3] for i in range(0, len(buttons), 3)]
    rows.append(
        [types.InlineKeyboardButton(text="🔄 Обновить превью", callback_data=f"dg:r:{digest_id}")]
    )
    for ch in session.get("channels", []):
        rows.append(
            [
                types.InlineKeyboardButton(
                    text=f"🚀 Отправить в «{ch['name']}»",
                    callback_data=f"dg:s:{digest_id}:{ch['channel_id']}",
                )
            ]
        )
    rows.append(
        [types.InlineKeyboardButton(text="🗑 Скрыть панель", callback_data=f"dg:x:{digest_id}")]
    )
    logging.info(
        "digest.controls.render digest_id=%s count=%s excluded=%s",
        digest_id,
        len(session["items"]),
        sorted(excluded),
    )
    return types.InlineKeyboardMarkup(inline_keyboard=rows)


async def _compose_from_session(
    session: dict, digest_id: str
) -> tuple[str, List[str], bool, int, List[int]]:
    excluded: set[int] = session.get("excluded", set())
    indices = [i for i in range(len(session["items"])) if i not in excluded]
    lines_html = [session["items"][i]["line_html"] for i in indices]
    caption, used_lines = await compose_digest_caption(
        session["intro_html"],
        lines_html,
        session["footer_html"],
        digest_id=digest_id,
    )
    used_indices = indices[: len(used_lines)]
    media_urls: List[str] = []
    seen_urls: set[str] = set()
    for i in used_indices:
        url = session["items"][i]["cover_url"]
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        media_urls.append(url)
    media = [types.InputMediaPhoto(media=url) for url in media_urls]
    attach, _ = attach_caption_if_fits(media, caption)
    vis_len = visible_caption_len(caption)
    logging.info(
        "digest.caption.visible_len digest_id=%s visible=%s attach=%s",
        digest_id,
        vis_len,
        int(attach),
    )
    return caption, media_urls, attach, vis_len, used_indices


async def _send_preview(session: dict, digest_id: str, bot: Bot):
    caption, media_urls, attach, vis_len, used_indices = await _compose_from_session(
        session, digest_id
    )
    session["current_caption_html"] = caption
    session["current_media_urls"] = media_urls
    session["current_attach"] = attach
    session["current_visible_len"] = vis_len
    session["current_used_indices"] = used_indices

    msg_ids: List[int] = []
    media: List[types.InputMediaPhoto] = []
    for i, url in enumerate(media_urls):
        if i == 0 and attach:
            media.append(
                types.InputMediaPhoto(
                    media=url, caption=caption, parse_mode="HTML"
                )
            )
        else:
            media.append(types.InputMediaPhoto(media=url))
    if media:
        sent = await bot.send_media_group(session["chat_id"], media)
        msg_ids.extend(m.message_id for m in sent)
    if not attach or not media:
        msg = await bot.send_message(
            session["chat_id"],
            caption,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        msg_ids.append(msg.message_id)
    panel = await bot.send_message(
        session["chat_id"],
        session.get("panel_text", DEFAULT_PANEL_TEXT),
        reply_markup=_build_digest_panel_markup(digest_id, session),
    )
    session["preview_msg_ids"] = msg_ids
    session["panel_msg_id"] = panel.message_id
    return caption, attach, vis_len, len(used_indices)




async def handle_digest_toggle(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, idx_str = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    index = int(idx_str) - 1
    excluded: set[int] = session.setdefault("excluded", set())
    if 0 <= index < len(session["items"]):
        if index in excluded:
            excluded.remove(index)
            active = True
        else:
            excluded.add(index)
            active = False
    else:
        active = False
    markup = _build_digest_panel_markup(digest_id, session)
    try:
        await bot.edit_message_reply_markup(
            session["chat_id"], session["panel_msg_id"], reply_markup=markup
        )
    except Exception:
        logging.exception(
            "digest.panel.toggle edit_error digest_id=%s message_id=%s",
            digest_id,
            session.get("panel_msg_id"),
        )
    logging.info(
        "digest.controls.toggle digest_id=%s idx=%s active=%s",
        digest_id,
        index,
        str(active).lower(),
    )
    await callback.answer()


async def handle_digest_refresh(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 3:
        return
    _, _, digest_id = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    excluded: set[int] = session.get("excluded", set())
    remaining = [i for i in range(len(session["items"])) if i not in excluded]
    if not remaining:
        noun = session.get("items_noun", "лекций")
        await callback.answer(f"Нет выбранных {noun}", show_alert=True)
        return

    logging.info(
        "digest.preview.recompose.start digest_id=%s kept=%s excluded=%s",
        digest_id,
        len(remaining),
        len(excluded),
    )

    digest_type = session.get("digest_type", "lectures")
    horizon_days = session.get("horizon_days", 0)
    start = _time.monotonic()
    logging.info(
        "digest.intro.llm.request digest_id=%s type=%s items=%s",
        digest_id,
        digest_type,
        len(remaining),
    )
    try:
        if digest_type == "masterclasses":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                    }
                )
            intro = await compose_masterclasses_intro_via_4o(
                len(payload), horizon_days, payload
            )
        elif digest_type == "exhibitions":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                start_date = item.get("date", "")
                end_date = item.get("end_date") or start_date
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                        "date_range": {"start": start_date, "end": end_date},
                    }
                )
            intro = await compose_exhibitions_intro_via_4o(
                len(payload), horizon_days, payload
            )
        elif digest_type == "psychology":
            payload = []
            for idx in remaining:
                item = session["items"][idx]
                payload.append(
                    {
                        "title": item.get("norm_title") or item.get("title", ""),
                        "description": item.get("norm_description", ""),
                        "topics": item.get("norm_topics", []),
                    }
                )
            intro = await compose_psychology_intro_via_4o(
                len(payload), horizon_days, payload
            )
        else:
            titles = [session["items"][i]["norm_title"] for i in remaining]
            intro = await compose_digest_intro_via_4o(
                len(remaining),
                horizon_days,
                titles,
                event_noun=session.get("items_noun", "лекций"),
            )
    except Exception as e:
        logging.error(
            "digest.intro.llm.error digest_id=%s err=\"%s\"",
            digest_id,
            e,
        )
    else:
        duration_ms = int((_time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response digest_id=%s type=%s text_len=%s took_ms=%s",
            digest_id,
            digest_type,
            len(intro),
            duration_ms,
        )
        session["intro_html"] = intro

    for mid in session.get("preview_msg_ids", []):
        try:
            await bot.delete_message(session["chat_id"], mid)
        except Exception:
            logging.error(
                "digest.panel.refresh delete_error digest_id=%s message_id=%s",
                digest_id,
                mid,
            )
    if session.get("panel_msg_id"):
        try:
            await bot.delete_message(session["chat_id"], session["panel_msg_id"])
        except Exception:
            logging.error(
                "digest.panel.refresh delete_error digest_id=%s message_id=%s",
                digest_id,
                session["panel_msg_id"],
            )

    caption, attach, vis_len, kept = await _send_preview(
        session, digest_id, bot
    )
    logging.info(
        "digest.preview.recompose.done digest_id=%s media=%s caption_attached=%s",
        digest_id,
        len(session.get("current_media_urls", [])),
        int(attach),
    )
    logging.info(
        "digest.panel.refresh.sent digest_id=%s panel_msg_id=%s",
        digest_id,
        session.get("panel_msg_id"),
    )
    await callback.answer()


async def handle_digest_send(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, ch_id_str = parts
    channel_id = int(ch_id_str)
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return

    excluded: set[int] = session.get("excluded", set())
    if len(session["items"]) - len(excluded) == 0:
        noun = session.get("items_noun", "лекций")
        await callback.answer(f"Нет выбранных {noun}", show_alert=True)
        return

    caption = session.get("current_caption_html", "")
    media_urls = session.get("current_media_urls", [])
    attach = session.get("current_attach", False)

    album_msg_ids: List[int] = []
    caption_msg_id: int | None = None
    media = [types.InputMediaPhoto(media=url) for url in media_urls]
    if attach and media:
        media[0].parse_mode = "HTML"
        media[0].caption = caption
    if media:
        sent = await bot.send_media_group(channel_id, media)
        album_msg_ids = [m.message_id for m in sent]
    if not attach or not media:
        msg = await bot.send_message(
            channel_id,
            caption,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        caption_msg_id = msg.message_id
    else:
        caption_msg_id = album_msg_ids[0] if album_msg_ids else None

    ch_info = next(
        (c for c in session.get("channels", []) if c["channel_id"] == channel_id),
        None,
    )
    if ch_info and caption_msg_id is not None:
        ch_obj = SimpleNamespace(
            channel_id=ch_info["channel_id"],
            title=ch_info.get("name"),
            username=ch_info.get("username"),
        )
        link = build_channel_post_url(ch_obj, caption_msg_id)
        await bot.send_message(session["chat_id"], link)

        used_indices = session.get("current_used_indices", [])
        items = session.get("items", [])
        event_ids = [
            items[i]["event_id"]
            for i in used_indices
            if i < len(items) and items[i].get("event_id")
        ]
        creator_ids = {
            items[i].get("creator_id")
            for i in used_indices
            if i < len(items) and items[i].get("creator_id")
        }
        draft_key = f"draft:digest:{digest_id}"
        raw = await get_setting_value(db, draft_key)
        data = json.loads(raw) if raw else {}
        published_to = data.setdefault("published_to", {})
        published_to[str(channel_id)] = {
            "message_url": link,
            "event_ids": event_ids,
            "notified_partner_ids": [],
        }
        await set_setting_value(db, draft_key, json.dumps(data))

        partners: list[User] = []
        if creator_ids:
            async with db.get_session() as session_db:
                result = await session_db.execute(
                    select(User).where(
                        User.user_id.in_(creator_ids), User.is_partner == True
                    )
                )
                partners = result.scalars().all()
        if partners:
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        types.InlineKeyboardButton(
                            text="Уведомить партнёров",
                            callback_data=f"dg:np:{digest_id}:{channel_id}",
                        )
                    ]
                ]
            )
            await bot.send_message(
                session["chat_id"],
                "В дайджесте есть события, добавленные партнёрами.",
                reply_markup=markup,
            )
        else:
            await bot.send_message(
                session["chat_id"],
                "В дайджесте нет событий, добавленных партнёрами.",
            )

    logging.info(
        "digest.publish digest_id=%s channel_id=%s message_id=%s attached=%s kept=%s",
        digest_id,
        channel_id,
        caption_msg_id,
        int(attach),
        len(media_urls),
    )
    await callback.answer("Опубликовано", show_alert=False)


async def handle_digest_notify_partners(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    _, _, digest_id, ch_id_str = parts
    channel_id = int(ch_id_str)
    draft_key = f"draft:digest:{digest_id}"
    raw = await get_setting_value(db, draft_key)
    if not raw:
        await callback.answer("Сначала опубликуйте дайджест", show_alert=True)
        return
    data = json.loads(raw)
    published_to = data.get("published_to", {})
    entry = published_to.get(str(channel_id))
    if not entry:
        await callback.answer("Сначала опубликуйте дайджест", show_alert=True)
        return
    event_ids = entry.get("event_ids", [])
    notified_ids = set(entry.get("notified_partner_ids", []))
    async with db.get_session() as session_db:
        if event_ids:
            res_ev = await session_db.execute(
                select(Event).where(Event.id.in_(event_ids))
            )
            events = res_ev.scalars().all()
            creator_ids = {ev.creator_id for ev in events if ev.creator_id}
        else:
            creator_ids = set()
        if creator_ids:
            res_users = await session_db.execute(
                select(User).where(
                    User.user_id.in_(creator_ids), User.is_partner == True
                )
            )
            partners = res_users.scalars().all()
        else:
            partners = []
    to_notify = [u for u in partners if u.user_id not in notified_ids]
    if not to_notify:
        await callback.answer("Уже уведомлено", show_alert=False)
        return
    notified_now: list[int] = []
    for u in to_notify:
        try:
            await bot.send_message(
                u.user_id, f"Ваше событие попало в дайджест: {entry.get('message_url')}"
            )
            notified_now.append(u.user_id)
        except Exception as e:
            logging.error("digest.notify_partner failed user_id=%s err=%s", u.user_id, e)
    usernames: list[str] = []
    for u in to_notify:
        if u.username:
            usernames.append(f"@{u.username}")
        else:
            usernames.append(f'<a href="tg://user?id={u.user_id}">Партнёр</a>')
    if callback.message:
        await bot.send_message(
            callback.message.chat.id,
            f"Уведомлено: {', '.join(usernames)}",
            parse_mode="HTML",
        )
    entry["notified_partner_ids"] = list(notified_ids | set(notified_now))
    published_to[str(channel_id)] = entry
    data["published_to"] = published_to
    await set_setting_value(db, draft_key, json.dumps(data))
    await callback.answer()


async def handle_digest_hide(callback: types.CallbackQuery, bot: Bot) -> None:
    parts = callback.data.split(":")
    if len(parts) != 3:
        return
    _, _, digest_id = parts
    session = digest_preview_sessions.get(digest_id)
    if not session:
        await callback.answer(
            "Сессия превью истекла, соберите новый дайджест командой /digest",
            show_alert=True,
        )
        return
    if session.get("panel_msg_id"):
        try:
            await bot.delete_message(session["chat_id"], session["panel_msg_id"])
        except Exception:
            logging.error(
                "digest.panel.hide delete_error digest_id=%s message_id=%s",
                digest_id,
                session["panel_msg_id"],
            )
        session["panel_msg_id"] = None
    await callback.answer()

    draft_key = f"draft:digest:{digest_id}"
    raw = await get_setting_value(db, draft_key)
    if not raw:
        await callback.answer("Черновик не найден", show_alert=False)
        return
    data = json.loads(raw)
    published_to = data.setdefault("published_to", {})
    if str(channel_id) in published_to:
        await callback.answer("Уже отправлено", show_alert=False)
        logging.info(
            "digest.publish.skip digest_id=%s channel_id=%s reason=already_sent",
            digest_id,
            channel_id,
        )
        return

    image_urls = data.get("image_urls") or []
    caption_text = data.get("caption_text") or ""

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

    chunks, markup = await build_exhibitions_message(db, tz)
    if not chunks:
        return
    first, *rest = chunks
    sent_messages: list[tuple[int, str]] = []
    first_msg = await bot.send_message(message.chat.id, first, reply_markup=markup)
    sent_messages.append((first_msg.message_id, first))
    for chunk in rest:
        msg = await bot.send_message(message.chat.id, chunk)
        sent_messages.append((msg.message_id, chunk))
    exhibitions_message_state[message.chat.id] = sent_messages


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


async def handle_weekendimg_cmd(message: types.Message, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            return
    today = datetime.now(LOCAL_TZ).date()
    first = next_weekend_start(today)
    dates = [first + timedelta(days=7 * i) for i in range(5)]
    rows = [
        [
            types.InlineKeyboardButton(
                text=f"Выходные {format_weekend_range(d)}",
                callback_data=f"weekimg:{d.isoformat()}",
            )
        ]
        for d in dates
    ]
    rows.append(
        [
            types.InlineKeyboardButton(
                text="Обложка лендинга фестивалей",
                callback_data=f"weekimg:{FESTIVALS_INDEX_MARKER}",
            )
        ]
    )
    kb = types.InlineKeyboardMarkup(inline_keyboard=rows)
    await message.answer("Выберите выходные для обложки:", reply_markup=kb)


async def handle_weekendimg_cb(callback: types.CallbackQuery, db: Database, bot: Bot):
    async with db.get_session() as session:
        user = await session.get(User, callback.from_user.id)
        if not user or not user.is_superadmin:
            await callback.answer("Недостаточно прав", show_alert=True)
            return
    if not callback.data:
        return
    start = callback.data.split(":", 1)[1]
    weekend_img_wait[callback.from_user.id] = start
    await callback.message.edit_reply_markup(reply_markup=None)
    if start == FESTIVALS_INDEX_MARKER:
        await callback.message.answer(
            "Выбрана обложка лендинга фестивалей.\n"
            "Пришлите обложку одним сообщением (фото или файл).",
        )
    else:
        try:
            start_date = date.fromisoformat(start)
        except ValueError:
            await callback.message.answer(
                "Не удалось распознать дату. Попробуйте выбрать вариант ещё раз."
            )
            weekend_img_wait.pop(callback.from_user.id, None)
            await callback.answer()
            return
        await callback.message.answer(
            f"Выбраны выходные {format_weekend_range(start_date)}.\n"
            "Пришлите обложку одним сообщением (фото или файл).",
        )
    await callback.answer()


async def handle_weekendimg_photo(message: types.Message, db: Database, bot: Bot):
    start = weekend_img_wait.get(message.from_user.id)
    if not start:
        return

    images = (await extract_images(message, bot))[:1]
    if not images:
        await message.reply("Не вижу изображения. Пришлите одно фото/файл в ответ.")
        return

    urls, _ = await upload_images(images, limit=1, force=True)
    if not urls:
        await message.reply("Не удалось загрузить в Catbox. Попробуйте другое фото.")
        return

    cover = urls[0]
    if start == FESTIVALS_INDEX_MARKER:
        landing_url = ""
        try:
            await set_setting_value(db, "festivals_index_cover", cover)
            await sync_festivals_index_page(db)
            _, landing_url = await rebuild_festivals_index_if_needed(
                db, force=True
            )
        except Exception:
            logging.exception("Failed to update festivals index cover")
            await message.reply(
                "Обложка сохранена, но страницу не удалось обновить. Попробуйте ещё раз."
            )
        else:
            if landing_url:
                await message.reply(
                    f"Готово! Обложка лендинга обновлена.\n{landing_url}"
                )
            else:
                await message.reply(
                    "Обложка сохранена, но ссылку на лендинг получить не удалось."
                )
        finally:
            weekend_img_wait.pop(message.from_user.id, None)
        return

    await set_setting_value(db, f"weekend_cover:{start}", cover)
    await sync_weekend_page(db, start, update_links=True, post_vk=False, force=True)

    async with db.get_session() as session:
        page = await session.get(WeekendPage, start)
    if page and page.url:
        await message.reply(f"Готово! Обложка добавлена.\n{page.url}")
    else:
        await message.reply(
            "Обложка сохранена, но страницу не удалось обновить. Попробуйте ещё раз."
        )

    weekend_img_wait.pop(message.from_user.id, None)


def _shift_month(d: date, offset: int) -> date:
    year = d.year + (d.month - 1 + offset) // 12
    month = (d.month - 1 + offset) % 12 + 1
    return date(year, month, 1)


def _expand_months(months: list[str], past: int, future: int) -> list[str]:
    if months:
        return sorted(months)
    today = date.today().replace(day=1)
    start = _shift_month(today, -past)
    total = past + future + 1
    res = []
    for i in range(total):
        m = _shift_month(start, i)
        res.append(m.strftime("%Y-%m"))
    return res


async def _future_months_with_events(db: Database) -> list[str]:
    today = date.today().replace(day=1)
    rows = await db.exec_driver_sql("SELECT date FROM event")
    months: set[str] = set()
    for (raw_date,) in rows:
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except Exception:
            continue
        if dt >= today:
            months.add(dt.strftime("%Y-%m"))
    return sorted(months)


def _weekends_for_months(months: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    weekends: set[str] = set()
    mapping: dict[str, list[str]] = defaultdict(list)
    for m in months:
        year, mon = map(int, m.split("-"))
        d = date(year, mon, 1)
        while d.month == mon:
            w = weekend_start_for_date(d)
            if w and w.month == mon:
                key = w.isoformat()
                if key not in mapping[m]:
                    mapping[m].append(key)
                    weekends.add(key)
            d += timedelta(days=1)
    return sorted(weekends), mapping


async def _perform_pages_rebuild(db: Database, months: list[str], force: bool = False) -> str:
    weekends, mapping = _weekends_for_months(months)
    global DISABLE_EVENT_PAGE_UPDATES
    DISABLE_EVENT_PAGE_UPDATES = True
    try:
        result = await rebuild_pages(db, months, weekends, force=force)
    finally:
        DISABLE_EVENT_PAGE_UPDATES = False

    lines = ["Telegraph month rebuild:"]
    for m in months:
        failed = result["months"]["failed"].get(m)
        urls = result["months"]["updated"].get(m, [])
        if failed:
            lines.append(f"❌ {m} — ошибка: {failed}")
        else:
            lines.append(f"✅ {m} — обновлено:")
            if len(urls) == 2:
                lines.append(f"  • Часть 1: {urls[0]}")
                lines.append(f"  • Часть 2: {urls[1]}")
            elif len(urls) == 1:
                lines.append(f"  • {urls[0]}")
            else:
                lines.append("  • отсутствует")

    lines.append("\nTelegraph weekends rebuild:")
    for m in months:
        w_list = mapping.get(m, [])
        total = len(w_list)
        success = 0
        month_lines: list[str] = []
        for w in w_list:
            label = format_weekend_range(date.fromisoformat(w))
            urls = result["weekends"]["updated"].get(w, [])
            err = result["weekends"]["failed"].get(w)
            if urls:
                success += 1
                month_lines.append(f"  • {label}: ✅ {urls[0]}")
            elif err:
                status = "⏳ перенесено" if "flood" in err.lower() else "❌"
                month_lines.append(f"  • {label}: {status} {err}")
            else:
                month_lines.append(f"  • {label}: ❌ неизвестно")
        if success == total:
            lines.append(f"✅ {m} — обновлено: {total} страниц")
        elif success == 0:
            lines.append(f"❌ {m} — ошибка:")
        else:
            lines.append(f"☑️ {success} из {total} обновлены:")
        lines.extend(month_lines)

    return "\n".join(lines)


def _parse_pages_rebuild_args(text: str) -> tuple[list[str], int, int, bool]:
    parts = text.split()[1:]
    months: list[str] = []
    past = 0
    future = 2
    force = False
    for p in parts:
        if p.startswith("--past="):
            try:
                past = int(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p.startswith("--future="):
            try:
                future = int(p.split("=", 1)[1])
            except ValueError:
                pass
        elif p == "--force":
            force = True
        else:
            months.append(p)
    return months, past, future, force


async def handle_pages_rebuild(message: types.Message, db: Database, bot: Bot):
    months, past, future, _ = _parse_pages_rebuild_args(message.text or "")
    if not months and (message.text or "").strip() == "/pages_rebuild":
        options = await _future_months_with_events(db)
        if not options:
            options = _expand_months([], past, future)
        buttons = [
            [types.InlineKeyboardButton(text=m, callback_data=f"pages_rebuild:{m}")]
            for m in options
        ]
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=buttons
            + [[types.InlineKeyboardButton(text="Все", callback_data="pages_rebuild:ALL")]]
        )
        await bot.send_message(
            message.chat.id,
            "Выберите месяц для пересборки или «Все»",
            reply_markup=markup,
        )
        return
    months_list = _expand_months(months, past, future)
    report = await _perform_pages_rebuild(db, months_list, force=True)
    await bot.send_message(message.chat.id, report)


async def handle_pages_rebuild_cb(
    callback: types.CallbackQuery, db: Database, bot: Bot
):
    await callback.answer()
    val = callback.data.split(":", 1)[1]
    if val.upper() == "ALL":
        months = await _future_months_with_events(db)
        if not months:
            months = _expand_months([], 0, 2)
    else:
        months = [val]
    report = await _perform_pages_rebuild(db, months, force=True)
    await bot.send_message(callback.message.chat.id, report)


async def handle_fest(message: types.Message, db: Database, bot: Bot):
    archive = False
    page = 1
    text = message.text or ""
    parts = text.split()
    for part in parts[1:]:
        if part.lower() == "archive":
            archive = True
        else:
            try:
                page = int(part)
            except ValueError:
                continue
    await send_festivals_list(
        message,
        db,
        bot,
        user_id=message.from_user.id if message.from_user else None,
        edit=False,
        page=page,
        archive=archive,
    )





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

    async def month_views(mp: MonthPage) -> int | None:
        paths: list[tuple[str, str | None]] = []
        if mp.path:
            paths.append((mp.path, mp.url))
        if mp.path2:
            paths.append((mp.path2, mp.url2))
        if not paths:
            return None

        total = 0
        has_value = False
        for path, url in paths:
            views = await fetch_views(path, url)
            if views is None:
                continue
            total += views
            has_value = True

        return total if has_value else None

    if mp_prev:

        views = await month_views(mp_prev)

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
        if not (mp.path or mp.path2):
            continue

        views = await month_views(mp)

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


async def collect_festivals_landing_stats(db: Database) -> str | None:
    """Return Telegraph view count for the festivals landing page."""
    path = await get_setting_value(db, "festivals_index_path") or await get_setting_value(
        db, "fest_index_path"
    )
    url = await get_setting_value(db, "festivals_index_url") or await get_setting_value(
        db, "fest_index_url"
    )
    if not path and url:
        try:
            path = url.split("//", 1)[1].split("/", 1)[1]
        except Exception:
            path = None
    if not path:
        return None
    views = await fetch_views(path, url)
    if views is None:
        return None
    return f"Лендинг фестивалей: {views} просмотров"


async def collect_festival_telegraph_stats(db: Database) -> list[str]:
    """Return Telegraph view counts for upcoming and recent festivals."""
    today = datetime.now(LOCAL_TZ).date()
    week_ago = today - timedelta(days=7)
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.telegraph_path.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int]] = []
    for f in fests:
        start = parse_iso_date(f.start_date) if f.start_date else None
        end = parse_iso_date(f.end_date) if f.end_date else start
        if not ((start and start >= today) or (end and end >= week_ago)):
            continue
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
        response = await vk_api("wall.getById", posts=f"{owner_id}_{post_id}")
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
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
    """Return VK view and reach counts for upcoming and recent festivals."""
    today = datetime.now(LOCAL_TZ).date()
    week_ago = today - timedelta(days=7)
    async with db.get_session() as session:
        result = await session.execute(
            select(Festival).where(Festival.vk_post_url.is_not(None))
        )
        fests = result.scalars().all()
    stats: list[tuple[str, int, int]] = []
    for f in fests:
        start = parse_iso_date(f.start_date) if f.start_date else None
        end = parse_iso_date(f.end_date) if f.end_date else start
        if not ((start and start >= today) or (end and end >= week_ago)):
            continue
        views, reach = await fetch_vk_post_stats(f.vk_post_url, db)
        if views is not None or reach is not None:
            stats.append((f.name, views or 0, reach or 0))
    stats.sort(key=lambda x: x[1], reverse=True)
    return [f"{name}: {views}, {reach}" for name, views, reach in stats]


async def handle_status(
    message: types.Message, db: Database, bot: Bot, app: web.Application
):
    parts = (message.text or "").split()
    if len(parts) > 1 and parts[1].isdigit():
        await send_job_status(message.chat.id, int(parts[1]), db, bot)
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    uptime = _time.time() - START_TIME
    qlen = add_event_queue.qsize()
    worker_task = app.get("add_event_worker")
    alive = "no"
    if isinstance(worker_task, asyncio.Task) and not worker_task.done():
        alive = "yes"
    last_dequeue = (
        f"{int(_time.monotonic() - _ADD_EVENT_LAST_DEQUEUE_TS)}s"
        if _ADD_EVENT_LAST_DEQUEUE_TS
        else "never"
    )
    jobs = list(JOB_HISTORY)[-5:]
    lines = [
        f"uptime: {int(uptime)}s",
        f"queue_len: {qlen}",
        f"worker_alive: {alive}",
        f"last_dequeue_ago: {last_dequeue}",
    ]
    if jobs:
        lines.append("last_jobs:")
        for j in reversed(jobs):
            when = j["when"].strftime("%H:%M:%S")
            lines.append(f"- {j['id']} {when} {j['status']} {j['took_ms']}ms")
    if LAST_RUN_ID:
        lines.append(f"last_run_id: {LAST_RUN_ID}")
    await bot.send_message(message.chat.id, "\n".join(lines))


async def handle_festivals_fix_nav(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    run_id = uuid.uuid4().hex
    logging.info(
        "festivals_fix_nav start", extra={"run_id": run_id, "user": message.from_user.id}
    )
    async with page_lock["festivals-index"]:
        await message.answer("Пересобираю навигацию и лендинг…")
        pages = changed = duplicates_removed = 0
        try:
            pages, changed, duplicates_removed, _ = await festivals_fix_nav(db, bot)
            status, url = await rebuild_festivals_index_if_needed(db, force=True)
        except Exception as e:
            status = f"ошибка: {e}"
            url = ""
        landing_line = f"Лендинг: {status}" + (f" {url}" if url else "")
        await message.answer(
            f"Готово. pages:{pages}, changed:{changed}, duplicates_removed:{duplicates_removed}\n{landing_line}"
        )


async def handle_ics_fix_nav(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    parts = (message.text or "").split()
    month = parts[1] if len(parts) > 1 else None
    count = await ics_fix_nav(db, month)
    await message.answer(f"Готово. события: {count}")


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


async def handle_backfill_topics(
    message: types.Message, db: Database, bot: Bot
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return

    parts = (message.text or "").split()
    days = 90
    if len(parts) > 1:
        try:
            days = int(parts[1])
        except Exception:
            await bot.send_message(message.chat.id, "Usage: /backfill_topics [days]")
            return
        if days < 0:
            await bot.send_message(message.chat.id, "Usage: /backfill_topics [days]")
            return

    today = date.today()
    end_date = today + timedelta(days=days)
    start_iso = today.isoformat()
    end_iso = end_date.isoformat()

    logging.info(
        "backfill_topics.start user_id=%s days=%s start=%s end=%s",
        message.from_user.id,
        days,
        start_iso,
        end_iso,
    )

    processed = 0
    updated = 0
    skipped = 0
    total = 0
    async with db.get_session() as session:
        stmt = (
            select(Event)
            .where(Event.date >= start_iso)
            .where(Event.date <= end_iso)
            .order_by(Event.date, Event.time, Event.id)
        )
        events = (await session.execute(stmt)).scalars().all()
        total = len(events)
        for event in events:
            if getattr(event, "topics_manual", False):
                skipped += 1
                continue

            processed += 1
            previous_topics = list(getattr(event, "topics", []) or [])
            original_manual = bool(getattr(event, "topics_manual", False))
            try:
                new_topics_raw = await classify_event_topics(event)
            except Exception:
                logging.exception(
                    "backfill_topics.classify_failed event_id=%s", getattr(event, "id", None)
                )
                continue

            seen: set[str] = set()
            normalized_topics: list[str] = []
            for topic in new_topics_raw:
                canonical = normalize_topic_identifier(topic)
                if canonical is None or canonical in seen:
                    continue
                seen.add(canonical)
                normalized_topics.append(canonical)

            event.topics = normalized_topics
            event.topics_manual = False

            if normalized_topics != previous_topics or original_manual:
                updated += 1
                session.add(event)

        if updated:
            await session.commit()

    logging.info(
        "backfill_topics.summary total=%s processed=%s updated=%s skipped=%s",
        total,
        processed,
        updated,
        skipped,
    )
    summary = (
        f"Backfilled topics {start_iso}..{end_iso} (days={days}): "
        f"processed={processed}, updated={updated}, skipped={skipped}"
    )
    await bot.send_message(message.chat.id, summary)


async def handle_queue_reap(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    args = shlex.split(message.text or "")[1:]
    parser = argparse.ArgumentParser(prog="/queue_reap", add_help=False)
    parser.add_argument("--type")
    parser.add_argument("--ym")
    parser.add_argument("--key-prefix")
    parser.add_argument("--status", default="running")
    parser.add_argument("--older-than")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--action", choices=["fail", "requeue"])
    parser.add_argument("--apply", action="store_true")
    try:
        opts = parser.parse_args(args)
    except Exception:
        await bot.send_message(message.chat.id, "Invalid arguments")
        return
    key_prefix = opts.key_prefix
    if not key_prefix and opts.type and opts.ym:
        key_prefix = f"{opts.type}:{opts.ym}"
    if not key_prefix and opts.type:
        key_prefix = f"{opts.type}:"
    older_sec = 0
    if opts.older_than:
        s = opts.older_than
        mult = 60
        if s.endswith("h"):
            mult = 3600
        elif s.endswith("d"):
            mult = 86400
        value = int(s[:-1]) if s[-1] in "mhd" else int(s)
        older_sec = value * mult
    now = datetime.now(timezone.utc)
    async with db.get_session() as session:
        stmt = select(JobOutbox).where(JobOutbox.status == JobStatus(opts.status))
        if key_prefix:
            stmt = stmt.where(JobOutbox.coalesce_key.like(f"{key_prefix}%"))
        if older_sec:
            thresh = now - timedelta(seconds=older_sec)
            stmt = stmt.where(JobOutbox.updated_at < thresh)
        stmt = stmt.order_by(JobOutbox.updated_at).limit(opts.limit)
        jobs = (await session.execute(stmt)).scalars().all()
    lines: list[str] = []
    header = "[DRY-RUN] " if not opts.apply else ""
    header += f"candidates={len(jobs)} status={opts.status}"
    if opts.older_than:
        header += f" older-than={opts.older_than}"
    if key_prefix:
        header += f" key-prefix={key_prefix}"
    lines.append(header)
    for idx, j in enumerate(jobs, 1):
        key = j.coalesce_key or f"{j.task.value}:{j.event_id}"
        started = j.updated_at.replace(microsecond=0).isoformat()
        delta = now - j.updated_at
        days, rem = divmod(int(delta.total_seconds()), 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts: list[str] = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if not days and minutes:
            parts.append(f"{minutes}m")
        age = "".join(parts) or "0m"
        lines.append(
            f"{idx}) id={j.id} key={key} owner_eid={j.event_id} started={started} age={age}"
        )
    if opts.apply and opts.action and jobs:
        async with db.get_session() as session:
            for j in jobs:
                obj = await session.get(JobOutbox, j.id)
                if not obj:
                    continue
                if opts.action == "fail":
                    obj.status = JobStatus.error
                    obj.last_error = "reaped_by_admin"
                else:
                    obj.status = JobStatus.pending
                    obj.attempts = 0
                    obj.last_error = None
                    obj.next_run_at = now
                obj.updated_at = now
                session.add(obj)
            await session.commit()
        lines.append(f"applied {opts.action} to {len(jobs)}")
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
        fest_landing = await collect_festivals_landing_stats(db)
        fest_tg = await collect_festival_telegraph_stats(db)
        fest_vk = await collect_festival_vk_stats(db)
        if fest_landing:
            lines.append("")
            lines.append(fest_landing)
        if fest_tg:
            lines.append("")
            lines.append("Фестивали (телеграм)")
            lines.extend(fest_tg)
        if fest_vk:
            lines.append("")
            lines.append("Фестивали (Вк) (просмотров, пользователи)")
            lines.extend(fest_vk)
    usage_snapshot = _get_four_o_usage_snapshot()
    usage_models = usage_snapshot.get("models", {})
    lines.extend(
        [
            f"Tokens gpt-4o: {usage_models.get('gpt-4o', 0)}",
            f"Tokens gpt-4o-mini: {usage_models.get('gpt-4o-mini', 0)}",
        ]
    )
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


def _coerce_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(stripped)
    return value


async def handle_tourist_export(message: types.Message, db: Database, bot: Bot) -> None:
    args = shlex.split(message.text or "")[1:]
    parser = argparse.ArgumentParser(prog="/tourist_export", add_help=False)
    parser.add_argument("--period")
    try:
        opts, extra = parser.parse_known_args(args)
    except SystemExit:
        await bot.send_message(message.chat.id, "Invalid arguments")
        return
    period_arg = opts.period
    if not period_arg and extra:
        token = extra[0]
        if token.startswith("period="):
            period_arg = token.split("=", 1)[1]
        else:
            period_arg = token
    start_date, end_date = (None, None)
    if period_arg:
        start_date, end_date = parse_period_range(period_arg)
        if not start_date and not end_date:
            await bot.send_message(message.chat.id, "Invalid period")
            return

    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not _user_can_label_event(user):
            await bot.send_message(message.chat.id, "Not authorized")
            return

        logger.info(
            "tourist_export.request user=%s period=%s",
            message.from_user.id,
            period_arg or "",
        )

        pragma_rows = await session.execute(text("PRAGMA table_info('event')"))
        event_columns = [row[1] for row in pragma_rows]
        base_fields = [
            "id",
            "title",
            "description",
            "festival",
            "date",
            "end_date",
            "time",
            "location_name",
            "location_address",
            "city",
            "ticket_price_min",
            "ticket_price_max",
            "ticket_link",
            "event_type",
            "emoji",
            "is_free",
            "pushkin_card",
            "telegraph_url",
            "source_post_url",
            "source_vk_post_url",
            "ics_url",
            "topics",
            "photo_urls",
        ]
        available_fields = [col for col in base_fields if col in event_columns]
        tourist_fields = [col for col in event_columns if col.startswith("tourist_")]
        seen: set[str] = set()
        selected_columns: list[str] = []
        for col in available_fields + tourist_fields:
            if col in seen:
                continue
            seen.add(col)
            selected_columns.append(col)
        if "id" not in seen and "id" in event_columns:
            selected_columns.insert(0, "id")
        if not selected_columns:
            await bot.send_message(message.chat.id, "No exportable fields")
            return

        query = text("SELECT {} FROM event".format(", ".join(selected_columns)))
        rows = await session.execute(query)
        records: list[dict[str, Any]] = []
        for row in rows:
            mapping = dict(row._mapping)
            start = parse_iso_date(str(mapping.get("date", "") or ""))
            end = parse_iso_date(str(mapping.get("end_date", "") or ""))
            if not end:
                end = start
            if start_date and end and end < start_date:
                continue
            if end_date and start and start > end_date:
                continue
            record: dict[str, Any] = {}
            for col in selected_columns:
                record[col] = _coerce_jsonish(mapping.get(col))
            records.append({
                "_sort": (
                    start or date.min,
                    mapping.get("id") or 0,
                ),
                "data": record,
            })

    logger.info(
        "tourist_export.start user=%s period=%s count=%s",
        message.from_user.id,
        period_arg or "",
        len(records),
    )

    if not records:
        await bot.send_message(message.chat.id, "No events found")
        return

    records.sort(key=lambda item: item["_sort"])
    lines = [json.dumps(item["data"], ensure_ascii=False) for item in records]
    payload = "\n".join(lines).encode("utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename_bits = ["tourist_export", timestamp]
    if start_date:
        filename_bits.insert(1, start_date.isoformat())
    if end_date and (not start_date or end_date != start_date):
        filename_bits.insert(2 if start_date else 1, end_date.isoformat())
    filename = "_".join(filename_bits) + ".jsonl"

    max_buffer = 45 * 1024 * 1024
    if len(payload) <= max_buffer:
        file = types.BufferedInputFile(payload, filename=filename)
        await bot.send_document(message.chat.id, file)
    else:
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".jsonl") as tmp:
                tmp.write(payload)
                tmp_path = tmp.name
            file = types.FSInputFile(tmp_path, filename=filename)
            await bot.send_document(message.chat.id, file)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(Exception):
                    os.remove(tmp_path)

    logger.info(
        "tourist_export.done user=%s count=%s bytes=%s",
        message.from_user.id,
        len(records),
        len(payload),
    )


async def handle_telegraph_fix_author(message: types.Message, db: Database, bot: Bot):
    await bot.send_message(
        message.chat.id,
        "Начинаю проставлять автора на всех Telegraph-страницах…",
    )
    token = get_telegraph_token()
    if not token:
        await bot.send_message(message.chat.id, "Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    pages: list[tuple[str, str]] = []
    async with db.get_session() as session:
        result = await session.execute(
            select(Event.title, Event.telegraph_path).where(
                Event.telegraph_path.is_not(None)
            )
        )
        pages.extend(result.all())
        result = await session.execute(
            select(Festival.name, Festival.telegraph_path).where(
                Festival.telegraph_path.is_not(None)
            )
        )
        pages.extend(result.all())
        result = await session.execute(select(MonthPage))
        for mp in result.scalars().all():
            pages.append((f"Month {mp.month}", mp.path))
            if mp.path2:
                pages.append((f"Month {mp.month} (2)", mp.path2))
        result = await session.execute(select(WeekendPage.start, WeekendPage.path))
        pages.extend([(f"Weekend {s}", p) for s, p in result.all()])

    updated: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    start = _time.perf_counter()
    for title, path in pages:
        try:
            await telegraph_edit_page(tg, path, title=title, caller="festival_build")
            updated.append((title, path))
        except Exception as e:  # pragma: no cover - network errors
            errors.append((title, str(e)))
        await asyncio.sleep(random.uniform(0.7, 1.2))
    dur = _time.perf_counter() - start
    lines = [
        f"Готово за {dur:.1f}с. Обновлено: {len(updated)}, ошибок: {len(errors)}"
    ]
    lines += [f"✓ {t} — https://telegra.ph/{p}" for t, p in updated[:50]]
    if errors:
        lines.append("\nОшибки:")
        lines += [f"✗ {t} — {err}" for t, err in errors[:50]]
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
        topics_meta: tuple[list[str], int, str | None, bool] | None = None
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
            elif field == "ticket_link":
                if value == "":
                    setattr(event, field, None)
                elif is_tg_folder_link(value):
                    await bot.send_message(
                        message.chat.id,
                        "Это ссылка на папку Telegram, не на регистрацию",
                    )
                    return
                else:
                    setattr(event, field, value)
                event.vk_ticket_short_url = None
                event.vk_ticket_short_key = None
            else:
                setattr(event, field, value)
        if field in {"title", "description", "source_text"} and not event.topics_manual:
            topics_meta = await assign_event_topics(event)
        await session.commit()
        if topics_meta:
            topics_list, text_len, error_text, manual_flag = topics_meta
            if manual_flag:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s manual=True",
                    event.id,
                    text_len,
                    topics_list,
                )
            elif error_text:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s error=%s",
                    event.id,
                    text_len,
                    topics_list,
                    error_text,
                )
            else:
                logging.info(
                    "event_topics_classify eid=%s text_len=%d topics=%s",
                    event.id,
                    text_len,
                    topics_list,
                )
        new_date = event.date.split("..", 1)[0]
        new_month = new_date[:7]
        new_fest = event.festival
    results = await schedule_event_update_tasks(db, event)
    await publish_event_progress(event, db, bot, message.chat.id, results)
    if field == "city":
        page_url = None
        async with db.get_session() as session:
            page = await session.get(MonthPage, new_month)
            page_url = page.url if page else None
        markup = None
        if page_url:
            label = f"Открыть {month_name_prepositional(new_month)}"
            markup = types.InlineKeyboardMarkup(
                inline_keyboard=[[types.InlineKeyboardButton(text=label, url=page_url)]]
            )
        await bot.send_message(
            message.chat.id,
            f"Город обновлён: {event.city}",
            reply_markup=markup,
        )
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
    add_event_sessions[message.from_user.id] = "event"
    logging.info(
        "handle_add_event_start session opened for user %s", message.from_user.id
    )
    await bot.send_message(message.chat.id, "Send event text and optional photo")


async def handle_add_festival_start(message: types.Message, db: Database, bot: Bot):
    """Initiate festival creation via the menu."""
    logging.info("handle_add_festival_start from user %s", message.from_user.id)
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    add_event_sessions[message.from_user.id] = "festival"
    logging.info(
        "handle_add_festival_start session opened for user %s",
        message.from_user.id,
    )
    await bot.send_message(message.chat.id, "Пришлите текст фестиваля…")


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


async def handle_vk_command(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        await bot.send_message(message.chat.id, "Access denied")
        return
    if not (VK_USER_TOKEN or VK_TOKEN or VK_TOKEN_AFISHA):
        await bot.send_message(message.chat.id, "VK token not configured")
        return
    buttons = [
        [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
        [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
        [
            types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
            types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
        ],
    ]
    markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
    await bot.send_message(message.chat.id, "VK мониторинг", reply_markup=markup)


async def handle_vk_add_start(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not (user and user.is_superadmin):
        await bot.send_message(message.chat.id, "Access denied")
        return
    vk_add_source_sessions.add(message.from_user.id)
    await bot.send_message(
        message.chat.id,
        "Отправьте ссылку или скриннейм, опционально локацию и время через |",
    )


async def handle_vk_add_message(message: types.Message, db: Database, bot: Bot) -> None:
    if message.from_user.id not in vk_add_source_sessions:
        return
    vk_add_source_sessions.discard(message.from_user.id)
    text = (message.text or "").strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if not parts:
        await bot.send_message(message.chat.id, "Пустой ввод")
        return
    screen = parts[-1]
    location = None
    default_time = None
    default_ticket_link = None
    for p in parts[:-1]:
        if re.match(r"^\d{1,2}:\d{2}$", p):
            default_time = p if len(p.split(":")[0]) == 2 else f"0{p}"
        elif p.startswith("http://") or p.startswith("https://"):
            default_ticket_link = p
        else:
            location = p
    try:
        gid, name, screen_name = await vk_resolve_group(screen)
    except Exception as e:
        logging.exception("vk_resolve_group failed")
        await bot.send_message(
            message.chat.id,
            "Не удалось определить сообщество.\n"
            "Проверьте ссылку/скриннейм (пример: https://vk.com/muzteatr39).\n"
            f"Технические детали: {e}.",
        )
        return
    async with db.raw_conn() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO vk_source(group_id, screen_name, name, location, default_time, default_ticket_link) VALUES(?,?,?,?,?,?)",
            (gid, screen_name, name, location, default_time, default_ticket_link),
        )
        await conn.commit()
    extra = []
    if location:
        extra.append(location)
    if default_time:
        extra.append(default_time)
    if default_ticket_link:
        extra.append(default_ticket_link)
    suffix = f" — {', '.join(extra)}" if extra else ""
    await bot.send_message(
        message.chat.id,
        f"Добавлено: {name} (vk.com/{screen_name}){suffix}",
    )


async def _fetch_vk_sources(
    db: Database,
) -> list[tuple[int, int, str, str, str | None, str | None, str | None, str | None]]:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            """
            SELECT
                s.id,
                s.group_id,
                s.screen_name,
                s.name,
                s.location,
                s.default_time,
                s.default_ticket_link,
                c.updated_at
            FROM vk_source AS s
            LEFT JOIN vk_crawl_cursor AS c ON c.group_id = s.group_id
            ORDER BY s.id
            """
        )
        rows = await cursor.fetchall()
    return rows


VK_SOURCES_PER_PAGE = 10
VK_STATUS_LABELS: Sequence[tuple[str, str]] = (
    ("pending", "Pending"),
    ("skipped", "Skipped"),
    ("imported", "Imported"),
    ("rejected", "Rejected"),
)


def _zero_vk_status_counts() -> dict[str, int]:
    return {key: 0 for key, _ in VK_STATUS_LABELS}


async def _fetch_vk_inbox_counts(db: Database) -> dict[int, dict[str, int]]:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT group_id, status, COUNT(*) FROM vk_inbox GROUP BY group_id, status"
        )
        rows = await cursor.fetchall()
    counts: dict[int, dict[str, int]] = {}
    for gid, status, amount in rows:
        bucket = counts.get(gid)
        if bucket is None:
            bucket = _zero_vk_status_counts()
            counts[gid] = bucket
        if status in bucket:
            bucket[status] = amount
    return counts


async def handle_vk_list(
    message: types.Message,
    db: Database,
    bot: Bot,
    edit: types.Message | None = None,
    page: int = 1,
) -> None:
    rows = await _fetch_vk_sources(db)
    if not rows:
        if edit:
            await edit.edit_text("Список пуст")
        else:
            await bot.send_message(message.chat.id, "Список пуст")
        return
    total_pages = max(1, (len(rows) + VK_SOURCES_PER_PAGE - 1) // VK_SOURCES_PER_PAGE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * VK_SOURCES_PER_PAGE
    end = start + VK_SOURCES_PER_PAGE
    page_rows = rows[start:end]
    inbox_counts = await _fetch_vk_inbox_counts(db)
    page_items: list[
        tuple[
            int,
            tuple[int, int, str, str, str | None, str | None, str | None, str | None],
            dict[str, int],
        ]
    ] = []
    for offset, row in enumerate(page_rows, start=start + 1):
        rid, gid, screen, name, loc, dtime, default_ticket_link, updated_at = row
        counts = inbox_counts.get(gid)
        if counts is None:
            counts = _zero_vk_status_counts()
        else:
            counts = dict(counts)
        page_items.append((offset, row, counts))

    if page_items:
        count_widths = {}
        for key, label in VK_STATUS_LABELS:
            max_value_len = max(len(str(item[2][key])) for item in page_items)
            base_width = max(len(label), max_value_len)
            count_widths[key] = max(1, math.ceil(base_width * 1.87))
    else:
        count_widths = {}
        for key, label in VK_STATUS_LABELS:
            base_width = len(label)
            count_widths[key] = max(1, math.ceil(base_width * 1.87))

    status_header_parts = [f" {label} " for _, label in VK_STATUS_LABELS]
    status_header_line = "|".join(status_header_parts)

    lines: list[str] = []
    buttons: list[list[types.InlineKeyboardButton]] = []
    for offset, row, counts in page_items:
        rid, gid, screen, name, loc, dtime, default_ticket_link, updated_at = row
        info_parts = [f"id={gid}"]
        if loc:
            info_parts.append(loc)
        if default_ticket_link:
            info_parts.append(f"билеты: {default_ticket_link}")
        info = ", ".join(info_parts)
        if updated_at:
            try:
                if isinstance(updated_at, (int, float)):
                    parsed_updated = datetime.fromtimestamp(updated_at, tz=timezone.utc)
                else:
                    parsed_updated = datetime.fromisoformat(updated_at)
                    if parsed_updated.tzinfo is None:
                        parsed_updated = parsed_updated.replace(tzinfo=timezone.utc)
                human_updated = parsed_updated.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                human_updated = str(updated_at)
        else:
            human_updated = "-"
        lines.append(
            f"{offset}. {name} (vk.com/{screen}) — {info}, типовое время: {dtime or '-'}, последняя проверка: {human_updated}"
        )
        value_parts = [
            f" {counts[key]:^{count_widths[key]}} "
            for key, _ in VK_STATUS_LABELS
        ]
        lines.append(status_header_line)
        lines.append("|".join(value_parts))
        buttons.append(
            [
                types.InlineKeyboardButton(
                    text=f"❌ {offset}", callback_data=f"vkdel:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"⚙️ {offset}", callback_data=f"vkset:{rid}:{page}"
                ),
            ]
        )
        buttons.append(
            [
                types.InlineKeyboardButton(
                    text=f"🕒 {offset}", callback_data=f"vkdt:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"🎟 {offset}", callback_data=f"vklink:{rid}:{page}"
                ),
                types.InlineKeyboardButton(
                    text=f"📍 {offset}", callback_data=f"vkloc:{rid}:{page}"
                ),
            ]
        )
        if counts["rejected"] > 0:
            buttons.append(
                [
                    types.InlineKeyboardButton(
                        text=f"🚫 Rejected: {counts['rejected']}",
                        callback_data=f"vkrejected:{gid}:{page}",
                    )
                ]
            )
    text = "\n".join(lines)
    if total_pages > 1:
        nav_row: list[types.InlineKeyboardButton] = []
        if page > 1:
            nav_row.append(
                types.InlineKeyboardButton(
                    text="◀️", callback_data=f"vksrcpage:{page - 1}"
                )
            )
        nav_row.append(
            types.InlineKeyboardButton(
                text=f"{page}/{total_pages}", callback_data=f"vksrcpage:{page}"
            )
        )
        if page < total_pages:
            nav_row.append(
                types.InlineKeyboardButton(
                    text="▶️", callback_data=f"vksrcpage:{page + 1}"
                )
            )
        buttons.append(nav_row)
    markup = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    if edit:
        await edit.edit_text(text, reply_markup=markup)
    else:
        await bot.send_message(message.chat.id, text, reply_markup=markup)


async def handle_vk_list_page_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    try:
        page = int(callback.data.split(":", 1)[1])
    except Exception:
        await callback.answer()
        return
    await handle_vk_list(callback.message, db, bot, edit=callback.message, page=page)
    await callback.answer()


async def handle_vk_delete_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        await conn.execute("DELETE FROM vk_source WHERE id=?", (vid,))
        await conn.commit()
    await callback.answer("Удалено")
    await handle_vk_list(callback.message, db, bot, edit=callback.message, page=page)


async def handle_vk_rejected_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":")
        group_id = int(parts[0])
    except Exception:
        await callback.answer()
        return

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, screen_name FROM vk_source WHERE group_id=?",
            (group_id,),
        )
        source_row = await cur.fetchone()
        await cur.close()
        cur = await conn.execute(
            """
            SELECT post_id
            FROM vk_inbox
            WHERE group_id=? AND status='rejected'
            ORDER BY
                COALESCE(created_at, '') DESC,
                id DESC
            LIMIT 30
            """,
            (group_id,),
        )
        rows = await cur.fetchall()
        await cur.close()

    if not rows:
        await callback.answer("Нет отклонённых постов", show_alert=True)
        return

    if source_row:
        name, screen = source_row
    else:
        name = None
        screen = None

    if name and screen:
        header = f"{name} (vk.com/{screen})"
    elif screen:
        header = f"vk.com/{screen}"
    elif name:
        header = name
    else:
        header = f"group {group_id}"

    links = [f"https://vk.com/wall-{group_id}_{post_id}" for (post_id,) in rows]
    message_text = "\n".join([f"🚫 Отклонённые посты — {header}"] + links)

    await bot.send_message(
        callback.message.chat.id,
        message_text,
        disable_web_page_preview=True,
    )
    await callback.answer()


async def handle_vk_settings_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location, default_time, default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    if not row:
        await callback.answer("Источник не найден", show_alert=True)
        return
    name, location, default_time, default_ticket_link = row
    lines = [f"{name}"]
    lines.append(f"Локация: {location or 'не указана'}")
    lines.append(f"Типовое время: {default_time or 'не установлено'}")
    lines.append(f"Ссылка на билеты: {default_ticket_link or 'не указана'}")
    lines.append("Используйте кнопки 🕒, 🎟 и 📍 для изменений.")
    if callback.message:
        await bot.send_message(callback.message.chat.id, "\n".join(lines))
    await callback.answer()


async def handle_vk_ticket_link_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_ticket_link_sessions[callback.from_user.id] = (
        VkDefaultTicketLinkSession(
            source_id=vid,
            page=page,
            message=callback.message,
        )
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, default_ticket_link FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: текущая ссылка на билеты — {current or 'не установлена'}. "
        "Отправьте ссылку, начинающуюся с http(s)://, или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_ticket_link_message(
    message: types.Message, db: Database, bot: Bot
) -> None:
    session = vk_default_ticket_link_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_link: str | None = None
    else:
        if not re.match(r"^https?://", text, re.IGNORECASE):
            await bot.send_message(
                message.chat.id,
                "Неверный формат. Укажите ссылку, начинающуюся с http(s)://, или '-' для удаления.",
            )
            vk_default_ticket_link_sessions[message.from_user.id] = session
            return
        new_link = text
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET default_ticket_link=? WHERE id=?",
            (new_link, vid),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT name FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_link:
        msg = f"Ссылка на билеты для {name} установлена: {new_link}"
    else:
        msg = f"Ссылка на билеты для {name} удалена"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def handle_vk_location_callback(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_location_sessions[callback.from_user.id] = VkDefaultLocationSession(
        source_id=vid,
        page=page,
        message=callback.message,
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: текущая локация — {current or 'не указана'}. "
        "Отправьте новую локацию или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_location_message(
    message: types.Message, db: Database, bot: Bot
) -> None:
    session = vk_default_location_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_location: str | None = None
    else:
        new_location = text
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET location=? WHERE id=?",
            (new_location, vid),
        )
        await conn.commit()
        cur = await conn.execute(
            "SELECT name FROM vk_source WHERE id=?",
            (vid,),
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_location:
        msg = f"Локация для {name} установлена: {new_location}"
    else:
        msg = f"Локация для {name} удалена"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def handle_vk_dtime_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    page = 1
    try:
        _, payload = callback.data.split(":", 1)
        parts = payload.split(":", 1)
        vid = int(parts[0])
        if len(parts) > 1:
            page = int(parts[1])
    except Exception:
        await callback.answer()
        return
    vk_default_time_sessions[callback.from_user.id] = VkDefaultTimeSession(
        source_id=vid,
        page=page,
        message=callback.message,
    )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, default_time FROM vk_source WHERE id=?", (vid,)
        )
        row = await cur.fetchone()
    name = row[0] if row else ""
    current = row[1] if row else None
    await bot.send_message(
        callback.message.chat.id,
        f"{name}: типовое время сейчас — {current or 'не установлено'}. "
        "Отправьте время в формате HH:MM или '-' для удаления.",
    )
    await callback.answer()


async def handle_vk_dtime_message(message: types.Message, db: Database, bot: Bot) -> None:
    session = vk_default_time_sessions.pop(message.from_user.id, None)
    if not session:
        return
    vid = session.source_id
    text = (message.text or "").strip()
    if text in {"", "-"}:
        new_time: str | None = None
    else:
        if not re.match(r"^\d{1,2}:\d{2}$", text):
            await bot.send_message(
                message.chat.id,
                "Неверный формат. Используйте HH:MM или '-' для удаления.",
            )
            return
        new_time = text if len(text.split(":")[0]) == 2 else f"0{text}"
    async with db.raw_conn() as conn:
        await conn.execute(
            "UPDATE vk_source SET default_time=? WHERE id=?", (new_time, vid)
        )
        await conn.commit()
        cur = await conn.execute("SELECT name FROM vk_source WHERE id=?", (vid,))
        row = await cur.fetchone()
    name = row[0] if row else ""
    if new_time:
        msg = f"Типовое время для {name} установлено: {new_time}"
    else:
        msg = f"Типовое время для {name} удалено"
    await bot.send_message(message.chat.id, msg)
    if session.message:
        await handle_vk_list(
            session.message,
            db,
            bot,
            edit=session.message,
            page=session.page,
        )


async def send_vk_tmp_post(chat_id: int, batch: str, idx: int, total: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT text, photos, url FROM vk_tmp_post WHERE batch=? ORDER BY date DESC, post_id DESC LIMIT 1 OFFSET ?",
            (batch, idx),
        )
        row = await cursor.fetchone()
    if not row:
        await bot.send_message(chat_id, "Это был последний пост.")
        return
    text, photos_json, url = row
    photos = json.loads(photos_json) if photos_json else []
    if len(photos) >= 2:
        media = [types.InputMediaPhoto(media=p) for p in photos[:10]]
        try:
            await bot.send_media_group(chat_id, media)
        except Exception:
            await bot.send_photo(chat_id, photos[0])
    elif len(photos) == 1:
        await bot.send_photo(chat_id, photos[0])
    msg = (text or "").strip()
    if len(msg) > 3500:
        msg = msg[:3500] + "…"
    msg = msg + f"\n\n{url}"
    if idx + 1 < total:
        markup = types.InlineKeyboardMarkup(
            inline_keyboard=[[types.InlineKeyboardButton(text="Следующее ▶️", callback_data=f"vknext:{batch}:{idx+1}")]]
        )
        await bot.send_message(chat_id, msg, reply_markup=markup)
    else:
        await bot.send_message(chat_id, msg)
        await bot.send_message(chat_id, "Это был последний пост.")


async def handle_vk_check(message: types.Message, db: Database, bot: Bot) -> None:
    """Start VK inbox review."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await message.answer("Not authorized")
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT batch_id FROM vk_review_batch
            WHERE operator_id=? AND finished_at IS NULL
            ORDER BY started_at DESC LIMIT 1
            """,
            (message.from_user.id,),
        )
        row = await cur.fetchone()
    if row:
        batch_id = row[0]
    else:
        batch_id = f"{int(unixtime.time())}:{message.from_user.id}"
        async with db.raw_conn() as conn:
            await conn.execute(
                "INSERT INTO vk_review_batch(batch_id, operator_id, months_csv) VALUES(?,?,?)",
                (batch_id, message.from_user.id, ""),
            )
            await conn.commit()
    await _vkrev_show_next(message.chat.id, batch_id, message.from_user.id, db, bot)


async def handle_vk_crawl_now(message: types.Message, db: Database, bot: Bot) -> None:
    """Manually trigger VK crawling."""
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or not user.is_superadmin:
            await bot.send_message(message.chat.id, "Not authorized")
            return
    text = message.text or ""
    try:
        tokens = shlex.split(text)
    except ValueError:
        await bot.send_message(
            message.chat.id,
            "Не удалось разобрать параметры команды",
        )
        return

    forced_backfill = False
    requested_backfill_days: int | None = None
    error_message: str | None = None

    for token in tokens:
        if not token.startswith("--backfill-days"):
            continue
        forced_backfill = True
        if token == "--backfill-days":
            error_message = "Используйте синтаксис --backfill-days=N"
            break
        if token.startswith("--backfill-days="):
            value = token.split("=", 1)[1]
            if not value:
                error_message = "Значение для --backfill-days отсутствует"
                break
            try:
                requested_backfill_days = int(value)
            except ValueError:
                error_message = "Значение --backfill-days должно быть целым числом"
                break
        else:
            error_message = "Используйте синтаксис --backfill-days=N"
            break

    if error_message:
        await bot.send_message(message.chat.id, error_message)
        return

    if forced_backfill and requested_backfill_days is not None and requested_backfill_days < 1:
        await bot.send_message(
            message.chat.id,
            "Значение --backfill-days должно быть положительным",
        )
        return

    stats = await vk_intake.crawl_once(
        db,
        broadcast=True,
        bot=bot,
        force_backfill=forced_backfill,
        backfill_days=requested_backfill_days if forced_backfill else None,
    )
    q = stats.get("queue", {})
    forced_note = ""
    if stats.get("forced_backfill"):
        used_days = stats.get("backfill_days_used") or vk_intake.VK_CRAWL_BACKFILL_DAYS
        requested_days = stats.get("backfill_days_requested")
        forced_note = f", режим: принудительный бэкафилл до {used_days} дн."
        if (
            requested_days is not None
            and requested_days != used_days
        ):
            forced_note += f" (запрошено {requested_days})"
    msg = (
        f"Проверено {stats['groups_checked']} сообществ, "
        f"просмотрено {stats['posts_scanned']} постов, "
        f"совпало {stats['matches']}, "
        f"дубликатов {stats['duplicates']}, "
        f"добавлено {stats['added']}, "
        f"всего постов {stats['inbox_total']} "
        f"(в очереди: {q.get('pending',0)}, locked: {q.get('locked',0)}, "
        f"skipped: {q.get('skipped',0)}, imported: {q.get('imported',0)}, "
        f"rejected: {q.get('rejected',0)})"
        f", страниц на группу: {'/'.join(str(p) for p in stats.get('pages_per_group', []))}, "
        f"перекрытие: {stats.get('overlap_sec')} сек"
        f"{forced_note}"
    )
    await bot.send_message(message.chat.id, msg)


async def handle_vk_queue(message: types.Message, db: Database, bot: Bot) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await bot.send_message(message.chat.id, "Not authorized")
        return
    await vk_review.release_stale_locks(db)
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, COUNT(*) FROM vk_inbox GROUP BY status"
        )
        rows = await cur.fetchall()
    counts = {r[0]: r[1] for r in rows}
    lines = [
        f"pending: {counts.get('pending', 0)}",
        f"locked: {counts.get('locked', 0)}",
        f"skipped: {counts.get('skipped', 0)}",
        f"imported: {counts.get('imported', 0)}",
        f"rejected: {counts.get('rejected', 0)}",
    ]
    schedule_raw = os.getenv(
        "VK_CRAWL_TIMES_LOCAL", "05:15,09:15,13:15,17:15,21:15,22:45"
    )
    schedule_times = [part.strip() for part in schedule_raw.split(",") if part.strip()]
    schedule_line = ", ".join(schedule_times)
    schedule_tz = os.getenv("VK_CRAWL_TZ")
    if schedule_tz:
        schedule_line = f"{schedule_line} ({schedule_tz})"
    if schedule_line:
        lines.insert(0, f"Обновление базы: {schedule_line}")
    markup = types.ReplyKeyboardMarkup(
        keyboard=[[types.KeyboardButton(text=VK_BTN_CHECK_EVENTS)]],
        resize_keyboard=True,
    )
    await bot.send_message(message.chat.id, "\n".join(lines), reply_markup=markup)


async def handle_vk_requeue_imported(
    message: types.Message, db: Database, bot: Bot
) -> None:
    parts = message.text.split()
    n = 1
    if len(parts) > 1:
        try:
            n = int(parts[1])
        except ValueError:
            await bot.send_message(message.chat.id, "Usage: /vk_requeue_imported [N]")
            return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
    if not user:
        await bot.send_message(message.chat.id, "Not authorized")
        return
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            """
            SELECT id FROM vk_inbox
            WHERE status='imported' AND review_batch IN (
                SELECT batch_id FROM vk_review_batch WHERE operator_id=?
            )
            ORDER BY id DESC LIMIT ?
            """,
            (message.from_user.id, n),
        )
        rows = await cur.fetchall()
        ids = [r[0] for r in rows]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            await conn.execute(
                f"""
                UPDATE vk_inbox
                SET status='pending', imported_event_id=NULL, review_batch=NULL,
                    locked_by=NULL, locked_at=NULL
                WHERE id IN ({placeholders})
                """,
                ids,
            )
            await conn.commit()
    await bot.send_message(message.chat.id, f"Requeued {len(ids)} item(s)")

async def _vkrev_queue_size(db: Database) -> int:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM vk_inbox WHERE status IN ('pending','skipped')",
        )
        (cnt,) = await cur.fetchone()
    return cnt


async def _vkrev_fetch_photos(group_id: int, post_id: int, db: Database, bot: Bot) -> list[str]:
    token: str | None = VK_SERVICE_TOKEN
    token_kind = "service"
    if not token:
        token = _vk_user_token()
        token_kind = "user"
    if not token:
        return []
    try:
        data = await _vk_api(
            "wall.getById",
            {"posts": f"-{group_id}_{post_id}"},
            db,
            bot,
            token=token,
            token_kind=token_kind,
            skip_captcha=True,
        )
    except VKAPIError as e:  # pragma: no cover
        logging.error(
            "wall.getById failed gid=%s post=%s actor=%s token=%s code=%s msg=%s",
            group_id,
            post_id,
            e.actor,
            e.token,
            e.code,
            e.message,
        )
        return []
    except Exception as e:  # pragma: no cover
        logging.error("wall.getById failed gid=%s post=%s: %s", group_id, post_id, e)
        return []
    response = data.get("response") if isinstance(data, dict) else data
    def best_url(sizes: list[dict]) -> str:
        if not sizes:
            return ""
        best = max(sizes, key=lambda s: s.get("width", 0) * s.get("height", 0))
        return best.get("url") or best.get("src", "")

    if isinstance(response, dict):
        items = response.get("items") or []
    else:
        items = response or []
    photos: list[str] = []
    seen: set[str] = set()

    def process_atts(atts: list[dict], source: str) -> bool:
        counts = {"photo": 0, "link": 0, "video_thumbs": 0, "doc": 0}
        for att in atts or []:
            url = ""
            if att.get("type") == "photo":
                url = best_url(att["photo"].get("sizes", []))
                if url:
                    counts["photo"] += 1
            elif att.get("type") == "link":
                sizes = ((att.get("link") or {}).get("photo") or {}).get("sizes", [])
                url = best_url(sizes)
                if url:
                    counts["link"] += 1
            elif att.get("type") == "video":
                images = att["video"].get("first_frame") or att["video"].get("image", [])
                url = best_url(images)
                if url:
                    counts["video_thumbs"] += 1
            elif att.get("type") == "doc":
                sizes = (
                    (att["doc"].get("preview") or {})
                    .get("photo", {})
                    .get("sizes", [])
                )
                url = best_url(sizes)
                if url:
                    counts["doc"] += 1
            if url and url not in seen:
                seen.add(url)
                photos.append(url)
                if len(photos) >= 10:
                    break
        total = sum(counts.values())
        logging.info(
            "found_photos=%s (photo=%s, link=%s, video_thumbs=%s, doc=%s) source=%s",
            total,
            counts["photo"],
            counts["link"],
            counts["video_thumbs"],
            counts["doc"],
            source,
        )
        return len(photos) >= 10

    for item in items:
        copy = (item.get("copy_history") or [{}])[0].get("attachments")
        if copy and process_atts(copy, "copy_history"):
            break
        if len(photos) >= 10:
            break
        atts = item.get("attachments") or []
        if process_atts(atts, "attachments"):
            break

    if not photos:
        logging.info("no media found for -%s_%s", group_id, post_id)
    return photos


def _vkrev_collect_photo_ids(items: list[dict], max_photos: int) -> list[str]:
    photos: list[str] = []
    seen: set[str] = set()

    def process(atts: list[dict]) -> bool:
        for att in atts or []:
            if att.get("type") != "photo":
                continue
            ph = att.get("photo", {})
            owner = ph.get("owner_id")
            pid = ph.get("id")
            if owner is None or pid is None:
                continue
            key = f"{owner}_{pid}"
            access = ph.get("access_key")
            if access:
                key = f"{key}_{access}"
            if key in seen:
                continue
            seen.add(key)
            photos.append("photo" + key)
            if len(photos) >= max_photos:
                return True
        return False

    for item in items:
        copy = (item.get("copy_history") or [{}])[0].get("attachments")
        if process(copy):
            break
        if process(item.get("attachments") or []):
            break
    return photos


async def build_short_vk_text(
    event: Event,
    source_text: str,
    max_sentences: int = 4,
    *,
    poster_texts: Sequence[str] | None = None,
) -> str:
    text = (source_text or "").strip()
    fallback_from_title = False
    if not text:
        desc = (event.description or "").strip()
        if desc:
            text = desc
        else:
            title = (event.title or "").strip()
            text = title
            fallback_from_title = True
    if not text:
        return ""
    if fallback_from_title:
        return text

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")

    def _truncate_sentences(source: str, limit: int) -> str:
        if limit <= 0:
            return ""
        paragraphs: list[str] = []
        remaining = limit
        for block in source.split("\n\n"):
            paragraph = block.strip()
            if not paragraph or remaining <= 0:
                continue
            sentences = [part.strip() for part in sentence_splitter.split(paragraph) if part.strip()]
            if not sentences:
                continue
            selected: list[str] = []
            for sentence in sentences:
                if remaining <= 0:
                    break
                selected.append(sentence)
                remaining -= 1
            if selected:
                paragraphs.append(" ".join(selected))
        return "\n\n".join(paragraphs).strip()

    def _fallback_summary() -> str:
        fallback = _truncate_sentences(text, min(max_sentences, 2))
        return fallback or text

    extra_blocks = [block.strip() for block in poster_texts or [] if block.strip()]
    prompt_text = text
    if extra_blocks:
        joined = "\n\n".join(extra_blocks)
        prompt_text = f"{prompt_text}\n\nДополнительный текст с афиш:\n{joined}"

    prompt = (
        "Сократи описание ниже без выдумок, сохраняя все важные детали "
        "и перечисленных ключевых участников, максимум до "
        f"{max_sentences} предложений. Разрешены эмодзи. "
        "Пиши дружелюбно и не добавляй прямых рекламных призывов (например, про покупку билетов). "
        "Сразу начинай с главной идеи — в первой строке не повторяй название события и не добавляй блок про дату, время, место или билеты. "
        "Название проекта или события можно упомянуть позже. "
        "Не повторяй дату, время и место события в абзацах — мы выводим их отдельными строками. "
        "Сделай первую фразу крючком, который вызывает любопытство: это может быть вопрос или интригующая деталь. "
        "Не используй фразу «Погрузитесь в мир» ни в каком виде. "
        f"Разбивай текст на абзацы для удобства чтения.\n\n{prompt_text}"
    )
    try:
        raw = await ask_4o(
            prompt,
            system_prompt=(
                "Ты сжимаешь текст фактически, без новых деталей и не упуская важные факты. "
                "Эмодзи допустимы. Делай текст читабельным и дружелюбным, разбивая его на абзацы. "
                "Не используй прямые рекламные формулировки, в том числе призывы покупать билеты. "
                "Сразу начинай с сути — в первой строке не повторяй название события и не добавляй блок про дату, время, место или билеты. "
                "Название проекта или события можно упомянуть позже. "
                "Не повторяй дату, время и место события в абзацах — они выводятся отдельно. "
                "Первая фраза должна быть крючком, вызывающим любопытство, и избегай фразы «Погрузитесь в мир»."
            ),
            max_tokens=400,
        )
    except Exception:
        return _fallback_summary()
    cleaned = raw.strip()
    if not cleaned:
        return _fallback_summary()

    banned_phrase_pattern = re.compile(r"погрузитесь в мир", re.IGNORECASE)

    def _remove_banned_sentences(value: str) -> str:
        if not banned_phrase_pattern.search(value):
            return value
        paragraphs: list[str] = []
        for block in value.split("\n\n"):
            sentences = [
                sentence.strip()
                for sentence in sentence_splitter.split(block)
                if sentence.strip()
            ]
            filtered = [
                sentence
                for sentence in sentences
                if not banned_phrase_pattern.search(sentence)
            ]
            if filtered:
                paragraphs.append(" ".join(filtered))
        return "\n\n".join(paragraphs).strip()

    def _ensure_curiosity_hook(value: str) -> str:
        stripped = value.lstrip()
        prefix = value[: len(value) - len(stripped)]
        if not stripped:
            return value
        match = re.search(r"^([^\n]*?[.!?])(\s|$)", stripped)
        if match:
            first_sentence = match.group(1).strip()
            separator = match.group(2) or ""
            remainder = separator + stripped[match.end():]
        else:
            first_sentence = stripped
            remainder = ""
        hook_prefixes = (
            "что если",
            "представьте",
            "знаете ли вы",
            "как насчет",
            "как насчёт",
            "готовы ли вы",
            "хотите узнать",
            "угадайте",
        )
        first_lower = first_sentence.casefold()
        has_hook = "?" in first_sentence or any(
            first_lower.startswith(prefix) for prefix in hook_prefixes
        )
        if has_hook:
            return prefix + stripped
        base = first_sentence.rstrip(".!?").strip()
        if not base:
            return prefix + stripped
        if len(base) > 1:
            body = base[0].lower() + base[1:]
        else:
            body = base.lower()
        new_first_sentence = f"Знаете ли вы, {body}?"
        remainder = remainder.lstrip()
        if remainder:
            if remainder.startswith("\n"):
                rebuilt = new_first_sentence + remainder
            else:
                rebuilt = new_first_sentence + " " + remainder
        else:
            rebuilt = new_first_sentence
        return prefix + rebuilt

    cleaned = _remove_banned_sentences(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return _fallback_summary()
    if banned_phrase_pattern.search(cleaned):
        cleaned = banned_phrase_pattern.sub("", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r" ?\n ?", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            return _fallback_summary()
    cleaned = _ensure_curiosity_hook(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return _fallback_summary()
    cleaned_lower = cleaned.casefold()
    if not cleaned:
        return _fallback_summary()
    if ("предостав" in cleaned_lower and "текст" in cleaned_lower) or (
        "provide" in cleaned_lower and "text" in cleaned_lower
    ):
        return _fallback_summary()
    summary = _truncate_sentences(cleaned, max_sentences)
    return summary or _fallback_summary()


VK_LOCATION_TAG_OVERRIDES: dict[str, str] = {
    "ицаэ": "#ИЦАЭ",
    "кгту": "#КГТУ",
    "коихм": "#КОИХМ",
}


VK_TOPIC_HASHTAGS: Mapping[str, str] = {
    "STANDUP": "#стендап",
    "QUIZ_GAMES": "#квиз",
    "OPEN_AIR": "#openair",
    "PARTIES": "#вечеринка",
    "CONCERTS": "#музыка",
    "MOVIES": "#кино",
    "EXHIBITIONS": "#искусство",
    "THEATRE": "#театр",
    "THEATRE_CLASSIC": "#классика",
    "THEATRE_MODERN": "#перфоманс",
    "LECTURES": "#лекция",
    "MASTERCLASS": "#мастеркласс",
    "PSYCHOLOGY": "#здоровье",
    "SCIENCE_POP": "#научпоп",
    "HANDMADE": "#маркет",
    "NETWORKING": "#митап",
    "ACTIVE": "#спорт",
    "HISTORICAL_IMMERSION": "#история",
    "FASHION": "#мода",
    "KIDS_SCHOOL": "#детям",
    "FAMILY": "#семье",
    "URBANISM": "#урбанистика",
    "KRAEVEDENIE_KALININGRAD_OBLAST": "#калининград",
}


async def build_short_vk_tags(
    event: Event, summary: str, used_type_hashtag: str | None = None
) -> list[str]:
    """Generate 5-7 hashtags for the short VK post."""
    day = int(event.date.split("-")[2])
    month = int(event.date.split("-")[1])
    month_name = MONTHS[month - 1]
    current_year = date.today().year
    tags: list[str] = []
    seen: set[str] = set()

    used_type_hashtag_normalized: str | None = None
    if used_type_hashtag:
        used_tag_clean = used_type_hashtag.strip()
        if used_tag_clean:
            if not used_tag_clean.startswith("#"):
                used_tag_clean = "#" + used_tag_clean.lstrip("#")
            used_type_hashtag_normalized = used_tag_clean.lower()
            seen.add(used_type_hashtag_normalized)

    def add_tag(tag: str) -> None:
        tag_clean = (tag or "").strip()
        if not tag_clean:
            return
        if not tag_clean.startswith("#"):
            tag_clean = "#" + tag_clean.lstrip("#")
        tag_lower = tag_clean.lower()
        if tag_lower in seen:
            return
        years = re.findall(r"\d{4}", tag_lower)
        for year_text in years:
            try:
                if int(year_text) < current_year:
                    return
            except ValueError:  # pragma: no cover
                continue
        tags.append(tag_clean)
        seen.add(tag_lower)

    add_tag(f"#{day}_{month_name}")
    add_tag(f"#{day}{month_name}")
    city = (event.city or "").strip()
    if city:
        normalized_city = re.sub(r"[^0-9a-zа-яё]+", "", city.lower())
        if normalized_city:
            add_tag(f"#{normalized_city}")
    seen_location_tokens: set[str] = set()
    for source in (event.location_name or "", event.location_address or ""):
        if not source:
            continue
        for token in re.findall(r"[А-ЯЁ]{2,}", source):
            normalized_token = token.lower()
            if normalized_token in seen_location_tokens:
                continue
            seen_location_tokens.add(normalized_token)
            tag = VK_LOCATION_TAG_OVERRIDES.get(normalized_token, f"#{token}")
            add_tag(tag)
    if event.event_type:
        raw_event_type = event.event_type.strip()
        if raw_event_type:
            event_type_lower = raw_event_type.casefold()
            normalized_event_type = re.sub(
                r"[^0-9a-zа-яё]+", "_", event_type_lower
            ).strip("_")
            if normalized_event_type:
                add_tag(f"#{normalized_event_type}")
            if re.search(r"[-–—]", raw_event_type):
                hyphen_free_variant = re.sub(
                    r"[^0-9a-zа-яё]", "", event_type_lower
                )
                if hyphen_free_variant:
                    hyphen_tag = f"#{hyphen_free_variant}"
                    if not (
                        used_type_hashtag_normalized
                        and hyphen_tag.lower() == used_type_hashtag_normalized
                    ):
                        add_tag(hyphen_tag)
    topic_values = getattr(event, "topics", None) or []
    for topic in topic_values:
        if len(tags) >= 7:
            break
        normalized_topic = (topic or "").strip().upper()
        if not normalized_topic:
            continue
        topic_tag = VK_TOPIC_HASHTAGS.get(normalized_topic)
        if not topic_tag:
            continue
        topic_tag_clean = topic_tag.strip()
        if not topic_tag_clean:
            continue
        if not topic_tag_clean.startswith("#"):
            topic_tag_clean = "#" + topic_tag_clean.lstrip("#")
        if (
            used_type_hashtag_normalized
            and topic_tag_clean.lower() == used_type_hashtag_normalized
        ):
            continue
        add_tag(topic_tag_clean)
    needed = 7 - len(tags)
    if needed > 0:
        prompt = (
            "Подбери ещё {n} коротких и актуальных хештегов "
            "для поста о событии. Используй русский язык, "
            "начинай каждый хештег с #, не добавляй пояснений. "
            "Добавь хештег с форматом события (например, #спектакль, #мастеркласс, #лекция). "
            "Не предлагай хештеги со старыми годами (раньше {current_year}).\n"
            "Название: {title}\nОписание: {desc}"
        ).format(
            n=needed,
            current_year=current_year,
            title=event.title,
            desc=summary,
        )
        try:
            raw = await ask_4o(
                prompt,
                system_prompt=(
                    "Ты подбираешь хештеги к событию, отвечай только хештегами. "
                    "Избегай устаревших годов и рекламных формулировок."
                ),
                max_tokens=60,
            )
            extra = re.findall(r"#[^\s#]+", raw.lower())
            for t in extra:
                add_tag(t)
                if len(tags) >= 7:
                    break
        except Exception:
            pass
    if len(tags) < 5:
        fallback = ["#афиша", "#кудапойти", "#событие", "#выходные", "#калининград"]
        for t in fallback:
            if len(tags) >= 5:
                break
            add_tag(t)
            if len(tags) >= 5:
                break
    return tags[:7]


async def build_short_vk_location(parts: Sequence[str]) -> str:
    cleaned_parts = [part.strip() for part in parts if part and part.strip()]
    if not cleaned_parts:
        return ""
    joined = ", ".join(cleaned_parts)
    prompt = (
        "Собери короткую и понятную формулировку адреса события для поста ВКонтакте. "
        "Используй все важные части, убери дубли и лишние слова. Не пиши ничего, кроме адреса, "
        "не добавляй слово «Локация» и эмодзи.\n"
        f"Части адреса: {joined}"
    )
    try:
        raw = await ask_4o(
            prompt,
            system_prompt=(
                "Ты формируешь краткую строку с адресом события. "
                "Верни только сам адрес без вводных слов и эмодзи."
            ),
            max_tokens=60,
        )
    except Exception:
        return joined
    location = raw.strip()
    if not location:
        return joined
    location = re.sub(r"\s+", " ", location)
    location = location.replace("📍", "").strip()
    location = re.sub(r"^[Лл]окация[:\-\s]*", "", location).strip()
    return location or joined


async def _vkrev_show_next(chat_id: int, batch_id: str, operator_id: int, db: Database, bot: Bot) -> None:
    post = await vk_review.pick_next(db, operator_id, batch_id)
    if not post:
        buttons = [
            [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
            [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
            [
                types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
                types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
            ],
        ]
        markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
        await bot.send_message(chat_id, "Очередь пуста", reply_markup=markup)
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                """
                SELECT batch_id, months_csv
                FROM vk_review_batch
                WHERE operator_id=? AND finished_at IS NULL AND months_csv<>''
                ORDER BY started_at DESC
                """,
                (operator_id,),
            )
            rows = await cur.fetchall()
        inline_buttons: list[list[types.InlineKeyboardButton]] = []
        summaries: list[str] = []
        for batch_id_db, months_csv in rows:
            months = [m for m in months_csv.split(',') if m]
            if not months:
                continue
            months_display = ", ".join(months)
            summaries.append(months_display)
            base_text = "🧹 Завершить и обновить страницы месяцев"
            suffix = f" ({months_display})" if months_display else ""
            button_text = base_text
            if suffix and len(base_text + suffix) <= 64:
                button_text = base_text + suffix
            inline_buttons.append(
                [
                    types.InlineKeyboardButton(
                        text=button_text,
                        callback_data=f"vkrev:finish:{batch_id_db}",
                    )
                ]
            )
        if inline_buttons:
            info_lines = ["Опубликованные события ждут обновления страниц месяцев."]
            if summaries:
                info_lines.append("; ".join(summaries))
            info_lines.append("Нажмите кнопку ниже, чтобы запустить обновление.")
            await bot.send_message(
                chat_id,
                "\n".join(info_lines),
                reply_markup=types.InlineKeyboardMarkup(inline_keyboard=inline_buttons),
            )
        return
    photos = await _vkrev_fetch_photos(post.group_id, post.post_id, db, bot)
    if photos:
        media = [types.InputMediaPhoto(media=p) for p in photos[:10]]
        with contextlib.suppress(Exception):
            await bot.send_media_group(chat_id, media)

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name FROM vk_source WHERE group_id=?",
            (post.group_id,),
        )
        row = await cur.fetchone()
    group_name = row[0] if row else f"group {post.group_id}"

    url = f"https://vk.com/wall-{post.group_id}_{post.post_id}"
    pending = await _vkrev_queue_size(db)
    if post.matched_kw == vk_intake.OCR_PENDING_SENTINEL:
        matched_kw_display = "ожидает OCR"
    elif post.matched_kw:
        matched_kw_display = post.matched_kw
    else:
        matched_kw_display = "-"
    status_line = (
        f"ключи: {matched_kw_display} | дата: {'да' if post.has_date else 'нет'} | в очереди: {pending}"
    )
    ts_hint = getattr(post, "event_ts_hint", None)
    heading_line: str | None = None
    event_lines: list[str] = []
    if ts_hint and ts_hint > 0:
        dt = datetime.fromtimestamp(ts_hint, tz=LOCAL_TZ)
        heading_line = f"{dt.day:02d} {MONTHS[dt.month - 1]} {dt.strftime('%H:%M')}"
        async with db.get_session() as session:
            result = await session.execute(
                select(Event).where(
                    Event.date == dt.date().isoformat(),
                    Event.time == dt.strftime("%H:%M"),
                )
            )
            matched_events = result.scalars().all()
        if matched_events:
            for event in matched_events:
                link = normalize_telegraph_url(event.telegraph_url)
                if not link and event.telegraph_path:
                    link = f"https://telegra.ph/{event.telegraph_path.lstrip('/')}"
                event_lines.append(
                    f"{event.title} — {link or 'Telegraph отсутствует'}"
                )
        else:
            event_lines.append("Совпадений нет")
    inline_keyboard = [
        [
            types.InlineKeyboardButton(text="✅ Добавить", callback_data=f"vkrev:accept:{post.id}"),
            types.InlineKeyboardButton(
                text="🎉 Добавить (+ фестиваль)",
                callback_data=f"vkrev:accept_fest:{post.id}",
            ),
        ],
        [
            types.InlineKeyboardButton(
                text="📝 Добавить с доп.инфо",
                callback_data=f"vkrev:accept_extra:{post.id}",
            ),
            types.InlineKeyboardButton(
                text="📝🎉 Добавить с доп.инфо (+ фестиваль)",
                callback_data=f"vkrev:accept_fest_extra:{post.id}",
            ),
        ],
        [
            types.InlineKeyboardButton(text="✖️ Отклонить", callback_data=f"vkrev:reject:{post.id}"),
            types.InlineKeyboardButton(text="⏭ Пропустить", callback_data=f"vkrev:skip:{post.id}"),
        ],
        [
            types.InlineKeyboardButton(
                text="Создать историю", callback_data=f"vkrev:story:{post.id}"
            )
        ],
        [types.InlineKeyboardButton(text="⏹ Стоп", callback_data=f"vkrev:stop:{batch_id}")],
        [
            types.InlineKeyboardButton(
                text="🧹 Завершить и обновить страницы месяцев",
                callback_data=f"vkrev:finish:{batch_id}",
            )
        ],
    ]
    imported_event_id = getattr(post, "imported_event_id", None)
    if imported_event_id:
        async with db.get_session() as session:
            event = await session.get(Event, imported_event_id)
        if event:
            inline_keyboard = append_tourist_block(inline_keyboard, event, "vk")
    markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
    post_text = post.text or ""
    def build_tail_lines(warning: str | None = None) -> list[str]:
        lines = [group_name, "", url]
        if heading_line:
            lines.extend(["", heading_line, *event_lines])
        lines.append("")
        if warning:
            lines.append(warning)
        lines.append(status_line)
        return lines

    tail_lines = build_tail_lines()
    tail_str = "\n".join(tail_lines)
    if post_text:
        message_text = post_text + "\n" + tail_str
    else:
        message_text = tail_str

    if len(message_text) > TELEGRAM_MESSAGE_LIMIT:
        warning_line = (
            f"⚠️ Текст поста был обрезан до {TELEGRAM_MESSAGE_LIMIT} символов"
        )
        tail_lines = build_tail_lines(warning_line)
        tail_str = "\n".join(tail_lines)
        if post_text:
            available = TELEGRAM_MESSAGE_LIMIT - len(tail_str) - 1
        else:
            available = TELEGRAM_MESSAGE_LIMIT - len(tail_str)
        available = max(0, available)
        if len(post_text) > available:
            truncated_text = post_text[: max(available - 1, 0)]
            if available > 0:
                truncated_text = truncated_text.rstrip()
                if truncated_text:
                    truncated_text += "…"
                else:
                    truncated_text = "…"
            post_text = truncated_text
        if post_text:
            message_text = post_text + "\n" + tail_str
        else:
            message_text = tail_str

    await bot.send_message(
        chat_id,
        message_text,
        reply_markup=markup,
    )


async def _vkrev_import_flow(
    chat_id: int,
    operator_id: int,
    inbox_id: int,
    batch_id: str,
    db: Database,
    bot: Bot,
    operator_extra: str | None = None,
    festival_hint: bool | None = None,
    *,
    force_festival: bool = False,
) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, text FROM vk_inbox WHERE id=?",
            (inbox_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(chat_id, "Инбокс не найден")
        return
    group_id, post_id, text = row
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT name, location, default_time, default_ticket_link FROM vk_source WHERE group_id=?",
            (group_id,),
        )
        source = await cur.fetchone()
    photos = await _vkrev_fetch_photos(group_id, post_id, db, bot)
    async with db.get_session() as session:
        res_f = await session.execute(select(Festival))
        festivals = res_f.scalars().all()
    festival_names = sorted(
        {
            (fest.name or "").strip()
            for fest in festivals
            if (fest.name or "").strip()
        }
    )
    festival_alias_pairs: list[tuple[str, int]] = []
    if festival_names:
        index_map = {name: idx for idx, name in enumerate(festival_names)}
        for fest in festivals:
            name = (fest.name or "").strip()
            if not name:
                continue
            idx = index_map.get(name)
            if idx is None:
                continue
            base_norm = normalize_alias(name)
            for alias in getattr(fest, "aliases", None) or []:
                norm = normalize_alias(alias)
                if not norm or norm == base_norm:
                    continue
                festival_alias_pairs.append((norm, idx))
        if festival_alias_pairs:
            seen_pairs: set[tuple[str, int]] = set()
            deduped: list[tuple[str, int]] = []
            for pair in festival_alias_pairs:
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                deduped.append(pair)
            festival_alias_pairs = deduped
    if festival_hint is None:
        festival_hint = force_festival
    source_name_val: str | None = None
    location_hint_val: str | None = None
    default_time_val: str | None = None
    default_ticket_link_val: str | None = None
    if source:
        source_name_val, location_hint_val, default_time_val, default_ticket_link_val = source

    drafts = await vk_intake.build_event_drafts(
        text,
        photos=photos,
        source_name=source_name_val,
        location_hint=location_hint_val,
        default_time=default_time_val,
        default_ticket_link=default_ticket_link_val,
        operator_extra=operator_extra,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs or None,
        festival_hint=festival_hint,
        db=db,
    )
    source_post_url = f"https://vk.com/wall-{group_id}_{post_id}"
    festival_info_raw = getattr(parse_event_via_4o, "_festival", None)
    setattr(parse_event_via_4o, "_festival", None)
    if isinstance(festival_info_raw, str):
        festival_info_raw = {"name": festival_info_raw}

    poster_urls: list[str] = []
    first_draft = drafts[0] if drafts else None
    if first_draft and first_draft.poster_media:
        poster_urls = [
            media.catbox_url
            for media in first_draft.poster_media
            if getattr(media, "catbox_url", None)
        ]
    poster_urls = [url for url in poster_urls if url]

    festival_obj: Festival | None = None
    fest_created = False
    fest_updated = False
    fest_status_line: str | None = None
    fest_data: dict[str, Any] | None = None
    fest_start_date: str | None = None
    fest_end_date: str | None = None
    if isinstance(festival_info_raw, dict):
        fest_data = festival_info_raw
        fest_name = clean_optional_str(
            fest_data.get("name") or fest_data.get("festival")
        )
    else:
        fest_name = None

    if force_festival and not fest_name:
        await bot.send_message(
            chat_id,
            "❌ Не удалось распознать фестиваль, импорт остановлен.",
        )
        return

    if fest_name:
        start_raw = None
        end_raw = None
        location_name = None
        city = None
        location_address = None
        website_url = None
        program_url = None
        ticket_url = None
        full_name = None
        if fest_data:
            start_raw = clean_optional_str(fest_data.get("start_date"))
            if not start_raw:
                start_raw = clean_optional_str(fest_data.get("date"))
            end_raw = clean_optional_str(fest_data.get("end_date"))
            location_name = clean_optional_str(fest_data.get("location_name"))
            city = clean_optional_str(fest_data.get("city"))
            location_address = clean_optional_str(fest_data.get("location_address"))
            website_url = clean_optional_str(fest_data.get("website_url"))
            program_url = clean_optional_str(fest_data.get("program_url"))
            ticket_url = clean_optional_str(fest_data.get("ticket_url"))
            full_name = clean_optional_str(fest_data.get("full_name"))
        fest_start_date = canonicalize_date(start_raw)
        fest_end_date = canonicalize_date(end_raw)
        location_address = strip_city_from_address(location_address, city)
        source_text_value = text
        if first_draft and first_draft.source_text:
            source_text_value = first_draft.source_text
        festival_obj, fest_created, fest_updated = await ensure_festival(
            db,
            fest_name,
            full_name=full_name,
            photo_url=poster_urls[0] if poster_urls else None,
            photo_urls=poster_urls,
            website_url=website_url,
            program_url=program_url,
            ticket_url=ticket_url,
            start_date=fest_start_date,
            end_date=fest_end_date,
            location_name=location_name,
            location_address=location_address,
            city=city,
            source_text=source_text_value,
            source_post_url=source_post_url,
        )
        if festival_obj:
            for draft in drafts:
                draft.festival = festival_obj.name
            status = "создан" if fest_created else "обновлён" if fest_updated else "без изменений"
            fest_status_line = f"Фестиваль: {festival_obj.name} ({status})"

    async def _sync_festival_updates() -> None:
        if festival_obj and (fest_created or fest_updated):
            try:
                await sync_festival_page(db, festival_obj.name)
            except Exception:
                logging.exception("festival page sync failed for %s", festival_obj.name)
            try:
                await sync_festivals_index_page(db)
            except Exception:
                logging.exception("festival index sync failed")
            try:
                await sync_festival_vk_post(db, festival_obj.name, bot, strict=True)
            except Exception:
                logging.exception("festival vk sync failed for %s", festival_obj.name)

    persist_results: list[
        tuple[vk_intake.EventDraft, vk_intake.PersistResult, Event | None]
    ] = []
    for draft in drafts:
        res = await vk_intake.persist_event_and_pages(
            draft, photos, db, source_post_url=source_post_url
        )
        async with db.get_session() as session:
            event_obj = await session.get(Event, res.event_id)
        persist_results.append((draft, res, event_obj))

    if not persist_results:
        if festival_obj:
            fest_month_hint = fest_start_date or fest_end_date or ""
            await vk_review.mark_imported(
                db, inbox_id, batch_id, operator_id, None, fest_month_hint
            )
            vk_review_actions_total["imported"] += 1
            message_lines = ["Импортировано только фестиваль"]
            if fest_status_line:
                message_lines.append(fest_status_line)
            message_lines.append("События не импортированы.")
            await bot.send_message(chat_id, "\n".join(message_lines))
            await _sync_festival_updates()
        else:
            await bot.send_message(chat_id, "LLM не вернул события")
        return

    first_res = persist_results[0][1]
    await vk_review.mark_imported(
        db, inbox_id, batch_id, operator_id, first_res.event_id, first_res.event_date
    )
    vk_review_actions_total["imported"] += 1
    link_lines: list[str] = []
    for idx, (_draft, res, _event_obj) in enumerate(persist_results, start=1):
        link_lines.append(f"Событие {idx}: ID {res.event_id}")
        link_lines.append(f"✅ Telegraph — {res.telegraph_url}")
        link_lines.append(f"✅ Календарь (ICS) — {res.ics_supabase_url}")
        link_lines.append(f"✅ ICS (Telegram) — {res.ics_tg_url}")
        if idx != len(persist_results):
            link_lines.append("")
    links = "\n".join(link_lines)
    admin_chat = os.getenv("ADMIN_CHAT_ID")
    if admin_chat:
        await bot.send_message(int(admin_chat), links)
    await bot.send_message(chat_id, links)

    await _sync_festival_updates()

    for idx, (draft, res, event_obj) in enumerate(persist_results, start=1):
        base_keyboard = [
            [
                types.InlineKeyboardButton(
                    text="↪️ Репостнуть в Vk",
                    callback_data=f"vkrev:repost:{res.event_id}",
                ),
                types.InlineKeyboardButton(
                    text="✂️ Сокращённый рерайт",
                    callback_data=f"vkrev:shortpost:{res.event_id}",
                ),
            ],
            [
                types.InlineKeyboardButton(
                    text="Редактировать",
                    callback_data=f"edit:{res.event_id}",
                )
            ],
        ]
        if event_obj:
            inline_keyboard = append_tourist_block(base_keyboard, event_obj, "vk")
        else:
            inline_keyboard = base_keyboard
        markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)

        def _display(value: str | None) -> str:
            return value if value else "—"

        detail_lines = [
            f"Тип: {_display(res.event_type)}",
            f"Дата начала: {_display(res.event_date)}",
            f"Дата окончания: {_display(res.event_end_date)}",
            f"Время: {_display(res.event_time)}",
            f"Бесплатное: {'да' if res.is_free else 'нет'}",
        ]
        if event_obj:
            detail_lines.append(
                _format_topics_line(
                    getattr(event_obj, "topics", None),
                    bool(getattr(event_obj, "topics_manual", False)),
                )
            )
        if fest_status_line:
            detail_lines.append(fest_status_line)
        if draft.poster_media and draft.ocr_tokens_remaining is not None:
            if getattr(draft, "ocr_limit_notice", None):
                detail_lines.append(draft.ocr_limit_notice)
            detail_lines.append(
                f"OCR: потрачено {draft.ocr_tokens_spent}, осталось {draft.ocr_tokens_remaining}"
            )

        header = "Импортировано"
        if len(persist_results) > 1:
            header = f"Импортировано #{idx}"

        if event_obj:
            message_text = build_event_card_message(
                header, event_obj, detail_lines
            )
        else:
            message_text = "\n".join([header, *detail_lines])

        await bot.send_message(chat_id, message_text, reply_markup=markup)


_VK_STORY_LINK_RE = re.compile(r"\[([^\[\]]+)\]")


def _vk_story_link_label(match: re.Match[str]) -> str:
    content = match.group(1)
    if "|" not in content:
        return content
    parts = [part.strip() for part in content.split("|")]
    for candidate in parts[1:]:
        if candidate:
            return candidate
    return ""


def _vkrev_story_title(text: str | None, group_id: int, post_id: int) -> str:
    if text:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            cleaned = _VK_STORY_LINK_RE.sub(_vk_story_link_label, stripped)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return cleaned[:64]
    return f"История VK {group_id}_{post_id}"


async def _vkrev_handle_story_choice(
    callback: types.CallbackQuery,
    placement: str,
    inbox_id_hint: int,
    db: Database,
    bot: Bot,
) -> None:
    operator_id = callback.from_user.id
    state = vk_review_story_sessions.get(operator_id)
    if not state:
        await bot.send_message(callback.message.chat.id, "Нет активной истории")
        return
    inbox_id = state.inbox_id
    if inbox_id_hint and inbox_id_hint != inbox_id:
        logging.info(
            "vk_review story inbox mismatch operator=%s stored=%s hint=%s",
            operator_id,
            inbox_id,
            inbox_id_hint,
        )
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, text FROM vk_inbox WHERE id=?",
            (inbox_id,),
        )
        row = await cur.fetchone()
    if not row:
        vk_review_story_sessions.pop(operator_id, None)
        await bot.send_message(callback.message.chat.id, "Инбокс не найден")
        return
    group_id, post_id, text = row
    title = _vkrev_story_title(text, group_id, post_id)
    source_url = f"https://vk.com/wall-{group_id}_{post_id}"
    photos = await _vkrev_fetch_photos(group_id, post_id, db, bot)
    image_mode = "inline" if placement == "middle" else "tail"
    source_text = text or ""
    editor_html: str | None = None
    pitch_text = ""
    if source_text.strip():
        try:
            pitch_text = await compose_story_pitch_via_4o(
                source_text,
                title=title,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "vk_review story pitch failed",  # pragma: no cover - logging only
                extra={
                    "operator": operator_id,
                    "inbox_id": inbox_id,
                    "error": str(exc),
                },
            )
            pitch_text = ""
        try:
            editor_candidate = await compose_story_editorial_via_4o(
                source_text,
                title=title,
            )
        except Exception as exc:
            logging.warning(
                "vk_review story editor request failed",  # pragma: no cover - logging only
                extra={
                    "operator": operator_id,
                    "inbox_id": inbox_id,
                    "error": str(exc),
                },
            )
        else:
            cleaned = editor_candidate.strip()
            if cleaned:
                editor_html = cleaned
            else:
                logging.warning(
                    "vk_review story editor returned empty response",
                    extra={"operator": operator_id, "inbox_id": inbox_id},
                )
    pitch_text = (pitch_text or "").strip()
    if pitch_text:
        pitch_html = f"<p><i>{html.escape(pitch_text)}</i></p>"
        if editor_html:
            editor_html = pitch_html + "\n" + editor_html
        else:
            editor_html = pitch_html
    try:
        result = await create_source_page(
            title,
            source_text,
            source_url,
            editor_html,
            db=None,
            catbox_urls=photos,
            image_mode=image_mode,
            page_mode="history",
        )
    except Exception as exc:  # pragma: no cover - network and external API
        logging.exception(
            "vk_review story creation failed",
            extra={"operator": operator_id, "inbox_id": inbox_id},
        )
        await bot.send_message(
            callback.message.chat.id, f"❌ Не удалось создать историю: {exc}"
        )
        return
    if not result:
        await bot.send_message(
            callback.message.chat.id, "❌ Не удалось создать историю"
        )
        return
    url, _path, catbox_msg, _uploaded = result
    if catbox_msg:
        logging.info("vkrev story catbox: %s", catbox_msg)
    if not url:
        await bot.send_message(
            callback.message.chat.id, "❌ Не удалось получить ссылку на Telegraph"
        )
        return
    placement_display = {
        "end": "в конце",
        "middle": "посреди текста",
    }.get(placement, placement or "неизвестно")
    with contextlib.suppress(Exception):
        await callback.message.edit_reply_markup()
    message_lines = [f"История готова ({placement_display}): {url}"]
    if pitch_text:
        message_lines.append(pitch_text)
    await bot.send_message(
        callback.message.chat.id,
        "\n".join(message_lines),
    )
    vk_review_story_sessions.pop(operator_id, None)


async def handle_vk_review_cb(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    assert callback.data
    parts = callback.data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    answered = False
    if action in {
        "accept",
        "accept_extra",
        "accept_fest",
        "accept_fest_extra",
        "reject",
        "skip",
    }:
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        batch_id = row[0] if row else ""
        if action in {"accept", "accept_fest"}:
            force_festival = action == "accept_fest"
            if action == "accept" and len(parts) > 3:
                force_arg = parts[3].strip().lower()
                force_festival = force_arg in {"1", "true", "fest", "festival", "force"}
            await callback.answer("Запускаю импорт…")
            answered = True
            await bot.send_message(
                callback.message.chat.id,
                "⏳ Начинаю импорт события…",
            )
            await _vkrev_import_flow(
                callback.message.chat.id,
                callback.from_user.id,
                inbox_id,
                batch_id,
                db,
                bot,
                force_festival=force_festival,
            )
        elif action in {"accept_extra", "accept_fest_extra"}:
            force_festival = action == "accept_fest_extra"
            vk_review_extra_sessions[callback.from_user.id] = (
                inbox_id,
                batch_id,
                force_festival,
            )
            await bot.send_message(
                callback.message.chat.id,
                "Отправьте доп. информацию одним сообщением",
            )
        elif action == "reject":
            await vk_review.mark_rejected(db, inbox_id)
            vk_review_actions_total["rejected"] += 1
            await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
        elif action == "skip":
            await vk_review.mark_skipped(db, inbox_id)
            vk_review_actions_total["skipped"] += 1
            await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
    elif action == "story":
        inbox_id = int(parts[2]) if len(parts) > 2 else 0
        async with db.raw_conn() as conn:
            cur = await conn.execute(
                "SELECT review_batch FROM vk_inbox WHERE id=?",
                (inbox_id,),
            )
            row = await cur.fetchone()
        batch_id = row[0] if row else ""
        vk_review_story_sessions[callback.from_user.id] = VkReviewStorySession(
            inbox_id=inbox_id,
            batch_id=batch_id,
        )
        placement_keyboard = types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    types.InlineKeyboardButton(
                        text="В конце",
                        callback_data=f"vkrev:storypos:end:{inbox_id}",
                    )
                ],
                [
                    types.InlineKeyboardButton(
                        text="Посреди текста",
                        callback_data=f"vkrev:storypos:middle:{inbox_id}",
                    )
                ],
            ]
        )
        await bot.send_message(
            callback.message.chat.id,
            "Где разместить иллюстрации?",
            reply_markup=placement_keyboard,
        )
        answered = True
        await callback.answer()
    elif action == "storypos":
        placement = parts[2] if len(parts) > 2 else ""
        inbox_id = int(parts[3]) if len(parts) > 3 else 0
        answered = True
        await callback.answer("Создаю историю…")
        await _vkrev_handle_story_choice(callback, placement, inbox_id, db, bot)
    elif action == "stop":
        async with db.raw_conn() as conn:
            await conn.execute(
                "UPDATE vk_inbox SET status='pending', locked_by=NULL, locked_at=NULL WHERE locked_by=?",
                (callback.from_user.id,),
            )
            await conn.commit()
        vk_review_extra_sessions.pop(callback.from_user.id, None)
        buttons = [
            [types.KeyboardButton(text=VK_BTN_ADD_SOURCE)],
            [types.KeyboardButton(text=VK_BTN_LIST_SOURCES)],
            [
                types.KeyboardButton(text=VK_BTN_CHECK_EVENTS),
                types.KeyboardButton(text=VK_BTN_QUEUE_SUMMARY),
            ],
        ]
        markup = types.ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
        await bot.send_message(callback.message.chat.id, "Остановлено", reply_markup=markup)
    elif action == "finish":
        batch_id = parts[2] if len(parts) > 2 else ""
        reports: list[tuple[str, str]] = []

        async def rebuild_cb(db_: Database, month: str) -> None:
            report = await _perform_pages_rebuild(db_, [month], force=True)
            reports.append((month, report))

        months = await vk_review.finish_batch(db, batch_id, rebuild_cb)
        if months:
            await bot.send_message(
                callback.message.chat.id,
                "Запущен rebuild для: " + ", ".join(months),
            )
            for _, report in reports:
                if report:
                    await bot.send_message(callback.message.chat.id, report)
        else:
            await bot.send_message(callback.message.chat.id, "Нет месяцев для обновления")
    elif action == "repost":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_handle_repost(callback, event_id, db, bot)
    elif action == "shortpost":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_handle_shortpost(callback, event_id, db, bot)
    elif action == "shortpost_pub":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        await _vkrev_publish_shortpost(
            event_id, db, bot, callback.message.chat.id, callback.from_user.id
        )
    elif action == "shortpost_edit":
        event_id = int(parts[2]) if len(parts) > 2 else 0
        vk_shortpost_edit_sessions[callback.from_user.id] = (
            event_id,
            callback.message.message_id,
        )
        await bot.send_message(
            callback.message.chat.id,
            "Отправьте новый текст поста одной строкой/сообщением",
        )
    if not answered:
        await callback.answer()


async def _vkrev_handle_repost(callback: types.CallbackQuery, event_id: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(callback.message.chat.id, "❌ Репост не удался: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev:
        await bot.send_message(callback.message.chat.id, "❌ Репост не удался: нет события")
        return

    if VK_ALLOW_TRUE_REPOST:
        object_id = f"wall-{group_id}_{post_id}"
        target_group = int(VK_AFISHA_GROUP_ID.lstrip('-')) if VK_AFISHA_GROUP_ID else None
        params = {"object": object_id}
        if target_group:
            params["group_id"] = target_group
        global vk_repost_attempts_total, vk_repost_errors_total
        vk_repost_attempts_total += 1
        try:
            data = await _vk_api("wall.repost", params, db, bot, token=VK_TOKEN_AFISHA)
            post = data.get("response", {}).get("post_id")
            if not post:
                raise RuntimeError("no post_id")
            url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
            await vk_review.save_repost_url(db, event_id, url)
            await bot.send_message(callback.message.chat.id, url)
        except Exception as e:  # pragma: no cover
            vk_repost_errors_total += 1
            logging.exception("vk repost failed")
            await bot.send_message(
                callback.message.chat.id,
                f"❌ Репост не удался: {getattr(e, 'code', getattr(e, 'message', str(e)))}",
            )
        await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)
        return

    try:
        response = await vk_api("wall.getById", posts=f"-{group_id}_{post_id}")
    except Exception:
        items: list[dict[str, Any]] = []
    else:
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
    photos = _vkrev_collect_photo_ids(items, VK_SHORTPOST_MAX_PHOTOS)
    attachments = ",".join(photos) if photos else vk_url
    message = f"Репост: {ev.title}\n\n[{vk_url}|Источник]"
    params = {
        "owner_id": f"-{VK_AFISHA_GROUP_ID.lstrip('-')}",
        "from_group": 1,
        "message": message,
        "attachments": attachments,
        "copyright": vk_url,
        "signed": 0,
    }
    try:
        data = await _vk_api(
            "wall.post",
            params,
            db,
            bot,
            token=VK_TOKEN_AFISHA,
            skip_captcha=True,
        )
        post = data.get("response", {}).get("post_id")
        if not post:
            raise RuntimeError("no post_id")
        url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
        await vk_review.save_repost_url(db, event_id, url)
        await bot.send_message(callback.message.chat.id, url)
    except VKAPIError as e:
        logging.error(
            "vk.repost_failed actor=%s token=%s code=%s msg=%s",
            e.actor,
            e.token,
            e.code,
            e.message,
        )
        await bot.send_message(
            callback.message.chat.id,
            f"❌ Репост не удался: {e.message}",
        )
    except Exception as e:  # pragma: no cover
        await bot.send_message(
            callback.message.chat.id,
            f"❌ Репост не удался: {getattr(e, 'message', str(e))}",
        )
    await _vkrev_show_next(callback.message.chat.id, batch_id, callback.from_user.id, db, bot)


def _normalize_location_part(part: str | None) -> str:
    if not part:
        return ""
    normalized = unicodedata.normalize("NFKC", part)
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


async def _vkrev_build_shortpost(
    ev: Event,
    vk_url: str,
    *,
    db: Database | None = None,
    session: AsyncSession | None = None,
    bot: Bot | None = None,
    for_preview: bool = False,
    poster_texts: Sequence[str] | None = None,
) -> tuple[str, str | None]:

    text_len = len(ev.source_text or "")
    if text_len < 200:
        max_sent = 1
    elif text_len < 500:
        max_sent = 2
    elif text_len < 800:
        max_sent = 3
    else:
        max_sent = 4
    summary = await build_short_vk_text(
        ev,
        ev.source_text or "",
        max_sent,
        poster_texts=poster_texts,
    )

    start_date: date | None = None
    try:
        start_date = date.fromisoformat(ev.date)
    except (TypeError, ValueError):
        start_date = None

    end_date_obj: date | None = None
    if ev.end_date:
        try:
            end_date_obj = date.fromisoformat(ev.end_date)
        except (TypeError, ValueError):
            end_date_obj = None

    if start_date:
        default_date_str = f"{start_date.day} {MONTHS[start_date.month - 1]}"
    else:
        try:
            parts = ev.date.split("-")
            day = int(parts[2])
            month = int(parts[1])
        except (AttributeError, IndexError, ValueError):
            default_date_str = ev.date
        else:
            default_date_str = f"{day} {MONTHS[month - 1]}"

    today = date.today()

    ongoing_exhibition = (
        ev.event_type == "выставка"
        and start_date is not None
        and end_date_obj is not None
        and start_date <= today <= end_date_obj
    )

    if ongoing_exhibition:
        end_month_name = MONTHS[end_date_obj.month - 1]
        year_suffix = ""
        if start_date.year != end_date_obj.year:
            year_suffix = f" {end_date_obj.year}"
        date_line = f"🗓 по {end_date_obj.day} {end_month_name}{year_suffix}"
    else:
        time_part = f" ⏰ {ev.time}" if ev.time and ev.time != "00:00" else ""
        date_line = f"🗓 {default_date_str}{time_part}"

    type_line: str | None = None
    type_line_used_tag: str | None = None
    raw_event_type = (ev.event_type or "").strip()
    if raw_event_type:
        event_type_lower = raw_event_type.casefold()
        normalized_event_type = re.sub(
            r"[^0-9a-zа-яё]+", "_", event_type_lower
        ).strip("_")
        if normalized_event_type:
            normalized_hashtag = f"#{normalized_event_type}"
            type_line = normalized_hashtag
            type_line_used_tag = normalized_hashtag
            if re.search(r"[-–—]", raw_event_type):
                hyphen_free = re.sub(r"[^0-9a-zа-яё]", "", event_type_lower)
                if hyphen_free:
                    type_line = f"#{hyphen_free}"
                    type_line_used_tag = type_line

    tags = await build_short_vk_tags(ev, summary, used_type_hashtag=type_line_used_tag)
    title_line = ev.title.upper() if ev.title else ""
    if getattr(ev, "is_free", False):
        title_line = f"🆓 {title_line}".strip()
    lines = [
        title_line,
        "",
    ]
    lines.append(date_line)
    if type_line:
        lines.append(type_line)
    ticket_url_for_message = (
        format_vk_short_url(ev.vk_ticket_short_url)
        if ev.vk_ticket_short_url
        else ev.ticket_link
    )
    if ev.ticket_link and not for_preview:
        short_result = await ensure_vk_short_ticket_link(
            ev,
            db,
            session=session,
            bot=bot,
            vk_api_fn=_vk_api,
        )
        if short_result:
            ticket_url_for_message = format_vk_short_url(short_result[0])
    if ev.ticket_link:
        if getattr(ev, "is_free", False):
            lines.append(
                f"🆓 Бесплатно, по регистрации {ticket_url_for_message}"
            )
        else:
            lines.append(f"🎟 Билеты: {ticket_url_for_message}")
    loc_parts: list[str] = []
    existing_normalized: set[str] = set()
    for part in (ev.location_name, ev.location_address):
        if part:
            loc_parts.append(part)
            normalized = _normalize_location_part(part)
            if normalized:
                existing_normalized.add(normalized)
    if ev.city:
        city_normalized = _normalize_location_part(ev.city)
        if not city_normalized or city_normalized not in existing_normalized:
            loc_parts.append(ev.city)
            if city_normalized:
                existing_normalized.add(city_normalized)
    location_text = await build_short_vk_location(loc_parts)
    if location_text:
        lines.append(f"📍 {location_text}")
    lines.append("")
    lines.append(summary)
    summary_idx = len(lines) - 1
    lines.append("")
    if for_preview:
        lines.append("Источник")
        lines.append(vk_url)
    else:
        lines.append(f"[{vk_url}|Источник]")
    lines.append("")
    lines.append(" ".join(tags))
    message = "\n".join(lines)
    if len(message) > 4096:
        excess = len(message) - 4096
        lines[summary_idx] = lines[summary_idx][: -excess]
        message = "\n".join(lines)
    link_attachment = ev.telegraph_url or vk_url
    return message, link_attachment


async def _vkrev_handle_shortpost(callback: types.CallbackQuery, event_id: int, db: Database, bot: Bot) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(callback.message.chat.id, "❌ Не удалось: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            await bot.send_message(
                callback.message.chat.id, "❌ Не удалось: нет события"
            )
            return

        poster_texts = await get_event_poster_texts(event_id, db)

        message, link_attachment = await _vkrev_build_shortpost(
            ev,
            vk_url,
            db=db,
            session=session,
            bot=bot,
            for_preview=True,
            poster_texts=poster_texts,
        )
    markup = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [
                types.InlineKeyboardButton(
                    text="Опубликовать",
                    callback_data=f"vkrev:shortpost_pub:{event_id}",
                ),
                types.InlineKeyboardButton(
                    text="Изменить",
                    callback_data=f"vkrev:shortpost_edit:{event_id}",
                ),
            ]
        ]
    )
    await bot.send_message(callback.message.chat.id, message, reply_markup=markup)
    logging.info(
        "shortpost_preview_sent",
        extra={"eid": event_id, "chat_id": callback.message.chat.id},
    )
    vk_shortpost_ops[event_id] = VkShortpostOpState(
        chat_id=callback.message.chat.id,
        preview_text=message,
        preview_link_attachment=link_attachment,
    )


async def _vkrev_publish_shortpost(
    event_id: int,
    db: Database,
    bot: Bot,
    actor_chat_id: int,
    operator_id: int,
    text: str | None = None,
    edited: bool = False,
) -> None:
    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT group_id, post_id, review_batch FROM vk_inbox WHERE imported_event_id=?",
            (event_id,),
        )
        row = await cur.fetchone()
    if not row:
        await bot.send_message(actor_chat_id, "❌ Не удалось: нет события")
        return
    group_id, post_id, batch_id = row
    vk_url = f"https://vk.com/wall-{group_id}_{post_id}"
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
        if not ev:
            await bot.send_message(actor_chat_id, "❌ Не удалось: нет события")
            return
        op_state = vk_shortpost_ops.get(event_id)
        poster_texts = await get_event_poster_texts(event_id, db)
    def _ensure_publish_markup(message: str) -> str:
        lines = message.split("\n")
        markup = f"[{vk_url}|Источник]"
        for idx, line in enumerate(lines):
            if line.strip() == "Источник":
                if idx + 1 < len(lines):
                    del lines[idx + 1]
                lines[idx] = markup
                return "\n".join(lines)
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("|Источник]"):
                lines[idx] = markup
                return "\n".join(lines)
        return message

    short_ticket = None
    if text is None:
        if op_state and op_state.preview_text is not None:
            message = _ensure_publish_markup(op_state.preview_text)
            link_attachment = (
                op_state.preview_link_attachment
                if op_state.preview_link_attachment is not None
                else ev.telegraph_url or vk_url
            )
            if ev.ticket_link:
                short_ticket = await ensure_vk_short_ticket_link(
                    ev,
                    db,
                    bot=bot,
                    vk_api_fn=_vk_api,
                )
                if short_ticket:
                    message = message.replace(
                        ev.ticket_link, format_vk_short_url(short_ticket[0])
                    )
        else:
            message, link_attachment = await _vkrev_build_shortpost(
                ev,
                vk_url,
                db=db,
                bot=bot,
                poster_texts=poster_texts,
            )
    else:
        message = _ensure_publish_markup(text)
        link_attachment = ev.telegraph_url or vk_url
        if ev.ticket_link:
            short_ticket = await ensure_vk_short_ticket_link(
                ev,
                db,
                bot=bot,
                vk_api_fn=_vk_api,
            )
            if short_ticket:
                message = message.replace(
                    ev.ticket_link, format_vk_short_url(short_ticket[0])
                )

    photo_attachments: list[str] = []
    try:
        response = await vk_api("wall.getById", posts=f"-{group_id}_{post_id}")
    except Exception as exc:  # pragma: no cover - logging only
        logging.error(
            "shortpost_fetch_photos_failed gid=%s post=%s: %s",
            group_id,
            post_id,
            exc,
        )
    else:
        if isinstance(response, dict):
            items = response.get("response") or (
                response["response"] if "response" in response else response
            )
        else:
            items = response or []
        if not isinstance(items, list):
            items = [items] if items else []
        photo_attachments.extend(
            _vkrev_collect_photo_ids(items, VK_SHORTPOST_MAX_PHOTOS)
        )

    if not photo_attachments and VK_PHOTOS_ENABLED and ev.photo_urls:
        token = VK_TOKEN_AFISHA or VK_TOKEN
        if token:
            uploaded: list[str] = []
            for url in ev.photo_urls[:VK_SHORTPOST_MAX_PHOTOS]:
                photo_id = await upload_vk_photo(
                    VK_AFISHA_GROUP_ID,
                    url,
                    db,
                    bot,
                    token=token,
                    token_kind="group",
                )
                if photo_id:
                    uploaded.append(photo_id)
                elif not VK_USER_TOKEN:
                    logging.info(
                        "shortpost_photo_upload_skipped gid=%s post=%s reason=user_token_required",
                        group_id,
                        post_id,
                    )
                    break
            photo_attachments.extend(uploaded)
        else:
            logging.info(
                "shortpost_photo_upload_skipped gid=%s post=%s reason=no_token",
                group_id,
                post_id,
            )

    attachments: list[str] = []
    if photo_attachments:
        attachments.extend(photo_attachments)
        if link_attachment:
            attachments.append(link_attachment)

    attachments_str = ",".join(attachments) if attachments else None

    params = {
        "owner_id": f"-{VK_AFISHA_GROUP_ID.lstrip('-')}",
        "from_group": 1,
        "message": message,
        "copyright": vk_url,
        "signed": 0,
    }
    if attachments_str:
        params["attachments"] = attachments_str
    operator_chat = op_state.chat_id if op_state else None
    try:
        data = await _vk_api(
            "wall.post",
            params,
            db,
            bot,
            token=VK_TOKEN_AFISHA,
            skip_captcha=True,
        )
        post = data.get("response", {}).get("post_id")
        if not post:
            raise RuntimeError("no post_id")
        url = f"https://vk.com/wall-{VK_AFISHA_GROUP_ID.lstrip('-')}_{post}"
        await vk_review.save_repost_url(db, event_id, url)
        await bot.send_message(actor_chat_id, f"✅ Опубликовано: {url}")
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, f"✅ Опубликовано: {url}")
        logging.info("shortpost_publish", extra={"eid": event_id, "edited": edited})
        vk_shortpost_ops.pop(event_id, None)
        await _vkrev_show_next(actor_chat_id, batch_id, operator_id, db, bot)
    except VKAPIError as e:
        if e.code == 14:
            msg = "Капча, публикацию не делаем. Попробуйте позже"
        else:
            msg = f"❌ Не удалось: {e.message}"
        await bot.send_message(actor_chat_id, msg)
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, msg)
        logging.warning(
            "shortpost_publish_failed code=%s actor=%s token=%s",
            e.code,
            e.actor,
            e.token,
            extra={"eid": event_id},
        )
    except Exception as e:  # pragma: no cover
        msg = f"❌ Не удалось: {getattr(e, 'message', str(e))}"
        await bot.send_message(actor_chat_id, msg)
        if operator_chat and operator_chat != actor_chat_id:
            await bot.send_message(operator_chat, msg)
        logging.warning("shortpost_publish_failed", extra={"eid": event_id, "error": str(e)})


def extract_message_text_with_links(message: types.Message) -> str:
    """Return message text where hidden links are exposed for downstream use."""

    base_text = message.text or message.caption or ""
    if not base_text:
        return ""

    html_text, _mode = ensure_html_text(message)
    if not html_text:
        return base_text

    def escape_md_label(value: str) -> str:
        return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")

    def escape_md_url(value: str) -> str:
        return value.replace("\\", "\\\\").replace(")", "\\)")

    def repl_anchor(match: re.Match[str]) -> str:
        href = match.group(1)
        label_html = match.group(2)
        label = re.sub(r"</?[^>]+>", "", label_html)
        label = html.unescape(label)
        label = label.replace("\xa0", " ")
        label_md = escape_md_label(label)
        if not label_md.strip():
            return href
        return f"[{label_md}]({escape_md_url(href)})"

    text = re.sub(r"(?is)<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>", repl_anchor, html_text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?i)</div>", "\n", text)
    text = re.sub(r"(?i)</li>", "\n", text)
    text = re.sub(r"(?i)<li>", "• ", text)
    text = re.sub(r"</?[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")

    return text or base_text


async def handle_vk_extra_message(message: types.Message, db: Database, bot: Bot) -> None:
    info = vk_review_extra_sessions.pop(message.from_user.id, None)
    if not info:
        return
    inbox_id, batch_id, force_festival = info
    operator_extra = extract_message_text_with_links(message)
    await _vkrev_import_flow(
        message.chat.id,
        message.from_user.id,
        inbox_id,
        batch_id,
        db,
        bot,
        operator_extra=operator_extra,
        force_festival=force_festival,
    )


async def handle_tourist_note_message(message: types.Message, db: Database, bot: Bot) -> None:
    try:
        session_state = tourist_note_sessions.pop(message.from_user.id)
    except KeyError:
        await bot.send_message(
            message.chat.id,
            "Сессия для комментария истекла, нажмите кнопку заново.",
        )
        return
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        event = await session.get(Event, session_state.event_id)
        if not event or not _user_can_label_event(user):
            await bot.send_message(message.chat.id, "Not authorized")
            return
        note_text = (message.text or "").strip()
        is_trimmed = False
        if len(note_text) > 500:
            note_text = note_text[:500]
            is_trimmed = True
        event.tourist_note = note_text or None
        event.tourist_label_by = message.from_user.id
        event.tourist_label_at = datetime.now(timezone.utc)
        event.tourist_label_source = "operator"
        session.add(event)
        await session.commit()
        await session.refresh(event)
    logging.info(
        "tourist_note_saved",
        extra={
            "event_id": session_state.event_id,
            "user_id": message.from_user.id,
            "has_note": bool(event.tourist_note),
        },
    )
    confirmation_text = (
        "Комментарий сохранён (обрезан до 500 символов)."
        if is_trimmed
        else "Комментарий сохранён."
    )
    await bot.send_message(message.chat.id, confirmation_text)
    base_markup = session_state.markup
    new_markup = replace_tourist_block(
        base_markup, event, session_state.source, menu=session_state.menu
    )
    original_text = session_state.message_text
    if original_text is not None:
        updated_text = apply_tourist_status_to_text(original_text, event)
        try:
            await bot.edit_message_text(
                chat_id=session_state.chat_id,
                message_id=session_state.message_id,
                text=updated_text,
                reply_markup=new_markup,
            )
        except TelegramBadRequest as exc:  # pragma: no cover - Telegram quirks
            logging.warning(
                "tourist_note_message_update_failed",
                extra={"event_id": session_state.event_id, "error": exc.message},
            )
            with contextlib.suppress(Exception):
                await bot.edit_message_reply_markup(
                    chat_id=session_state.chat_id,
                    message_id=session_state.message_id,
                    reply_markup=new_markup,
                )
    else:
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(
                chat_id=session_state.chat_id,
                message_id=session_state.message_id,
                reply_markup=new_markup,
            )


async def handle_vk_shortpost_edit_message(message: types.Message, db: Database, bot: Bot) -> None:
    info = vk_shortpost_edit_sessions.pop(message.from_user.id, None)
    if not info:
        return
    event_id, _ = info
    await _vkrev_publish_shortpost(
        event_id,
        db,
        bot,
        message.chat.id,
        message.from_user.id,
        text=message.text or "",
        edited=True,
    )


async def handle_vk_next_callback(callback: types.CallbackQuery, db: Database, bot: Bot) -> None:
    try:
        _, batch, idx = callback.data.split(":", 2)
        index = int(idx)
    except Exception:
        await callback.answer()
        return
    async with db.raw_conn() as conn:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM vk_tmp_post WHERE batch=?", (batch,)
        )
        total = (await cursor.fetchone())[0]
    await send_vk_tmp_post(callback.message.chat.id, batch, index, total, db, bot)
    await callback.answer()


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
        elif field == "program":
            fest.program_url = None if text in {"", "-"} else text
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
    await rebuild_fest_nav_if_changed(db)



@dataclass
class AlbumImage:
    data: bytes
    name: str
    seq: int
    file_unique_id: str | None = None


@dataclass
class AlbumState:
    images: list[AlbumImage]
    text: str | None = None
    html: str | None = None
    html_mode: str = "native"
    message: types.Message | None = None
    timer: asyncio.Task | None = None
    created: float = field(default_factory=_time.monotonic)


pending_albums: dict[str, AlbumState] = {}
processed_media_groups: set[str] = set()


async def _drop_album_after_ttl(gid: str) -> None:
    await asyncio.sleep(ALBUM_PENDING_TTL_S)
    state = pending_albums.get(gid)
    if state and not state.text:
        age = int(_time.monotonic() - state.created)
        logging.info(
            "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
            gid,
            len(state.images),
            age,
        )
        pending_albums.pop(gid, None)


async def _finalize_album_after_delay(gid: str, db: Database, bot: Bot) -> None:
    await asyncio.sleep(ALBUM_FINALIZE_DELAY_MS / 1000)
    await finalize_album(gid, db, bot)


async def finalize_album(gid: str, db: Database, bot: Bot) -> None:
    state = pending_albums.pop(gid, None)
    if not state or not state.text or not state.message:
        return
    start = _time.monotonic()
    images_total = len(state.images)
    processed_media_groups.add(gid)
    images_sorted = sorted(state.images, key=lambda im: im.seq)
    uniq: list[AlbumImage] = []
    seen: set[str] = set()
    for im in images_sorted:
        uid = im.file_unique_id
        if uid and uid in seen:
            continue
        if uid:
            seen.add(uid)
        uniq.append(im)
    used_images = uniq[:MAX_ALBUM_IMAGES]
    order = [im.seq for im in used_images]
    logging.info(
        "album_finalize gid=%s images=%d order=%s",
        gid,
        len(used_images),
        order,
    )
    logging.info("html_mode=%s", state.html_mode)
    media = [(im.data, im.name) for im in used_images]
    poster_items, catbox_msg = await process_media(
        media, need_catbox=True, need_ocr=False
    )
    global LAST_CATBOX_MSG, LAST_HTML_MODE
    LAST_CATBOX_MSG = catbox_msg
    LAST_HTML_MODE = state.html_mode
    msg = state.message
    session_mode = None
    if msg and msg.from_user:
        session_mode = add_event_sessions.get(msg.from_user.id)
    if msg.forward_date or msg.forward_from_chat or getattr(msg, "forward_origin", None):
        await _process_forwarded(
            msg,
            db,
            bot,
            state.text,
            state.html,
            media,
            poster_media=poster_items,
            catbox_msg=catbox_msg,
        )
    else:
        mode_for_call: AddEventMode = session_mode or "event"
        await handle_add_event(
            msg,
            db,
            bot,
            session_mode=mode_for_call,
            force_festival=mode_for_call == "festival",
            media=media,
            poster_media=poster_items,
            catbox_msg=catbox_msg,
        )
        add_event_sessions.pop(msg.from_user.id, None)
    took = int((_time.monotonic() - start) * 1000)
    logging.info(
        "album_finalize_done gid=%s images_total=%d took_ms=%d used_images=%d catbox_result=%s",
        gid,
        images_total,
        took,
        len(used_images),
        LAST_CATBOX_MSG,
    )


async def _process_forwarded(
    message: types.Message,
    db: Database,
    bot: Bot,
    text: str,
    html: str | None,
    media: list[tuple[bytes, str]] | None,
    poster_media: Sequence[PosterMedia] | None = None,
    catbox_msg: str | None = None,
) -> None:
    async with db.get_session() as session:
        user = await session.get(User, message.from_user.id)
        if not user or user.blocked:
            logging.info("user %s not registered or blocked", message.from_user.id)
            return
    link = None
    msg_id = None
    chat_id: int | None = None
    channel_name: str | None = None

    allowed: bool | None = None
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
    # determine where to update buttons: default to forwarded message itself
    target_chat_id = message.chat.id
    target_message_id = message.message_id
    if chat_id and msg_id:
        target_chat_id = chat_id
        target_message_id = msg_id
    logging.info(
        "FWD link=%s channel_id=%s name=%s allowed=%s",
        link,
        chat_id,
        channel_name,
        allowed,
    )
    if media is None:
        normalized_media: list[tuple[bytes, str]] = []
    elif isinstance(media, tuple):
        normalized_media = [media]
    else:
        normalized_media = list(media)
    poster_items: list[PosterMedia] = []
    local_catbox_msg = catbox_msg or ""
    if poster_media is not None:
        poster_items = list(poster_media)
    elif normalized_media:
        poster_items, local_catbox_msg = await process_media(
            normalized_media, need_catbox=True, need_ocr=False
        )
    global LAST_CATBOX_MSG
    LAST_CATBOX_MSG = local_catbox_msg
    logging.info(
        "FWD summary text_len=%d media_len=%d posters=%d",
        len(text or ""),
        len(normalized_media),
        len(poster_items),
    )
    logging.info("parsing forwarded text via LLM")
    try:
        results = await add_events_from_text(
            db,
            text,
            link,
            html,
            normalized_media,
            poster_media=poster_items,
            source_chat_id=target_chat_id,
            source_message_id=target_message_id,
            creator_id=user.user_id,
            source_channel=channel_name,
            bot=None,
        )
    except Exception as e:
        logging.exception("forward parse failed")
        snippet = (text or "")[:200]
        msg = f"Не удалось обработать сообщение: {type(e).__name__}: {e}"
        if snippet:
            msg += f"\n\n{snippet}"
        if link:
            msg += f"\n{link}"
        await notify_superadmin(db, bot, msg)
        return
    logging.info("forward parsed %d events", len(results))
    ocr_line = None
    if normalized_media and results.ocr_tokens_remaining is not None:
        base_line = (
            f"OCR: потрачено {results.ocr_tokens_spent}, осталось "
            f"{results.ocr_tokens_remaining}"
        )
        if results.ocr_limit_notice:
            ocr_line = f"{results.ocr_limit_notice}\n{base_line}"
        else:
            ocr_line = base_line
    if not results:
        logging.info("no events parsed from forwarded text")
        await bot.send_message(
            message.chat.id,
            "Я не смог найти события в пересланном посте. "
            "Попробуйте добавить событие вручную или свяжитесь с поддержкой.",
        )
        return
    for saved, added, lines, status in results:
        if status == "missing":
            buttons: list[list[types.InlineKeyboardButton]] = []
            if "time" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить время", callback_data="asktime")]
                )
                buttons.append(
                    [types.InlineKeyboardButton(text="Изменить дату", callback_data="askdate")]
                )
            if "location_name" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить локацию", callback_data="askloc")]
                )
            if "city" in lines:
                buttons.append(
                    [types.InlineKeyboardButton(text="Добавить город", callback_data="askcity")]
                )
            saved_id = getattr(saved, "id", None)
            if saved_id is not None:
                buttons.append(
                    [
                        types.InlineKeyboardButton(
                            text="Редактировать",
                            callback_data=f"edit:{saved_id}",
                        )
                    ]
                )
            keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
            await bot.send_message(
                message.chat.id,
                "Отсутствуют обязательные поля: " + ", ".join(lines),
                reply_markup=keyboard,
            )
            continue
        if isinstance(saved, Festival):
            async with db.get_session() as session:
                count = (
                    await session.scalar(
                        select(func.count()).where(Event.festival == saved.name)
                    )
                ) or 0
            logging.info(
                "festival_notify",
                extra={
                    "festival": saved.name,
                    "action": "created" if added else "updated",
                    "events_count_at_moment": count,
                },
            )
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
            text_out = "Festival added\n" + "\n".join(lines)
            if ocr_line:
                text_out = f"{text_out}\n{ocr_line}"
                ocr_line = None
            await bot.send_message(message.chat.id, text_out, reply_markup=markup)
            continue
        buttons: list[types.InlineKeyboardButton] = []
        if not saved.city:
            buttons.append(
                types.InlineKeyboardButton(
                    text="Добавить город", callback_data="askcity"
                )
            )
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
        buttons.append(
            types.InlineKeyboardButton(
                text="Редактировать",
                callback_data=f"edit:{saved.id}",
            )
        )
        inline_keyboard: list[list[types.InlineKeyboardButton]]
        if len(buttons) > 1:
            inline_keyboard = [buttons[:-1], [buttons[-1]]]
        else:
            inline_keyboard = [[buttons[0]]]
        inline_keyboard = append_tourist_block(inline_keyboard, saved, "tg")
        markup = types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)
        extra_lines = [ocr_line] if ocr_line else None
        text_out = build_event_card_message(
            f"Event {status}", saved, lines, extra_lines=extra_lines
        )
        if ocr_line:
            ocr_line = None
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
    if message.media_group_id:
        gid = message.media_group_id
        if gid in processed_media_groups:
            logging.info("skip already processed album %s", gid)
            return
        state = pending_albums.get(gid)
        if not state:
            if len(pending_albums) >= MAX_PENDING_ALBUMS:
                old_gid, old_state = min(
                    pending_albums.items(), key=lambda kv: kv[1].created
                )
                if old_state.timer:
                    old_state.timer.cancel()
                age = int(_time.monotonic() - old_state.created)
                logging.info(
                    "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
                    old_gid,
                    len(old_state.images),
                    age,
                )
                pending_albums.pop(old_gid, None)
            state = AlbumState(images=[])
            state.timer = asyncio.create_task(_drop_album_after_ttl(gid))
            pending_albums[gid] = state
        seq = message.forward_from_message_id
        if not seq:
            fo = getattr(message, "forward_origin", None)
            if isinstance(fo, dict):
                seq = fo.get("message_id")
            else:
                seq = getattr(fo, "message_id", None)
        if not seq:
            seq = message.message_id or int(message.date.timestamp())
        img_count = len(images or [])
        if images and len(state.images) < MAX_ALBUM_IMAGES:
            add = min(img_count, MAX_ALBUM_IMAGES - len(state.images))
            file_uid = None
            if message.photo:
                file_uid = message.photo[-1].file_unique_id
            elif (
                message.document
                and message.document.mime_type
                and message.document.mime_type.startswith("image/")
            ):
                file_uid = message.document.file_unique_id
            for data, name in images[:add]:
                state.images.append(AlbumImage(data=data, name=name, seq=seq, file_unique_id=file_uid))
        logging.info(
            "album_collect gid=%s seq=%s msg_id=%s has_text=%s images_in_msg=%d buf_size_after=%d",
            gid,
            seq,
            message.message_id,
            bool(text),
            len(images or []),
            len(state.images),
        )
        if text and not state.text:
            state.text = text
            state.html, state.html_mode = ensure_html_text(message)
            state.message = message
            if state.timer:
                state.timer.cancel()
            logging.info(
                "album_caption_seen gid=%s delay_ms=%d",
                gid,
                ALBUM_FINALIZE_DELAY_MS,
            )
            state.timer = asyncio.create_task(
                _finalize_album_after_delay(gid, db, bot)
            )
        return
    if not text:
        logging.info("forwarded message has no text")
        return
    media = images[:MAX_ALBUM_IMAGES] if images else None
    logging.info("IMG single message media_len=%d", len(media or []))
    html, _mode = ensure_html_text(message)
    await _process_forwarded(
        message,
        db,
        bot,
        text,
        html,
        media,
    )


async def handle_add_event_media_group(
    message: types.Message, db: Database, bot: Bot
) -> None:
    """Collect media group messages for /addevent sessions."""
    logging.info(
        "received add_event media group message %s from %s",
        message.message_id,
        message.from_user.id,
    )
    gid = message.media_group_id
    if not gid:
        return
    if gid in processed_media_groups:
        logging.info("skip already processed album %s", gid)
        return
    images = await extract_images(message, bot)
    state = pending_albums.get(gid)
    if not state:
        if len(pending_albums) >= MAX_PENDING_ALBUMS:
            old_gid, old_state = min(
                pending_albums.items(), key=lambda kv: kv[1].created
            )
            if old_state.timer:
                old_state.timer.cancel()
            age = int(_time.monotonic() - old_state.created)
            logging.info(
                "album_drop_no_caption gid=%s buf_size=%d age_s=%d",
                old_gid,
                len(old_state.images),
                age,
            )
            pending_albums.pop(old_gid, None)
        state = AlbumState(images=[])
        state.timer = asyncio.create_task(_drop_album_after_ttl(gid))
        pending_albums[gid] = state
    seq = message.message_id or int(message.date.timestamp())
    img_count = len(images or [])
    if images and len(state.images) < MAX_ALBUM_IMAGES:
        add = min(img_count, MAX_ALBUM_IMAGES - len(state.images))
        file_uid = None
        if message.photo:
            file_uid = message.photo[-1].file_unique_id
        elif (
            message.document
            and message.document.mime_type
            and message.document.mime_type.startswith("image/")
        ):
            file_uid = message.document.file_unique_id
        for data, name in images[:add]:
            state.images.append(
                AlbumImage(data=data, name=name, seq=seq, file_unique_id=file_uid)
            )
    text = message.text or message.caption
    logging.info(
        "album_collect gid=%s seq=%s msg_id=%s has_text=%s images_in_msg=%d buf_size_after=%d",
        gid,
        seq,
        message.message_id,
        bool(text),
        len(images or []),
        len(state.images),
    )
    if text and not state.text:
        state.text = text
        state.html, state.html_mode = ensure_html_text(message)
        state.message = message
        if state.timer:
            state.timer.cancel()
        logging.info(
            "album_caption_seen gid=%s delay_ms=%d",
            gid,
            ALBUM_FINALIZE_DELAY_MS,
        )
        state.timer = asyncio.create_task(
            _finalize_album_after_delay(gid, db, bot)
        )


async def telegraph_test():
    token = get_telegraph_token()
    if not token:
        print("Unable to obtain Telegraph token")
        return
    tg = Telegraph(access_token=token)
    page = await telegraph_create_page(
        tg, "Test Page", html_content="<p>test</p>", caller="event_pipeline"
    )
    logging.info("Created %s", page["url"])
    print("Created", page["url"])
    await telegraph_edit_page(
        tg,
        page["path"],
        title="Test Page",
        html_content="<p>updated</p>",
        caller="event_pipeline",
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
            catbox_msg = ""
        else:
            catbox_urls, catbox_msg = await upload_images(images)
            urls = catbox_urls
        has_cover = "<img" in html_content
        if has_cover:
            cover: list[str] = []
            tail = urls
        else:
            cover = urls[:1]
            tail = urls[1:]
        if cover:
            cover_html = f'<figure><img src="{html.escape(cover[0])}"/></figure>'
            html_content = cover_html + html_content
        new_html = normalize_hashtag_dates(new_html)
        cleaned = re.sub(r"</?tg-(?:emoji|spoiler)[^>]*>", "", new_html)
        cleaned = cleaned.replace(
            "\U0001f193\U0001f193\U0001f193\U0001f193", "Бесплатно"
        )
        new_block = (
            f"<p>{CONTENT_SEPARATOR}</p><p>" + cleaned.replace("\n", "<br/>") + "</p>"
        )
        hr_idx = html_content.lower().rfind("<hr")
        if hr_idx != -1:
            hr_end = html_content.find(">", hr_idx)
            if hr_end != -1:
                html_content = html_content[:hr_idx] + new_block + html_content[hr_idx:]
            else:
                html_content += new_block
        else:
            html_content += new_block
        nav_html = None
        if db:
            nav_html = await build_month_nav_html(db)
            html_content = apply_month_nav(html_content, nav_html)
        existing_imgs = html_content.count("<img")
        for url in tail:
            html_content += f'<img src="{html.escape(url)}"/>'
        total_imgs = existing_imgs + len(tail)
        if nav_html and total_imgs >= 2:
            html_content += nav_html
        html_content = apply_footer_link(html_content)
        html_content = lint_telegraph_html(html_content)
        logging.info(
            "Editing telegraph page %s", path,
        )
        await telegraph_edit_page(
            tg,
            path,
            title=title,
            html_content=html_content,
            caller="event_pipeline",
        )
        logging.info(
            "Updated telegraph page %s", path,
        )
        logging.info(
            "update_source_page: cover=%d tail=%d nav_dup=%s",
            len(cover),
            len(tail),
            bool(nav_html and total_imgs >= 2),
        )
        return catbox_msg, len(urls)
    except Exception as e:
        logging.error("Failed to update telegraph page: %s", e)
        return f"error: {e}", 0


async def update_source_page_ics(event_id: int, db: Database, url: str | None):
    """Insert or remove the ICS link in a Telegraph page."""
    async with db.get_session() as session:
        ev = await session.get(Event, event_id)
    if not ev or not ev.telegraph_path:
        return
    token = get_telegraph_token()
    if not token:
        logging.error("Telegraph token unavailable")
        return
    tg = Telegraph(access_token=token)
    path = ev.telegraph_path
    title = ev.title or "Event"
    try:
        logging.info("Editing telegraph ICS for %s", path)
        page = await telegraph_call(tg.get_page, path, return_html=True)
        html_content = page.get("content") or page.get("content_html") or ""
        html_content = apply_ics_link(html_content, url)
        html_content = apply_footer_link(html_content)
        await telegraph_edit_page(
            tg,
            path,
            title=title,
            html_content=html_content,
            caller="event_pipeline",
            eid=ev.id,
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
    posters = await _fetch_event_posters(event.id, db)
    poster_texts = await get_event_poster_texts(event.id, db, posters=posters)
    poster_summary = _summarize_event_posters(posters)
    try:
        parse_kwargs: dict[str, Any] = {}
        if poster_texts:
            parse_kwargs["poster_texts"] = poster_texts
        if poster_summary:
            parse_kwargs["poster_summary"] = poster_summary
        parsed = await parse_event_via_4o(text, **parse_kwargs)
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


@dataclass(slots=True)
class SourcePageEventSummary:
    date: str | None = None
    end_date: str | None = None
    time: str | None = None
    event_type: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    city: str | None = None
    ticket_price_min: int | None = None
    ticket_price_max: int | None = None
    ticket_link: str | None = None
    is_free: bool = False


def _format_summary_anchor_text(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = (parsed.netloc or "").strip()
    if not host:
        return url
    host = host.rstrip("/")
    path = (parsed.path or "").strip()
    path = path.rstrip("/")
    if path:
        display = f"{host}{path}"
    else:
        display = host
    if len(display) > 48:
        display = display[:47] + "…"
    return display or url


def _render_summary_anchor(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        return ""
    href = cleaned
    try:
        parsed = urlparse(cleaned)
    except ValueError:
        parsed = None
    if parsed and not parsed.scheme:
        if cleaned.startswith("//"):
            href = "https:" + cleaned
        else:
            href = "https://" + cleaned.lstrip("/")
    text = _format_summary_anchor_text(href)
    return f'<a href="{html.escape(href)}">{html.escape(text)}</a>'


def _format_ticket_price(min_price: int | None, max_price: int | None) -> str:
    if (
        min_price is not None
        and max_price is not None
        and min_price != max_price
    ):
        return f"от {min_price} до {max_price} руб."
    if min_price is not None:
        return f"{min_price} руб."
    if max_price is not None:
        return f"{max_price} руб."
    return ""


async def _build_source_summary_block(
    event_summary: SourcePageEventSummary | None,
) -> str:
    if not event_summary:
        return ""

    parts: list[str] = []

    start_date: date | None = None
    try:
        if event_summary.date:
            start_date = date.fromisoformat(event_summary.date)
    except ValueError:
        start_date = None

    if start_date:
        default_date_str = f"{start_date.day} {MONTHS[start_date.month - 1]}"
    elif event_summary.date:
        try:
            _, month, day = event_summary.date.split("-")
            default_date_str = f"{int(day)} {MONTHS[int(month) - 1]}"
        except Exception:
            default_date_str = event_summary.date
    else:
        default_date_str = ""

    end_date_obj: date | None = None
    if event_summary.end_date:
        try:
            end_date_obj = date.fromisoformat(event_summary.end_date)
        except ValueError:
            end_date_obj = None

    ongoing_exhibition = (
        (event_summary.event_type or "").strip().casefold() == "выставка"
        and start_date is not None
        and end_date_obj is not None
        and start_date <= date.today() <= end_date_obj
    )

    if ongoing_exhibition and end_date_obj is not None:
        end_month_name = MONTHS[end_date_obj.month - 1]
        year_suffix = ""
        if start_date.year != end_date_obj.year:
            year_suffix = f" {end_date_obj.year}"
        date_line = f"🗓 по {end_date_obj.day} {end_month_name}{year_suffix}"
    elif default_date_str:
        time_part = ""
        if event_summary.time and event_summary.time != "00:00":
            time_part = f" ⏰ {event_summary.time}"
        date_line = f"🗓 {default_date_str}{time_part}"
    else:
        date_line = ""

    if date_line.strip():
        parts.append(f"<p>{html.escape(date_line.strip())}</p>")

    location_parts: list[str] = []
    existing_normalized: set[str] = set()
    for part in (event_summary.location_name, event_summary.location_address):
        if part and part.strip():
            location_parts.append(part)
            normalized = _normalize_location_part(part)
            if normalized:
                existing_normalized.add(normalized)
    if event_summary.city and event_summary.city.strip():
        city_normalized = _normalize_location_part(event_summary.city)
        if not city_normalized or city_normalized not in existing_normalized:
            location_parts.append(event_summary.city)
            if city_normalized:
                existing_normalized.add(city_normalized)

    location_line = ""
    if location_parts:
        try:
            location_text = await build_short_vk_location(location_parts)
        except Exception:
            location_text = ", ".join(part.strip() for part in location_parts if part.strip())
        if location_text.strip():
            location_line = f"📍 {location_text.strip()}"
    if location_line:
        parts.append(f"<p>{html.escape(location_line)}</p>")

    ticket_line = ""
    anchor_html = ""
    link_value = (event_summary.ticket_link or "").strip()
    price_text = _format_ticket_price(
        event_summary.ticket_price_min, event_summary.ticket_price_max
    )
    if event_summary.is_free:
        ticket_line = "🆓 Бесплатно"
        if link_value:
            ticket_line += ", по регистрации"
            anchor_html = _render_summary_anchor(link_value)
    elif link_value:
        if price_text:
            ticket_line = f"🎟 Билеты {price_text}"
        else:
            ticket_line = "🎟 Билеты"
        anchor_html = _render_summary_anchor(link_value)
    elif price_text:
        ticket_line = f"🎟 Билеты {price_text}"

    if ticket_line:
        parts.append(f"<p>{html.escape(ticket_line)}</p>")
    if anchor_html:
        parts.append(f"<p>{anchor_html}</p>")

    return "".join(parts)


async def build_source_page_content(
    title: str,
    text: str,
    source_url: str | None,
    html_text: str | None = None,
    media: list[tuple[bytes, str]] | tuple[bytes, str] | None = None,
    ics_url: str | None = None,
    db: Database | None = None,
    *,
    event_summary: SourcePageEventSummary | None = None,
    display_link: bool = True,
    catbox_urls: list[str] | None = None,
    image_mode: Literal["tail", "inline"] = "tail",
    page_mode: Literal["default", "history"] = "default",
) -> tuple[str, str, int]:
    if image_mode not in {"tail", "inline"}:
        raise ValueError(f"unknown image_mode={image_mode}")
    if page_mode not in {"default", "history"}:
        raise ValueError(f"unknown page_mode={page_mode}")
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
        input_count = len(catbox_urls)
    else:
        catbox_urls, catbox_msg = await upload_images(images)
        urls = catbox_urls
        input_count = 0
    # filter out video links and limit to first 12 images
    urls = [
        u for u in urls if not re.search(r"\.(?:mp4|webm|mkv|mov)(?:\?|$)", u, re.I)
    ][:12]
    cover = urls[:1]
    tail = urls[1:]
    if cover:
        html_content += f'<figure><img src="{html.escape(cover[0])}"/></figure>'
        if ics_url:
            html_content += (
                f'<p>\U0001f4c5 <a href="{html.escape(ics_url)}">Добавить в календарь</a></p>'
            )
    else:
        if ics_url:
            html_content += (
                f'<p>\U0001f4c5 <a href="{html.escape(ics_url)}">Добавить в календарь</a></p>'
            )
    summary_html = await _build_source_summary_block(event_summary)
    summary_added = bool(summary_html)
    if summary_html:
        html_content += summary_html
    emoji_pat = re.compile(r"<tg-emoji[^>]*>(.*?)</tg-emoji>", re.DOTALL)
    spoiler_pat = re.compile(r"<tg-spoiler[^>]*>(.*?)</tg-spoiler>", re.DOTALL)
    tg_emoji_cleaned = 0
    tg_spoiler_unwrapped = 0
    paragraphs: list[str] = []
    blank_paragraph_re = re.compile(
        r"<p>(?:&nbsp;|&#8203;|\s|<br\s*/?>)*</p>", re.IGNORECASE
    )
    def _wrap_plain_chunks(raw_chunk: str) -> list[str]:
        chunk = raw_chunk.strip()
        if not chunk:
            return []
        normalized = re.sub(r"<br\s*/?>", "<br/>", chunk, flags=re.IGNORECASE)
        normalized = normalized.replace("\r", "")
        parts = [
            part.strip()
            for part in re.split(r"(?:<br/>\s*){2,}|\n{2,}", normalized)
            if part.strip()
        ]
        wrapped: list[str] = []
        for part in parts:
            segment = part.replace("\n", "<br/>")
            segment = re.sub(r"^(?:<br/>\s*)+", "", segment, flags=re.IGNORECASE)
            segment = re.sub(r"(?:<br/>\s*)+$", "", segment, flags=re.IGNORECASE)
            wrapped.append(f"<p>{segment}</p>")
        return wrapped

    def _split_paragraph_block(block: str) -> list[str]:
        match = re.match(r"(<p[^>]*>)(.*?)(</p>)", block, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return [block]
        start_tag, body, end_tag = match.groups()
        pieces = re.split(r"(?:<br\s*/?>\s*){2,}", body, flags=re.IGNORECASE)
        result: list[str] = []
        for piece in pieces:
            cleaned = piece.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^(?:<br\s*/?>\s*)+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"(?:<br\s*/?>\s*)+$", "", cleaned, flags=re.IGNORECASE)
            result.append(f"{start_tag}{cleaned}{end_tag}")
        return result or [block]

    def _fix_heading_paragraph_mismatches(raw: str) -> str:
        tag_re = re.compile(r"<(/?)(h[1-6]|p)([^>]*)>", re.IGNORECASE)
        block_tags = {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
        result: list[str] = []
        pos = 0
        stack: list[str] = []

        def _flush_stack() -> None:
            while stack:
                result.append(f"</{stack.pop()}>")

        for match in tag_re.finditer(raw):
            start, end = match.span()
            result.append(raw[pos:start])
            closing = match.group(1) == "/"
            tag = match.group(2).lower()
            tail = match.group(3) or ""
            if not closing:
                if tag in block_tags and stack:
                    _flush_stack()
                stack.append(tag)
                result.append(f"<{tag}{tail}>")
            else:
                if not stack:
                    pos = end
                    continue
                if tag not in stack:
                    _flush_stack()
                else:
                    while stack and stack[-1] != tag:
                        result.append(f"</{stack.pop()}>")
                    if stack and stack[-1] == tag:
                        stack.pop()
                        result.append(f"</{tag}>")
            pos = end
        result.append(raw[pos:])
        while stack:
            result.append(f"</{stack.pop()}>")
        return "".join(result)

    def _editor_html_blocks(raw: str) -> list[str]:
        text_value = raw.strip()
        if not text_value:
            return []
        looks_like_html = bool(re.search(r"<\w+[^>]*>", text_value))
        if looks_like_html:
            sanitized = sanitize_telegram_html(text_value)
            sanitized = linkify_for_telegraph(sanitized)
        else:
            sanitized = md_to_html(text_value)
        sanitized = re.sub(r"<(\/?)h[12](\b)", r"<\1h3\2", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"<br\s*/?>", "<br/>", sanitized, flags=re.IGNORECASE)
        sanitized = sanitize_telegram_html(sanitized)
        block_re = re.compile(
            r"<(?P<tag>h[1-6]|p|ul|ol|blockquote|pre|table|figure)[^>]*>.*?</(?P=tag)>|<hr\b[^>]*>|<img\b[^>]*>",
            re.IGNORECASE | re.DOTALL,
        )
        blocks: list[str] = []
        pos = 0
        for match in block_re.finditer(sanitized):
            start, end = match.span()
            if start > pos:
                blocks.extend(_wrap_plain_chunks(sanitized[pos:start]))
            block_value = match.group(0)
            if block_value.lower().startswith("<p"):
                blocks.extend(_split_paragraph_block(block_value))
            else:
                blocks.append(block_value)
            pos = end
        if pos < len(sanitized):
            blocks.extend(_wrap_plain_chunks(sanitized[pos:]))
        return [block for block in blocks if block.strip()]

    if html_text:
        html_text = strip_title(html_text)
        html_text = normalize_hashtag_dates(html_text)
        html_text = html_text.replace("\r\n", "\n")
        html_text = sanitize_telegram_html(html_text)
        for k, v in CUSTOM_EMOJI_MAP.items():
            html_text = html_text.replace(k, v)
        html_text = _fix_heading_paragraph_mismatches(html_text)
        paragraphs = _editor_html_blocks(html_text)
    else:
        clean_text = strip_title(text)
        clean_text = normalize_hashtag_dates(clean_text)
        tg_emoji_cleaned = len(emoji_pat.findall(clean_text))
        tg_spoiler_unwrapped = len(spoiler_pat.findall(clean_text))
        clean_text = emoji_pat.sub(r"\1", clean_text)
        clean_text = spoiler_pat.sub(r"\1", clean_text)
        for k, v in CUSTOM_EMOJI_MAP.items():
            clean_text = clean_text.replace(k, v)
        for line in clean_text.splitlines():
            escaped = html.escape(line)
            linked = linkify_for_telegraph(escaped)
            if linked.strip():
                paragraphs.append(f"<p>{linked}</p>")
            else:
                paragraphs.append(BODY_SPACER_HTML)
    if paragraphs:
        if summary_added:
            html_content += BODY_SPACER_HTML
        normalized_paragraphs: list[str] = []
        for block in paragraphs:
            block_stripped = block.strip()
            if blank_paragraph_re.fullmatch(block_stripped):
                normalized_paragraphs.append(BODY_SPACER_HTML)
            else:
                normalized_paragraphs.append(block)
        paragraphs = normalized_paragraphs
    inline_used = 0
    if page_mode == "history" and paragraphs:
        anchor_re = re.compile(r"<a\b[^>]*href=(['\"])(.*?)\1[^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)

        def _parse_href(raw: str | None) -> ParseResult | None:
            if not raw:
                return None
            candidate = html.unescape(raw).strip()
            if not candidate:
                return None
            parsed = urlparse(candidate)
            if parsed.scheme:
                final = parsed
            elif candidate.startswith("//"):
                final = urlparse("https:" + candidate)
            else:
                final = urlparse("https://" + candidate.lstrip("/"))
            return final

        def _normalized_parts(raw: str | None) -> tuple[str, str, str, str, str] | None:
            parsed = _parse_href(raw)
            if not parsed:
                return None
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            path = parsed.path or "/"
            if path != "/":
                path = path.rstrip("/")
            return host, path, parsed.params, parsed.query, parsed.fragment

        source_parts = _normalized_parts(source_url)

        def _is_vk_href(raw: str | None) -> bool:
            parsed = _parse_href(raw)
            if not parsed:
                return False
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host == "vk.com" or host.endswith(".vk.com")

        def _replace_anchor(match: re.Match[str]) -> str:
            href = match.group(2)
            parts = _normalized_parts(href)
            if source_parts and parts == source_parts:
                return match.group(0)
            if _is_vk_href(href):
                return match.group(3)
            return match.group(0)

        paragraphs = [anchor_re.sub(_replace_anchor, para) for para in paragraphs]
    if paragraphs:
        spacer = BODY_SPACER_HTML
        if image_mode == "inline" and tail:
            text_paragraphs = [p for p in paragraphs if p != BODY_SPACER_HTML]
            paragraph_count = len(text_paragraphs)
            image_count = len(tail)
            base_count = min(image_count, paragraph_count)
            positions: list[int] = []
            if base_count:
                step = paragraph_count / (base_count + 1)
                positions = [math.ceil((idx + 1) * step) for idx in range(base_count)]
            body_blocks: list[str] = []
            base_index = 0
            text_index = 0
            for block in paragraphs:
                if block == BODY_SPACER_HTML:
                    if body_blocks and body_blocks[-1] != spacer:
                        body_blocks.append(spacer)
                    continue
                if body_blocks and body_blocks[-1] != spacer:
                    body_blocks.append(spacer)
                body_blocks.append(block)
                text_index += 1
                inserted_for_para = False
                while base_index < base_count and positions[base_index] == text_index:
                    if not inserted_for_para:
                        body_blocks.append(spacer)
                        inserted_for_para = True
                    body_blocks.append(
                        f'<img src="{html.escape(tail[base_index])}"/>'
                    )
                    base_index += 1
            for extra_url in tail[base_index:]:
                if body_blocks:
                    body_blocks.append(spacer)
                body_blocks.append(f'<img src="{html.escape(extra_url)}"/>')
                base_index += 1
            inline_used = base_index
        else:
            body_blocks = []
            for block in paragraphs:
                if block == BODY_SPACER_HTML:
                    if body_blocks and body_blocks[-1] != spacer:
                        body_blocks.append(spacer)
                    continue
                if body_blocks and body_blocks[-1] != spacer:
                    body_blocks.append(spacer)
                body_blocks.append(block)
        html_content += "".join(body_blocks)
    elif image_mode == "inline" and tail:
        if summary_added:
            html_content += BODY_SPACER_HTML
        for extra_url in tail:
            html_content += f'<img src="{html.escape(extra_url)}"/>'
        inline_used = len(tail)
    if db and hasattr(db, "get_session") and text and text.strip():
        from models import Event, Festival
        from sqlalchemy import select

        async with db.get_session() as session:
            res = await session.execute(
                select(Event.festival, Festival.telegraph_path, Festival.telegraph_url)
                .join(Festival, Event.festival == Festival.name)
                .where(Event.source_text == text)
            )
            row = res.first()
            if row and row.telegraph_path:
                href = row.telegraph_url or f"https://telegra.ph/{row.telegraph_path.lstrip('/')}"
                html_content += (
                    BODY_SPACER_HTML
                    + f'<p>✨ <a href="{html.escape(href)}">{html.escape(row.festival)}</a></p>'
                    + BODY_SPACER_HTML
                )
    nav_html = None
    if db and page_mode != "history":
        nav_html = await build_month_nav_html(db)
        html_content = apply_month_nav(html_content, nav_html)
    if image_mode == "tail":
        for url in tail:
            html_content += f'<img src="{html.escape(url)}"/>'
    if nav_html and len(urls) >= 2:
        html_content += nav_html
    if page_mode == "history" and display_link and source_url:
        html_content += f'<p><a href="{html.escape(source_url)}">Источник</a></p>'
    if page_mode == "history":
        html_content = html_content.replace(FOOTER_LINK_HTML, "").rstrip()
        html_content = re.sub(
            r'(?:<p>(?:&nbsp;|&#8203;)</p>)?<p><a href="https://t\.me/kgdstories">[^<]+</a></p>',
            "",
            html_content,
        ).rstrip()
        html_content += HISTORY_FOOTER_HTML
    else:
        html_content = apply_footer_link(html_content)
    html_content = lint_telegraph_html(html_content)
    mode = "html" if html_text else "plain"
    logging.info("SRC build mode=%s urls_total=%d input_urls=%d", mode, len(urls), input_count)
    logging.info(
        "html_mode=%s tg_emoji_cleaned=%d tg_spoiler_unwrapped=%d",
        LAST_HTML_MODE,
        tg_emoji_cleaned,
        tg_spoiler_unwrapped,
    )
    logging.info(
        "build_source_page_content: cover=%d tail=%d nav_dup=%s catbox_msg=%s",
        len(cover),
        len(tail),
        bool(nav_html and len(urls) >= 2),
        catbox_msg,
    )
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
    image_mode: Literal["tail", "inline"] = "tail",
    page_mode: Literal["default", "history"] = "default",
) -> tuple[str, str, str, int] | None:
    """Create a Telegraph page with the original event text."""
    if db and text and text.strip():
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
        image_mode=image_mode,
        page_mode=page_mode,
    )
    logging.info("SRC page compose uploaded=%d catbox_msg=%s", uploaded, catbox_msg)
    from telegraph.utils import html_to_nodes, InvalidHTML

    try:
        nodes = html_to_nodes(html_content)
    except InvalidHTML as exc:
        if not html_text:
            raise
        logging.warning(
            "Invalid HTML in source page, rebuilding without editor markup: %s", exc
        )
        html_content, catbox_msg, uploaded = await build_source_page_content(
            title,
            text,
            source_url,
            None,
            media,
            ics_url,
            db,
            display_link=display_link,
            catbox_urls=catbox_urls,
            image_mode=image_mode,
            page_mode=page_mode,
        )
        try:
            nodes = html_to_nodes(html_content)
        except InvalidHTML:
            logging.exception("Fallback source page content is still invalid")
            raise
    author_name = (
        "Полюбить Калининград Истории"
        if page_mode == "history"
        else "Полюбить Калининград Анонсы"
    )
    try:
        page = await telegraph_create_page(
            tg,
            title=title,
            author_name=author_name,
            content=nodes,
            return_content=False,
            caller="event_pipeline",
        )
    except Exception as e:
        logging.error("Failed to create telegraph page: %s", e)
        return None
    url = normalize_telegraph_url(page.get("url"))
    logging.info(
        "SRC page created title=%s uploaded=%d url=%s",
        title,
        uploaded,
        url,
    )
    return url, page.get("path"), catbox_msg, uploaded


def create_app() -> web.Application:
    global db
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

    async def help_wrapper(message: types.Message):
        await handle_help(message, db, bot)

    async def ocrtest_wrapper(message: types.Message):
        await handle_ocrtest(message, db, bot)

    async def ocr_detail_wrapper(callback: types.CallbackQuery):
        await vision_test.select_detail(callback, bot)

    async def ocr_photo_wrapper(message: types.Message):
        images = await extract_images(message, bot)
        await vision_test.handle_photo(message, bot, images)

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

    async def digest_wrapper(message: types.Message):
        await show_digest_menu(message, db, bot)

    async def digest_select_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_lectures(callback, db, bot)

    async def digest_select_masterclasses_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_masterclasses(callback, db, bot)

    async def digest_select_exhibitions_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_exhibitions(callback, db, bot)

    async def digest_select_psychology_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_psychology(callback, db, bot)

    async def digest_select_science_pop_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_science_pop(callback, db, bot)

    async def digest_select_kraevedenie_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_kraevedenie(callback, db, bot)

    async def digest_select_networking_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_networking(callback, db, bot)

    async def digest_select_entertainment_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_entertainment(callback, db, bot)

    async def digest_select_markets_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_markets(callback, db, bot)

    async def digest_select_theatre_classic_wrapper(
        callback: types.CallbackQuery,
    ):
        await handle_digest_select_theatre_classic(callback, db, bot)

    async def digest_select_theatre_modern_wrapper(
        callback: types.CallbackQuery,
    ):
        await handle_digest_select_theatre_modern(callback, db, bot)

    async def digest_select_meetups_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_meetups(callback, db, bot)

    async def digest_select_movies_wrapper(callback: types.CallbackQuery):
        await handle_digest_select_movies(callback, db, bot)

    async def digest_disabled_wrapper(callback: types.CallbackQuery):
        await callback.answer("Ещё не реализовано", show_alert=False)

    async def digest_toggle_wrapper(callback: types.CallbackQuery):
        await handle_digest_toggle(callback, bot)

    async def digest_refresh_wrapper(callback: types.CallbackQuery):
        await handle_digest_refresh(callback, bot)

    async def digest_send_wrapper(callback: types.CallbackQuery):
        await handle_digest_send(callback, db, bot)

    async def digest_notify_partners_wrapper(callback: types.CallbackQuery):
        await handle_digest_notify_partners(callback, db, bot)

    async def digest_hide_wrapper(callback: types.CallbackQuery):
        await handle_digest_hide(callback, bot)

    async def pages_wrapper(message: types.Message):
        await handle_pages(message, db, bot)

    async def pages_rebuild_wrapper(message: types.Message):
        await handle_pages_rebuild(message, db, bot)

    async def weekendimg_cmd_wrapper(message: types.Message):
        await handle_weekendimg_cmd(message, db, bot)

    async def weekendimg_cb_wrapper(callback: types.CallbackQuery):
        await handle_weekendimg_cb(callback, db, bot)

    async def weekendimg_photo_wrapper(message: types.Message):
        await handle_weekendimg_photo(message, db, bot)

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

    async def tourist_export_wrapper(message: types.Message):
        await handle_tourist_export(message, db, bot)

    async def telegraph_fix_author_wrapper(message: types.Message):
        await handle_telegraph_fix_author(message, db, bot)

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

    captcha_handler = partial(handle_vk_captcha, db=db, bot=bot)

    async def askloc_wrapper(callback: types.CallbackQuery):
        await handle_askloc(callback, db, bot)

    async def askcity_wrapper(callback: types.CallbackQuery):
        await handle_askcity(callback, db, bot)

    async def pages_rebuild_cb_wrapper(callback: types.CallbackQuery):
        await handle_pages_rebuild_cb(callback, db, bot)

    async def captcha_prompt_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_prompt(callback, db, bot)

    async def captcha_delay_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_delay(callback, db, bot)
    async def captcha_refresh_wrapper(callback: types.CallbackQuery):
        await handle_vk_captcha_refresh(callback, db, bot)

    async def menu_wrapper(message: types.Message):
        await handle_menu(message, db, bot)

    async def events_menu_wrapper(message: types.Message):
        await handle_events_menu(message, db, bot)

    async def events_date_wrapper(message: types.Message):
        await handle_events_date_message(message, db, bot)

    async def add_event_start_wrapper(message: types.Message):
        logging.info("add_event_start_wrapper start: user=%s", message.from_user.id)
        await handle_add_event_start(message, db, bot)

    async def add_festival_start_wrapper(message: types.Message):
        logging.info(
            "add_festival_start_wrapper start: user=%s", message.from_user.id
        )
        await handle_add_festival_start(message, db, bot)

    async def add_event_session_wrapper(message: types.Message):
        logging.info("add_event_session_wrapper start: user=%s", message.from_user.id)
        session_mode = add_event_sessions.get(message.from_user.id)
        if message.media_group_id:
            await handle_add_event_media_group(message, db, bot)
        else:
            await enqueue_add_event(
                message, db, bot, session_mode=session_mode
            )

    async def vk_link_cmd_wrapper(message: types.Message):
        logging.info("vk_link_cmd_wrapper start: user=%s", message.from_user.id)
        await handle_vk_link_command(message, db, bot)

    async def vk_cmd_wrapper(message: types.Message):
        await handle_vk_command(message, db, bot)

    async def vk_add_start_wrapper(message: types.Message):
        await handle_vk_add_start(message, db, bot)

    async def vk_add_msg_wrapper(message: types.Message):
        await handle_vk_add_message(message, db, bot)

    async def vk_list_wrapper(message: types.Message):
        await handle_vk_list(message, db, bot)

    async def vk_check_wrapper(message: types.Message):
        await handle_vk_check(message, db, bot)

    async def vk_delete_wrapper(callback: types.CallbackQuery):
        await handle_vk_delete_callback(callback, db, bot)

    async def vk_list_page_wrapper(callback: types.CallbackQuery):
        await handle_vk_list_page_callback(callback, db, bot)

    async def vk_rejected_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_rejected_callback(callback, db, bot)

    async def vk_settings_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_settings_callback(callback, db, bot)

    async def vk_dtime_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_dtime_callback(callback, db, bot)

    async def vk_dtime_msg_wrapper(message: types.Message):
        await handle_vk_dtime_message(message, db, bot)

    async def vk_ticket_link_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_ticket_link_callback(callback, db, bot)

    async def vk_ticket_link_msg_wrapper(message: types.Message):
        await handle_vk_ticket_link_message(message, db, bot)

    async def vk_location_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_location_callback(callback, db, bot)

    async def vk_location_msg_wrapper(message: types.Message):
        await handle_vk_location_message(message, db, bot)

    async def vk_next_wrapper(callback: types.CallbackQuery):
        await handle_vk_next_callback(callback, db, bot)

    async def vk_review_cb_wrapper(callback: types.CallbackQuery):
        await handle_vk_review_cb(callback, db, bot)

    async def vk_extra_msg_wrapper(message: types.Message):
        await handle_vk_extra_message(message, db, bot)

    async def tourist_note_wrapper(message: types.Message):
        await handle_tourist_note_message(message, db, bot)

    async def vk_shortpost_edit_msg_wrapper(message: types.Message):
        await handle_vk_shortpost_edit_message(message, db, bot)

    async def vk_crawl_now_wrapper(message: types.Message):
        await handle_vk_crawl_now(message, db, bot)

    async def vk_queue_wrapper(message: types.Message):
        await handle_vk_queue(message, db, bot)

    async def vk_requeue_imported_wrapper(message: types.Message):
        await handle_vk_requeue_imported(message, db, bot)
    async def status_wrapper(message: types.Message):
        await handle_status(message, db, bot, app)

    async def trace_wrapper(message: types.Message):
        await handle_trace(message, db, bot)

    async def last_errors_wrapper(message: types.Message):
        await handle_last_errors(message, db, bot)

    async def debug_wrapper(message: types.Message):
        await handle_debug(message, db, bot)

    async def queue_reap_wrapper(message: types.Message):
        await handle_queue_reap(message, db, bot)

    async def mem_wrapper(message: types.Message):
        await handle_mem(message, db, bot)

    async def festmerge_do_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_do_callback(callback, db, bot)

    async def festmerge_to_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_to_callback(callback, db, bot)

    async def festmerge_page_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_page_callback(callback, db, bot)

    async def festmerge_wrapper(callback: types.CallbackQuery):
        await handle_festmerge_callback(callback, db, bot)

    async def backfill_topics_wrapper(message: types.Message):
        await handle_backfill_topics(message, db, bot)

    async def festivals_fix_nav_wrapper(message: types.Message):
        await handle_festivals_fix_nav(message, db, bot)

    async def ics_fix_nav_wrapper(message: types.Message):
        await handle_ics_fix_nav(message, db, bot)

    dp.message.register(help_wrapper, Command("help"))
    dp.message.register(ocrtest_wrapper, Command("ocrtest"))
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
        or c.data.startswith("makefest:")
        or c.data.startswith("makefest_create:")
        or c.data.startswith("makefest_bind:")
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
        or c.data.startswith("festpage:")
        or c.data.startswith("festdel:")
        or c.data.startswith("setfest:")
        or c.data.startswith("festdays:")
        or c.data.startswith("festimgs:")
        or c.data.startswith("festsetcover:")
        or c.data.startswith("festcover:")
        or c.data.startswith("requeue:")
        or c.data.startswith("tourist:")
    ,
    )
    dp.callback_query.register(
        festmerge_do_wrapper, lambda c: c.data and c.data.startswith("festmerge_do:")
    )
    dp.callback_query.register(
        festmerge_to_wrapper, lambda c: c.data and c.data.startswith("festmerge_to:")
    )
    dp.callback_query.register(
        festmerge_page_wrapper, lambda c: c.data and c.data.startswith("festmergep:")
    )
    dp.callback_query.register(
        festmerge_wrapper, lambda c: c.data and c.data.startswith("festmerge:")
    )
    dp.callback_query.register(
        ocr_detail_wrapper,
        lambda c: c.data and c.data.startswith("ocr:detail:"),
    )
    dp.callback_query.register(askloc_wrapper, lambda c: c.data == "askloc")
    dp.callback_query.register(askcity_wrapper, lambda c: c.data == "askcity")
    dp.callback_query.register(
        pages_rebuild_cb_wrapper, lambda c: c.data and c.data.startswith("pages_rebuild:")
    )
    dp.callback_query.register(captcha_prompt_wrapper, lambda c: c.data == "captcha_input")
    dp.callback_query.register(captcha_delay_wrapper, lambda c: c.data == "captcha_delay")
    dp.callback_query.register(captcha_refresh_wrapper, lambda c: c.data == "captcha_refresh")
    dp.callback_query.register(
        vk_rejected_cb_wrapper, lambda c: c.data and c.data.startswith("vkrejected:")
    )
    dp.message.register(tz_wrapper, Command("tz"))
    dp.message.register(
        add_event_session_wrapper, lambda m: m.from_user.id in add_event_sessions
    )
    dp.message.register(
        ocr_photo_wrapper,
        lambda m: vision_test.is_waiting(m.from_user.id),
        F.photo | F.document,
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
    dp.message.register(captcha_handler, Command("captcha"))
    dp.message.register(captcha_handler, F.reply_to_message)
    dp.message.register(menu_wrapper, Command("menu"))
    dp.message.register(events_menu_wrapper, lambda m: m.text == MENU_EVENTS)
    dp.message.register(events_date_wrapper, lambda m: m.from_user.id in events_date_sessions)
    dp.message.register(add_event_start_wrapper, lambda m: m.text == MENU_ADD_EVENT)
    dp.message.register(
        add_festival_start_wrapper, lambda m: m.text == MENU_ADD_FESTIVAL
    )
    dp.message.register(vk_link_cmd_wrapper, Command("vklink"))
    dp.message.register(vk_cmd_wrapper, Command("vk"))
    dp.message.register(vk_crawl_now_wrapper, Command("vk_crawl_now"))
    dp.message.register(vk_queue_wrapper, Command("vk_queue"))
    dp.message.register(vk_requeue_imported_wrapper, Command("vk_requeue_imported"))
    dp.message.register(vk_add_start_wrapper, lambda m: m.text == VK_BTN_ADD_SOURCE)
    dp.message.register(vk_list_wrapper, lambda m: m.text == VK_BTN_LIST_SOURCES)
    dp.message.register(vk_check_wrapper, lambda m: m.text == VK_BTN_CHECK_EVENTS)
    dp.message.register(vk_queue_wrapper, lambda m: m.text == VK_BTN_QUEUE_SUMMARY)
    dp.message.register(vk_add_msg_wrapper, lambda m: m.from_user.id in vk_add_source_sessions)
    dp.message.register(vk_extra_msg_wrapper, lambda m: m.from_user.id in vk_review_extra_sessions)
    dp.message.register(
        tourist_note_wrapper, lambda m: m.from_user.id in tourist_note_sessions
    )
    dp.message.register(
        vk_shortpost_edit_msg_wrapper,
        lambda m: m.from_user.id in vk_shortpost_edit_sessions,
    )
    dp.message.register(partner_info_wrapper, lambda m: m.from_user.id in partner_info_sessions)
    dp.message.register(channels_wrapper, Command("channels"))
    dp.message.register(reg_daily_wrapper, Command("regdailychannels"))
    dp.message.register(daily_wrapper, Command("daily"))
    dp.message.register(exhibitions_wrapper, Command("exhibitions"))
    dp.message.register(digest_wrapper, Command("digest"))
    dp.callback_query.register(
        digest_select_wrapper, lambda c: c.data.startswith("digest:select:lectures:")
    )
    dp.callback_query.register(
        digest_select_masterclasses_wrapper,
        lambda c: c.data.startswith("digest:select:masterclasses:"),
    )
    dp.callback_query.register(
        digest_select_exhibitions_wrapper,
        lambda c: c.data.startswith("digest:select:exhibitions:"),
    )
    dp.callback_query.register(
        digest_select_psychology_wrapper,
        lambda c: c.data.startswith("digest:select:psychology:"),
    )
    dp.callback_query.register(
        digest_select_science_pop_wrapper,
        lambda c: c.data.startswith("digest:select:science_pop:"),
    )
    dp.callback_query.register(
        digest_select_kraevedenie_wrapper,
        lambda c: c.data.startswith("digest:select:kraevedenie:"),
    )
    dp.callback_query.register(
        digest_select_networking_wrapper,
        lambda c: c.data.startswith("digest:select:networking:"),
    )
    dp.callback_query.register(
        digest_select_entertainment_wrapper,
        lambda c: c.data.startswith("digest:select:entertainment:"),
    )
    dp.callback_query.register(
        digest_select_markets_wrapper,
        lambda c: c.data.startswith("digest:select:markets:"),
    )
    dp.callback_query.register(
        digest_select_theatre_classic_wrapper,
        lambda c: c.data.startswith("digest:select:theatre_classic:"),
    )
    dp.callback_query.register(
        digest_select_theatre_modern_wrapper,
        lambda c: c.data.startswith("digest:select:theatre_modern:"),
    )
    dp.callback_query.register(
        digest_select_meetups_wrapper,
        lambda c: c.data.startswith("digest:select:meetups:"),
    )
    dp.callback_query.register(
        digest_select_movies_wrapper,
        lambda c: c.data.startswith("digest:select:movies:"),
    )
    dp.callback_query.register(
        digest_disabled_wrapper, lambda c: c.data == "digest:disabled"
    )
    dp.callback_query.register(
        digest_toggle_wrapper, lambda c: c.data.startswith("dg:t:")
    )
    dp.callback_query.register(
        digest_refresh_wrapper, lambda c: c.data.startswith("dg:r:")
    )
    dp.callback_query.register(
        digest_send_wrapper, lambda c: c.data.startswith("dg:s:")
    )
    dp.callback_query.register(
        digest_notify_partners_wrapper, lambda c: c.data.startswith("dg:np:")
    )
    dp.callback_query.register(
        digest_hide_wrapper, lambda c: c.data.startswith("dg:x:")
    )
    dp.message.register(fest_wrapper, Command("fest"))

    dp.message.register(weekendimg_cmd_wrapper, Command("weekendimg"))
    dp.callback_query.register(
        weekendimg_cb_wrapper, lambda c: c.data and c.data.startswith("weekimg:")
    )
    dp.message.register(
        weekendimg_photo_wrapper, lambda m: m.from_user.id in weekend_img_wait
    )

    dp.message.register(pages_wrapper, Command("pages"))
    dp.message.register(pages_rebuild_wrapper, Command("pages_rebuild"))
    dp.message.register(stats_wrapper, Command("stats"))
    dp.message.register(status_wrapper, Command("status"))
    dp.message.register(trace_wrapper, Command("trace"))
    dp.message.register(last_errors_wrapper, Command("last_errors"))
    dp.message.register(debug_wrapper, Command("debug"))
    dp.message.register(queue_reap_wrapper, Command("queue_reap"))
    dp.message.register(mem_wrapper, Command("mem"))
    dp.message.register(backfill_topics_wrapper, Command("backfill_topics"))
    dp.message.register(festivals_fix_nav_wrapper, Command("festivals_fix_nav"))
    dp.message.register(festivals_fix_nav_wrapper, Command("festivals_nav_dedup"))
    dp.message.register(ics_fix_nav_wrapper, Command("ics_fix_nav"))
    dp.message.register(users_wrapper, Command("users"))
    dp.message.register(dumpdb_wrapper, Command("dumpdb"))
    dp.message.register(tourist_export_wrapper, Command("tourist_export"))
    dp.message.register(telegraph_fix_author_wrapper, Command("telegraph_fix_author"))
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
        vk_dtime_msg_wrapper, lambda m: m.from_user.id in vk_default_time_sessions
    )
    dp.message.register(
        vk_ticket_link_msg_wrapper,
        lambda m: m.from_user.id in vk_default_ticket_link_sessions,
    )
    dp.message.register(
        vk_location_msg_wrapper,
        lambda m: m.from_user.id in vk_default_location_sessions,
    )
    dp.callback_query.register(
        vk_list_page_wrapper, lambda c: c.data.startswith("vksrcpage:")
    )
    dp.callback_query.register(vk_delete_wrapper, lambda c: c.data.startswith("vkdel:"))
    dp.callback_query.register(
        vk_settings_cb_wrapper, lambda c: c.data.startswith("vkset:")
    )
    dp.callback_query.register(vk_dtime_cb_wrapper, lambda c: c.data.startswith("vkdt:"))
    dp.callback_query.register(
        vk_ticket_link_cb_wrapper, lambda c: c.data.startswith("vklink:")
    )
    dp.callback_query.register(
        vk_location_cb_wrapper, lambda c: c.data.startswith("vkloc:")
    )
    dp.callback_query.register(vk_next_wrapper, lambda c: c.data.startswith("vknext:"))
    dp.callback_query.register(vk_review_cb_wrapper, lambda c: c.data and c.data.startswith("vkrev:"))
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
    app.router.add_get("/metrics", metrics_handler)

    async def on_startup(app: web.Application):
        await init_db_and_scheduler(app, db, bot, webhook)

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


def _normalize_event_description(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


async def _handle_digest_select(
    callback: types.CallbackQuery,
    db: Database,
    bot: Bot,
    *,
    digest_type: str,
    preview_builder: Callable[[str, Database, datetime], Awaitable[tuple[str, List[str], int, List[Event], List[str]]]],
    items_noun: str,
    panel_text: str,
) -> None:
    parts = callback.data.split(":")
    if len(parts) != 4 or parts[2] != digest_type:
        return
    digest_id = parts[3]

    chat_id = callback.message.chat.id if callback.message else None
    if chat_id is None:
        await callback.answer()
        return

    logging.info(
        "digest.type.selected digest_id=%s type=%s chat_id=%s user_id=%s callback_id=%s",
        digest_id,
        digest_type,
        chat_id,
        callback.from_user.id,
        callback.id,
    )

    offset = await get_tz_offset(db)
    tz = offset_to_timezone(offset)
    now = datetime.now(tz).replace(tzinfo=None)

    intro, lines, horizon, events, norm_titles = await preview_builder(
        digest_id, db, now
    )
    if not events:
        await bot.send_message(
            chat_id,
            f"Пока ничего нет в ближайшие {horizon} дней с учётом правила “+2 часа”.",
        )
        return

    items: List[dict] = []
    for idx, (ev, line, norm_title) in enumerate(
        zip(events, lines, norm_titles), start=1
    ):
        cover_url = None
        if ev.telegraph_url:
            try:
                covers = await extract_catbox_covers_from_telegraph(
                    ev.telegraph_url, event_id=ev.id
                )
                cover_url = covers[0] if covers else None
            except Exception:
                cover_url = None
        norm_topics = normalize_topics(getattr(ev, "topics", []))
        items.append(
            {
                "event_id": ev.id,
                "creator_id": ev.creator_id,
                "index": idx,
                "title": ev.title,
                "norm_title": norm_title,
                "event_type": ev.event_type,
                "norm_description": _normalize_event_description(ev.description),
                "norm_topics": norm_topics,
                "date": ev.date,
                "end_date": ev.end_date,
                "link": pick_display_link(ev),
                "cover_url": cover_url,
                "line_html": line,
            }
        )

    async with db.get_session() as session_db:
        result = await session_db.execute(
            select(Channel).where(Channel.daily_time.is_not(None))
        )
        channels = result.scalars().all()

    session_data = {
        "chat_id": chat_id,
        "preview_msg_ids": [],
        "panel_msg_id": None,
        "items": items,
        "intro_html": intro,
        "footer_html": '<a href="https://t.me/kenigevents">Полюбить Калининград | Анонсы</a>',
        "excluded": set(),
        "horizon_days": horizon,
        "channels": [
            {
                "channel_id": ch.channel_id,
                "name": ch.title or ch.username or str(ch.channel_id),
                "username": ch.username,
            }
            for ch in channels
        ],
        "items_noun": items_noun,
        "panel_text": panel_text,
        "digest_type": digest_type,
    }

    digest_preview_sessions[digest_id] = session_data

    caption, attach, vis_len, kept = await _send_preview(
        session_data, digest_id, bot
    )

    logging.info(
        "digest.panel.new digest_id=%s type=%s total=%s caption_len_visible=%s attached=%s",
        digest_id,
        digest_type,
        len(items),
        vis_len,
        int(attach),
    )

    await callback.answer()


async def handle_digest_select_lectures(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="lectures",
        preview_builder=build_lectures_digest_preview,
        items_noun="лекций",
        panel_text="Управление дайджестом лекций\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_masterclasses(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="masterclasses",
        preview_builder=build_masterclasses_digest_preview,
        items_noun="мастер-классов",
        panel_text="Управление дайджестом мастер-классов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_exhibitions(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="exhibitions",
        preview_builder=build_exhibitions_digest_preview,
        items_noun="выставок",
        panel_text="Управление дайджестом выставок\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_psychology(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="psychology",
        preview_builder=build_psychology_digest_preview,
        items_noun="психологических событий",
        panel_text="Управление дайджестом психологии\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_science_pop(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="science_pop",
        preview_builder=build_science_pop_digest_preview,
        items_noun="научно-популярных событий",
        panel_text="Управление дайджестом научпопа\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_kraevedenie(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="kraevedenie",
        preview_builder=build_kraevedenie_digest_preview,
        items_noun="краеведческих событий",
        panel_text="Управление дайджестом краеведения\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_networking(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="networking",
        preview_builder=build_networking_digest_preview,
        items_noun="нетворкингов",
        panel_text="Управление дайджестом нетворкингов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_entertainment(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="entertainment",
        preview_builder=build_entertainment_digest_preview,
        items_noun="развлечений",
        panel_text="Управление дайджестом развлечений\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_markets(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="markets",
        preview_builder=build_markets_digest_preview,
        items_noun="маркетов",
        panel_text="Управление дайджестом маркетов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_theatre_classic(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="theatre_classic",
        preview_builder=build_theatre_classic_digest_preview,
        items_noun="классических спектаклей",
        panel_text="Управление дайджестом классического театра\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_theatre_modern(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="theatre_modern",
        preview_builder=build_theatre_modern_digest_preview,
        items_noun="современных спектаклей",
        panel_text="Управление дайджестом современного театра\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_meetups(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="meetups",
        preview_builder=build_meetups_digest_preview,
        items_noun="встреч и клубов",
        panel_text="Управление дайджестом встреч и клубов\nВыключите лишнее и нажмите «Обновить превью».",
    )


async def handle_digest_select_movies(
    callback: types.CallbackQuery, db: Database, bot: Bot
) -> None:
    await _handle_digest_select(
        callback,
        db,
        bot,
        digest_type="movies",
        preview_builder=build_movies_digest_preview,
        items_noun="кинопоказов",
        panel_text="Управление дайджестом кинопоказов\nВыключите лишнее и нажмите «Обновить превью».",
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_telegraph":
        asyncio.run(telegraph_test())
    else:
        web.run_app(create_app(), port=int(os.getenv("PORT", 8080)))
