from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import List, Any
from datetime import datetime, timedelta

from db import Database

from sections import MONTHS_RU

# Crawl tuning parameters
VK_CRAWL_PAGE_SIZE = int(os.getenv("VK_CRAWL_PAGE_SIZE", "30"))
VK_CRAWL_MAX_PAGES_INC = int(os.getenv("VK_CRAWL_MAX_PAGES_INC", "1"))
VK_CRAWL_OVERLAP_SEC = int(os.getenv("VK_CRAWL_OVERLAP_SEC", "300"))
VK_CRAWL_PAGE_SIZE_BACKFILL = int(os.getenv("VK_CRAWL_PAGE_SIZE_BACKFILL", "50"))
VK_CRAWL_MAX_PAGES_BACKFILL = int(os.getenv("VK_CRAWL_MAX_PAGES_BACKFILL", "3"))
VK_CRAWL_BACKFILL_DAYS = int(os.getenv("VK_CRAWL_BACKFILL_DAYS", "14"))
VK_CRAWL_BACKFILL_AFTER_IDLE_H = int(os.getenv("VK_CRAWL_BACKFILL_AFTER_IDLE_H", "24"))
VK_USE_PYMORPHY = os.getenv("VK_USE_PYMORPHY", "false").lower() == "true"

# optional pymorphy3 initialisation
MORPH = None
if VK_USE_PYMORPHY:  # pragma: no cover - optional dependency
    try:
        import pymorphy3

        MORPH = pymorphy3.MorphAnalyzer()
    except Exception:
        VK_USE_PYMORPHY = False

# Keyword patterns for regex-based matching
KEYWORD_PATTERNS = [
    r"лекци(я|и|й|е|ю|ями|ях)",
    r"спектакл(ь|я|ю|ем|е|и|ей|ям|ями|ях)",
    r"концерт(ы|а|у|е|ом|ов|ам|ами|ах)",
    r"фестивал(ь|я|ю|е|ем|и|ей|ям|ями|ях)|festival",
    r"м(?:а|а?стер)[-\s]?класс(ы|а|е|ом|ов|ам|ами|ах)|мк\b",
    r"воркшоп(ы|а|е|ом|ов|ам|ами|ах)|workshop",
    r"показ(ы|а|е|ом|ов|ам|ами|ах)|кинопоказ",
    r"лекто(р|рия|рий|рии|риями|риях)|кинолекторий",
    r"выставк(а|и|е|у|ой|ам|ами|ах)",
    r"экскурси(я|и|е|ю|ей|ям|ями|ях)",
    r"читк(а|и|е|у|ой|ам|ами|ах)",
    r"перформанс(ы|а|е|ом|ов|ам|ами|ах)",
    r"встреч(а|и|е|у|ей|ам|ами|ах)",
    r"бронировани(е|я|ю|ем)|билет(ы|а|ов)|регистраци(я|и|ю|ей)|афиш(а|и|е|у)",
]
KEYWORD_RE = re.compile(r"\b#?(?:" + "|".join(KEYWORD_PATTERNS) + r")\b", re.I | re.U)

# Canonical keywords for morphological mode
KEYWORD_LEMMAS = {
    "лекция",
    "спектакль",
    "концерт",
    "фестиваль",
    "мастер-класс",
    "воркшоп",
    "показ",
    "кинопоказ",
    "лекторий",
    "кинолекторий",
    "выставка",
    "экскурсия",
    "читка",
    "перформанс",
    "встреча",
    "бронирование",
    "билет",
    "регистрация",
    "афиша",
}

# Date/time patterns used for quick detection
MONTH_NAMES_DET = "|".join(sorted(re.escape(m) for m in MONTHS_RU.keys()))
DATE_PATTERNS = [
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b",
    r"\b\d{1,2}[–-]\d{1,2}(?:[./]\d{1,2})\b",
    rf"\b\d{{1,2}}\s+(?:{MONTH_NAMES_DET})\.?\b",
    r"\b(понед(?:ельник)?|вторник|сред(?:а)?|четверг|пятниц(?:а)?|суббот(?:а)?|воскресень(?:е|е)|пн|вт|ср|чт|пт|сб|вс)\b",
    r"\b(сегодня|завтра|послезавтра|в эти выходные)\b",
    r"\b([01]?\d|2[0-3])[:.][0-5]\d\b",
    r"\bв\s*([01]?\d|2[0-3])\s*(?:ч|час(?:а|ов)?)\b",
    r"\bс\s*([01]?\d|2[0-3])(?:[:.][0-5]\d)?\s*до\s*([01]?\d|2[0-3])(?:[:.][0-5]\d)?\b",
    r"\b20\d{2}\b",
]

COMPILED_DATE_PATTERNS = [re.compile(p, re.I | re.U) for p in DATE_PATTERNS]

NUM_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b")
DATE_RANGE_RE = re.compile(r"\b(\d{1,2})[–-](\d{1,2})(?:[./](\d{1,2}))\b")
MONTH_NAME_RE = re.compile(r"\b(\d{1,2})\s+([а-яё.]+)\b", re.I)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")
TIME_H_RE = re.compile(r"\bв\s*([01]?\d|2[0-3])\s*(?:ч|час(?:а|ов)?)\b")
TIME_RANGE_RE = re.compile(
    r"\bс\s*([01]?\d|2[0-3])(?:[:.](\d{2}))?\s*до\s*([01]?\d|2[0-3])(?:[:.](\d{2}))?\b"
)
DOW_RE = re.compile(
    r"\b(понед(?:ельник)?|вторник|сред(?:а)?|четверг|пятниц(?:а)?|суббот(?:а)?|воскресень(?:е|е)|пн|вт|ср|чт|пт|сб|вс)\b",
    re.I,
)
WEEKEND_RE = re.compile(r"в\s+эти\s+выходны", re.I)

# Maximum age of a past date mention that should not be rolled over to the next year
RECENT_PAST_THRESHOLD = timedelta(days=92)

# cumulative processing time for VK event intake (seconds)
processing_time_seconds_total: float = 0.0


def match_keywords(text: str) -> tuple[bool, list[str]]:
    """Return True and list of matched keywords if any are found."""
    text_low = text.lower()
    if VK_USE_PYMORPHY and MORPH:
        tokens = re.findall(r"\w+", text_low)
        matched: list[str] = []
        for t in tokens:
            lemma = MORPH.parse(t)[0].normal_form
            if lemma in KEYWORD_LEMMAS and lemma not in matched:
                matched.append(lemma)
        return bool(matched), matched
    matched = [m.group(0).lstrip("#") for m in KEYWORD_RE.finditer(text_low)]
    return bool(matched), matched


def detect_date(text: str) -> bool:
    """Heuristically detect a date or time mention in the text."""
    return any(p.search(text) for p in COMPILED_DATE_PATTERNS)


def extract_event_ts_hint(text: str, default_time: str | None = None) -> int | None:
    """Return Unix timestamp for the nearest future datetime mentioned in text."""
    from main import LOCAL_TZ

    now = datetime.now(LOCAL_TZ)
    text_low = text.lower()

    day = month = year = None
    m = NUM_DATE_RE.search(text_low)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if m.group(3):
            y = m.group(3)
            year = int("20" + y if len(y) == 2 else y)
    else:
        m = DATE_RANGE_RE.search(text_low)
        if m:
            day = int(m.group(1))
            month = int(m.group(3))
        else:
            m = MONTH_NAME_RE.search(text_low)
            if m:
                day = int(m.group(1))
                mon_word = m.group(2).rstrip(".")
                month = MONTHS_RU.get(mon_word)
                y = re.search(r"\b20\d{2}\b", text_low[m.end():])
                if y:
                    year = int(y.group(0))

    if day is None or month is None:
        if "сегодня" in text_low:
            dt = now
        elif "завтра" in text_low:
            dt = now + timedelta(days=1)
        elif "послезавтра" in text_low:
            dt = now + timedelta(days=2)
        else:
            dow_m = DOW_RE.search(text_low)
            if dow_m:
                dow_map = {
                    "понедельник": 0,
                    "понед": 0,
                    "пн": 0,
                    "вторник": 1,
                    "вт": 1,
                    "среда": 2,
                    "ср": 2,
                    "четверг": 3,
                    "чт": 3,
                    "пятница": 4,
                    "пт": 4,
                    "суббота": 5,
                    "сб": 5,
                    "воскресенье": 6,
                    "вс": 6,
                }
                key = dow_m.group(1).lower().rstrip(".")
                dow = dow_map.get(key)
                if dow is None:
                    dow = dow_map.get(key[:2])
                days_ahead = (dow - now.weekday()) % 7
                dt = now + timedelta(days=days_ahead)
            elif WEEKEND_RE.search(text_low):
                days_ahead = (5 - now.weekday()) % 7
                dt = now + timedelta(days=days_ahead)
            else:
                return None
    else:
        explicit_year = year is not None
        year = year or now.year
        try:
            dt = datetime(year, month, day, tzinfo=LOCAL_TZ)
        except ValueError:
            return None
        if dt < now:
            skip_year_rollover = False
            if not explicit_year and now - dt <= RECENT_PAST_THRESHOLD:
                skip_year_rollover = True
            if not skip_year_rollover:
                try:
                    dt = datetime(year + 1, month, day, tzinfo=LOCAL_TZ)
                except ValueError:
                    return None

    tm = TIME_RE.search(text_low)
    if tm:
        hhmm = tm.group(0).replace(".", ":")
        hour, minute = map(int, hhmm.split(":"))
    else:
        tr = TIME_RANGE_RE.search(text_low)
        if tr:
            hour = int(tr.group(1))
            minute = int(tr.group(2) or 0)
        else:
            th = TIME_H_RE.search(text_low)
            if th:
                hour = int(th.group(1))
                minute = 0
            elif default_time:
                try:
                    hour, minute = map(int, default_time.split(":"))
                except Exception:
                    hour = minute = 0
            else:
                hour = minute = 0

    dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if dt < now:
        return None
    return int(dt.timestamp())


@dataclass
class EventDraft:
    title: str
    date: str | None = None
    time: str | None = None
    venue: str | None = None
    description: str | None = None
    festival: str | None = None
    location_address: str | None = None
    city: str | None = None
    ticket_price_min: int | None = None
    ticket_price_max: int | None = None
    event_type: str | None = None
    emoji: str | None = None
    end_date: str | None = None
    is_free: bool = False
    pushkin_card: bool = False
    links: List[str] | None = None
    source_text: str | None = None


@dataclass
class PersistResult:
    event_id: int
    telegraph_url: str
    ics_supabase_url: str
    ics_tg_url: str
    event_date: str
    event_end_date: str | None
    event_time: str
    event_type: str | None
    is_free: bool


async def build_event_payload_from_vk(
    text: str,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    operator_extra: str | None = None,
    festival_names: list[str] | None = None,
) -> EventDraft:
    """Return a normalised event draft extracted from a VK post.

    The function delegates parsing to the same LLM helper used by ``/add`` and
    forwarded posts.  When ``operator_extra`` is supplied it takes precedence
    over conflicting fragments of the original text.  ``source_name`` and
    ``location_hint`` are passed to the extractor for additional context and
    ``default_time`` is used when the post does not mention a time explicitly.
    The extractor is also instructed to apply ``default_time`` when no time is
    present in the post.

    The resulting :class:`EventDraft` contains normalised event attributes such
    as title, schedule, venue, ticket details and other metadata needed by the
    import pipeline.
    """
    from main import parse_event_via_4o

    llm_text = text
    if operator_extra:
        llm_text = f"{llm_text}\n{operator_extra}"
    if default_time:
        llm_text = f"{llm_text}\nЕсли время не указано, предположи начало в {default_time}."

    extra: dict[str, str] = {}
    if source_name:
        # ``parse_event_via_4o`` accepts ``channel_title`` for context
        extra["channel_title"] = source_name

    parsed = await parse_event_via_4o(
        llm_text, festival_names=festival_names, **extra
    )
    if not parsed:
        raise RuntimeError("LLM returned no event")
    data = parsed[0]

    combined_text = text or ""
    extra_clean = (operator_extra or "").strip()
    if extra_clean:
        trimmed = combined_text.rstrip()
        combined_text = f"{trimmed}\n\n{extra_clean}" if trimmed else extra_clean

    def clean_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return int(float(value))
            except ValueError:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def clean_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return str(value)

    def clean_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            val = value.strip().lower()
            if not val:
                return False
            if val in {"true", "1", "yes", "да", "y"}:
                return True
            if val in {"false", "0", "no", "нет", "n"}:
                return False
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return bool(value)

    ticket_price_min = clean_int(data.get("ticket_price_min"))
    ticket_price_max = clean_int(data.get("ticket_price_max"))
    links = [data["ticket_link"]] if data.get("ticket_link") else None
    return EventDraft(
        title=data.get("title", ""),
        date=data.get("date"),
        time=data.get("time") or default_time,
        venue=data.get("location_name"),
        description=data.get("short_description"),
        festival=clean_str(data.get("festival")),
        location_address=clean_str(data.get("location_address")),
        city=clean_str(data.get("city")),
        ticket_price_min=ticket_price_min,
        ticket_price_max=ticket_price_max,
        event_type=clean_str(data.get("event_type")),
        emoji=clean_str(data.get("emoji")),
        end_date=clean_str(data.get("end_date")),
        is_free=clean_bool(data.get("is_free")),
        pushkin_card=clean_bool(data.get("pushkin_card")),
        links=links,
        source_text=combined_text,
    )


async def persist_event_and_pages(
    draft: EventDraft, photos: list[str], db: Database, source_post_url: str | None = None
) -> PersistResult:
    """Store a drafted event and produce all public artefacts.

    The helper encapsulates the legacy import pipeline used by the bot.  It
    persists the event to the database, uploads images to Catbox and creates the
    Telegraph page, generates an ICS file and posts it to the asset channel.
    Links to these artefacts are returned in :class:`PersistResult`.
    """
    from datetime import datetime
    from models import Event
    import sys

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if main_mod is None:  # pragma: no cover - defensive
        raise RuntimeError("main module not found")
    upsert_event = main_mod.upsert_event
    schedule_event_update_tasks = main_mod.schedule_event_update_tasks

    event = Event(
        title=draft.title,
        description=(draft.description or ""),
        festival=(draft.festival or None),
        date=draft.date or datetime.utcnow().date().isoformat(),
        time=draft.time or "00:00",
        location_name=draft.venue or "",
        location_address=draft.location_address or None,
        city=draft.city or None,
        ticket_price_min=draft.ticket_price_min,
        ticket_price_max=draft.ticket_price_max,
        ticket_link=(draft.links[0] if draft.links else None),
        event_type=draft.event_type or None,
        emoji=draft.emoji or None,
        end_date=draft.end_date or None,
        is_free=bool(draft.is_free),
        pushkin_card=bool(draft.pushkin_card),
        source_text=draft.source_text or draft.title,
        photo_urls=photos,
        photo_count=len(photos),
        source_post_url=source_post_url,
    )

    async with db.get_session() as session:
        saved, _ = await upsert_event(session, event)
    async with db.get_session() as session:
        saved = await session.get(Event, saved.id)
    logging.info(
        "persist_event_and_pages: source_post_url=%s", saved.source_post_url
    )
    await schedule_event_update_tasks(db, saved)

    async with db.get_session() as session:
        saved = await session.get(Event, saved.id)

    return PersistResult(
        event_id=saved.id,
        telegraph_url=saved.telegraph_url or "",
        ics_supabase_url=saved.ics_url or "",
        ics_tg_url=saved.ics_post_url or "",
        event_date=saved.date,
        event_end_date=saved.end_date,
        event_time=saved.time,
        event_type=saved.event_type,
        is_free=bool(saved.is_free),
    )


async def process_event(
    text: str,
    photos: list[str] | None = None,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    operator_extra: str | None = None,
    db: Database,
) -> PersistResult:
    """Process VK post text into an event and track processing time."""
    start = time.perf_counter()
    from sqlalchemy import select
    from models import Festival

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival.name))
        festival_names = [row[0] for row in res_f.fetchall()]
    draft = await build_event_payload_from_vk(
        text,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        operator_extra=operator_extra,
        festival_names=festival_names,
    )
    result = await persist_event_and_pages(draft, photos or [], db)
    duration = time.perf_counter() - start
    global processing_time_seconds_total
    processing_time_seconds_total += duration
    try:
        import sys

        main_mod = sys.modules.get("main") or sys.modules.get("__main__")
        if main_mod is not None:
            main_mod.vk_import_duration_sum += duration
            main_mod.vk_import_duration_count += 1
            for bound in main_mod.vk_import_duration_buckets:
                if duration <= bound:
                    main_mod.vk_import_duration_buckets[bound] += 1
                    break
    except Exception:
        pass
    return result


async def crawl_once(db, *, broadcast: bool = False, bot: Any | None = None) -> dict[str, int]:
    """Crawl configured VK groups once and enqueue matching posts.

    The function scans groups listed in ``vk_source`` and uses cursors from
    ``vk_crawl_cursor`` to fetch only new posts. Posts containing event
    keywords and a date mention are inserted into ``vk_inbox`` with status
    ``pending``. Basic statistics are returned for reporting purposes.

    If ``broadcast`` is True and ``bot`` is supplied, a crawl summary is sent
    to the admin chat specified by ``ADMIN_CHAT_ID`` environment variable.
    """

    from main import vk_wall_since  # imported lazily to avoid circular import

    start = time.perf_counter()
    stats = {
        "groups_checked": 0,
        "posts_scanned": 0,
        "matches": 0,
        "duplicates": 0,
        "added": 0,
        "errors": 0,
        "inbox_total": 0,
        "queue": {},
    }

    async with db.raw_conn() as conn:
        cutoff = int(time.time()) + 2 * 3600
        await conn.execute(
            "UPDATE vk_inbox SET status='rejected' WHERE status IN ('pending','skipped') AND (event_ts_hint IS NULL OR event_ts_hint < ?)",
            (cutoff,),
        )
        cur = await conn.execute("SELECT group_id, default_time FROM vk_source")
        groups = [(row[0], row[1]) for row in await cur.fetchall()]
        await conn.commit()

    random.shuffle(groups)
    logging.info(
        "vk.crawl start groups=%d overlap=%s", len(groups), VK_CRAWL_OVERLAP_SEC
    )

    pages_per_group: list[int] = []

    now_ts = int(time.time())
    for gid, default_time in groups:
        stats["groups_checked"] += 1
        await asyncio.sleep(random.uniform(0.7, 1.2))  # safety pause
        try:
            async with db.raw_conn() as conn:
                cur = await conn.execute(
                    "SELECT last_seen_ts, last_post_id, updated_at FROM vk_crawl_cursor WHERE group_id=?",
                    (gid,),
                )
                row = await cur.fetchone()
            if row:
                last_seen_ts, last_post_id, updated_at = row
                updated_at_ts = int(
                    datetime.fromisoformat(updated_at).timestamp() if isinstance(updated_at, str) else updated_at
                )
            else:
                last_seen_ts = last_post_id = 0
                updated_at_ts = 0

            idle_h = (now_ts - updated_at_ts) / 3600 if updated_at_ts else None
            backfill = last_seen_ts == 0 or (
                idle_h is not None and idle_h >= VK_CRAWL_BACKFILL_AFTER_IDLE_H
            )

            posts: list[dict] = []
            pages_loaded = 0

            if backfill:
                horizon = now_ts - VK_CRAWL_BACKFILL_DAYS * 86400
                offset = 0
                while pages_loaded < VK_CRAWL_MAX_PAGES_BACKFILL:
                    page = await vk_wall_since(
                        gid, 0, count=VK_CRAWL_PAGE_SIZE_BACKFILL, offset=offset
                    )
                    pages_loaded += 1
                    posts.extend(p for p in page if p["date"] >= horizon)
                    if len(page) < VK_CRAWL_PAGE_SIZE_BACKFILL:
                        break
                    if page and min(p["date"] for p in page) < horizon:
                        break
                    offset += VK_CRAWL_PAGE_SIZE_BACKFILL
            else:
                since = max(0, last_seen_ts - VK_CRAWL_OVERLAP_SEC)
                offset = 0
                while True:
                    page = await vk_wall_since(
                        gid, since, count=VK_CRAWL_PAGE_SIZE, offset=offset
                    )
                    pages_loaded += 1
                    posts.extend(page)
                    if (
                        len(page) == VK_CRAWL_PAGE_SIZE
                        and page
                        and min(p["date"] for p in page) >= since
                        and pages_loaded < 1 + VK_CRAWL_MAX_PAGES_INC
                    ):
                        offset += VK_CRAWL_PAGE_SIZE
                        continue
                    break

            pages_per_group.append(pages_loaded)

            group_posts = 0
            group_matched = 0
            max_ts, max_pid = last_seen_ts, last_post_id

            for post in posts:
                ts = post["date"]
                pid = post["post_id"]
                if ts < last_seen_ts or (ts == last_seen_ts and pid <= last_post_id):
                    continue
                stats["posts_scanned"] += 1
                group_posts += 1
                kw_ok, kws = match_keywords(post["text"])
                has_date = detect_date(post["text"])
                event_ts_hint = extract_event_ts_hint(post["text"], default_time)
                if kw_ok and has_date:
                    if event_ts_hint is None or event_ts_hint < int(time.time()) + 2 * 3600:
                        continue
                    stats["matches"] += 1
                    try:
                        async with db.raw_conn() as conn:
                            cur = await conn.execute(
                                """
                                INSERT OR IGNORE INTO vk_inbox(
                                    group_id, post_id, date, text, matched_kw, has_date, event_ts_hint, status
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                                """,
                                (
                                    gid,
                                    pid,
                                    ts,
                                    post["text"],
                                    ",".join(kws),
                                    int(has_date),
                                    event_ts_hint,
                                ),
                            )
                            await conn.commit()
                        if cur.rowcount == 0:
                            stats["duplicates"] += 1
                        else:
                            stats["added"] += 1
                            group_matched += 1
                    except Exception:
                        stats["errors"] += 1

                if ts > max_ts or (ts == max_ts and pid > max_pid):
                    max_ts, max_pid = ts, pid

            async with db.raw_conn() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO vk_crawl_cursor(group_id, last_seen_ts, last_post_id) VALUES(?,?,?)",
                    (gid, max_ts, max_pid),
                )
                await conn.commit()

            mode = "backfill" if backfill else "inc"
            logging.info(
                "vk.crawl group=%s posts=%s matched=%s pages=%s mode=%s",
                gid,
                group_posts,
                group_matched,
                pages_loaded,
                mode,
            )
        except Exception:
            stats["errors"] += 1

    async with db.raw_conn() as conn:
        cur = await conn.execute(
            "SELECT status, COUNT(*) FROM vk_inbox GROUP BY status"
        )
        rows = await cur.fetchall()
    for st, cnt in rows:
        stats["queue"][st] = cnt
    stats["inbox_total"] = sum(stats["queue"].values())
    stats["pages_per_group"] = pages_per_group
    stats["overlap_sec"] = VK_CRAWL_OVERLAP_SEC
    try:
        import main
        main.vk_crawl_groups_total += stats["groups_checked"]
        main.vk_crawl_posts_scanned_total += stats["posts_scanned"]
        main.vk_crawl_matched_total += stats["matches"]
        main.vk_crawl_duplicates_total += stats["duplicates"]
        main.vk_inbox_inserted_total += stats["added"]
    except Exception:
        pass

    took_ms = int((time.perf_counter() - start) * 1000)
    logging.info(
        "vk.crawl.finish groups=%s posts_scanned=%s matches=%s dups=%s added=%s inbox_total=%s pages=%s overlap=%s took_ms=%s",
        stats["groups_checked"],
        stats["posts_scanned"],
        stats["matches"],
        stats["duplicates"],
        stats["added"],
        stats["inbox_total"],
        "/".join(str(p) for p in pages_per_group),
        VK_CRAWL_OVERLAP_SEC,
        took_ms,
    )
    if broadcast and bot:
        admin_chat = os.getenv("ADMIN_CHAT_ID")
        if admin_chat:
            q = stats.get("queue", {})
            msg = (
                f"Проверено {stats['groups_checked']} сообществ, "
                f"просмотрено {stats['posts_scanned']} постов, "
                f"совпало {stats['matches']}, "
                f"дубликатов {stats['duplicates']}, "
                f"добавлено {stats['added']}, "
                f"теперь в очереди {stats['inbox_total']} "
                f"(pending: {q.get('pending',0)}, locked: {q.get('locked',0)}, "
                f"skipped: {q.get('skipped',0)}, imported: {q.get('imported',0)}, "
                f"rejected: {q.get('rejected',0)}), "
                f"страниц на группу: {'/'.join(str(p) for p in stats['pages_per_group'])}, "
                f"перекрытие: {stats['overlap_sec']} сек"
            )
            try:
                await bot.send_message(int(admin_chat), msg)
            except Exception:
                logging.exception("vk.crawl.broadcast.error")
    return stats
