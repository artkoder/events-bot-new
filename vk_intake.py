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

from sections import MONTHS_RU

# Keywords used to detect potential event posts
KEYWORDS: list[str] = [
    "показ",
    "кинопоказ",
    "премьера",
    "мюзикл",
    "спектакль",
    "лекция",
    "лектор",
    "концерт",
    "фестиваль",
    "мастер-класс",
    "воркшоп",
    "встреча",
    "экскурсия",
    "читка",
    "выставка",
    "перформанс",
    "кинолекторий",
    "бронирование",
    "билеты",
    "регистрация",
    "афиша",
]

# Date patterns roughly covering common forms seen in Russian posts
DATE_PATTERNS: list[str] = [
    r"\b\d{1,2}[./-]\d{1,2}\b",
    r"\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    r"\bсегодня\b",
    r"\bзавтра\b",
    r"\bпослезавтра\b",
    r"\b(понедельник|вторник|среда|четверг|пятница|суббота|воскресенье|пн|вт|ср|чт|пт|сб|вс)\b",
    r"\b([01]?\d|2[0-3]):[0-5]\d\b",
]

COMPILED_DATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in DATE_PATTERNS]

NUM_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})\b")
MONTH_NAME_RE = re.compile(
    r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")

# cumulative processing time for VK event intake (seconds)
processing_time_seconds_total: float = 0.0


def match_keywords(text: str) -> tuple[bool, list[str]]:
    """Return True and list of matched keywords if any are found."""
    matched: list[str] = []
    for kw in KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            matched.append(kw)
    return bool(matched), matched


def detect_date(text: str) -> bool:
    """Heuristically detect a date or time mention in the text."""
    return any(p.search(text) for p in COMPILED_DATE_PATTERNS)


def _extract_event_ts_hint(text: str, default_time: str | None = None) -> int | None:
    """Return Unix timestamp for the nearest future datetime mentioned in text."""
    from main import LOCAL_TZ

    now = datetime.now(LOCAL_TZ)
    text_low = text.lower()

    day = month = None
    m = NUM_DATE_RE.search(text_low)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
    else:
        m = MONTH_NAME_RE.search(text_low)
        if m:
            day = int(m.group(1))
            month = MONTHS_RU.get(m.group(2).lower())

    if day is None or month is None:
        if "сегодня" in text_low:
            dt = now
        elif "завтра" in text_low:
            dt = now + timedelta(days=1)
        elif "послезавтра" in text_low:
            dt = now + timedelta(days=2)
        else:
            return None
    else:
        year = now.year
        try:
            dt = datetime(year, month, day, tzinfo=LOCAL_TZ)
        except ValueError:
            return None
        if dt < now:
            try:
                dt = datetime(year + 1, month, day, tzinfo=LOCAL_TZ)
            except ValueError:
                return None

    tm = TIME_RE.search(text_low)
    if tm:
        hour, minute = map(int, tm.group(0).split(":"))
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
    price: str | None = None
    links: List[str] | None = None
    source_text: str | None = None


@dataclass
class PersistResult:
    event_id: int
    telegraph_url: str
    ics_supabase_url: str
    ics_tg_url: str
    event_date: str


async def build_event_payload_from_vk(
    text: str,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    operator_extra: str | None = None,
) -> EventDraft:
    """Return a normalised event draft extracted from a VK post.

    The function delegates parsing to the same LLM helper used by ``/add`` and
    forwarded posts.  When ``operator_extra`` is supplied it takes precedence
    over conflicting fragments of the original text.  ``source_name`` and
    ``location_hint`` are passed to the extractor for additional context and
    ``default_time`` is used when the post does not mention a time explicitly.

    The resulting :class:`EventDraft` contains basic event attributes such as
    title, date, time, venue, price and relevant links.
    """
    from main import parse_event_via_4o

    llm_text = text
    if operator_extra:
        llm_text = f"{llm_text}\n{operator_extra}"

    extra: dict[str, str] = {}
    if source_name:
        # ``parse_event_via_4o`` accepts ``channel_title`` for context
        extra["channel_title"] = source_name

    parsed = await parse_event_via_4o(llm_text, **extra)
    if not parsed:
        raise RuntimeError("LLM returned no event")
    data = parsed[0]

    price: str | None = None
    if data.get("ticket_price_min") or data.get("ticket_price_max"):
        lo = data.get("ticket_price_min")
        hi = data.get("ticket_price_max")
        if lo and hi and lo != hi:
            price = f"{lo}-{hi}"
        else:
            price = str(lo or hi)

    links = [data["ticket_link"]] if data.get("ticket_link") else None
    return EventDraft(
        title=data.get("title", ""),
        date=data.get("date"),
        time=data.get("time") or default_time,
        venue=data.get("location_name"),
        price=price,
        links=links,
        source_text=text,
    )


async def persist_event_and_pages(
    draft: EventDraft, photos: list[str]
) -> PersistResult:
    """Store a drafted event and produce all public artefacts.

    The helper encapsulates the legacy import pipeline used by the bot.  It
    persists the event to the database, uploads images to Catbox and creates the
    Telegraph page, generates an ICS file and posts it to the asset channel.
    Links to these artefacts are returned in :class:`PersistResult`.
    """
    from datetime import datetime
    from main import (
        db,
        Event,
        upsert_event,
        schedule_event_update_tasks,
    )

    event = Event(
        title=draft.title,
        description="",
        festival=None,
        date=draft.date or datetime.utcnow().date().isoformat(),
        time=draft.time or "00:00",
        location_name=draft.venue or "",
        source_text=draft.source_text or draft.title,
        ticket_link=(draft.links[0] if draft.links else None),
        photo_urls=photos,
        photo_count=len(photos),
    )

    async with db.get_session() as session:
        saved, _ = await upsert_event(session, event)

    await schedule_event_update_tasks(db, saved)

    async with db.get_session() as session:
        saved = await session.get(Event, saved.id)

    return PersistResult(
        event_id=saved.id,
        telegraph_url=saved.telegraph_url or "",
        ics_supabase_url=saved.ics_url or "",
        ics_tg_url=saved.ics_post_url or "",
        event_date=saved.date,
    )


async def process_event(
    text: str,
    photos: list[str] | None = None,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    operator_extra: str | None = None,
) -> PersistResult:
    """Process VK post text into an event and track processing time."""
    start = time.perf_counter()
    draft = await build_event_payload_from_vk(
        text,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        operator_extra=operator_extra,
    )
    result = await persist_event_and_pages(draft, photos or [])
    duration = time.perf_counter() - start
    global processing_time_seconds_total
    processing_time_seconds_total += duration
    try:
        import main

        main.vk_import_duration_sum += duration
        main.vk_import_duration_count += 1
        for bound in main.vk_import_duration_buckets:
            if duration <= bound:
                main.vk_import_duration_buckets[bound] += 1
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
            "UPDATE vk_inbox SET status='rejected' WHERE status IN ('pending','skipped') AND event_ts_hint IS NOT NULL AND event_ts_hint < ?",
            (cutoff,),
        )
        cur = await conn.execute("SELECT group_id, default_time FROM vk_source")
        groups = [(row[0], row[1]) for row in await cur.fetchall()]
        await conn.commit()

    logging.info("vk.crawl start groups=%d", len(groups))

    for gid, default_time in groups:
        stats["groups_checked"] += 1
        # pause between groups (safety: 0.7–1.2s)
        await asyncio.sleep(random.uniform(0.7, 1.2))
        try:
            async with db.raw_conn() as conn:
                cur = await conn.execute(
                    "SELECT last_seen_ts, last_post_id FROM vk_crawl_cursor WHERE group_id=?",
                    (gid,),
                )
                row = await cur.fetchone()
                last_seen_ts, last_post_id = row if row else (0, 0)

            posts = await vk_wall_since(gid, last_seen_ts)
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
                event_ts_hint = _extract_event_ts_hint(post["text"], default_time)
                if kw_ok and has_date:
                    if event_ts_hint is not None and event_ts_hint < int(time.time()) + 2 * 3600:
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

            logging.info(
                "vk.crawl group=%s posts=%s matched=%s", gid, group_posts, group_matched
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
        "vk.crawl.finish groups=%s posts_scanned=%s matches=%s dups=%s added=%s inbox_total=%s took_ms=%s",
        stats["groups_checked"],
        stats["posts_scanned"],
        stats["matches"],
        stats["duplicates"],
        stats["added"],
        stats["inbox_total"],
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
                f"rejected: {q.get('rejected',0)})"
            )
            try:
                await bot.send_message(int(admin_chat), msg)
            except Exception:
                logging.exception("vk.crawl.broadcast.error")
    return stats
