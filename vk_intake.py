from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import List

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
    """Parse VK post text into an :class:`EventDraft` using the existing LLM."""
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
    """Persist a drafted event and schedule page generation tasks."""
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
    return result


async def crawl_once(db, *, broadcast: bool = False) -> dict[str, int]:
    """Crawl configured VK groups once and enqueue matching posts.

    The function scans groups listed in ``vk_source`` and uses cursors from
    ``vk_crawl_cursor`` to fetch only new posts. Posts containing event
    keywords and a date mention are inserted into ``vk_inbox`` with status
    ``pending``. Basic statistics are returned for reporting purposes.
    ``broadcast`` is accepted for API parity but is not used directly here.
    """

    from main import vk_wall_since  # imported lazily to avoid circular import

    start = time.perf_counter()
    stats = {
        "groups_checked": 0,
        "posts_scanned": 0,
        "posts_matched": 0,
        "duplicates": 0,
        "errors": 0,
        "inbox_total": 0,
    }

    async with db.raw_conn() as conn:
        cur = await conn.execute("SELECT group_id FROM vk_source")
        groups = [row[0] for row in await cur.fetchall()]

    logging.info("vk.crawl start groups=%d", len(groups))

    for gid in groups:
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
                if kw_ok and has_date:
                    try:
                        async with db.raw_conn() as conn:
                            cur = await conn.execute(
                                """
                                INSERT OR IGNORE INTO vk_inbox(
                                    group_id, post_id, date, text, matched_kw, has_date, status
                                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
                                """,
                                (
                                    gid,
                                    pid,
                                    ts,
                                    post["text"],
                                    ",".join(kws),
                                    int(has_date),
                                ),
                            )
                            await conn.commit()
                        if cur.rowcount == 0:
                            stats["duplicates"] += 1
                        else:
                            stats["posts_matched"] += 1
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
            "SELECT COUNT(*) FROM vk_inbox WHERE status='pending'"
        )
        stats["inbox_total"] = (await cur.fetchone())[0]

    took_ms = int((time.perf_counter() - start) * 1000)
    logging.info(
        "vk.crawl.finish groups=%s posts_scanned=%s matched=%s dups=%s inbox_total=%s took_ms=%s",
        stats["groups_checked"],
        stats["posts_scanned"],
        stats["posts_matched"],
        stats["duplicates"],
        stats["inbox_total"],
        took_ms,
    )
    return stats
