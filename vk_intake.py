from __future__ import annotations

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
    """Placeholder for LLM-based event extraction from VK posts."""
    raise NotImplementedError


async def persist_event_and_pages(
    draft: EventDraft, photos: list[str]
) -> PersistResult:
    """Placeholder for persistence and page generation pipeline."""
    raise NotImplementedError


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
