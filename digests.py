from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import logging
import re
import time
from typing import Iterable, List, Tuple

from sqlalchemy import select

from db import Database
from models import Event

# Mapping of canonical topic -> set of synonyms (in lowercase)
TOPIC_SYNONYMS: dict[str, set[str]] = {
    "искусство": {"искусство", "культура"},
    "история россии": {"история россии", "российская история"},
    "технологии": {"технологии", "ит", "тех"},
    "психология": {"психология", "mental health"},
    "урбанистика": {"урбанистика", "город"},
    "литература": {"литература", "книги"},
    "кино": {"кино", "фильм", "кинематограф"},
    "музыка": {"музыка", "саунд", "sound"},
    "бизнес": {"бизнес", "предпринимательство"},
}

# Reverse lookup for quick normalization
_REVERSE_SYNONYMS = {
    syn: canon
    for canon, syns in TOPIC_SYNONYMS.items()
    for syn in syns
}


def parse_start_time(raw: str) -> tuple[int, int] | None:
    """Вернёт (hh, mm) из строки времени.

    Берём ПЕРВОЕ валидное вхождение. Поддерживаем разделители ``:`` и ``.``
    и сложные строки вроде ``18:30–20:00`` или ``18:30.15:30``. Часы и
    минуты нормализуются в диапазоны 0–23 и 0–59 соответственно. Если время
    не распознано, возвращается ``None``.
    """

    match = re.search(r"\b(\d{1,2})[:\.](\d{2})\b", raw)
    if not match:
        return None

    hh = max(0, min(23, int(match.group(1))))
    mm = max(0, min(59, int(match.group(2))))
    return hh, mm


def _event_start_datetime(event: Event, digest_id: str | None = None) -> datetime:
    """Combine ``date`` and ``time`` fields of an event into a datetime."""

    cached = getattr(event, "_start_dt", None)
    if cached is not None:
        return cached

    day = datetime.strptime(event.date, "%Y-%m-%d")
    raw = event.time or ""
    parsed = parse_start_time(raw)

    if parsed is None:
        logging.warning(
            "digest.time.parse digest_id=%s event_id=%s title=%r time_raw=%r parsed=null reason_if_null=no_match",
            digest_id,
            event.id,
            event.title,
            raw,
        )
        dt = day
    else:
        hh, mm = parsed
        logging.info(
            "digest.time.parse digest_id=%s event_id=%s title=%r time_raw=%r parsed=%02d:%02d",
            digest_id,
            event.id,
            event.title,
            raw,
            hh,
            mm,
        )
        dt = day.replace(hour=hh, minute=mm)

    setattr(event, "_start_dt", dt)
    return dt


async def build_lectures_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select lecture events for the digest.

    Parameters
    ----------
    db:
        Database instance.
    now:
        Current moment in local timezone.

    Returns
    -------
    tuple[list[Event], int]
        A tuple with selected events ordered by start datetime and the
        horizon in days (7 or 14) that was used.
    """

    start_date = now.date().isoformat()
    end_date = (now + timedelta(days=14)).date().isoformat()

    async with db.get_session() as session:
        res = await session.execute(
            select(Event)
            .where(
                Event.event_type == "лекция",
                Event.date >= start_date,
                Event.date <= end_date,
            )
            .order_by(Event.date, Event.time)
        )
        events = list(res.scalars().all())

    cutoff = now + timedelta(hours=2)
    events = [e for e in events if _event_start_datetime(e, digest_id) >= cutoff]
    events.sort(key=lambda e: _event_start_datetime(e, digest_id))

    horizon = 7
    end_7 = now + timedelta(days=7)
    selected = [e for e in events if _event_start_datetime(e, digest_id) <= end_7]

    if len(selected) < 6:
        horizon = 14
        end_14 = now + timedelta(days=14)
        selected = [e for e in events if _event_start_datetime(e, digest_id) <= end_14]

    return selected[:9], horizon


def normalize_topics(topics: Iterable[str]) -> List[str]:
    """Normalize topics using the synonym map.

    The function lowercases input topics, maps known synonyms to their
    canonical form, removes duplicates and returns a sorted list.
    """

    normalized = [_REVERSE_SYNONYMS.get(t.strip().lower(), t.strip().lower()) for t in topics]
    return sorted(set(normalized))


def aggregate_digest_topics(events: Iterable[Event]) -> List[str]:
    """Aggregate topics from events and return top-3 by frequency.

    Events are expected to have ``topics`` attribute containing a list of
    strings. Unknown topics are included as-is after normalization.
    """

    counter: Counter[str] = Counter()
    for ev in events:
        for topic in normalize_topics(getattr(ev, "topics", [])):
            counter[topic] += 1

    if not counter:
        return []

    ranked = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:3]]


def make_intro(n: int, horizon_days: int, topics3: List[str]) -> str:
    """Construct intro phrase for the digest.

    Parameters
    ----------
    n:
        Number of lectures.
    horizon_days:
        Either 7 or 14 depending on the search horizon.
    topics3:
        List of up to three topics.
    """

    if n == 1:
        lectures_part = "интересную лекцию"
    elif 2 <= n <= 4:
        lectures_part = "интересные лекции"
    else:
        lectures_part = "интересных лекций"

    horizon_part = "ближайшей недели" if horizon_days == 7 else "ближайших двух недель"

    if not topics3:
        topics_part = "на разные темы"
    elif len(topics3) == 1:
        topics_part = f"на {topics3[0]}"
    elif len(topics3) == 2:
        topics_part = f"на {topics3[0]} и {topics3[1]}"
    else:
        topics_part = f"на {topics3[0]}, {topics3[1]} и {topics3[2]}"

    return f"Подобрали для вас {n} {lectures_part} {horizon_part} {topics_part}."


async def build_lectures_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, int, List[Event]]:
    """Build digest preview text for lectures."""

    start = time.monotonic()
    logging.info(
        "digest.collect.start digest_id=%s window_days=14 now=%s limit=9",
        digest_id,
        now.isoformat(),
    )
    events, horizon = await build_lectures_digest_candidates(db, now, digest_id)
    duration_ms = int((time.monotonic() - start) * 1000)
    cutoff_plus_2h = now + timedelta(hours=2)
    logging.info(
        "digest.collect.end digest_id=%s window_days=%s now=%s cutoff_plus_2h=%s count_found=%s count_after_filters=%s limit=9 duration_ms=%s",
        digest_id,
        horizon,
        now.isoformat(),
        cutoff_plus_2h.isoformat(),
        len(events),
        len(events),
        duration_ms,
    )

    if not events:
        return "", horizon, []

    topics = aggregate_digest_topics(events)
    intro = make_intro(len(events), horizon, topics)
    lines = [intro, ""]
    for ev in events:
        lines.append(f"{ev.date} | {ev.title}")
    text = "\n".join(lines)
    return text, horizon, events
