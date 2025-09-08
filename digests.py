from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
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


def _event_start_datetime(event: Event) -> datetime:
    """Combine ``date`` and ``time`` fields of an event into a datetime."""
    return datetime.strptime(f"{event.date} {event.time}", "%Y-%m-%d %H:%M")


async def build_lectures_digest_candidates(
    db: Database, now: datetime
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
    events = [e for e in events if _event_start_datetime(e) >= cutoff]
    events.sort(key=_event_start_datetime)

    horizon = 7
    end_7 = now + timedelta(days=7)
    selected = [e for e in events if _event_start_datetime(e) <= end_7]

    if len(selected) < 6:
        horizon = 14
        end_14 = now + timedelta(days=14)
        selected = [e for e in events if _event_start_datetime(e) <= end_14]

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
