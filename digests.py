from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import html
import logging
import re
import time
from typing import Iterable, List, Tuple, Callable, Awaitable
import httpx

from sqlalchemy import select

from db import Database
from models import Event, normalize_topic_identifier

# Mapping of canonical topic -> set of synonyms (in lowercase)
TOPIC_SYNONYMS: dict[str, set[str]] = {
    "EXHIBITIONS": {"art", "искусство", "выставка", "выставки", "галерея"},
    "THEATRE": {"theatre", "театр", "спектакль", "спектакли", "performance"},
    "THEATRE_CLASSIC": {
        "classic theatre",
        "классический театр",
        "классический спектакль",
        "драма",
        "драматический театр",
    },
    "THEATRE_MODERN": {
        "modern theatre",
        "современный театр",
        "современные спектакли",
        "модерн",
        "experimental theatre",
        "экспериментальный театр",
    },
    "CONCERTS": {"concert", "music", "музыка", "концерт", "sound"},
    "MOVIES": {"cinema", "movie", "film", "кино", "фильм"},
    "LECTURES": {
        "lecture",
        "lectures",
        "лекция",
        "лекции",
        "история",
        "история россии",
        "книги",
        "business",
        "встреча",
    },
    "MASTERCLASS": {"masterclass", "мастер-класс", "воркшоп"},
    "PARTIES": {"party", "вечеринка", "вечеринки"},
    "STANDUP": {"standup", "стендап", "стендапы", "комедия"},
    "QUIZ_GAMES": {"quiz", "квиз", "квизы", "игра", "настолки"},
    "OPEN_AIR": {"open-air", "open air", "фестиваль", "фестивали", "openair"},
    "SCIENCE_POP": {"science", "science_pop", "научпоп", "технологии"},
    "PSYCHOLOGY": {"психология", "psychology", "mental health"},
    "HANDMADE": {
        "handmade",
        "hand-made",
        "ярмарка",
        "ярмарки",
        "маркет",
        "маркеты",
        "маркетплейс",
        "маркетплейсы",
        "хендмейд",
    },
    "NETWORKING": {
        "networking",
        "network",
        "нетворкинг",
        "нетворк",
        "знакомства",
        "карьера",
        "деловые встречи",
        "бизнес-завтрак",
        "business breakfast",
        "карьерный вечер",
    },
    "ACTIVE": {
        "active",
        "sport",
        "sports",
        "спорт",
        "спортивные",
        "активности",
        "активный отдых",
        "фитнес",
        "йога",
        "yoga",
    },
    "PERSONALITIES": {
        "personalities",
        "personality",
        "персоны",
        "личности",
        "встреча с автором",
        "встреча с героем",
        "встреча с артистом",
        "книжный клуб",
        "книжные клубы",
        "book club",
    },
    "KIDS_SCHOOL": {
        "kids",
        "kids_school",
        "дети",
        "детям",
        "детские",
        "школа",
        "школьники",
        "образование",
    },
    "FAMILY": {
        "family",
        "семья",
        "семейные",
        "семейный",
        "для всей семьи",
    },
    "URBANISM": {
        "urbanism",
        "урбанистика",
        "урбанистический",
        "урбанистике",
    },
    "KRAEVEDENIE_KALININGRAD_OBLAST": {
        "краеведение",
        "краевед",
        "краеведческий",
        "краеведческие",
        "калининград",
        "kaliningrad",
        "калининградская область",
        "калининградской области",
        "кёнигсберг",
        "кенигсберг",
        "königsberg",
        "konigsberg",
        "koenigsberg",
        "kenigsberg",
        "kenig",
        "янтарный край",
        "янтарного края",
        "39 регион",
        "39-й регион",
        "39й регион",
        "39йрегион",
        "#калининград",
    },
}

# Reverse lookup for quick normalization
_REVERSE_SYNONYMS = {
    syn.casefold(): canon
    for canon, syns in TOPIC_SYNONYMS.items()
    for syn in syns
}


MEETUPS_INTRO_FORBIDDEN_WORDINGS: tuple[str, ...] = (
    "«Погрузитесь»",
    "«не упустите шанс»",
    "конструкции с «мир …»",
    "«Откройте для себя»",
    "любые упоминания «горизонтов»",
)


def _format_forbidden_wordings(wordings: Iterable[str]) -> str:
    items = list(wordings)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} и {items[1]}"
    return ", ".join(items[:-1]) + f" и {items[-1]}"


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


async def _build_digest_candidates(
    event_type: str | None,
    db: Database,
    now: datetime,
    digest_id: str | None = None,
    *,
    topic_identifier: str | None = None,
    topic_identifiers: Iterable[str] | None = None,
) -> Tuple[List[Event], int]:
    """Select events within the digest window with optional filters.

    Parameters
    ----------
    event_type:
        ``Event.event_type`` value to filter by. When ``None`` the filter is
        omitted which allows combining with topic-based selection.
    topic_identifier:
        Canonical topic identifier (e.g. ``"PSYCHOLOGY"``). When provided the
        resulting events must contain the topic after normalization.
    """

    start_date = now.date().isoformat()
    end_date = (now + timedelta(days=14)).date().isoformat()

    async with db.get_session() as session:
        query = (
            select(Event)
            .where(
                Event.date >= start_date,
                Event.date <= end_date,
            )
            .order_by(Event.date, Event.time)
        )
        if event_type is not None:
            query = query.where(Event.event_type == event_type)
        res = await session.execute(query)
        events = list(res.scalars().all())

    topic_filters: set[str] = set()
    if topic_identifier:
        normalized_topic = normalize_topic_identifier(topic_identifier) or topic_identifier
        topic_filters.add(normalized_topic)
    if topic_identifiers:
        for raw in topic_identifiers:
            if not raw:
                continue
            canonical = normalize_topic_identifier(raw) or raw
            topic_filters.add(canonical)

    if topic_filters:
        filtered: List[Event] = []
        for event in events:
            topics = set(normalize_topics(getattr(event, "topics", [])))
            if topics.intersection(topic_filters):
                filtered.append(event)
        events = filtered

    cutoff = now + timedelta(hours=2)
    events = [e for e in events if _event_start_datetime(e, digest_id) >= cutoff]
    events.sort(key=lambda e: _event_start_datetime(e, digest_id))

    horizon = 7
    end_7 = now + timedelta(days=7)
    result: List[Event] = []
    for ev in events:
        if _event_start_datetime(ev, digest_id) > end_7:
            continue
        if pick_display_link(ev) is None:
            logging.info(
                "digest.skip.no_link event_id=%s title=%r event_type=%s",
                getattr(ev, "id", None),
                ev.title,
                event_type,
            )
            continue
        result.append(ev)
        if len(result) == 9:
            break

    if len(result) < 9:
        horizon = 14
        end_14 = now + timedelta(days=14)
        result = []
        for ev in events:
            if _event_start_datetime(ev, digest_id) > end_14:
                continue
            if pick_display_link(ev) is None:
                logging.info(
                    "digest.skip.no_link event_id=%s title=%r event_type=%s",
                    getattr(ev, "id", None),
                    ev.title,
                    event_type,
                )
                continue
            result.append(ev)
            if len(result) == 9:
                break

    return result, horizon


async def build_lectures_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select lecture events for the digest."""

    return await _build_digest_candidates("лекция", db, now, digest_id)


async def build_masterclasses_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select master-class events for the digest."""

    return await _build_digest_candidates("мастер-класс", db, now, digest_id)


async def build_psychology_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select psychology-tagged events for the digest."""

    return await _build_digest_candidates(
        None, db, now, digest_id, topic_identifier="PSYCHOLOGY"
    )


async def build_science_pop_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select science-pop events for the digest."""

    return await _build_digest_candidates(
        None, db, now, digest_id, topic_identifier="SCIENCE_POP"
    )


async def build_kraevedenie_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select Kaliningrad regional heritage events for the digest."""

    return await _build_digest_candidates(
        None,
        db,
        now,
        digest_id,
        topic_identifier="KRAEVEDENIE_KALININGRAD_OBLAST",
    )


async def build_networking_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select networking-focused events for the digest."""

    return await _build_digest_candidates(
        None, db, now, digest_id, topic_identifier="NETWORKING"
    )


async def build_entertainment_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select stand-up, quiz, party and open-air events for the digest."""

    return await _build_digest_candidates(
        None,
        db,
        now,
        digest_id,
        topic_identifiers=("STANDUP", "QUIZ_GAMES", "PARTIES", "OPEN_AIR"),
    )


async def build_markets_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select handmade market events for the digest."""

    return await _build_digest_candidates(
        None, db, now, digest_id, topic_identifier="HANDMADE"
    )


async def build_theatre_classic_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select classic theatre performances for the digest."""

    return await _build_digest_candidates(
        "спектакль",
        db,
        now,
        digest_id,
        topic_identifier="THEATRE_CLASSIC",
    )


async def build_theatre_modern_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select modern theatre performances for the digest."""

    return await _build_digest_candidates(
        "спектакль",
        db,
        now,
        digest_id,
        topic_identifier="THEATRE_MODERN",
    )


async def build_meetups_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select meetups and club events for the digest."""

    return await _build_digest_candidates(
        None, db, now, digest_id, topic_identifier="PERSONALITIES"
    )


async def build_movies_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select movie screening events for the digest."""

    return await _build_digest_candidates("кинопоказ", db, now, digest_id)


def _event_end_date(event: Event) -> datetime | None:
    """Return ``datetime`` for ``event.end_date`` if possible."""

    raw = getattr(event, "end_date", None)
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        logging.warning(
            "digest.end_date.parse event_id=%s title=%r end_date_raw=%r parsed=null",  # noqa: G003
            getattr(event, "id", None),
            event.title,
            raw,
        )
        return None


async def build_exhibitions_digest_candidates(
    db: Database, now: datetime, digest_id: str | None = None
) -> Tuple[List[Event], int]:
    """Select exhibition events for the digest."""

    today_iso = now.date().isoformat()

    async with db.get_session() as session:
        res = await session.execute(
            select(Event)
            .where(
                Event.event_type == "выставка",
                Event.end_date.is_not(None),
                Event.end_date >= today_iso,
            )
            .order_by(Event.date, Event.time)
        )
        events = list(res.scalars().all())

    cutoff = now + timedelta(hours=2)
    filtered: List[Event] = []
    for ev in events:
        start_dt = _event_start_datetime(ev, digest_id)
        if start_dt >= now and start_dt < cutoff:
            continue
        filtered.append(ev)

    events = filtered
    events.sort(
        key=lambda e: (
            _event_end_date(e) or datetime.max,
            _event_start_datetime(e, digest_id),
        )
    )

    def _within_horizon(ev: Event, days: int) -> bool:
        end_dt = _event_end_date(ev)
        if end_dt is None:
            return False
        return end_dt.date() <= (now + timedelta(days=days)).date()

    horizon = 7
    result: List[Event] = []
    for ev in events:
        if not _within_horizon(ev, horizon):
            continue
        if pick_display_link(ev) is None:
            logging.info(
                "digest.skip.no_link event_id=%s title=%r event_type=%s",
                getattr(ev, "id", None),
                ev.title,
                "выставка",
            )
            continue
        result.append(ev)
        if len(result) == 9:
            break

    if len(result) < 9:
        horizon = 14
        result = []
        for ev in events:
            if not _within_horizon(ev, horizon):
                continue
            if pick_display_link(ev) is None:
                logging.info(
                    "digest.skip.no_link event_id=%s title=%r event_type=%s",
                    getattr(ev, "id", None),
                    ev.title,
                    "выставка",
                )
                continue
            result.append(ev)
            if len(result) == 9:
                break

    return result, horizon


def normalize_topics(topics: Iterable[str]) -> List[str]:
    """Normalize topics using the synonym map.

    The function lowercases input topics, maps known synonyms to their
    canonical form, removes duplicates and returns a sorted list.
    """

    normalized: list[str] = []
    for topic in topics:
        canonical = normalize_topic_identifier(topic)
        if canonical:
            normalized.append(canonical)
            continue
        if not isinstance(topic, str):
            continue
        cleaned = topic.strip()
        if not cleaned:
            continue
        mapped = _REVERSE_SYNONYMS.get(cleaned.casefold())
        if mapped:
            normalized.append(mapped)
        else:
            normalized.append(cleaned)
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


async def extract_catbox_covers_from_telegraph(
    page_url: str, *, event_id: str | int | None = None
) -> List[str]:
    """Extract ``files.catbox.moe`` image URLs from a Telegraph page.

    Parameters
    ----------
    page_url:
        URL of the Telegraph page.

    Returns
    -------
    list[str]
        List of URLs in order of appearance. Only ``files.catbox.moe`` links
        with typical image extensions are returned. No HEAD requests are
        performed – validation is based solely on the file extension.
    """

    import uuid

    run_id = uuid.uuid4().hex
    start = time.monotonic()
    logging.info(
        "digest.cover.fetch.start event_id=%s run_id=%s telegraph_url=%s",
        event_id,
        run_id,
        page_url,
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(page_url)
    except Exception:
        logging.info(
            "digest.cover.missing event_id=%s run_id=%s reason=network_error",
            event_id,
            run_id,
        )
        return []
    took_ms = int((time.monotonic() - start) * 1000)
    logging.info(
        "digest.cover.fetch.html.ok event_id=%s run_id=%s bytes=%s took_ms=%s",
        event_id,
        run_id,
        len(resp.content or b""),
        took_ms,
    )
    html = resp.text
    pattern = re.compile(
        r'(?:src|href)="(https://files\.catbox\.moe/[^"]+\.(?:jpg|jpeg|png|webp|gif))"',
        re.IGNORECASE,
    )
    matches = pattern.findall(html)
    results: List[str] = []
    for url in matches:
        ext = url.rsplit(".", 1)[-1].lower()
        logging.info(
            "digest.cover.candidate event_id=%s url=%s ext=%s",
            event_id,
            url,
            ext,
        )
        results.append(url)

    if results:
        logging.info(
            "digest.cover.accepted event_id=%s url=%s",
            event_id,
            results[0],
        )
    else:
        logging.info(
            "digest.cover.missing event_id=%s reason=no_catbox_in_html",
            event_id,
        )

    return results
 

async def compose_digest_intro_via_4o(
    n: int, horizon_days: int, titles: List[str], *, event_noun: str = "лекций"
) -> str:
    """Generate an intro phrase for the digest via model 4o.

    The helper imports :func:`main.ask_4o` lazily to avoid circular imports.
    Request and response metrics are logged with ``digest.intro.llm.*`` tags.
    """

    from main import ask_4o, FOUR_O_TIMEOUT  # local import to avoid cycle
    import uuid

    run_id = uuid.uuid4().hex
    horizon_word = "неделю" if horizon_days == 7 else "две недели"
    titles_str = "; ".join(titles[:9])
    prompt = (
        "Сформулируй 1–2 предложения (≤140 символов) во вступление к дайджесту"
        f" из {n} {event_noun} на ближайшую {horizon_word} без приветствий. Используй форму: "
        f"'N {event_noun} на ближайшую ... — от X до Y.' X и Y выбери из названий ниже."
    )
    if titles_str:
        prompt += f" Названия: {titles_str}."

    logging.info(
        "digest.intro.llm.request run_id=%s n=%s horizon=%s titles_count=%s prompt_len=%s",
        run_id,
        n,
        horizon_days,
        len(titles),
        len(prompt),
    )

    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=120)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response run_id=%s ok=error text_len=0 took_ms=%s",
            run_id,
            took_ms,
        )
        raise

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    m = re.search(r"от\s+([^\s].*?)\s+до\s+([^\.]+)", text, re.IGNORECASE)
    x = m.group(1).strip() if m else ""
    y = m.group(2).strip() if m else ""
    logging.info(
        "digest.intro.llm.response run_id=%s ok=ok text_len=%s took_ms=%s x=%s y=%s",
        run_id,
        len(text),
        took_ms,
        x,
        y,
    )
    return text


def _detect_meetup_formats(event: Event, normalized: dict[str, str]) -> List[str]:
    """Return a list of human-friendly meetup formats derived from text."""

    haystack_parts = [
        normalized.get("title_clean") or event.title,
        event.description or "",
        event.event_type or "",
    ]
    haystack = " ".join(haystack_parts).lower()

    formats: List[str] = []

    def add_format(condition: bool, label: str) -> None:
        if condition and label not in formats:
            formats.append(label)

    add_format("клуб" in haystack or "club" in haystack, "клуб")
    add_format(
        re.search(r"\bвстреч", haystack) is not None
        or "meetup" in haystack
        or "meeting" in haystack,
        "встреча",
    )
    add_format(
        ("творчес" in haystack and "вечер" in haystack)
        or "creative evening" in haystack,
        "творческий вечер",
    )
    add_format("нетворкин" in haystack or "network" in haystack, "нетворкинг")
    add_format("q&a" in haystack or "q & a" in haystack, "Q&A")
    add_format("дискус" in haystack or "discussion" in haystack, "дискуссия")
    add_format("форум" in haystack or "forum" in haystack, "форум")

    return formats


_MEETUPS_TONE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "интрига": ("секрет", "закулисье", "впервые"),
    "простота": ("открытая встреча", "без подготовки"),
    "любопытство": ("узнаете", "редкие факты"),
}

_MEETUPS_TONE_PRIORITY: dict[str, int] = {
    "любопытство": 0,
    "интрига": 1,
    "простота": 2,
}

_DEFAULT_MEETUPS_TONE_HINT = "простота+любопытство"


def _update_meetups_tone(counter: Counter[str], text: str) -> None:
    """Increment tone counters based on keywords found in ``text``."""

    haystack = text.casefold()
    for tone, keywords in _MEETUPS_TONE_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            counter[tone] += 1


def _select_meetups_tone_hint(counter: Counter[str]) -> str:
    """Return combined tone hint from accumulated keyword ``counter``."""

    if not counter:
        return _DEFAULT_MEETUPS_TONE_HINT

    sorted_items = sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            _MEETUPS_TONE_PRIORITY.get(item[0], len(_MEETUPS_TONE_PRIORITY)),
        ),
    )

    top_tones = [tone for tone, count in sorted_items if count > 0][:2]

    if not top_tones:
        return _DEFAULT_MEETUPS_TONE_HINT

    return "+".join(top_tones)


async def compose_masterclasses_intro_via_4o(
    n: int, horizon_days: int, masterclasses: List[dict[str, str]]
) -> str:
    """Generate intro phrase for master-class digest via model 4o."""

    from main import ask_4o  # local import to avoid cycle
    import json
    import uuid

    run_id = uuid.uuid4().hex
    horizon_word = "неделю" if horizon_days == 7 else "две недели"
    data_json = json.dumps(masterclasses[:9], ensure_ascii=False)
    prompt = (
        "Ты помогаешь телеграм-дайджесту мероприятий."  # context
        f" Сохрани каркас «{n} мастер-классов на ближайшую {horizon_word} — …»"
        " и общий тон лекционного дайджеста без приветствий."
        " Текст сделай динамичным и коротким: 1–2 предложения до ~200 символов,"
        " помни, что слишком длинный ответ нежелателен."
        " Используй подходящие эмодзи."
        " После тире перечисли основные активности из описаний, постарайся"
        " упомянуть каждое направление, если они разные (например,"
        " «рисование, работа с голосом, создание духов»)."
        " Опирайся только на реальные активности из аннотаций ниже, не выдумывай фактов."
        " Если в описании явно указан возраст или формат группы, упомяни это."
        " Данные о мастер-классах в JSON (title — нормализованное название,"
        f" description — полная аннотация): {data_json}"
    )

    logging.info(
        "digest.intro.llm.request run_id=%s kind=masterclass n=%s horizon=%s items_count=%s prompt_len=%s",
        run_id,
        n,
        horizon_days,
        len(masterclasses),
        len(prompt),
    )

    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=160)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response run_id=%s kind=masterclass ok=error text_len=0 took_ms=%s",
            run_id,
            took_ms,
        )
        raise

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.intro.llm.response run_id=%s kind=masterclass ok=ok text_len=%s took_ms=%s",
        run_id,
        len(text),
        took_ms,
    )
    return text


async def compose_exhibitions_intro_via_4o(
    n: int, horizon_days: int, exhibitions: List[dict[str, object]]
) -> str:
    """Generate intro phrase for exhibition digest via model 4o."""

    from main import ask_4o  # local import to avoid cycle
    import json
    import uuid

    run_id = uuid.uuid4().hex
    horizon_word = "неделю" if horizon_days == 7 else "две недели"
    data_json = json.dumps(exhibitions[:9], ensure_ascii=False)
    prompt = (
        "Ты помогаешь телеграм-дайджесту мероприятий."
        f" Начни интро так: «Не пропустите, в ближайшие {horizon_word} заканчивается {n} выставок — …»."
        " Всего оставь 1–2 связных предложения до ~200 символов без списков, делая их динамичными."
        " Используй подходящие эмодзи."
        " Сделай акцент на скором завершении и ключевых темах выставок по описаниям."
        " Обязательно опирайся на поля description и date_range каждой выставки и называй даты окончания по полю end."
        " Опирайся только на факты из данных, не выдумывай детали."
        " Данные о выставках в JSON (title — нормализованное название,"
        ' description — полная аннотация, date_range — {"start": "YYYY-MM-DD",'
        ' "end": "YYYY-MM-DD"}): '
        f"{data_json}"
    )

    logging.info(
        "digest.intro.llm.request run_id=%s kind=exhibition n=%s horizon=%s items_count=%s prompt_len=%s",
        run_id,
        n,
        horizon_days,
        len(exhibitions),
        len(prompt),
    )

    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=160)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response run_id=%s kind=exhibition ok=error text_len=0 took_ms=%s",
            run_id,
            took_ms,
        )
        raise

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.intro.llm.response run_id=%s kind=exhibition ok=ok text_len=%s took_ms=%s",
        run_id,
        len(text),
        took_ms,
    )
    return text


async def compose_psychology_intro_via_4o(
    n: int, horizon_days: int, events: List[dict[str, object]]
) -> str:
    """Generate intro phrase for psychology digest via model 4o."""

    from main import ask_4o  # local import to avoid cycle
    import json
    import uuid

    run_id = uuid.uuid4().hex
    horizon_word = "неделю" if horizon_days == 7 else "две недели"
    data_json = json.dumps(events[:9], ensure_ascii=False)
    prompt = (
        "Ты помогаешь телеграм-дайджесту психологических событий."  # context
        f" Сохрани каркас «{n} психологических событий на ближайшую {horizon_word} — …»."
        " Ответ сделай на русском языке, 1–2 предложения до ~200 символов без приветствий."
        " Добавь 1–2 подходящих эмодзи."
        " Обязательно опирайся на поля topics и description, кратко объединяя основные темы"
        " (например, ментальное здоровье, осознанность, поддержка)."
        " Не выдумывай фактов, используй только данные из JSON."
        " Данные о событиях в JSON со структурой"
        ' {"title": "…", "description": "…", "topics": ["PSYCHOLOGY", …]}: '
        f"{data_json}"
    )

    logging.info(
        "digest.intro.llm.request run_id=%s kind=psychology n=%s horizon=%s items_count=%s prompt_len=%s",
        run_id,
        n,
        horizon_days,
        len(events),
        len(prompt),
    )

    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=160)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response run_id=%s kind=psychology ok=error text_len=0 took_ms=%s",
            run_id,
            took_ms,
        )
        raise

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.intro.llm.response run_id=%s kind=psychology ok=ok text_len=%s took_ms=%s",
        run_id,
        len(text),
        took_ms,
    )
    return text


async def compose_meetups_intro_via_4o(
    n: int,
    horizon_days: int,
    meetups: List[dict[str, object]],
    tone_hint: str | None = None,
) -> str:
    """Generate intro phrase for meetup digest via model 4o."""

    from main import ask_4o  # local import to avoid cycle
    import json
    import uuid

    run_id = uuid.uuid4().hex
    horizon_word = "неделю" if horizon_days == 7 else "две недели"
    data_json = json.dumps(meetups[:9], ensure_ascii=False)
    has_club = any(
        "клуб" in [fmt.casefold() for fmt in item.get("formats", [])]
        for item in meetups
    )
    has_club_flag = "true" if has_club else "false"

    hint = tone_hint or _DEFAULT_MEETUPS_TONE_HINT
    tone_tokens = [token for token in (hint or "").split("+") if token]
    if tone_tokens:
        formatted_tokens = [tone_tokens[0].capitalize()]
        formatted_tokens.extend(token.lower() for token in tone_tokens[1:])
        tone_pattern = " + ".join(formatted_tokens)
        tone_instruction = (
            f" Используй паттерн «{tone_pattern}», избегай остальных тонов."
        )
    else:
        tone_instruction = ""

    forbidden_guidance = _format_forbidden_wordings(
        MEETUPS_INTRO_FORBIDDEN_WORDINGS
    )

    prompt = (
        "Ты помогаешь телеграм-дайджесту мероприятий."
        f" Сформулируй живое интро на 1–2 предложения до ~200 символов к подборке из {n} встреч"
        f" и клубов на ближайшую {horizon_word}."
        " Добавь 1–2 уместных эмодзи, избегай приветствий и списков."
        " Опирайся на поля title, description, event_type и formats каждого события,"
        " чтобы выделить ключевые темы и форматы."
        f" Метаданные: has_club={has_club_flag}."
        f"{tone_instruction}"
        f" Избегай рекламных клише: не используй {forbidden_guidance}."
        " Обязательно подчеркни живое общение: знакомство с интересными людьми, живое Q&A"
        " и нетворкинг."
        " Если has_club=false, сделай на этом акцент ещё заметнее."
        " Не выдумывай фактов, используй только данные из JSON ниже."
        f" Данные о встречах в JSON: {data_json}"
    )

    logging.info(
        "digest.intro.llm.request run_id=%s kind=meetups n=%s horizon=%s items_count=%s prompt_len=%s has_club=%s tone_hint=%s",
        run_id,
        n,
        horizon_days,
        len(meetups),
        len(prompt),
        int(has_club),
        hint,
    )

    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=160)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.intro.llm.response run_id=%s kind=meetups ok=error text_len=0 took_ms=%s",
            run_id,
            took_ms,
        )
        raise

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.intro.llm.response run_id=%s kind=meetups ok=ok text_len=%s took_ms=%s",
        run_id,
        len(text),
        took_ms,
    )
    return text


def _normalize_exhibition_title(title: str) -> dict[str, str]:
    """Return exhibition title without lecture-specific formatting."""

    emoji, rest = _split_leading_emoji(title)
    cleaned = re.sub(r"\s+", " ", rest).strip()
    if not cleaned:
        cleaned = rest.strip() or title.strip()
    return {"emoji": emoji, "title_clean": cleaned}


def _align_event_contexts(
    events: Iterable[Any] | None, expected_len: int
) -> List[Any | None]:
    """Return a list of ``events`` padded/truncated to ``expected_len``."""

    if expected_len <= 0:
        return []

    if events is None:
        return [None] * expected_len

    seq = list(events)
    if len(seq) < expected_len:
        seq.extend([None] * (expected_len - len(seq)))
    else:
        seq = seq[:expected_len]
    return seq


def _prepare_meetup_context(event: Any | None) -> dict[str, str]:
    """Extract context fields from ``event`` for meetup normalization."""

    if event is None:
        return {"event_type": "", "description": ""}

    event_type = getattr(event, "event_type", "") or ""
    description = getattr(event, "description", "") or ""
    return {
        "event_type": str(event_type),
        "description": re.sub(r"\s+", " ", str(description)).strip(),
    }


def _truncate_context(text: str, *, limit: int = 220) -> str:
    """Trim ``text`` to ``limit`` characters while keeping whole words."""

    if len(text) <= limit:
        return text

    truncated = text[: limit + 1].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    if not truncated:
        truncated = text[:limit].rstrip()
    return truncated + "…"


def _should_add_meetup_exhibition_clarifier(
    event: Any | None, *, title: str
) -> bool:
    """Return ``True`` when meetup title should mention exhibition context."""

    lower_title = title.casefold()
    if "творческая встреча и открытие выставки" in lower_title:
        return False
    if "выстав" in lower_title:
        return False

    if event is None:
        return False

    event_type = (getattr(event, "event_type", "") or "").casefold()
    description = (getattr(event, "description", "") or "").casefold()

    if "выстав" in event_type:
        return True
    if "выстав" in description:
        return True
    return False


def _apply_meetup_postprocessing(title: str, event: Any | None) -> str:
    """Append clarifier for meetup events tied to exhibitions when needed."""

    if _should_add_meetup_exhibition_clarifier(event, title=title):
        return f"{title} — творческая встреча и открытие выставки"
    return title


async def normalize_titles_via_4o(
    titles: List[str], *, event_kind: str = "lecture", events: Iterable[Any] | None = None
) -> List[dict[str, str]]:
    """Normalize event titles using model 4o with regex fallback."""

    if event_kind == "exhibition":
        return [_normalize_exhibition_title(t) for t in titles]

    events_list = _align_event_contexts(events, len(titles))

    if event_kind == "meetups":
        from main import ask_4o  # local import to avoid a cycle
        import json

        contexts: List[str] = []
        for idx, (title, event) in enumerate(zip(titles, events_list), start=1):
            pieces = [f"Название: «{title}»"]
            ctx = _prepare_meetup_context(event)
            if ctx["event_type"]:
                pieces.append(f"Тип: {ctx['event_type']}")
            if ctx["description"]:
                pieces.append(f"Описание: {_truncate_context(ctx['description'])}")
            contexts.append(f"{idx}. " + " | ".join(pieces))

        prompt_titles = "\n".join(contexts) if contexts else " | ".join(titles)
        prompt = (
            "Язык: русский.\n\n"
            "Задача: вернуть JSON-массив объектов вида:\n\n"
            '{"emoji": "👥" | "", "title_clean": "Название"}\n\n'
            "Требования:\n"
            "- Очистить лишние пробелы и переносы строк.\n"
            "- Перенести ведущий эмодзи (если был) в поле emoji.\n"
            "- Сохранить кавычки и ключевые слова в названии.\n"
            "- Если по типу события или описанию видно, что встреча совмещена с открытием выставки, добавь в конец заголовка пояснение «— творческая встреча и открытие выставки».\n"
            "- Без дополнительных слов, только JSON.\n\n"
            "События:\n"
            + prompt_titles
        )

        logging.info(
            "digest.titles.llm.request kind=%s n=%s prompt_len=%s",
            "meetups",
            len(titles),
            len(prompt),
        )
        start = time.monotonic()
        try:
            text = await ask_4o(prompt, max_tokens=300)
        except Exception:
            took_ms = int((time.monotonic() - start) * 1000)
            logging.info(
                "digest.titles.llm.response error text_len=0 took_ms=%s", took_ms
            )
            return [
                _normalize_title_fallback(t, event_kind="meetups", event=ev)
                for t, ev in zip(titles, events_list)
            ]

        took_ms = int((time.monotonic() - start) * 1000)
        text = text.strip()
        logging.info(
            "digest.titles.llm.response ok text_len=%s took_ms=%s",
            len(text),
            took_ms,
        )

        try:
            data = json.loads(text)
            result: List[dict[str, str]] = []
            for orig, item, event in zip(titles, data, events_list):
                emoji = item.get("emoji") or ""
                title_clean_raw = item.get("title_clean") or item.get("title") or orig
                title_clean = _apply_meetup_postprocessing(title_clean_raw, event)
                result.append({"emoji": emoji, "title_clean": title_clean})
                logging.info(
                    "digest.titles.llm.sample kind=%s before=%r after=%r emoji=%s",
                    "meetups",
                    orig,
                    title_clean,
                    emoji,
                )
            if len(result) == len(titles):
                return result
        except Exception:
            pass

        return [
            _normalize_title_fallback(t, event_kind="meetups", event=ev)
            for t, ev in zip(titles, events_list)
        ]

    from main import ask_4o  # local import to avoid a cycle
    import json

    prompt_titles = " | ".join(titles)
    kind = event_kind if event_kind in {"lecture", "masterclass"} else "lecture"
    if kind == "masterclass":
        event_word = "Мастер-класс"
        removal_phrase = "«Мастер-класс», «Мастер класс»"
        role_word = "ведущего"
        examples = [
            'Вход: «🎨 Мастер-класс Марии Ивановой «Ботаническая иллюстрация»» → {"emoji":"🎨","title_clean":"Мастер-класс Марии Ивановой: Ботаническая иллюстрация"}\n',
            'Вход: «Мастер класс “Готовим штрудель”» → {"emoji":"","title_clean":"Готовим штрудель"}\n',
            'Вход: «🧵 Мастер-класс «Вышивка гладью для начинающих»» → {"emoji":"🧵","title_clean":"Вышивка гладью для начинающих"}\n',
        ]
    else:
        event_word = "Лекция"
        removal_phrase = "«Лекция», «Лекторий»"
        role_word = "лектора"
        examples = [
            'Вход: «📚 Лекция Алёны Мирошниченко «Мода Франции…»» → {"emoji":"📚","title_clean":"Лекция Алёны Мирошниченко: Мода Франции…"}\n',
            'Вход: «Лекторий Ильи Дементьева “От каменного века…”» → {"emoji":"","title_clean":"Лекция Ильи Дементьева: От каменного века…"}\n',
            'Вход: «Лекция «Древнерусское искусство. Мастера и эпохи»» → {"emoji":"","title_clean":"Древнерусское искусство. Мастера и эпохи"}\n',
        ]

    examples_str = "".join(examples)
    prompt = (
        "Язык: русский.\n\n"
        "Задача: вернуть JSON-массив объектов вида:\n\n"
        f'{{"emoji": "📚" | "", "title_clean": "{event_word} Имя Фамилия: Название" | "Название"}}\n\n'
        "Требования:\n\n"
        f"Удалять слова {removal_phrase} и т.п. из исходного заголовка.\n\n"
        f"Если в исходном названии есть имя {role_word} (в любой форме), привести имя и фамилию к родительному падежу (Р.п.) без отчества и сформировать заголовок: '{event_word} Имя Фамилия: Название'.\n\n"
        f"Если {role_word} нет — просто 'Название' без слова '{event_word}'.\n\n"
        "Ведущий эмодзи (если был) вернуть в поле emoji (не внутри title_clean).\n\n"
        "Без дополнительных слов/пояснений, только JSON.\n\n"
        "Примеры:\n"
        + examples_str
        + "\n"
        "Заголовки: "
        + prompt_titles
    )
    logging.info(
        "digest.titles.llm.request kind=%s n=%s prompt_len=%s",
        kind,
        len(titles),
        len(prompt),
    )
    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=300)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.titles.llm.response error text_len=0 took_ms=%s", took_ms
        )
        return [
            _normalize_title_fallback(t, event_kind=kind, event=ev)
            for t, ev in zip(titles, events_list)
        ]

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.titles.llm.response ok text_len=%s took_ms=%s", len(text), took_ms
    )

    try:
        data = json.loads(text)
        result: List[dict[str, str]] = []
        for orig, item, event in zip(titles, data, events_list):
            emoji = item.get("emoji") or ""
            title_clean = item.get("title_clean") or item.get("title") or orig
            result.append({"emoji": emoji, "title_clean": title_clean})
            logging.info(
                "digest.titles.llm.sample kind=%s before=%r after=%r emoji=%s",
                kind,
                orig,
                title_clean,
                emoji,
            )
        if len(result) == len(titles):
            return result
    except Exception:
        pass

    return [
        _normalize_title_fallback(t, event_kind=kind, event=ev)
        for t, ev in zip(titles, events_list)
    ]




_LEADING_EMOJI_RE = re.compile(
    r"^([\U0001F300-\U0010FFFF](?:\uFE0F|[\U0001F3FB-\U0001F3FF])?"
    r"(?:\u200D[\U0001F300-\U0010FFFF](?:\uFE0F|[\U0001F3FB-\U0001F3FF])?)*)"
)


def _split_leading_emoji(title: str) -> tuple[str, str]:
    match = _LEADING_EMOJI_RE.match(title)
    if not match:
        return "", title
    emoji = match.group(0)
    return emoji, title[len(emoji) :]


_NAME_PART_RE = re.compile(r"^[A-ZА-ЯЁ][a-zа-яё]+(?:-[A-ZА-ЯЁ][a-zа-яё]+)*$")


def _looks_like_full_name(candidate: str) -> bool:
    parts = candidate.split()
    if len(parts) != 2:
        return False
    return all(_NAME_PART_RE.match(part) for part in parts)


def _normalize_title_fallback(
    title: str, *, event_kind: str = "lecture", event: Any | None = None
) -> dict[str, str]:
    """Fallback normalization used when LLM is unavailable."""

    emoji, rest = _split_leading_emoji(title)

    if event_kind == "exhibition":
        return _normalize_exhibition_title(title)

    if event_kind == "meetups":
        cleaned = re.sub(r"\s+", " ", rest).strip() or rest.strip() or title.strip()
        cleaned = _apply_meetup_postprocessing(cleaned, event)
        return {"emoji": emoji, "title_clean": cleaned}

    kind = event_kind if event_kind in {"lecture", "masterclass"} else "lecture"
    if kind == "masterclass":
        removal_pattern = r"^(?:[^\w]*?)*(?:Мастер[\s-]*класс)[\s:—-]*"
        prefix = "Мастер-класс"
    else:
        removal_pattern = r"^(?:[^\w]*?)*(?:Лекция|Лекторий)[\s:—-]*"
        prefix = "Лекция"

    title = re.sub(removal_pattern, "", rest, flags=re.IGNORECASE)
    title = re.sub(r"^от\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()

    m = re.match(
        r"^(?P<who>[\wЁёА-Яа-я-]+\s+[\wЁёА-Яа-я-]+)[\s—:-]+(?P<what>.+)$",
        title,
    )
    if m:
        who = m.group("who").strip()
        what = m.group("what").strip()
        if _looks_like_full_name(who):
            title = f"{prefix} {who}: {what}"

    return {"emoji": emoji, "title_clean": title}


def pick_display_link(event: Event) -> str | None:
    """Return a display link for ``event`` if available.

    Priority:
    1. ``source_post_url``
    2. ``telegraph_url``
    3. ``telegraph_path`` (normalized to ``https://telegra.ph``)

    Additionally logs the chosen source.
    """

    chosen = "none"
    url = None
    if event.source_post_url:
        url = event.source_post_url
        chosen = "source_post"
    elif event.telegraph_url:
        url = event.telegraph_url
        chosen = "telegraph"
    elif event.telegraph_path:
        url = f"https://telegra.ph/{event.telegraph_path.lstrip('/')}"
        chosen = "path"
    logging.info(
        "digest.link.pick event_id=%s chosen=%s url=%s",
        getattr(event, "id", None),
        chosen,
        url,
    )
    return url


def format_event_line_html(
    event: Event,
    link_url: str | None,
    *,
    emoji: str = "",
    title_override: str | None = None,
) -> str:
    """Format event information for digest list as HTML.

    Parameters
    ----------
    event:
        Source event.
    link_url:
        URL to wrap the title into. ``None`` disables the link.
    emoji:
        Optional emoji placed before the title. When present the ``|``
        separator is omitted as the emoji acts as a visual marker.
    title_override:
        Normalized title to use instead of ``event.title``.
    """

    dt = datetime.strptime(event.date, "%Y-%m-%d")
    date_part = dt.strftime("%d.%m")
    is_exhibition = (event.event_type or "").lower() == "выставка"
    if is_exhibition:
        end_raw = getattr(event, "end_date", None)
        if end_raw:
            try:
                end_dt = datetime.strptime(end_raw, "%Y-%m-%d")
            except ValueError:
                logging.warning(
                    "digest.end_date.format event_id=%s end_date_raw=%r",
                    getattr(event, "id", None),
                    end_raw,
                )
            else:
                date_part = f"по {end_dt.strftime('%d.%m')}"
        else:
            logging.warning(
                "digest.end_date.missing event_id=%s event_type=%r",
                getattr(event, "id", None),
                event.event_type,
            )
        if not date_part.startswith("по "):
            date_part = f"по {dt.strftime('%d.%m')}"
        time_part = ""
    else:
        time_part = ""
        parsed = parse_start_time(event.time or "")
        if parsed is not None:
            hh, mm = parsed
            time_part = f" {hh:02d}:{mm:02d}"
        else:
            logging.warning(
                "digest.time.format event_id=%s time_raw=%r parsed=none",
                getattr(event, "id", None),
                event.time,
            )

    title = title_override or event.title
    if link_url:
        title_part = f'<a href="{link_url}">{title}</a>'
    else:
        title_part = title

    if emoji:
        return f"{date_part}{time_part} {emoji} {title_part}".strip()
    return f"{date_part}{time_part} | {title_part}"


def _truncate_html_title(line: str, limit: int) -> str:
    """Truncate title text inside HTML link in ``line`` to ``limit`` chars."""

    def repl(match: re.Match[str]) -> str:
        url = match.group("url")
        title = match.group("title")
        if len(title) > limit:
            title = title[: limit - 3] + "..."
        return f'<a href="{url}">{title}</a>'

    return re.sub(r'<a href="(?P<url>[^"]+)">(?P<title>[^<]+)</a>', repl, line)


def _strip_quotes_dashes(line: str) -> str:
    """Remove angle quotes and long dashes from title part of ``line``."""

    def repl(match: re.Match[str]) -> str:
        url = match.group("url")
        title = match.group("title").replace("«", "").replace("»", "").replace("—", "-")
        return f'<a href="{url}">{title}</a>'

    return re.sub(r'<a href="(?P<url>[^"]+)">(?P<title>[^<]+)</a>', repl, line)


def visible_caption_len(html_text: str) -> int:
    """Return length of caption text visible to humans.

    The function removes HTML tags while keeping the inner text of
    anchors, strips raw ``http(s)://`` URLs and collapses repeating
    whitespace and newlines. The result mirrors the length counted by
    Telegram for media group captions.
    """

    # Replace anchors with their inner text
    s = re.sub(r"<a\s+[^>]*>(.*?)</a>", r"\1", html_text, flags=re.IGNORECASE | re.DOTALL)
    # Drop all remaining tags
    s = re.sub(r"<[^>]+>", "", s)
    # Decode HTML entities
    s = html.unescape(s)
    # Remove raw URLs that might remain in text
    s = re.sub(r"https?://\S+", "", s)
    # Normalize whitespace and newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", "\n", s)
    s = s.strip()
    return len(s)


def attach_caption_if_fits(
    media: List["types.InputMediaPhoto"], caption_html: str
) -> tuple[bool, int]:
    """Attach caption to first media item if visible length allows.

    Returns a tuple ``(attached, visible_len)`` where ``attached`` indicates
    whether the caption was placed on the first photo and ``visible_len`` is the
    human‑visible length of ``caption_html``.  The function mutates ``media`` in
    place when attaching the caption.
    """

    from aiogram import types

    visible_len = visible_caption_len(caption_html)
    if media and visible_len <= 1024:
        first = media[0]
        media[0] = types.InputMediaPhoto(
            media=first.media, caption=caption_html, parse_mode="HTML"
        )
        return True, visible_len
    return False, visible_len


async def compose_digest_caption(
    intro_text: str,
    lines_html: List[str],
    footer_html: str,
    excluded: Iterable[int] | None = None,
    *,
    digest_id: str | None = None,
) -> tuple[str, List[str]]:
    """Compose digest caption respecting Telegram visible length limits."""

    excluded_set = set(excluded or [])
    lines = [
        line for idx, line in enumerate(lines_html) if idx not in excluded_set
    ]

    def build_caption(current: List[str]) -> str:
        body = intro_text.strip()
        if current:
            body += "\n\n" + "\n".join(current)
        body += "\n\n" + footer_html
        return body

    caption = build_caption(lines)
    visible_len = visible_caption_len(caption)
    while lines and visible_len > 4096:
        lines.pop()
        caption = build_caption(lines)
        visible_len = visible_caption_len(caption)
    logging.info(
        "digest.caption.compose digest_id=%s visible_len=%s fit_1024=%s",
        digest_id,
        visible_len,
        int(visible_len <= 1024),
    )
    return caption, lines


async def assemble_compact_caption(
    intro: str, items_html: List[str], *, digest_id: str | None = None
) -> tuple[str, List[str]]:
    """Assemble caption trimmed to Telegram's 4096 char visible limit."""

    footer = '<a href="https://t.me/kenigevents">Полюбить Калининград | Анонсы</a>'
    return await compose_digest_caption(
        intro, items_html, footer, excluded=None, digest_id=digest_id
    )


async def _build_digest_preview(
    digest_id: str,
    db: Database,
    now: datetime,
    *,
    kind: str,
    event_noun: str,
    event_kind: str,
    candidates_builder: Callable[
        [Database, datetime, str | None], Awaitable[Tuple[List[Event], int]]
    ],
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Generic helper for assembling digest previews."""

    start = time.monotonic()
    logging.info(
        "digest.collect.start digest_id=%s kind=%s window_days=14 now=%s limit=9",
        digest_id,
        kind,
        now.isoformat(),
    )
    events, horizon = await candidates_builder(db, now, digest_id)
    duration_ms = int((time.monotonic() - start) * 1000)
    cutoff_plus_2h = now + timedelta(hours=2)
    logging.info(
        "digest.collect.end digest_id=%s kind=%s window_days=%s now=%s cutoff_plus_2h=%s count_found=%s count_after_filters=%s limit=9 duration_ms=%s",
        digest_id,
        kind,
        horizon,
        now.isoformat(),
        cutoff_plus_2h.isoformat(),
        len(events),
        len(events),
        duration_ms,
    )

    if not events:
        return "", [], horizon, [], []

    titles = [e.title for e in events]
    normalized = await normalize_titles_via_4o(
        titles, event_kind=event_kind, events=events
    )

    if event_kind == "masterclass":
        masterclasses_payload: List[dict[str, str]] = []
        for ev, norm in zip(events, normalized):
            title_clean = norm.get("title_clean") or ev.title
            masterclasses_payload.append(
                {
                    "title": title_clean,
                    "description": (ev.description or "").strip(),
                }
            )
        intro = await compose_masterclasses_intro_via_4o(
            len(events), horizon, masterclasses_payload
        )
    elif event_kind == "exhibition":
        exhibitions_payload: List[dict[str, object]] = []
        for ev, norm in zip(events, normalized):
            title_clean = norm.get("title_clean") or ev.title
            exhibitions_payload.append(
                {
                    "title": title_clean,
                    "description": (ev.description or "").strip(),
                    "date_range": {
                        "start": ev.date,
                        "end": ev.end_date or ev.date,
                    },
                }
            )
        intro = await compose_exhibitions_intro_via_4o(
            len(events), horizon, exhibitions_payload
        )
    elif event_kind == "psychology":
        psychology_payload: List[dict[str, object]] = []
        for ev, norm in zip(events, normalized):
            title_clean = norm.get("title_clean") or ev.title
            psychology_payload.append(
                {
                    "title": title_clean,
                    "description": (ev.description or "").strip(),
                    "topics": normalize_topics(getattr(ev, "topics", [])),
                }
            )
        intro = await compose_psychology_intro_via_4o(
            len(events), horizon, psychology_payload
        )
    elif event_kind == "meetups":
        meetups_payload: List[dict[str, object]] = []
        tone_counter: Counter[str] = Counter()
        for ev, norm in zip(events, normalized):
            title_clean = norm.get("title_clean") or ev.title
            description = (ev.description or "").strip()
            meetups_payload.append(
                {
                    "title": title_clean,
                    "description": description,
                    "event_type": (ev.event_type or "").strip(),
                    "formats": _detect_meetup_formats(ev, norm),
                }
            )
            _update_meetups_tone(tone_counter, f"{title_clean} {description}")
        tone_hint = _select_meetups_tone_hint(tone_counter)
        intro = await compose_meetups_intro_via_4o(
            len(events), horizon, meetups_payload, tone_hint
        )
    else:
        intro = await compose_digest_intro_via_4o(
            len(events), horizon, titles, event_noun=event_noun
        )
    lines: List[str] = []
    norm_titles: List[str] = []
    for ev, norm in zip(events, normalized):
        link = pick_display_link(ev)
        title_clean = norm.get("title_clean")
        norm_titles.append(title_clean or ev.title)
        lines.append(
            format_event_line_html(
                ev,
                link,
                emoji=norm.get("emoji", ""),
                title_override=title_clean,
            )
        )
    return intro, lines, horizon, events, norm_titles


async def build_lectures_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for lectures."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="lectures",
        event_noun="лекций",
        event_kind="lecture",
        candidates_builder=build_lectures_digest_candidates,
    )


async def build_masterclasses_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for master-classes."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="masterclasses",
        event_noun="мастер-классов",
        event_kind="masterclass",
        candidates_builder=build_masterclasses_digest_candidates,
    )


async def build_exhibitions_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for exhibitions."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="exhibitions",
        event_noun="выставок",
        event_kind="exhibition",
        candidates_builder=build_exhibitions_digest_candidates,
    )


async def build_psychology_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for psychology events."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="psychology",
        event_noun="психологических событий",
        event_kind="psychology",
        candidates_builder=build_psychology_digest_candidates,
    )


async def build_science_pop_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for science-pop events."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="science_pop",
        event_noun="научно-популярных событий",
        event_kind="science_pop",
        candidates_builder=build_science_pop_digest_candidates,
    )


async def build_kraevedenie_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for Kaliningrad heritage events."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="kraevedenie",
        event_noun="краеведческих событий",
        event_kind="kraevedenie",
        candidates_builder=build_kraevedenie_digest_candidates,
    )


async def build_networking_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for networking events."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="networking",
        event_noun="нетворкингов",
        event_kind="networking",
        candidates_builder=build_networking_digest_candidates,
    )


async def build_entertainment_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for entertainment events."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="entertainment",
        event_noun="развлечений",
        event_kind="entertainment",
        candidates_builder=build_entertainment_digest_candidates,
    )


async def build_markets_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for handmade markets."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="markets",
        event_noun="маркетов",
        event_kind="markets",
        candidates_builder=build_markets_digest_candidates,
    )


async def build_theatre_classic_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for classic theatre performances."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="theatre_classic",
        event_noun="классических спектаклей",
        event_kind="theatre_classic",
        candidates_builder=build_theatre_classic_digest_candidates,
    )


async def build_theatre_modern_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for modern theatre performances."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="theatre_modern",
        event_noun="современных спектаклей",
        event_kind="theatre_modern",
        candidates_builder=build_theatre_modern_digest_candidates,
    )


async def build_meetups_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for meetups and clubs."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="meetups",
        event_noun="встреч и клубов",
        event_kind="meetups",
        candidates_builder=build_meetups_digest_candidates,
    )


async def build_movies_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for movie screenings."""

    return await _build_digest_preview(
        digest_id,
        db,
        now,
        kind="movies",
        event_noun="кинопоказов",
        event_kind="movies",
        candidates_builder=build_movies_digest_candidates,
    )


