from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import html
import logging
import re
import time
from typing import Iterable, List, Tuple
import httpx

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
    result: List[Event] = []
    for ev in events:
        if _event_start_datetime(ev, digest_id) > end_7:
            continue
        if pick_display_link(ev) is None:
            logging.info(
                "digest.skip.no_link event_id=%s title=%r", getattr(ev, "id", None), ev.title
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
                    "digest.skip.no_link event_id=%s title=%r", getattr(ev, "id", None), ev.title
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
    n: int, horizon_days: int, titles: List[str]
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
        f" из {n} лекций на ближайшую {horizon_word} без приветствий. Используй форму: "
        "'N лекций на ближайшую ... — от X до Y.' X и Y выбери из названий ниже."
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


async def normalize_titles_via_4o(titles: List[str]) -> List[dict[str, str]]:
    """Normalize lecture titles using model 4o with regex fallback."""

    from main import ask_4o  # local import to avoid a cycle
    import json

    prompt_titles = " | ".join(titles)
    prompt = (
        "Язык: русский.\n\n"
        "Задача: вернуть JSON-массив объектов вида:\n\n"
        '{"emoji": "📚" | "", "title_clean": "Лекция Имя Фамилия: Название" | "Название"}\n\n'
        "Требования:\n\n"
        "Удалять слова «Лекция», «Лекторий» и т.п. из исходного заголовка.\n\n"
        "Если в исходном названии есть имя лектора (в любой форме), привести имя и фамилию к родительному падежу (Р.п.) без отчества и сформировать заголовок: 'Лекция Имя Фамилия: Название'.\n\n"
        "Если лектора нет — просто 'Название' без слова 'Лекция'.\n\n"
        "Ведущий эмодзи (если был) вернуть в поле emoji (не внутри title_clean).\n\n"
        "Без дополнительных слов/пояснений, только JSON.\n\n"
        "Примеры:\n"
        'Вход: «📚 Лекция Алёны Мирошниченко «Мода Франции…»» → {"emoji":"📚","title_clean":"Лекция Алёны Мирошниченко: Мода Франции…"}\n'
        'Вход: «Лекторий Ильи Дементьева “От каменного века…”» → {"emoji":"","title_clean":"Лекция Ильи Дементьева: От каменного века…"}\n'
        'Вход: «Лекция «Древнерусское искусство. Мастера и эпохи»» → {"emoji":"","title_clean":"Древнерусское искусство. Мастера и эпохи"}\n\n'
        "Заголовки: "
        + prompt_titles
    )
    logging.info(
        "digest.titles.llm.request n=%s prompt_len=%s", len(titles), len(prompt)
    )
    start = time.monotonic()
    try:
        text = await ask_4o(prompt, max_tokens=300)
    except Exception:
        took_ms = int((time.monotonic() - start) * 1000)
        logging.info(
            "digest.titles.llm.response error text_len=0 took_ms=%s", took_ms
        )
        return [_normalize_title_fallback(t) for t in titles]

    took_ms = int((time.monotonic() - start) * 1000)
    text = text.strip()
    logging.info(
        "digest.titles.llm.response ok text_len=%s took_ms=%s", len(text), took_ms
    )

    try:
        data = json.loads(text)
        result: List[dict[str, str]] = []
        for orig, item in zip(titles, data):
            emoji = item.get("emoji") or ""
            title_clean = item.get("title_clean") or item.get("title") or orig
            result.append({"emoji": emoji, "title_clean": title_clean})
            logging.info(
                "digest.titles.llm.sample before=%r after=%r emoji=%s",
                orig,
                title_clean,
                emoji,
            )
        if len(result) == len(titles):
            return result
    except Exception:
        pass

    return [_normalize_title_fallback(t) for t in titles]


def _normalize_title_fallback(title: str) -> dict[str, str]:
    """Fallback normalization used when LLM is unavailable."""

    # Extract leading emoji if any
    emoji_match = re.match(r"^[\U0001F300-\U0010FFFF]", title)
    emoji = ""
    if emoji_match:
        emoji = emoji_match.group(0)
        title = title[len(emoji) :]

    title = re.sub(
        r"^(?:[^\w]*?)*(?:Лекция|Лекторий)[\s:—-]*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^от\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()

    m = re.match(
        r"^(?P<who>[\wЁёА-Яа-я-]+\s+[\wЁёА-Яа-я-]+)[\s—:-]+(?P<what>.+)$",
        title,
    )
    if m:
        who = m.group("who").strip()
        what = m.group("what").strip()
        title = f"Лекция {who}: {what}"

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


async def build_lectures_digest_preview(
    digest_id: str, db: Database, now: datetime
) -> tuple[str, List[str], int, List[Event], List[str]]:
    """Build digest preview text for lectures.

    Returns intro phrase, list of formatted event lines, horizon in days,
    the underlying events and normalized titles.
    """

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
        return "", [], horizon, [], []

    intro = await compose_digest_intro_via_4o(
        len(events), horizon, [e.title for e in events]
    )

    normalized = await normalize_titles_via_4o([e.title for e in events])
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
