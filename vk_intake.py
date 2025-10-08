from __future__ import annotations

import asyncio
import calendar
import hashlib
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, List, Sequence
from datetime import date, datetime, timedelta, timezone

from db import Database
from poster_media import (
    PosterMedia,
    apply_ocr_results_to_media,
    build_poster_summary,
    collect_poster_texts,
    process_media,
)
import poster_ocr

from sections import MONTHS_RU
from runtime import require_main_attr
from supabase_export import SBExporter

logger = logging.getLogger(__name__)

# Crawl tuning parameters
VK_CRAWL_PAGE_SIZE = int(os.getenv("VK_CRAWL_PAGE_SIZE", "30"))
VK_CRAWL_MAX_PAGES_INC = int(os.getenv("VK_CRAWL_MAX_PAGES_INC", "1"))
VK_CRAWL_OVERLAP_SEC = int(os.getenv("VK_CRAWL_OVERLAP_SEC", "300"))
VK_CRAWL_PAGE_SIZE_BACKFILL = int(os.getenv("VK_CRAWL_PAGE_SIZE_BACKFILL", "50"))
VK_CRAWL_MAX_PAGES_BACKFILL = int(os.getenv("VK_CRAWL_MAX_PAGES_BACKFILL", "3"))
VK_CRAWL_BACKFILL_DAYS = int(os.getenv("VK_CRAWL_BACKFILL_DAYS", "14"))
VK_CRAWL_BACKFILL_AFTER_IDLE_H = int(os.getenv("VK_CRAWL_BACKFILL_AFTER_IDLE_H", "24"))
VK_CRAWL_BACKFILL_OVERRIDE_MAX_DAYS = int(
    os.getenv("VK_CRAWL_BACKFILL_OVERRIDE_MAX_DAYS", "60")
)
VK_USE_PYMORPHY = os.getenv("VK_USE_PYMORPHY", "false").lower() == "true"

# Sentinel used to flag posts awaiting poster OCR before keyword/date checks.
OCR_PENDING_SENTINEL = "__ocr_pending__"

HISTORY_MATCHED_KEYWORD = "history"


def _normalize_group_title(value: str | None) -> str | None:
    if not value:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()
    if not normalized:
        return None
    return normalized.casefold()


def _display_group_title(value: str | None, gid: int) -> str:
    if not value:
        return f"club{gid}"
    display = unicodedata.normalize("NFKC", value)
    display = display.replace("\xa0", " ")
    display = re.sub(r"\s+", " ", display)
    display = display.strip()
    if not display:
        return f"club{gid}"
    return display


def _normalize_group_screen_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\xa0", " ")
    normalized = normalized.strip().lstrip("@")
    if not normalized:
        return None
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized:
        return None
    return normalized.casefold()


def _display_group_screen_name(value: str | None, gid: int) -> str:
    if not value:
        return f"club{gid}"
    display = unicodedata.normalize("NFKC", value)
    display = display.replace("\xa0", " ")
    display = display.strip().lstrip("@")
    if not display:
        return f"club{gid}"
    display = re.sub(r"\s+", "", display)
    if not display:
        return f"club{gid}"
    return display


# optional pymorphy3 initialisation
MORPH = None
if VK_USE_PYMORPHY:  # pragma: no cover - optional dependency
    try:
        import pymorphy3

        MORPH = pymorphy3.MorphAnalyzer()
    except Exception:
        VK_USE_PYMORPHY = False

# Keyword patterns for regex-based matching
GROUP_CONTEXT_PATTERN = r"групп[аы]\s+[\"«'][^\"»']+[\"»']"

KEYWORD_PATTERNS = [
    r"лекци(я|и|й|е|ю|ями|ях)",
    r"спектакл(ь|я|ю|ем|е|и|ей|ям|ями|ях)",
    r"концерт(ы|а|у|е|ом|ов|ам|ами|ах)",
    r"фестивал(ь|я|ю|е|ем|и|ей|ям|ями|ях)|festival",
    r"ф[её]ст(а|у|ом|е|ы|ов|ам|ами|ах)?",
    r"fest",
    r"м(?:а|а?стер)[-\s]?класс(ы|а|е|ом|ов|ам|ами|ах)|мк\b",
    r"воркшоп(ы|а|е|ом|ов|ам|ами|ах)|workshop",
    r"показ(ы|а|е|ом|ов|ам|ами|ах)|кинопоказ",
    r"лекто(р|рия|рий|рии|риями|риях)|кинолекторий",
    r"выставк(а|и|е|у|ой|ам|ами|ах)",
    r"экскурси(я|и|е|ю|ей|ям|ями|ях)",
    r"читк(а|и|е|у|ой|ам|ами|ах)",
    r"перформанс(ы|а|е|ом|ов|ам|ами|ах)",
    r"встреч(а|и|е|у|ей|ам|ами|ах)",
    r"событ(?:ие|ия|ий|иях|иями|ию|ием|иям)",
    r"праздник(и|а|у|е|ом|ов|ам|ами|ах)?",
    r"праздничн(?:ый|ая|ое|ые|ого|ому|ым|ых|ую|ой|ыми|ом)",
    r"музыкальн(?:ое|ый|ая|ые|ым|ых|ом|ой|ому|ыми)",
    r"музык(?:а|и|е|у|ой|ою)",
    r"стих(?:и|отворен\w*)",
    r"песн(?:я|и|ей|е|ю|ями|ях|ью)",
    r"фортепиан(?:о|ный|ная|ные|ной|ном|ного|ному|ным|ных|нюю|ными)",
    r"сыгра\w*",
    r"жив(?:ой|ого|ым|ом)\s+звук(?:а|ом|у|и|ов)?",
    r"жив(?:ое|ом)?\s+исполнен\w*",
    r"выступлени(?:е|я|ю|ем|ями|ях)",
    r"хит(?:ы|ов|ом|ам|ами|ах)?",
    r"в\s+исполнен(?:ии|ием|ию)",
    r"в\s+программе[^\n,.!?]{0,40}?произведен(?:ие|ия|ий)",
    r"композитор(?:а|ов|ы)",
    GROUP_CONTEXT_PATTERN,
    r"band",
    r"бронировани(е|я|ю|ем)|билет(ы|а|ов)|регистраци(я|и|ю|ей)|афиш(а|и|е|у)",
    r"ведущ(ий|ая|ее|ие|его|ему|ем|им|их|ими|ую|ей)",
    r"караок[её]",
    r"трибь?ют|трибут|tribute(?:\s+show)?",
    r"дайджест(ы|а|у|ом|ах)?",
    r"приглашаем\s+(?:вас\s+)?на",
    r"пушкинск(?:ая|ой)\s+карт(?:а|у|е)",
]
KEYWORD_RE = re.compile(r"(?<!\w)#?(?:" + "|".join(KEYWORD_PATTERNS) + r")(?!\w)", re.I | re.U)
GROUP_CONTEXT_RE = re.compile(GROUP_CONTEXT_PATTERN, re.I | re.U)
GROUP_NAME_RE = re.compile(
    r"групп[аы]\s+[A-ZА-ЯЁ0-9][^\s,.:;!?]*(?:\s+[A-ZА-ЯЁ0-9][^\s,.:;!?]*){0,2}",
    re.U,
)

# Pricing patterns provide an additional hint for event-like posts
PRICE_AMOUNT_PATTERN = "\\d+(?:[ \\t\\u00a0\\u202f]\\d+)*"
PRICE_PATTERNS = [
    r"вход\s+свободн(?:ый|а|о)",
    r"бесплатн(?:о|ый|ая|ое|ые|ую|ым|ыми|ом|ых)",
    r"\bплатн(?:о|ый|ая|ое|ые|ую|ым|ыми|ом|ых)\b",
    r"\bстоимост[ьи]\b",
    r"\bпо\s+донат(?:у|ам)?\b",
    r"\bдонат(?:а|у|ом|ы)?\b",
    r"\bпожертвовани[еяюомьях]*\b",
    r"\bвзнос\b",
    r"\bоплат\w*\b",
    rf"(?:₽|руб(?:\.|лей|ля|ль)?|р\.?)\s*{PRICE_AMOUNT_PATTERN}",
    rf"\b{PRICE_AMOUNT_PATTERN}\s*(?:₽|руб(?:\.|лей|ля|ль)?|р\.?)",
    r"\bруб(?:\.|лей|ля|ль|ы)?\b",
]
PRICE_RE = re.compile("(?:" + "|".join(PRICE_PATTERNS) + ")", re.I | re.U)

# Canonical keywords for morphological mode
KEYWORD_LEMMAS = {
    "лекция",
    "спектакль",
    "концерт",
    "фестиваль",
    "фест",
    "fest",
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
    "событие",
    "праздник",
    "музыка",
    "музыкальный",
    "стих",
    "поэзия",
    "песня",
    "фортепиано",
    "сыграть",
    "хит",
    "исполнение",
    "выступление",
    "произведение",
    "композитор",
    "бронирование",
    "билет",
    "регистрация",
    "афиша",
    "ведущий",
    "караоке",
    "трибьют",
    "трибут",
    "tribute",
    "band",
    "дайджест",
    "приглашать",
}

# Date/time patterns used for quick detection
MONTH_NAMES_DET = "|".join(sorted(re.escape(m) for m in MONTHS_RU.keys()))
DATE_PATTERNS = [
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?(?!-\d)\b",
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

DATE_PATTERN_STRONG_INDEXES = (0, 1, 2, 3, 4, 8)

PAST_EVENT_RE = re.compile(
    r"\b("
    r"состоял(?:ась|ось|ся|и|а)?|"
    r"прош[её]л(?:и|а)?|"
    r"проходил(?:и|а|о)?|"
    r"завершил(?:ись|ась|ось|ся|и|а|о)?|"
    r"отгремел(?:а|и|о)?"
    r")\b",
    re.I,
)

HISTORICAL_TOPONYMS = [
    "кёнигсберг",
    "кенигсберг",
    "гумбинен",
    "инстербург",
    "тильзит",
    "мемель",
    "тапиау",
    "кранц",
    "раушен",
    "пиллау",
    "роминта",
    "гердауэн",
    "гёрдауэн",
    "гердауен",
    "гёрдауен",
    "пруссия",
    "восточная пруссия",
    "восточной пруссии",
]
HISTORICAL_YEAR_RE = re.compile(r"\b(1\d{3})\b")

NUM_DATE_RE = re.compile(
    r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!-\d)\b"
)
PHONE_LIKE_RE = re.compile(r"^(?:\d{2}-){2,}\d{2}$")
FEDERAL_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\+7|8)\D*\d(?:\D*\d){9}")
CITY_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)\d(?:\D*\d){5}(?!\d)")
PHONE_CANDIDATE_RE = re.compile(
    rf"(?:{FEDERAL_PHONE_CANDIDATE_RE.pattern})|(?:{CITY_PHONE_CANDIDATE_RE.pattern})"
)
PHONE_CONTEXT_RE = re.compile(
    r"(\bтел(?:[.:]|ефон\w*|\b|(?=\d))|\bзвоните\b|\bзвонок\w*)",
    re.I | re.U,
)
EVENT_LOCATION_PREFIXES = (
    "клуб",
    "бар",
    "каф",
    "рест",
    "театр",
    "музе",
    "дом",
    "дк",
    "центр",
    "парк",
    "сад",
    "площад",
    "зал",
    "галер",
    "библиот",
    "филармон",
    "кин",
    "сц",
    "арен",
    "лофт",
    "коворк",
    "конгресс",
    "форум",
    "павиль",
    "дворц",
    "манеж",
    "усадь",
    "гостин",
    "отел",
    "hotel",
    "пансион",
    "санатор",
    "лагер",
    "база",
    "стадион",
)
EVENT_ADDRESS_PREFIXES = (
    "ул",
    "улиц",
    "пр",
    "просп",
    "пл",
    "пер",
    "наб",
    "бульв",
    "бул",
    "шос",
    "тракт",
    "дор",
    "мкр",
    "микр",
    "проезд",
    "пр-д",
    "б-р",
    "корп",
    "строен",
    "офис",
)
EVENT_ACTION_PREFIXES = (
    "собира",
    "встреч",
    "приглаш",
    "ждем",
    "ждём",
    "приход",
    "начал",
    "старт",
    "будет",
    "проход",
    "пройдет",
    "пройдёт",
    "состо",
    "откры",
    "ждет",
    "ждёт",
    "обсужд",
    "танцу",
    "игра",
    "мастер",
    "лекци",
    "семинар",
    "экскурс",
    "кинопоказ",
    "показ",
    "фестив",
    "ярмар",
    "праздн",
)
DATE_RANGE_RE = re.compile(r"\b(\d{1,2})[–-](\d{1,2})(?:[./](\d{1,2}))\b")
MONTH_NAME_RE = re.compile(r"\b(\d{1,2})\s+([а-яё.]+)\b", re.I)
TIME_RE = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")
TIME_H_RE = re.compile(r"\bв\s*([01]?\d|2[0-3])\s*(?:ч|час(?:а|ов)?)\b")
BARE_TIME_H_RE = re.compile(r"\b([01]?\d|2[0-3])\s*(?:ч|час(?:а|ов)?)\b")
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
    """Return True and list of matched keywords or pricing hints."""

    text_low = text.lower()
    price_matches = [m.group(0).strip() for m in PRICE_RE.finditer(text_low)]

    if VK_USE_PYMORPHY and MORPH:
        tokens = re.findall(r"\w+", text_low)
        matched: list[str] = []
        lemmas: list[str] = []
        for t in tokens:
            lemma = MORPH.parse(t)[0].normal_form
            lemmas.append(lemma)
            if lemma in KEYWORD_LEMMAS and lemma not in matched:
                matched.append(lemma)
        for idx, (first, second) in enumerate(zip(lemmas, lemmas[1:])):
            if first == "живой" and second == "звук":
                if "живой звук" not in matched:
                    matched.append("живой звук")
            if first == "пушкинский" and second == "карта":
                phrase = f"{tokens[idx]} {tokens[idx + 1]}"
                if phrase not in matched:
                    matched.append(phrase)
        for m in GROUP_CONTEXT_RE.finditer(text):
            group_match = m.group(0).lower()
            if group_match and group_match not in matched:
                matched.append(group_match)
        for m in GROUP_NAME_RE.finditer(text):
            group_match = m.group(0).lower()
            if group_match and group_match not in matched:
                matched.append(group_match)
        for hint in price_matches:
            if hint and hint not in matched:
                matched.append(hint)
        return bool(matched), matched

    matched = [m.group(0).lower().lstrip("#") for m in KEYWORD_RE.finditer(text_low)]
    for m in GROUP_CONTEXT_RE.finditer(text):
        group_match = m.group(0).lower()
        if group_match and group_match not in matched:
            matched.append(group_match)
    for m in GROUP_NAME_RE.finditer(text):
        group_match = m.group(0).lower()
        if group_match and group_match not in matched:
            matched.append(group_match)
    for hint in price_matches:
        if hint and hint not in matched:
            matched.append(hint)
    return bool(matched), matched


def detect_date(text: str) -> bool:
    """Heuristically detect a date or time mention in the text."""
    return any(
        COMPILED_DATE_PATTERNS[index].search(text)
        for index in DATE_PATTERN_STRONG_INDEXES
    )


def detect_historical_context(text: str) -> bool:
    """Return True if text mentions a pre-1995 year or historical toponyms."""

    text_low = text.lower()
    for match in HISTORICAL_YEAR_RE.findall(text_low):
        try:
            year = int(match)
        except ValueError:
            continue
        if year <= 1994:
            return True
    return any(name in text_low for name in HISTORICAL_TOPONYMS)


def normalize_phone_candidates(text: str) -> str:
    """Strip separators from phone-like sequences without touching valid dates."""

    date_intervals: list[tuple[int, int]] = []

    def _collect_intervals(pattern: re.Pattern[str]) -> None:
        for match in pattern.finditer(text):
            date_intervals.append((match.start(), match.end()))

    for date_pattern in (DATE_RANGE_RE, NUM_DATE_RE, MONTH_NAME_RE):
        _collect_intervals(date_pattern)

    phone_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in PHONE_CANDIDATE_RE.finditer(text)
    ]

    filtered_intervals: list[tuple[int, int]] = []
    for start, end in date_intervals:
        skip_interval = False
        for p_start, p_end in phone_spans:
            if p_start <= start and end <= p_end:
                if p_start < start or end < p_end:
                    skip_interval = True
                    break
                context_start = max(0, start - 20)
                context = text[context_start:start]
                if PHONE_CONTEXT_RE.search(context):
                    skip_interval = True
                    break
        if not skip_interval:
            filtered_intervals.append((start, end))

    date_intervals = sorted(filtered_intervals)

    def is_in_date_interval(index: int) -> bool:
        for interval_start, interval_end in date_intervals:
            if interval_end <= index:
                continue
            if interval_start > index:
                break
            return interval_start <= index < interval_end
        return False

    result: List[str] = []
    pos = 0
    separators = set(" +()\t\r\n.-–\u00a0\u202f")
    while True:
        match = PHONE_CANDIDATE_RE.search(text, pos)
        if not match:
            break
        start = match.start()
        result.append(text[pos:start])
        original = match.group(0)
        trimmed_end = 0
        for rel_idx, ch in enumerate(original):
            if ch.isdigit() or ch in separators:
                trimmed_end = rel_idx + 1
            else:
                break
        trimmed = original[:trimmed_end]
        if trimmed_end:
            normalized_chars: list[str] = []
            for rel_idx, ch in enumerate(trimmed):
                if ch.isdigit():
                    absolute_idx = start + rel_idx
                    if is_in_date_interval(absolute_idx):
                        normalized_chars.append(ch)
                    else:
                        normalized_chars.append("x")
                else:
                    normalized_chars.append(ch)
            result.append("".join(normalized_chars))
        else:
            result.append(trimmed)
        pos = start + trimmed_end
    result.append(text[pos:])
    return "".join(result)


def extract_event_ts_hint(
    text: str,
    default_time: str | None = None,
    *,
    tz: timezone | None = None,
    publish_ts: datetime | int | float | None = None,
    allow_past: bool = False,
) -> int | None:
    """Return Unix timestamp for the nearest future datetime mentioned in text."""
    tzinfo = tz or require_main_attr("LOCAL_TZ")

    if publish_ts is None:
        now = datetime.now(tzinfo)
    elif isinstance(publish_ts, datetime):
        if publish_ts.tzinfo is None:
            now = publish_ts.replace(tzinfo=tzinfo)
        else:
            now = publish_ts.astimezone(tzinfo)
    else:
        now = datetime.fromtimestamp(publish_ts, tzinfo)
    raw_text_low = text.lower()
    text_low = normalize_phone_candidates(raw_text_low)

    day = month = year = None
    m = None
    date_span: tuple[int, int] | None = None
    for candidate in NUM_DATE_RE.finditer(text_low):
        start = candidate.start()
        prev_idx = start - 1
        while prev_idx >= 0 and text_low[prev_idx].isspace():
            prev_idx -= 1
        if prev_idx >= 0 and text_low[prev_idx] in "./-":
            digit_count = 0
            check_idx = prev_idx - 1
            while check_idx >= 0 and text_low[check_idx].isdigit():
                digit_count += 1
                check_idx -= 1
            if digit_count >= 3:
                continue
        trailing_chars = " \t\r\n.;:!?()[]{}«»\"'—–-"
        trailing_idx = candidate.end()
        while trailing_idx < len(text_low) and text_low[trailing_idx] in trailing_chars:
            trailing_idx += 1
        if trailing_idx < len(text_low):
            raw_remainder = raw_text_low[trailing_idx:]
            trimmed_remainder = raw_remainder.lstrip(trailing_chars)
            if trimmed_remainder and trimmed_remainder[0].isdigit():
                continue
        remainder = text_low[trailing_idx:] if trailing_idx < len(text_low) else ""

        if PHONE_LIKE_RE.match(candidate.group(0)):
            context_start = max(0, start - 30)
            context_end = min(len(text_low), candidate.end() + 10)
            context_slice = text_low[context_start:context_end]
            skip_candidate = False
            has_event_tail = False
            next_alpha_word = None
            following_is_phone_tail = False
            skip_due_to_action_tail = False
            skip_due_to_location_tail = False
            if trailing_idx < len(text_low):
                word_match = re.match(r"[a-zа-яё]+", remainder)
                if word_match:
                    next_alpha_word = word_match.group(0)
                    if PHONE_CONTEXT_RE.match(next_alpha_word):
                        following_is_phone_tail = True
                if PHONE_CONTEXT_RE.match(remainder):
                    following_is_phone_tail = True
                if not following_is_phone_tail:
                    def _tail_has_datetime(segment: str) -> bool:
                        return bool(
                            NUM_DATE_RE.search(segment)
                            or DATE_RANGE_RE.search(segment)
                            or TIME_RE.search(segment)
                            or TIME_H_RE.search(segment)
                            or TIME_RANGE_RE.search(segment)
                            or MONTH_NAME_RE.search(segment)
                        )

                    if TIME_RE.match(remainder) or TIME_H_RE.match(remainder) or TIME_RANGE_RE.match(remainder):
                        has_event_tail = True
                    elif DOW_RE.match(remainder):
                        has_event_tail = True
                    else:
                        if remainder.startswith("по адресу"):
                            after_location = remainder[len("по адресу") :]
                            after_location = after_location.lstrip(
                                " \t\r\n.;:!?()[]{}«»\"'—–-"
                            )
                            if _tail_has_datetime(after_location):
                                skip_due_to_location_tail = True
                        elif next_alpha_word and next_alpha_word.startswith(
                            EVENT_ADDRESS_PREFIXES
                        ):
                            address_tail = remainder[len(next_alpha_word) :]
                            address_tail = address_tail.lstrip(
                                " \t\r\n.;:!?()[]{}«»\"'—–-"
                            )
                            if _tail_has_datetime(address_tail):
                                has_event_tail = True
                                skip_due_to_location_tail = True
                        else:
                            loc_match = re.match(r"(?:в|на)\s+([a-zа-яё.]+)", remainder)
                            if loc_match:
                                loc_word = loc_match.group(1).strip(".")
                                if loc_word.startswith(EVENT_LOCATION_PREFIXES):
                                    after_location = remainder[loc_match.end() :]
                                    after_location = after_location.lstrip(
                                        " \t\r\n.;:!?()[]{}«»\"'—–-"
                                    )
                                    if _tail_has_datetime(after_location):
                                        skip_due_to_location_tail = True
                        if (
                            not has_event_tail
                            and next_alpha_word
                            and next_alpha_word.startswith(EVENT_ACTION_PREFIXES)
                        ):
                            action_tail = remainder[len(next_alpha_word) :]
                            action_tail = action_tail.lstrip(
                                " \t\r\n.;:!?()[]{}«»\"'—–-"
                            )
                            if action_tail:
                                has_action_tail_datetime = bool(
                                    NUM_DATE_RE.search(action_tail)
                                    or DATE_RANGE_RE.search(action_tail)
                                    or TIME_RE.search(action_tail)
                                    or TIME_H_RE.search(action_tail)
                                    or TIME_RANGE_RE.search(action_tail)
                                    or MONTH_NAME_RE.search(action_tail)
                                )
                                if has_action_tail_datetime:
                                    has_event_tail = True
                                    skip_due_to_action_tail = True
            if skip_due_to_action_tail:
                continue
            if skip_due_to_location_tail:
                continue
            if not has_event_tail:
                for phone_match in PHONE_CONTEXT_RE.finditer(context_slice):
                    match_end = context_start + phone_match.end()
                    if match_end <= start:
                        intervening = text_low[match_end:start]
                        if "\n" in intervening or "\r" in intervening:
                            continue
                        trimmed = intervening.strip()
                        if not trimmed:
                            skip_candidate = True
                            break
                        if "," in trimmed:
                            break
                        if re.search(r"[a-zа-яё]", trimmed):
                            break
                        if (
                            re.search(r"\d", trimmed)
                            and re.search(r"[a-zа-яё]", remainder)
                            and not re.search(r"\d", remainder)
                        ):
                            skip_candidate = True
                            continue
                        compact = trimmed.replace(" ", "")
                        compact = re.sub(r"^[.,:;-–—]+", "", compact)
                        if not compact or re.fullmatch(r"[\d()+\-–—]*", compact):
                            skip_candidate = True
                            break
            if skip_candidate:
                continue
        m = candidate
        date_span = candidate.span()
        break

    if m:
        day, month = int(m.group(1)), int(m.group(2))
        if m.group(3):
            y = m.group(3)
            year = int("20" + y if len(y) == 2 else y)
        if date_span is None:
            date_span = m.span()
    else:
        m = DATE_RANGE_RE.search(text_low)
        if m:
            day = int(m.group(1))
            month = int(m.group(3))
            date_span = m.span()
        else:
            m = MONTH_NAME_RE.search(text_low)
            if m:
                day = int(m.group(1))
                mon_word = m.group(2).rstrip(".")
                month = MONTHS_RU.get(mon_word)
                y = re.search(r"\b20\d{2}\b", text_low[m.end():])
                if y:
                    year = int(y.group(0))
                if month is not None:
                    date_span = m.span()

    if day is None or month is None:
        if "сегодня" in text_low:
            dt = now
            idx = text_low.find("сегодня")
            if idx != -1:
                date_span = (idx, idx + len("сегодня"))
        elif "завтра" in text_low:
            dt = now + timedelta(days=1)
            idx = text_low.find("завтра")
            if idx != -1:
                date_span = (idx, idx + len("завтра"))
        elif "послезавтра" in text_low:
            dt = now + timedelta(days=2)
            idx = text_low.find("послезавтра")
            if idx != -1:
                date_span = (idx, idx + len("послезавтра"))
        else:
            dow_matches = list(DOW_RE.finditer(text_low))
            dow_m = None
            for candidate in dow_matches:
                context_start = max(0, candidate.start() - 40)
                context_end = min(len(text_low), candidate.end() + 40)
                context_slice = text_low[context_start:context_end]
                if PAST_EVENT_RE.search(context_slice):
                    continue
                dow_m = candidate
                break
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
                date_span = (dow_m.start(), dow_m.end())
            elif dow_matches:
                return None
            elif (weekend_m := WEEKEND_RE.search(text_low)):
                days_ahead = (5 - now.weekday()) % 7
                dt = now + timedelta(days=days_ahead)
                date_span = (weekend_m.start(), weekend_m.end())
            else:
                return None
    else:
        explicit_year = year is not None
        year = year or now.year
        try:
            dt = datetime(year, month, day, tzinfo=tzinfo)
        except ValueError:
            return None
        if dt < now:
            skip_year_rollover = explicit_year
            if not explicit_year and now - dt <= RECENT_PAST_THRESHOLD:
                skip_year_rollover = True
            if not skip_year_rollover:
                try:
                    dt = datetime(year + 1, month, day, tzinfo=tzinfo)
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
            else:
                bare_th = None
                bare_hour_rejected = False
                if date_span is not None:
                    allowed_connector_words = {
                        "в",
                        "к",
                        "ровно",
                        "начало",
                        "начала",
                        "начнем",
                        "начнём",
                        "начнется",
                        "начнётся",
                        "начинаем",
                        "старт",
                        "стартуем",
                        "стартует",
                    }
                    duration_hint_prefixes = ("жив", "длит", "продолж", "програм")

                    for candidate in BARE_TIME_H_RE.finditer(text_low):
                        if candidate.start() < date_span[1]:
                            continue
                        between = text_low[date_span[1] : candidate.start()]
                        if re.search(r"[.!?]", between):
                            continue
                        between_stripped = between.strip()
                        reject_candidate = False
                        if between_stripped:
                            normalized_between = between_stripped
                            normalized_between = re.sub(r"[—–-]", " ", normalized_between)
                            normalized_between = re.sub(r"[,;:]", " ", normalized_between)
                            normalized_between = re.sub(r"\s+", " ", normalized_between).strip()
                            if normalized_between:
                                tokens = normalized_between.split(" ")
                                if any(token not in allowed_connector_words for token in tokens):
                                    reject_candidate = True
                        trailing_segment = text_low[candidate.end() :]
                        trailing_segment = trailing_segment.lstrip(
                            " \t\r\n,.;:!?()[]{}«»\"'—–-"
                        )
                        if trailing_segment:
                            next_word_match = re.match(r"[a-zа-яё]+", trailing_segment)
                            if next_word_match and next_word_match.group(0).startswith(
                                duration_hint_prefixes
                            ):
                                reject_candidate = True
                        if reject_candidate:
                            bare_hour_rejected = True
                            continue
                        bare_th = candidate
                        break
                if bare_th:
                    hour = int(bare_th.group(1))
                    minute = 0
                elif bare_hour_rejected:
                    return None
                elif default_time:
                    try:
                        hour, minute = map(int, default_time.split(":"))
                    except Exception:
                        hour = minute = 0
                else:
                    hour = minute = 0

    dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if dt < now and not allow_past:
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
    poster_media: list[PosterMedia] = field(default_factory=list)
    poster_summary: str | None = None
    ocr_tokens_spent: int = 0
    ocr_tokens_remaining: int | None = None
    ocr_limit_notice: str | None = None


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


async def _download_photo_media(urls: Sequence[str]) -> list[tuple[bytes, str]]:
    if not urls:
        return []
    import sys

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if main_mod is None:  # pragma: no cover - defensive
        raise RuntimeError("main module not found")
    session = main_mod.get_http_session()
    semaphore = main_mod.HTTP_SEMAPHORE
    timeout = main_mod.HTTP_TIMEOUT
    max_size = main_mod.MAX_DOWNLOAD_SIZE
    ensure_jpeg = main_mod.ensure_jpeg
    detect_image_type = getattr(main_mod, "detect_image_type", None)
    if detect_image_type is None:  # pragma: no cover - defensive
        raise RuntimeError("detect_image_type not found")
    validate_jpeg_markers = getattr(main_mod, "validate_jpeg_markers", None)
    if validate_jpeg_markers is None:  # pragma: no cover - defensive
        raise RuntimeError("validate_jpeg_markers not found")
    limit = getattr(main_mod, "MAX_ALBUM_IMAGES", 3)
    results: list[tuple[bytes, str]] = []

    request_headers = getattr(main_mod, "VK_PHOTO_FETCH_HEADERS", None)
    if request_headers is None:
        request_headers = {
            "User-Agent": getattr(
                main_mod,
                "VK_BROWSER_USER_AGENT",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 "
                "Safari/537.36",
            ),
            "Accept": getattr(
                main_mod,
                "VK_BROWSER_ACCEPT",
                "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            ),
            "Referer": getattr(main_mod, "VK_BROWSER_REFERER", "https://vk.com/"),
            "Sec-Fetch-Dest": getattr(
                main_mod, "VK_BROWSER_SEC_FETCH_DEST", "image"
            ),
            "Sec-Fetch-Mode": getattr(
                main_mod, "VK_BROWSER_SEC_FETCH_MODE", "no-cors"
            ),
            "Sec-Fetch-Site": getattr(
                main_mod, "VK_BROWSER_SEC_FETCH_SITE", "same-origin"
            ),
        }
    else:
        request_headers = dict(request_headers)

    for idx, url in enumerate(urls[:limit]):

        async def _fetch() -> tuple[bytes, str | None, str | None]:
            async with semaphore:
                async with session.get(url, headers=request_headers) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type")
                    content_length = resp.headers.get("Content-Length")
                    if resp.content_length and resp.content_length > max_size:
                        raise ValueError("file too large")
                    buf = bytearray()
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        buf.extend(chunk)
                        if len(buf) > max_size:
                            raise ValueError("file too large")
                    return bytes(buf), content_type, content_length

        size = None
        content_type: str | None = None
        content_length: str | None = None
        try:
            data, content_type, content_length = await asyncio.wait_for(
                _fetch(), timeout
            )
            size = len(data)
            if size > max_size:
                raise ValueError("file too large")
            if content_length:
                try:
                    expected_size = int(content_length)
                except ValueError as exc:
                    raise ValueError("invalid Content-Length header") from exc
                if expected_size != size:
                    raise ValueError("content-length mismatch")
            orig_subtype = detect_image_type(data)
            if orig_subtype == "jpeg":
                validate_jpeg_markers(data)
            data, name = ensure_jpeg(data, f"vk_poster_{idx + 1}.jpg")
            subtype = detect_image_type(data)
            if subtype == "jpeg":
                validate_jpeg_markers(data)
        except Exception as exc:  # pragma: no cover - network dependent
            logging.warning(
                "vk.download_photo_failed url=%s size=%s content_type=%s "
                "content_length=%s error=%s",
                url,
                size if size is not None else "unknown",
                content_type or "unknown",
                content_length or "unknown",
                exc,
            )
            continue
        logging.info(
            "vk.photo_media processed idx=%s url=%s size=%d subtype=%s "
            "filename=%s content_type=%s content_length=%s",
            idx,
            url,
            size if size is not None else 0,
            subtype or "unknown",
            name,
            content_type or "unknown",
            content_length or "unknown",
        )
        results.append((data, name))
    return results


async def build_event_drafts_from_vk(
    text: str,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    poster_media: Sequence[PosterMedia] | None = None,
    ocr_tokens_spent: int = 0,
    ocr_tokens_remaining: int | None = None,
) -> list[EventDraft]:
    """Return normalised event drafts extracted from a VK post.

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
    parse_event_via_4o = require_main_attr("parse_event_via_4o")

    fallback_ticket_link = (
        default_ticket_link.strip()
        if isinstance(default_ticket_link, str)
        else default_ticket_link
    )
    if isinstance(fallback_ticket_link, str) and not fallback_ticket_link:
        fallback_ticket_link = None

    llm_text = text
    if operator_extra:
        llm_text = f"{llm_text}\n{operator_extra}"
    if default_time:
        llm_text = f"{llm_text}\nЕсли время не указано, предположи начало в {default_time}."
    if fallback_ticket_link:
        llm_text = (
            f"{llm_text}\n"
            f"Если в посте нет ссылки на билеты или регистрацию, используй {fallback_ticket_link} как ссылку по умолчанию. "
            "Не заменяй ссылки, которые уже указаны."
        )
    if festival_hint:
        llm_text = (
            f"{llm_text}\n"
            "Оператор подтверждает, что пост описывает фестиваль. "
            "Сопоставь с существующими фестивалями (JSON ниже) или создай новый."
        )

    poster_items = list(poster_media or [])
    poster_texts = collect_poster_texts(poster_items)
    poster_summary = build_poster_summary(poster_items)

    extra: dict[str, str] = {}
    if source_name:
        # ``parse_event_via_4o`` accepts ``channel_title`` for context
        extra["channel_title"] = source_name

    parse_kwargs: dict[str, Any] = {}
    if poster_texts:
        parse_kwargs["poster_texts"] = poster_texts
    if poster_summary:
        parse_kwargs["poster_summary"] = poster_summary
    if festival_alias_pairs:
        parse_kwargs["festival_alias_pairs"] = festival_alias_pairs

    parsed = await parse_event_via_4o(
        llm_text, festival_names=festival_names, **extra, **parse_kwargs
    )
    festival_payload = getattr(parse_event_via_4o, "_festival", None)
    parsed_events = list(parsed) if parsed else []
    if not parsed_events and not festival_payload:
        raise RuntimeError("LLM returned no event")

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

    drafts: list[EventDraft] = []
    for data in parsed_events:
        ticket_price_min = clean_int(data.get("ticket_price_min"))
        ticket_price_max = clean_int(data.get("ticket_price_max"))
        ticket_link = clean_str(data.get("ticket_link"))
        links: list[str] | None
        if ticket_link:
            links = [ticket_link]
        elif fallback_ticket_link:
            links = [fallback_ticket_link]
        else:
            links = None
        drafts.append(
            EventDraft(
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
                poster_media=poster_items,
                poster_summary=poster_summary,
                ocr_tokens_spent=ocr_tokens_spent,
                ocr_tokens_remaining=ocr_tokens_remaining,
            )
        )

    combined_lower = (combined_text or "").lower()
    paid_keywords = ("руб", "₽", "платн", "стоимост", "взнос", "донат")
    has_paid_keywords = any(keyword in combined_lower for keyword in paid_keywords)

    for draft in drafts:
        venue_text = (draft.venue or "").lower()
        address_text = (draft.location_address or "").lower()
        if "библиотек" not in venue_text and "библиотек" not in address_text:
            continue
        if draft.ticket_price_min is not None or draft.ticket_price_max is not None:
            continue
        if has_paid_keywords:
            continue
        if not draft.is_free:
            draft.is_free = True

    return drafts


async def build_event_payload_from_vk(
    text: str,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    festival_names: list[str] | None = None,
    poster_media: Sequence[PosterMedia] | None = None,
    ocr_tokens_spent: int = 0,
    ocr_tokens_remaining: int | None = None,
) -> EventDraft:
    drafts = await build_event_drafts_from_vk(
        text,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        default_ticket_link=default_ticket_link,
        operator_extra=operator_extra,
        festival_names=festival_names,
        poster_media=poster_media,
        ocr_tokens_spent=ocr_tokens_spent,
        ocr_tokens_remaining=ocr_tokens_remaining,
    )
    if not drafts:
        raise RuntimeError("LLM returned no event")
    return drafts[0]


async def build_event_drafts(
    text: str,
    *,
    photos: Sequence[str] | None = None,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: list[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    db: Database,
) -> list[EventDraft]:
    """Download posters, run OCR and return event drafts for a VK post."""
    photo_bytes = await _download_photo_media(photos or [])
    poster_items: list[PosterMedia] = []
    ocr_tokens_spent = 0
    ocr_tokens_remaining: int | None = None
    ocr_limit_notice: str | None = None
    hash_to_indices: dict[str, list[int]] | None = None
    if photo_bytes:
        hash_to_indices = {}
        for idx, (payload, _name) in enumerate(photo_bytes):
            digest = hashlib.sha256(payload).hexdigest()
            hash_to_indices.setdefault(digest, []).append(idx)
        poster_items, catbox_msg = await process_media(
            photo_bytes, need_catbox=True, need_ocr=False
        )
        ocr_source = source_name or "vk"
        ocr_log_context = {"event_id": None, "source": ocr_source}
        ocr_results: list[poster_ocr.PosterOcrCache] = []
        try:
            (
                ocr_results,
                ocr_tokens_spent,
                ocr_tokens_remaining,
            ) = await poster_ocr.recognize_posters(
                db, photo_bytes, log_context=ocr_log_context
            )
        except poster_ocr.PosterOcrLimitExceededError as exc:
            logging.warning(
                "vk.build_event_draft OCR skipped: %s",
                exc,
                extra=ocr_log_context,
            )
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
        logging.info(
            "vk.build_event_draft posters=%d catbox=%s",
            len(poster_items),
            catbox_msg or "",
        )
    else:
        ocr_source = source_name or "vk"
        ocr_log_context = {"event_id": None, "source": ocr_source}
        _, _, ocr_tokens_remaining = await poster_ocr.recognize_posters(
            db, [], log_context=ocr_log_context
        )
    drafts = await build_event_drafts_from_vk(
        text,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        default_ticket_link=default_ticket_link,
        operator_extra=operator_extra,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs,
        festival_hint=festival_hint,
        poster_media=poster_items,
        ocr_tokens_spent=ocr_tokens_spent,
        ocr_tokens_remaining=ocr_tokens_remaining,
    )
    for draft in drafts:
        draft.ocr_limit_notice = ocr_limit_notice
    return drafts


async def build_event_draft(
    text: str,
    *,
    photos: Sequence[str] | None = None,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: list[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    db: Database,
) -> EventDraft:
    drafts = await build_event_drafts(
        text,
        photos=photos,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        default_ticket_link=default_ticket_link,
        operator_extra=operator_extra,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs,
        festival_hint=festival_hint,
        db=db,
    )
    if not drafts:
        raise RuntimeError("LLM returned no event")
    return drafts[0]


_DASH_CHAR_PATTERN = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_MONTH_NAME_PATTERN = "|".join(sorted(MONTHS_RU.keys(), key=len, reverse=True))
_TEXT_RANGE_TWO_MONTHS_RE = re.compile(
    rf"^\s*(?P<start_day>\d{{1,2}})\s*(?P<start_month>{_MONTH_NAME_PATTERN})\s*-\s*(?P<end_day>\d{{1,2}})\s*(?P<end_month>{_MONTH_NAME_PATTERN})\s*$",
    re.IGNORECASE,
)
_TEXT_RANGE_SAME_MONTH_RE = re.compile(
    rf"^\s*(?P<start_day>\d{{1,2}})\s*-\s*(?P<end_day>\d{{1,2}})\s*(?P<month>{_MONTH_NAME_PATTERN})\s*$",
    re.IGNORECASE,
)
_TEXT_SINGLE_RE = re.compile(
    rf"^\s*(?P<day>\d{{1,2}})\s*(?P<month>{_MONTH_NAME_PATTERN})\s*$",
    re.IGNORECASE,
)


def _month_from_token(token: str) -> int | None:
    lookup = token.strip().strip(".,").casefold()
    return MONTHS_RU.get(lookup)


def _safe_construct_date(year: int, month: int, day: int) -> date | None:
    if not (1 <= month <= 12):
        return None
    if day < 1:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        try:
            last_day = calendar.monthrange(year, month)[1]
        except Exception:
            return None
        day = min(day, last_day)
        try:
            return date(year, month, day)
        except ValueError:
            return None


def _parse_single_date_token(token: str, target_year: int) -> date | None:
    token = token.strip()
    if not token:
        return None

    token = token.strip(".,")
    dot_match = re.match(r"^(?P<day>\d{1,2})\.(?P<month>\d{1,2})$", token)
    if dot_match:
        day = int(dot_match.group("day"))
        month = int(dot_match.group("month"))
        return _safe_construct_date(target_year, month, day)

    legacy_match = re.match(r"^(?P<month>\d{1,2})-(?P<day>\d{1,2})$", token)
    if legacy_match:
        month = int(legacy_match.group("month"))
        day = int(legacy_match.group("day"))
        return _safe_construct_date(target_year, month, day)

    text_match = _TEXT_SINGLE_RE.match(token)
    if text_match:
        month = _month_from_token(text_match.group("month"))
        day = int(text_match.group("day"))
        if month is None:
            return None
        return _safe_construct_date(target_year, month, day)

    return None


def _holiday_date_range(record: Any, target_year: int) -> tuple[str | None, str | None]:
    raw = (record.date or "").strip()
    if not raw:
        return None, None

    normalized = _DASH_CHAR_PATTERN.sub("-", raw)
    normalized = re.sub(r"\s+", " ", normalized.strip())
    normalized = normalized.strip(".,")
    if not normalized:
        return None, None

    if ".." in normalized:
        parts = [part.strip() for part in normalized.split("..") if part.strip()]
        if not parts:
            return None, None
        start = _parse_single_date_token(parts[0], target_year)
        end_token = parts[-1]
        end = _parse_single_date_token(end_token, target_year)
    else:
        if re.match(r"^\d{1,2}-\d{1,2}$", normalized):
            start = _parse_single_date_token(normalized, target_year)
            end = start
        else:
            dot_range = re.match(
                r"^(?P<start_day>\d{1,2})\.(?P<start_month>\d{1,2})\s*-\s*(?P<end_day>\d{1,2})\.(?P<end_month>\d{1,2})$",
                normalized,
            )
            partial_numeric = re.match(
                r"^(?P<start_day>\d{1,2})\s*-\s*(?P<end_day>\d{1,2})\.(?P<month>\d{1,2})$",
                normalized,
            )
            text_range = _TEXT_RANGE_TWO_MONTHS_RE.match(normalized)
            partial_text = re.match(
                r"^(?P<start_day>\d{1,2})\s*-\s*(?P<end_day>\d{1,2})\s+(?P<month>[\wё]+)\.?$",
                normalized,
                flags=re.IGNORECASE,
            )
            text_same_month = _TEXT_RANGE_SAME_MONTH_RE.match(normalized)

            if dot_range:
                start = _safe_construct_date(
                    target_year,
                    int(dot_range.group("start_month")),
                    int(dot_range.group("start_day")),
                )
                end = _safe_construct_date(
                    target_year,
                    int(dot_range.group("end_month")),
                    int(dot_range.group("end_day")),
                )
            elif partial_numeric:
                month = int(partial_numeric.group("month"))
                start = _safe_construct_date(
                    target_year,
                    month,
                    int(partial_numeric.group("start_day")),
                )
                end = _safe_construct_date(
                    target_year,
                    month,
                    int(partial_numeric.group("end_day")),
                )
            elif text_range:
                start_month = _month_from_token(text_range.group("start_month"))
                end_month = _month_from_token(text_range.group("end_month"))
                start = (
                    _safe_construct_date(
                        target_year,
                        start_month,
                        int(text_range.group("start_day")),
                    )
                    if start_month is not None
                    else None
                )
                end = (
                    _safe_construct_date(
                        target_year,
                        end_month,
                        int(text_range.group("end_day")),
                    )
                    if end_month is not None
                    else None
                )
            elif partial_text:
                month = _month_from_token(partial_text.group("month"))
                if month is not None:
                    start = _safe_construct_date(
                        target_year,
                        month,
                        int(partial_text.group("start_day")),
                    )
                    end = _safe_construct_date(
                        target_year,
                        month,
                        int(partial_text.group("end_day")),
                    )
                else:
                    start = None
                    end = None
            elif text_same_month:
                month = _month_from_token(text_same_month.group("month"))
                if month is not None:
                    start = _safe_construct_date(
                        target_year, month, int(text_same_month.group("start_day"))
                    )
                    end = _safe_construct_date(
                        target_year, month, int(text_same_month.group("end_day"))
                    )
                else:
                    start = None
                    end = None
            else:
                parts = [part.strip() for part in re.split(r"\s*-\s*", normalized) if part.strip()]
                if len(parts) >= 2:
                    start = _parse_single_date_token(parts[0], target_year)
                    end = _parse_single_date_token(parts[-1], target_year)
                else:
                    start = _parse_single_date_token(normalized, target_year)
                    end = start

    if start and end and end < start:
        rollover = _safe_construct_date(end.year + 1, end.month, end.day)
        end = rollover if rollover else end

    start_iso = start.isoformat() if start else None
    end_iso = end.isoformat() if end else None
    return start_iso, end_iso


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except Exception:
        return None


def _event_date_range(
    event_date: str | None, event_end_date: str | None
) -> tuple[date | None, date | None]:
    if not event_date:
        return None, None

    if ".." in event_date:
        parts = [part.strip() for part in event_date.split("..") if part.strip()]
        if not parts:
            return None, None
        start = _parse_iso_date(parts[0])
        end = _parse_iso_date(parts[-1])
    else:
        start = _parse_iso_date(event_date.strip())
        if event_end_date:
            end = _parse_iso_date(event_end_date.strip())
        else:
            end = start

    if start and end and end < start:
        start, end = end, start

    return start, end


def _event_date_matches_holiday(
    record: Any,
    event_date: str | None,
    event_end_date: str | None,
    tolerance_days: int | None,
) -> bool:
    if record is None:
        return False

    start, end = _event_date_range(event_date, event_end_date)
    if start is None and end is None:
        return False

    tolerance = tolerance_days if tolerance_days is not None else 0
    if tolerance < 0:
        tolerance = 0

    event_start = start or end
    event_end = end or start or event_start
    if event_start is None or event_end is None:
        return False
    if event_end < event_start:
        event_start, event_end = event_end, event_start

    years: set[int] = set()
    years.add(event_start.year)
    years.add(event_end.year)
    expanded_years: set[int] = set()
    for year in years:
        expanded_years.add(year)
        expanded_years.add(year - 1)
        expanded_years.add(year + 1)

    tolerance_delta = timedelta(days=tolerance)

    for year in sorted(expanded_years):
        start_iso, end_iso = _holiday_date_range(record, year)
        if not start_iso and not end_iso:
            continue
        holiday_start = _parse_iso_date(start_iso)
        holiday_end = _parse_iso_date(end_iso)
        if holiday_start is None and holiday_end is None:
            continue
        if holiday_start is None:
            holiday_start = holiday_end
        if holiday_end is None:
            holiday_end = holiday_start
        if holiday_start is None or holiday_end is None:
            continue
        if holiday_end < holiday_start:
            holiday_start, holiday_end = holiday_end, holiday_start

        window_start = holiday_start - tolerance_delta
        window_end = holiday_end + tolerance_delta
        if event_end >= window_start and event_start <= window_end:
            return True

    return False


async def persist_event_and_pages(
    draft: EventDraft,
    photos: list[str],
    db: Database,
    source_post_url: str | None = None,
    *,
    holiday_tolerance_days: int | None = None,
) -> PersistResult:
    """Store a drafted event and produce all public artefacts.

    The helper encapsulates the legacy import pipeline used by the bot.  It
    persists the event to the database, uploads images to Catbox and creates the
    Telegraph page, generates an ICS file and posts it to the asset channel.
    Links to these artefacts are returned in :class:`PersistResult`.
    """
    from datetime import datetime
    from models import Event, Festival
    from sqlalchemy import select
    import sys

    main_mod = sys.modules.get("main") or sys.modules.get("__main__")
    if main_mod is None:  # pragma: no cover - defensive
        raise RuntimeError("main module not found")
    upsert_event = main_mod.upsert_event
    upsert_event_posters = main_mod.upsert_event_posters
    assign_event_topics = main_mod.assign_event_topics
    schedule_event_update_tasks = main_mod.schedule_event_update_tasks
    rebuild_fest_nav_if_changed = main_mod.rebuild_fest_nav_if_changed
    ensure_festival = main_mod.ensure_festival
    get_holiday_record = getattr(main_mod, "get_holiday_record", None)
    sync_festival_page = getattr(main_mod, "sync_festival_page", None)

    poster_urls = [m.catbox_url for m in draft.poster_media if m.catbox_url]
    photo_urls = poster_urls or list(photos or [])

    event = Event(
        title=draft.title,
        description=(draft.description or ""),
        festival=(draft.festival or None),
        date=draft.date or datetime.now(timezone.utc).date().isoformat(),
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
        photo_urls=photo_urls,
        photo_count=len(photo_urls),
        source_post_url=source_post_url,
    )

    topics, text_length, error_text, manual_flag = await assign_event_topics(event)

    async with db.get_session() as session:
        saved, _ = await upsert_event(session, event)
        await upsert_event_posters(session, saved.id, draft.poster_media)
    if manual_flag:
        logging.info(
            "event_topics_classify eid=%s text_len=%d topics=%s manual=True",
            saved.id,
            text_length,
            list(saved.topics or []),
        )
    elif error_text:
        logging.info(
            "event_topics_classify eid=%s text_len=%d topics=%s error=%s",
            saved.id,
            text_length,
            list(saved.topics or []),
            error_text,
        )
    else:
        logging.info(
            "event_topics_classify eid=%s text_len=%d topics=%s",
            saved.id,
            text_length,
            list(saved.topics or []),
        )
    async with db.get_session() as session:
        saved = await session.get(Event, saved.id)
    logging.info(
        "persist_event_and_pages: source_post_url=%s", saved.source_post_url
    )

    holiday_record = (
        get_holiday_record(saved.festival) if callable(get_holiday_record) else None
    )
    tolerance_value = holiday_tolerance_days
    if tolerance_value is None and holiday_record is not None:
        tolerance_value = getattr(holiday_record, "tolerance_days", None)

    if holiday_record and _event_date_matches_holiday(
        holiday_record, saved.date, saved.end_date, tolerance_value
    ):
        canonical_name = holiday_record.canonical_name
        start_iso, end_iso = _holiday_date_range(holiday_record, date.today().year)
        ensure_kwargs: dict[str, Any] = {}
        if holiday_record.description:
            ensure_kwargs["description"] = holiday_record.description
            ensure_kwargs["source_text"] = holiday_record.description
        if start_iso:
            ensure_kwargs["start_date"] = start_iso
        if end_iso:
            ensure_kwargs["end_date"] = end_iso
        aliases_payload = [
            alias for alias in getattr(holiday_record, "normalized_aliases", ()) if alias
        ]
        if aliases_payload:
            ensure_kwargs["aliases"] = aliases_payload
        fest_obj, fest_created, fest_updated = await ensure_festival(
            db,
            canonical_name,
            **ensure_kwargs,
        )
        if saved.festival != canonical_name:
            async with db.get_session() as session:
                event_obj = await session.get(Event, saved.id)
                if event_obj:
                    event_obj.festival = canonical_name
                    session.add(event_obj)
                    await session.commit()
                    saved = event_obj
        if (fest_created or fest_updated) and callable(sync_festival_page):
            asyncio.create_task(sync_festival_page(db, canonical_name))

    nav_update_needed = False
    if saved.festival:
        parts = [p.strip() for p in (saved.date or "").split("..") if p.strip()]
        start_str = parts[0] if parts else None
        end_str = parts[-1] if len(parts) > 1 else None
        if not end_str:
            end_str = saved.end_date or start_str
        if start_str or end_str:
            async with db.get_session() as session:
                res = await session.execute(
                    select(Festival).where(Festival.name == saved.festival)
                )
                festival = res.scalar_one_or_none()
                if festival is not None:
                    changed = False
                    if start_str and festival.start_date is None:
                        festival.start_date = start_str
                        changed = True
                    if end_str and festival.end_date is None:
                        festival.end_date = end_str
                        changed = True
                    if changed:
                        session.add(festival)
                        await session.commit()
                        nav_update_needed = True
    if nav_update_needed:
        await rebuild_fest_nav_if_changed(db)
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
) -> list[PersistResult]:
    """Process VK post text into an event and track processing time."""
    start = time.perf_counter()
    from sqlalchemy import select
    from models import Festival

    async with db.get_session() as session:
        res_f = await session.execute(select(Festival.name))
        festival_names = [row[0] for row in res_f.fetchall()]
    drafts = await build_event_drafts(
        text,
        photos=photos or [],
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        operator_extra=operator_extra,
        festival_names=festival_names,
        festival_hint=False,
        db=db,
    )
    results: list[PersistResult] = []
    for draft in drafts:
        results.append(
            await persist_event_and_pages(draft, photos or [], db)
        )
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
    return results


async def crawl_once(
    db,
    *,
    broadcast: bool = False,
    bot: Any | None = None,
    force_backfill: bool = False,
    backfill_days: int | None = None,
) -> dict[str, Any]:
    """Crawl configured VK groups once and enqueue matching posts.

    The function scans groups listed in ``vk_source`` and uses cursors from
    ``vk_crawl_cursor`` to fetch only new posts. Posts containing event
    keywords and a date mention are inserted into ``vk_inbox`` with status
    ``pending``. Basic statistics are returned for reporting purposes.

    If ``broadcast`` is True and ``bot`` is supplied, a crawl summary is sent
    to the admin chat specified by ``ADMIN_CHAT_ID`` environment variable.
    """

    vk_wall_since = require_main_attr(
        "vk_wall_since"
    )  # imported lazily to avoid circular import
    get_supabase_client = require_main_attr("get_supabase_client")
    get_tz_offset = require_main_attr("get_tz_offset")
    mark_vk_import_result = require_main_attr("mark_vk_import_result")
    VkImportRejectCode = require_main_attr("VkImportRejectCode")
    await get_tz_offset(db)
    local_tz = require_main_attr("LOCAL_TZ")
    exporter = SBExporter(get_supabase_client)

    def _record_rejection(
        group_id: int,
        post_id: int,
        url: str,
        code: Any,
        note: str | None = None,
    ) -> None:
        try:
            code_value = getattr(code, "value", code)
            mark_vk_import_result(
                group_id=group_id,
                post_id=post_id,
                url=url,
                outcome="rejected",
                event_id=None,
                reject_code=str(code_value),
                reject_note=note,
            )
        except Exception:
            logging.exception("vk_import_result.supabase_failed")

    start = time.perf_counter()
    override_backfill_days = (
        max(1, min(backfill_days, VK_CRAWL_BACKFILL_OVERRIDE_MAX_DAYS))
        if backfill_days is not None
        else None
    )

    stats = {
        "groups_checked": 0,
        "posts_scanned": 0,
        "matches": 0,
        "duplicates": 0,
        "added": 0,
        "errors": 0,
        "inbox_total": 0,
        "queue": {},
        "safety_cap_hits": 0,
        "deep_backfill_triggers": 0,
        "forced_backfill": force_backfill,
        "backfill_days_used": (
            override_backfill_days
            if override_backfill_days is not None
            else (VK_CRAWL_BACKFILL_DAYS if force_backfill else None)
        ),
        "backfill_days_requested": backfill_days if force_backfill else None,
    }

    async with db.raw_conn() as conn:
        cutoff = int(time.time()) + 2 * 3600
        await conn.execute(
            "UPDATE vk_inbox SET status='rejected' WHERE status IN ('pending','skipped') AND (event_ts_hint IS NULL OR event_ts_hint < ?)",
            (cutoff,),
        )
        cur = await conn.execute(
            """
            SELECT group_id, screen_name, name, location, default_time, default_ticket_link
            FROM vk_source
            """
        )
        groups = [
            {
                "group_id": row[0],
                "screen_name": row[1],
                "name": row[2],
                "location": row[3],
                "default_time": row[4],
                "default_ticket_link": row[5],
            }
            for row in await cur.fetchall()
        ]
        await conn.commit()

    random.shuffle(groups)
    logging.info(
        "vk.crawl start groups=%d overlap=%s", len(groups), VK_CRAWL_OVERLAP_SEC
    )

    pages_per_group: list[int] = []

    now_ts = int(time.time())
    for group in groups:
        gid = group["group_id"]
        group_title_norm = _normalize_group_title(group.get("name"))
        group_screen_name_norm = _normalize_group_screen_name(
            group.get("screen_name")
        )
        group_title_display = _display_group_title(group.get("name"), gid)
        group_screen_name_display = _display_group_screen_name(
            group.get("screen_name"), gid
        )
        default_time = group.get("default_time")
        stats["groups_checked"] += 1
        await asyncio.sleep(random.uniform(0.7, 1.2))  # safety pause
        exporter.upsert_group_meta(
            gid,
            screen_name=group.get("screen_name"),
            name=group.get("name"),
            location=group.get("location"),
            default_time=default_time,
            default_ticket_link=group.get("default_ticket_link"),
        )
        backfill = False
        pages_loaded = 0
        group_posts = 0
        group_matches = 0
        group_added = 0
        group_duplicates = 0
        group_blank_single_photo_matches = 0
        group_history_matches = 0
        group_errors = 0
        safety_cap_triggered = False
        hard_cap_triggered = False
        reached_cursor_overlap = False
        deep_backfill_scheduled = False
        mode = "inc"
        try:
            async with db.raw_conn() as conn:
                cur = await conn.execute(
                    "SELECT last_seen_ts, last_post_id, updated_at, checked_at FROM vk_crawl_cursor WHERE group_id=?",
                    (gid,),
                )
                row = await cur.fetchone()
            cursor_updated_at_existing_raw: Any = None
            if row:
                last_seen_ts, last_post_id, updated_at, _checked_at = row
                cursor_updated_at_existing_raw = updated_at
                if isinstance(updated_at, str):
                    try:
                        updated_at_ts = int(
                            datetime.fromisoformat(updated_at).timestamp()
                        )
                    except ValueError:
                        try:
                            updated_at_ts = int(updated_at)
                        except (TypeError, ValueError):
                            updated_at_ts = 0
                elif updated_at:
                    updated_at_ts = int(updated_at)
                else:
                    updated_at_ts = 0
            else:
                last_seen_ts = last_post_id = 0
                updated_at_ts = 0
                cursor_updated_at_existing_raw = None

            idle_h = (now_ts - updated_at_ts) / 3600 if updated_at_ts else None
            backfill = force_backfill or last_seen_ts == 0 or (
                idle_h is not None and idle_h >= VK_CRAWL_BACKFILL_AFTER_IDLE_H
            )
            mode = "backfill" if backfill else "inc"

            posts: list[dict] = []

            next_cursor_ts = last_seen_ts
            next_cursor_pid = last_post_id
            cursor_updated_at_override: int | None = None
            cursor_payload: tuple[int, int, Any, int] | None = None
            has_new_posts = False

            if backfill:
                window_days = (
                    override_backfill_days
                    if override_backfill_days is not None
                    else VK_CRAWL_BACKFILL_DAYS
                )
                stats["backfill_days_used"] = window_days
                horizon = now_ts - window_days * 86400
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
                safety_cap_threshold = max(1, VK_CRAWL_MAX_PAGES_INC)
                hard_cap = safety_cap_threshold * 10
                while True:
                    page = await vk_wall_since(
                        gid, since, count=VK_CRAWL_PAGE_SIZE, offset=offset
                    )
                    pages_loaded += 1
                    posts.extend(page)

                    if page:
                        oldest_page_post = min(
                            page, key=lambda p: (p["date"], p["post_id"])
                        )
                        if oldest_page_post["date"] < last_seen_ts or (
                            oldest_page_post["date"] == last_seen_ts
                            and oldest_page_post["post_id"] <= last_post_id
                        ):
                            reached_cursor_overlap = True

                    if not page or len(page) < VK_CRAWL_PAGE_SIZE:
                        break

                    if reached_cursor_overlap:
                        break

                    if pages_loaded >= safety_cap_threshold:
                        safety_cap_triggered = True
                    if pages_loaded >= hard_cap:
                        hard_cap_triggered = True
                        logging.warning(
                            "vk.crawl.inc.hard_cap group=%s pages=%s since=%s last_seen=%s",
                            gid,
                            pages_loaded,
                            since,
                            last_seen_ts,
                        )
                        break

                    offset += VK_CRAWL_PAGE_SIZE

                if safety_cap_triggered:
                    stats["safety_cap_hits"] += 1
                    logging.warning(
                        "vk.crawl.inc.safety_cap group=%s pages=%s threshold=%s", 
                        gid,
                        pages_loaded,
                        safety_cap_threshold,
                    )
                    try:
                        import main

                        main.vk_crawl_safety_cap_total += 1
                    except Exception:
                        pass

            max_ts, max_pid = last_seen_ts, last_post_id

            for post in posts:
                ts = post["date"]
                pid = post["post_id"]
                matched_kw_value = ""
                has_date_value = 0
                event_ts_hint: int | None = None
                matched_kw_list: list[str] = []
                is_match = False
                history_hit = False
                has_date = False
                kw_ok = False
                if ts < last_seen_ts or (ts == last_seen_ts and pid <= last_post_id):
                    continue
                if ts > max_ts or (ts == max_ts and pid > max_pid):
                    max_ts, max_pid = ts, pid
                stats["posts_scanned"] += 1
                group_posts += 1
                post_text = post.get("text", "")
                photos = post.get("photos", []) or []
                post_url = post.get("url")
                miss_url = post_url or f"https://vk.com/wall-{gid}_{pid}"
                blank_single_photo = not post_text.strip() and len(photos) == 1

                if blank_single_photo:
                    matched_kw_value = OCR_PENDING_SENTINEL
                    matched_kw_list = [OCR_PENDING_SENTINEL]
                    is_match = True
                else:
                    history_hit = detect_historical_context(post_text)
                    kw_ok, kws = match_keywords(post_text)
                    has_date = detect_date(post_text)
                    seen_kws: set[str] = set()
                    unique_kws: list[str] = []
                    for kw in kws:
                        if kw not in seen_kws:
                            seen_kws.add(kw)
                            unique_kws.append(kw)
                    if kw_ok and has_date:
                        log_keywords = list(unique_kws)
                        if history_hit and HISTORY_MATCHED_KEYWORD not in seen_kws:
                            log_keywords.append(HISTORY_MATCHED_KEYWORD)
                        event_ts_hint = extract_event_ts_hint(
                            post_text,
                            default_time,
                            publish_ts=ts,
                            tz=local_tz,
                        )
                        min_event_ts = int(time.time()) + 2 * 3600
                        fallback_applied = False
                        if event_ts_hint is None or event_ts_hint < min_event_ts:
                            allow_without_hint = False
                            year_match = re.search(r"\b20\d{2}\b", post_text)
                            if year_match:
                                try:
                                    year_val = int(year_match.group(0))
                                except ValueError:
                                    year_val = None
                                else:
                                    publish_year = datetime.fromtimestamp(
                                        ts, local_tz
                                    ).year
                                    if year_val is not None and year_val > publish_year:
                                        allow_without_hint = True
                            if not allow_without_hint:
                                exporter.log_miss(
                                    group_id=gid,
                                    group_title=group_title_display,
                                    group_screen_name=group_screen_name_display,
                                    post_id=pid,
                                    url=post_url,
                                    ts=int(time.time()),
                                    reason="past_event",
                                    matched_kw=log_keywords,
                                    kw_ok=bool(kw_ok),
                                    has_date=bool(has_date),
                                )
                                _record_rejection(
                                    gid,
                                    pid,
                                    miss_url,
                                    VkImportRejectCode.PAST_EVENT,
                                    "past_event",
                                )
                                continue
                            fallback_applied = True
                        if not fallback_applied:
                            far_threshold = int(time.time()) + 2 * 365 * 86400
                            if event_ts_hint > far_threshold:
                                exporter.log_miss(
                                    group_id=gid,
                                    group_title=group_title_display,
                                    group_screen_name=group_screen_name_display,
                                    post_id=pid,
                                    url=post_url,
                                    ts=int(time.time()),
                                    reason="too_far",
                                    matched_kw=log_keywords,
                                    kw_ok=bool(kw_ok),
                                    has_date=bool(has_date),
                                )
                                _record_rejection(
                                    gid,
                                    pid,
                                    miss_url,
                                    VkImportRejectCode.TOO_FAR,
                                    "too_far",
                                )
                                continue
                        matched_kw_list = log_keywords
                        matched_kw_value = ",".join(matched_kw_list)
                        has_date_value = 1
                        if fallback_applied:
                            event_ts_hint = None
                        is_match = True
                    elif history_hit:
                        matched_kw_value = HISTORY_MATCHED_KEYWORD
                        matched_kw_list = [HISTORY_MATCHED_KEYWORD]
                        has_date_value = int(has_date)
                        is_match = True
                    else:
                        reason = "no_date" if kw_ok else "no_keywords"
                        exporter.log_miss(
                            group_id=gid,
                            group_title=group_title_display,
                            group_screen_name=group_screen_name_display,
                            post_id=pid,
                            url=post_url,
                            ts=int(time.time()),
                            reason=reason,
                            matched_kw=unique_kws,
                            kw_ok=bool(kw_ok),
                            has_date=bool(has_date),
                        )
                        code = (
                            VkImportRejectCode.NO_DATE
                            if reason == "no_date"
                            else VkImportRejectCode.NO_KEYWORDS
                        )
                        _record_rejection(gid, pid, miss_url, code, reason)
                        continue

                stats["matches"] += 1
                group_matches += 1
                if history_hit:
                    group_history_matches += 1
                if blank_single_photo:
                    group_blank_single_photo_matches += 1
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
                                matched_kw_value,
                                has_date_value,
                                event_ts_hint,
                            ),
                        )
                        await conn.commit()
                        if cur.rowcount == 0:
                            stats["duplicates"] += 1
                            group_duplicates += 1
                            existing_status: str | None = None
                            async with db.raw_conn() as conn:
                                cur_status = await conn.execute(
                                    "SELECT status FROM vk_inbox WHERE group_id=? AND post_id=? LIMIT 1",
                                    (gid, pid),
                                )
                                row_status = await cur_status.fetchone()
                            if row_status:
                                existing_status = row_status[0]
                            reason = (
                                "already_inbox"
                                if existing_status in {"pending", "locked", "skipped"}
                                else "duplicate"
                            )
                            exporter.log_miss(
                                group_id=gid,
                                group_title=group_title_display,
                                group_screen_name=group_screen_name_display,
                                post_id=pid,
                                url=post_url,
                                ts=int(time.time()),
                                reason=reason,
                                matched_kw=matched_kw_list,
                                kw_ok=bool(kw_ok),
                                has_date=bool(has_date),
                            )
                            code = (
                                VkImportRejectCode.ALREADY_INBOX
                                if reason == "already_inbox"
                                else VkImportRejectCode.DUPLICATE
                            )
                            _record_rejection(gid, pid, miss_url, code, reason)
                        else:
                            stats["added"] += 1
                            group_added += 1
                            has_new_posts = True
                except Exception:
                    stats["errors"] += 1
                    group_errors += 1
                    continue

            next_cursor_ts = max_ts
            next_cursor_pid = max_pid
            if hard_cap_triggered and max_ts > 0 and not reached_cursor_overlap:
                deep_backfill_scheduled = True
                next_cursor_ts = last_seen_ts
                next_cursor_pid = last_post_id
                idle_threshold = VK_CRAWL_BACKFILL_AFTER_IDLE_H * 3600
                cursor_updated_at_override = max(0, now_ts - idle_threshold - 60)
            elif safety_cap_triggered and max_ts > 0:
                adjusted_ts = max(last_seen_ts, max_ts - VK_CRAWL_OVERLAP_SEC)
                if adjusted_ts < next_cursor_ts:
                    next_cursor_ts = adjusted_ts
                    next_cursor_pid = 0

            if deep_backfill_scheduled:
                stats["deep_backfill_triggers"] += 1
                logging.warning(
                    "vk.crawl.inc.deep_backfill_trigger group=%s pages=%s last_seen=%s next_ts=%s",
                    gid,
                    pages_loaded,
                    last_seen_ts,
                    max_ts,
                )

            mode = "backfill" if backfill else "inc"
            logging.info(
                "vk.crawl group=%s posts=%s matched=%s pages=%s mode=%s",
                gid,
                group_posts,
                group_added,
                pages_loaded,
                mode,
            )
            cursor_checked_at = int(time.time())
            if cursor_updated_at_override is not None:
                cursor_updated_at = cursor_updated_at_override
            elif has_new_posts:
                cursor_updated_at = now_ts
            else:
                cursor_updated_at = cursor_updated_at_existing_raw
            cursor_payload = (
                next_cursor_ts,
                next_cursor_pid,
                cursor_updated_at,
                cursor_checked_at,
            )
        except Exception:
            stats["errors"] += 1
            group_errors += 1
            cursor_payload = None
        else:
            if cursor_payload is not None:
                async with db.raw_conn() as conn:
                    await conn.execute(
                        "INSERT OR REPLACE INTO vk_crawl_cursor(group_id, last_seen_ts, last_post_id, updated_at, checked_at) VALUES(?,?,?,?,?)",
                        (gid, *cursor_payload),
                    )
                    await conn.commit()
        finally:
            pages_per_group.append(pages_loaded)
            match_rate = group_matches / max(1, group_posts)
            snapshot_counters = {
                "posts_scanned": group_posts,
                "matched": group_matches,
                "duplicates": group_duplicates,
                "errors": group_errors,
                "pages_loaded": pages_loaded,
            }
            exporter.write_snapshot(
                group_id=gid,
                group_title=group.get("name"),
                group_screen_name=group.get("screen_name"),
                ts=int(time.time()),
                match_rate=match_rate,
                errors=group_errors,
                counters=snapshot_counters,
            )

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
            forced_note = ""
            if stats.get("forced_backfill"):
                used_days = stats.get("backfill_days_used") or VK_CRAWL_BACKFILL_DAYS
                requested_days = stats.get("backfill_days_requested")
                forced_note = f", принудительный бэкафилл до {used_days} дн."
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
                f"теперь в очереди {stats['inbox_total']} "
                f"(pending: {q.get('pending',0)}, locked: {q.get('locked',0)}, "
                f"skipped: {q.get('skipped',0)}, imported: {q.get('imported',0)}, "
                f"rejected: {q.get('rejected',0)}), "
                f"страниц на группу: {'/'.join(str(p) for p in stats['pages_per_group'])}, "
                f"перекрытие: {stats['overlap_sec']} сек"
                f"{forced_note}"
            )
            try:
                await bot.send_message(int(admin_chat), msg)
            except Exception:
                logging.exception("vk.crawl.broadcast.error")
    exporter.retention()
    return stats
