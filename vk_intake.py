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
    is_supabase_storage_url,
    process_media,
)
import poster_ocr
from source_parsing.date_utils import normalize_implicit_iso_date_to_anchor

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

_VK_PARSE_PREFILTER_VISIT_HINT_RE = re.compile(
    r"\b("
    r"билет\w*|регистрац\w*|вход\w*|стоимост\w*|донат\w*|"
    r"приглаша(?:ем|ют)\w*|приходите|жд[её]м|"
    r"состоит(?:ся|есь)\w*|пройдет\w*|пройд[её]т\w*|"
    r"начал\w*|открыти\w*|сеанс\w*|"
    r"экскурси\w*|лекци\w*|концерт\w*|спектакл\w*|"
    r"выставк\w*|кинопоказ\w*|мастер[ -]класс\w*|фестивал\w*"
    r")\b",
    re.I | re.U,
)
_VK_PARSE_PREFILTER_ADMIN_RE = re.compile(
    r"\b("
    r"администрац\w*|жител\w*|голосовани\w*|итог\w*|"
    r"проект[а-я-]*победител\w*|благоустройств\w*|"
    r"муниципальн\w*|округ\w*|район\w*|нацпроект\w*|"
    r"заседан\w*|депутат\w*|совет\w*|"
    r"поздрав\w*|наград\w*|юбиляр\w*"
    r")\b",
    re.I | re.U,
)
_VK_PARSE_GIVEAWAY_RE = re.compile(
    r"\b(розыгрыш|разыгрыва\w*|розыгра\w*|выигра\w*|конкурс|giveaway)\b",
    re.I | re.U,
)
_VK_PARSE_GIVEAWAY_TICKETS_RE = re.compile(
    r"\b(билет\w*|пригласительн\w*|абонемент\w*)\b",
    re.I | re.U,
)
_VK_PARSE_GIVEAWAY_MECHANICS_RE = re.compile(
    r"\b("
    r"услови\w*|участв\w*|подпиш\w*|репост\w*|коммент\w*|"
    r"отмет\w*|лайк\w*|победител\w*|итог\w*|приз\w*"
    r")\b",
    re.I | re.U,
)


def _read_int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _vk_parse_should_add_giveaway_prize_hint(
    text: str,
    *,
    poster_texts: Sequence[str] | None = None,
) -> bool:
    combined_parts: list[str] = [text or ""]
    for block in list(poster_texts or [])[:2]:
        if isinstance(block, str) and block.strip():
            combined_parts.append(block)
    combined = "\n".join(part for part in combined_parts if part and part.strip())
    if not combined:
        return False
    norm = unicodedata.normalize("NFKC", combined).casefold().replace("ё", "е")
    if not (_VK_PARSE_GIVEAWAY_RE.search(norm) and _VK_PARSE_GIVEAWAY_TICKETS_RE.search(norm)):
        return False
    return bool(_VK_PARSE_GIVEAWAY_MECHANICS_RE.search(norm))


def _normalize_prompt_ocr_block(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\xa0", " ")
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines).strip()


def _truncate_prompt_block(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 80:
        return text[:limit].rstrip()
    head = max(1, int(limit * 0.65))
    tail = max(0, limit - head - 5)
    if tail <= 0:
        return text[:limit].rstrip()
    return f"{text[:head].rstrip()}\n...\n{text[-tail:].lstrip()}".strip()


def _budget_vk_parse_poster_texts(post_text: str, poster_texts: Sequence[str]) -> list[str]:
    cleaned = [
        block
        for block in (_normalize_prompt_ocr_block(text) for text in poster_texts)
        if block
    ]
    if not cleaned:
        return []

    main_text_len = len((post_text or "").strip())
    skip_main_text_chars = max(0, _read_int_env("VK_PARSE_POSTER_TEXT_SKIP_MAIN_TEXT_CHARS", 1600))
    if skip_main_text_chars and main_text_len >= skip_main_text_chars:
        logger.info(
            "vk.parse budget: skip poster OCR for long post text_len=%s posters=%s",
            main_text_len,
            len(cleaned),
        )
        return []

    max_blocks = max(1, _read_int_env("VK_PARSE_POSTER_TEXT_MAX_BLOCKS", 3))
    max_block_chars = max(80, _read_int_env("VK_PARSE_POSTER_TEXT_MAX_BLOCK_CHARS", 500))
    max_total_chars = max(max_block_chars, _read_int_env("VK_PARSE_POSTER_TEXT_MAX_TOTAL_CHARS", 1200))

    selected: list[str] = []
    remaining = max_total_chars
    for block in cleaned:
        if len(selected) >= max_blocks or remaining <= 0:
            break
        limit = min(max_block_chars, remaining)
        trimmed = _truncate_prompt_block(block, limit).strip()
        if not trimmed:
            continue
        selected.append(trimmed)
        remaining -= len(trimmed)

    if len(selected) != len(cleaned):
        logger.info(
            "vk.parse budget: poster OCR reduced blocks=%s->%s total_chars=%s->%s",
            len(cleaned),
            len(selected),
            sum(len(block) for block in cleaned),
            sum(len(block) for block in selected),
        )
    return selected


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


def _vk_parse_preclassify(
    text: str,
    *,
    source_name: str | None = None,
    poster_texts: Sequence[str] | None = None,
    publish_ts: datetime | int | float | None = None,
    event_ts_hint: int | None = None,
    operator_extra: str | None = None,
    festival_hint: bool = False,
) -> tuple[str, str | None]:
    """Cheap conservative gate before the full VK parse prompt.

    The goal is not to classify every post, only to skip obvious long-form
    non-events that would otherwise reserve >12k TPM and still end up rejected.
    Anything even slightly ambiguous stays in ``maybe_event`` and proceeds to
    the normal LLM parser unchanged.
    """
    if festival_hint or (operator_extra or "").strip():
        return "maybe_event", None

    enabled = (os.getenv("VK_AUTO_IMPORT_PREFILTER_OBVIOUS_NON_EVENTS", "1") or "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return "maybe_event", None

    text_clean = (text or "").strip()
    if not text_clean:
        return "maybe_event", None

    history_min_chars = max(800, _read_int_env("VK_AUTO_IMPORT_PREFILTER_HISTORY_MIN_CHARS", 2200))
    admin_min_chars = max(800, _read_int_env("VK_AUTO_IMPORT_PREFILTER_ADMIN_MIN_CHARS", 1800))
    if len(text_clean) < min(history_min_chars, admin_min_chars):
        return "maybe_event", None

    context_parts: list[str] = [text_clean]
    source_clean = (source_name or "").strip()
    if source_clean:
        context_parts.append(source_clean)
    for block in list(poster_texts or [])[:3]:
        block_clean = (block or "").strip()
        if block_clean:
            context_parts.append(block_clean)
    combined_text = "\n".join(context_parts)
    combined_norm = unicodedata.normalize("NFKC", combined_text).casefold().replace("ё", "е")

    future_hint = int(event_ts_hint) if isinstance(event_ts_hint, int) and event_ts_hint > 0 else None
    if future_hint is None:
        try:
            tzinfo = require_main_attr("LOCAL_TZ")
            future_hint = extract_event_ts_hint(
                combined_text,
                default_time=None,
                publish_ts=publish_ts,
                allow_past=False,
                tz=tzinfo,
            )
        except Exception:
            future_hint = None
    if future_hint:
        return "maybe_event", None

    kw_ok, _matched = match_keywords(combined_text)
    visitable_signal = bool(
        kw_ok
        or PRICE_RE.search(combined_norm)
        or _VK_PARSE_PREFILTER_VISIT_HINT_RE.search(combined_norm)
    )
    if visitable_signal:
        return "maybe_event", None

    historical_years = {
        int(match)
        for match in HISTORICAL_YEAR_RE.findall(combined_norm)
        if str(match).isdigit() and int(match) <= 1994
    }
    historical_hit = detect_historical_context(combined_norm)
    if len(text_clean) >= history_min_chars and historical_hit and historical_years:
        return (
            "non_event",
            "Длинный исторический/справочный пост без признаков будущего посещаемого события",
        )

    admin_hits = len(_VK_PARSE_PREFILTER_ADMIN_RE.findall(combined_norm))
    if len(text_clean) >= admin_min_chars and admin_hits >= 3:
        return (
            "non_event",
            "Длинный административный/новостной пост без признаков посещаемого события",
        )

    return "maybe_event", None


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
            # MONTH_NAME_RE is intentionally broad (number + word) to catch
            # "13 февраля", but it also matches unrelated fragments like
            # "3 этаж" or "2 зал". Pick the first match that is a real month.
            for cand in MONTH_NAME_RE.finditer(text_low):
                mon_word = cand.group(2).rstrip(".")
                mon_num = MONTHS_RU.get(mon_word)
                if mon_num is None:
                    continue
                m = cand
                day = int(cand.group(1))
                month = int(mon_num)
                y = re.search(r"\b20\d{2}\b", text_low[cand.end() :])
                if y:
                    year = int(y.group(0))
                date_span = cand.span()
                break

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
    time_is_default: bool = False
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
    search_digest: str | None = None
    reject_reason: str | None = None


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
    # Smart Update outcome (for unified operator report)
    smart_status: str | None = None
    smart_created: bool = False
    smart_merged: bool = False
    smart_added_posters: int = 0


def _vk_wall_source_ids_from_url(source_post_url: str | None) -> tuple[int | None, int | None]:
    if not source_post_url:
        return None, None
    m = re.search(r"wall-?(\d+)_([0-9]+)", source_post_url)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _collapse_same_post_exact_drafts(drafts_in: list[EventDraft]) -> list[EventDraft]:
    """Collapse obviously duplicated child drafts emitted from one VK multi-post.

    Safety: collapse only when date + explicit time + venue + normalized title are
    identical inside the same parsed draft set. This targets duplicated poster/card
    extraction, not real parallel events from one schedule post.
    """

    if len(drafts_in) < 2:
        return drafts_in

    def _norm_text(value: str | None, *, keep_digits: bool = True) -> str:
        text = unicodedata.normalize("NFKC", (value or "")).casefold().replace("ё", "е")
        pattern = r"[^a-zа-я0-9]+" if keep_digits else r"[^a-zа-я]+"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE).strip()
        return re.sub(r"\s+", " ", text)

    def _norm_title(value: str | None) -> str:
        return _norm_text(value, keep_digits=True)

    def _norm_location(value: str | None) -> str:
        return _norm_text(value, keep_digits=True)

    def _norm_link(value: str | None) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        return raw.rstrip("/")

    def _draft_score(draft: EventDraft) -> tuple[int, int, int, int]:
        score = 0
        score += min(len((_norm_title(draft.title) or "").split()), 8)
        score += min(len(str(draft.description or "").split()), 20)
        score += 4 if (draft.location_address or "").strip() else 0
        score += 2 if (draft.city or "").strip() else 0
        score += 2 if draft.links else 0
        score += min(len(draft.poster_media or []), 3)
        score += 1 if (draft.search_digest or "").strip() else 0
        score += 1 if (draft.event_type or "").strip() else 0
        score += 1 if draft.ticket_price_min is not None or draft.ticket_price_max is not None else 0
        return score, len(draft.source_text or ""), len(draft.description or ""), len(draft.title or "")

    def _poster_key(item: PosterMedia) -> tuple[str, str, str, str]:
        return (
            str(item.digest or ""),
            str(item.supabase_url or ""),
            str(item.catbox_url or ""),
            str(item.name or ""),
        )

    def _merge_links(primary: list[str] | None, secondary: list[str] | None) -> list[str] | None:
        merged: list[str] = []
        seen: set[str] = set()
        for url in list(primary or []) + list(secondary or []):
            norm = _norm_link(url)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(url)
        return merged or None

    def _prefer_longer(current: str | None, candidate: str | None) -> str | None:
        cur = (current or "").strip()
        cand = (candidate or "").strip()
        if len(cand) > len(cur):
            return candidate
        return current

    groups: dict[tuple[str, str, str, str], list[EventDraft]] = {}
    passthrough: list[EventDraft] = []
    for draft in drafts_in:
        date_key = str(draft.date or "").strip()
        time_key = str(draft.time or "").strip()
        title_key = _norm_title(draft.title)
        venue_key = _norm_location(draft.venue or draft.location_address)
        if not (date_key and time_key and title_key and venue_key):
            passthrough.append(draft)
            continue
        groups.setdefault((date_key, time_key, venue_key, title_key), []).append(draft)

    collapsed_any = False
    out: list[EventDraft] = list(passthrough)
    for _, group in groups.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        collapsed_any = True
        keep = max(group, key=_draft_score)
        posters_by_key: dict[tuple[str, str, str, str], PosterMedia] = {
            _poster_key(item): item for item in list(keep.poster_media or [])
        }
        for other in group:
            if other is keep:
                continue
            keep.description = _prefer_longer(keep.description, other.description)
            keep.source_text = _prefer_longer(keep.source_text, other.source_text)
            keep.poster_summary = _prefer_longer(keep.poster_summary, other.poster_summary)
            keep.search_digest = _prefer_longer(keep.search_digest, other.search_digest)
            keep.location_address = _prefer_longer(keep.location_address, other.location_address)
            keep.city = _prefer_longer(keep.city, other.city)
            keep.event_type = keep.event_type or other.event_type
            keep.emoji = keep.emoji or other.emoji
            keep.festival = keep.festival or other.festival
            keep.is_free = bool(keep.is_free or other.is_free)
            keep.pushkin_card = bool(keep.pushkin_card or other.pushkin_card)
            if keep.ticket_price_min is None:
                keep.ticket_price_min = other.ticket_price_min
            if keep.ticket_price_max is None:
                keep.ticket_price_max = other.ticket_price_max
            keep.links = _merge_links(keep.links, other.links)
            keep.ocr_tokens_spent += int(other.ocr_tokens_spent or 0)
            for poster in list(other.poster_media or []):
                posters_by_key.setdefault(_poster_key(poster), poster)
        keep.poster_media = list(posters_by_key.values())
        keep.poster_summary = build_poster_summary(keep.poster_media)
        out.append(keep)

    if collapsed_any:
        logger.info(
            "vk_intake: collapsed same-post exact duplicate drafts dropped=%s kept=%s",
            len(drafts_in) - len(out),
            len(out),
        )
    return out


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


async def vk_intake_parse_llm(
    prompt_text: str,
    *,
    source_name: str | None = None,
    festival_names: Sequence[str] | None = None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None = None,
    poster_media: Sequence[PosterMedia] | None = None,
    rate_limit_max_wait_sec: float | int | str | None = None,
) -> Any:
    """Parse a VK post text into structured events using the universal LLM parser.

    Default backend is Gemma; set `EVENT_PARSE_LLM=4o` to force the legacy OpenAI parser.
    """
    parse_event_via_llm = require_main_attr("parse_event_via_llm")

    extra: dict[str, str] = {}
    if source_name:
        # ``parse_event_via_llm`` accepts ``channel_title`` for context.
        extra["channel_title"] = source_name

    parse_kwargs: dict[str, Any] = {}
    poster_items = list(poster_media or [])
    poster_texts = _budget_vk_parse_poster_texts(
        prompt_text,
        collect_poster_texts(poster_items),
    )
    poster_summary = build_poster_summary(poster_items)
    if poster_texts:
        parse_kwargs["poster_texts"] = poster_texts
    if poster_summary:
        parse_kwargs["poster_summary"] = poster_summary
    if festival_alias_pairs:
        parse_kwargs["festival_alias_pairs"] = festival_alias_pairs
    if rate_limit_max_wait_sec is not None:
        parse_kwargs["rate_limit_max_wait_sec"] = str(rate_limit_max_wait_sec)

    return await parse_event_via_llm(
        prompt_text,
        festival_names=festival_names,
        **extra,
        **parse_kwargs,
    )


async def build_event_drafts_from_vk(
    text: str,
    *,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    publish_ts: datetime | int | float | None = None,
    event_ts_hint: int | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: Sequence[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    poster_media: Sequence[PosterMedia] | None = None,
    ocr_tokens_spent: int = 0,
    ocr_tokens_remaining: int | None = None,
    rate_limit_max_wait_sec: float | int | str | None = None,
    prefilter_obvious_non_events: bool = False,
) -> tuple[list[EventDraft], dict[str, Any] | None]:
    """Return normalised event drafts extracted from a VK post.

    The function delegates parsing to the same LLM helper used by ``/add`` and
    forwarded posts.  When ``operator_extra`` is supplied it takes precedence
    over conflicting fragments of the original text.  ``source_name`` and
    ``location_hint`` are passed to the extractor for additional context and
    ``default_time`` (if set for the VK source) is used as a low-priority
    fallback when the post has no explicit time. Such time is marked with
    ``draft.time_is_default=True`` so Smart Update treats it as a weak anchor
    and can override it when explicit time arrives from other sources.

    The resulting :class:`EventDraft` contains normalised event attributes such
    as title, schedule, venue, ticket details and other metadata needed by the
    import pipeline.  The function returns a tuple ``(drafts, festival_payload)``
    where ``festival_payload`` is the raw festival structure, if any, provided
    by :func:`main.parse_event_via_llm`.
    """
    timings_on = (os.getenv("PIPELINE_TIMINGS") or "").strip().lower() in {"1", "true", "yes", "on"}
    poster_items = list(poster_media or [])
    poster_texts = collect_poster_texts(poster_items)
    poster_summary = build_poster_summary(poster_items)

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

    # LLM-first hinting: if the source explicitly says it's a standup/comedy show,
    # nudge the parser to make the format visible in the title (without hardcoding
    # deterministic renames after parsing).
    try:
        hint_parts: list[str] = [text or ""]
        if poster_texts:
            hint_parts.extend([p for p in poster_texts if isinstance(p, str) and p.strip()])
        hint_norm = unicodedata.normalize("NFKC", "\n".join(hint_parts)).casefold().replace("ё", "е")
        if re.search(r"\b(?:стендап|стенд-?ап|stand\s*-?up|комик\w*|юмор\w*)\b", hint_norm, flags=re.IGNORECASE):
            llm_text += (
                "\nЕсли это стендап/комедия, сделай это явно в title (например «Стендап: …»), "
                "даже если оригинальное название звучит как «медитация». Не выдумывай детали."
            )
    except Exception:
        pass
    if _vk_parse_should_add_giveaway_prize_hint(text, poster_texts=poster_texts):
        llm_text += (
            "\nЕсли это розыгрыш/конкурс и мероприятие упомянуто только как приз "
            "(например билеты на матч/концерт), не создавай событие и верни `[]`. "
            "Извлекай событие только если пост отдельно описывает само посещаемое "
            "мероприятие, а не только механику розыгрыша."
        )
    if location_hint:
        hint_clean = str(location_hint).strip()
        if hint_clean:
            llm_text = (
                f"{llm_text}\n"
                "Хинт по локации (используй ТОЛЬКО если пост действительно описывает посещаемое событие, "
                f"но место не указано явно): {hint_clean}. "
                "Не создавай событие только из-за этого хинта. Если пост не про событие — верни `[]`."
            )
    if fallback_ticket_link:
        llm_text = (
            f"{llm_text}\n"
            "Хинт по ссылке: если и только если это событие и в посте нет ссылки на билеты/регистрацию, "
            f"используй {fallback_ticket_link} как ссылку по умолчанию. "
            "Не заменяй ссылки, которые уже указаны. Если пост не про событие — верни `[]`."
        )
    if festival_hint:
        llm_text = (
            f"{llm_text}\n"
            "Оператор подтверждает, что пост описывает фестиваль. "
            "Сопоставь с существующими фестивалями (JSON ниже) или создай новый."
        )

    if prefilter_obvious_non_events:
        verdict, reason = _vk_parse_preclassify(
            text,
            source_name=source_name,
            poster_texts=poster_texts,
            publish_ts=publish_ts,
            event_ts_hint=event_ts_hint,
            operator_extra=operator_extra,
            festival_hint=festival_hint,
        )
        if verdict == "non_event" and reason:
            logger.info(
                "vk.parse prefilter verdict=%s reason=%s source=%s text_len=%s posters=%s",
                verdict,
                reason,
                source_name or "vk",
                len((text or "").strip()),
                len(poster_items),
            )
            return [
                EventDraft(
                    title="",
                    source_text=text or None,
                    reject_reason=reason,
                )
            ], None

    t0 = time.monotonic()
    parsed = await vk_intake_parse_llm(
        llm_text,
        source_name=source_name,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs,
        poster_media=poster_media,
        rate_limit_max_wait_sec=rate_limit_max_wait_sec,
    )
    if timings_on:
        try:
            logger.info(
                "timing vk_intake_parse_llm events_hint=%s posters=%s took_sec=%.3f",
                "unknown",
                len(list(poster_media or [])),
                float(time.monotonic() - t0),
            )
        except Exception:
            pass
    festival_payload = getattr(parsed, "festival", None)
    parsed_events = list(parsed or [])
    if not parsed_events and not festival_payload:
        # For VK auto-import we treat "no events extracted" as a valid outcome (0 drafts),
        # not a technical failure. Callers that require an event (manual flows) can
        # enforce that at a higher level (see build_event_draft/build_event_payload_from_vk).
        return [], None

    combined_text = text or ""
    extra_clean = (operator_extra or "").strip()
    if extra_clean:
        trimmed = combined_text.rstrip()
        combined_text = f"{trimmed}\n\n{extra_clean}" if trimmed else extra_clean

    # Date normalization logic
    tzinfo = require_main_attr("LOCAL_TZ")
    if publish_ts is None:
        anchor_dt = datetime.now(tzinfo)
    elif isinstance(publish_ts, datetime):
        anchor_dt = publish_ts.astimezone(tzinfo) if publish_ts.tzinfo else publish_ts.replace(tzinfo=tzinfo)
    else:
        anchor_dt = datetime.fromtimestamp(publish_ts, tzinfo)

    effective_ts_hint = event_ts_hint
    if operator_extra or effective_ts_hint is None:
        computed = extract_event_ts_hint(
            combined_text,
            default_time=None,
            publish_ts=publish_ts,
            allow_past=False,
            tz=tzinfo
        )
        if computed:
            effective_ts_hint = computed

    hint_dt = None
    if effective_ts_hint:
        hint_dt = datetime.fromtimestamp(effective_ts_hint, tzinfo)

    _numeric_year_re = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-](19|20)\d{2}\b")
    _month_names_patt = "|".join(sorted(MONTHS_RU.keys(), key=len, reverse=True))
    _textual_year_re = re.compile(rf"\b\d{{1,2}}\s+(?:{_month_names_patt})\s+(?:19|20)\d{{2}}\b", re.IGNORECASE)
    has_explicit_year_in_text = bool(_numeric_year_re.search(combined_text) or _textual_year_re.search(combined_text))

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

    # Title grounding: prevent hallucinated/garbled short tokens in titles from leaking into UI.
    ground_parts: list[str] = []
    if isinstance(text, str) and text.strip():
        ground_parts.append(text)
    if poster_texts:
        ground_parts.extend([p for p in poster_texts if isinstance(p, str) and p.strip()])
    ground_norm = unicodedata.normalize("NFKC", "\n".join(ground_parts)).casefold().replace("ё", "е")

    _title_word_re = re.compile(r"[а-яё]{3,}", re.IGNORECASE)
    _title_prefix_re = re.compile(r"^([^a-zа-я0-9]+\\s+)", re.IGNORECASE)

    def _token_in_ground(token: str) -> bool:
        tok = token.casefold().replace("ё", "е").strip()
        if not tok:
            return True
        if tok in ground_norm:
            return True
        # Best-effort: allow simple inflection differences.
        if len(tok) >= 5 and tok[:-1] in ground_norm:
            return True
        if len(tok) >= 6 and tok[:-2] in ground_norm:
            return True
        return False

    def _missing_title_tokens(title: str) -> list[str]:
        words = [w for w in _title_word_re.findall(title or "") if w]
        missing: list[str] = []
        for w in words:
            if _token_in_ground(w):
                continue
            missing.append(w)
        return missing

    def _fallback_title(title: str, *, event_type: str | None, venue: str | None) -> str:
        prefix = ""
        m = _title_prefix_re.match(title or "")
        if m:
            prefix = m.group(1)
        et = (event_type or "").strip().casefold()
        if any(k in et for k in ("выстав", "экспоз")):
            base = "Интерактивная экспозиция"
        elif et:
            base = (event_type or "").strip().capitalize()
        else:
            base = "Событие"
        venue_short = (venue or "").split(",", 1)[0].strip()
        core = f"{base} — {venue_short}" if venue_short else base
        return (prefix + core).strip()

    drafts: list[EventDraft] = []

    def _source_norm_text() -> str:
        parts = [combined_text]
        if poster_texts:
            parts.extend([p for p in poster_texts if isinstance(p, str) and p.strip()])
        s = unicodedata.normalize("NFKC", "\n".join([p for p in parts if p]))
        s = s.replace("\xa0", " ")
        return s.casefold().replace("ё", "е")

    source_norm = _source_norm_text()

    def _sanitize_false_time_from_date(
        *,
        draft_date: str | None,
        draft_time: str | None,
    ) -> str | None:
        """Fix common LLM confusion: date token DD.MM -> time HH:MM.

        Example: source mentions "21.02" (Feb 21), but model outputs time "21:02".
        We treat this as a correctness fix (not an editorial rewrite) and only apply
        it when date and time numerals match exactly.
        """
        t_raw = (draft_time or "").strip()
        if not t_raw:
            return draft_time
        m = re.match(r"^\s*(\d{1,2})[:.](\d{2})\s*$", t_raw)
        if not m:
            return draft_time
        try:
            hh = int(m.group(1))
            mm = int(m.group(2))
        except Exception:
            return draft_time
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return draft_time
        d_raw = (draft_date or "").strip()
        if not d_raw:
            return draft_time
        try:
            d_obj = date.fromisoformat(d_raw.split("..", 1)[0].strip())
        except Exception:
            return draft_time
        # Strong signal: time numerals equal the event date day/month.
        if (d_obj.day, d_obj.month) != (hh, mm):
            return draft_time

        date_dot = f"{d_obj.day}.{d_obj.month:02d}"
        date_dot2 = f"{d_obj.day:02d}.{d_obj.month:02d}"
        time_colon = f"{hh:02d}:{mm:02d}"
        # Source contains DD.MM, but not HH:MM -> likely date, not time.
        if (date_dot in source_norm or date_dot2 in source_norm) and (time_colon not in source_norm):
            # If there is any other explicit time token in the source, do not touch.
            # (We only fix the "time copied from date" case when the source otherwise has no time.)
            other_times = re.findall(r"\b\d{1,2}[:.]\d{2}\b", source_norm)
            # Filter out the date-like token itself.
            other_times = [x for x in other_times if x.replace(".", ":") != time_colon]
            if not other_times:
                return None
        return draft_time

    def _looks_like_program_schedule_source() -> bool:
        if not source_norm:
            return False
        if not re.search(r"\b(?:программ\w*|расписан\w*|тайминг|тайм-?инг|в\s+программ\w*)\b", source_norm):
            return False
        times = re.findall(r"\b\d{1,2}[:.]\d{2}\b", source_norm)
        # Require 2+ time tokens to treat it as a schedule/program.
        return len(times) >= 2

    def _maybe_collapse_program_schedule_drafts(drafts_in: list[EventDraft]) -> list[EventDraft]:
        """Collapse duplicate drafts produced from a single program/schedule post.

        Applies only when the source clearly looks like one umbrella event with a program.
        Guardrails are intentionally strict to avoid collapsing theatre multi-show posts.
        """
        if len(drafts_in) < 2:
            return drafts_in
        if not _looks_like_program_schedule_source():
            return drafts_in

        def _norm_title(value: str | None) -> str:
            t = unicodedata.normalize("NFKC", (value or "")).strip()
            t = re.sub(r"^[^a-zа-яё0-9]+", "", t, flags=re.IGNORECASE).strip()
            t = re.sub(r"[^a-zа-яё0-9]+", " ", t, flags=re.IGNORECASE).strip()
            return t.casefold().replace("ё", "е")

        def _parse_hhmm(value: str | None) -> tuple[int, int] | None:
            s = (value or "").strip()
            if not s:
                return None
            s = s.split("..", 1)[0].strip()
            m2 = re.match(r"^(\d{1,2})[:.](\d{2})$", s)
            if not m2:
                return None
            try:
                hh2 = int(m2.group(1))
                mm2 = int(m2.group(2))
            except Exception:
                return None
            if not (0 <= hh2 <= 23 and 0 <= mm2 <= 59):
                return None
            return hh2, mm2

        def _fmt(hh2: int, mm2: int) -> str:
            return f"{hh2:02d}:{mm2:02d}"

        # Use times from source, not from drafts, to get the full range.
        src_times: list[tuple[int, int]] = []
        for tok in re.findall(r"\b\d{1,2}[:.]\d{2}\b", source_norm):
            m2 = re.match(r"^(\d{1,2})[:.](\d{2})$", tok)
            if not m2:
                continue
            try:
                hh2 = int(m2.group(1))
                mm2 = int(m2.group(2))
            except Exception:
                continue
            if 0 <= hh2 <= 23 and 0 <= mm2 <= 59:
                src_times.append((hh2, mm2))
        src_times = sorted(set(src_times))
        if len(src_times) < 2:
            return drafts_in
        start_hh, start_mm = src_times[0]
        end_hh, end_mm = src_times[-1]
        if (end_hh, end_mm) <= (start_hh, start_mm):
            return drafts_in
        merged_time = f"{_fmt(start_hh, start_mm)}..{_fmt(end_hh, end_mm)}"

        # Group by date + venue + normalized title; collapse only within a single clear group.
        groups: dict[tuple[str, str, str], list[EventDraft]] = {}
        for d in drafts_in:
            d_date = (d.date or "").strip()
            d_venue = (d.venue or "").strip()
            key = (d_date, d_venue, _norm_title(d.title))
            groups.setdefault(key, []).append(d)

        # Pick the largest eligible group.
        best_key = None
        best_group: list[EventDraft] = []
        for key, grp in groups.items():
            if len(grp) < 2:
                continue
            # Skip theatre shows: multi-time theatre posts should remain separate.
            et = (grp[0].event_type or "").strip().casefold()
            if et == "спектакль":
                continue
            # All titles should match after normalization.
            if len({_norm_title(x.title) for x in grp}) != 1:
                continue
            # All must have the same date and venue already by key.
            if not key[0] or not key[1] or not key[2]:
                continue
            # Require that drafts appear to differ mainly by time.
            parsed_times = [_parse_hhmm(x.time) for x in grp]
            if not all(pt is not None for pt in parsed_times):
                continue
            # Prefer collapsing when there are 3+ time mentions (typical "program").
            if len(src_times) < 3:
                continue
            if len(grp) > len(best_group):
                best_key = key
                best_group = grp

        if not best_group or best_key is None:
            return drafts_in

        # Keep one draft (stable choice: earliest time among group), override its time with the range.
        best_group_sorted = sorted(
            best_group,
            key=lambda d: _parse_hhmm(d.time) or (99, 99),
        )
        keep = best_group_sorted[0]
        keep.time = merged_time
        out: list[EventDraft] = []
        for d in drafts_in:
            if d is keep:
                out.append(d)
                continue
            if d in best_group:
                continue
            out.append(d)
        return out

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

        raw_date = clean_str(data.get("date"))
        raw_time = clean_str(data.get("time"))
        final_date = raw_date
        final_time = raw_time
        time_is_default = False

        if raw_date and not has_explicit_year_in_text:
            final_date = _maybe_rollover_llm_iso_date(
                raw_date,
                anchor_date=anchor_dt.date(),
                has_explicit_year_in_text=has_explicit_year_in_text,
            )
            if hint_dt:
                try:
                    d_check = date.fromisoformat((final_date or raw_date).split("..", 1)[0].strip())
                    if (d_check.month, d_check.day) == (hint_dt.month, hint_dt.day):
                        if d_check.year != hint_dt.year:
                            final_date = hint_dt.date().isoformat()
                        if not final_time and hint_dt.strftime("%H:%M") != "00:00":
                            final_time = hint_dt.strftime("%H:%M")
                except ValueError:
                    pass

        final_time = _sanitize_false_time_from_date(draft_date=final_date, draft_time=final_time)
        if final_date and (not final_time) and default_time:
            final_time = default_time
            time_is_default = True

        title_raw = clean_str(data.get("title")) or ""
        event_type_val = clean_str(data.get("event_type"))
        missing_tokens = _missing_title_tokens(title_raw)
        # Heuristic: if the title contains any Cyrillic word (3+ chars) absent from the source
        # text/OCR, it's likely a hallucination/typo (e.g. "Утя"). Prefer a safe fallback.
        if missing_tokens:
            venue_val = clean_str(data.get("location_name")) or source_name or ""
            fallback = _fallback_title(
                title_raw,
                event_type=event_type_val,
                venue=venue_val,
            )
            logger.warning(
                "vk_intake suspicious_title replaced title=%r missing=%s fallback=%r",
                title_raw,
                ",".join(missing_tokens[:4]),
                fallback,
            )
            title_raw = fallback

        drafts.append(
            EventDraft(
                title=title_raw,
                date=final_date,
                time=final_time,
                time_is_default=time_is_default,
                venue=data.get("location_name"),
                description=data.get("short_description"),
                festival=clean_str(data.get("festival")),
                location_address=clean_str(data.get("location_address")),
                city=clean_str(data.get("city")),
                ticket_price_min=ticket_price_min,
                ticket_price_max=ticket_price_max,
                event_type=event_type_val,
                emoji=clean_str(data.get("emoji")),
                end_date=clean_str(data.get("end_date")),
                is_free=clean_bool(data.get("is_free")),
                pushkin_card=clean_bool(data.get("pushkin_card")),
                links=links,
                source_text=combined_text,
                poster_media=list(poster_items),
                poster_summary=poster_summary,
                ocr_tokens_spent=ocr_tokens_spent,
                ocr_tokens_remaining=ocr_tokens_remaining,
                search_digest=clean_str(data.get("search_digest")),
            )
        )

    # If a single VK post describes multiple events, do not blindly attach the whole
    # poster gallery to every event: this often results in the wrong cover/poster
    # on Telegraph pages. Instead, try to assign posters to drafts by OCR relevance
    # and drop ambiguous matches.
    if len(drafts) > 1 and poster_items:
        try:
            month_words: dict[int, set[str]] = {}
            for word, num in MONTHS_RU.items():
                try:
                    num_i = int(num)
                except Exception:
                    continue
                month_words.setdefault(num_i, set()).add(str(word).casefold())

            stop_tokens = {
                "афиша",
                "вход",
                "билет",
                "билеты",
                "руб",
                "рублей",
                "цена",
                "стоимость",
                "начало",
                "начнется",
                "начнётся",
                "сбор",
                "регистрация",
            }

            def _norm_text(value: str | None) -> str:
                text_val = (value or "").strip().casefold().replace("ё", "е")
                text_val = unicodedata.normalize("NFKC", text_val)
                text_val = text_val.replace("\xa0", " ")
                return text_val

            def _tokens(value: str | None) -> set[str]:
                raw = _norm_text(value)
                if not raw:
                    return set()
                found = re.findall(r"[a-zа-я0-9]{3,}", raw, flags=re.IGNORECASE)
                return {t for t in found if t and t not in stop_tokens}

            def _date_bonus(draft: EventDraft, ocr_raw: str) -> float:
                d_raw = (draft.date or "").strip()
                if not d_raw:
                    return 0.0
                try:
                    d_obj = date.fromisoformat(d_raw.split("..", 1)[0].strip())
                except Exception:
                    return 0.0
                day = d_obj.day
                month = d_obj.month
                # Numeric formats: 14.02, 14/02, 14-02, allow 1-digit month/day.
                if re.search(rf"\\b0?{day}[./-]0?{month}\\b", ocr_raw):
                    return 3.0
                # Text month formats: "14 февраля"
                words = month_words.get(month) or set()
                if words:
                    if any(w in ocr_raw for w in words) and re.search(rf"\\b{day}\\b", ocr_raw):
                        return 2.0
                return 0.0

            def _time_bonus(draft: EventDraft, ocr_raw: str) -> float:
                t = (draft.time or "").strip()
                if not t or t == "00:00":
                    return 0.0
                hhmm = re.sub(r"\\s+", "", t)
                if not re.match(r"^\\d{1,2}:\\d{2}$", hhmm):
                    return 0.0
                hh, mm = hhmm.split(":", 1)
                hh = hh.zfill(2)
                needle1 = f"{hh}:{mm}"
                needle2 = f"{hh}.{mm}"
                if needle1 in ocr_raw or needle2 in ocr_raw:
                    return 1.5
                return 0.0

            def _poster_score(draft: EventDraft, poster: PosterMedia) -> float:
                ocr_combined = " ".join(
                    x
                    for x in [
                        (poster.ocr_title or "").strip(),
                        (poster.ocr_text or "").strip(),
                    ]
                    if x
                ).strip()
                if not ocr_combined:
                    return 0.0
                ocr_raw = _norm_text(ocr_combined)

                draft_text = " ".join(
                    x
                    for x in [
                        (draft.title or "").strip(),
                        (draft.venue or "").strip(),
                        (draft.festival or "").strip(),
                    ]
                    if x
                )

                draft_tokens = _tokens(draft_text)
                ocr_tokens = _tokens(ocr_combined)
                overlap = len(draft_tokens & ocr_tokens) if (draft_tokens and ocr_tokens) else 0

                score = float(min(12, overlap * 2))
                score += _date_bonus(draft, ocr_raw)
                score += _time_bonus(draft, ocr_raw)

                # If OCR title contains a substantial part of the event title, boost.
                title_norm = _norm_text(draft.title)
                if title_norm and len(title_norm) >= 10 and title_norm in ocr_raw:
                    score += 4.0

                return score

            max_per_draft = 3
            assigned: dict[int, list[PosterMedia]] = {i: [] for i in range(len(drafts))}

            for poster in poster_items:
                scores = [(_poster_score(d, poster), idx) for idx, d in enumerate(drafts)]
                scores.sort(key=lambda x: x[0], reverse=True)
                best_score, best_idx = scores[0]
                second = scores[1][0] if len(scores) > 1 else 0.0

                # Guardrails: require confident match; otherwise drop to avoid wrong posters.
                if best_score < 3.0:
                    continue
                if (best_score - second) < 1.5:
                    continue
                if len(assigned[best_idx]) >= max_per_draft:
                    continue
                assigned[best_idx].append(poster)

            for idx, draft in enumerate(drafts):
                draft.poster_media = list(assigned.get(idx) or [])
                draft.poster_summary = build_poster_summary(draft.poster_media)
        except Exception:
            logging.warning("vk_intake: poster assignment failed", exc_info=True)

    def _venue_looks_like_organizer_not_place(venue: str | None, address: str | None) -> bool:
        name = (venue or "").strip()
        if not name:
            return False
        if (address or "").strip():
            return False
        low = name.casefold()
        # Heuristic: some LLM parses put an organizer/artist into location_name when the post
        # doesn't contain an explicit venue. Reject such drafts to avoid garbage events.
        bad_tokens = (
            "оркестр",
            "ансамбль",
            "коллектив",
            "солист",
            "дириж",
            "лауреат",
            "исполнитель",
        )
        good_tokens = (
            "театр",
            "музей",
            "библиотек",
            "центр",
            "дк",
            "дом культуры",
            "зал",
            "кино",
            "галере",
            "филармон",
            "клуб",
            "студ",
            "выставоч",
        )
        if any(t in low for t in good_tokens):
            return False
        if any(t in low for t in bad_tokens):
            return True
        # Very long names without an address-like token are suspicious.
        if len(name) >= 52 and not re.search(r"\\b(ул\\.?|просп\\.?|пр-т|наб\\.?|пл\\.?|дом|д\\.)\\b", low):
            return True
        return False

    kept: list[EventDraft] = []
    dropped = 0
    for draft in drafts:
        if _venue_looks_like_organizer_not_place(draft.venue, draft.location_address):
            dropped += 1
            continue
        kept.append(draft)
    if dropped:
        logging.info(
            "vk_intake: dropped drafts due to suspicious venue: dropped=%s kept=%s",
            dropped,
            len(kept),
        )
    drafts = kept

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

    # Guardrail: do not accept a parsed `date` when the source contains no explicit/relative
    # datetime signals. This protects VK auto-import from "today" hallucinations on non-event posts.
    datetime_signal_re = re.compile(
        r"(?iu)\b("
        r"\d{1,2}[./-]\d{1,2}(?:[./-](?:19|20)\d{2})?|"
        r"(?:[01]?\d|2[0-3])[:.][0-5]\d|"
        r"(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*|"
        r"сегодня|завтра|послезавтра|"
        r"выходн\w*|"
        r"понедел\w*|вторник\w*|сред\w*|четверг\w*|пятниц\w*|суббот\w*|воскресен\w*"
        r")\b"
    )
    has_datetime_evidence = bool(datetime_signal_re.search(source_norm or ""))
    if not has_datetime_evidence:
        for draft in drafts:
            if (draft.reject_reason or "").strip():
                continue
            if (draft.date or "").strip() or (draft.end_date or "").strip():
                draft.reject_reason = "Нет сигналов даты/времени в источнике"

    # Guardrail: do not create one-off events that are already in the past relative to
    # the post publish time. Recap posts may contain past dates (for context), but those
    # should not become standalone events.
    for draft in drafts:
        if (draft.reject_reason or "").strip():
            continue
        start_d, end_d = _parse_iso_date_range(draft.date, end_value=draft.end_date)
        if not start_d:
            continue
        end_d = end_d or start_d
        if end_d >= anchor_dt.date():
            continue
        event_type_cf = str(getattr(draft, "event_type", "") or "").strip().casefold()
        # For long-running events without an explicit end date, allow (best-effort).
        if ".." not in str(draft.date or "") and not str(draft.end_date or "").strip():
            if event_type_cf in {"выставка", "экспозиция", "ярмарка"}:
                continue
        draft.reject_reason = f"Событие в прошлом: {end_d.isoformat()}"

    # Low-confidence guardrail: do not create events when the extracted title appears
    # to be copied from a recap of a past event, while the future announcement lacks
    # an explicit title. Mark drafts as rejected so callers can skip with a clear reason.
    for draft in drafts:
        if (draft.reject_reason or "").strip():
            continue
        reason = _looks_like_recap_title_copied_to_future_event(
            source_text=combined_text,
            title=draft.title,
            draft_date=draft.date,
            draft_time=draft.time,
            anchor_date=anchor_dt.date(),
        )
        if reason:
            draft.reject_reason = reason

    # Additional guardrail for recap-style posts: if the post looks like a recent recap,
    # and the "future mention" is too generic (e.g. "тематический концерт"), skip it.
    recap_reason = _looks_like_recent_recap_with_past_date(
        source_text=combined_text,
        anchor_date=anchor_dt.date(),
    )
    if recap_reason:
        for draft in drafts:
            if (draft.reject_reason or "").strip():
                continue
            if not _looks_like_vague_teaser_title(draft.title):
                continue
            try:
                d_obj = date.fromisoformat((draft.date or "").split("..", 1)[0].strip())
            except Exception:
                continue
            if d_obj < anchor_dt.date():
                continue
            draft.reject_reason = recap_reason

    drafts = _maybe_collapse_program_schedule_drafts(drafts)
    drafts = _collapse_same_post_exact_drafts(drafts)

    return drafts, festival_payload


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
) -> tuple[EventDraft, dict[str, Any] | None]:
    drafts, festival_payload = await build_event_drafts_from_vk(
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
    return drafts[0], festival_payload


async def build_event_drafts(
    text: str,
    *,
    photos: Sequence[str] | None = None,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    publish_ts: datetime | int | float | None = None,
    event_ts_hint: int | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: list[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    rate_limit_max_wait_sec: float | int | str | None = None,
    prefilter_obvious_non_events: bool = False,
    db: Database,
) -> tuple[list[EventDraft], dict[str, Any] | None]:
    """Download posters, run OCR and return event drafts for a VK post.

    Returns a tuple ``(drafts, festival_payload)`` mirroring
    :func:`build_event_drafts_from_vk`.
    """
    timings_on = (os.getenv("PIPELINE_TIMINGS") or "").strip().lower() in {"1", "true", "yes", "on"}
    timing: dict[str, float] = {}
    def _tmark(name: str, sec: float) -> None:
        if timings_on:
            timing[name] = float(sec)

    t_all = time.monotonic()
    t0 = time.monotonic()
    photo_bytes = await _download_photo_media(photos or [])
    _tmark("download_photos", time.monotonic() - t0)
    poster_items: list[PosterMedia] = []
    ocr_tokens_spent = 0
    ocr_tokens_remaining: int | None = None
    ocr_limit_notice: str | None = None
    hash_to_indices: dict[str, list[int]] | None = None
    ocr_disabled = (os.getenv("POSTER_OCR_DISABLED") or "").strip().lower() in {"1", "true", "yes", "on"}
    if photo_bytes:
        hash_to_indices = {}
        for idx, (payload, _name) in enumerate(photo_bytes):
            digest = hashlib.sha256(payload).hexdigest()
            hash_to_indices.setdefault(digest, []).append(idx)
        t0 = time.monotonic()
        poster_items, catbox_msg = await process_media(
            photo_bytes, need_catbox=True, need_ocr=False
        )
        _tmark("upload_catbox", time.monotonic() - t0)
        ocr_source = source_name or "vk"
        ocr_log_context = {"event_id": None, "source": ocr_source}
        ocr_results: list[poster_ocr.PosterOcrCache] = []
        if ocr_disabled:
            logging.info("vk.build_event_draft OCR disabled via POSTER_OCR_DISABLED=1", extra=ocr_log_context)
        else:
            try:
                t0 = time.monotonic()
                (
                    ocr_results,
                    ocr_tokens_spent,
                    ocr_tokens_remaining,
                ) = await poster_ocr.recognize_posters(
                    db, photo_bytes, log_context=ocr_log_context
                )
                _tmark("ocr_posters", time.monotonic() - t0)
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
            except Exception as exc:
                # OCR is a best-effort enrichment. Do not fail the entire VK post import
                # when OCR backend is temporarily unavailable (network/provider errors).
                logging.warning(
                    "vk.build_event_draft OCR failed: %s",
                    exc,
                    extra=ocr_log_context,
                    exc_info=True,
                )
                ocr_results = []
                ocr_limit_notice = "OCR недоступен: ошибка распознавания, распознавание пропущено."
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
        if not ocr_disabled:
            _, _, ocr_tokens_remaining = await poster_ocr.recognize_posters(
                db, [], log_context=ocr_log_context
            )
    drafts, festival_payload = await build_event_drafts_from_vk(
        text,
        source_name=source_name,
        location_hint=location_hint,
        default_time=default_time,
        default_ticket_link=default_ticket_link,
        operator_extra=operator_extra,
        publish_ts=publish_ts,
        event_ts_hint=event_ts_hint,
        festival_names=festival_names,
        festival_alias_pairs=festival_alias_pairs,
        festival_hint=festival_hint,
        poster_media=poster_items,
        ocr_tokens_spent=ocr_tokens_spent,
        ocr_tokens_remaining=ocr_tokens_remaining,
        rate_limit_max_wait_sec=rate_limit_max_wait_sec,
        prefilter_obvious_non_events=prefilter_obvious_non_events,
    )
    _tmark("build_drafts_from_vk_total", time.monotonic() - t_all)
    for draft in drafts:
        draft.ocr_limit_notice = ocr_limit_notice
    if timings_on:
        try:
            logger.info(
                "timing vk_intake_build_drafts photos=%s posters=%s stages=%s",
                len(photos or []),
                len(poster_items or []),
                {k: round(v, 3) for k, v in sorted(timing.items())},
            )
        except Exception:
            pass
    return drafts, festival_payload


async def build_event_draft(
    text: str,
    *,
    photos: Sequence[str] | None = None,
    source_name: str | None = None,
    location_hint: str | None = None,
    default_time: str | None = None,
    default_ticket_link: str | None = None,
    operator_extra: str | None = None,
    publish_ts: datetime | int | float | None = None,
    event_ts_hint: int | None = None,
    festival_names: list[str] | None = None,
    festival_alias_pairs: list[tuple[str, int]] | None = None,
    festival_hint: bool = False,
    db: Database,
) -> tuple[EventDraft, dict[str, Any] | None]:
    drafts, festival_payload = await build_event_drafts(
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
    return drafts[0], festival_payload


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


def _looks_like_vague_teaser_title(title: str | None) -> bool:
    raw = (title or "").strip()
    if not raw:
        return False
    t = unicodedata.normalize("NFKC", raw)
    t = t.replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # Drop leading emoji/prefix noise.
    t = re.sub(r"^[^a-zа-яё0-9]+", "", t, flags=re.IGNORECASE).strip()
    words = re.findall(r"[а-яё]+", t, flags=re.IGNORECASE)
    words = [w.casefold().replace("ё", "е") for w in words if w]
    if len(words) != 2:
        return False
    return words[0].startswith("тематичес") and words[1] == "концерт"


def _looks_like_recent_recap_with_past_date(
    *,
    source_text: str | None,
    anchor_date: date,
) -> str | None:
    """Detect recap-style posts that mention a very recent past date.

    We use this as a context signal: such posts often contain a small "teaser" for a
    different future event without a proper title. Creating a standalone event from
    that teaser is high-risk.
    """
    raw_text = (source_text or "").strip()
    if not raw_text:
        return None

    def _norm(s: str) -> str:
        s2 = unicodedata.normalize("NFKC", s)
        s2 = s2.replace("\xa0", " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        return s2.casefold().replace("ё", "е")

    text_norm = _norm(raw_text)
    recap_markers = (
        "позавчера",
        "вчера",
        "прош",
        "состоял",
        "состоялась",
        "состоялось",
        "вновь",
        "исполнил",
        "исполнила",
        "исполн",
        "прозвуч",
    )
    if not any(tok in text_norm for tok in recap_markers):
        return None

    # Require a recent past date mention to avoid false positives on generic
    # "история/справка" posts.
    max_past = timedelta(days=14)
    month_names_patt = "|".join(sorted(MONTHS_RU.keys(), key=len, reverse=True))
    date_mentions: list[date] = []

    for m in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-]((?:19|20)\d{2}))?\b", text_norm):
        try:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else int(anchor_date.year)
        except Exception:
            continue
        d = _safe_construct_date(year, month, day)
        if d:
            date_mentions.append(d)

    for m in re.finditer(
        rf"\b(\d{{1,2}})\s+({month_names_patt})(?:\s+((?:19|20)\d{{2}}))?\b",
        text_norm,
        flags=re.IGNORECASE,
    ):
        try:
            day = int(m.group(1))
            mon = _month_from_token(m.group(2))
            if not mon:
                continue
            year = int(m.group(3)) if m.group(3) else int(anchor_date.year)
        except Exception:
            continue
        d = _safe_construct_date(year, int(mon), day)
        if d:
            date_mentions.append(d)

    if not date_mentions:
        return None

    has_recent_past = any(
        (d < anchor_date and (anchor_date - d) <= max_past)
        for d in date_mentions
    )
    if not has_recent_past:
        return None

    return (
        "Низкая уверенность: пост похож на отчёт о недавнем прошедшем событии, "
        "а будущее упоминание слишком общее (нет явного названия)."
    )


def _maybe_rollover_llm_iso_date(
    raw_date: str | None,
    *,
    anchor_date: date,
    has_explicit_year_in_text: bool,
) -> str | None:
    """Roll over an LLM-produced ISO date to the next year when the year is implicit.

    VK posts often contain "DD month" without a year. The parser may resolve it to
    the current year; if that date is in the far past relative to the publish date,
    it likely refers to the next year's event.

    IMPORTANT: do NOT roll over *recent* past mentions (recaps), otherwise we create
    bogus future events (e.g. 12 Feb -> 12 Feb next year).
    """
    rd = (raw_date or "").strip()
    if not rd:
        return raw_date
    if has_explicit_year_in_text:
        return raw_date
    normalized = normalize_implicit_iso_date_to_anchor(
        rd,
        anchor_date=anchor_date,
        recent_past_days=int(RECENT_PAST_THRESHOLD.days),
    )
    return normalized or raw_date


def _parse_iso_date_maybe(value: str | None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = raw.split("..", 1)[0].strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


def _parse_iso_date_range(value: str | None, *, end_value: str | None) -> tuple[date | None, date | None]:
    raw = (value or "").strip()
    if not raw:
        return None, None
    if ".." in raw:
        left, right = raw.split("..", 1)
        start = _parse_iso_date_maybe(left)
        end = _parse_iso_date_maybe(right) or start
    else:
        start = _parse_iso_date_maybe(raw)
        end = start
    # Prefer explicit end_date field when present.
    end_override = _parse_iso_date_maybe(end_value)
    if end_override:
        end = end_override
    return start, end


def _looks_like_recap_title_copied_to_future_event(
    *,
    source_text: str | None,
    title: str | None,
    draft_date: str | None,
    draft_time: str | None,
    anchor_date: date,
) -> str | None:
    """Detect a common VK pattern: recap of a past event + mention of a future event without a name.

    Example (real-world): "12 февраля ... вновь исполнила программу «X» ...", then
    "19 марта ... исполнят тематический концерт" (no explicit title). LLM may
    incorrectly reuse the past program title for the future date.

    Returns a human-readable reject reason when confidence is low.
    """
    raw_text = (source_text or "").strip()
    raw_title = (title or "").strip()
    raw_date = (draft_date or "").strip()
    if not (raw_text and raw_title and raw_date):
        return None
    try:
        d_obj = date.fromisoformat(raw_date.split("..", 1)[0].strip())
    except Exception:
        return None

    def _norm(s: str) -> str:
        s2 = unicodedata.normalize("NFKC", s)
        s2 = s2.replace("\xa0", " ")
        for ch in ("«", "»", "“", "”", "„", "‟", '"', "'"):
            s2 = s2.replace(ch, " ")
        s2 = re.sub(r"\s+", " ", s2).strip()
        s2 = s2.casefold().replace("ё", "е")
        return s2

    text_norm = _norm(raw_text)
    title_norm = _norm(raw_title)
    title_norm = re.sub(r"[^\w\s]+", " ", title_norm, flags=re.UNICODE)
    title_norm = re.sub(r"\s+", " ", title_norm).strip()
    if len(title_norm) < 6:
        return None

    title_positions: list[int] = []
    start = 0
    while True:
        idx = text_norm.find(title_norm, start)
        if idx < 0:
            break
        title_positions.append(int(idx))
        start = idx + max(1, len(title_norm))
        if len(title_positions) >= 6:
            break
    if not title_positions:
        return None

    # Extract date mentions with rough positions (in the normalized text).
    month_names_patt = "|".join(sorted(MONTHS_RU.keys(), key=len, reverse=True))
    date_mentions: list[tuple[date, int]] = []

    # Numeric: dd.mm, dd/mm, dd-mm (year optional).
    for m in re.finditer(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-]((?:19|20)\d{2}))?\b", text_norm):
        try:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else int(anchor_date.year)
        except Exception:
            continue
        d = _safe_construct_date(year, month, day)
        if d:
            date_mentions.append((d, int(m.start())))

    # Text: "12 февраля" (+ optional year).
    for m in re.finditer(
        rf"\b(\d{{1,2}})\s+({month_names_patt})(?:\s+((?:19|20)\d{{2}}))?\b",
        text_norm,
        flags=re.IGNORECASE,
    ):
        try:
            day = int(m.group(1))
            mon = _month_from_token(m.group(2))
            if not mon:
                continue
            year = int(m.group(3)) if m.group(3) else int(anchor_date.year)
        except Exception:
            continue
        d = _safe_construct_date(year, int(mon), day)
        if d:
            date_mentions.append((d, int(m.start())))

    if len(date_mentions) < 2:
        return None

    draft_date_positions = [pos for d, pos in date_mentions if (d.day, d.month) == (d_obj.day, d_obj.month)]
    if not draft_date_positions:
        return None

    def _min_dist(pos: int, others: list[int]) -> int:
        return int(min(abs(pos - x) for x in others)) if others else 10**9

    def _no_sentence_boundary_between(a: int, b: int) -> bool:
        lo = max(0, min(a, b))
        hi = min(len(text_norm), max(a, b))
        if hi <= lo:
            return True
        between = text_norm[lo:hi]
        # Treat sentence ends and explicit newlines as hard boundaries.
        return not bool(re.search(r"[.!?]\s|[\r\n]", between))

    # Find the date mention closest to the title (likely the date the title belongs to).
    near_title_date, near_title_pos = min(
        date_mentions,
        key=lambda item: _min_dist(item[1], title_positions),
    )

    # If the title appears near the draft's date mention, it's probably fine.
    close_to_draft_date = False
    for pos in title_positions:
        # Require that the mention is in the same sentence/fragment; otherwise a recap paragraph
        # followed by a new-sentence future date can look "close" in short posts.
        nearest_dpos = min(draft_date_positions, key=lambda p: abs(p - pos))
        if abs(pos - nearest_dpos) <= 220 and _no_sentence_boundary_between(pos, nearest_dpos):
            close_to_draft_date = True
            break
    if close_to_draft_date:
        return None

    # Strong signal: title is anchored to a past date in the same post, while the extracted
    # draft date is in the future (relative to the post publish date).
    if near_title_date == d_obj:
        return None
    if not (near_title_date < anchor_date <= d_obj):
        return None

    # Check that the title lives in a "recap" window (past tense / "вновь исполнил").
    # This reduces false positives where multiple future dates are listed.
    nearest_title_pos = min(title_positions, key=lambda p: abs(p - near_title_pos))
    win = text_norm[max(0, nearest_title_pos - 220) : min(len(text_norm), nearest_title_pos + 220)]
    recap_markers = (
        "позавчера",
        "вчера",
        "прош",
        "состоял",
        "состоялась",
        "состоялось",
        "вновь",
        "исполнил",
        "исполнила",
        "исполн",
        "прозвуч",
    )
    if not any(tok in win for tok in recap_markers):
        return None

    # If time is explicitly present (not placeholder), allow — the announcement is more likely grounded.
    t_raw = (draft_time or "").strip().replace(".", ":")
    if t_raw and t_raw not in {"00:00", "0:00"}:
        return None

    near_title_iso = near_title_date.isoformat()
    return (
        "Низкая уверенность: заголовок выглядит как название прошедшего концерта "
        f"({near_title_iso}), а анонс на {d_obj.isoformat()} не содержит явного названия."
    )


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


def _orthodox_easter_gregorian(target_year: int) -> date:
    """Return Orthodox Easter (Pascha) date in Gregorian calendar.

    Algorithm: compute Julian Easter (Meeus Julian algorithm) and convert to
    Gregorian by adding the calendar offset for the given year.
    """
    a = target_year % 4
    b = target_year % 7
    c = target_year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31  # 3=March, 4=April (Julian)
    day = ((d + e + 114) % 31) + 1

    julian_easter = date(target_year, month, day)
    gregorian_delta_days = target_year // 100 - target_year // 400 - 2
    return julian_easter + timedelta(days=gregorian_delta_days)


def _movable_holiday_date_range(token: str, target_year: int) -> tuple[date | None, date | None]:
    key = token.split(":", 1)[1].strip().casefold()
    if key in {"maslenitsa", "масленица"}:
        easter = _orthodox_easter_gregorian(target_year)
        start = easter - timedelta(days=55)
        end = easter - timedelta(days=49)
        return start, end
    return None, None


def _holiday_date_range(record: Any, target_year: int) -> tuple[str | None, str | None]:
    raw = (record.date or "").strip()
    if not raw:
        return None, None

    if raw.casefold().startswith("movable:"):
        start, end = _movable_holiday_date_range(raw, target_year)
        start_iso = start.isoformat() if start else None
        end_iso = end.isoformat() if end else None
        return start_iso, end_iso

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
    wait_for_telegraph_url: bool = True,
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
    schedule_event_update_tasks = main_mod.schedule_event_update_tasks
    rebuild_fest_nav_if_changed = main_mod.rebuild_fest_nav_if_changed
    normalize_event_type = getattr(main_mod, "normalize_event_type", None)

    from smart_event_update import EventCandidate, PosterCandidate, smart_event_update

    if (getattr(draft, "reject_reason", None) or "").strip():
        # Keep the error string compatible with vk_auto_queue handler that treats
        # "smart_update rejected:" as an expected rejection (not a technical failure).
        raise RuntimeError(
            "smart_update rejected: rejected_low_confidence "
            f"reason={str(getattr(draft, 'reject_reason', '')).strip()}"
        )

    posters = _build_smart_update_posters(
        draft,
        photos=photos,
        poster_cls=PosterCandidate,
    )

    normalized_event_type = (
        normalize_event_type(
            draft.title or "",
            f"{draft.description or ''}\n{draft.source_text or ''}".strip(),
            draft.event_type,
        )
        if callable(normalize_event_type)
        else (draft.event_type or None)
    )

    vk_source_chat_id, vk_source_message_id = _vk_wall_source_ids_from_url(source_post_url)

    candidate = EventCandidate(
        source_type="vk",
        source_url=source_post_url,
        source_chat_id=vk_source_chat_id,
        source_message_id=vk_source_message_id,
        source_text=draft.source_text or draft.title,
        title=draft.title,
        # Never default missing date/time to "today" or "00:00": it creates pseudo-events.
        # Let Smart Update reject/skip incomplete drafts (vk_auto_queue treats it as an expected skip).
        date=draft.date or None,
        time=draft.time or "",
        time_is_default=bool(getattr(draft, "time_is_default", False)),
        end_date=draft.end_date or None,
        festival=draft.festival or None,
        location_name=draft.venue or "",
        location_address=draft.location_address or None,
        city=draft.city or None,
        ticket_link=(draft.links[0] if draft.links else None),
        ticket_price_min=draft.ticket_price_min,
        ticket_price_max=draft.ticket_price_max,
        event_type=normalized_event_type,
        emoji=draft.emoji or None,
        is_free=bool(draft.is_free),
        pushkin_card=bool(draft.pushkin_card),
        search_digest=draft.search_digest,
        raw_excerpt=draft.description or "",
        posters=posters,
    )

    update_result = await smart_event_update(
        db,
        candidate,
        check_source_url=False,
    )
    if str(getattr(update_result, "status", "") or "").startswith("rejected_"):
        raise RuntimeError(
            f"smart_update rejected: {getattr(update_result, 'status', None)} "
            f"reason={getattr(update_result, 'reason', None)}"
        )
    if not getattr(update_result, "event_id", None):
        raise RuntimeError(
            "smart_update returned no event_id: "
            f"status={getattr(update_result, 'status', None)} "
            f"reason={getattr(update_result, 'reason', None)}"
        )
    async with db.get_session() as session:
        saved = (
            await session.get(Event, update_result.event_id)
            if update_result.event_id
            else None
        )
    if saved is None:
        raise RuntimeError(
            "smart_update failed to persist event: "
            f"event_id={getattr(update_result, 'event_id', None)} "
            f"status={getattr(update_result, 'status', None)} "
            f"reason={getattr(update_result, 'reason', None)}"
        )
    text_length = len(saved.title or "") + len(saved.description or "") + len(saved.source_text or "")
    logging.info(
        "event_topics_classify eid=%s text_len=%d topics=%s manual=%s",
        saved.id,
        text_length,
        list(saved.topics or []),
        bool(saved.topics_manual),
    )
    logging.info(
        "persist_event_and_pages: source_post_url=%s", saved.source_post_url
    )

    nav_update_needed = False
    if saved.festival:
        parts = [p.strip() for p in (saved.date or "").split("..") if p.strip()]
        start_str = parts[0] if parts else None
        end_str = parts[-1] if len(parts) > 1 else None
        explicit_end = bool(saved.end_date) or len(parts) > 1
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
                    if start_str and explicit_end:
                        if (
                            festival.start_date is None
                            or start_str < festival.start_date
                        ):
                            festival.start_date = start_str
                            changed = True
                    if end_str:
                        if (
                            festival.end_date is None
                            or (explicit_end and end_str > festival.end_date)
                        ):
                            festival.end_date = end_str
                            changed = True
                    if changed:
                        session.add(festival)
                        await session.commit()
                        nav_update_needed = True
    if nav_update_needed:
        await rebuild_fest_nav_if_changed(db)
    if update_result.status in ("skipped_nochange", "skipped_same_source_url"):
        await schedule_event_update_tasks(db, saved)

    if wait_for_telegraph_url:
        # Wait for Telegraph URL to become available (async job). Callers that
        # already run inline Telegraph jobs can skip this extra wait.
        start_wait = time.time()
        for _ in range(20):  # Wait up to 10 seconds
            async with db.get_session() as session:
                saved = await session.get(Event, saved.id)
            if saved.telegraph_url:
                elapsed = time.time() - start_wait
                logging.info("persist_event_and_pages: telegraph_url appeared after %.2fs", elapsed)
                break
            await asyncio.sleep(0.5)

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
        smart_status=getattr(update_result, "status", None),
        smart_created=bool(getattr(update_result, "created", False)),
        smart_merged=bool(getattr(update_result, "merged", False)),
        smart_added_posters=int(getattr(update_result, "added_posters", 0) or 0),
    )


def _build_smart_update_posters(
    draft: EventDraft,
    *,
    photos: Sequence[str] | None,
    poster_cls: type,
) -> list[object]:
    """Build Smart Update poster candidates with VK URL fallback.

    Catbox can be disabled in tests/live runs. In this case we still want event
    posters rendered on Telegraph by passing original VK media URLs through the
    same `catbox_url` field consumed by the unified event-page pipeline.
    """
    poster_urls = [m.catbox_url for m in draft.poster_media if m.catbox_url]
    photo_urls = poster_urls or list(photos or [])
    posters: list[object] = []
    for idx, item in enumerate(draft.poster_media):
        url = (item.catbox_url or "").strip()
        if not url and idx < len(photo_urls):
            url = str(photo_urls[idx] or "").strip()
        supabase_url = (item.supabase_url or "").strip() or (
            url if is_supabase_storage_url(url) else None
        )
        catbox_url = url if url and not is_supabase_storage_url(url) else None
        posters.append(
            poster_cls(
                catbox_url=catbox_url,
                supabase_url=supabase_url,
                sha256=item.digest,
                phash=None,
                ocr_text=item.ocr_text,
                ocr_title=item.ocr_title,
                prompt_tokens=int(getattr(item, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(item, "completion_tokens", 0) or 0),
                total_tokens=int(getattr(item, "total_tokens", 0) or 0),
            )
        )
    if not posters and photo_urls:
        posters = [
            poster_cls(
                catbox_url=(u if u and not is_supabase_storage_url(u) else None),
                supabase_url=(u if is_supabase_storage_url(u) else None),
            )
            for u in (str(url).strip() for url in photo_urls)
        ]
    return posters


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
    drafts, _ = await build_event_drafts(
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

    try:
        from source_parsing.post_metrics import compute_age_day as _compute_age_day
        from source_parsing.post_metrics import normalize_age_day as _normalize_age_day
        from source_parsing.post_metrics import upsert_vk_post_metric as _upsert_vk_post_metric
    except Exception:  # pragma: no cover - optional helper
        _compute_age_day = None
        _normalize_age_day = None
        _upsert_vk_post_metric = None

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
                    try:
                        collected_ts = int(time.time())
                        age_raw = (
                            _compute_age_day(published_ts=int(ts), collected_ts=int(collected_ts))
                            if _compute_age_day
                            else None
                        )
                        age_day = _normalize_age_day(age_raw) if _normalize_age_day else age_raw
                        if (
                            _upsert_vk_post_metric
                            and isinstance(age_day, int)
                            and age_day >= 0
                        ):
                            views = post.get("views")
                            likes = post.get("likes")
                            if isinstance(views, int) or isinstance(likes, int):
                                await _upsert_vk_post_metric(
                                    db,
                                    group_id=int(gid),
                                    post_id=int(pid),
                                    age_day=int(age_day),
                                    source_url=miss_url,
                                    post_ts=int(ts),
                                    views=int(views) if isinstance(views, int) else None,
                                    likes=int(likes) if isinstance(likes, int) else None,
                                    collected_ts=int(collected_ts),
                                )
                    except Exception:
                        logging.warning(
                            "vk.crawl.metrics persist failed gid=%s post_id=%s",
                            gid,
                            pid,
                            exc_info=True,
                        )
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
