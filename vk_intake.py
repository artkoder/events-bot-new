from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Sequence
from datetime import datetime, timedelta, timezone

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

NUM_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?\b")
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
    return any(p.search(text) for p in COMPILED_DATE_PATTERNS)


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

    def is_valid_date_sequence(parts: List[str]) -> bool:
        if len(parts) < 3:
            return False
        try:
            day = int(parts[0])
            month = int(parts[1])
        except ValueError:
            return False
        if not (1 <= day <= 31 and 1 <= month <= 12):
            return False
        year_part = parts[2]
        if len(year_part) < 2:
            return False
        return True

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
        if DATE_RANGE_RE.search(trimmed):
            result.append(trimmed)
        else:
            parts = re.findall(r"\d+", trimmed)
            if is_valid_date_sequence(parts):
                result.append(trimmed)
            else:
                normalized = re.sub(r"\d", "x", trimmed)
                result.append(normalized)
        pos = start + trimmed_end
    result.append(text[pos:])
    return "".join(result)


def extract_event_ts_hint(
    text: str,
    default_time: str | None = None,
    *,
    tz: timezone | None = None,
    publish_ts: datetime | int | float | None = None,
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
    text_low = normalize_phone_candidates(text.lower())

    day = month = year = None
    m = None
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
        if PHONE_LIKE_RE.match(candidate.group(0)):
            context_start = max(0, start - 30)
            context_end = min(len(text_low), candidate.end() + 10)
            context_slice = text_low[context_start:context_end]
            skip_candidate = False
            trailing_idx = candidate.end()
            trailing_chars = " \t\r\n.;:!?()[]{}«»\"'—–-"
            while trailing_idx < len(text_low) and text_low[trailing_idx] in trailing_chars:
                trailing_idx += 1
            has_event_tail = False
            next_alpha_word = None
            following_is_phone_tail = False
            remainder = ""
            skip_due_to_action_tail = False
            skip_due_to_location_tail = False
            if trailing_idx < len(text_low):
                remainder = text_low[trailing_idx:]
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
        break

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
            elif dow_matches:
                return None
            elif WEEKEND_RE.search(text_low):
                days_ahead = (5 - now.weekday()) % 7
                dt = now + timedelta(days=days_ahead)
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
            backfill = force_backfill or last_seen_ts == 0 or (
                idle_h is not None and idle_h >= VK_CRAWL_BACKFILL_AFTER_IDLE_H
            )

            posts: list[dict] = []
            pages_loaded = 0
            safety_cap_triggered = False
            hard_cap_triggered = False
            reached_cursor_overlap = False

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

            pages_per_group.append(pages_loaded)

            group_posts = 0
            group_matched = 0
            max_ts, max_pid = last_seen_ts, last_post_id
            deep_backfill_scheduled = False

            for post in posts:
                ts = post["date"]
                pid = post["post_id"]
                if ts < last_seen_ts or (ts == last_seen_ts and pid <= last_post_id):
                    continue
                stats["posts_scanned"] += 1
                group_posts += 1
                post_text = post.get("text", "")
                photos = post.get("photos", []) or []
                blank_single_photo = not post_text.strip() and len(photos) == 1

                history_hit = False
                if blank_single_photo:
                    matched_kw_value = OCR_PENDING_SENTINEL
                    has_date_value = 0
                    event_ts_hint = None
                else:
                    history_hit = detect_historical_context(post_text)
                    kw_ok, kws = match_keywords(post_text)
                    has_date = detect_date(post_text)
                    if kw_ok and has_date:
                        event_ts_hint = extract_event_ts_hint(
                            post_text, default_time, publish_ts=ts
                        )
                        if event_ts_hint is None or event_ts_hint < int(time.time()) + 2 * 3600:
                            continue
                        matched_kw_value = ",".join(kws)
                        has_date_value = int(has_date)
                    elif history_hit:
                        matched_kw_value = HISTORY_MATCHED_KEYWORD
                        has_date_value = int(has_date)
                        event_ts_hint = None
                    else:
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
                                matched_kw_value,
                                has_date_value,
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

            next_cursor_ts = max_ts
            next_cursor_pid = max_pid
            cursor_updated_at = now_ts
            if hard_cap_triggered and max_ts > 0 and not reached_cursor_overlap:
                deep_backfill_scheduled = True
                next_cursor_ts = last_seen_ts
                next_cursor_pid = last_post_id
                idle_threshold = VK_CRAWL_BACKFILL_AFTER_IDLE_H * 3600
                cursor_updated_at = max(0, now_ts - idle_threshold - 60)
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
            
            async with db.raw_conn() as conn:
                await conn.execute(
                    "INSERT OR REPLACE INTO vk_crawl_cursor(group_id, last_seen_ts, last_post_id, updated_at) VALUES(?,?,?,?)",
                    (gid, next_cursor_ts, next_cursor_pid, cursor_updated_at),
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
    return stats
