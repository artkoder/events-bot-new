from __future__ import annotations

import asyncio
from calendar import monthrange
import math
import json
import logging
import os
import time
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Iterable, Sequence

from sqlalchemy import and_, delete, or_, select

from db import Database
from models import Event, EventPoster, EventSource, EventSourceFact, PosterOcrCache
from sections import MONTHS_RU

logger = logging.getLogger(__name__)

_HALL_HINT_RE = re.compile(
    r"\b(зал|аудитория|лекторий|сцена|фойе|этаж|корпус)\b\s+([^\s,.;:]+)(?:\s+([^\s,.;:]+))?(?:\s+([^\s,.;:]+))?",
    re.IGNORECASE,
)
# Telegram custom emoji placeholders can land in PUA (Private Use Area) ranges.
# Keep this broader than just BMP to avoid "tofu" boxes on Telegraph pages.
_PRIVATE_USE_RE = re.compile(r"[\uE000-\uF8FF\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060]")

# Ticket giveaways must not become standalone "events", but real announcements that
# include a giveaway block should still import/merge the underlying event facts.
_GIVEAWAY_RE = re.compile(
    r"\b(розыгрыш|разыгрыва\w*|розыгра\w*|выигра\w*|конкурс|giveaway)\b",
    re.IGNORECASE,
)
_TICKETS_RE = re.compile(
    r"\b(билет\w*|пригласительн\w*|абонемент\w*)\b",
    re.IGNORECASE,
)

# Lines that are usually giveaway mechanics ("subscribe/repost/comment") rather than event facts.
_GIVEAWAY_LINE_RE = re.compile(
    r"\b("
    r"услови\w*|"
    r"участв\w*|"
    r"подпиш\w*|"
    r"репост\w*|"
    r"коммент\w*|"
    r"отмет\w*|"
    r"лайк\w*|"
    r"победител\w*|"
    r"итог\w*|"
    r"розыгрыш|разыгрыва\w*|розыгра\w*|"
    r"конкурс|giveaway|"
    r"приз\w*"
    r")\b",
    re.IGNORECASE,
)

_GIVEAWAY_MECHANICS_RE = re.compile(
    r"\b("
    r"услови\w*|"
    r"участв\w*|"
    r"подпиш\w*|"
    r"репост\w*|"
    r"коммент\w*|"
    r"отмет\w*|"
    r"лайк\w*|"
    r"победител\w*|"
    r"итог\w*|"
    r"приз\w*"
    r")\b",
    re.IGNORECASE,
)

_EVENT_SIGNAL_RE = re.compile(
    r"\b("
    r"спектакл\w*|"
    r"концерт\w*|"
    r"выставк\w*|"
    r"лекци\w*|"
    r"показ\w*|"
    r"встреч\w*|"
    r"мастер-?класс\w*|"
    r"презентац\w*|"
    r"экскурс\w*|"
    r"перформанс\w*|"
    r"кино\w*|фильм\w*"
    r")\b",
    re.IGNORECASE,
)

# Promotions are often mixed into real event announcements. Product requirement:
# strip purely promotional fragments, but keep actual event facts (date/time/place/contacts).
_PROMO_STRIP_RE = re.compile(
    r"\b("
    r"акци(?:я|и|ю|ях)|"
    r"скидк\w*|"
    r"промокод\w*|"
    r"спецпредложен\w*|"
    r"бонус\w*|"
    r"кэшбек\w*|кэшбэк\w*|кэшбэ\w*|"
    r"подарок\w*|"
    r"сертификат\w*"
    r")\b",
    re.IGNORECASE,
)
_CONGRATS_RE = re.compile(
    r"\b(поздравля\w*|с\s+дн[её]м\s+рождени\w*|юбиле\w*)\b",
    re.IGNORECASE,
)
_CONGRATS_CONTEXT_RE = re.compile(
    r"\b(ближайш\w*|спектакл\w*|концерт\w*|мероприят\w*|событи\w*)\b",
    re.IGNORECASE,
)

_CHANNEL_PROMO_STRIP_RE = re.compile(
    r"(?i)"
    r"(?=.*(?:t\.me/|telegram|телеграм))"
    r"(?=.*\b(?:канал\w*|чат\w*|групп\w*)\b)"
    r"(?=.*(?:анонс\w*|афиш\w*|подпис\w*|следит\w*|информац\w*\s+о\s+(?:событи\w*|мероприят\w*)))"
)

_POSTER_PROMO_RE = re.compile(
    r"\b(акци(?:я|и|ю|ях)|скидк\w*|промокод\w*|купон\w*|sale)\b|%",
    re.IGNORECASE,
)

SMART_UPDATE_LLM = os.getenv("SMART_UPDATE_LLM", "gemma").strip().lower()
SMART_UPDATE_LLM_DISABLED = SMART_UPDATE_LLM in {"off", "none", "disabled", "0"}
# Product requirement: Smart Update uses Gemma as the primary model.
# OpenAI (4o) is allowed only as a *fallback* when Gemma calls fail/unavailable.
if not SMART_UPDATE_LLM_DISABLED and SMART_UPDATE_LLM != "gemma":
    logger.warning(
        "smart_update: SMART_UPDATE_LLM=%r is not supported; forcing 'gemma' (4o is fallback-only)",
        SMART_UPDATE_LLM,
    )
    SMART_UPDATE_LLM = "gemma"
SMART_UPDATE_MODEL = os.getenv(
    "SMART_UPDATE_MODEL",
    "gemma-3-27b-it",
).strip()
if not SMART_UPDATE_MODEL or "gemma" not in SMART_UPDATE_MODEL.lower():
    logger.warning(
        "smart_update: SMART_UPDATE_MODEL=%r is not a Gemma model; forcing 'gemma-3-27b-it'",
        SMART_UPDATE_MODEL,
    )
    SMART_UPDATE_MODEL = "gemma-3-27b-it"
SMART_UPDATE_YO_RULE = (
    "Уважай букву «ё»: если слово в норме пишется через «ё», не заменяй её на «е»."
)
SMART_UPDATE_PRESERVE_LISTS_RULE = (
    "Если в источнике есть нумерованный/маркированный список (песни/треклист/репертуар/программа/пункты формата), "
    "НЕ сворачивай его в одну общую фразу. Перенеси список полностью, сохрани порядок и нумерацию/маркеры. "
    "Названия песен/произведений/имён НЕ перефразируй: копируй дословно."
)
SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE = (
    "Если в источнике есть короткий список программы/репертуара/треклиста/участников/фильмов (2–12 строк подряд), "
    "верни каждый пункт отдельным фактом ДОСЛОВНО, сохрани порядок (можно занять на это большую часть лимита facts). "
    "Пример: после строки `Программа:` идут 5 строк `«Название» — Автор` → верни эти 5 строк как отдельные facts. "
    "Не сворачивай список в общую фразу и не пиши «и другие»."
)
SMART_UPDATE_OPTIONAL_HEADINGS_RULE = (
    "Структурируй description: добавь 1–3 коротких подзаголовка в Markdown формате `### ...` "
    "(например «Программа», «Условия участия», «Состав/участники», «Как добраться»). "
    "Если текст содержит 2+ абзаца — хотя бы 1 подзаголовок обязателен. "
    "Подзаголовки должны быть КОРОТКИМИ (до ~60 символов, без точек/полных предложений) и стоять отдельной строкой. "
    "Не создавай пустых подзаголовков: после строки `### ...` должен идти хотя бы 1 непустой абзац/список/цитата "
    "до следующего подзаголовка. Не ставь два `### ...` подряд без текста между ними. "
    "Если хочешь сделать общий раздел и подпункты — используй вложенный уровень `#### ...` для подпунктов "
    "или оставь только конкретные `### ...` без общего контейнера. "
    "Не помещай целые абзацы в подзаголовки и не используй технические метки вроде `Facts:`/`Факты:`. "
    "Не делай больше 4 подзаголовков."
)
SMART_UPDATE_VISITOR_CONDITIONS_RULE = (
    "Условия участия/посещения (длительность, возраст, максимальный размер группы, формат/что взять/как одеться, "
    "что входит/не входит в оплату, нужен ли отдельный входной билет) считаются фактами о событии и должны попадать "
    "в описание (description). "
    "В description описывай это обычным связным текстом (или под подходящим подзаголовком), а НЕ отдельным блоком «фактов». "
    "Если отдельно формируется список facts (атомарных фактов) — включай туда 1–3 факта про условия участия/посещения. "
    "Для description: не вставляй ссылки/телефоны и не указывай точные цены — пиши нейтрально "
    "(например «оплачивается отдельно», «входной билет нужен дополнительно»). "
    "Для фактов: точная сумма допускается только если она уточняет, что цена относится к части услуги "
    "(например «стоимость экскурсии X; входной билет отдельно»); не более 1 такого факта."
)

# Smart Update description sizing:
# - Telegraph pages can be long, but Telegram UI messages are capped at 4096 chars.
# - Keep a reasonable default and allow overrides via ENV.
def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return min(hi, max(lo, value))


SMART_UPDATE_DESCRIPTION_MAX_CHARS = _env_int(
    # Telegraph pages can hold much longer text; keep a generous default to
    # avoid "too short" descriptions when sources are rich.
    "SMART_UPDATE_DESCRIPTION_MAX_CHARS",
    12000,
    lo=1200,
    hi=20000,
)
SMART_UPDATE_REWRITE_MAX_TOKENS = _env_int(
    # Default kept fairly high: we want a full description, not a short snippet.
    "SMART_UPDATE_REWRITE_MAX_TOKENS", 1400, lo=120, hi=6500
)

# If an event is extracted far into the future, treat poster date mismatches as a high-risk signal.
# Default matches the operator expectation: > 6 months ahead requires more scrutiny.
SMART_UPDATE_FAR_FUTURE_REVIEW_MONTHS = _env_int(
    "SMART_UPDATE_FAR_FUTURE_REVIEW_MONTHS", 6, lo=0, hi=24
)

# Optional: allow light emoji usage in *full* public descriptions (Telegraph/body).
# Must not affect `search_digest` (explicitly emoji-free by prompt).
# Default: enabled (light). Can be disabled via ENV if it turns out noisy.
SMART_UPDATE_DESCRIPTION_EMOJI_MODE = (os.getenv("SMART_UPDATE_DESCRIPTION_EMOJI_MODE", "light") or "").strip().lower()
if SMART_UPDATE_DESCRIPTION_EMOJI_MODE in {"1", "true", "yes", "on", "light"}:
    SMART_UPDATE_DESCRIPTION_EMOJI_MODE = "light"
else:
    SMART_UPDATE_DESCRIPTION_EMOJI_MODE = "off"
SMART_UPDATE_DESCRIPTION_MAX_EMOJIS = _env_int(
    "SMART_UPDATE_DESCRIPTION_MAX_EMOJIS",
    3,
    lo=0,
    hi=8,
)
SMART_UPDATE_DESCRIPTION_EMOJI_ALLOWLIST = (os.getenv("SMART_UPDATE_DESCRIPTION_EMOJI_ALLOWLIST") or "").strip()

# Fact-first mode: build public narrative from extracted facts (not from raw sources).
# Enabled by default; can be disabled for rollback/experiments.
SMART_UPDATE_FACT_FIRST = (os.getenv("SMART_UPDATE_FACT_FIRST", "1") or "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Serialize Smart Update calls within a single bot process to avoid LLM/provider contention
# and to keep operator-visible logs deterministic.
_SMART_UPDATE_LOCK = asyncio.Lock()
SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS = _env_int(
    # How much of candidate.source_text we feed into the rewrite prompt.
    # Telegraph pages can be long; for rewrite we still cap to keep prompts bounded.
    "SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS",
    12000,
    lo=1200,
    hi=20000,
)

# Smart Update merge prompt sizing.
SMART_UPDATE_MERGE_MAX_TOKENS = _env_int(
    "SMART_UPDATE_MERGE_MAX_TOKENS", 1200, lo=300, hi=1600
)
SMART_UPDATE_MERGE_EVENT_DESC_MAX_CHARS = _env_int(
    "SMART_UPDATE_MERGE_EVENT_DESC_MAX_CHARS", 4000, lo=800, hi=20000
)
SMART_UPDATE_MERGE_CANDIDATE_TEXT_MAX_CHARS = _env_int(
    "SMART_UPDATE_MERGE_CANDIDATE_TEXT_MAX_CHARS", 6000, lo=800, hi=20000
)


@dataclass(slots=True)
class PosterCandidate:
    catbox_url: str | None = None
    supabase_url: str | None = None
    supabase_path: str | None = None
    sha256: str | None = None
    phash: str | None = None
    ocr_text: str | None = None
    ocr_title: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class EventCandidate:
    source_type: str
    source_url: str | None
    source_text: str
    title: str | None = None
    date: str | None = None
    time: str | None = None
    # True when time comes from per-source default_time (a low-priority guess).
    time_is_default: bool = False
    end_date: str | None = None
    end_date_is_inferred: bool = False
    festival: str | None = None
    festival_context: str | None = None
    festival_full: str | None = None
    festival_dedup_links: list[str] = field(default_factory=list)
    festival_source: bool | None = None
    festival_series: str | None = None
    location_name: str | None = None
    location_address: str | None = None
    city: str | None = None
    ticket_link: str | None = None
    ticket_price_min: int | None = None
    ticket_price_max: int | None = None
    ticket_status: str | None = None
    event_type: str | None = None
    emoji: str | None = None
    is_free: bool | None = None
    pushkin_card: bool | None = None
    search_digest: str | None = None
    raw_excerpt: str | None = None
    posters: list[PosterCandidate] = field(default_factory=list)
    poster_scope_hashes: list[str] = field(default_factory=list)
    source_chat_username: str | None = None
    source_chat_id: int | None = None
    source_message_id: int | None = None
    creator_id: int | None = None
    trust_level: str | None = None
    metrics: dict[str, Any] | None = None
    links_payload: Any | None = None


@dataclass(slots=True)
class SmartUpdateResult:
    status: str
    event_id: int | None = None
    created: bool = False
    merged: bool = False
    added_posters: int = 0
    added_sources: bool = False
    added_facts: list[str] = field(default_factory=list)
    skipped_conflicts: list[str] = field(default_factory=list)
    reason: str | None = None
    queue_notes: list[str] = field(default_factory=list)


MATCH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventMatch",
        "schema": {
            "type": "object",
            "properties": {
                "match_event_id": {"type": ["integer", "null"]},
                "confidence": {"type": "number"},
                "reason_short": {"type": "string"},
            },
            "required": ["match_event_id", "confidence", "reason_short"],
            "additionalProperties": False,
        },
    },
}

MERGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventMerge",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "search_digest": {"type": ["string", "null"]},
                "ticket_link": {"type": ["string", "null"]},
                "ticket_price_min": {"type": ["integer", "null"]},
                "ticket_price_max": {"type": ["integer", "null"]},
                "ticket_status": {"type": ["string", "null"]},
                "added_facts": {"type": "array", "items": {"type": "string"}},
                "duplicate_facts": {"type": "array", "items": {"type": "string"}},
                "conflict_facts": {"type": "array", "items": {"type": "string"}},
                "skipped_conflicts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["description", "added_facts", "duplicate_facts", "conflict_facts", "skipped_conflicts"],
            "additionalProperties": False,
        },
    },
}

MATCH_SCHEMA = MATCH_RESPONSE_FORMAT["json_schema"]["schema"]
MERGE_SCHEMA = MERGE_RESPONSE_FORMAT["json_schema"]["schema"]


CREATE_BUNDLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventCreateBundle",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "search_digest": {"type": ["string", "null"]},
                "short_description": {"type": ["string", "null"]},
                "facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["description", "facts"],
            "additionalProperties": False,
        },
    },
}

CREATE_BUNDLE_SCHEMA = CREATE_BUNDLE_RESPONSE_FORMAT["json_schema"]["schema"]

MATCH_CREATE_BUNDLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventMatchOrCreateBundle",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["match", "create"]},
                "match_event_id": {"type": ["integer", "null"]},
                "confidence": {"type": "number"},
                "reason_short": {"type": "string"},
                "bundle": {
                    "anyOf": [
                        CREATE_BUNDLE_SCHEMA,
                        {"type": "null"},
                    ]
                },
            },
            "required": ["action", "match_event_id", "confidence", "reason_short", "bundle"],
            "additionalProperties": False,
        },
    },
}

MATCH_CREATE_BUNDLE_SCHEMA = MATCH_CREATE_BUNDLE_RESPONSE_FORMAT["json_schema"]["schema"]


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_LOCATION_NOISE_PREFIXES_RE = re.compile(
    r"^(?:"
    r"кинотеатр|"
    r"бар|bar|"
    r"арт[- ]?пространство|"
    r"пространство"
    r")\s+",
    re.IGNORECASE,
)


def _strip_private_use(text: str | None) -> str | None:
    """Remove PUA chars that may appear as Telegram custom emoji placeholders."""
    if not text:
        return None
    cleaned = _PRIVATE_USE_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n", "\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def _fix_inline_bullet_lists(text: str | None) -> str | None:
    """Convert jammed '•' bullets into a proper multiline Markdown list.

    Gemma sometimes outputs list items on a single line like:
      "-Пункт 1 •Пункт 2 •Пункт 3"
    Telegraph renders this as a single paragraph. We only touch paragraphs that
    clearly look like inline bullet sequences.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    paras = [p for p in re.split(r"\n{2,}", raw) if p.strip()]
    out: list[str] = []
    for para in paras:
        p = para.strip()
        if "\n" in p:
            out.append(p)
            continue
        if "•" not in p:
            out.append(p)
            continue
        if p.count("•") < 1:
            out.append(p)
            continue
        if not p.lstrip().startswith(("-", "•")):
            out.append(p)
            continue
        parts = [x.strip() for x in p.split("•") if x.strip()]
        items: list[str] = []
        for part in parts:
            item = part.lstrip("-•").strip()
            if not item:
                continue
            if item.startswith("-") and len(item) > 1 and not item[1].isspace():
                item = f"- {item[1:].lstrip()}"
            items.append(f"- {item}" if not item.startswith("- ") else item)
        if len(items) >= 2:
            out.append("\n".join(items).strip())
        else:
            out.append(p)
    return "\n\n".join(out).strip() or None


_BULLET_LINE_PREFIX_RE = re.compile(r"(?m)^[ \t]*[•·▪‣⁃]+[ \t]*")


def _normalize_bullet_markers(text: str | None) -> str | None:
    """Normalize non-ASCII bullet markers to Markdown list items.

    Telegram sources often use middle-dot bullets (`·`) which models may not treat
    as a list. Converting them to `- ` increases the chance of list preservation
    in the rewritten Telegraph description.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return None
    normalized = _BULLET_LINE_PREFIX_RE.sub("- ", raw)
    return normalized if normalized.strip() else None


def _extract_small_source_list_items(text: str | None, *, max_items: int = 8) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []
    out: list[str] = []
    for ln in raw.splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        m = re.match(r"^(?:[-*•·]|\d+[.)])\s+(\S.*)$", s)
        if not m:
            continue
        item = (m.group(1) or "").strip()
        if not item:
            continue
        # Keep it safe: do not pull links/handles into description via this deterministic fallback.
        if "http://" in item or "https://" in item or "@" in item:
            continue
        low = item.casefold()
        if low.startswith(("билеты", "вход", "стоимость", "сбор гостей", "начало")):
            continue
        if "подписаться" in low or "зарегистрироваться" in low:
            continue
        out.append(item)
        if len(out) >= max_items:
            break
    return out


def _append_missing_small_list(
    *,
    description: str | None,
    source_text: str | None,
    source_type: str | None,
) -> str | None:
    """Append a small bullet list from the source when the rewrite dropped it.

    Conservative: only targets Telegram-style short lists (2–6 items) and avoids
    links/contacts (they belong to the infoblock).
    """
    desc = (description or "").strip()
    src = (source_text or "").strip()
    if not desc or not src:
        return desc or src or None
    if str(source_type or "").strip().lower() not in {"telegram", "tg"}:
        return desc
    # If the output already contains a list, do not interfere.
    if re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S+", desc):
        return desc

    items = _extract_small_source_list_items(src, max_items=8)
    if not (2 <= len(items) <= 6):
        return desc

    desc_cf = desc.casefold()
    missing = [it for it in items if it.casefold() not in desc_cf]
    if len(missing) < max(2, int(len(items) * 0.5)):
        return desc

    block = "### Что вас ждёт\n" + "\n".join(f"- {it}" for it in items)
    return (desc + "\n\n" + block).strip()


_HEADING_LINE_RE = re.compile(r"(?m)^#{1,6}\s+\S")


def _ensure_minimal_description_headings(text: str | None) -> str | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if _HEADING_LINE_RE.search(raw):
        return raw
    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    if len(blocks) < 2:
        return raw
    # Deterministic fallback: if the model returned 2+ paragraphs but no headings,
    # inject a single neutral subheading to keep Telegraph pages readable.
    heading = "### О событии"
    first = blocks[0]
    rest = "\n\n".join(blocks[1:]).strip()
    if not rest:
        return raw
    if re.match(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S+", first) or first.lstrip().startswith(">"):
        return f"{heading}\n\n{raw}".strip()
    return f"{first}\n\n{heading}\n\n{rest}".strip()


def _normalize_plaintext_paragraphs(text: str | None) -> str | None:
    """Normalize LLM output while preserving paragraph breaks.

    NOTE: event.description is rendered to Telegraph through our Markdown/HTML pipeline
    (see build_source_page_content). So we keep lightweight Markdown that improves
    readability: headings, blockquotes and emphasis.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Drop fenced code blocks (they are almost always accidental/noise for event pages).
    raw = re.sub(r"(?s)```.*?```", "", raw)
    raw = raw.replace("`", "")
    # Replace Markdown links with link text to avoid noisy URL-heavy descriptions.
    raw = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", raw)
    # Keep paragraphs: collapse 3+ newlines into 2.
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    # Normalize spaces without destroying newlines.
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    raw = re.sub(r"\n[ \t]+", "\n", raw)
    raw = raw.strip()
    raw = _fix_inline_bullet_lists(raw) or raw

    # Drop "orphan" headings: cases where the model outputs two headings in a row
    # (or a trailing heading at the end) without any paragraph/list/quote under the first one.
    # This is a pure formatting cleanup: we do not invent any missing text.
    def _drop_orphan_headings(value: str) -> str:
        def _strip_zw(s: str) -> str:
            return _ZERO_WIDTH_RE.sub("", s or "")

        blocks: list[str] = []
        for b in re.split(r"\n{2,}", value.strip()):
            if not b:
                continue
            if not _strip_zw(b).strip():
                continue
            blocks.append(b.strip())
        if len(blocks) < 2:
            return value.strip()
        heading_re = re.compile(r"^\s*(#{1,6})\s+\S")

        def _nonempty_lines(block: str) -> list[str]:
            lines: list[str] = []
            for ln in (block or "").splitlines():
                if not _strip_zw(ln).strip():
                    continue
                lines.append(ln.strip())
            return lines

        def _heading_level(line: str) -> int | None:
            m = heading_re.match(line)
            if not m:
                return None
            return len(m.group(1))

        def _is_heading_only(block: str) -> tuple[bool, int | None]:
            lines = _nonempty_lines(block)
            if len(lines) != 1:
                return False, None
            level = _heading_level(lines[0])
            return (level is not None), level

        def _starts_with_heading(block: str) -> int | None:
            lines = _nonempty_lines(block)
            if not lines:
                return None
            return _heading_level(lines[0])

        out: list[str] = []
        i = 0
        while i < len(blocks):
            cur = blocks[i]
            is_heading, cur_level = _is_heading_only(cur)
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            if is_heading:
                if nxt is None:
                    i += 1
                    continue
                nxt_level = _starts_with_heading(nxt or "")
                if nxt_level is not None and cur_level is not None and nxt_level <= cur_level:
                    i += 1
                    continue
            out.append(cur)
            i += 1
        cleaned = "\n\n".join(out).strip()
        return cleaned or value.strip()

    raw = _drop_orphan_headings(raw)

    # NOTE: We intentionally avoid heuristic paragraph splitting here.
    # Paragraphing is part of LLM output quality. If the model returns a single
    # wall-of-text, we prefer an explicit LLM rewrite pass rather than applying
    # deterministic formatting that can cut semantics at the wrong boundaries.
    return raw or None


def _fix_broken_initial_paragraph_splits(text: str | None) -> str | None:
    """Fix accidental paragraph splits like `... в переводе Н.` + `Любимова.`.

    This is not "formatting"; it's a cleanup for a common LLM artifact that
    makes the text look machine-produced.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if len(paras) < 2:
        return raw

    out: list[str] = []
    i = 0
    while i < len(paras):
        cur = paras[i]
        nxt = paras[i + 1] if i + 1 < len(paras) else None
        if nxt:
            cur_cf = cur.casefold()
            # Join when we ended a paragraph on a single-letter initial and the next
            # paragraph starts with a surname-like token.
            if (
                re.search(r"(?:^|\s)[А-ЯЁA-Z]\.$", cur)
                and re.match(r"^[А-ЯЁ][а-яё]+\b", nxt)
                and ("перевод" in cur_cf or "в переводе" in cur_cf)
            ):
                cur = f"{cur} {nxt}"
                i += 2
                out.append(cur)
                continue
        out.append(cur)
        i += 1

    return "\n\n".join(out).strip() or None


_NEURAL_CLICHE_RE = re.compile(
    r"(?i)\bобеща\w+\s+(?:стать|быть)\b|\bярк\w+\s+событ\w+\b|\bзаметн\w+\s+событ\w+\b|"
    r"\bкультурн\w+\s+жизн\w+\b|\bне\s+остав\w+\s+равнодуш\w+\b|\bнезабываем\w+\b|"
    r"\bуникальн\w+\s+возможн\w+\b|\bэто\s+созда[её]т\b|\bсозда[её]т\s+атмосфер\w*\b|"
    r"\bатмосфер\w+\s+(?:спонтанност|вовлеч[её]нност)\w*\b"
)

_LIST_ITEM_LINE_RE = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+\S")

_DEFAULT_DESCRIPTION_EMOJI_ALLOWLIST = "🎭 🎨 🎵 🎬 🎤 📚 🖼️"


def _description_emoji_prompt_rule() -> str:
    """Prompt rule for optional emoji usage in full descriptions.

    Keep it accessibility-friendly: emojis are optional, few, never replace words,
    and should not be used as list markers.
    """
    if SMART_UPDATE_DESCRIPTION_EMOJI_MODE != "light":
        return "Без эмодзи. "
    max_emojis = int(SMART_UPDATE_DESCRIPTION_MAX_EMOJIS or 0)
    if max_emojis <= 0:
        return "Без эмодзи. "
    allowlist = SMART_UPDATE_DESCRIPTION_EMOJI_ALLOWLIST or _DEFAULT_DESCRIPTION_EMOJI_ALLOWLIST
    allowlist = re.sub(r"\s+", " ", allowlist).strip()
    return (
        f"Эмодзи допускаются умеренно: максимум {max_emojis} на весь текст description; "
        "не ставь несколько эмодзи подряд и не используй их как маркеры списка. "
        "Не заменяй слова эмодзи (эмодзи только как мягкий визуальный акцент). "
        "Лучше всего ставить 0–1 эмодзи в конце подзаголовков `### ...` (не в начале строки). "
        f"Используй только уместные эмодзи из списка: {allowlist}. "
    )


# Emoji limiting is a *display-level* safeguard; it should not change factual content.
_EMOJI_BASE_CLASS = (
    r"[\U0001F1E6-\U0001F1FF"  # flags
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F700-\U0001F77F"  # alchemical symbols (rare but emoji-like)
    r"\U0001F780-\U0001F7FF"  # geometric extended
    r"\U0001F800-\U0001F8FF"  # arrows-C
    r"\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    r"\U0001FA00-\U0001FAFF"  # symbols & pictographs extended-A
    r"\u2600-\u26FF"          # misc symbols
    r"\u2700-\u27BF"          # dingbats
    r"]"
)
_EMOJI_SEQ_RE = re.compile(
    rf"(?:{_EMOJI_BASE_CLASS})(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?"
    rf"(?:\u200D(?:{_EMOJI_BASE_CLASS})(?:[\uFE0E\uFE0F])?(?:[\U0001F3FB-\U0001F3FF])?)*"
)


def _limit_emoji_sequences(text: str, *, max_keep: int) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if max_keep < 0:
        max_keep = 0
    matches = list(_EMOJI_SEQ_RE.finditer(raw))
    if not matches:
        return raw
    if len(matches) <= max_keep:
        return raw
    out: list[str] = []
    last = 0
    kept = 0
    for m in matches:
        out.append(raw[last : m.start()])
        if kept < max_keep:
            out.append(m.group(0))
            kept += 1
        last = m.end()
    out.append(raw[last:])
    cleaned = "".join(out)
    # Fix excessive spaces left after emoji removal; keep paragraph breaks intact.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _limit_description_emojis(text: str | None) -> str | None:
    if text is None:
        return None
    if SMART_UPDATE_DESCRIPTION_EMOJI_MODE != "light":
        return text
    max_keep = int(SMART_UPDATE_DESCRIPTION_MAX_EMOJIS or 0)
    return _limit_emoji_sequences(text, max_keep=max_keep) or None


def _looks_like_list_block(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    hits = sum(1 for ln in lines if _LIST_ITEM_LINE_RE.match(ln))
    if hits < 2:
        return False
    return hits >= max(2, int(len(lines) * 0.6))


def _looks_like_structured_block(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if re.search(r"(?m)^\s*#{1,6}\s+\S", raw):
        return True
    if re.search(r"(?m)^\s*>", raw):
        return True
    if _looks_like_list_block(raw):
        return True
    return False


def _sanitize_description_output(
    text: str | None,
    *,
    source_text: str | None,
) -> str | None:
    """Best-effort *non-semantic* cleanup of LLM output for public Telegraph pages.

    Project rule: meaning-bearing operations on text should be done by LLM, not by
    deterministic regex cutting. This function therefore limits itself to:
    - trimming/normalizing whitespace
    - removing standalone internal/technical headings that must not leak publicly

    Anything more aggressive (cliche removal, logistics removal, etc.) must be
    handled in prompts / LLM editor passes.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    internal_heading_re = re.compile(
        r"(?i)^\s*(?:#{1,6}\s*)?(?:"
        r"facts\s*(?:/\s*)?added\s*facts|facts|added\s*facts|"
        r"факты\s*(?:/\s*)?добавленные\s*факты|"
        r"факты\s+о\s+событии|"
        r"факты\s+для\s+лога\s+источник\w*|"
        r"факты|добавленные\s*факты"
        r")\s*:?\s*$"
    )
    # "Facts for source log" is strictly internal and must never leak publicly.
    # If the LLM emits a whole paragraph that starts with such heading, drop it.
    internal_log_prefix_re = re.compile(r"(?i)^\s*(?:факты\s+для\s+лога|facts\s+for\s+source)\b")

    parts: list[str] = []
    for para in re.split(r"\n{2,}", raw):
        s = para.strip()
        if not s:
            continue
        if internal_heading_re.match(s):
            continue
        # If a paragraph starts with an internal heading (e.g. "Facts:"), strip only that
        # heading line and keep the content (non-semantic cleanup of a display artifact).
        lines = [ln.rstrip() for ln in s.splitlines()]
        # Find first non-empty line.
        first_idx = None
        for i, ln in enumerate(lines):
            if ln.strip():
                first_idx = i
                break
        if first_idx is not None:
            first = lines[first_idx].strip()
            if internal_log_prefix_re.match(first):
                # Entire block is internal (facts for /log), drop it.
                continue
            if internal_heading_re.match(first):
                # Drop only the heading line.
                lines = lines[:first_idx] + lines[first_idx + 1 :]
                s2 = "\n".join([ln for ln in lines]).strip()
                if not s2:
                    continue
                s = s2
        parts.append(s)
    cleaned = "\n\n".join(parts).strip()

    # Unescape backslash-escaped quotes that sometimes leak from JSON-ish sources
    # (e.g. `\\\"Сигнал\\\"` or `\"Сигнал\"`). This is a display-level cleanup
    # and should not change meaning.
    cleaned = cleaned.replace("\\\\\"", "\"")
    cleaned = cleaned.replace("\\\"", "\"")

    def _demote_overlong_headings(value: str) -> str:
        """Demote headings that look like full paragraphs (formatting-only fix)."""
        if not value:
            return value
        out_lines: list[str] = []
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
        for ln in value.splitlines():
            m = heading_re.match(ln)
            if not m:
                out_lines.append(ln)
                continue
            hashes = m.group(1)
            content = (m.group(2) or "").strip()
            # Strip internal "Facts:" prefix even if it got embedded into a heading line.
            content = re.sub(r"(?i)^(?:facts|факты)\s*:\s*", "", content).strip()
            # Heuristic: headings should be short; if it's long or looks like a sentence, demote.
            word_count = len(re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", content))
            # Keep headings short for Telegraph readability: long headings look like giant paragraphs.
            too_long = len(content) >= 80 or word_count >= 12
            if too_long:
                out_lines.append(content)
            else:
                out_lines.append(f"{hashes} {content}".rstrip())
        return "\n".join(out_lines).strip()

    def _strip_inline_facts_prefixes(value: str) -> str:
        """Strip 'Facts:' / 'Факты:' prefixes (display artifact)."""
        if not value:
            return value
        patt = re.compile(
            r"(?i)(?:^|(?<=\s))(?:\*\*|__)?(?:facts|факты)\s*:\s*(?:\*\*|__)?(?:\s+|$)"
        )
        out: list[str] = []
        for ln in value.splitlines():
            out.append(patt.sub("", ln))
        return "\n".join(out).strip()

    # Avoid leading blank lines/spacers before the first heading.
    cleaned = re.sub(r"^\s*\n+", "", cleaned)
    cleaned = _demote_overlong_headings(cleaned)
    cleaned = _strip_inline_facts_prefixes(cleaned)
    cleaned = cleaned.strip()
    cleaned = _limit_description_emojis(cleaned) or cleaned
    return cleaned or None


def _normalize_for_similarity(text: str | None, *, drop_structured: bool = True) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    raw = unicodedata.normalize("NFKC", raw)
    raw = raw.replace("\xa0", " ")
    lines: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if drop_structured:
            # Ignore structured/quoted blocks: they may be verbatim by design.
            if s.startswith(">"):
                continue
            if re.search(r"^\s*#{1,6}\s+\S", s):
                continue
            if _LIST_ITEM_LINE_RE.match(s):
                continue
        lines.append(s)
    cleaned = " ".join(lines)
    cleaned = _LOGISTICS_URL_RE.sub(" ", cleaned)
    cleaned = _LOGISTICS_PHONE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.casefold().replace("ё", "е")
    return cleaned


def _description_too_verbatim(description: str | None, *, source_text: str | None) -> bool:
    desc_norm = _normalize_for_similarity(description, drop_structured=True)
    src_norm = _normalize_for_similarity(source_text, drop_structured=True)
    desc_norm_strict = _normalize_for_similarity(description, drop_structured=False)
    src_norm_strict = _normalize_for_similarity(source_text, drop_structured=False)
    if not (desc_norm and src_norm) and not (desc_norm_strict and src_norm_strict):
        return False
    # Only enforce on sufficiently long texts; small snippets can be legitimately similar.
    if (
        (desc_norm and len(desc_norm) < 180)
        or (src_norm and len(src_norm) < 240)
    ) and (
        (desc_norm_strict and len(desc_norm_strict) < 220)
        or (src_norm_strict and len(src_norm_strict) < 260)
    ):
        return False
    try:
        from difflib import SequenceMatcher
    except Exception:  # pragma: no cover
        return False
    if desc_norm and src_norm and desc_norm in src_norm and len(desc_norm) >= 240:
        return True
    if desc_norm_strict and src_norm_strict and desc_norm_strict in src_norm_strict and len(desc_norm_strict) >= 280:
        return True
    ratio = (
        SequenceMatcher(None, desc_norm, src_norm).ratio()
        if (desc_norm and src_norm)
        else 0.0
    )
    ratio_strict = (
        SequenceMatcher(None, desc_norm_strict, src_norm_strict).ratio()
        if (desc_norm_strict and src_norm_strict)
        else 0.0
    )
    return ratio >= 0.88 or ratio_strict >= 0.90


_LOGISTICS_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?7|8)\s*\(?\d{3}\)?\s*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)|(?<!\d)\d{10,11}(?!\d)"
)
_LOGISTICS_URL_RE = re.compile(r"(?i)\bhttps?://\S+")
_LOGISTICS_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_LOGISTICS_PRICE_RE = re.compile(r"(?i)\b\d{2,6}\s*(?:₽|руб\.?|рублей|рубля|р\.?)\b")
_LOGISTICS_DDMM_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}(?:[./]20\d{2})?\b")
_LOGISTICS_ADDR_WORD_RE = re.compile(
    r"(?i)\b("
    r"ул\.?|улиц\w*|"
    r"пр\.?|проспект\w*|"
    r"пер\.?|переул\w*|"
    r"наб\.?|набережн\w*|"
    r"пл\.?|площад\w*|"
    r"бульвар\w*|бул\.?|"
    r"шоссе|"
    r"дом|д\.|"
    r"корпус|корп\.?|к\.|"
    r"офис|этаж|"
    r"г\.|город"
    r")\b"
)
_LOGISTICS_TICKET_WORD_RE = re.compile(r"(?i)\b(билет\w*|регистрац\w*|запис\w*|брон\w*|вход)\b")
_LOGISTICS_TICKET_CONDITION_KEEP_RE = re.compile(
    r"(?i)\b("
    r"входн\w*\s+билет|"
    r"нужн\w*|понадобит\w*|необходим\w*|"
    r"дополнительно|отдельно|помимо|кроме|"
    r"не\s+входит|входит\s+в|"
    r"оплачива\w*\s+отдельно"
    r")\b"
)
_LOGISTICS_TICKET_BOILERPLATE_DROP_RE = re.compile(
    r"(?i)\b("
    r"билет\w*\s+(?:доступн\w*|в\s+продаже)|"
    r"купит\w+\s+билет\w*|"
    r"по\s+ссылке|"
    r"подробнее|"
    r"регистрац\w*.*\bссылк\w*"
    r")\b"
)

_DESCRIPTION_CHANNEL_PROMO_SENT_RE = re.compile(
    r"(?i)\b("
    r"информац\w*\s+о\s+(?:событи\w*|мероприят\w*).{0,80}?(?:telegram|телеграм)[- ]?канал|"
    r"следит\w*\s+за\s+(?:анонс\w*|афиш\w*).{0,80}?(?:telegram|телеграм)|"
    r"подпис\w*\s+на\s+(?:наш\s+)?(?:telegram|телеграм)[- ]?канал|"
    r"(?:telegram|телеграм)[- ]?канал.{0,80}?(?:анонс\w*|афиш\w*)"
    r")\b"
)

_DESCRIPTION_CHANNEL_PROMO_PHRASE_RE = re.compile(
    r"(?i)\b("
    r"информац\w*\s+о\s+(?:событи\w*|мероприят\w*)|"
    r"следит\w*\s+за\s+(?:анонс\w*|афиш\w*)|"
    r"подпис\w*\s+на\s+(?:наш\s+)?(?:telegram|телеграм)[- ]?канал|"
    r"(?:telegram|телеграм)[- ]?канал"
    r")\b"
)


def _format_ru_date_phrase(iso_value: str | None) -> str | None:
    if not iso_value:
        return None
    raw = iso_value.split("..", 1)[0].strip()
    if not raw:
        return None
    try:
        d = date.fromisoformat(raw)
    except Exception:
        return None
    months = {v: k for k, v in _RU_MONTHS_GENITIVE.items()}
    month_word = months.get(d.month)
    if not month_word:
        return None
    return f"{d.day} {month_word}"


def _strip_infoblock_logistics_from_description(
    text: str | None,
    *,
    candidate: EventCandidate,
) -> str | None:
    """Remove obvious logistics duplicates from narrative description.

    Telegraph pages already render a quick facts infoblock (date/time/location/tickets),
    so repeating these details inside the narrative bloats the text.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    ru_date = _format_ru_date_phrase(candidate.date)
    needles: list[str] = []
    price_values: set[int] = set()
    for pv in (candidate.ticket_price_min, candidate.ticket_price_max):
        if isinstance(pv, int) and pv > 0:
            price_values.add(pv)
    for val in (
        candidate.date,
        candidate.time,
        candidate.location_address,
        ru_date,
        _format_ticket_price(candidate.ticket_price_min, candidate.ticket_price_max),
    ):
        v = str(val or "").strip()
        if v:
            needles.append(v)
            if v == candidate.time and ":" in v:
                needles.append(v.replace(":", "."))
    # Avoid stripping plain venue names from narrative text: it's often part of the story
    # ("в баре …") and removing it can make sentences awkward. Keep stripping when the
    # configured "location_name" itself looks like a full address line.
    loc_name = str(getattr(candidate, "location_name", "") or "").strip()
    if loc_name and (
        re.search(r"\d", loc_name)
        or _LOGISTICS_ADDR_WORD_RE.search(loc_name)
        or loc_name.count(",") >= 2
    ):
        needles.append(loc_name)
    # Also add DD.MM / DD.MM.YYYY derived from candidate.date when available.
    if candidate.date:
        try:
            d = date.fromisoformat(candidate.date.split("..", 1)[0].strip())
            ddmm = d.strftime("%d.%m")
            ddmmyyyy = d.strftime("%d.%m.%Y")
            needles.extend([ddmm, ddmmyyyy, ddmm.replace(".", "/"), ddmmyyyy.replace(".", "/")])
        except Exception:
            pass

    def _strip_sentence(sentence: str) -> str:
        s = sentence
        had_price = bool(_LOGISTICS_PRICE_RE.search(sentence)) or any(
            (isinstance(pv, int) and pv > 0 and str(pv) in sentence) for pv in price_values
        )
        had_ticket_word = bool(_LOGISTICS_TICKET_WORD_RE.search(sentence))
        s = _LOGISTICS_URL_RE.sub("", s)
        s = _LOGISTICS_PHONE_RE.sub("", s)
        if price_values:
            for pv in sorted(price_values, reverse=True):
                s = re.sub(
                    rf"(?i)(?<!\d){pv}\s*(?:₽|руб\.?|рублей|рубля|р\.?)(?!\w)",
                    "",
                    s,
                )
        for needle in needles:
            if len(needle) < 4:
                continue
            s = re.sub(re.escape(needle), "", s, flags=re.IGNORECASE)

        # Remove common logistics lead-ins that become noise after stripping.
        s = re.sub(r"(?i)\b(сбор\s+гост\w*|начал\w*|время\s+начала)\b\s*[:\-–—]?\s*", "", s)
        s = re.sub(r"(?i)\b(по\s+адресу|адрес)\b\s*[:\-–—]?\s*", "", s)
        s = re.sub(r"(?i)\b(стоимость|цена)\s+билет\w*\b\s*[:\-–—]?\s*", "", s)
        s = re.sub(r"(?i)\b(телефон|по\s+телефон\w*|звон\w*|контакт\w*)\b\s*[:\-–—]?\s*", "", s)

        # Cleanup punctuation/whitespace.
        s = s.replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"^[,.;:!?]+\s*", "", s).strip()
        # Keep sentence-ending punctuation (., !, ?, …). Removing it makes Telegraph text
        # look ungrammatical (common operator complaint). We only trim "soft" trailing
        # punctuation that often becomes dangling after stripping URLs/prices.
        s = re.sub(r"\s*[,;:]+\s*$", "", s).strip()

        # If we stripped the key payload (price/ticket) and left a dangling clause,
        # drop the sentence entirely to avoid broken Russian like "... составит".
        if (had_price or had_ticket_word) and not re.search(r"\d", s):
            if re.search(
                r"(?i)\b(составит|составят|будет|будут|стоит|стоить|обойдется|обойдётся)\b$",
                s,
            ):
                return ""
            if re.search(r"(?i)\b(стоимость|цена)\b", s):
                return ""
        return s

    out_paras: list[str] = []
    sent_split = re.compile(r"(?<=[.!?…])\s+")
    for para in re.split(r"\n{2,}", raw):
        p = para.strip()
        if not p:
            continue
        # Preserve headings/quotes as-is (quotes may include source wording).
        if p.lstrip().startswith(">") or re.match(r"^\s*#{1,6}\s+\S", p):
            out_paras.append(p)
            continue

        # For list-like blocks keep formatting and strip logistics line-by-line.
        if _looks_like_list_block(p) or re.match(r"^\s*[-*•]\s+\S", p):
            kept_lines: list[str] = []
            for line in p.splitlines():
                if not line.strip():
                    continue
                stripped = _strip_sentence(line)
                if not stripped:
                    continue
                # Drop "empty logistics" leftovers like "Билеты доступны" after removing link/price.
                if (
                    _LOGISTICS_TICKET_WORD_RE.search(stripped)
                    and (
                        (
                            len(stripped) < 28
                            and not _LOGISTICS_TICKET_CONDITION_KEEP_RE.search(stripped)
                        )
                        or (
                            _LOGISTICS_TICKET_BOILERPLATE_DROP_RE.search(stripped)
                            and not _LOGISTICS_TICKET_CONDITION_KEEP_RE.search(stripped)
                        )
                    )
                ):
                    continue
                if not re.search(r"[A-Za-zА-Яа-яЁё]", stripped):
                    continue
                kept_lines.append(stripped)
            if kept_lines:
                out_paras.append("\n".join(kept_lines).strip())
            continue

        sents = [s.strip() for s in sent_split.split(re.sub(r"\s*\n\s*", " ", p)) if s.strip()]
        kept: list[str] = []
        for sent in sents:
            stripped = _strip_sentence(sent)
            if not stripped:
                continue
            # Drop "empty logistics" leftovers like "Билеты доступны" after removing link/price.
            if (
                _LOGISTICS_TICKET_WORD_RE.search(stripped)
                and (
                    (
                        len(stripped) < 28
                        and not _LOGISTICS_TICKET_CONDITION_KEEP_RE.search(stripped)
                    )
                    or (
                        _LOGISTICS_TICKET_BOILERPLATE_DROP_RE.search(stripped)
                        and not _LOGISTICS_TICKET_CONDITION_KEEP_RE.search(stripped)
                    )
                )
            ):
                continue
            # Keep only sentences with some letters left.
            if not re.search(r"[A-Za-zА-Яа-яЁё]", stripped):
                continue
            if len(stripped) < 18 and len(stripped.split()) < 3:
                continue
            kept.append(stripped)
        if kept:
            out_paras.append(" ".join(kept).strip())
    cleaned = "\n\n".join(out_paras).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


def _description_needs_infoblock_logistics_strip(
    text: str | None,
    *,
    candidate: EventCandidate,
) -> bool:
    """Cheap gate to reduce deterministic вмешательство в текст.

    We only run the heavy stripping pass when we see clear logistics signals.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    if _LOGISTICS_URL_RE.search(raw):
        return True
    if _LOGISTICS_PHONE_RE.search(raw):
        return True
    if _LOGISTICS_PRICE_RE.search(raw):
        return True
    if _LOGISTICS_TIME_RE.search(raw):
        return True
    if _LOGISTICS_DDMM_RE.search(raw):
        return True
    if _LOGISTICS_ADDR_WORD_RE.search(raw):
        return True
    if _LOGISTICS_TICKET_WORD_RE.search(raw):
        return True
    # Candidate anchors occasionally leak verbatim; strip only if present.
    for val in (
        getattr(candidate, "location_address", None),
        _format_ru_date_phrase(getattr(candidate, "date", None)),
        getattr(candidate, "time", None),
        getattr(candidate, "date", None),
    ):
        v = str(val or "").strip()
        if v and v.casefold() in raw.casefold():
            return True
    return False


def _description_needs_channel_promo_strip(text: str | None) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if not ("телеграм" in raw.casefold() or "telegram" in raw.casefold() or "t.me/" in raw.casefold()):
        return False
    return bool(_DESCRIPTION_CHANNEL_PROMO_SENT_RE.search(raw))


def _strip_channel_promo_from_description(text: str | None) -> str | None:
    # Deterministic sentence-level cutting is not allowed (LLM handles this).
    raw = (text or "").strip()
    return raw or None


def _norm_text_for_fact_presence(text: str) -> str:
    """Deterministic normalization for 'fact presence' substring checks.

    We intentionally keep this conservative: it's used only to detect obvious
    omissions (e.g. short slogan-like quoted facts) and should not attempt
    semantic matching.
    """
    raw = (text or "").casefold()
    raw = raw.replace("ё", "е")
    raw = raw.translate(
        str.maketrans(
            {
                "«": '"',
                "»": '"',
                "“": '"',
                "”": '"',
                "„": '"',
                "’": "'",
                "–": "-",
                "—": "-",
                "\u00a0": " ",
                "\u2009": " ",
                "\u202f": " ",
                "\ufeff": "",
                "\u200b": "",
                "\u2060": "",
            }
        )
    )
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _is_anchor_or_service_fact(fact: str) -> bool:
    f = (fact or "").strip()
    if not f:
        return True
    # Do not force anchors / service notes into narrative coverage checks.
    if re.search(r"(?i)^(дата|время|локац\w*|адрес\w*|город\w*|источник)\b", f):
        return True
    # Dates/times often appear as free-form sentences, not only as `Дата:`/`Время:`.
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", f):  # 2026-03-03
        return True
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", f):  # 18:30 / 18.30
        return True
    if re.search(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b", f):  # 03.03 / 03.03.2026
        return True
    if re.search(
        r"(?i)\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
        f,
    ):
        return True
    # Location/logistics phrasing (must stay in infoblock, not in narrative).
    if re.search(r"(?i)\b(по\s+адресу|\d+\s*этаж\w*|этаж\w*|зал\w*|аудитори\w*)\b", f):
        return True
    # Event type is provided separately to the narrative generator; keep `Тип:` out of text_clean.
    if re.search(r"(?i)^тип\b", f):
        return True
    if re.search(r"(?i)^(текст\s+очищен|llm\s+недоступна|добавлена\s+афиша)\b", f):
        return True
    if "http://" in f or "https://" in f or "www." in f.casefold():
        if _fact_first_urls_are_allowed(f):
            return False
        return True
    return False


def _find_missing_facts_in_description(
    *, description: str, facts: Sequence[str], max_items: int = 5
) -> list[str]:
    """Return a small list of facts that are very likely missing from description."""
    desc_n = _norm_text_for_fact_presence(description)
    missing: list[str] = []
    for fact in facts:
        f = str(fact or "").strip()
        if not f or _is_anchor_or_service_fact(f):
            continue
        is_quoted = bool(re.fullmatch(r'["«].+["»]\s*', f)) or ("«" in f and "»" in f) or ('"' in f)
        # Only enforce coverage for short facts, unless they are explicit slogans/quotes
        # or an allowlisted content URL (playlist).
        if not is_quoted and len(f) > 90 and not _fact_first_urls_are_allowed(f):
            continue
        # Prefer checking the "inner" content for quoted slogan-like facts.
        inner = f
        m = re.fullmatch(r'["«](.+?)["»]\s*', f)
        if m:
            inner = m.group(1).strip()
        needle = _norm_text_for_fact_presence(inner)
        if not needle:
            continue
        if needle not in desc_n:
            missing.append(f)
            if len(missing) >= max_items:
                break
    return missing


async def _llm_integrate_missing_facts_into_description(
    *, description: str, missing_facts: Sequence[str], source_text: str, label: str
) -> str | None:
    """Ask LLM to integrate missing facts into description without adding new facts."""
    if SMART_UPDATE_LLM_DISABLED:
        return None
    desc = (description or "").strip()
    if not desc:
        return None
    facts = [str(f).strip() for f in (missing_facts or []) if str(f or "").strip()]
    if not facts:
        return None
    payload = {
        "description": _clip(desc, 5000),
        "missing_facts": facts[:8],
        "source_text": _clip(source_text or "", 2500),
    }
    prompt = (
        "В тексте описания события отсутствуют некоторые факты.\n"
        "Твоя задача: аккуратно встроить `missing_facts` в `description` так, чтобы текст читался связно.\n"
        "Правила:\n"
        "- НЕЛЬЗЯ добавлять новые факты (только из `missing_facts`).\n"
        "- НЕЛЬЗЯ менять якорные поля (дата/время/площадка/адрес).\n"
        "- НЕ добавляй в текст логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены): она уже показана отдельным блоком.\n"
        "- НЕ добавляй промо-упоминания «где следить за анонсами» и ссылки на каналы/чаты с афишей.\n"
        "- Не дублируй в тексте строки формата `Дата:`, `Время:`, `Локация:`, `Билеты:`: эти данные уже показаны в карточке сверху.\n"
        "- Факты в кавычках (слоганы/характеристики) сохраняй ДОСЛОВНО, лучше в «ёлочках», "
        "и атрибутируй как слова/характеристики из афиши/поста, а не как объективный прогноз.\n"
        "- Не добавляй рекламных клише и прогнозов.\n"
        "- Сохраняй существующие цитаты в формате blockquote (`>`).\n"
        "- Не оставляй обрывов фраз после правок (например «стоимость … составит» без суммы): перефразируй или удали.\n"
        "- Самопроверка: все предложения грамматически завершены; не появилось странных/непонятных слов.\n"
        f"{SMART_UPDATE_YO_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=900,
        label=label,
        temperature=0.0,
    )
    return text.strip() if text else None


_FACT_FIRST_CTA_RE = re.compile(
    r"(?i)\b("
    r"приглашаем|приходите|жд[её]м\s+вас|не\s+пропуст\w*|успей\w*|"
    r"присоединяйтесь|подписывайтесь|подпиш\w*|репост\w*|лайк\w*|"
    r"делитесь|расскажите|рекомендуем|советуем|жми|жмите|"
    r"покупай\w*|купите|брониру\w*|зарегистриру\w*|записывайтесь|"
    r"встречайте"
    r")\b"
)
_FACT_FIRST_TICKET_WORD_RE = re.compile(
    r"(?i)\b(билет\w*|вход\w*|регистрац\w*|запис\w*|брон\w*|бесплатн\w*)\b"
)
_FACT_FIRST_PRICE_RE = re.compile(r"(?i)\b\d{1,6}\s*(?:₽|руб\.?|р\.)\b")
_FACT_FIRST_AGE_RE = re.compile(r"\b\d{1,2}\+\b")
_FACT_FIRST_PUSHKIN_RE = re.compile(r"(?i)\bпушкинск\w*\s+карт\w*\b")
_FACT_FIRST_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_FACT_FIRST_PHONE_RE = re.compile(r"(?i)\+7\D*\d{3}\D*\d{3}\D*\d{2}\D*\d{2}")
_FACT_FIRST_HASH_RE = re.compile(r"(?m)(?:^|\s)#[A-Za-zА-Яа-яЁё0-9_]{2,}")
_FACT_FIRST_ALLOWED_CONTENT_URL_RE = re.compile(
    r"(?i)https?://music\.yandex\.ru/users/[^\s/]+/playlists/\d+\S*"
)


def _fact_first_extract_urls(text: str) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    return [m.group(0) for m in _FACT_FIRST_URL_RE.finditer(raw)]


def _fact_first_urls_are_allowed(text: str) -> bool:
    urls = _fact_first_extract_urls(text)
    if not urls:
        return False
    return all(bool(_FACT_FIRST_ALLOWED_CONTENT_URL_RE.search(url)) for url in urls)


def _fact_first_has_disallowed_urls(text: str) -> bool:
    urls = _fact_first_extract_urls(text)
    if not urls:
        return False
    return any(not _FACT_FIRST_ALLOWED_CONTENT_URL_RE.search(url) for url in urls)


def _fact_first_bucket(fact: str) -> str:
    """Classify a fact into a bucket for fact-first narrative generation.

    Buckets:
    - text_clean: publishable narrative facts
    - infoblock: logistics/tickets/prices/contacts/anchors (must not go into narrative)
    - drop: promo/CTA/noise (must not go into narrative)
    """
    f = str(fact or "").strip()
    if not f:
        return "drop"
    if _is_anchor_or_service_fact(f):
        return "infoblock"
    if _FACT_FIRST_PHONE_RE.search(f):
        return "infoblock"
    if _FACT_FIRST_URL_RE.search(f) and not _fact_first_urls_are_allowed(f):
        return "infoblock"
    if _FACT_FIRST_TICKET_WORD_RE.search(f) or _FACT_FIRST_PRICE_RE.search(f):
        return "infoblock"
    if _FACT_FIRST_AGE_RE.search(f) or _FACT_FIRST_PUSHKIN_RE.search(f):
        return "infoblock"
    if re.search(r"(?i)\b(афиш\w*|постер\w*)\b", f):
        return "infoblock"
    if _FACT_FIRST_HASH_RE.search(f) or _FACT_FIRST_CTA_RE.search(f):
        return "drop"
    return "text_clean"


def _facts_text_clean_from_facts(
    facts: Sequence[str],
    *,
    max_items: int = 28,
    anchors: Sequence[str] | None = None,
) -> list[str]:
    anchors_norm: list[str] = []
    for a in anchors or []:
        s = str(a or "").strip()
        if not s:
            continue
        # Avoid filtering by very short/common tokens.
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s)
        if not tokens:
            continue
        if len(s) < 5 and all(len(t) < 5 for t in tokens):
            continue
        anchors_norm.append(s.casefold())

    out: list[str] = []
    for fact in facts or []:
        cleaned = _normalize_fact_item(str(fact or ""), limit=180)
        if not cleaned:
            continue
        bucket = _fact_first_bucket(cleaned)
        if bucket != "text_clean":
            # Participant chats are useful as a *fact*, but their links are forbidden in narrative text.
            # Keep the meaning, drop the URL.
            if (
                bucket == "infoblock"
                and _FACT_FIRST_URL_RE.search(cleaned)
                and re.search(r"(?iu)\bчат\w*\b", cleaned)
            ):
                without_urls = _FACT_FIRST_URL_RE.sub("", cleaned)
                without_urls = re.sub(r"\s+", " ", without_urls).strip()
                without_urls = re.sub(r"[\s:—–-]+$", "", without_urls).strip()
                if without_urls and _fact_first_bucket(without_urls) == "text_clean":
                    cleaned = without_urls
                else:
                    continue
            else:
                continue
        if anchors_norm:
            cleaned_cf = cleaned.casefold()
            if any(a in cleaned_cf for a in anchors_norm):
                continue
        out.append(cleaned)
        if len(out) >= max_items * 2:
            break
    return _dedupe_source_facts(out)[:max_items]


def _pick_epigraph_fact(facts_text_clean: Sequence[str]) -> str | None:
    facts = [str(f or "").strip() for f in (facts_text_clean or []) if str(f or "").strip()]
    if not facts:
        return None
    # Prefer a direct quote-like fact.
    for f in facts:
        if ("«" in f and "»" in f) or re.search(r"(?i)^\s*цитата\b", f):
            return f
    # Fallback: short "tagline" fact with an em dash, avoiding key-value facts.
    for f in facts:
        if ":" in f:
            continue
        if re.search(r"(?i)^(ведущ\w*|лектор\w*|спикер\w*|гост\w*)\b", f):
            continue
        if "—" in f and 20 <= len(f) <= 140:
            return f
    return None


_FACT_FIRST_POSV_WORD_RE = re.compile(r"(?i)\bпосвящ\w*\b")


def _sanitize_fact_text_clean_for_prompt(fact: str) -> str:
    """Pre-sanitize text_clean facts for the narrative prompt.

    Goal: keep the meaning, but avoid strictly forbidden lexical markers that
    models tend to copy verbatim from facts (e.g. "посвящ..." in Russian).
    This is *not* persisted as a source fact; it only affects the description prompt.
    """
    s = str(fact or "").strip()
    if not s or not _FACT_FIRST_POSV_WORD_RE.search(s):
        return s

    # Common pattern: "<event> посвящена/посвящён ... <topic>" -> "Тема: <topic>."
    m = re.match(
        r"(?i)^\s*(?:лекци\w*|встреч\w*|бесед\w*|показ\w*|концерт\w*|спектакл\w*|"
        r"мастер-?класс\w*|мастерск\w*|заняти\w*|экскурс\w*|презентац\w*|выставк\w*)\s+"
        r"посвящ\w+\s+(.+?)\s*[.!?]?\s*$",
        s,
    )
    if m:
        topic = (m.group(1) or "").strip()
        if topic:
            return f"Тема: {topic}."
    return s


def _estimate_fact_first_description_budget_chars(facts_text_clean: Sequence[str]) -> int:
    facts_chars = sum(len(str(f or "")) for f in (facts_text_clean or []) if str(f or "").strip())
    # Budget should scale with fact volume: if we have many (already de-duped) facts,
    # the narrative should be allowed to be longer (Telegraph can render long texts).
    budget = int(facts_chars * 1.10 + 420)
    return max(800, min(SMART_UPDATE_DESCRIPTION_MAX_CHARS, budget))


def _estimate_fact_first_description_max_tokens(
    *, budget_chars: int, floor: int = 1700, ceil: int = 4500
) -> int:
    """Best-effort output token budget for fact-first description.

    We keep this deterministic and conservative: it only sets an upper bound for the
    provider call. The prompt still instructs the model to stay within
    `description_budget_chars`.
    """
    # Rough heuristic for Russian text + Markdown: ~3.2 chars per token.
    est = int(math.ceil(float(max(0, int(budget_chars or 0))) / 3.2)) + 200
    return max(int(floor), min(int(ceil), est))


def _fact_first_description_prompt(
    *,
    title: str | None,
    event_type: str | None,
    facts_text_clean: Sequence[str],
    epigraph_fact: str | None,
) -> str:
    facts_block = "\n".join(f"- {str(f).strip()}" for f in (facts_text_clean or []) if str(f or "").strip())
    budget_chars = _estimate_fact_first_description_budget_chars(facts_text_clean)
    return textwrap.dedent(
        f"""\
        Ты пишешь Markdown‑анонс события для Telegram в стиле культурного журналиста: живо, конкретно, без рекламы.

        Источник истины: ТОЛЬКО список facts_text_clean ниже. Нельзя добавлять новые сведения.

        Цель: связный текст, где каждая деталь из фактов упомянута и при этом нет смысловых повторов.

        {SMART_UPDATE_YO_RULE}

        Структура:
        - Если epigraph_fact не null:
          - Если в нём есть длинная прямая речь в «...!»/«...?» — оформи эпиграф 2 строками:
            `> «...»
            > — ...` (атрибуция только из слов epigraph_fact; без новых глаголов типа «восхищается/говорит»).
          - Иначе — одна строка `> epigraph_fact`.
          - После эпиграфа: пустая строка, затем лид ОДНИМ абзацем (1–2 предложения) без заголовка (без переносов строки).
          - В теле текста НЕ повторяй и НЕ пересказывай epigraph_fact: он уже прозвучал в эпиграфе.
        - Если epigraph_fact null: просто лид ОДНИМ абзацем (1–2 предложения) без заголовка (без переносов строки).
        - Затем 2–3 блока с подзаголовками `### ...` (только `###`).
        - Подзаголовки короткие (до ~60 символов), без точек, не полные предложения; не делай пустых блоков.
        - Подзаголовки должны быть информативными; избегай общих вроде «Подробности».
        - Под каждым `###` должно быть либо 2–4 предложения, либо список (2+ пунктов).
        - Абзацы средней длины: обычно 2–4 предложения; избегай микро‑абзацев из одной короткой фразы.
        - Объём: старайся уложиться примерно в лимит `description_budget_chars` символов, без воды.
        - Эмодзи: 1–2 штуки, как навигация (в лид/в 1 заголовке), без «ёлки».
        - Если факты дают несколько тем/пунктов (вопросы лекции, пункты программы и т.п.) — сгруппируй их в ОДИН раздел и оформи списком; не делай отдельный `###` для каждого пункта.

        Стиль C («Сцена → смысл → детали»):
        - Начни с короткой «картинки/настроения» из фактов.
          Если в фактах есть строка `Атмосфера ...` — используй её ДОСЛОВНО.
          Если такой строки нет — НЕ придумывай новую «атмосферу», начни с конкретики из фактов.
        - Во 2‑й фразе лида дай понять, что это за событие (спектакль/концерт/показ фильма/игра/мастерская и т.п.).
        - Если среди фактов есть строка `Формат: ...` — обозначь формат ЯВНО (ключевые слова после `:` должны прозвучать).
        - Условия участия/просмотра из фактов (длительность, язык/субтитры, что взять/что предоставляется, количество участников/игроков) — ОБЯЗАТЕЛЬНО упомяни явно.

        Как работать с фактами:
        - Объединяй и перестраивай фразы, но аккуратно: не добавляй новых смыслов/обещаний/обобщений.
        - Не превращай текст в «перечень фактов» и не копируй подряд несколько фактов как готовые предложения: перепиши связно, сохранив детали.
        - Если в факте есть список/перечень/треклист/набор пунктов — сохрани ВСЕ элементы ДОСЛОВНО и лучше оформи списком (каждый пункт на отдельной строке).
        - Сохраняй ключевые слова и образы из фактов; не подменяй «образные» слова синонимами.
        - Если в факте есть ЦИФРЫ/диапазон/рейтинги — эти же цифры должны появиться в тексте.

        Запреты:
        - Нельзя: дату/время/город/площадку/адрес, URL/телефоны (кроме ссылок на плейлист Я.Музыки из facts_text_clean), билеты/вход/регистрацию/запись, цены/донаты, возраст, «Пушкинская карта», афиши.
        - Нельзя CTA‑слова: «приглашаем», «приходите», «ждём/ждем вас», «не пропустите», «успейте», «присоединяйтесь», «предлагаем», «встречайте».
        - Не используй слово «посвящён/посвящена/посвящено».
          Если в фактах встречается «... посвящ... ...» — перефразируй без этого слова: «лекция о ...», «разговор про ...», «в центре — ...».
        - Запрещён штамп: «это ... не ..., а ...».

        Контекст:
        - title: {(title or '').strip()}
        - event_type: {(event_type or '').strip()}
        - epigraph_fact (если null — эпиграф НЕ нужен): {epigraph_fact if epigraph_fact is not None else 'null'}
        - description_budget_chars: {budget_chars}

        Факты (facts_text_clean):
        {facts_block}

        Верни только Markdown‑текст описания (без JSON).
        """
    ).strip()


def _fact_first_coverage_prompt(
    *,
    facts_text_clean: Sequence[str],
    description: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "facts_text_clean": [str(x).strip() for x in (facts_text_clean or []) if str(x or "").strip()],
        "description_md": _clip((description or "").strip(), 6500),
    }
    prompt = (
        "Ты проверяешь полноту и строгость описания относительно списка фактов.\n\n"
        "Найди:\n"
        "- missing: факты из списка facts_text_clean, которые НЕ отражены в описании.\n"
        "- extra: утверждения из описания, которые НЕ подтверждаются ни одним фактом.\n\n"
        "ВАЖНО:\n"
        "- Считай факт отражённым, если он перефразирован/встроен в предложение и при этом сохранены ключевые сущности и детали.\n"
        "- Допустимы синонимы и формы, если смысл и ключевые детали сохранены.\n"
        "- Если факт содержит несколько элементов, считай его отражённым только если в тексте присутствуют ВСЕ элементы.\n"
        "- Служебные перефразы для структурных фактов НЕ считаются extra, если они явно отсылают к фактам.\n"
        "- Для missing используй ТОЛЬКО строки фактов ВЕРБАТИМ (копируй из списка facts_text_clean; не перефразируй).\n"
        "- Для extra используй ТОЛЬКО короткие ВЕРБАТИМ-фрагменты из description_md (копируй из текста; не выдумывай), до 12 слов.\n\n"
        "Верни JSON по схеме.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    schema = {
        "type": "object",
        "properties": {
            "missing": {"type": "array", "items": {"type": "string"}},
            "extra": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["missing", "extra"],
        "additionalProperties": False,
    }
    return prompt, schema


def _fact_first_revise_prompt(
    *,
    title: str | None,
    event_type: str | None,
    epigraph_fact: str | None,
    facts_text_clean: Sequence[str],
    description: str,
    missing: Sequence[str],
    extra: Sequence[str],
    policy_issues: Sequence[str],
) -> str:
    budget_chars = _estimate_fact_first_description_budget_chars(facts_text_clean)
    facts_block = "\n".join(f"- {str(f).strip()}" for f in (facts_text_clean or []) if str(f or "").strip())
    return textwrap.dedent(
        f"""\
        Отредактируй Markdown‑анонс события. Если проще — перепиши его заново, но строго по фактам.

        Источник истины: только `facts_text_clean` (нельзя добавлять новые сведения вне списка).

        Цели (строго):
        - missing = [] (все факты отражены)
        - extra = [] (нет неподтверждённых утверждений)
        - нет запрещённых маркеров/логистики/промо
        - нет смысловых повторов

        {SMART_UPDATE_YO_RULE}
        Объём: старайся уложиться примерно в `description_budget_chars` символов, без воды.

        Контекст:
        - title: {(title or '').strip()}
        - event_type: {(event_type or '').strip()}
        - epigraph_fact: {epigraph_fact if epigraph_fact is not None else 'null'}
        - description_budget_chars: {budget_chars}

        Факты (facts_text_clean):
        {facts_block}

        Исправь проблемы:
        - policy_issues:
        {chr(10).join('  - ' + str(i) for i in (policy_issues or [])) if policy_issues else '  - (нет)'}
        - missing (добавь, не теряя деталей; цифры/имена/названия/элементы списков сохранить):
        {chr(10).join('  - ' + str(f) for f in (missing or [])) if missing else '  - (нет)'}
        - extra (удали/перепиши так, чтобы этих подстрок не осталось):
        {chr(10).join('  - ' + str(e) for e in (extra or [])) if extra else '  - (нет)'}

        Текущее описание:
        {description}

        Правила редактуры:
        - Списки/перечни/треклисты: элементы НЕ перефразируй; каждый пункт — отдельной строкой (можно списком `- ...`).
        - Факты вида `Формат: ...`: отрази формат явно; ключевые слова после `:` должны прозвучать.
        - Факты с цифрами/рейтинги/диапазоны: цифры должны совпадать с фактами.
        - Эпиграф: если epigraph_fact не null — blockquote до первого `###` и только один раз.
        - Структура: лид (1–2 предложения) без заголовка, ОДНИМ абзацем → 2–3 блока `### ...` → абзацы средней длины.
          Под каждым `###` — 2+ предложения ИЛИ список (2+ пунктов). Не дроби на микро‑разделы по 1 фразе.
        - Подзаголовки должны быть информативными; избегай общих вроде «Подробности».
        - Эмодзи: 1–2 штуки, без «ёлки».
        - Запреты: нет даты/времени/города/площадки/адреса; нет URL/телефонов (кроме ссылок на плейлист Я.Музыки из facts_text_clean); нет цен/донатов; нет билетов/входа/регистрации/записи; нет возраста; нет «Пушкинская карта»; нет афиш.
          Не используй «посвящ...»: перефразируй через «о/про/в центре — ...».

        Верни только обновлённый Markdown‑текст (без JSON).
        """
    ).strip()


def _fact_first_remove_posv_prompt(
    *,
    title: str | None,
    event_type: str | None,
    epigraph_fact: str | None,
    facts_text_clean: Sequence[str],
    description: str,
) -> str:
    """Targeted cleanup prompt for stubborn `посвящ...` leaks.

    Some models keep copying "лекция посвящена ..." despite explicit bans. This
    prompt is a last-mile, single-purpose fix: keep the same facts and structure,
    but ensure the forbidden root never appears.
    """
    payload = {
        "title": (title or "").strip(),
        "event_type": (event_type or "").strip(),
        "epigraph_fact": epigraph_fact if epigraph_fact is not None else None,
        "facts_text_clean": [str(x).strip() for x in (facts_text_clean or []) if str(x or "").strip()],
        "description_md": _clip((description or "").strip(), 3200),
    }
    return (
        "В описании найден запрещённый корень «посвящ…» (посвящён/посвящена/посвящено и т.п.).\n"
        "Твоя задача: отредактировать `description_md` так, чтобы этот корень НЕ встречался нигде.\n\n"
        "Правила:\n"
        "- Верни ПОЛНЫЙ Markdown-текст (не частями).\n"
        "- Нельзя добавлять новые факты: источник истины — только `facts_text_clean`.\n"
        "- Сохрани структуру: эпиграф (если есть) остаётся blockquote до первого `###`; затем лид; затем 2–3 `###`.\n"
        "- Нельзя добавлять логистику/CTA/ссылки/контакты/цены/возраст/афиши.\n"
        "- Слово/корень «посвящ» запрещён полностью. Перефразируй через «о/про/в центре — …», "
        "исправляя падежи/согласование.\n\n"
        f"{SMART_UPDATE_YO_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _fact_first_forbidden_reasons(description: str, *, anchors: Sequence[str]) -> list[str]:
    text = str(description or "")
    reasons: list[str] = []
    if re.search(r"(?m)^\s*#{1,2}\s+\S", text):
        reasons.append("h1h2_heading")
    if _fact_first_has_disallowed_urls(text):
        reasons.append("url")
    if _FACT_FIRST_PHONE_RE.search(text):
        reasons.append("phone")
    if _FACT_FIRST_PRICE_RE.search(text):
        reasons.append("price")
    if _FACT_FIRST_TICKET_WORD_RE.search(text):
        reasons.append("tickets")
    if _FACT_FIRST_AGE_RE.search(text):
        reasons.append("age")
    if _FACT_FIRST_PUSHKIN_RE.search(text):
        reasons.append("pushkin_card")
    if re.search(r"(?i)\bафиш\w*\b", text):
        reasons.append("poster")
    if re.search(r"(?i)\bпосвящ\w*\b", text):
        reasons.append("посвящ*")
    if _FACT_FIRST_CTA_RE.search(text) or _FACT_FIRST_HASH_RE.search(text):
        reasons.append("cta_or_hashtag")
    if re.search(r"(?i)\bэто\b[^.!?\n]{0,60}\bне\b[^.!?\n]{0,60}\bа\b", text):
        reasons.append("cliche_not_about_but_about")
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        reasons.append("date_iso")
    if re.search(r"\b\d{1,2}\.\d{1,2}(?:\.\d{2,4})?\b", text):
        reasons.append("date_ddmm")
    if re.search(
        r"(?i)\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
        text,
    ):
        reasons.append("date_ru_words")
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", text):
        reasons.append("time_hhmm")

    text_lc = text.lower()
    for a in anchors or []:
        s = str(a or "").strip()
        if not s:
            continue
        # Avoid flagging short/common tokens.
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", s)
        if not tokens:
            continue
        if len(s) < 5 and all(len(t) < 5 for t in tokens):
            continue
        if s.lower() in text_lc:
            reasons.append("anchor_leak")
            break
    return _dedupe_source_facts(reasons)


def _fact_first_lead_paragraph_count(description_md: str) -> int:
    """Count lead paragraphs (text before the first `### ...`), excluding epigraph blockquote."""
    raw = (description_md or "").strip()
    if not raw:
        return 0
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.splitlines()

    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    # Drop epigraph (blockquote at the top).
    while i < len(lines) and lines[i].lstrip().startswith(">"):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1

    lead_lines: list[str] = []
    for j in range(i, len(lines)):
        if re.match(r"^\s*###\s+\S", lines[j] or ""):
            break
        lead_lines.append(lines[j])
    lead = "\n".join(lead_lines).strip()
    if not lead:
        return 0
    paras = [p.strip() for p in re.split(r"\n{2,}", lead) if p.strip()]
    return len(paras)


def _fact_first_micro_h3_headings(description_md: str) -> list[str]:
    """Return headings whose bodies are too short (micro-sections)."""
    raw = (description_md or "").strip()
    if not raw:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    h3_re = re.compile(r"^\s*###\s+(\S.*)$")

    micro: list[str] = []
    current_heading: str | None = None
    body_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_heading, body_lines
        if current_heading is None:
            return
        body = "\n".join(body_lines).strip()
        list_items = sum(1 for ln in body.splitlines() if _LIST_ITEM_LINE_RE.match((ln or "").strip()))
        sentence_count = len(re.findall(r"[.!?]+", body))
        body_chars = len(body)

        ok = False
        if list_items >= 2:
            ok = True
        elif sentence_count >= 2:
            ok = True
        elif body_chars >= 220:
            ok = True

        if not ok:
            micro.append(current_heading.strip())
        current_heading = None
        body_lines = []

    for line in raw.splitlines():
        m = h3_re.match(line or "")
        if m:
            _flush()
            current_heading = (m.group(1) or "").strip()
            body_lines = []
            continue
        if current_heading is not None:
            body_lines.append(line)
    _flush()
    return micro


async def _llm_fact_first_description_md(
    *,
    title: str | None,
    event_type: str | None,
    facts_text_clean: Sequence[str],
    anchors: Sequence[str],
    label: str,
) -> str | None:
    """Generate description strictly from `facts_text_clean` (fact-first, C+D).

    Uses LLM for:
    - writing the narrative text,
    - coverage check (missing/extra),
    - revision loop (bounded and intentionally small to keep Gemma call counts predictable).
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None
    facts = [str(f or "").strip() for f in (facts_text_clean or []) if str(f or "").strip()]
    if not facts:
        return None

    # Keep the prompt bounded; fact-first relies on extracted facts, not raw sources.
    facts = _dedupe_source_facts(facts)[:28]
    facts = _dedupe_source_facts([_sanitize_fact_text_clean_for_prompt(f) for f in facts])[:28]
    epigraph_fact = _pick_epigraph_fact(facts)
    budget_chars = _estimate_fact_first_description_budget_chars(facts)
    desc_max_tokens = _estimate_fact_first_description_max_tokens(budget_chars=budget_chars, floor=1700)
    revise_max_tokens = _estimate_fact_first_description_max_tokens(budget_chars=budget_chars, floor=1900)

    def _cleanup_description(value: str | None) -> str | None:
        """Format-only cleanup (must not change meaning)."""
        raw = (value or "").strip()
        if not raw:
            return None
        raw = _strip_private_use(raw) or raw
        raw = _fix_inline_bullet_lists(raw) or raw
        raw = _normalize_bullet_markers(raw) or raw
        raw = _promote_review_bullets_to_blockquotes(raw) or raw
        raw = _normalize_blockquote_markers(raw) or raw
        raw = _limit_description_emojis(raw) or raw
        raw = _sanitize_description_output(raw, source_text="") or raw
        # Important: sanitize can normalize headings (e.g. strip "Facts:" prefix), so
        # dedupe/orphan-heading cleanup must happen AFTER it to avoid duplicate empty sections.
        raw = _dedupe_description(raw) or raw
        raw = _normalize_plaintext_paragraphs(raw) or raw
        raw = _ensure_minimal_description_headings(raw) or raw
        return raw.strip() or None

    def _collect_policy_issues(value: str) -> list[str]:
        desc_s = str(value or "")
        issues: list[str] = []

        # Headings count: keep it readable and consistent with the prompt.
        h3 = len(re.findall(r"(?m)^###\s+\S", desc_s))
        if h3 < 2 or h3 > 3:
            issues.append(
                f"Сейчас заголовков `###` = {h3}; нужно 2–3. "
                "Объедини близкие разделы и оставь ровно 2–3 информативных подзаголовка."
            )

        lead_paras = _fact_first_lead_paragraph_count(desc_s)
        if lead_paras == 0:
            issues.append("Добавь лид одним абзацем (1–2 предложения) перед первым `###`.")
        elif lead_paras > 1:
            issues.append(
                "Лид до первого `###` должен быть ОДНИМ абзацем (1–2 предложения), без лишних переносов строки."
            )

        h3_titles = [
            re.sub(r"\s+", " ", (m.group(1) or "")).strip()
            for m in re.finditer(r"(?m)^###\s+(.+?)\s*$", desc_s)
            if (m.group(1) or "").strip()
        ]
        if any(t.casefold() == "подробности" for t in h3_titles):
            issues.append("Не используй общий заголовок «Подробности»: убери его или замени на конкретный.")

        # Prevent repeated identical headings (they create empty/micro sections on Telegraph).
        if h3_titles:
            seen_keys: set[str] = set()
            dup_titles: list[str] = []
            for t in h3_titles:
                t_norm = re.sub(r"\s+", " ", t).strip()
                t_norm = _EMOJI_SEQ_RE.sub("", t_norm).strip()
                key = re.sub(r"\s+", " ", t_norm).strip().casefold()
                if not key:
                    continue
                if key in seen_keys:
                    dup_titles.append(t.strip())
                seen_keys.add(key)
            if dup_titles:
                uniq = _dedupe_source_facts([d for d in dup_titles if d])
                shown = ", ".join(uniq[:4])
                more = "" if len(uniq) <= 4 else f" (+{len(uniq) - 4})"
                issues.append(
                    "Повторяются одинаковые подзаголовки `### ...`: "
                    f"{shown}{more}. Заголовки должны быть уникальными — объедини секции или переименуй."
                )

        micro_h3 = _fact_first_micro_h3_headings(desc_s)
        if micro_h3:
            shown = ", ".join(micro_h3[:4])
            more = "" if len(micro_h3) <= 4 else f" (+{len(micro_h3) - 4})"
            issues.append(
                "Слишком дробные секции под `###`: "
                f"{shown}{more}. Объедини близкие разделы: 2–3 `###` всего; "
                "под каждым — 2+ предложения или список (2+ пунктов)."
            )

        if epigraph_fact:
            if not re.search(r"(?m)^>\s*\S", desc_s):
                issues.append("Если epigraph_fact не null — добавь эпиграф blockquote перед лидом (до первого `###`).")
        else:
            if re.search(r"(?m)^>\s*\S", desc_s):
                issues.append("Если epigraph_fact = null — убери blockquote-эпиграф из начала текста.")

        forbidden = _fact_first_forbidden_reasons(desc_s, anchors=anchors)
        if forbidden:
            for r in forbidden:
                if r == "посвящ*":
                    issues.append(
                        "Запрещено слово/корень «посвящ…» — убери его полностью и перефразируй без него "
                        "(например: «лекция о …», «разговор про …», «в центре — …»)."
                    )
                    continue
                issues.append(f"forbidden_marker({r})")

        return issues

    description = await _ask_gemma_text(
        _fact_first_description_prompt(
            title=title,
            event_type=event_type,
            facts_text_clean=facts,
            epigraph_fact=epigraph_fact,
        ),
        max_tokens=desc_max_tokens,
        label=f"{label}:fact_first_desc",
        temperature=0.0,
    )
    description = _cleanup_description(description)
    if not description:
        return None

    cov_prompt, cov_schema = _fact_first_coverage_prompt(facts_text_clean=facts, description=description)
    cov = await _ask_gemma_json(
        cov_prompt,
        cov_schema,
        max_tokens=500,
        label=f"{label}:fact_first_cov",
    )
    missing: list[str] = []
    extra: list[str] = []
    if isinstance(cov, dict):
        missing = [str(x).strip() for x in (cov.get("missing") or []) if str(x or "").strip()]
        extra = [str(x).strip() for x in (cov.get("extra") or []) if str(x or "").strip()]
    missing = _dedupe_source_facts(missing)[:12]
    extra = _dedupe_source_facts(extra)[:12]

    policy_issues = _collect_policy_issues(description)
    if not missing and not extra and not policy_issues:
        return description.strip() or None

    revised = await _ask_gemma_text(
        _fact_first_revise_prompt(
            title=title,
            event_type=event_type,
            epigraph_fact=epigraph_fact,
            facts_text_clean=facts,
            description=description,
            missing=missing,
            extra=extra,
            policy_issues=policy_issues,
        ),
        max_tokens=revise_max_tokens,
        label=f"{label}:fact_first_revise",
        temperature=0.0,
    )
    description2 = _cleanup_description(revised) or description
    policy_issues2 = _collect_policy_issues(description2)
    if not policy_issues2:
        return description2.strip() or None

    # Final attempt: fix policy issues only (do not redo coverage to keep call count bounded).
    # Prefer a dedicated "посвящ..." remover when it's the only remaining forbidden marker.
    forbidden2 = _fact_first_forbidden_reasons(description2, anchors=anchors)
    if forbidden2 == ["посвящ*"] and len(policy_issues2) == 1:
        revised3 = await _ask_gemma_text(
            _fact_first_remove_posv_prompt(
                title=title,
                event_type=event_type,
                epigraph_fact=epigraph_fact,
                facts_text_clean=facts,
                description=description2,
            ),
            max_tokens=revise_max_tokens,
            label=f"{label}:fact_first_remove_posv",
            temperature=0.0,
        )
        fixed = _cleanup_description(revised3) or description2
        return fixed.strip() or None

    revised2 = await _ask_gemma_text(
        _fact_first_revise_prompt(
            title=title,
            event_type=event_type,
            epigraph_fact=epigraph_fact,
            facts_text_clean=facts,
            description=description2,
            missing=missing,
            extra=extra,
            policy_issues=policy_issues2,
        ),
        max_tokens=revise_max_tokens,
        label=f"{label}:fact_first_revise_policy",
        temperature=0.0,
    )
    description3 = _cleanup_description(revised2) or description2
    return description3.strip() or None


def _has_overlong_paragraph(text: str | None, *, limit: int = 900) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    for para in re.split(r"\n{2,}", raw):
        p = para.strip()
        if not p:
            continue
        if len(p) > limit:
            return True
    return False


async def _llm_reflow_description_paragraphs(text: str) -> str | None:
    """Ask LLM to reflow paragraphs (no new facts), keeping markdown structure."""
    if SMART_UPDATE_LLM_DISABLED:
        return None
    raw = (text or "").strip()
    if not raw or len(raw) < 300:
        return None
    payload = {
        "text": _clip(raw, 6500),
    }
    prompt = (
        "Переформатируй текст описания события.\n"
        "Задача: разбить на короткие читаемые абзацы и убрать перегруженные стены текста.\n\n"
        "Правила:\n"
        "- Верни ПОЛНЫЙ текст.\n"
        "- Не добавляй новых фактов и не выдумывай.\n"
        "- Не меняй смысл и не делай рекламных клише.\n"
        "- Не добавляй и не оставляй хэштеги (`#...`) в тексте.\n"
        "- Сохраняй существующие цитаты в формате blockquote (`>`), не превращай их в обычный текст.\n"
        "- Сохраняй существующие нумерованные/маркированные списки; не превращай их в абзацы и не сокращай.\n"
        "- Можно добавить 1-2 коротких подзаголовка `###`, если это улучшает структуру.\n"
        "- В каждом абзаце держи 1-2 предложения (максимум 3 только если иначе теряется смысл).\n"
        "- Не дублируй в тексте строки формата `Дата:`, `Время:`, `Локация:`, `Билеты:`: эти данные уже показаны в карточке сверху.\n"
        "- Каждому абзацу старайся держать длину <= 600-800 символов.\n"
        "- Не оставляй обрывов фраз/предложений после правок.\n"
        f"{SMART_UPDATE_YO_RULE}\n"
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE}\n\n"
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    out = await _ask_gemma_text(prompt, max_tokens=1200, label="reflow", temperature=0.0)
    return out.strip() if out else None


_FIRST_PERSON_QUOTE_RE = re.compile(
    r"(?is)^\s*(?:мне кажется|я думаю|я считаю|я вижу|я замечаю|я уверен)\b"
)

_REPORTED_SPEECH_RE = re.compile(
    r"(?is)\b(?:отмечает|подч[её]ркивает|говорит|считает|пишет)\s*,?\s*что\s+(.+)$"
)

_SCENE_HINT_RE = re.compile(r"(?is)\b(основн\w+|мал\w+)\s+сцен\w+\b")
_REVIEW_CONTEXT_RE = re.compile(
    r"(?iu)\b("
    r"отзыв\w*|"
    r"реценз\w*|"
    r"комментар\w*|"
    r"впечатлен\w*|"
    r"мнения\w*|"
    r"зрител\w*|"
    r"восторг\w*|"
    r"говорят|"
    r"пишут"
    r")\b"
)
_REVIEW_LIST_ITEM_RE = re.compile(
    r"(?iu)^\s*(?:[-*•]|\d{1,3}[.)])\s+"
    r"(?P<who>[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,24}(?:\s+[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,24}){0,2}(?:\s*(?:,|\(|—)\s*[^:]{1,60})?)"
    r"\s*:\s*(?P<body>\S.+?)\s*$"
)


def _promote_first_person_quotes_to_blockquotes(text: str | None) -> str | None:
    """Format direct speech as Markdown blockquotes when it looks like a quote.

    This improves Telegraph readability and avoids "quote-like" sentences blending
    into narration. We keep this conservative to avoid over-formatting.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    out_paras: list[str] = []
    sent_split = re.compile(r"(?<=[.!?…])\s+")
    for para in paragraphs:
        if para.lstrip().startswith(">"):
            out_paras.append(para)
            continue
        # Only touch "normal" paragraphs (no headings/lists).
        if re.match(r"^\s*#{1,6}\s+\S", para) or re.match(r"^\s*[-*•]\s+\S", para):
            out_paras.append(para)
            continue
        sents = [s.strip() for s in sent_split.split(para) if s.strip()]
        if not sents:
            out_paras.append(para)
            continue
        if len(sents) == 1:
            only = re.sub(r"\s+", " ", sents[0]).strip()
            if 25 <= len(only) <= 220 and _FIRST_PERSON_QUOTE_RE.match(only.lower()):
                out_paras.append(f"> {only}")
            else:
                out_paras.append(para)
            continue
        kept: list[str] = []
        quotes: list[str] = []
        for s in sents:
            s_norm = re.sub(r"\s+", " ", s).strip()
            if 25 <= len(s_norm) <= 220 and _FIRST_PERSON_QUOTE_RE.match(s_norm.lower()):
                quotes.append(s_norm)
            else:
                kept.append(s_norm)
        if kept:
            out_paras.append(" ".join(kept).strip())
        for q in quotes[:2]:
            out_paras.append(f"> {q}")
    merged = "\n\n".join(p for p in out_paras if p.strip()).strip()
    return merged or None


def _promote_inline_quoted_direct_speech_to_blockquotes(text: str | None) -> str | None:
    """Turn inline direct speech in «...» into a standalone Markdown blockquote.

    This is a deterministic fallback for cases where the model put the quote inside
    a normal paragraph like:
      `... отмечает: «Мне кажется, ...»`
    but we want Telegraph to render it as `<blockquote>`.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if re.search(r"(?m)^>\s+", raw):
        return raw
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return raw

    out: list[str] = []
    promoted = False

    quote_re = re.compile(r"(?s)«(?P<q>[^»]{25,900})»")
    context_re = re.compile(
        r"(?i)\b(цитат\w*|по\s+словам|говорит|отмечает|подч[её]ркивает|пишет|организатор\w*|автор\w*)\b"
    )
    for para in paragraphs:
        if promoted:
            out.append(para)
            continue
        if para.lstrip().startswith(">"):
            out.append(para)
            continue
        if re.match(r"^\s*#{1,6}\s+\S", para) or re.match(r"^\s*[-*•]\s+\S", para):
            out.append(para)
            continue

        m = quote_re.search(para)
        if not m:
            out.append(para)
            continue
        q = re.sub(r"\s+", " ", (m.group("q") or "").strip())
        if not q:
            out.append(para)
            continue
        q_words = re.findall(r"[a-zа-яё0-9]{2,}", q, flags=re.IGNORECASE)
        looks_like_title = len(q_words) <= 6 and not re.search(r"[.!?…]", q)
        promote = bool(_FIRST_PERSON_QUOTE_RE.match(q.lower()))
        if not promote:
            if context_re.search(para):
                promote = True
            elif not looks_like_title and len(q_words) >= 10 and len(q) >= 60:
                promote = True
        if not promote:
            out.append(para)
            continue

        before = (para[: m.start()] or "").rstrip()
        after = (para[m.end() :] or "").lstrip()
        if before.endswith(":"):
            before = before[:-1].rstrip() + "."
        merged = (before + " " + after).strip()
        merged = re.sub(r"\s+", " ", merged).strip()
        if merged:
            out.append(merged)
        out.append(f"> {q}")
        promoted = True

    updated = "\n\n".join(p for p in out if p.strip()).strip()
    return updated or None


def _promote_review_bullets_to_blockquotes(text: str | None) -> str | None:
    """Render simple audience reviews as Markdown blockquotes.

    Pattern:
      - `<Name>: <review text>`
    under a nearby review context ("отзывы", "зрители", etc.)

    This is a formatting-only helper to improve Telegraph readability; it must not
    drop or rewrite text.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return None

    out: list[str] = []

    def _is_fully_quoted(v: str) -> bool:
        s = (v or "").strip()
        if len(s) < 2:
            return False
        pairs = [
            ("«", "»"),
            ('"', '"'),
            ("“", "”"),
            ("„", "“"),
        ]
        return any(s.startswith(op) and s.endswith(cl) for op, cl in pairs)

    def _quote_review_body(v: str) -> str:
        s = (v or "").strip()
        if not s:
            return s
        if _is_fully_quoted(s):
            return s
        # If the author already started/ended quoting, don't try to "fix" it.
        if s.startswith(("«", '"', "“", "„")) or s.endswith(("»", '"', "”", "“")):
            return s
        return f"«{s}»"

    for para in paragraphs:
        lines = [ln.rstrip() for ln in (para or "").splitlines()]
        if not lines:
            continue

        # Find a contiguous list block inside this paragraph.
        start: int | None = None
        for i, ln in enumerate(lines):
            if _LIST_ITEM_LINE_RE.match((ln or "").strip()):
                start = i
                break
        if start is None:
            out.append(para)
            continue

        end = start
        while end < len(lines) and _LIST_ITEM_LINE_RE.match((lines[end] or "").strip()):
            end += 1

        list_lines = [lines[i].strip() for i in range(start, end) if (lines[i] or "").strip()]
        if len(list_lines) < 2:
            out.append(para)
            continue

        items: list[tuple[str, str]] = []
        for ln in list_lines:
            m = _REVIEW_LIST_ITEM_RE.match(ln)
            if not m:
                items = []
                break
            who = (m.group("who") or "").strip()
            body = (m.group("body") or "").strip()
            if not who or not body:
                items = []
                break
            items.append((who, body))
        if not items:
            out.append(para)
            continue

        preface = " ".join((ln or "").strip() for ln in lines[:start] if (ln or "").strip()).strip()
        prev = out[-1] if out else ""
        context = "\n".join([preface, prev]).strip()
        if not _REVIEW_CONTEXT_RE.search(context):
            out.append(para)
            continue

        if preface:
            out.append(preface)
        for who, body in items:
            q = _quote_review_body(body)
            # Two-line quote: quote text + attribution.
            out.append(f"> {q}\n> — {who}")

        tail = "\n".join((ln or "").rstrip() for ln in lines[end:] if (ln or "").strip()).strip()
        if tail:
            out.append(tail)

    merged = "\n\n".join(p for p in out if (p or "").strip()).strip()
    return merged or None


def _drop_reported_speech_duplicates(text: str | None) -> str | None:
    """Remove paraphrased "X notes that ..." if the same clause exists as a direct quote.

    Goal: avoid duplicate meaning when we have both:
      - "Режиссёр ... отмечает, что <clause>."
      - "> Мне кажется, что <clause>."
    """
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return None

    quote_clauses: list[str] = []
    for p in paragraphs:
        if not p.lstrip().startswith(">"):
            continue
        q = p.lstrip()[1:].strip()
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            continue
        # Prefer the part after "что" to match reported speech.
        parts = re.split(r"(?i)\bчто\b", q, maxsplit=1)
        clause = (parts[1] if len(parts) == 2 else q).strip()
        clause = clause.strip(" .,!?:;—-").strip()
        if len(clause) >= 20:
            quote_clauses.append(clause.casefold())
    if not quote_clauses:
        return raw

    sent_split = re.compile(r"(?<=[.!?…])\s+")
    out_paras: list[str] = []
    for para in paragraphs:
        if para.lstrip().startswith(">"):
            out_paras.append(para)
            continue
        # Keep headings/lists as is.
        if re.match(r"^\s*#{1,6}\s+\S", para) or re.match(r"^\s*[-*•]\s+\S", para):
            out_paras.append(para)
            continue
        sents = [s.strip() for s in sent_split.split(para) if s.strip()]
        kept: list[str] = []
        for s in sents:
            s_norm = re.sub(r"\s+", " ", s).strip()
            m = _REPORTED_SPEECH_RE.search(s_norm)
            if m:
                clause = (m.group(1) or "").strip()
                clause = clause.strip(" .,!?:;—-").strip()
                clause_cf = clause.casefold()
                if len(clause_cf) >= 20 and any(
                    (clause_cf in qc) or (qc in clause_cf) for qc in quote_clauses
                ):
                    # Drop the paraphrase if we already have the direct quote.
                    continue
            kept.append(s_norm)
        merged = " ".join(kept).strip()
        if merged:
            out_paras.append(merged)
    return "\n\n".join(out_paras).strip() or None


def _normalize_blockquote_markers(text: str | None) -> str | None:
    """Ensure Markdown blockquotes are standalone paragraphs (so Telegraph renders <blockquote>)."""
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # If a blockquote marker leaked into the middle of a paragraph, split it out.
    raw = re.sub(r"(?<=\S)[ \t]+>\s+", "\n\n> ", raw)
    # Remove leading spaces before a blockquote marker.
    raw = re.sub(r"(?m)^[ \t]+(>\s+)", r"\1", raw)
    # Ensure a blank line before the *start* of any blockquote block.
    # Do not insert blank lines between consecutive quote lines (`> ...\n> ...`),
    # otherwise multi-line quotes (e.g. quote + attribution) break apart.
    lines = raw.split("\n")
    out_lines: list[str] = []
    quote_line_re = re.compile(r"^\s*>\s+\S")
    for ln in lines:
        if quote_line_re.match(ln):
            if out_lines:
                prev = out_lines[-1]
                if prev.strip() and not quote_line_re.match(prev):
                    out_lines.append("")
            out_lines.append(ln.lstrip())
        else:
            out_lines.append(ln)
    raw = "\n".join(out_lines)
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
    return raw or None


def _dedupe_paragraphs_preserving_formatting(text: str | None) -> str | None:
    """Remove repeated paragraphs while preserving paragraph boundaries."""
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if len(paragraphs) < 2:
        return raw
    seen: set[str] = set()
    out: list[str] = []
    for p in paragraphs:
        cleaned = _ZERO_WIDTH_RE.sub("", p).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        key = cleaned.lower().rstrip(".!?…")
        # Only dedupe "meaningful" paragraphs, keep small fragments intact.
        if len(key) >= 40 and key in seen:
            continue
        seen.add(key)
        out.append(p)
    return "\n\n".join(out).strip() or None


def _split_overlong_first_person_blockquotes(text: str | None) -> str | None:
    """Keep first-person quotes as blockquotes, but avoid swallowing narration into the quote."""
    raw = (text or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return None
    sent_split = re.compile(r"(?<=[.!?…])\s+")
    out: list[str] = []
    for para in paragraphs:
        if not para.lstrip().startswith(">"):
            out.append(para)
            continue
        # Collapse multi-line blockquotes into a single text.
        q = re.sub(r"(?m)^\s*>\s*", "", para).strip()
        q = re.sub(r"\s+", " ", q).strip()
        if not q:
            continue
        sents = [s.strip() for s in sent_split.split(q) if s.strip()]
        if len(sents) <= 1:
            out.append(f"> {q}")
            continue
        first = re.sub(r"\s+", " ", sents[0]).strip()
        if _FIRST_PERSON_QUOTE_RE.match(first.lower()):
            out.append(f"> {first}")
            tail = " ".join(re.sub(r"\s+", " ", s).strip() for s in sents[1:] if s.strip()).strip()
            if tail:
                out.append(tail)
        else:
            out.append(f"> {q}")
    return "\n\n".join(out).strip() or None


def _preserve_blockquotes_from_previous_description(
    *,
    before_description: str | None,
    merged_description: str | None,
    event_title: str | None,
    max_quotes: int = 2,
) -> str | None:
    """Preserve meaningful existing blockquotes across LLM merges.

    LLM merges (especially when adding site/parser info) can sometimes "flatten" direct speech
    into reported speech and drop the original quote. Product expectation: if we already had a
    relevant direct quote for the event, keep it as a Markdown blockquote in the merged text.

    We keep this conservative:
    - only preserve explicit Markdown blockquote paragraphs from the previous description;
    - only preserve quotes that mention the event title tokens (to avoid carrying quotes about
      other events from multi-event posts);
    - only append quotes that are missing from the merged description.
    """
    before = (before_description or "").strip()
    after = (merged_description or "").strip()
    if not before or not after:
        return merged_description

    tokens = _title_tokens(event_title)
    before_norm = before.replace("\r\n", "\n").replace("\r", "\n")
    after_cf = after.casefold()

    quotes: list[str] = []
    for para in [p.strip() for p in re.split(r"\n{2,}", before_norm) if p.strip()]:
        if not para.lstrip().startswith(">"):
            continue
        q = para.lstrip()[1:].strip()
        q = re.sub(r"\s+", " ", q).strip()
        q = q.strip("\u200b\u200c\u200d\u2060").strip()
        if not q:
            continue
        if len(q) < 20 or len(q) > 280:
            continue
        if tokens and not any(tok in q.casefold() for tok in tokens):
            continue
        if q.casefold() in after_cf:
            continue
        quotes.append(q)
        if len(quotes) >= max_quotes:
            break

    if not quotes:
        return merged_description

    appended = after.rstrip() + "\n\n" + "\n\n".join(f"> {q}" for q in quotes)
    return _normalize_plaintext_paragraphs(appended) or appended


def _append_missing_scene_hint(
    *,
    description: str | None,
    source_text: str | None,
) -> str | None:
    """Deterministic safety-net: keep 'Основная/Малая сцена' hints when present in sources."""
    desc = (description or "").strip()
    if not desc:
        return None
    if re.search(r"(?is)\b(основн\w+|мал\w+)\s+сцен\w+\b", desc):
        return desc
    src = (source_text or "").strip()
    if not src:
        return desc
    m = _SCENE_HINT_RE.search(src)
    if not m:
        return desc
    kind = (m.group(1) or "").lower()
    phrase = "на Основной сцене" if "основ" in kind else "на Малой сцене"
    sentence = f"Спектакль пройдёт {phrase}."
    if sentence.lower() in desc.lower():
        return desc
    return (desc + "\n\n" + sentence).strip()


def _fallback_digest_from_description(description: str | None) -> str | None:
    """Deterministic fallback digest: use the first 1-2 sentences from description."""
    raw = (description or "").strip()
    if not raw:
        return None
    # Drop headings and blockquotes.
    lines = []
    for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = (ln or "").strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith(">"):
            continue
        lines.append(s)
    if not lines:
        return None
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return None
    digest = parts[0]
    if len(digest) < 80 and len(parts) >= 2:
        digest = f"{digest} {parts[1]}".strip()
    digest = _clip_to_readable_boundary(digest, 240)
    return _clean_search_digest(digest)

def _clip_to_readable_boundary(text: str | None, limit: int) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if len(raw) <= limit:
        return raw
    # Prefer cutting at sentence/paragraph boundaries to avoid dangling tails.
    boundary = max(
        raw.rfind("\n\n", 0, limit + 1),
        raw.rfind(". ", 0, limit + 1),
        raw.rfind("! ", 0, limit + 1),
        raw.rfind("? ", 0, limit + 1),
        raw.rfind("… ", 0, limit + 1),
    )
    if boundary >= int(limit * 0.65):
        return raw[: boundary + 1].rstrip()
    return _clip(raw, limit)


_STYLE_TERM_RE = re.compile(
    r"\bв\s+(?:стиле|жанре)\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{3,})",
    re.IGNORECASE,
)


def _append_missing_fact_sentences(
    *,
    base: str,
    rewritten: str,
    max_sentences: int = 2,
    ensure_coverage: bool = False,
) -> str:
    """Append a small number of factual sentences that the rewrite missed.

    Deterministic safety-net: do not invent facts, only reuse snippets from the source.
    """
    base_raw = (base or "").strip()
    out_raw = (rewritten or "").strip()
    if not base_raw or not out_raw:
        return out_raw or base_raw

    base_cf = base_raw.casefold()
    out_cf = out_raw.casefold()

    required_terms: set[str] = set()
    for m in _STYLE_TERM_RE.finditer(base_raw):
        term = (m.group(1) or "").strip().casefold()
        if term and term not in out_cf:
            required_terms.add(term)
    if "фламенко" in base_cf and "фламенко" not in out_cf:
        required_terms.add("фламенко")

    out_norm = re.sub(r"\s+", " ", out_raw).strip().lower()

    candidates: list[str] = []
    for chunk in re.split(r"(?:\n{2,}|(?<=[.!?])\s+)", base_raw):
        s = (chunk or "").strip()
        if len(s) < 30:
            continue
        candidates.append(s)

    added: list[str] = []
    for term in sorted(required_terms):
        if len(added) >= max_sentences:
            break
        for s in candidates:
            s_norm = re.sub(r"\s+", " ", s).strip().lower()
            if term in s.casefold():
                if s_norm in out_norm:
                    break
                added.append(s)
                break

    if ensure_coverage and len(added) < max_sentences:
        missing: list[str] = []
        seen_missing: set[str] = set()
        critical_missing: list[str] = []
        for chunk in re.split(r"(?:\n{2,}|(?<=[.!?…])\s+|\n)", base_raw):
            sent = _normalize_candidate_sentence(chunk)
            is_critical = _is_coverage_critical_sentence(sent)
            if _is_low_signal_sentence(sent) and not is_critical:
                continue
            sent_norm = re.sub(r"\s+", " ", sent).strip().lower()
            if not sent_norm:
                continue
            if sent_norm in out_norm:
                continue
            if sent_norm in seen_missing:
                continue
            seen_missing.add(sent_norm)
            missing.append(sent)
            if is_critical:
                critical_missing.append(sent)

        if missing:
            for critical in critical_missing:
                if len(added) >= max_sentences:
                    break
                critical_norm = re.sub(r"\s+", " ", critical).strip().lower()
                if not critical_norm or critical_norm in out_norm:
                    continue
                if any(
                    re.sub(r"\s+", " ", a).strip().lower() == critical_norm
                    for a in added
                ):
                    continue
                added.append(critical)

            ranked = sorted(
                range(len(missing)),
                key=lambda idx: (
                    _sentence_quality_score(missing[idx])
                    + (400 if _is_coverage_critical_sentence(missing[idx]) else 0),
                    -idx,
                ),
                reverse=True,
            )
            for idx in ranked:
                if len(added) >= max_sentences:
                    break
                candidate_sent = missing[idx]
                candidate_norm = re.sub(r"\s+", " ", candidate_sent).strip().lower()
                if candidate_norm in out_norm:
                    continue
                if any(
                    re.sub(r"\s+", " ", a).strip().lower() == candidate_norm
                    for a in added
                ):
                    continue
                added.append(candidate_sent)

    if not added:
        return out_raw
    merged = out_raw.rstrip() + "\n\n" + "\n\n".join(added)
    return _normalize_plaintext_paragraphs(merged) or merged


def _looks_like_ticket_giveaway(*texts: str | None) -> bool:
    combined = "\n".join(t for t in texts if t and t.strip())
    if not combined:
        return False
    value = combined.casefold()
    # Require both giveaway + tickets signals to reduce false positives.
    return bool(_GIVEAWAY_RE.search(value) and _TICKETS_RE.search(value))


def _looks_like_promo_or_congrats(*texts: str | None) -> bool:
    combined = "\n".join(t for t in texts if t and t.strip())
    if not combined:
        return False
    value = combined.casefold()
    # Congratulation posts are treated as non-event content by product requirements.
    if _CONGRATS_RE.search(value):
        return True
    # Pure promotions (discounts/coupons) without event anchors must not become events/sources.
    if _PROMO_STRIP_RE.search(value):
        if not _has_datetime_signals(combined) and not _EVENT_SIGNAL_RE.search(value):
            # Keep it conservative: if there's no date/time and no event-type signals, it's promo-only.
            return True
    return False


_RU_MONTHS_GENITIVE_RE = (
    "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
)
_DEADLINE_RE = re.compile(
    rf"(?i)\b(?:до|дедлайн|срок(?:\s+подачи)?|успеть)\s+"
    rf"(?:\d{{1,2}}[./]\d{{1,2}}(?:[./](?:19|20)\d{{2}})?|\d{{1,2}}\s+(?:{_RU_MONTHS_GENITIVE_RE}))\b"
)
_FROM_DATE_RE = re.compile(
    rf"(?iu)\bс\s+\d{{1,2}}(?:[./]\d{{1,2}}(?:[./](?:19|20)\d{{2}})?|\s+(?:{_RU_MONTHS_GENITIVE_RE}))\b"
)
_OPEN_CALL_RE = re.compile(
    r"(?iu)\b("
    r"open\s*call|"
    r"опен\s*колл|"
    r"опенколл|"
    r"конкурсн\w*\s+отбор|"
    r"при[её]м\s+заявок|"
    r"подать\s+заявк\w*|"
    r"заявк\w*\s+принима\w*"
    r")\b"
)
_NON_EVENT_NOTICE_RE = re.compile(
    r"(?i)\b("
    r"налогов\w*\s+вычет|"
    r"госуслуг\w*|"
    r"государственн\w*\s+услуг\w*|"
    r"субсид\w*|"
    r"льгот\w*|"
    r"компенсац\w*|"
    r"пособи\w*|"
    r"постановлени\w*|"
    r"перечен\w*|"
    r"утвержда\w*|"
    r"заявк\w*\s+на\s+включени\w*"
    r")\b"
)
_VENUE_STATUS_UPDATE_RE = re.compile(
    r"(?iu)\b("
    r"отсрочк\w*|"
    r"продл(?:или|им|ят|ение|ен\w*)|"
    r"город\s+может\s+потерять|"
    r"можем\s+потерять|"
    r"потеря\w*|"
    r"закры(вают|ваем|вается|тие|ют)\w*|"
    r"высел\w*|"
    r"аренд\w*|"
    r"съезжа\w*|"
    r"петици\w*|"
    r"сбор\s+средств|"
    r"поддерж(ите|ать)|"
    r"нужна\s+помощь|"
    r"помогите"
    r")\b"
)
_COURSE_PROMO_RE = re.compile(
    r"(?i)\b("
    r"старт\s+курс\w*|"
    r"набор\s+на\s+курс|"
    r"курс\w*|"
    r"обучени\w*|"
    r"программ\w*\s+курс\w*|"
    r"поток\s+|"
    r"модул\w*|"
    r"домашн\w*\s+задан\w*|"
    r"куратор\w*|"
    r"сертификат\w*|"
    r"професси\w*\s+переподготовк\w*"
    r")\b"
)

_SERVICE_PROMO_OCCASION_RE = re.compile(
    r"(?iu)\b("
    r"выпускн\w*|"
    r"свадьб\w*|"
    r"корпоратив\w*|"
    r"тимбилдинг\w*|"
    r"день\s+рождени\w*|"
    r"юбиле\w*|"
    r"банкет\w*|"
    r"фуршет\w*"
    r")\b"
)
_SERVICE_PROMO_OFFER_RE = re.compile(
    r"(?iu)\b("
    r"пакетн\w*\s+программ\w*|"
    r"комплексн\w*\s+программ\w*|"
    r"бронирован\w*|"
    r"брониру\w*|"
    r"заброниру\w*|"
    r"бронь\b|"
    r"заказ\w*|"
    r"организу\w*|"
    r"помож\w*\s+организ\w*|"
    r"стоимость\s*(?:—|-|:)?\s*от\b|"
    r"/\s*чел\b|"
    r"на\s+чел(?:овек[ао]?)?\b"
    r")\b"
)
_SERVICE_PROMO_CONTACT_RE = re.compile(
    r"(?iu)\b("
    r"телефон|"
    r"whatsapp|"
    r"вайбер|"
    r"viber|"
    r"telegram|"
    r"tg|"
    r"max|"
    r"звоните|"
    r"пишите|"
    r"контакт\w*"
    r")\b|"
    r"(?:\+7|8)\s*\(?\d{3}\)?\s*\d{3}[\s-]*\d{2}[\s-]*\d{2}"
)

_WORK_SCHEDULE_RE = re.compile(
    r"(?iu)\b("
    r"график\s+работ\w*|"
    r"расширенн\w*\s+график\w*|"
    r"режим\s+работ\w*|"
    r"часы\s+работ\w*|"
    r"в\s+праздничн\w*\s+дни|"
    r"в\s+выходн\w*\s+дни|"
    r"праздничн\w*\s+дни|"
    r"выходн\w*\s+дни|"
    r"санитарн\w*\s+день|"
    r"не\s+работа(?:ет|ют)|"
    r"работаем\s+по\s+(?:обычн\w*|нов\w*)\s+график\w*|"
    r"музе[йя]\s+работа(?:ет|ют)"
    r")\b"
)

_WORK_SCHEDULE_DETAIL_RE = re.compile(
    r"(?iu)\b("
    r"понедельник|вторник|среда|четверг|пятниц[ауы]|суббот[ауы]|воскресень[ея]|"
    r"пн|вт|ср|чт|пт|сб|вс|"
    r"выходн\w*|санитарн\w*|"
    r"с\s*\d{1,2}[:.]\d{2}\s*до\s*\d{1,2}[:.]\d{2}|"
    r"\d{1,2}[:.]\d{2}\s*[–—-]\s*\d{1,2}[:.]\d{2}|"
    r"\d{1,2}[./]\d{1,2}"
    r")\b"
)

_EVENT_ACTION_INVITE_RE = re.compile(
    r"(?iu)\b("
    r"состо(ится|ятся)|"
    r"пройд(ёт|ет|ут)|"
    r"приглаша(ем|ю|ет)|"
    r"встречаемс\w*|"
    r"открыт(ие|ый)\s+урок|"
    r"открыти(е|я)\s+выставк\w*|"
    r"старт(?:ует|уем)\s+в\b|"
    r"начал(о|а)\s+в\b"
    r")\b"
)

_PRICE_CONTEXT_RE = re.compile(
    r"(?iu)(?:"
    r"\b(?:цена|стоимост\w*|руб(?:\.|л[её]й|ля|лей)?)\b"
    r"|₽"
    r"|р\."
    r")"
)
_PRICE_NUMBER_RE = re.compile(r"(?u)\b\d{2,6}\b")
_NON_TICKET_MONEY_CONTEXT_RE = re.compile(
    r"(?iu)\b("
    r"компенсац\w*|"
    r"вознагражден\w*|"
    r"выплат\w*|"
    r"гонорар\w*|"
    r"стипенди\w*|"
    r"зарплат\w*|"
    r"оклад\w*|"
    r"преми\w*|"
    r"приз\w*|"
    r"подар\w*|"
    r"к[эе]шб[эе]к|"
    r"cashback"
    r")\b"
)
_TICKET_PRICE_CONTEXT_RE = re.compile(
    r"(?iu)\b("
    r"билет\w*|"
    r"вход\w*|"
    r"стоимост\w*|"
    r"цена|"
    r"взнос\w*|"
    r"донат\w*|"
    r"платн\w*"
    r")\b"
)
_BLOOD_DONATION_CONTEXT_RE = re.compile(
    r"(?iu)\b("
    r"день\s+донора|"
    r"донорск\w*\s+акци\w*|"
    r"донор\w*\s+(?:кров\w*|плазм\w*|тромбоцит\w*|костн\w*\s+мозг\w*)|"
    r"донорств\w*\s+(?:кров\w*|плазм\w*|тромбоцит\w*|костн\w*\s+мозг\w*)|"
    r"сдач\w*\s+(?:кров\w*|плазм\w*|тромбоцит\w*)|"
    r"станц\w*\s+перелив\w*\s+кров\w*|"
    r"центр\w*\s+кров\w*|"
    r"служб\w*\s+кров\w*|"
    r"кроводач\w*"
    r")\b"
)
_EVENT_INVITE_RE = re.compile(
    r"(?i)\b("
    r"состо(ится|ятся)|"
    r"пройд(ёт|ет|ут)|"
    r"приглаша(ем|ю|ет)|"
    r"встречаемс\w*|"
    r"открыт(ие|ый)\s+урок|"
    r"мастер-?класс|"
    r"лекци\w*|"
    r"спектакл\w*|"
    r"концерт\w*|"
    r"показ\w*|"
    r"выставк\w*"
    r")\b"
)
_EVENT_HAPPENS_VERB_RE = re.compile(
    r"(?i)\b("
    r"состо(ится|ятся)|"
    r"пройд(ёт|ет|ут)|"
    r"приглаша(ем|ю|ет)|"
    r"встречаемс\w*|"
    r"начал(о|а|ется|нутся)|"
    r"стартует|"
    r"начинаем"
    r")\b"
)

_TOO_SOON_NOTICE_RE = re.compile(
    r"(?iu)\b("
    r"уже\s+через\s+\d{1,3}\s+минут|"
    r"через\s+\d{1,3}\s+минут\s+(?:старт|начал\w*|начинаем)|"
    r"стартует\s+через\s+\d{1,3}\s+минут|"
    r"начинаем\s+через\s+\d{1,3}\s+минут"
    r")\b"
)
_ONLINE_EVENT_RE = re.compile(
    r"(?iu)\b("
    r"онлайн|zoom|вебинар|webinar|телемост|"
    r"стрим|трансляц\w*|youtube|"
    r"подключайтес\w*|ссылка\s+на\s+подключен\w*"
    r")\b"
)
_BOOK_REVIEW_RE = re.compile(r"(?iu)(#книг\w*|\bкниг\w*\b|\bчтени\w*\b|\bфл[эе]т\s+уайт\b)")
_BOOK_EVENT_KEEP_RE = re.compile(
    r"(?iu)\b(обсуждени\w*|книжн\w*\s+клуб|встреч\w*|дискусси\w*|презентац\w*|лекци\w*)\b"
)
_PHOTO_DAY_RE = re.compile(r"(?iu)\b(фото\s*дня|фотодня|photo\s+of\s+the\s+day)\b")
_PHOTO_DAY_KEEP_ACTION_RE = re.compile(
    r"(?iu)\b("
    r"приглаша(ем|ю|ет)|"
    r"приходите|"
    r"жд[её]м\s+вас|"
    r"состо(ится|ятся)|"
    r"пройд(ёт|ет|ут)|"
    r"открыти(е|я)|"
    r"начал(о|а)|"
    r"презентац\w*|"
    r"экскурси\w*|"
    r"лекци\w*|"
    r"концерт\w*|"
    r"спектакл\w*|"
    r"мастер-?класс"
    r")\b"
)
_PHOTO_DAY_TICKET_RE = re.compile(r"(?iu)\b(регистрац\w*|билет\w*|вход\w*|оплат\w*|бронь)\b")
_PHOTO_DAY_PERIOD_RE = re.compile(
    rf"(?iu)\b(?:с\s+\d{{1,2}}\s+(?:{_RU_MONTHS_GENITIVE_RE})\s+по\s+\d{{1,2}}\s+(?:{_RU_MONTHS_GENITIVE_RE})|"
    rf"до\s+\d{{1,2}}\s+(?:{_RU_MONTHS_GENITIVE_RE})|"
    rf"по\s+\d{{1,2}}\s+(?:{_RU_MONTHS_GENITIVE_RE}))\b"
)


def _looks_like_too_soon_notice(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    return bool(_TOO_SOON_NOTICE_RE.search(combined))


def _looks_like_online_event(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    return bool(_ONLINE_EVENT_RE.search(combined))


def _looks_like_book_review_not_event(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    if not _BOOK_REVIEW_RE.search(combined):
        return False
    # If it's framed as an event (discussion/club/lecture), keep it.
    if _BOOK_EVENT_KEEP_RE.search(combined):
        return False
    # If there is a concrete start time/date anchor, treat as a real event instead of a review.
    if re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", combined):
        return False
    if re.search(r"\b\d{1,2}[./]\d{1,2}\b", combined):
        return False
    if re.search(r"(?iu)\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\b", combined):
        return False
    return True


def _looks_like_photo_day_not_event(
    title: str | None,
    text: str | None,
    *,
    candidate: EventCandidate | None = None,
) -> bool:
    """Detect rubric posts like "Фото дня" that should not become events.

    "Фото дня" is a strong non-event hint, but we keep the post if it still contains
    strong event signals (explicit time, period, tickets/registration, or an invite).
    """
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    if not _PHOTO_DAY_RE.search(low):
        return False
    # Keep if the candidate already has explicit scheduling/ticket fields.
    if candidate is not None:
        if str(getattr(candidate, "time", "") or "").strip():
            return False
        if str(getattr(candidate, "end_date", "") or "").strip():
            return False
        if str(getattr(candidate, "ticket_link", "") or "").strip():
            return False
        if (
            getattr(candidate, "ticket_price_min", None) is not None
            or getattr(candidate, "ticket_price_max", None) is not None
        ):
            return False
    # Keep if there's an explicit start time in the text.
    if re.search(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b", combined):
        return False
    # Keep if the post contains an exhibition-like period ("с ... по ...", "до ...", "по ...").
    if _PHOTO_DAY_PERIOD_RE.search(low):
        return False
    # Keep if there's ticket/registration language.
    if _PHOTO_DAY_TICKET_RE.search(low):
        return False
    # Keep if the post looks like an actual invite/announcement.
    if _PHOTO_DAY_KEEP_ACTION_RE.search(low):
        return False
    return True


def _looks_like_non_event_notice(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    if not _NON_EVENT_NOTICE_RE.search(low):
        return False
    # If the post looks like an actionable announcement for a specific event, keep it.
    if _EVENT_INVITE_RE.search(combined):
        return False
    # Deadline-heavy informational posts are not events.
    if _DEADLINE_RE.search(combined):
        return True
    # Strong "service notice" keywords alone are enough in practice.
    return True


def _looks_like_venue_status_update_not_event(title: str | None, text: str | None) -> bool:
    """Detect non-event status updates about a venue/organisation.

    Examples: "город может потерять площадку с 1 мая", "дана отсрочка до 1 июня",
    eviction/lease issues, petitions, fundraising — these should not become events
    to attend.
    """
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    if not _VENUE_STATUS_UPDATE_RE.search(low):
        return False
    # Require a deadline-like date anchor to avoid skipping unrelated discussions.
    if not (_DEADLINE_RE.search(combined) or _FROM_DATE_RE.search(low)):
        return False
    # If the post looks like a real event invite/announcement, keep it.
    if _EVENT_INVITE_RE.search(combined):
        return False
    # If there is an explicit start time, treat as a likely event.
    if re.search(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b", combined):
        return False
    return True


def _looks_like_open_call_not_event(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    if not _OPEN_CALL_RE.search(low):
        return False
    # If the post is clearly an actionable one-off event invite with a concrete start time,
    # do not treat it as an open-call notice.
    if re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", combined) and _EVENT_INVITE_RE.search(combined):
        return False
    return True


def _looks_like_course_promo(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    if not _COURSE_PROMO_RE.search(combined):
        return False
    # If it's clearly a one-off masterclass/lecture with concrete start time, keep it.
    if re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", combined) and _EVENT_INVITE_RE.search(combined):
        return False
    # "Старт курса" and multi-session language strongly indicates non-event promo content.
    if re.search(r"(?i)\bна\s+кажд(ом|ом\s+из)\s+заняти\w*\b", combined):
        return True
    if re.search(r"(?i)\bстарт\s+курс\w*\b", combined):
        return True
    return True


def _looks_like_service_promo_not_event(title: str | None, text: str | None) -> bool:
    """Detect promotions for "event hosting services" (packages/booking), not one-off events.

    Example: "Выпускные 2026 ... пакетные программы ... бронирование открыто ... телефон ...".
    """
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    # If there are explicit date/time anchors, treat it as a potential real event.
    # This filter targets promos without a concrete schedule.
    if _has_datetime_signals(combined):
        return False
    low = combined.casefold()
    if not _SERVICE_PROMO_OCCASION_RE.search(low):
        return False
    if not _SERVICE_PROMO_OFFER_RE.search(low):
        return False
    # Reduce false positives: require either explicit contacts or price-like signals.
    if _SERVICE_PROMO_CONTACT_RE.search(combined):
        return True
    if _PRICE_CONTEXT_RE.search(combined) and _PRICE_NUMBER_RE.search(combined):
        return True
    return False


def _looks_like_work_schedule_notice(title: str | None, text: str | None) -> bool:
    """Institution work schedules must not become events."""
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    has_schedule_headline = bool(_WORK_SCHEDULE_RE.search(low))
    has_schedule_details = bool(_WORK_SCHEDULE_DETAIL_RE.search(combined))
    if not (has_schedule_headline or (("музей" in low or "библиотек" in low) and has_schedule_details)):
        return False
    # If the post is clearly announcing a concrete attendable event, keep it.
    # Use action verbs, not generic nouns ("выставка/концерт"), otherwise
    # work-schedule notices with occasional cultural terms slip through.
    if _EVENT_ACTION_INVITE_RE.search(combined):
        return False
    # Explicit timetable-like details are a strong non-event signal.
    if has_schedule_details:
        return True
    return True


def _looks_like_congrats_notice_not_event(title: str | None, text: str | None) -> bool:
    """Congratulation/holiday promos are not events unless there's an explicit event invite."""
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    low = combined.casefold()
    if not _CONGRATS_RE.search(low):
        return False
    # If it looks like a real event invite/announcement, keep.
    if _EVENT_INVITE_RE.search(combined):
        return False
    return True


def _has_price_evidence(text: str | None, *values: int | None) -> bool:
    """Return True when source text contains price-like context + numbers.

    Deterministic guardrail to prevent LLM/extractor hallucinations for ticket prices.
    """
    raw = str(text or "")
    if not raw:
        return False
    if not (_PRICE_CONTEXT_RE.search(raw) and _PRICE_NUMBER_RE.search(raw)):
        return False
    nums = {int(v) for v in values if isinstance(v, int) and v > 0}
    if not nums:
        return True
    for n in nums:
        # Allow optional whitespace between digits: "1500" vs "1 500" vs "1 500".
        digits = str(n)
        pattern = r"\s*".join(re.escape(ch) for ch in digits)
        for m in re.finditer(rf"(?u)\b{pattern}\b", raw):
            window_raw = raw[max(0, m.start() - 70) : min(len(raw), m.end() + 70)]
            # Money in a "compensation/payout/reward" context is not a ticket price.
            # Exception: if the same window also contains explicit ticket/entry words, keep it.
            if _NON_TICKET_MONEY_CONTEXT_RE.search(window_raw) and not _TICKET_PRICE_CONTEXT_RE.search(
                window_raw
            ):
                continue
            return True
    # If we have price-like context but none of the specific values appear as ticket-price-like,
    # treat as unsupported.
    return False


def _looks_like_blood_donation_event(title: str | None, text: str | None) -> bool:
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return False
    return bool(_BLOOD_DONATION_CONTEXT_RE.search(combined))


_UTILITY_OUTAGE_RE = re.compile(
    r"(?iu)\b("
    r"отключен\w*\s+(?:электроэнерг\w*|свет\w*|вод\w*|тепл\w*|газ\w*)|"
    r"времен\w*\s+отключен\w*|"
    r"планов\w*\s+отключен\w*|"
    r"аварийн\w*\s+отключен\w*|"
    r"перерыв\w*\s+в\s+(?:электроснабжен\w*|водоснабжен\w*|теплоснабжен\w*|газоснабжен\w*)|"
    r"(?:электроснабжен\w*|водоснабжен\w*|теплоснабжен\w*|газоснабжен\w*)\s+(?:будет|ограничен\w*|отключен\w*)|"
    r"перекрыт\w*\s+(?:движен\w*|дорог\w*|улиц\w*|проезд\w*)|"
    r"ограничен\w*\s+(?:движен\w*|проезд\w*)|"
    r"закрыт\w*\s+(?:движен\w*|проезд\w*)"
    r")\b"
)


def _looks_like_utility_outage_or_road_closure(title: str | None, text: str | None) -> str | None:
    """
    Municipal notices (utility outages / road closures) must not become events.
    Deterministic guard: no LLM, no content rewriting.
    Returns a reason suffix for SmartUpdateResult.reason when matched.
    """
    combined = "\n".join([str(title or ""), str(text or "")]).strip()
    if not combined:
        return None
    if not _UTILITY_OUTAGE_RE.search(combined):
        return None
    low = combined.casefold()
    if "перекры" in low or "огранич" in low or "проезд" in low or "движен" in low:
        return "road_closure"
    return "utility_outage"


def _strip_promo_lines(text: str | None) -> str | None:
    # Deterministic line-level cutting is not allowed (LLM handles this).
    raw = str(text or "").replace("\\n", "\n").strip()
    return raw or None


def _strip_giveaway_lines(text: str | None) -> str | None:
    # Deterministic line-level cutting is not allowed (LLM handles this).
    raw = str(text or "").replace("\\n", "\n").strip()
    return raw or None


def _candidate_has_event_anchors(candidate: EventCandidate) -> bool:
    # Minimal anchor set for a real event.
    #
    # Important: location_name alone is NOT a reliable anchor (often defaulted from a channel/source
    # and can appear in promo/congrats posts). Prefer anchors that are present in the text/title.
    title = (candidate.title or "").strip()
    if not (candidate.date and title):
        return False

    # Prefer checking anchors against *both* the short excerpt and the full source text.
    # The excerpt is typically `short_description` which must not contain date/time by prompt design,
    # so relying only on it can produce false "promo_or_congrats" skips for real events.
    excerpt = (candidate.raw_excerpt or "").strip()
    src = (candidate.source_text or "").strip()
    text_parts = [p for p in (excerpt, src) if p]
    text = "\n".join(text_parts).strip()
    combined = (title + "\n" + text).strip()

    if _EVENT_SIGNAL_RE.search(combined):
        return True
    if _has_datetime_signals(src) or _has_datetime_signals(excerpt):
        return True
    return False

def _has_datetime_signals(text: str | None) -> bool:
    if not text:
        return False
    value = text.lower()
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", value):
        return True
    if re.search(r"\b\d{1,2}[./]\d{1,2}\b", value):
        return True
    if re.search(r"\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\b", value):
        return True
    return False


def _giveaway_has_underlying_event_facts(text: str | None) -> bool:
    raw = str(text or "").replace("\\n", "\n").strip()
    if not raw:
        return False

    has_time = False
    has_date = False
    has_event_signal = False

    for line in raw.splitlines():
        value = line.strip()
        if not value:
            continue
        for part in re.split(r"(?<=[.!?…])\s+", value):
            chunk = part.strip()
            if not chunk:
                continue

            if _EVENT_INVITE_RE.search(chunk) or _EVENT_SIGNAL_RE.search(chunk):
                has_event_signal = True

            is_mechanics = bool(_GIVEAWAY_MECHANICS_RE.search(chunk) and not _EVENT_HAPPENS_VERB_RE.search(chunk))

            if re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", chunk):
                if re.search(r"(?iu)\bдо\s+([01]?\d|2[0-3])[:.]([0-5]\d)\b", chunk):
                    continue
                if is_mechanics:
                    continue
                has_time = True

            if re.search(r"\b\d{1,2}[./]\d{1,2}\b", chunk) or re.search(
                r"(?iu)\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\b",
                chunk,
            ):
                if _DEADLINE_RE.search(chunk):
                    continue
                if is_mechanics:
                    continue
                has_date = True

    if has_time:
        return True
    return has_date and has_event_signal


def _title_tokens(title: str | None) -> set[str]:
    if not title:
        return set()
    words = re.findall(r"[a-zа-яё0-9]{4,}", title.lower(), flags=re.IGNORECASE)
    return {w for w in words if w and not w.isdigit()}


def _extract_quote_candidates(text: str | None, *, max_items: int = 2) -> list[str]:
    """Extract short first-person quote candidates from source text (best-effort).

    This is used to help the LLM keep valuable direct speech as a quote block,
    instead of paraphrasing it away.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    raw = raw.replace("\r", "\n")
    candidates: list[str] = []
    seen: set[str] = set()

    # Deterministic direct speech in «...»: keep valuable quotes verbatim even when
    # they're not first-person. This helps Telegraph formatting (blockquote) and
    # prevents the LLM from paraphrasing away the original wording.
    for m in re.finditer(r"(?s)«([^»]{25,900})»", raw):
        q = re.sub(r"\s+", " ", (m.group(1) or "").strip())
        if not q:
            continue
        words = re.findall(r"[a-zа-яё0-9]{2,}", q, flags=re.IGNORECASE)
        looks_like_title = len(words) <= 6 and not re.search(r"[.!?…]", q)
        if looks_like_title:
            continue
        if len(words) < 8:
            continue
        cleaned_q = _normalize_fact_item(q, limit=170)
        if not cleaned_q:
            continue
        key_q = cleaned_q.lower()
        if key_q in seen:
            continue
        seen.add(key_q)
        candidates.append(cleaned_q)
        if len(candidates) >= max_items:
            return candidates
    # Split by sentence-ish boundaries while keeping it simple and deterministic.
    chunks = re.split(r"[.!?…]\s+|\n{2,}|\n", raw)
    sched_re = re.compile(r"^\s*\d{1,2}\.\d{1,2}\s*\|\s*.+$")
    # Russian first-person / opinion markers.
    fp_re = re.compile(
        r"\b(я|мне|мой|моя|моё|кажется|думаю|считаю|вижу|замечаю|по[- ]моему)\b",
        re.IGNORECASE,
    )
    for chunk in chunks:
        s = re.sub(r"\s+", " ", (chunk or "").strip())
        if not s or len(s) < 20:
            continue
        if sched_re.match(s):
            continue
        if not fp_re.search(s):
            continue
        cleaned = _normalize_fact_item(s, limit=170)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cleaned)
        if len(candidates) >= max_items:
            break
    return candidates


def _extract_director_name_hint(
    *,
    candidate_text: str | None,
    facts_before: Sequence[str] | None,
) -> str | None:
    """Best-effort extraction of the director name for quote attribution.

    We keep this conservative and deterministic: it's only used to label a direct
    quote block (operator readability + E2E assertion), not to invent facts.
    """
    text = (candidate_text or "").replace("\r", "\n")
    facts = [str(f or "") for f in (facts_before or [])]

    # Prefer explicit known name in either source text or existing facts.
    if re.search(r"(?i)\bегор\s+равинск", text) or any(
        re.search(r"(?i)\bегор\s+равинск", f) for f in facts
    ):
        return "Егор Равинский"

    # Generic RU "First Last" name capture near "режисс".
    # Example: "Режиссёр спектакля — Егор Равинский."
    name_re = re.compile(r"(?i)\bрежисс\w*\b[^\n]{0,80}?([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)")
    m = name_re.search(text)
    if m:
        return m.group(1).strip()
    for f in facts:
        m2 = name_re.search(f)
        if m2:
            return m2.group(1).strip()
    return None


def _inject_direct_quote_blockquote(
    *,
    description: str,
    quote: str,
    attribution_name: str | None,
) -> str:
    """Insert a Markdown blockquote with optional attribution into a description.

    Used as a hard safety-net when the LLM fails to keep a detected direct quote
    formatted as a blockquote.
    """
    desc = (description or "").strip()
    q = (quote or "").strip()
    if not desc or not q:
        return description
    if re.search(r"(?m)^>\s+", desc):
        return description

    # Avoid duplicating the same quote if it already appears verbatim.
    if q.casefold() in desc.casefold():
        return description

    block = f"> {q}"
    if attribution_name and attribution_name.strip():
        name = attribution_name.strip()
        # Put attribution inside the blockquote so Telegraph renders it together.
        if name.casefold() not in q.casefold():
            block = f"> {q}\n> — {name}"

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", desc) if p.strip()]
    if not paragraphs:
        return f"{desc}\n\n{block}".strip()

    insert_at = 1  # by default: after the first paragraph
    anchor = (attribution_name or "").split()[-1].casefold() if attribution_name else ""
    for i, p in enumerate(paragraphs):
        pc = p.casefold()
        if (anchor and anchor in pc) or ("режисс" in pc):
            insert_at = i + 1
            break
    paragraphs.insert(min(insert_at, len(paragraphs)), block)
    return "\n\n".join(paragraphs).strip()


def _ensure_blockquote_has_attribution(
    *,
    description: str,
    attribution_name: str | None,
) -> str:
    """Ensure at least one Markdown blockquote contains the attribution name.

    If we have a direct quote block but the speaker name is only mentioned in narration,
    operators (and tests) cannot reliably tell whose quote it is. We fix that by adding
    a short attribution line inside the first quote block.
    """
    desc = (description or "").strip()
    name = (attribution_name or "").strip()
    if not desc or not name:
        return description
    lines = desc.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    quote_line_idxs = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(">")]
    if not quote_line_idxs:
        return description

    name_cf = name.casefold()
    # Does any quote line already mention the name?
    for i in quote_line_idxs:
        if name_cf in lines[i].casefold():
            return description

    # Find the first contiguous quote block and append an attribution line to it.
    start = quote_line_idxs[0]
    end = start
    while end + 1 < len(lines) and lines[end + 1].lstrip().startswith(">"):
        end += 1

    # Avoid adding duplicate attribution markers.
    if end >= start and re.search(r"(?i)^\s*>\s*[—-]\s*\S", lines[end] or ""):
        return description

    lines.insert(end + 1, f"> — {name}")
    updated = "\n".join(lines).strip()
    return updated


async def _ensure_direct_quote_blockquote(
    *,
    description: str,
    quote_candidates: Sequence[str] | None,
    candidate_text: str | None,
    facts_before: Sequence[str] | None,
    label: str,
) -> str:
    """Ensure we have a Markdown blockquote when we detected quote candidates.

    Strategy:
    1) Ask LLM to integrate it.
    2) If LLM still doesn't produce a blockquote, deterministically inject it.
    """
    desc = (description or "").strip()
    if not desc:
        return description
    if re.search(r"(?m)^>\s+", desc):
        return description

    qc = [str(q or "").strip() for q in (quote_candidates or []) if str(q or "").strip()]
    if not qc:
        return description

    quote = qc[0]
    enforced = await _llm_enforce_blockquote(description=desc, quote=quote, label=label)
    if enforced and re.search(r"(?m)^>\s+", enforced):
        director = _extract_director_name_hint(candidate_text=candidate_text, facts_before=facts_before)
        return _ensure_blockquote_has_attribution(description=enforced, attribution_name=director)

    director = _extract_director_name_hint(candidate_text=candidate_text, facts_before=facts_before)
    injected = _inject_direct_quote_blockquote(
        description=desc,
        quote=quote,
        attribution_name=director,
    )
    injected = _ensure_blockquote_has_attribution(description=injected, attribution_name=director)
    return injected


async def _poster_is_relevant(candidate: EventCandidate, poster: PosterCandidate) -> tuple[bool, str | None]:
    """Decide whether a poster image is relevant to the event.

    Goal: avoid attaching generic promo banners (discounts, promos) as event posters.
    """
    ocr = (poster.ocr_text or "").strip()
    if not ocr:
        return True, None
    if not _POSTER_PROMO_RE.search(ocr):
        return True, None

    # Heuristic: promo + no datetime signals + no overlap with title tokens => likely unrelated.
    title_tokens = _title_tokens(candidate.title)
    overlap = 0
    if title_tokens:
        ocr_tokens = set(re.findall(r"[a-zа-яё0-9]{4,}", ocr.lower(), flags=re.IGNORECASE))
        overlap = len(title_tokens & ocr_tokens)

    if not _has_datetime_signals(ocr) and overlap == 0:
        return False, "promo_no_datetime"

    # Borderline cases: ask Gemma (best-effort).
    if SMART_UPDATE_LLM_DISABLED:
        return True, None
    client = _get_gemma_client()
    if client is None:
        return True, None
    schema = {
        "type": "object",
        "properties": {
            "relevant": {"type": "boolean"},
            "reason_short": {"type": "string"},
        },
        "required": ["relevant", "reason_short"],
        "additionalProperties": False,
    }
    payload = {
        "event": {
            "title": candidate.title,
            "date": candidate.date,
            "time": candidate.time,
            "location_name": candidate.location_name,
        },
        "poster_ocr": _clip(ocr, 1200),
    }
    prompt = (
        "Ты решаешь, относится ли афиша к КОНКРЕТНОМУ событию или это общий промо-баннер (скидки/акции/промокоды).\n"
        "Верни JSON: {relevant: true|false, reason_short: '...'}.\n"
        "Если на изображении в основном скидка/акция и нет признаков конкретного события (название/дата/время/площадка), верни relevant=false.\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    data = await _ask_gemma_json(prompt, schema, max_tokens=140, label="poster_relevance")
    if isinstance(data, dict) and isinstance(data.get("relevant"), bool):
        return bool(data["relevant"]), str(data.get("reason_short") or "").strip() or None
    return True, None


def _format_ticket_price(
    price_min: int | None, price_max: int | None
) -> str | None:
    if price_min is None and price_max is None:
        return None
    if price_min is not None and price_max is not None:
        if price_min == price_max:
            return f"{price_min} ₽"
        return f"{price_min}–{price_max} ₽"
    if price_min is not None:
        return f"от {price_min} ₽"
    return f"до {price_max} ₽"


def _normalize_fact_item(value: str | None, limit: int = 200) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    if not cleaned:
        return None
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


_RU_MONTHS_GENITIVE: dict[str, int] = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def _semantic_fact_key(
    fact: str | None,
    *,
    event_date: str | None,
    event_time: str | None,
) -> str | None:
    """Build a semantic key for anchor-like facts to avoid meaning-duplicates.

    Examples:
      "Дата: 2026-02-12" -> "date:2026-02-12"
      "Спектакль будет показан 12 февраля." (event_date=2026-02-12) -> "date:2026-02-12"
      "Начало спектакля в 19:00." -> "time:19:00"
    """
    raw = (fact or "").strip()
    if not raw:
        return None
    s = re.sub(r"\s+", " ", raw).strip()
    low = s.lower()

    def _iso_date_in_text(text: str) -> str | None:
        m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
        if not m:
            return None
        return m.group(1)

    def _parse_ru_date(text: str) -> str | None:
        # 12 февраля [2026]
        m = re.search(
            r"\b(?P<d>\d{1,2})\s+(?P<m>[а-яё]+)(?:\s+(?P<y>20\d{2}))?\b",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        day = int(m.group("d"))
        month_word = (m.group("m") or "").casefold()
        month = _RU_MONTHS_GENITIVE.get(month_word)
        if not month:
            return None
        year = int(m.group("y")) if (m.group("y") or "").strip().isdigit() else None
        # If event_date is known and matches day/month, reuse it (handles year ambiguity around New Year).
        if event_date:
            try:
                ev_d = date.fromisoformat(event_date.split("..", 1)[0].strip())
                if ev_d.day == day and ev_d.month == month:
                    return ev_d.isoformat()
                if year is None:
                    year = ev_d.year
            except Exception:
                pass
        if year is None:
            return None
        try:
            return date(year, month, day).isoformat()
        except Exception:
            return None

    def _parse_time(text: str) -> str | None:
        m = re.search(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\b", text)
        if not m:
            return None
        hh = int(m.group("h"))
        mm = int(m.group("m"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        return f"{hh:02d}:{mm:02d}"

    if low.startswith("дата окончания:"):
        iso = _iso_date_in_text(low) or _parse_ru_date(low)
        return f"end_date:{iso}" if iso else None
    if low.startswith("дата:"):
        iso = _iso_date_in_text(low) or _parse_ru_date(low)
        return f"date:{iso}" if iso else None
    if low.startswith("время:"):
        t = _parse_time(low)
        return f"time:{t}" if t else None

    # Free-form: detect date/time mentions.
    iso = _iso_date_in_text(low) or _parse_ru_date(low)
    if iso:
        return f"date:{iso}"
    t = _parse_time(low)
    if t:
        return f"time:{t}"
    return None


def _fact_preference_score(fact: str) -> int:
    """Higher score = we prefer to keep this form in ✅ when keys collide."""
    low = (fact or "").strip().lower()
    if low.startswith(("дата:", "дата окончания:", "время:")):
        return 3
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", low) or re.search(r"\b\d{1,2}:\d{2}\b", low):
        return 2
    return 1


def _demote_redundant_anchor_facts(
    added_log: list[str],
    duplicate_log: list[str],
    *,
    event_date: str | None,
    event_time: str | None,
    updated_keys: set[str],
) -> tuple[list[str], list[str]]:
    """Move meaning-duplicates of existing anchors from ✅ to ↩️.

    If event_date/time already exist and weren't updated in this merge, we treat
    any date/time mentions in LLM facts as duplicates (operator UX).
    """
    kept: list[str | None] = [None] * len(added_log)
    best_by_key: dict[str, tuple[int, str]] = {}

    # Determine current anchors after merge (event_db already has final values).
    anchor_date = (event_date or "").split("..", 1)[0].strip() or None
    anchor_time = (event_time or "").strip() or None
    date_was_updated = "date" in updated_keys
    time_was_updated = "time" in updated_keys

    for i, fact in enumerate(list(added_log or [])):
        f = (fact or "").strip()
        if not f:
            continue
        k = _semantic_fact_key(f, event_date=anchor_date, event_time=anchor_time)
        if not k:
            kept[i] = f
            continue

        # If anchor already existed (not updated), treat restatements as duplicates.
        if k.startswith("date:") and (not date_was_updated) and anchor_date and k == f"date:{anchor_date}":
            duplicate_log.append(f)
            kept[i] = None
            continue
        if k.startswith("time:") and (not time_was_updated) and anchor_time and k == f"time:{anchor_time}":
            duplicate_log.append(f)
            kept[i] = None
            continue

        prev = best_by_key.get(k)
        if not prev:
            best_by_key[k] = (i, f)
            kept[i] = f
            continue

        prev_i, prev_f = prev
        if _fact_preference_score(f) > _fact_preference_score(prev_f):
            duplicate_log.append(prev_f)
            kept[prev_i] = None
            best_by_key[k] = (i, f)
            kept[i] = f
        else:
            duplicate_log.append(f)
            kept[i] = None

    new_added = [x for x in kept if x]
    return new_added, duplicate_log


def _initial_textual_facts(candidate: EventCandidate, *, max_items: int = 2) -> list[str]:
    """Extract a couple of textual (non-service) facts for operator source log on create."""
    snippets = _collect_new_candidate_sentences(candidate, before_norm="")
    out: list[str] = []
    seen: set[str] = set()
    for sent in snippets:
        cleaned = _normalize_fact_item(sent, limit=170)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(f"Тезис: {cleaned}")
        if len(out) >= max_items:
            break
    return out


def _render_location_fact_value(
    *,
    location_name: str | None,
    location_address: str | None,
    city: str | None,
) -> str | None:
    name = (location_name or "").strip()
    if not name:
        return None
    # If location_name already looks like a canonical full line ("name, address, city"),
    # do not duplicate address/city from separate fields in source logs.
    if name.count(",") >= 2:
        return name
    parts = [
        name,
        (location_address or "").strip(),
        (city or "").strip(),
    ]
    value = ", ".join(part for part in parts if part)
    return value or None


def _initial_added_facts(candidate: EventCandidate) -> list[str]:
    facts: list[str] = []
    if candidate.date:
        facts.append(f"Дата: {candidate.date}")
    if candidate.end_date:
        facts.append(f"Дата окончания: {candidate.end_date}")
    if candidate.time:
        facts.append(f"Время: {candidate.time}")
    location = _render_location_fact_value(
        location_name=candidate.location_name,
        location_address=candidate.location_address,
        city=candidate.city,
    )
    if location:
        facts.append(f"Локация: {location}")
    if candidate.is_free is True:
        facts.append("Бесплатно")
    price_text = _format_ticket_price(
        candidate.ticket_price_min, candidate.ticket_price_max
    )
    if price_text:
        facts.append(f"Цена: {price_text}")
    if candidate.ticket_status == "sold_out":
        facts.append("Билеты все проданы")
    if candidate.ticket_link:
        label = "Регистрация" if candidate.is_free else "Билеты"
        facts.append(f"{label}: {candidate.ticket_link}")
    if candidate.event_type:
        facts.append(f"Тип: {candidate.event_type}")
    if candidate.festival:
        facts.append(f"Фестиваль: {candidate.festival}")
    if candidate.festival_full:
        facts.append(f"Выпуск фестиваля: {candidate.festival_full}")
    if candidate.pushkin_card is True:
        facts.append("Пушкинская карта")
    # IMPORTANT: Do not emit "Тезис: ..." pseudo-facts. Operator log must contain facts only.

    normalized: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        cleaned = _normalize_fact_item(fact)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized[:12]


def _candidate_anchor_facts_for_log(candidate: EventCandidate) -> list[str]:
    """Anchor-only facts for source log (no free-form textual theses)."""
    facts: list[str] = []
    if candidate.date:
        facts.append(f"Дата: {candidate.date}")
    if candidate.end_date:
        facts.append(f"Дата окончания: {candidate.end_date}")
    if candidate.time:
        facts.append(f"Время: {candidate.time}")
    location = _render_location_fact_value(
        location_name=candidate.location_name,
        location_address=candidate.location_address,
        city=candidate.city,
    )
    if location:
        facts.append(f"Локация: {location}")
    if candidate.is_free is True:
        facts.append("Бесплатно")
    price_text = _format_ticket_price(candidate.ticket_price_min, candidate.ticket_price_max)
    if price_text:
        facts.append(f"Цена: {price_text}")
    if candidate.ticket_status == "sold_out":
        facts.append("Билеты все проданы")
    if candidate.ticket_link:
        label = "Регистрация" if candidate.is_free else "Билеты"
        facts.append(f"{label}: {candidate.ticket_link}")
    if candidate.event_type:
        facts.append(f"Тип: {candidate.event_type}")
    if candidate.festival:
        facts.append(f"Фестиваль: {candidate.festival}")
    if candidate.festival_full:
        facts.append(f"Выпуск фестиваля: {candidate.festival_full}")
    if candidate.pushkin_card is True:
        facts.append("Пушкинская карта")

    normalized: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        cleaned = _normalize_fact_item(fact)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized[:12]


_CANONICAL_SCI_LIBRARY_NAME = "Научная библиотека"
_CANONICAL_SCI_LIBRARY_ADDRESS = "Мира 9"
_CANONICAL_SCI_LIBRARY_CITY = "Калининград"

_CANONICAL_DOM_KITOBOYA_NAME = "Дом китобоя"
_CANONICAL_DOM_KITOBOYA_ADDRESS = "Мира 9"
_CANONICAL_DOM_KITOBOYA_CITY = "Калининград"

_CANONICAL_ZAKHEIM_NAME = "Закхаймские ворота"
_CANONICAL_ZAKHEIM_ADDRESS = "Литовский Вал 61"
_CANONICAL_ZAKHEIM_CITY = "Калининград"


def _normalize_location_compact(value: str | None) -> str:
    if not value:
        return ""
    normalized = _norm_space(value)
    normalized = re.sub(r"[,.]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _looks_like_scientific_library_alias(norm_compact: str) -> bool:
    if not norm_compact:
        return False
    if "бфу" in norm_compact:
        return False
    return (
        norm_compact == "научная библиотека"
        or norm_compact == "научная библиотека мира 9 калининград"
        or "калининградская областная научная библиотека" in norm_compact
    )


def _looks_like_dom_kitoboya_alias(norm_compact: str) -> bool:
    if not norm_compact:
        return False
    return "дом китобоя" in norm_compact


def _looks_like_zakheim_alias(norm_compact: str) -> bool:
    if not norm_compact:
        return False
    norm_soft = norm_compact.replace("-", " ").replace("—", " ")
    norm_soft = re.sub(r"\s+", " ", norm_soft).strip()
    if "фридланд" in norm_soft:
        return False
    if "закхайм" in norm_soft or "закхейм" in norm_soft:
        return True
    return norm_soft in {"ворота", "арт пространство ворота", "артпространство ворота"}


def _canonicalize_location_fields(
    *,
    location_name: str | None,
    location_address: str | None,
    city: str | None,
    source_chat_username: str | None = None,
    source_url: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    name = (location_name or "").strip() or None
    address = (location_address or "").strip() or None
    city_value = (city or "").strip() or None

    name_norm = _normalize_location_compact(name)
    address_norm = _normalize_location_compact(address)
    combined_norm = " ".join([name_norm, address_norm]).strip()
    source_hint = " ".join(
        [
            (source_chat_username or "").strip().casefold(),
            (source_url or "").strip().casefold(),
        ]
    ).strip()

    if _looks_like_scientific_library_alias(combined_norm):
        return (
            _CANONICAL_SCI_LIBRARY_NAME,
            _CANONICAL_SCI_LIBRARY_ADDRESS,
            _CANONICAL_SCI_LIBRARY_CITY,
        )

    if _looks_like_dom_kitoboya_alias(combined_norm):
        return (
            _CANONICAL_DOM_KITOBOYA_NAME,
            _CANONICAL_DOM_KITOBOYA_ADDRESS,
            _CANONICAL_DOM_KITOBOYA_CITY,
        )

    zakheim_by_source = bool(
        source_hint
        and "vorotagallery" in source_hint
        and (not name_norm or "ворота" in name_norm or "закх" in name_norm)
    )
    if _looks_like_zakheim_alias(combined_norm) or zakheim_by_source:
        return (
            _CANONICAL_ZAKHEIM_NAME,
            _CANONICAL_ZAKHEIM_ADDRESS,
            _CANONICAL_ZAKHEIM_CITY,
        )

    # Normalize common address abbreviations for known locations.
    if address and ("мира 9" in address_norm):
        if name_norm and "дом китобоя" in name_norm:
            address = _CANONICAL_DOM_KITOBOYA_ADDRESS
        elif _looks_like_scientific_library_alias(name_norm):
            address = _CANONICAL_SCI_LIBRARY_ADDRESS

    return name, address, city_value


def _normalize_location(value: str | None) -> str:
    if not value:
        return ""
    # Normalize for matching only (not for public display): remove punctuation noise and
    # make "Янтарь-холл" ~= "Янтарь холл, Ленина 11".
    norm = _normalize_location_compact(value)
    norm = norm.replace("-", " ").replace("—", " ")
    norm = re.sub(r"[«»\"']", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()
    norm = _LOCATION_NOISE_PREFIXES_RE.sub("", norm).strip()
    # Canonicalize aliases of Kaliningrad Regional Scientific Library.
    # Do NOT merge BFU library names into this bucket.
    norm_compact = _normalize_location_compact(norm)
    if _looks_like_scientific_library_alias(norm_compact):
        return "научная библиотека"
    if _looks_like_dom_kitoboya_alias(norm_compact):
        return "дом китобоя"
    if _looks_like_zakheim_alias(norm_compact):
        return "закхаймские ворота"
    return norm


def _location_matches(a: str | None, b: str | None) -> bool:
    na = _normalize_location(a)
    nb = _normalize_location(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


_ADDRESS_NOISE_RE = re.compile(
    r"(?iu)\b(?:ул(?:ица)?|пр(?:оспект|осп)?|пр-?т|пер(?:еулок)?|б-р|бульвар|пл(?:ощадь)?|наб(?:ережная)?)\.?\b"
)


def _normalize_address_for_match(value: str | None, city: str | None = None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    norm = _normalize_location_compact(raw)
    norm = norm.replace("-", " ").replace("—", " ")
    norm = re.sub(r"[«»\"']", " ", norm)
    norm = _ADDRESS_NOISE_RE.sub(" ", norm)
    if city:
        city_norm = _normalize_location_compact(city)
        if city_norm:
            norm = re.sub(
                rf"(?iu)(?:,\s*|\s+)#?{re.escape(city_norm)}$",
                "",
                norm,
            )
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _address_matches(
    a: str | None,
    b: str | None,
    *,
    city_a: str | None = None,
    city_b: str | None = None,
) -> bool:
    na = _normalize_address_for_match(a, city_a)
    nb = _normalize_address_for_match(b, city_b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return False


def _event_candidate_location_matches(event: Event, candidate: "EventCandidate") -> bool:
    if _location_matches(getattr(event, "location_name", None), candidate.location_name):
        return True
    return _address_matches(
        getattr(event, "location_address", None),
        candidate.location_address,
        city_a=getattr(event, "city", None),
        city_b=candidate.city,
    )


def _apply_soft_city_filter(stmt, city: str | None):
    """Filter by candidate city, but keep legacy rows with empty city.

    Telegram/VK imports can create an event with `city=""` first, then a later
    source of the same event may resolve city correctly (e.g. "Калининград").
    Strict city equality would hide the existing row from shortlist and create
    a duplicate instead of merge.
    """
    city_value = str(city or "").strip()
    if not city_value:
        return stmt
    return stmt.where(
        or_(
            Event.city == city_value,
            Event.city.is_(None),
            Event.city == "",
        )
    )


@lru_cache(maxsize=1)
def _get_gemma_client():
    try:
        from google_ai import GoogleAIClient, SecretsProvider
        from main import get_supabase_client, notify_llm_incident
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("smart_update: gemma client unavailable: %s", exc)
        return None
    supabase = get_supabase_client()
    return GoogleAIClient(
        supabase_client=supabase,
        secrets_provider=SecretsProvider(),
        consumer="smart_update",
        incident_notifier=notify_llm_incident,
    )


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = cleaned.replace("```", "")
    return cleaned.strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


async def _ask_gemma_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    max_tokens: int,
    label: str,
) -> dict[str, Any] | None:
    # Retry Gemma a few times, then fall back to 4o (operator-visible) if configured.
    max_tries = int(os.getenv("SMART_UPDATE_GEMMA_RETRIES", "3"))
    base_sleep = float(os.getenv("SMART_UPDATE_GEMMA_RETRY_BASE_SEC", "1.0"))
    # When we are rate-limited, prefer waiting (do not count it as a "try") to
    # keep the new GOOGLE_API_KEY within quota and avoid burning 4o fallback.
    rl_max_wait_sec = float(os.getenv("SMART_UPDATE_GEMMA_RATE_LIMIT_MAX_WAIT_SEC", "180") or "180")
    rl_max_wait_sec = max(0.0, min(rl_max_wait_sec, 1800.0))
    max_tries = max(1, min(max_tries, 5))
    base_sleep = max(0.1, min(base_sleep, 10.0))
    client = _get_gemma_client()
    schema_text = json.dumps(schema, ensure_ascii=False)
    full_prompt = (
        f"{prompt}\n\n"
        "Верни только JSON без markdown и комментариев.\n"
        f"JSON schema:\n{schema_text}"
    )
    last_exc: Exception | None = None
    raw_last = ""

    # Best-effort: ask provider for JSON MIME when supported to reduce invalid JSON outputs.
    global _GEMMA_JSON_MIME_SUPPORTED
    try:
        _GEMMA_JSON_MIME_SUPPORTED
    except NameError:  # pragma: no cover - module init
        # Gemma models frequently reject JSON MIME mode. Keep it opt-in.
        _GEMMA_JSON_MIME_SUPPORTED = (  # type: ignore[assignment]
            (os.getenv("SMART_UPDATE_GEMMA_JSON_MIME", "0") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
    json_gen_cfg = {"temperature": 0}
    if _GEMMA_JSON_MIME_SUPPORTED:
        json_gen_cfg["response_mime_type"] = "application/json"

    rl_deadline = time.monotonic() + rl_max_wait_sec
    attempt = 1
    while attempt <= max_tries:
        if client is None:
            last_exc = RuntimeError("gemma client unavailable")
        else:
            try:
                logger.info(
                    "smart_update: gemma json_call label=%s model=%s max_tokens=%s attempt=%d/%d",
                    label,
                    SMART_UPDATE_MODEL,
                    max_tokens,
                    attempt,
                    max_tries,
                )
                while True:
                    try:
                        raw, _usage = await client.generate_content_async(
                            model=SMART_UPDATE_MODEL,
                            prompt=full_prompt,
                            generation_config=json_gen_cfg,
                            max_output_tokens=max_tokens,
                        )
                        break
                    except Exception as exc:
                        msg_l = str(exc).lower()
                        if (
                            _GEMMA_JSON_MIME_SUPPORTED
                            and any(
                                k in msg_l
                                for k in (
                                    "response_mime_type",
                                    "mime",
                                    "unknown field",
                                    "json mode is not enabled",
                                    "json mode",
                                )
                            )
                        ):
                            # Provider/library does not support this key; disable for the rest of the process.
                            _GEMMA_JSON_MIME_SUPPORTED = False  # type: ignore[assignment]
                            json_gen_cfg = {"temperature": 0}
                            continue
                        # Rate-limit handling: wait and retry without consuming an attempt.
                        try:
                            from google_ai.exceptions import (
                                ProviderError as _ProviderError,
                                RateLimitError as _RateLimitError,
                            )
                        except Exception:
                            _ProviderError = None
                            _RateLimitError = None
                        retry_ms = 0
                        if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                            retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if _ProviderError is not None and isinstance(exc, _ProviderError):
                            if int(getattr(exc, "status_code", 0) or 0) == 429:
                                retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if retry_ms > 0 and time.monotonic() < rl_deadline:
                            await asyncio.sleep(min(60.0, max(0.2, (retry_ms / 1000.0) + 0.2)))
                            continue
                        raise
                raw_last = raw or ""
                data = _extract_json(raw_last)
                if data is not None:
                    return data
                fix_prompt = (
                    "Исправь JSON под схему. Верни только JSON без markdown.\n"
                    f"Schema:\n{schema_text}\n\n"
                    f"Input:\n{raw_last}"
                )
                while True:
                    try:
                        raw_fix, _usage = await client.generate_content_async(
                            model=SMART_UPDATE_MODEL,
                            prompt=fix_prompt,
                            generation_config=json_gen_cfg,
                            max_output_tokens=max_tokens,
                        )
                        break
                    except Exception as exc:
                        msg_l = str(exc).lower()
                        if (
                            _GEMMA_JSON_MIME_SUPPORTED
                            and any(
                                k in msg_l
                                for k in (
                                    "response_mime_type",
                                    "mime",
                                    "unknown field",
                                    "json mode is not enabled",
                                    "json mode",
                                )
                            )
                        ):
                            _GEMMA_JSON_MIME_SUPPORTED = False  # type: ignore[assignment]
                            json_gen_cfg = {"temperature": 0}
                            continue
                        try:
                            from google_ai.exceptions import (
                                ProviderError as _ProviderError,
                                RateLimitError as _RateLimitError,
                            )
                        except Exception:
                            _ProviderError = None
                            _RateLimitError = None
                        retry_ms = 0
                        if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                            retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if _ProviderError is not None and isinstance(exc, _ProviderError):
                            if int(getattr(exc, "status_code", 0) or 0) == 429:
                                retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if retry_ms > 0 and time.monotonic() < rl_deadline:
                            await asyncio.sleep(min(60.0, max(0.2, (retry_ms / 1000.0) + 0.2)))
                            continue
                        raise
                raw_last = raw_fix or raw_last
                fixed = _extract_json(raw_fix or "")
                if fixed is not None:
                    return fixed
                last_exc = RuntimeError("gemma returned invalid json")
            except Exception as exc:  # pragma: no cover - provider failures
                last_exc = exc
                # If it's a rate limit, wait (not an "attempt") until the max wait budget.
                try:
                    from google_ai.exceptions import ProviderError as _ProviderError, RateLimitError as _RateLimitError
                except Exception:
                    _ProviderError = None
                    _RateLimitError = None
                retry_ms = 0
                if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                    retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                if _ProviderError is not None and isinstance(exc, _ProviderError):
                    if int(getattr(exc, "status_code", 0) or 0) == 429:
                        retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                if retry_ms > 0 and time.monotonic() < rl_deadline:
                    await asyncio.sleep(min(60.0, max(0.2, (retry_ms / 1000.0) + 0.2)))
                    continue
                logger.warning(
                    "smart_update: gemma %s failed attempt=%d/%d: %s",
                    label,
                    attempt,
                    max_tries,
                    exc,
                )

        if attempt < max_tries:
            await asyncio.sleep(base_sleep * (2 ** (attempt - 1)))
        attempt += 1

    # Fallback to 4o after Gemma retries.
    try:
        from main import ask_4o, notify_llm_incident
    except Exception:
        ask_4o = None
        notify_llm_incident = None
    if ask_4o is None:
        return None
    try:
        if notify_llm_incident is not None:
            await notify_llm_incident(
                "smart_update_gemma_fallback_4o",
                {
                    "severity": "warning",
                    "consumer": "smart_update",
                    "requested_model": SMART_UPDATE_MODEL,
                    "model": SMART_UPDATE_MODEL,
                    "attempt_no": max_tries,
                    "max_retries": max_tries,
                    "next_model": "gpt-4o",
                    "message": f"Gemma JSON call failed for label={label}; switching to 4o",
                    "error": repr(last_exc) if last_exc else "unknown",
                },
            )
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": f"SmartUpdate_{label}", "schema": schema},
        }
        raw_4o = await ask_4o(
            prompt,
            response_format=response_format,
            max_tokens=max_tokens,
            meta={"consumer": "smart_update", "label": label, "fallback": "gemma_failed"},
        )
        data = _extract_json(raw_4o or "")
        return data
    except Exception as exc:  # pragma: no cover - network / token failures
        logger.warning("smart_update: 4o fallback failed label=%s: %s", label, exc)
        return None


async def _ask_gemma_text(
    prompt: str,
    *,
    max_tokens: int,
    label: str,
    temperature: float = 0.0,
) -> str | None:
    max_tries = int(os.getenv("SMART_UPDATE_GEMMA_RETRIES", "3"))
    base_sleep = float(os.getenv("SMART_UPDATE_GEMMA_RETRY_BASE_SEC", "1.0"))
    rl_max_wait_sec = float(os.getenv("SMART_UPDATE_GEMMA_RATE_LIMIT_MAX_WAIT_SEC", "180") or "180")
    rl_max_wait_sec = max(0.0, min(rl_max_wait_sec, 1800.0))
    max_tries = max(1, min(max_tries, 5))
    base_sleep = max(0.1, min(base_sleep, 10.0))
    client = _get_gemma_client()
    last_exc: Exception | None = None

    rl_deadline = time.monotonic() + rl_max_wait_sec
    attempt = 1
    while attempt <= max_tries:
        if client is None:
            last_exc = RuntimeError("gemma client unavailable")
        else:
            try:
                logger.info(
                    "smart_update: gemma text_call label=%s model=%s max_tokens=%s temperature=%s attempt=%d/%d",
                    label,
                    SMART_UPDATE_MODEL,
                    max_tokens,
                    temperature,
                    attempt,
                    max_tries,
                )
                while True:
                    try:
                        raw, _usage = await client.generate_content_async(
                            model=SMART_UPDATE_MODEL,
                            prompt=prompt,
                            generation_config={"temperature": temperature},
                            max_output_tokens=max_tokens,
                        )
                        break
                    except Exception as exc:
                        try:
                            from google_ai.exceptions import (
                                ProviderError as _ProviderError,
                                RateLimitError as _RateLimitError,
                            )
                        except Exception:
                            _ProviderError = None
                            _RateLimitError = None
                        retry_ms = 0
                        if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                            retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if _ProviderError is not None and isinstance(exc, _ProviderError):
                            if int(getattr(exc, "status_code", 0) or 0) == 429:
                                retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                        if retry_ms > 0 and time.monotonic() < rl_deadline:
                            await asyncio.sleep(min(60.0, max(0.2, (retry_ms / 1000.0) + 0.2)))
                            continue
                        raise
                cleaned = _strip_code_fences(raw or "").strip()
                if cleaned:
                    return cleaned
                last_exc = RuntimeError("gemma returned empty text")
            except Exception as exc:  # pragma: no cover - provider failures
                last_exc = exc
                try:
                    from google_ai.exceptions import ProviderError as _ProviderError, RateLimitError as _RateLimitError
                except Exception:
                    _ProviderError = None
                    _RateLimitError = None
                retry_ms = 0
                if _RateLimitError is not None and isinstance(exc, _RateLimitError):
                    retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                if _ProviderError is not None and isinstance(exc, _ProviderError):
                    if int(getattr(exc, "status_code", 0) or 0) == 429:
                        retry_ms = int(getattr(exc, "retry_after_ms", 0) or 0)
                if retry_ms > 0 and time.monotonic() < rl_deadline:
                    await asyncio.sleep(min(60.0, max(0.2, (retry_ms / 1000.0) + 0.2)))
                    continue
                logger.warning(
                    "smart_update: gemma %s failed attempt=%d/%d: %s",
                    label,
                    attempt,
                    max_tries,
                    exc,
                )
        if attempt < max_tries:
            await asyncio.sleep(base_sleep * (2 ** (attempt - 1)))
        attempt += 1

    # Fallback to 4o after Gemma retries.
    try:
        from main import ask_4o, notify_llm_incident
    except Exception:
        ask_4o = None
        notify_llm_incident = None
    if ask_4o is None:
        return None
    try:
        if notify_llm_incident is not None:
            await notify_llm_incident(
                "smart_update_gemma_fallback_4o",
                {
                    "severity": "warning",
                    "consumer": "smart_update",
                    "requested_model": SMART_UPDATE_MODEL,
                    "model": SMART_UPDATE_MODEL,
                    "attempt_no": max_tries,
                    "max_retries": max_tries,
                    "next_model": "gpt-4o",
                    "message": f"Gemma text call failed for label={label}; switching to 4o",
                    "error": repr(last_exc) if last_exc else "unknown",
                },
            )
        raw_4o = await ask_4o(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            meta={"consumer": "smart_update", "label": label, "fallback": "gemma_failed"},
        )
        cleaned = _strip_code_fences(raw_4o or "").strip()
        return cleaned or None
    except Exception as exc:  # pragma: no cover - network / token failures
        logger.warning("smart_update: 4o fallback failed label=%s: %s", label, exc)
        return None


async def _llm_extract_candidate_facts(
    candidate: EventCandidate,
    *,
    text_for_facts: str | None = None,
) -> list[str]:
    """Extract atomic event facts from a single candidate for global fact log/dedup.

    Notes:
    - Facts are used for operator source log and for global de-duplication between sources.
    - Do not include anchor fields (date/time/location) here: they are logged deterministically.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return []
    if candidate.source_type in ("bot", "manual"):
        return []

    schema = {
        "type": "object",
        "properties": {
            "facts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facts"],
        "additionalProperties": False,
    }
    payload = {
        "today": date.today().isoformat(),
        "title": candidate.title,
        "date": candidate.date,
        "time": candidate.time,
        "end_date": candidate.end_date,
        "location_name": candidate.location_name,
        "location_address": candidate.location_address,
        "city": candidate.city,
        "ticket_link": candidate.ticket_link,
        "ticket_status": candidate.ticket_status,
        "source_type": candidate.source_type,
        "source_url": candidate.source_url,
        "text": _clip(
            (text_for_facts or "").strip()
            or (_strip_promo_lines(candidate.source_text) or candidate.source_text),
            2800,
        ),
        "raw_excerpt": _clip(_strip_promo_lines(candidate.raw_excerpt) or candidate.raw_excerpt, 800),
        "poster_texts": [_clip(p.ocr_text, 700) for p in candidate.posters if (p.ocr_text or "").strip()][:3],
    }
    prompt = (
        "Ты извлекаешь атомарные факты о КОНКРЕТНОМ событии из текста источника.\n"
        "Верни JSON строго по схеме.\n\n"
        "Правила:\n"
        "- Верни 6–18 коротких фактов (1 строка = 1 факт), только про это событие.\n"
        "- Пиши факты как короткие именные группы (по возможности без глаголов 'является', 'будет', 'обещает').\n"
        "- Для оценочных характеристик и лозунгов используй формулировку из источника максимально близко к тексту "
        "(если в источнике есть кавычки, сохрани кавычки).\n"
        "- НЕ включай дату/время/адрес/город как отдельные факты (они фиксируются отдельно).\n"
        "- НЕ включай строки расписания вида `DD.MM | Название`.\n"
        f"- {SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE}\n"
        "- Не используй хэштеги (`#...`) в формулировках фактов.\n"
        "- НЕ включай рекламные призывы, скидки/промокоды, механику розыгрыша.\n"
        "- НЕ включай промо-упоминания «где следить за анонсами» и ссылки на каналы/чаты с афишей "
        "(например «Информация о событиях ... доступна в Telegram-канале ...»).\n"
        "- Включай условия участия/посещения (длительность, возраст, максимальный размер группы, формат/что взять/как одеться, "
        "что входит/не входит в оплату, нужен ли отдельный входной билет). Не вставляй ссылки/телефоны, "
        "КРОМЕ: если в источнике есть ссылка на плейлист Я.Музыки (music.yandex.ru/users/.../playlists/...), "
        "можно вернуть 1 факт с этой ссылкой; "
        "точную сумму указывай только если это важно, чтобы пояснить «что оплачивается отдельно» (не более 1 факта).\n"
        "- НЕ включай факты про общие новости площадки/организации, если они не описывают само событие "
        "(например отчёты о работе филиала, планы на год, пресс-анонсы о будущих репортажах).\n"
        "- НЕ включай нейросетевые клише, пустые оценки и прогнозы, которых нет в источнике: "
        "например 'обещает стать заметным событием', 'яркое событие культурной жизни', "
        "'не оставит равнодушным', 'незабываемые эмоции', 'уникальная возможность'.\n"
        "- НЕ выдумывай факты. Если чего-то нет в данных, не добавляй.\n"
        "- Если есть прямая речь и понятно, кто говорит (например режиссёр), оформи как факт:\n"
        "  `Цитата (Имя Фамилия): ...`.\n"
        "- Избегай дублирования: если мысль повторяется, оставь один факт.\n\n"
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    data = await _ask_gemma_json(prompt, schema, max_tokens=500, label="facts_extract")
    raw_facts = []
    if isinstance(data, dict):
        raw_facts = list(data.get("facts") or [])

    # Normalize + drop anchor-like meaning duplicates.
    anchor_date = (candidate.date or "").strip() or None
    anchor_time = (candidate.time or "").strip() or None
    out: list[str] = []
    seen: set[str] = set()
    for item in raw_facts:
        cleaned = _normalize_fact_item(str(item or ""), limit=180)
        if not cleaned:
            continue
        # Do not claim "premiere" unless it is explicitly present in the source text.
        if re.search(r"(?i)\bпремьер\w+\b", cleaned) and "премьер" not in (payload.get("text") or "").casefold():
            continue
        # Drop generic evaluative/predictive phrases: they are not factual and break
        # the "facts -> telegraph coverage" invariant.
        if re.search(
            r"(?i)\bобеща\w+\s+(?:стать|быть)\b|\bярк\w+\s+событ\w+\b|\bзаметн\w+\s+событ\w+\b|"
            r"\bкультурн\w+\s+жизн\w+\b|\bне\s+остав\w+\s+равнодуш\w+\b|\bнезабываем\w+\b|\bуникальн\w+\s+возможн\w+\b",
            cleaned,
        ):
            continue
        # If it repeats an anchor (e.g. "12 февраля") treat as noise for the global fact list.
        k = _semantic_fact_key(cleaned, event_date=anchor_date, event_time=anchor_time)
        if k:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= 20:
            break
    return out


async def _llm_enforce_blockquote(
    *,
    description: str,
    quote: str,
    label: str,
) -> str | None:
    """Ask LLM to integrate a direct quote as a blockquote into an existing description."""
    if SMART_UPDATE_LLM_DISABLED:
        return None
    desc = (description or "").strip()
    q = (quote or "").strip()
    if not desc or not q:
        return None
    payload = {
        "description": _clip(desc, 5000),
        "quote": _clip(q, 400),
    }
    prompt = (
        "Вставь прямую цитату в описание события.\n"
        "Правила:\n"
        "- Верни полный обновлённый текст описания.\n"
        "- Цитату вставь как отдельный блок `>` (blockquote) ДОСЛОВНО.\n"
        "- Если в описании упоминается автор цитаты (например режиссёр), добавь атрибуцию сразу после цитаты "
        "короткой строкой (например `— Егор Равинский`).\n"
        "- Не добавляй новых фактов и не меняй смысл остального текста.\n"
        f"{SMART_UPDATE_YO_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=900,
        label=label,
        temperature=0.0,
    )
    return text.strip() if text else None


async def _llm_remove_infoblock_logistics(
    *,
    description: str,
    candidate: EventCandidate,
    label: str,
) -> str | None:
    """Ask LLM to remove duplicated logistics from narrative text (infoblock already covers it).

    NOTE: We intentionally do NOT do deterministic regex-based cutting here. If the model
    fails, we prefer keeping duplicates over breaking grammar or deleting meaning.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None
    desc = (description or "").strip()
    if not desc:
        return None
    payload = {
        "title": candidate.title,
        "date": candidate.date,
        "time": candidate.time,
        "end_date": candidate.end_date,
        "location_name": candidate.location_name,
        "location_address": candidate.location_address,
        "city": candidate.city,
        "ticket_link": candidate.ticket_link,
        "ticket_status": candidate.ticket_status,
        "ticket_price_min": candidate.ticket_price_min,
        "ticket_price_max": candidate.ticket_price_max,
        "is_free": candidate.is_free,
        "event_type": candidate.event_type,
        "festival": candidate.festival,
        "description": _clip(desc, 6500),
    }
    prompt = (
        "Ты — редактор текста анонса.\n"
        "Задача: убрать из описания повторы логистики, потому что она уже показывается отдельным инфоблоком сверху.\n"
        "Что убрать (если встречается в описании как отдельная логистика):\n"
        "- дата/время/диапазон дат\n"
        "- площадка, точный адрес, город\n"
        "- ссылки/телефон/контакты\n"
        "- точные цены/стоимость билетов/регистрации\n"
        "Правила:\n"
        "- Верни ПОЛНЫЙ обновлённый текст описания.\n"
        "- Не вырезай смысловые фрагменты и не ломай грамматику.\n"
        "- Не добавляй новых фактов. Не меняй стиль.\n"
        "- Сохраняй пунктуацию и абзацы. Не превращай текст в список, если он был прозой.\n"
        f"{SMART_UPDATE_YO_RULE}\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=900,
        label=label,
        temperature=0.0,
    )
    return text.strip() if text else None


async def _rewrite_description_journalistic(
    candidate: EventCandidate,
    *,
    strict_nonverbatim: bool = False,
) -> str | None:
    """Produce a non-verbatim, journalist-style description for external imports.

    Keep this best-effort: failures must not block event creation/merge.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None
    if candidate.source_type in ("bot", "manual"):
        return None

    # For site imports we often have a short `raw_excerpt` (search-style snippet),
    # while `source_text` contains the full article/program. Prefer the fuller
    # source when the excerpt is clearly shorter to avoid generating a "too short"
    # description for Telegraph.
    excerpt_raw = (candidate.raw_excerpt or "").strip()
    source_raw = (candidate.source_text or "").strip()
    poster_raw = "\n\n".join(
        [
            (p.ocr_text or "").strip()
            for p in (candidate.posters or [])[:2]
            if (p.ocr_text or "").strip()
        ]
    ).strip()
    base = excerpt_raw or source_raw or poster_raw
    if _should_prefer_source_text_for_description(source_raw, excerpt_raw):
        base = source_raw
    base = _strip_promo_lines(base) or base
    base = _strip_private_use(base) or base
    base = _normalize_bullet_markers(base) or base
    if len(base) < 80 and poster_raw:
        poster_clean = _strip_promo_lines(poster_raw) or poster_raw
        poster_clean = _strip_private_use(poster_clean) or poster_clean
        poster_clean = _normalize_bullet_markers(poster_clean) or poster_clean
        if len((poster_clean or "").strip()) >= 80:
            base = (poster_clean or "").strip()
    if len(base) < 80:
        return None

    payload = {
        "title": candidate.title,
        "date": candidate.date,
        "time": candidate.time,
        "end_date": candidate.end_date,
        "location_name": candidate.location_name,
        "location_address": candidate.location_address,
        "city": candidate.city,
        "ticket_link": candidate.ticket_link,
        "ticket_status": candidate.ticket_status,
        "is_free": candidate.is_free,
        "event_type": candidate.event_type,
        "festival": candidate.festival,
        "source_type": candidate.source_type,
        "raw_excerpt": _clip(_strip_promo_lines(candidate.raw_excerpt) or candidate.raw_excerpt, 1200),
        "source_text": _clip(
            _normalize_bullet_markers(_strip_promo_lines(candidate.source_text) or candidate.source_text)
            or (_strip_promo_lines(candidate.source_text) or candidate.source_text),
            SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS,
        ),
        "quote_candidates": _extract_quote_candidates(
            _strip_promo_lines(candidate.source_text) or candidate.source_text,
            max_items=2,
        ),
        "poster_texts": [_clip(p.ocr_text, 500) for p in candidate.posters if p.ocr_text][:3],
        "poster_titles": [
            _clip(p.ocr_title, 140)
            for p in candidate.posters
            if (p.ocr_title or "").strip()
        ][:3],
    }
    strict_block = ""
    if strict_nonverbatim:
        strict_block = (
            "СТРОГОЕ требование к анти-дословности:\n"
            "- Не копируй предложения из источника дословно.\n"
            "- Не допускай длинных совпадающих фрагментов (ориентир: не более 8–10 слов подряд).\n"
            "- Если перефразировать нельзя без потери смысла — лучше сократи этот фрагмент.\n\n"
        )
    telegram_block = ""
    tone_hint = "Передай суть и атмосферу"
    if candidate.source_type == "telegram":
        tone_hint = "Передай суть"
        short_base_len = len(base.strip())
        if 80 <= short_base_len <= 350:
            cap_hint = max(220, short_base_len + 100)
            telegram_block = (
                "Особенности Telegram (короткий источник):\n"
                f"- Длина результата: не длиннее {cap_hint} символов.\n"
                "- Не раздувай текст и не добавляй общие фразы «про атмосферу/вовлечённость/уникальность».\n"
                "- Запрещены штампы вроде «это создаёт ...».\n"
                "- Не добавляй технические заголовки/секции про факты: `Facts`, `Added Facts`, `Facts/Added Facts`, "
                "а также «Факты…», «Факты о событии».\n"
                "- Если есть прямая речь в «...» — сохрани её ДОСЛОВНО и оформи как blockquote (`>`).\n\n"
            )
        else:
            telegram_block = (
                "Особенности Telegram:\n"
                "- Избегай клише и пустых фраз (особенно «это создаёт ...»).\n"
                "- Не добавляй технические заголовки/секции про факты: `Facts`, `Added Facts`, `Facts/Added Facts`, "
                "а также «Факты…», «Факты о событии».\n\n"
            )
    prompt = (
        "Ты — культурный журналист. Сделай журналистский рерайт анонса мероприятия. "
        f"{tone_hint}, но НЕ копируй исходные фразы дословно. "
        "Не добавляй выдуманных фактов, используй только то, что есть в данных. "
        "Запрещено придумывать утверждения вроде 'премьера', 'впервые', 'аншлаг' и т.п., "
        "если это явно не сказано в источнике. "
        f"{SMART_UPDATE_YO_RULE} "
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE} "
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE} "
        f"{SMART_UPDATE_OPTIONAL_HEADINGS_RULE} "
        f"{_description_emoji_prompt_rule()}"
        "Без хэштегов. "
        "Не добавляй отдельные секции/заголовки про факты (например «Факты…», «Факты о событии», "
        "«Facts/Added Facts»): факты формируются и показываются отдельно. "
        "Важно: НЕ повторяй в описании логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены) — "
        "она показывается отдельным инфоблоком сверху. "
        "Убери промо чужих/вспомогательных каналов с анонсами и призывы подписаться "
        "(например «Информация о событиях ... доступна в Telegram-канале ...»): это не факт про само событие. "
        "Можно использовать минимальную разметку для читабельности: "
        "заголовки `###`, цитаты блоком `> ...`, редкое выделение `**...**`. "
        "НЕ используй Markdown-ссылки вида `[текст](url)` и не вставляй таблицы. "
        "Согласуй время повествования с датой события: "
        "если дата события в будущем (относительно поля `today`) — используй будущее время "
        "(например «пройдёт», «состоится»), а не «проходит». "
        "Убери рекламные и акционные фрагменты (скидки/промокоды/акция) и механику розыгрыша, если они не являются частью сути события. "
        "Не включай малозначимые и повторяющиеся строки (например `DD.MM | Название`, повтор даты/заголовка, «представление состоится ...» при уже указанной дате/времени). "
        "Если в источнике есть обрыв фразы/текста (в т.ч. обрезано на середине слова), не вставляй это дословно: либо перефразируй, либо опусти. "
        "Не экранируй кавычки обратным слэшем (не пиши `\\\"...\\\"`). "
        "Если в тексте есть прямая речь/цитата (1-е лицо: 'я/мне/кажется/думаю' и т.п.), "
        "НЕ переписывай её в косвенную речь: включи её ДОСЛОВНО как цитату блоком `>` и не дублируй ту же мысль пересказом рядом. "
        "Если понятно, кто автор цитаты (например режиссёр), добавь атрибуцию: `— Имя Фамилия` рядом с цитатой. "
        "Если `quote_candidates` не пуст, обязательно включи хотя бы одну из этих цитат ДОСЛОВНО как blockquote. "
        "Сделай ПОЛНОЕ развернутое описание события, сохранив ВСЕ значимые факты из входных данных, "
        "кроме логистики (она уже показана отдельно). "
        "Не превращай текст в краткий дайджест: если исходный текст длинный, результат тоже может быть длинным "
        "(например 10-25 предложений, при необходимости больше). "
        "Если в исходных данных перечислены элементы программы/сюжета/формата/участники/условия посещения, отрази их. "
        "Структуру делай абзацами: разделяй абзацы пустой строкой. Текст должен читаться как единое связное повествование.\n\n"
        "Техническое требование к форматированию:\n"
        "- В одном абзаце держи 1-2 предложения (максимум 3 только если иначе теряется смысл).\n"
        "- НЕ разрывай предложения пустой строкой на середине.\n"
        "- НЕ ставь пустую строку между инициалом и фамилией (например `Н. Любимова`).\n"
        "- Не дублируй в основном тексте строки-анкеры (`Дата:`, `Время:`, `Локация:`, `Билеты:`) и их явные перефразы.\n"
        "- Избегай нейросетевых клише и прогнозов (например 'обещает стать заметным событием', 'не оставит равнодушным').\n\n"
        "Самопроверка перед ответом:\n"
        "- В тексте НЕТ ссылок/телефонов/точных адресов/цен/времени/дат (они уже в инфоблоке).\n"
        "- НЕТ обрывов фраз (например «стоимость … составит» без продолжения).\n"
        "- НЕТ странных/непонятных слов и опечаток; если слово выглядит ошибочным — перефразируй.\n\n"
        f"{telegram_block}"
        f"{strict_block}"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=SMART_UPDATE_REWRITE_MAX_TOKENS,
        label="rewrite",
        temperature=0.0,
    )
    if not text:
        return None
    cleaned = text
    cleaned = _strip_private_use(cleaned) or cleaned
    cleaned = _normalize_plaintext_paragraphs(cleaned)
    if not cleaned:
        return None
    cleaned = _fix_broken_initial_paragraph_splits(cleaned) or cleaned
    cleaned = (
        _sanitize_description_output(
            cleaned,
            source_text=_strip_promo_lines(candidate.source_text) or candidate.source_text,
        )
        or cleaned
    )
    cleaned = _strip_channel_promo_from_description(cleaned) or cleaned
    # Ensure direct quotes stay as quotes (blockquote) when we detected candidates in the source.
    quote_candidates = payload.get("quote_candidates") or []
    director_name_hint = _extract_director_name_hint(
        candidate_text=_strip_promo_lines(candidate.source_text) or candidate.source_text,
        facts_before=[],
    )
    cleaned = _ensure_blockquote_has_attribution(
        description=cleaned,
        attribution_name=director_name_hint,
    )
    if quote_candidates and not re.search(r"(?m)^>\s+", cleaned):
        cleaned = await _ensure_direct_quote_blockquote(
            description=cleaned,
            quote_candidates=quote_candidates,
            candidate_text=_strip_promo_lines(candidate.source_text) or candidate.source_text,
            facts_before=[],
            label="rewrite_quote_enforce",
        )
        cleaned = _normalize_plaintext_paragraphs(cleaned) or cleaned
        cleaned = _normalize_blockquote_markers(cleaned) or cleaned
        cleaned = _drop_reported_speech_duplicates(cleaned) or cleaned
        cleaned = _ensure_blockquote_has_attribution(
            description=cleaned,
            attribution_name=director_name_hint,
        )

    if _description_needs_infoblock_logistics_strip(cleaned, candidate=candidate):
        try:
            edited = await _llm_remove_infoblock_logistics(
                description=cleaned,
                candidate=candidate,
                label="rewrite_remove_logistics",
            )
        except Exception:  # pragma: no cover - network failures
            edited = None
        if edited:
            edited = _normalize_plaintext_paragraphs(edited) or edited
            cleaned = edited
    if _description_needs_channel_promo_strip(cleaned):
        cleaned = _strip_channel_promo_from_description(cleaned) or cleaned

    cleaned = (
        _append_missing_small_list(
            description=cleaned,
            source_text=base,
            source_type=candidate.source_type,
        )
        or cleaned
    )

    # For short Telegram snippets (1-2 lines), keep rewrite volume near source size.
    # This prevents aggressive expansion/hallucinated "long reads" when source is concise.
    if candidate.source_type == "telegram":
        short_base_len = len(base.strip())
        if 80 <= short_base_len <= 350:
            max_allowed = min(
                SMART_UPDATE_DESCRIPTION_MAX_CHARS,
                max(220, int(short_base_len) + 100),
            )
            if len(cleaned) > max_allowed:
                logger.info(
                    "smart_update: rewrite overexpanded short telegram source (base_len=%s, out_len=%s, cap=%s)",
                    short_base_len,
                    len(cleaned),
                    max_allowed,
                )
                cleaned = _clip_to_readable_boundary(cleaned, max_allowed)
    return _clip(cleaned, SMART_UPDATE_DESCRIPTION_MAX_CHARS)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = value.split("..", 1)[0].strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except Exception:
        return None


_DAY_MONTH_NUM_RE = re.compile(r"\b(\d{1,2})\s*[./-]\s*(\d{1,2})\b")
_MONTH_WORD_PATTERN = "|".join(sorted((re.escape(k) for k in MONTHS_RU.keys()), key=len, reverse=True))
_DAY_MONTH_WORD_RE = (
    re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_WORD_PATTERN})\b", re.IGNORECASE)
    if _MONTH_WORD_PATTERN
    else None
)


def _extract_day_month_pairs(text: str | None) -> set[tuple[int, int]]:
    raw = str(text or "").strip()
    if not raw:
        return set()
    normalized = unicodedata.normalize("NFKC", raw).casefold().replace("ё", "е")
    pairs: set[tuple[int, int]] = set()
    for m in _DAY_MONTH_NUM_RE.finditer(normalized):
        try:
            day = int(m.group(1))
            month = int(m.group(2))
        except Exception:
            continue
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        pairs.add((day, month))
    if _DAY_MONTH_WORD_RE is not None:
        for m in _DAY_MONTH_WORD_RE.finditer(normalized):
            try:
                day = int(m.group(1))
            except Exception:
                continue
            mon_word = (m.group(2) or "").casefold().replace("ё", "е")
            month = MONTHS_RU.get(mon_word)
            if not month or not (1 <= day <= 31):
                continue
            pairs.add((day, int(month)))
    return pairs


def _poster_day_month_pairs(posters: Sequence[PosterCandidate]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for poster in posters or []:
        pairs |= _extract_day_month_pairs(getattr(poster, "ocr_title", None))
        pairs |= _extract_day_month_pairs(getattr(poster, "ocr_text", None))
    return pairs


def _format_day_month_pairs(pairs: set[tuple[int, int]]) -> str:
    if not pairs:
        return ""
    return ", ".join(f"{d:02d}/{m:02d}" for d, m in sorted(pairs, key=lambda x: (x[1], x[0])))


def _far_future_poster_date_mismatch_note(
    *,
    candidate_date: str | None,
    posters: Sequence[PosterCandidate],
    months_threshold: int,
) -> str | None:
    """Return operator note when a far-future extracted date conflicts with poster OCR."""
    if months_threshold <= 0:
        return None
    start = _parse_iso_date(candidate_date)
    if not start:
        return None
    today = datetime.now(timezone.utc).date()
    try:
        from dateutil.relativedelta import relativedelta

        far_cutoff = today + relativedelta(months=int(months_threshold))
    except Exception:
        far_cutoff = today + timedelta(days=31 * int(months_threshold))
    if start < far_cutoff:
        return None
    pairs = _poster_day_month_pairs(posters)
    if not pairs:
        return None
    if (start.day, start.month) in pairs:
        return None
    pairs_label = _format_day_month_pairs(pairs)
    extracted_label = f"{start.day:02d}/{start.month:02d}"
    return (
        f"⚠️ Дата: конфликт с афишей (OCR={pairs_label}, extracted={extracted_label}) → event.silent=1"
    )


def _add_one_calendar_month(start: date) -> date:
    year = start.year
    month = start.month + 1
    if month > 12:
        month = 1
        year += 1
    day = min(start.day, monthrange(year, month)[1])
    return date(year, month, day)


_LONG_EVENT_TEXT_HINT_RE = re.compile(
    r"\b("
    r"выставк\w*|"
    r"экспозиц\w*|"
    r"ярмарк\w*|"
    r"маркет\w*|"
    r"инсталляци\w*|"
    r"экспозици\w*"
    r")\b",
    re.IGNORECASE,
)

_ACTION_TITLE_RE = re.compile(r"(?i)\bакци\w*\b")
_ONE_DAY_ACTION_HINT_RE = re.compile(
    r"(?i)\b("
    r"билет\w*\s+действ\w*.*\bтолько\b|"
    r"только\s+(?:сегодня|завтра)|"
    r"на\s+завтра|"
    r"на\s+сегодня|"
    r"указанн\w+\s+дат\w+|"
    r"одн[ау]\s+дат[ау]\b"
    r")\b"
)


def _has_long_event_duration_signals(text: str | None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    month_pat = "|".join(sorted(map(re.escape, _RU_MONTHS_GENITIVE.keys()), key=len, reverse=True))
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\s*\.\.\s*20\d{2}-\d{2}-\d{2}\b", raw):
        return True
    if re.search(r"\b\d{1,2}[./]\d{1,2}\s*[-–—]\s*\d{1,2}[./]\d{1,2}\b", raw):
        return True
    if month_pat:
        if re.search(
            rf"\bс\s+\d{{1,2}}\s+(?:{month_pat})\b.*\bпо\s+\d{{1,2}}\s+(?:{month_pat})\b",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return True
        if re.search(rf"\b(до|по)\s+\d{{1,2}}\s+(?:{month_pat})\b", raw, flags=re.IGNORECASE):
            return True
        if re.search(rf"\b\d{{1,2}}\s+(?:{month_pat})\s*[-–—]\s*\d{{1,2}}\s+(?:{month_pat})\b", raw, flags=re.IGNORECASE):
            return True
    return False


def _maybe_apply_default_end_date_for_long_event(candidate: EventCandidate) -> str | None:
    if candidate.end_date:
        return None
    inferred_type = _normalize_event_type_value(
        candidate.title,
        candidate.raw_excerpt or candidate.source_text,
        candidate.event_type,
    )
    if inferred_type != "выставка":
        return None
    # Guardrail: event_type can be misclassified by upstream LLMs.
    # Apply a default 1-month end_date only when the source text looks like a long event
    # (exhibition/exposition) or contains explicit duration signals.
    hay = "\n".join(
        [
            str(candidate.title or ""),
            str(candidate.raw_excerpt or ""),
            str(candidate.source_text or ""),
        ]
    ).strip()
    # "Акция" posts (free ticket days, service campaigns) are not long-running exhibitions.
    # Even if they mention "экспозиции", they should not become 1-month ranges by default.
    if _ACTION_TITLE_RE.search(str(candidate.title or "")) and _ONE_DAY_ACTION_HINT_RE.search(hay):
        return None
    if hay and not (_LONG_EVENT_TEXT_HINT_RE.search(hay) or _has_long_event_duration_signals(hay)):
        return None
    start = _parse_iso_date(candidate.date)
    if not start:
        return None
    candidate.end_date = _add_one_calendar_month(start).isoformat()
    candidate.end_date_is_inferred = True
    return candidate.end_date


def _apply_event_end_date(
    event_db: Event,
    *,
    end_date: str | None,
    inferred: bool,
    updated_keys: list[str],
) -> bool:
    if not end_date:
        return False
    current_end_date = getattr(event_db, "end_date", None)
    current_inferred = bool(getattr(event_db, "end_date_is_inferred", False))

    if current_end_date and (not current_inferred) and inferred:
        return False

    changed = False
    if current_end_date != end_date:
        event_db.end_date = end_date
        changed = True
        if "end_date" not in updated_keys:
            updated_keys.append("end_date")
    if current_inferred != inferred:
        event_db.end_date_is_inferred = inferred
        changed = True
        if "end_date_is_inferred" not in updated_keys:
            updated_keys.append("end_date_is_inferred")
    return changed


def _event_date_range(ev: Event) -> tuple[date | None, date | None]:
    start = _parse_iso_date(ev.date or "")
    end = _parse_iso_date(ev.end_date) if ev.end_date else None
    if not end and ev.date and ".." in ev.date:
        end = _parse_iso_date(ev.date.split("..", 1)[1])
    if start and not end:
        end = start
    return start, end


def _candidate_date_range(candidate: EventCandidate) -> tuple[date | None, date | None]:
    start = _parse_iso_date(candidate.date)
    end = _parse_iso_date(candidate.end_date) if candidate.end_date else None
    if not end and candidate.date and ".." in candidate.date:
        try:
            end = _parse_iso_date(candidate.date.split("..", 1)[1])
        except Exception:
            end = None
    if start and not end:
        end = start
    return start, end


def _smart_update_skip_past_events_enabled() -> bool:
    raw = (os.getenv("SMART_UPDATE_SKIP_PAST_EVENTS") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _smart_update_today_local() -> date:
    try:
        from event_utils import LOCAL_TZ as _LOCAL_TZ
    except Exception:
        _LOCAL_TZ = timezone.utc
    return datetime.now(_LOCAL_TZ).date()


def _should_skip_past_smart_update_candidate(candidate: EventCandidate) -> bool:
    """Skip candidates that have fully ended before today (local).

    This is a guardrail for automated ingestion (VK/TG/parsers) to avoid creating
    useless past events + Telegraph/ICS load.
    """
    if not _smart_update_skip_past_events_enabled():
        return False
    source_type = str(candidate.source_type or "").strip().lower()
    if source_type == "bot":
        return False
    cand_start, cand_end = _candidate_date_range(candidate)
    if not cand_start or not cand_end:
        return False
    today = _smart_update_today_local()
    return cand_end < today


def _ranges_overlap(a_start: date | None, a_end: date | None, b_start: date | None, b_end: date | None) -> bool:
    if not a_start or not a_end or not b_start or not b_end:
        return False
    return not (a_end < b_start or b_end < a_start)


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    value = url.strip()
    if not value:
        return None
    low = value.lower()
    if low.startswith(("t.me/", "vk.cc/", "clck.ru/")):
        value = f"https://{value}"
        low = value.lower()
    if low.startswith("http://"):
        value = "https://" + value[len("http://") :]
    if value.startswith("http://") or value.startswith("https://"):
        value = value.rstrip("/")
    return value


def _is_http_url(url: str | None) -> bool:
    if not url:
        return False
    value = url.strip().lower()
    return value.startswith("http://") or value.startswith("https://")


_VK_WALL_URL_RE = re.compile(
    r"^https?://(?:m\.)?vk\.com/wall-?\d+_\d+/?$",
    re.IGNORECASE,
)


def _is_vk_wall_url(url: str | None) -> bool:
    if not url:
        return False
    if not _is_http_url(url):
        return False
    return bool(_VK_WALL_URL_RE.match(url.strip()))


def _infer_source_type_from_url(url: str | None) -> str:
    """Infer EventSource.source_type for legacy source urls.

    We historically stored a single source link in Event.source_post_url / Event.source_vk_post_url.
    With Smart Update we moved to an explicit event_source table. When merging/updating an older
    event, we backfill that legacy link so the operator can see >=2 sources after a merge.
    """
    value = (url or "").strip().lower()
    if not value:
        return "site"
    if "t.me/" in value:
        return "telegram"
    if _is_vk_wall_url(value):
        return "vk"
    return "site"


async def _ensure_legacy_event_sources(session, event: Event | None) -> int:
    """Ensure legacy single-source fields are represented in event_source.

    Returns number of sources added.
    """
    if not event or not event.id:
        return 0

    urls: list[str] = []
    if _is_http_url(event.source_post_url):
        urls.append(str(event.source_post_url).strip())
    if _is_http_url(event.source_vk_post_url):
        urls.append(str(event.source_vk_post_url).strip())
    if not urls:
        return 0

    clean_source_text = _strip_private_use(event.source_text) or event.source_text
    now = datetime.now(timezone.utc)
    added = 0
    for url in urls:
        exists = (
            await session.execute(
                select(EventSource.id).where(
                    EventSource.event_id == event.id,
                    EventSource.source_url == url,
                )
            )
        ).scalar_one_or_none()
        if exists:
            continue
        session.add(
            EventSource(
                event_id=event.id,
                source_type=_infer_source_type_from_url(url),
                source_url=url,
                source_text=clean_source_text,
                imported_at=now,
            )
        )
        added += 1
    return added


SMART_UPDATE_LEGACY_DESC_FACT_MAX_CHARS = _env_int(
    "SMART_UPDATE_LEGACY_DESC_FACT_MAX_CHARS", 900, lo=200, hi=2500
)
SMART_UPDATE_LEGACY_DESC_FACT_EXTRACT_MIN_CHARS = _env_int(
    # Avoid extracting facts from ultra-short legacy descriptions: the risk of hallucination is high.
    "SMART_UPDATE_LEGACY_DESC_FACT_EXTRACT_MIN_CHARS",
    160,
    lo=80,
    hi=2000,
)

_LEGACY_LEAK_RE = re.compile(r"(?i)\bsmart\s*update\b")


def _drop_legacy_leak_from_description(description: str | None) -> str | None:
    """Remove a previously injected legacy snapshot block from a description.

    This is a backward-compat cleanup for a historical bug where a legacy snapshot
    was treated as an "added fact" and ended up in the public description.
    The heuristic is intentionally narrow: it only drops paragraphs that mention
    "Smart Update" and look like a quoted service block.
    """
    raw = (description or "").strip()
    if not raw:
        return None
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paras:
        return None
    kept: list[str] = []
    for p in paras:
        low = p.casefold()
        looks_like_quote = p.startswith(("«", ">", "“", '"', "'"))
        looks_like_service = _LEGACY_LEAK_RE.search(p) and (":" in p[:60]) and ("текст" in low or "legacy" in low)
        if looks_like_quote and looks_like_service:
            continue
        kept.append(p)
    cleaned = "\n\n".join(kept).strip()
    return cleaned or None


def _legacy_description_to_fact(description: str | None) -> str | None:
    raw = (description or "").strip()
    if not raw:
        return None

    # Keep it compact and neutral: a one-time backfill for legacy events that existed
    # before Smart Update/source facts. This must not inject new claims.
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*#+\s+", "", text)  # markdown headings
    text = re.sub(r"(?m)^\s*>\s*", "", text)  # markdown quotes
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return _clip(text, SMART_UPDATE_LEGACY_DESC_FACT_MAX_CHARS)


async def _ensure_legacy_description_fact(
    session,
    *,
    event: Event | None,
    legacy_description: str | None,
) -> list[str]:
    """Persist a legacy description snapshot (and optionally backfill baseline facts).

    For events created before Smart Update/source facts, a merge can overwrite the previous
    narrative too aggressively. We store a compact snapshot of the old description as a
    dedicated legacy source. In fact-first mode, we also extract atomic baseline facts from
    that legacy text and store them under the legacy source so future fact-first builds do
    not lose pre-existing details.

    Important: we do NOT feed legacy facts into `facts_before` for LLM merge (see query that
    excludes `source_type == "legacy"`). This prevents non-source "snapshots" from polluting
    per-source merge reasoning while still allowing fact-first to use the backfilled facts.
    """

    if not event or not event.id:
        return []

    now = datetime.now(timezone.utc).replace(microsecond=0)
    legacy_url = f"legacy:event_description:{int(event.id)}"
    source = (
        await session.execute(
            select(EventSource).where(
                EventSource.event_id == int(event.id),
                EventSource.source_url == legacy_url,
            )
        )
    ).scalar_one_or_none()
    created_source = False

    snapshot: str | None = None
    if source and getattr(source, "source_text", None):
        snapshot = str(getattr(source, "source_text", None) or "").strip() or None

    if not source:
        snapshot = _legacy_description_to_fact(legacy_description)
        if not snapshot:
            return []

        # Only create a legacy snapshot when the event has no canonical facts yet (legacy/pre-Smart Update).
        # Notes/service rows do not count: we need at least one `added/duplicate` fact from a real source.
        exists_non_legacy = (
            await session.execute(
                select(EventSourceFact.id)
                .join(EventSource, EventSourceFact.source_id == EventSource.id)
                .where(
                    EventSourceFact.event_id == int(event.id),
                    EventSourceFact.status.in_(("added", "duplicate")),
                    EventSource.source_type != "legacy",
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if exists_non_legacy is not None:
            return []

        source = EventSource(
            event_id=int(event.id),
            source_type="legacy",
            source_url=legacy_url,
            source_text=snapshot,
            imported_at=now,
            trust_level="high",
        )
        session.add(source)
        await session.flush()
        created_source = True

    if not snapshot:
        return []
    exists_legacy_facts = (
        await session.execute(
            select(EventSourceFact.id)
            .join(EventSource, EventSourceFact.source_id == EventSource.id)
            .where(
                EventSourceFact.event_id == int(event.id),
                EventSourceFact.status.in_(("added", "duplicate")),
                EventSource.source_type == "legacy",
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if exists_legacy_facts is not None:
        return []

    # Preserve legacy narrative as part of source_texts so merge can keep facts,
    # but do NOT treat it as an "added fact" (facts are for atomic source contributions).
    try:
        texts = list(getattr(event, "source_texts", None) or [])
        if snapshot not in texts:
            texts = [snapshot] + texts
            # Keep it bounded: legacy snapshot is only needed as a one-time baseline.
            setattr(event, "source_texts", texts[:6])
    except Exception:
        pass

    legacy_candidate = EventCandidate(
        source_type="legacy",
        source_url=legacy_url,
        source_text=snapshot,
        title=(getattr(event, "title", None) or "").strip() or None,
        date=(getattr(event, "date", None) or "").strip() or None,
        time=(getattr(event, "time", None) or "").strip() or None,
        end_date=(getattr(event, "end_date", None) or "").strip() or None,
        location_name=(getattr(event, "location_name", None) or "").strip() or None,
        location_address=(getattr(event, "location_address", None) or "").strip() or None,
        city=(getattr(event, "city", None) or "").strip() or None,
        ticket_link=(getattr(event, "ticket_link", None) or "").strip() or None,
        ticket_status=(getattr(event, "ticket_status", None) or "").strip() or None,
        event_type=(getattr(event, "event_type", None) or "").strip() or None,
        trust_level="high",
    )

    fact_records: list[tuple[str, str]] = [("Снапшот описания до Smart Update сохранён", "note")]

    extracted: list[str] = []
    if SMART_UPDATE_FACT_FIRST and not SMART_UPDATE_LLM_DISABLED:
        try:
            raw_for_facts = snapshot
            if created_source:
                raw_for_facts = _drop_legacy_leak_from_description(legacy_description) or legacy_description or snapshot
            cleaned_full = raw_for_facts
            cleaned_full = str(cleaned_full or "").strip()
            if cleaned_full:
                cleaned_full = cleaned_full.replace("\r\n", "\n").replace("\r", "\n")
                cleaned_full = re.sub(r"(?m)^\s*#+\s+", "", cleaned_full)  # markdown headings
                cleaned_full = re.sub(r"(?m)^\s*>\s*", "", cleaned_full)  # markdown quotes
                cleaned_full = re.sub(r"\n{3,}", "\n\n", cleaned_full).strip()
                if len(cleaned_full) < SMART_UPDATE_LEGACY_DESC_FACT_EXTRACT_MIN_CHARS:
                    fact_records.append(
                        (
                            "Legacy описание слишком короткое: факты не извлекались",
                            "note",
                        )
                    )
                    cleaned_full = ""
            if cleaned_full:
                extracted = await _llm_extract_candidate_facts(
                    legacy_candidate,
                    text_for_facts=cleaned_full,
                )
                if extracted:
                    for f in extracted[:18]:
                        fact_records.append((f, "added"))
                    fact_records.append(("Факты извлечены из legacy описания", "note"))
        except Exception:  # pragma: no cover - defensive
            logger.warning("smart_update: legacy fact backfill failed", exc_info=True)
            extracted = []

    try:
        if fact_records:
            await _record_source_facts(session, int(event.id), legacy_candidate, fact_records)
    except Exception:  # pragma: no cover - defensive
        logger.warning("smart_update: failed to persist legacy facts", exc_info=True)

    return extracted


def _normalize_event_type_value(
    title: str | None, description: str | None, event_type: str | None
) -> str | None:
    if not event_type:
        return None
    raw = str(event_type).strip()
    if not raw:
        return None
    aliases = {
        "exhibition": "выставка",
        "fair": "ярмарка",
    }
    canonical = aliases.get(raw.casefold(), raw)
    try:
        from main import normalize_event_type
    except Exception:  # pragma: no cover - defensive
        return canonical
    return normalize_event_type(title or "", description or "", canonical)


def _clean_search_digest(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from digest_helper import clean_search_digest
    except Exception:  # pragma: no cover - defensive
        return value.strip()
    return clean_search_digest(value) or None


def _clean_short_description(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from digest_helper import clean_short_description
    except Exception:  # pragma: no cover - defensive
        return value.strip()
    return clean_short_description(value) or None


def _is_short_description_acceptable(
    value: str | None,
    *,
    min_words: int = 12,
    max_words: int = 16,
) -> bool:
    try:
        from digest_helper import is_short_description_acceptable
    except Exception:  # pragma: no cover - defensive
        cleaned = _clean_short_description(value)
        if not cleaned:
            return False
        words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", cleaned)
        return bool(cleaned.endswith((".", "!", "?")) and min_words <= len(words) <= max_words)
    return bool(
        is_short_description_acceptable(
            value,
            min_words=min_words,
            max_words=max_words,
        )
    )


def _fallback_short_description_from_text(text: str | None) -> str | None:
    try:
        from digest_helper import fallback_one_sentence
    except Exception:
        return None
    raw = fallback_one_sentence(text, max_words=16)
    cleaned = _clean_short_description(raw)
    if not cleaned:
        return None
    if cleaned and cleaned[-1].isalnum():
        cleaned += "."
    return cleaned


async def _llm_build_short_description(
    *,
    title: str | None,
    description: str | None,
    event_type: str | None,
) -> str | None:
    """Build 1-sentence short description for public lists (festival/daily)."""
    if SMART_UPDATE_LLM_DISABLED:
        return None
    desc = (description or "").strip()
    if len(desc) < 80:
        return None
    payload = {
        "title": (title or "").strip(),
        "event_type": (event_type or "").strip(),
        "description": _clip(desc, 1200),
    }
    prompt = (
        "Сделай короткое описание события для публичных списков.\n"
        "Формат: ровно 1 законченное предложение на 12–16 слов.\n"
        "Запрещено: многоточия, обрывы фраз, дата/время/адрес/город/ссылки/эмодзи/хэштеги.\n"
        "Текст должен объяснять суть события простым и ясным языком, без выдумок.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=90,
        label="short_description",
        temperature=0.0,
    )
    cleaned = _clean_short_description(text)
    if not cleaned:
        return None
    if not _is_short_description_acceptable(cleaned, min_words=12, max_words=16):
        return None
    return cleaned


async def _llm_build_search_digest(
    *,
    title: str | None,
    description: str | None,
    event_type: str | None,
) -> str | None:
    """Build/refresh search_digest from the current merged description.

    This text is used as a short "what is this event" snippet (cards/search),
    and is inserted into the Telegraph page before long descriptions.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None
    desc = (description or "").strip()
    if len(desc) < 200:
        return None

    payload = {
        "title": (title or "").strip(),
        "event_type": (event_type or "").strip(),
        "description": _clip(desc, 1800),
    }
    prompt = (
        "Сделай краткий дайджест события для поиска/карточек. "
        "Один абзац: 1 предложение, 120–220 символов (если нужно, максимум 260). "
        "Не указывай дату, время, адрес и город (они показываются отдельно). "
        "Не используй эмодзи, хэштеги, кавычки-цитаты и списки. "
        "Не повторяй название дословно в начале, если оно уже понятно по контексту. "
        "Не добавляй выдуманных фактов.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    text = await _ask_gemma_text(
        prompt,
        max_tokens=180,
        label="search_digest",
        temperature=0.0,
    )
    cleaned = _clean_search_digest(text)
    if not cleaned:
        return None
    cleaned = cleaned.strip().strip("-•").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # If the model returned something too long, prefer deterministic fallback digest
    # rather than cutting mid-word and showing a broken sentence.
    if len(cleaned) > 280:
        return None
    return cleaned or None


async def _llm_create_description_facts_and_digest(
    candidate: EventCandidate,
    *,
    clean_title: str,
    clean_source_text: str,
    clean_raw_excerpt: str | None,
    normalized_event_type: str | None,
) -> dict[str, Any] | None:
    """Bundle create-time LLM work into a single Gemma JSON call.

    This replaces three separate LLM calls previously used on create:
    - rewrite description,
    - extract atomic facts,
    - build search_digest.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None
    enabled = (os.getenv("SMART_UPDATE_CREATE_BUNDLE", "1") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return None

    payload = {
        "title": clean_title,
        "date": candidate.date,
        "time": candidate.time,
        "end_date": candidate.end_date,
        "location_name": candidate.location_name,
        "location_address": candidate.location_address,
        "city": candidate.city,
        "ticket_link": candidate.ticket_link,
        "ticket_status": candidate.ticket_status,
        "is_free": bool(candidate.is_free),
        "event_type": normalized_event_type or candidate.event_type,
        "festival": candidate.festival,
        "source_type": candidate.source_type,
        "source_url": candidate.source_url,
        "source_text": _clip(clean_source_text, SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS),
        "raw_excerpt": _clip(clean_raw_excerpt or "", 1200),
        "poster_texts": [_clip(p.ocr_text, 700) for p in candidate.posters if (p.ocr_text or "").strip()][
            :3
        ],
        "poster_titles": [
            _clip(p.ocr_title, 140) for p in candidate.posters if (p.ocr_title or "").strip()
        ][:3],
    }
    # Budget for public description length: keep close to total source volume to avoid
    # hallucination-prone over-expansion on short posts.
    try:
        payload["description_budget_chars"] = _estimate_description_budget_chars(
            source_type=candidate.source_type,
            source_text=clean_source_text,
            raw_excerpt=clean_raw_excerpt,
            poster_texts=[p.ocr_text for p in candidate.posters if (p.ocr_text or "").strip()],
        )
    except Exception:  # pragma: no cover - defensive
        payload["description_budget_chars"] = 820
    prompt = (
        "Ты готовишь данные для создания события.\n"
        "Верни JSON строго по схеме.\n\n"
        "0) title:\n"
        "- Верни короткое и осмысленное название события (обычно 3–12 слов).\n"
        "- НЕ включай дату/время/адрес/город/цены/ссылки.\n"
        "- Если `poster_titles` содержит крупный заголовок афиши и он относится к событию, используй его как основу title.\n"
        "- Если в source_text/raw_excerpt/poster_texts есть явное собственное название (проект/тур/постановка/шоу), используй его как основу title.\n"
        "- НЕ делай title в формате «<event_type> — <площадка>», если в данных есть имя/бренд события (пример: «ЕвроДэнс'90», а не «Концерт — Янтарь холл»).\n"
        "- Не теряй ключевые смысловые маркеры (например «Масленица», «концерт», «кинопоказ», «лекция»), если они есть в данных, но НЕ подменяй ими собственное название.\n\n"
        "1) description:\n"
        "- Напиши ПОЛНОЕ развернутое описание события как культурный журналист.\n"
        "- Сохрани ВСЕ значимые факты из source_text/raw_excerpt/poster_texts (кроме логистики).\n"
        "- Если source_text короткий или пустой, опирайся на poster_texts (OCR афиш) как на основной источник фактов.\n"
        "- Объём: описание должно быть близко по объёму к источникам и НЕ превышать `description_budget_chars` символов.\n"
        "  Если источники короткие, описание тоже должно быть коротким (без воды/«атмосферных» вступлений).\n"
        "- Не копируй дословно длинными кусками; перефразируй, но не сокращай смысл.\n"
        "- Анти-дословность: избегай фрагментов, где подряд совпадает больше ~8–10 слов с источником.\n"
        "- Структура: абзацы, разделяй пустой строкой; 1–2 предложения в абзаце (для списков правило не применимо).\n"
        "- Запрещено: хэштеги, рекламные клише/прогнозы, механика розыгрыша.\n"
        "- НЕ добавляй секции/заголовки про факты (например «Факты…», «Факты о событии», `Facts/Added Facts`):\n"
        "  факты вернутся отдельным полем `facts`.\n"
        "- НЕ экранируй кавычки обратным слэшем (не пиши `\\\"...\\\"`).\n"
        "- ВАЖНО: НЕ включай в текст логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены)\n"
        "  и не дублируй строки `Дата:`, `Время:`, `Локация:`, `Билеты:`.\n"
        f"{SMART_UPDATE_YO_RULE}\n"
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE}\n\n"
        f"{SMART_UPDATE_OPTIONAL_HEADINGS_RULE}\n\n"
        f"{_description_emoji_prompt_rule()}\n\n"
        "Если в описании есть список:\n"
        "- Каждый пункт списка должен быть на отдельной строке.\n"
        "- Маркер списка пиши с пробелом: `- пункт` или `1. пункт`.\n\n"
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE}\n\n"
        "2) facts:\n"
        "- Верни 6–18 атомарных фактов (1 факт = 1 строка), только про ЭТО событие.\n"
        "- НЕ включай дату/время/адрес/город как отдельные факты.\n"
        f"- {SMART_UPDATE_FACTS_PRESERVE_COMPACT_PROGRAM_LISTS_RULE}\n"
        "- НЕ включай скидки/промокоды/призывы подписаться/ссылки на каналы.\n"
        "- Включай условия участия/посещения (длительность, возраст, размер группы, формат/что взять/как одеться, "
        "что входит/не входит в оплату), без ссылок/телефонов; сумму указывай только если это важно, чтобы пояснить "
        "что оплачивается отдельно (не более 1 факта).\n\n"
        "- Если есть прямая речь и понятно, кто говорит, оформи как `Цитата (Имя Фамилия): ...`.\n\n"
        "3) search_digest:\n"
        "- 1 предложение, 120–220 символов (макс 260), без эмодзи/хэштегов/списков.\n"
        "- Не указывай дату/время/адрес/город/цены/ссылки.\n"
        "- Не повторяй название дословно в начале.\n\n"
        "4) short_description:\n"
        "- Ровно 1 законченное предложение на 12–16 слов.\n"
        "- Без многоточий и обрывов фраз.\n"
        "- Не указывай дату/время/адрес/город/цены/ссылки.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    max_tokens = SMART_UPDATE_REWRITE_MAX_TOKENS
    data = await _ask_gemma_json(prompt, CREATE_BUNDLE_SCHEMA, max_tokens=max_tokens, label="create_bundle")
    if not isinstance(data, dict):
        return None
    return data


def _trust_priority(level: str | None) -> int:
    if not level:
        return 2
    key = level.strip().lower()
    if key == "high":
        return 3
    if key == "medium":
        return 2
    if key == "low":
        return 1
    return 2


def _max_trust_level(levels: Sequence[str | None]) -> tuple[str | None, int]:
    best_level: str | None = None
    best_priority = -1
    for lvl in levels:
        pr = _trust_priority(lvl)
        if pr > best_priority:
            best_priority = pr
            best_level = lvl
    if best_priority < 0:
        return None, _trust_priority(None)
    return best_level, best_priority


def _is_long_event_type_value(event_type: str | None) -> bool:
    if not event_type:
        return False
    return str(event_type).strip().casefold() in {"выставка", "ярмарка"}


def _extract_hall_hint(text: str | None) -> str | None:
    if not text:
        return None
    match = _HALL_HINT_RE.search(text)
    if not match:
        return None
    parts = [p for p in match.groups() if p]
    if not parts:
        return None
    return _norm_space(" ".join(parts))


@lru_cache(maxsize=1)
def _load_location_flags() -> dict[str, dict[str, Any]]:
    path = os.path.join("docs", "reference", "location-flags.md")
    flags: dict[str, dict[str, Any]] = {}
    if not os.path.exists(path):
        return flags
    current: str | None = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m_loc = re.match(r"-\s*location_name:\s*\"?(.+?)\"?$", line)
                if m_loc:
                    current = m_loc.group(1).strip()
                    flags[current] = {"allow_parallel_events": False}
                    continue
                if current:
                    m_flag = re.match(r"allow_parallel_events:\s*(true|false)", line, re.I)
                    if m_flag:
                        flags[current]["allow_parallel_events"] = m_flag.group(1).lower() == "true"
    except Exception as exc:
        logger.warning("smart_update: failed to read location flags: %s", exc)
    return flags


def _allow_parallel_events(location_name: str | None) -> bool:
    if not location_name:
        return False
    flags = _load_location_flags()
    for name, data in flags.items():
        if _normalize_location(name) == _normalize_location(location_name):
            return bool(data.get("allow_parallel_events"))
    return False


def _clip(text: str | None, limit: int = 1200) -> str:
    if not text:
        return ""
    raw = text.strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3].rstrip() + "..."


def _clip_title(text: str | None, limit: int = 80) -> str:
    if not text:
        return ""
    raw = text.strip()
    return raw if len(raw) <= limit else raw[: limit - 1].rstrip() + "…"


def _estimate_description_budget_chars(
    *,
    source_type: str | None,
    source_text: str | None,
    raw_excerpt: str | None,
    poster_texts: Sequence[str] | None,
) -> int:
    """Estimate a safe max length for public event descriptions.

    Goal: avoid LLM "over-expanding" short sources (hallucination-prone water/cliches).
    This is a *budget* for LLM prompts / LLM rewrite passes, not a deterministic cutter.
    """
    src = (source_text or "").strip()
    excerpt = (raw_excerpt or "").strip()
    # Avoid double-counting the excerpt if it is already inside the source.
    extra_excerpt = ""
    if excerpt and excerpt not in src:
        extra_excerpt = excerpt

    # OCR can be huge; include it but cap the contribution to keep budgets sane.
    ocr_chunks: list[str] = []
    for t in (poster_texts or [])[:5]:
        s = (t or "").strip()
        if s:
            ocr_chunks.append(s)
    ocr_total = sum(min(len(s), 900) for s in ocr_chunks)
    ocr_total = min(ocr_total, 2400)

    base = len(src) + len(extra_excerpt) + ocr_total
    if base <= 0:
        return 520  # fallback for empty sources (still keep it modest)

    st = (source_type or "").strip().lower()
    # Be stricter for Telegram: many posts are short, and "water" is very visible.
    factor = 1.05 if st.startswith("telegram") else 1.12

    budget = int(base * factor)
    # Clamp to a practical range for Telegraph pages.
    budget = max(220, min(1800, budget))
    # For extremely short sources, allow a small fixed slack for readability.
    if base < 260:
        budget = max(260, min(budget, base + 120))
    return int(budget)


async def _llm_shrink_description_to_budget(
    *,
    source_type: str | None,
    source_url: str | None,
    description: str,
    source_text: str | None,
    facts: Sequence[str] | None,
    max_chars: int,
) -> str | None:
    """LLM-only shrinking pass to keep description close to source volume (no new facts)."""
    if SMART_UPDATE_LLM_DISABLED:
        return None
    text = (description or "").strip()
    if not text:
        return None
    try:
        max_chars_i = int(max_chars)
    except Exception:
        max_chars_i = 520
    max_chars_i = max(180, min(2200, max_chars_i))

    payload = {
        "max_chars": max_chars_i,
        "source_type": (source_type or "").strip(),
        "source_url": (source_url or "").strip(),
        "facts": [str(x).strip() for x in (facts or []) if str(x).strip()][:18],
        "source_text": _clip((source_text or "").strip(), 1800),
        "description": _clip(text, 3200),
    }
    prompt = (
        "Сократи описание события до указанного лимита символов.\n"
        "Правила:\n"
        "- НЕ добавляй новых фактов и деталей.\n"
        "- Убирай воду и нейросетевые клише.\n"
        "- Не указывай дату/время/адрес/город/ссылки/телефоны.\n"
        "- Сохрани смысловые маркеры (жанр/формат/что будет происходить).\n"
        "- Пиши по-русски, естественно, без канцелярита.\n"
        "- Верни только готовый текст описания, без заголовков типа Facts/Факты.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    # Rough token budget: ~1 token per 4 chars + some slack.
    max_tokens = max(140, min(520, int(max_chars_i / 4) + 80))
    out = await _ask_gemma_text(prompt, max_tokens=max_tokens, label="shrink_desc", temperature=0.0)
    out = (out or "").strip()
    out = _sanitize_description_output(out, source_text=source_text) or out
    out = _normalize_plaintext_paragraphs(out) or out
    out = _normalize_blockquote_markers(out) or out
    if not out.strip():
        return None
    if len(out) > max_chars_i * 1.15:
        return None
    return out.strip()


async def _fetch_event_posters_map(
    db: Database, event_ids: Sequence[int]
) -> dict[int, list[EventPoster]]:
    if not event_ids:
        return {}
    async with db.get_session() as session:
        result = await session.execute(
            select(EventPoster).where(EventPoster.event_id.in_(event_ids))
        )
        posters = list(result.scalars().all())
    grouped: dict[int, list[EventPoster]] = {}
    for poster in posters:
        grouped.setdefault(poster.event_id, []).append(poster)
    return grouped


def _poster_hashes(posters: Iterable[PosterCandidate]) -> set[str]:
    hashes: set[str] = set()
    for poster in posters:
        if poster.sha256:
            hashes.add(poster.sha256)
    return hashes


async def _llm_match_event(
    candidate: EventCandidate,
    events: Sequence[Event],
    *,
    posters_map: dict[int, list[EventPoster]] | None = None,
) -> tuple[int | None, float, str]:
    if not events:
        return None, 0.0, "shortlist_empty"
    if SMART_UPDATE_LLM_DISABLED:
        return None, 0.0, "llm_disabled"

    posters_map = posters_map or {}
    candidates_payload: list[dict[str, Any]] = []
    for ev in events:
        posters = posters_map.get(ev.id or 0, [])
        poster_texts = [p.ocr_text for p in posters if p.ocr_text][:2]
        poster_titles = [p.ocr_title for p in posters if (p.ocr_title or "").strip()][:2]
        candidates_payload.append(
            {
                "id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "time": ev.time,
                "time_is_default": bool(getattr(ev, "time_is_default", False)),
                "end_date": ev.end_date,
                "location_name": ev.location_name,
                "location_address": ev.location_address,
                "city": ev.city,
                "ticket_link": ev.ticket_link,
                "description": _clip(ev.description, 600),
                "source_text": _clip(ev.source_text, 600),
                "poster_texts": poster_texts,
                "poster_titles": poster_titles,
            }
        )

    payload = {
        "candidate": {
            "title": candidate.title,
            "date": candidate.date,
            "time": candidate.time,
            "time_is_default": bool(getattr(candidate, "time_is_default", False)),
            "end_date": candidate.end_date,
            "location_name": candidate.location_name,
            "location_address": candidate.location_address,
            "city": candidate.city,
            "ticket_link": candidate.ticket_link,
            "text": _clip(_strip_promo_lines(candidate.source_text) or candidate.source_text, 1200),
            "raw_excerpt": _clip(_strip_promo_lines(candidate.raw_excerpt) or candidate.raw_excerpt, 800),
            "poster_texts": [
                _clip(p.ocr_text, 400) for p in candidate.posters if p.ocr_text
            ][:3],
            "poster_titles": [
                _clip(p.ocr_title, 140) for p in candidate.posters if (p.ocr_title or "").strip()
            ][:3],
        },
        "events": candidates_payload[:10],
    }
    prompt = (
        "Ты сопоставляешь анонс события с уже существующими событиями.\n"
        "Найди наиболее вероятное совпадение или верни null.\n"
        "Учитывай дату, время, площадку, участников, ссылки, афиши и OCR.\n"
        "Важно про дубли:\n"
        "- `time=00:00` и/или `time_is_default=true` считай неизвестным временем (слабый якорь, не конфликт).\n"
        "- Если совпадают дата + площадка + контекст (участник/афиша/OCR), а название сформулировано по-разному "
        "(общее vs конкретное), это всё равно один и тот же ивент: выбирай match.\n"
        "- Для длинных событий (выставка/ярмарка/экспозиция с `end_date`) пересечение периодов + площадка НЕ означает дубль:\n"
        "  в одном музее может идти несколько разных выставок одновременно. Матчь только если совпадает конкретное событие "
        "(название/автор/тематика/афиша/OCR/ссылка).\n"
        "- Если среди `events` есть событие с той же датой, тем же началом времени (или время пустое/placeholder), "
        "той же площадкой и тем же названием (или очевидно тем же) — это дубль: выбери его и поставь высокую confidence.\n"
        "- Не возвращай null, если есть правдоподобный матч: лучше выбрать наиболее вероятное и снизить confidence.\n"
        "Ответь строго JSON.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    data = await _ask_gemma_json(
        prompt,
        MATCH_SCHEMA,
        max_tokens=400,
        label="match",
    )
    if data is None:
        return None, 0.0, "llm_bad_json"
    match_id = data.get("match_event_id")
    confidence = data.get("confidence")
    reason = data.get("reason_short") or ""
    try:
        conf_val = float(confidence)
    except Exception:
        conf_val = 0.0
    if match_id is None:
        return None, conf_val, reason
    try:
        match_id = int(match_id)
    except Exception:
        return None, conf_val, reason
    return match_id, conf_val, reason


async def _llm_match_or_create_bundle(
    candidate: EventCandidate,
    events: Sequence[Event],
    *,
    posters_map: dict[int, list[EventPoster]] | None = None,
    threshold: float,
    clean_title: str,
    clean_source_text: str,
    clean_raw_excerpt: str | None,
    normalized_event_type: str | None,
) -> dict[str, Any] | None:
    """Single-call LLM step: match existing event OR return create bundle.

    Intended for VK/TG imports to reduce roundtrips on the "no match -> create" path.
    """
    if not events:
        return None
    if SMART_UPDATE_LLM_DISABLED:
        return None

    posters_map = posters_map or {}
    candidates_payload: list[dict[str, Any]] = []
    for ev in events:
        posters = posters_map.get(ev.id or 0, [])
        poster_texts = [p.ocr_text for p in posters if p.ocr_text][:2]
        poster_titles = [p.ocr_title for p in posters if (p.ocr_title or "").strip()][:2]
        candidates_payload.append(
            {
                "id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "time": ev.time,
                "time_is_default": bool(getattr(ev, "time_is_default", False)),
                "end_date": ev.end_date,
                "location_name": ev.location_name,
                "location_address": ev.location_address,
                "city": ev.city,
                "ticket_link": ev.ticket_link,
                "description": _clip(ev.description, 600),
                "source_text": _clip(ev.source_text, 600),
                "poster_texts": poster_texts,
                "poster_titles": poster_titles,
            }
        )

    payload = {
        "threshold": float(threshold),
        "candidate": {
            "title": clean_title,
            "date": candidate.date,
            "time": candidate.time,
            "time_is_default": bool(getattr(candidate, "time_is_default", False)),
            "end_date": candidate.end_date,
            "location_name": candidate.location_name,
            "location_address": candidate.location_address,
            "city": candidate.city,
            "ticket_link": candidate.ticket_link,
            "ticket_status": candidate.ticket_status,
            "is_free": bool(candidate.is_free),
            "event_type": normalized_event_type or candidate.event_type,
            "festival": candidate.festival,
            "source_type": candidate.source_type,
            "source_url": candidate.source_url,
            "source_text": _clip(clean_source_text, SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS),
            "raw_excerpt": _clip(clean_raw_excerpt or "", 1200),
            "poster_texts": [_clip(p.ocr_text, 400) for p in candidate.posters if p.ocr_text][:3],
            "poster_titles": [
                _clip(p.ocr_title, 140) for p in candidate.posters if (p.ocr_title or "").strip()
            ][:3],
        },
        "events": candidates_payload[:10],
    }
    prompt = (
        "Ты сопоставляешь новый анонс события с уже существующими событиями.\n"
        "Верни JSON строго по схеме.\n\n"
        "Шаг 1) MATCH:\n"
        "- Найди наиболее вероятное совпадение среди `events` или реши, что совпадения нет.\n"
        "- `confidence` от 0 до 1.\n"
        "- Если `confidence >= threshold`, верни `action=match` и `match_event_id`.\n"
        "- Если `confidence < threshold`, верни `action=create` и `match_event_id=null`.\n\n"
        "Анти-дубли (важно):\n"
        "- `time=00:00` и/или `time_is_default=true` считай неизвестным временем (слабый якорь, не конфликт).\n"
        "- Если совпадают дата + площадка + контекст (участник/афиша/OCR), но формулировка названия отличается "
        "(общее vs конкретное), это один и тот же ивент: выбирай `action=match`.\n"
        "- Для длинных событий (выставка/ярмарка/экспозиция с `end_date`) пересечение периодов + площадка НЕ означает дубль:\n"
        "  в одном музее может идти несколько разных выставок одновременно. Выбирай `action=match` только если совпадает "
        "конкретное событие (название/автор/тематика/афиша/OCR/ссылка).\n"
        "- Если хотя бы одно событие в `events` совпадает по якорям (дата + начало времени/пустое время + площадка) "
        "и названию/участникам, это дубль — выбирай `action=match` и ставь `confidence` заметно выше `threshold`.\n\n"
        "Шаг 2) CREATE (только если action=create):\n"
        "- Верни `bundle` с полями title/description/facts/search_digest/short_description как для создания нового события.\n"
        "- Если action=match, верни bundle=null.\n\n"
        "Правила для bundle.title:\n"
        "- Короткое осмысленное название (обычно 3–12 слов), без даты/времени/адреса/города/цен/ссылок.\n"
        "- Без эмодзи.\n"
        "- Если `candidate.poster_titles` содержит крупный заголовок афиши и он относится к событию, используй его как основу.\n"
        "- Если в candidate.source_text/raw_excerpt/poster_texts есть явное собственное название (проект/тур/постановка/шоу), используй его как основу.\n"
        "- НЕ делай title в формате «<event_type> — <площадка>», если в данных есть имя/бренд события (пример: «ЕвроДэнс'90», а не «Концерт — Янтарь холл»).\n"
        "- Не экранируй кавычки обратным слэшем (не пиши `\\\"...\\\"`).\n\n"
        "Правила для bundle.description:\n"
        "- Напиши ПОЛНОЕ развернутое описание как культурный журналист.\n"
        "- Сохрани ВСЕ значимые факты из source_text/raw_excerpt/poster_texts (кроме логистики).\n"
        "- Если source_text короткий или пустой, опирайся на poster_texts (OCR афиш) как на основной источник фактов.\n"
        "- Не копируй дословно длинными кусками; перефразируй, но не сокращай смысл.\n"
        "- Структура: абзацы, разделяй пустой строкой; 1–2 предложения в абзаце.\n"
        "- Запрещено: хэштеги, рекламные клише/прогнозы, механика розыгрыша.\n"
        f"- {_description_emoji_prompt_rule().strip()}\n"
        "- НЕ добавляй секции/заголовки про факты (например «Факты…», «Факты о событии», `Facts/Added Facts`):\n"
        "  факты вернутся отдельным полем bundle.facts.\n"
        "- НЕ экранируй кавычки обратным слэшем (не пиши `\\\"...\\\"`).\n"
        "- ВАЖНО: НЕ включай логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены)\n"
        "  и не дублируй строки `Дата:`, `Время:`, `Локация:`, `Билеты:`.\n"
        f"{SMART_UPDATE_YO_RULE}\n"
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE}\n\n"
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE}\n\n"
        "Правила для bundle.facts:\n"
        "- 6–18 атомарных фактов (1 факт = 1 строка), только про ЭТО событие.\n"
        "- НЕ включай дату/время/адрес/город как отдельные факты.\n"
        "- НЕ включай скидки/промокоды/призывы подписаться/ссылки на каналы.\n"
        "- Включай условия участия/посещения (длительность, возраст, размер группы, формат/что взять/как одеться, "
        "что входит/не входит в оплату), без ссылок/телефонов; сумму указывай только если это важно, чтобы пояснить "
        "что оплачивается отдельно (не более 1 факта).\n\n"
        "Правила для bundle.search_digest:\n"
        "- 1 предложение, 120–220 символов (макс 260), без эмодзи/хэштегов/списков.\n"
        "- Не указывай дату/время/адрес/город/цены/ссылки.\n"
        "- Не повторяй название дословно в начале.\n\n"
        "Правила для bundle.short_description:\n"
        "- Ровно 1 законченное предложение на 12–16 слов.\n"
        "- Без многоточий и обрывов фраз.\n"
        "- Не указывай дату/время/адрес/город/цены/ссылки.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    data = await _ask_gemma_json(
        prompt,
        MATCH_CREATE_BUNDLE_SCHEMA,
        max_tokens=SMART_UPDATE_REWRITE_MAX_TOKENS,
        label="match_create_bundle",
    )
    if not isinstance(data, dict):
        return None
    action = (data.get("action") or "").strip().lower()
    if action not in {"match", "create"}:
        return None
    if action == "match":
        return {
            "action": "match",
            "match_event_id": data.get("match_event_id"),
            "confidence": data.get("confidence"),
            "reason_short": data.get("reason_short") or "",
            "bundle": None,
        }
    bundle = data.get("bundle")
    if not isinstance(bundle, dict):
        return None
    return {
        "action": "create",
        "match_event_id": None,
        "confidence": data.get("confidence"),
        "reason_short": data.get("reason_short") or "",
        "bundle": bundle,
    }


async def _llm_merge_event(
    candidate: EventCandidate,
    event: Event,
    *,
    conflicting_anchor_fields: dict[str, Any] | None = None,
    poster_texts: Sequence[str] | None = None,
    facts_before: Sequence[str] | None = None,
    event_trust_level: str | None = None,
    candidate_trust_level: str | None = None,
) -> dict[str, Any] | None:
    if SMART_UPDATE_LLM_DISABLED:
        return None

    payload = {
        "event_before": {
            "title": event.title,
            "description": _clip(event.description, SMART_UPDATE_MERGE_EVENT_DESC_MAX_CHARS),
            "facts": [
                _clip(str(f), 220) for f in (facts_before or []) if isinstance(f, str) and f.strip()
            ][:60],
            "trust_level": event_trust_level,
            "trust_priority": _trust_priority(event_trust_level),
            "ticket_link": event.ticket_link,
            "ticket_price_min": event.ticket_price_min,
            "ticket_price_max": event.ticket_price_max,
            "ticket_status": getattr(event, "ticket_status", None),
            "source_texts": [
                _clip(t, 1200)
                for t in (getattr(event, "source_texts", None) or [])
                if isinstance(t, str) and t.strip()
            ][:4],
        },
        "candidate": {
            "title": candidate.title,
            "raw_excerpt": _clip(_strip_promo_lines(candidate.raw_excerpt) or candidate.raw_excerpt, 1200),
            "text": _clip(
                _strip_promo_lines(candidate.source_text) or candidate.source_text,
                SMART_UPDATE_MERGE_CANDIDATE_TEXT_MAX_CHARS,
            ),
            "trust_level": candidate_trust_level,
            "trust_priority": _trust_priority(candidate_trust_level),
            "ticket_link": candidate.ticket_link,
            "ticket_price_min": candidate.ticket_price_min,
            "ticket_price_max": candidate.ticket_price_max,
            "ticket_status": candidate.ticket_status,
            "source_url": candidate.source_url,
            "quote_candidates": _extract_quote_candidates(
                _strip_promo_lines(candidate.source_text) or candidate.source_text,
                max_items=2,
            ),
            "poster_texts": [
                _clip(p.ocr_text, 400) for p in candidate.posters if p.ocr_text
            ][:3],
            "poster_titles": [
                _clip(p.ocr_title, 140) for p in candidate.posters if (p.ocr_title or "").strip()
            ][:3],
        },
        "constraints": {
            "anchor_fields_do_not_change": [
                "date",
                "time",
                "end_date",
                "location_name",
                "location_address",
            ],
            "conflicting_do_not_use": conflicting_anchor_fields or {},
        },
    }
    if poster_texts:
        payload["candidate"]["existing_poster_texts"] = list(poster_texts)[:3]

    prompt = (
        "Ты объединяешь информацию о событии. "
        "Никогда не меняй якорные поля (дата/время/площадка/адрес). "
        "Если кандидат содержит противоречия в якорных полях, игнорируй их. "
        "Добавляй только непротиворечивые факты. "
        "Считай `event_before.facts` каноническим набором уже известных фактов о событии. "
        "Твоя задача: (1) выделить из candidate ТОЛЬКО новые факты, которых ещё нет в event_before.facts, "
        "(2) выделить факты из candidate, которые уже есть (это дубли), "
        "(3) выявить факты, которые ПРОТИВОРЕЧАТ уже известным фактам (conflict), "
        "(4) собрать цельное, связное описание события на основе event_before.facts + новых фактов. "
        "Конфликты фактов выявляй логически: если новый факт противоречит старому, это conflict. "
        "Какую версию оставить в описании — решай по уровню доверия источников: "
        "если `candidate.trust_priority` выше, можно заменить старую версию на новую, "
        "если ниже или равен — сохраняй старую версию. "
        "Любой конфликт обязательно опиши в `conflict_facts` с указанием, какая версия выбрана "
        "(например `Старый факт -> Новый факт (выбран: candidate)` или `(выбран: event_before)`). "
        "Обязательно старайся добавлять конкретные новые детали из кандидата, которых нет в текущем описании (имена/участники/уникальные детали/программа). "
        "Не повторяй уже имеющиеся факты (убирай дубли). "
        "Если `candidate.poster_titles` содержит короткий крупный заголовок афиши и он относится к событию, "
        "то при необходимости улучши `title` так, чтобы он был близок по смыслу к этому заголовку "
        "(но не включай дату/время/адрес/город/цены/ссылки и не используй эмодзи). "
        f"{SMART_UPDATE_YO_RULE} "
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE} "
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE} "
        "Описание должно читаться как единый связный текст-повествование (не рваное). "
        "Разбиение на абзацы делай осмысленно. НЕ разрывай предложения пустой строкой на середине, "
        "и особенно не ставь пустую строку между инициалом и фамилией (например `Н. Любимова`). "
        f"{_description_emoji_prompt_rule()}"
        "Без хэштегов. "
        "Описание должно быть журналистским рерайтом (не дословно), без выдуманных деталей. "
        "Запрещено придумывать факты, которых нет в данных (в т.ч. нельзя писать 'премьера', 'впервые', 'аншлаг' и т.п., "
        "если это явно не сказано в источниках). "
        "Избегай нейросетевых клише и пустых оценок/прогнозов: "
        "не пиши фразы вроде 'обещает стать заметным событием', 'не оставит равнодушным', 'уникальная возможность', "
        "'незабываемые эмоции' и т.п. Если оценка есть в источнике, атрибутируй её ('по словам организаторов/в анонсе'). "
        "Не включай в описание нерелевантные новости о площадке/организации, которые не относятся к самому событию "
        "(например отчёты о работе филиала, планы на год, анонс посторонних интервью). "
        "Сохраняй ПОЛНОЕ содержание события: включай существенные факты из event_before.description, source_texts и candidate.text. "
        "Не делай текст чрезмерно коротким: если источники длинные, итоговое описание тоже должно быть развернутым и подробным. "
        "Убери рекламные/акционные детали (скидки/промокоды/акция) и механику розыгрыша, если они не являются сутью события. "
        "Если в тексте есть URL или телефоны, не искажай их (лучше перенеси в конец, чем потерять). "
        "Можно использовать минимальную разметку для читабельности: "
        "заголовки `###`, цитаты блоком `> ...`, редкое выделение `**...**`. "
        "НЕ используй Markdown-ссылки вида `[текст](url)` и не вставляй таблицы. "
        "Не включай малозначимые и повторяющиеся строки (например `DD.MM | Название`, повтор заголовка, повтор даты/времени/площадки отдельной строкой). "
        "Если в источнике есть обрыв фразы/текста (в т.ч. обрезано на середине слова), не вставляй это дословно: либо перефразируй, либо опусти. "
        "Не экранируй кавычки обратным слэшем (не пиши `\\\"...\\\"`). "
        "Если в материалах есть прямая речь/цитата (1-е лицо: 'я/мне/кажется/думаю' и т.п.), "
        "НЕ переписывай её в косвенную речь: включи её ДОСЛОВНО как цитату блоком `>` и не дублируй ту же мысль пересказом рядом. "
        "Если `candidate.quote_candidates` не пуст, обязательно включи хотя бы одну из этих цитат ДОСЛОВНО как blockquote. "
        "Если цитата принадлежит конкретному человеку (например режиссёру), укажи это явно: "
        "либо перед цитатой, либо сразу после неё в виде краткой атрибуции (например `— Егор Равинский`). "
        "Структуру делай абзацами: разделяй абзацы пустой строкой. "
        "В каждом абзаце держи 1-2 предложения (максимум 3 только если иначе теряется смысл). "
        "Не дублируй в основном тексте строки-анкеры (`Дата:`, `Время:`, `Локация:`, `Билеты:`) и их явные перефразы: "
        "эти данные уже показываются отдельным блоком. "
        "Также верни `search_digest`: 1 предложение, 120–220 символов (макс 260), без эмодзи/хэштегов/списков; "
        "не указывай дату/время/адрес/город/цены/ссылки; не начинай с дословного повторения title. "
        "Верни JSON с полями title (если нужно улучшить), description (обязательно), search_digest, "
        "ticket_link, ticket_price_min/max, ticket_status, added_facts, duplicate_facts, conflict_facts, skipped_conflicts. "
        "added_facts должен содержать список КОНКРЕТНЫХ НОВЫХ фактов (короткими пунктами), которых НЕ было в event_before.facts. "
        "duplicate_facts должен содержать список фактов из candidate, которые уже есть в event_before.facts (дубли). "
        "conflict_facts должен содержать список конфликтов (см. выше) и выбранную сторону по доверию. "
        "Не включай в added_facts и duplicate_facts служебные заметки. "
        "\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    data = await _ask_gemma_json(
        prompt,
        MERGE_SCHEMA,
        max_tokens=SMART_UPDATE_MERGE_MAX_TOKENS,
        label="merge",
    )
    if data is None:
        logger.warning("smart_update: merge invalid json (gemma)")
        return None
    if isinstance(data, dict) and ("duplicate_facts" not in data or data.get("duplicate_facts") is None):
        data["duplicate_facts"] = []
    if isinstance(data, dict) and ("conflict_facts" not in data or data.get("conflict_facts") is None):
        data["conflict_facts"] = []
    return data


def _apply_ticket_fields(
    event: Event,
    *,
    ticket_link: str | None,
    ticket_price_min: int | None,
    ticket_price_max: int | None,
    ticket_status: str | None,
    candidate_trust: str | None,
) -> list[str]:
    added: list[str] = []
    cand_priority = _trust_priority(candidate_trust)
    existing_priority = _trust_priority(getattr(event, "ticket_trust_level", None))

    def _can_override(existing: Any) -> bool:
        if existing in (None, ""):
            return True
        return cand_priority > existing_priority

    if ticket_link and _can_override(event.ticket_link):
        event.ticket_link = ticket_link
        event.ticket_trust_level = candidate_trust
        added.append("ticket_link")
    if ticket_price_min is not None and _can_override(event.ticket_price_min):
        event.ticket_price_min = ticket_price_min
        event.ticket_trust_level = candidate_trust
        added.append("ticket_price_min")
    if ticket_price_max is not None and _can_override(event.ticket_price_max):
        event.ticket_price_max = ticket_price_max
        event.ticket_trust_level = candidate_trust
        added.append("ticket_price_max")
    if ticket_status and _can_override(getattr(event, "ticket_status", None)):
        setattr(event, "ticket_status", ticket_status)
        event.ticket_trust_level = candidate_trust
        added.append("ticket_status")
    return added


def _candidate_has_new_text(candidate: EventCandidate, event: Event) -> bool:
    def _normalize(text: str | None) -> str:
        raw = _strip_private_use(text) or (text or "")
        raw = _strip_promo_lines(raw) or raw
        raw = _strip_giveaway_lines(raw) or raw
        return raw.strip()

    def _sentences(text: str) -> list[str]:
        chunks = re.split(r"[.!?…]\s+|\n{2,}|\n", text)
        out: list[str] = []
        for chunk in chunks:
            c = re.sub(r"\s+", " ", chunk).strip()
            if c:
                out.append(c)
        return out

    event_text = _normalize(event.description)
    candidates = [_normalize(candidate.source_text), _normalize(candidate.raw_excerpt)]
    candidates = [c for c in candidates if c]
    if not candidates:
        return False
    if not event_text:
        return True

    event_lower = event_text.lower()
    for cand in candidates:
        if len(cand) < 40:
            continue
        # Prefer sentence-level detection: raw_excerpt may omit new details even when source_text contains them.
        for sent in _sentences(cand):
            if len(sent) < 35:
                continue
            if sent.lower() not in event_lower:
                return True
        # Fallback: simple containment check.
        if cand.lower() not in event_lower:
            return True
    return False


def _dedupe_description(description: str | None) -> str | None:
    """Remove obvious duplicate sentences/lines in a description.

    This is a deterministic safety net on top of LLM merge (prevents repeated facts like the same award twice).
    """
    if not description:
        return None
    raw = str(description).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return None
    raw = re.sub(r"\n{3,}", "\n\n", raw)

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return None

    seen_line_keys: set[str] = set()
    out_paras: list[str] = []

    def _dedupe_lines_keep_newlines(block: str) -> str:
        heading_re = re.compile(r"^\s*#{1,6}\s+\S")
        kept: list[str] = []
        for ln in block.splitlines():
            s = ln.strip()
            if not s:
                continue
            key = re.sub(r"\s+", " ", s).strip().lower()
            # Even short lines can be duplicated facts (e.g. awards). Dedupe more aggressively.
            # Headings are short by nature, but duplicate headings create empty sections and
            # ugly Telegraph pages, so we always dedupe them regardless of key length.
            if (heading_re.match(s) or len(key) >= 15) and key in seen_line_keys:
                continue
            seen_line_keys.add(key)
            kept.append(s)
        return "\n".join(kept).strip()

    def _dedupe_sentences_in_paragraph(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""
        parts = re.split(r"(?<=[.!?…])\s+", normalized)
        seen_sent: set[str] = set()
        kept_sent: list[str] = []
        for part in parts:
            sent = part.strip()
            if not sent:
                continue
            key = re.sub(r"\s+", " ", sent).strip().lower().rstrip(".!?…")
            # Dedupe repeated short sentences too (common LLM artifact and source-copy noise).
            if len(key) >= 18 and key in seen_sent:
                continue
            seen_sent.add(key)
            kept_sent.append(sent)

        # Drop sentences that are strict substrings of another sentence (helps with
        # truncated tails and "same idea twice" cases).
        norm_sents: list[tuple[str, str]] = []
        for sent in kept_sent:
            key = re.sub(r"\s+", " ", sent).strip().lower().rstrip(".!?…")
            norm_sents.append((sent, key))
        drop_idx: set[int] = set()
        for i, (_s_i, k_i) in enumerate(norm_sents):
            if i in drop_idx:
                continue
            if len(k_i) < 40:
                continue
            for j, (_s_j, k_j) in enumerate(norm_sents):
                if i == j or j in drop_idx:
                    continue
                if len(k_j) < len(k_i):
                    continue
                if len(k_j) - len(k_i) < 10:
                    continue
                if k_i and k_i in k_j:
                    drop_idx.add(i)
                    break

        kept2 = [s for idx, (s, _k) in enumerate(norm_sents) if idx not in drop_idx]
        return " ".join(kept2).strip()

    for para in paragraphs:
        if _looks_like_structured_block(para):
            cleaned = _dedupe_lines_keep_newlines(para)
            if cleaned:
                out_paras.append(cleaned)
            continue
        cleaned = _dedupe_sentences_in_paragraph(para)
        if cleaned:
            out_paras.append(cleaned)

    cleaned = "\n\n".join(out_paras).strip()
    return cleaned or None


def _normalize_candidate_sentence(chunk: str) -> str:
    sent = re.sub(r"\s+", " ", chunk).strip()
    if not sent:
        return ""
    # Replace Markdown links with link text to avoid noisy URL-heavy snippets.
    sent = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1", sent)
    sent = re.sub(r"\s+", " ", sent).strip(" *_`~|").strip()
    return sent


def _is_low_signal_sentence(sent: str) -> bool:
    if not sent:
        return True
    if len(sent) < 35:
        return True
    low = sent.lower()
    if "http://" in low or "https://" in low:
        return True
    # Skip schedule-like headers (common in multi-event Telegram posts):
    # "04.02 | ..." / "04/02 — ..." etc.
    if re.match(r"^\s*\d{1,2}[./]\d{1,2}\s*(?:\\||[-–—])\s*", sent):
        return True
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", sent)
    # Skip date/title-only fragments (common in noisy Telegram captions).
    if len(words) < 5:
        return True
    return False


_COVERAGE_CRITICAL_PATTERNS = (
    re.compile(r"\b\d{1,2}[:.]\d{2}\b", re.IGNORECASE),
    re.compile(r"\bнач(?:ало|н[её]т(?:с[яь])?)\b", re.IGNORECASE),
    re.compile(r"\b(?:основн\w*\s+сцен\w*|камерн\w*\s+сцен\w*|мал\w*\s+сцен\w*)\b", re.IGNORECASE),
    re.compile(r"\b(?:театральн\w+\s+хит|хит)\b", re.IGNORECASE),
)


def _is_coverage_critical_sentence(sent: str) -> bool:
    raw = (sent or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if "http://" in low or "https://" in low:
        return False
    return any(p.search(raw) for p in _COVERAGE_CRITICAL_PATTERNS)


def _enforce_merge_non_shrinking_description(
    *,
    before_description: str,
    merged_description: str,
    candidate: EventCandidate,
    has_new_text: bool,
) -> str:
    """Prevent LLM merge from collapsing a rich description into a short digest.

    If the merged description is substantially shorter than the previous one,
    prefer keeping the previous description and deterministically appending new
    factual sentences from the candidate.
    """
    before = (before_description or "").strip()
    merged = (merged_description or "").strip()
    if not merged:
        return before
    if not before:
        return merged
    # Only protect sufficiently rich descriptions; allow short texts to change freely.
    before_len = len(before)
    merged_len = len(merged)
    if before_len >= 500 and merged_len < int(before_len * 0.75):
        keep = before
        if has_new_text:
            before_norm = re.sub(r"\s+", " ", keep).strip().lower()
            new_sentences = _collect_new_candidate_sentences(candidate, before_norm=before_norm)
            if new_sentences:
                ranked = sorted(
                    range(len(new_sentences)),
                    key=lambda idx: (_sentence_quality_score(new_sentences[idx]), -idx),
                    reverse=True,
                )
                picked_idx = sorted(ranked[:2])
                picked = [new_sentences[idx] for idx in picked_idx]
                keep = (keep + "\n" + " ".join(picked)).strip()
        return keep

    # Also protect against the "too short compared to a rich new source" case:
    # when candidate text is long but the model returns a short digest.
    cand_text = _strip_private_use(candidate.source_text) or (candidate.source_text or "")
    cand_text = _strip_promo_lines(cand_text) or cand_text
    cand_text = _strip_giveaway_lines(cand_text) or cand_text
    cand_text = _strip_foreign_schedule_noise(
        cand_text,
        event_date=candidate.date,
        end_date=candidate.end_date,
        event_title=candidate.title,
    )
    cand_len = len((cand_text or "").strip())
    if cand_len >= 1200:
        min_expected = max(450, int(cand_len * 0.35))
        if merged_len < min_expected:
            # Prefer the richer previous description if it already has substance,
            # otherwise fall back to the candidate text (verbatim) to keep facts.
            if before_len >= min_expected:
                return before
            if cand_text.strip():
                return cand_text.strip()
    return merged


def _pick_richest_source_text_for_description(event: Event, candidate: EventCandidate) -> str:
    """Pick the richest available source text for building a full description.

    Priority is the longest cleaned text among event/source aggregates and the candidate.
    """
    texts: list[str] = []
    poster_texts: list[str] = []
    try:
        for p in (getattr(candidate, "posters", None) or [])[:5]:
            t = getattr(p, "ocr_text", None)
            if isinstance(t, str) and t.strip():
                poster_texts.append(t)
    except Exception:
        poster_texts = []
    for t in [
        getattr(event, "source_text", None),
        *(getattr(event, "source_texts", None) or []),
        getattr(candidate, "source_text", None),
        getattr(candidate, "raw_excerpt", None),
        *poster_texts,
    ]:
        if not isinstance(t, str):
            continue
        cleaned = _strip_private_use(t) or (t or "")
        cleaned = _strip_promo_lines(cleaned) or cleaned
        cleaned = _strip_giveaway_lines(cleaned) or cleaned
        cleaned = _strip_foreign_schedule_noise(
            cleaned,
            event_date=getattr(event, "date", None) or candidate.date,
            end_date=getattr(event, "end_date", None) or candidate.end_date,
            event_title=getattr(event, "title", None) or candidate.title,
        )
        cleaned = cleaned.strip()
        if cleaned:
            texts.append(cleaned)
    if not texts:
        return ""
    return max(texts, key=len)


def _build_fact_seed_text(
    event: Event,
    candidate: EventCandidate,
    *,
    poster_texts: Sequence[str] | None = None,
    max_chars: int = 16000,
) -> str:
    """Build a conservative "facts seed" text for deterministic post-processing.

    Smart Update merges are free to paraphrase and reorder, but they must not drop
    important facts (genre/style markers, unique details) that exist in the source
    materials. We use this combined seed only for *appending missing sentences*,
    not for generating new content.
    """

    def _clean(t: str | None) -> str:
        if not t or not isinstance(t, str):
            return ""
        cleaned = _strip_private_use(t) or (t or "")
        cleaned = _strip_promo_lines(cleaned) or cleaned
        cleaned = _strip_giveaway_lines(cleaned) or cleaned
        cleaned = _strip_foreign_schedule_noise(
            cleaned,
            event_date=getattr(event, "date", None) or candidate.date,
            end_date=getattr(event, "end_date", None) or candidate.end_date,
            event_title=getattr(event, "title", None) or candidate.title,
        ) or cleaned
        return cleaned.strip()

    chunks: list[str] = []
    for t in [
        getattr(event, "source_text", None),
        *(getattr(event, "source_texts", None) or []),
        getattr(event, "description", None),
        getattr(candidate, "source_text", None),
        getattr(candidate, "raw_excerpt", None),
        *(list(poster_texts or [])[:5]),
    ]:
        cleaned = _clean(t)
        if cleaned:
            chunks.append(cleaned)

    if not chunks:
        return ""

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for c in chunks:
        key = c.casefold()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    combined = "\n\n".join(uniq).strip()
    if not combined:
        return ""
    return _clip(combined, max_chars)


async def _rewrite_description_full_from_sources(event: Event, candidate: EventCandidate) -> str | None:
    """Second-pass rewrite used when merge returns an over-compressed digest.

    This uses the richest available source text (usually site import) and event metadata.
    """
    if SMART_UPDATE_LLM_DISABLED:
        return None

    base = _pick_richest_source_text_for_description(event, candidate)
    if len(base) < 120:
        return None

    payload = {
        "title": getattr(event, "title", None) or candidate.title,
        "date": getattr(event, "date", None) or candidate.date,
        "time": getattr(event, "time", None) or candidate.time,
        "end_date": getattr(event, "end_date", None) or candidate.end_date,
        "location_name": getattr(event, "location_name", None) or candidate.location_name,
        "location_address": getattr(event, "location_address", None) or candidate.location_address,
        "city": getattr(event, "city", None) or candidate.city,
        "ticket_link": getattr(event, "ticket_link", None) or candidate.ticket_link,
        "ticket_status": getattr(event, "ticket_status", None) or candidate.ticket_status,
        "is_free": bool(getattr(event, "is_free", False)),
        "event_type": getattr(event, "event_type", None) or candidate.event_type,
        "festival": getattr(event, "festival", None) or candidate.festival,
        "source_text": _clip(base, SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS),
    }

    prompt = (
        "Ты — культурный журналист. Сделай ПОЛНОЕ развернутое описание события на основе source_text и метаданных. "
        "Сохрани ВСЕ значимые факты, не превращай в короткий дайджест. "
        "Не добавляй выдуманных фактов. Не копируй фразы дословно, но и не сокращай содержание. "
        f"{SMART_UPDATE_YO_RULE} "
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE} "
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE} "
        f"{_description_emoji_prompt_rule()}"
        "Без хэштегов. Убери промо/акции и механику розыгрыша (если не часть сути). "
        "Не добавляй секции/заголовки про факты (например «Факты…», «Факты о событии», `Facts/Added Facts`): "
        "факты формируются отдельно. "
        "Важно: НЕ повторяй в описании логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены) — "
        "она показывается отдельным инфоблоком сверху.\n\n"
        "Убери промо чужих/вспомогательных каналов с анонсами и призывы подписаться "
        "(например «Информация о событиях ... доступна в Telegram-канале ...»): это не факт про само событие.\n\n"
        "Запрещено придумывать утверждения вроде 'премьера', 'впервые', 'аншлаг' и т.п., "
        "если это явно не сказано в source_text.\n"
        "Избегай нейросетевых клише и прогнозов (например 'обещает стать заметным событием', 'не оставит равнодушным').\n\n"
        "Можно использовать минимальную разметку для читабельности: "
        "заголовки `###`, цитаты блоком `> ...`, редкое выделение `**...**`. "
        "НЕ используй Markdown-ссылки вида `[текст](url)` и не вставляй таблицы. "
        "Не включай малозначимые и повторяющиеся строки (например `DD.MM | Название`, повтор заголовка, повтор даты/времени/площадки отдельной строкой). "
        "Не включай в описание нерелевантные новости о площадке/организации, которые не относятся к самому событию "
        "(например отчёты о работе филиала, планы на год, анонс посторонних интервью). "
        "Не дублируй в основном тексте строки-анкеры (`Дата:`, `Время:`, `Локация:`, `Билеты:`) и их явные перефразы: "
        "эти данные уже показываются отдельным блоком. "
        "Если в исходном тексте есть обрыв фразы/текста (в т.ч. обрезано на середине слова), не вставляй это дословно: либо перефразируй, либо опусти. "
        "Структуру делай абзацами: разделяй абзацы пустой строкой. "
        "В каждом абзаце держи 1-2 предложения (максимум 3 только если иначе теряется смысл).\n\n"
        "Самопроверка перед ответом:\n"
        "- В тексте НЕТ ссылок/телефонов/точных адресов/цен/времени/дат (они уже в инфоблоке).\n"
        "- НЕТ обрывов фраз после правок.\n"
        "- НЕТ странных/непонятных слов и опечаток.\n\n"
        f"Данные:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    # Allow a bit more than the default rewrite budget for the "fix too short merge" case.
    max_tokens = min(1600, max(300, SMART_UPDATE_REWRITE_MAX_TOKENS + 300))
    text = await _ask_gemma_text(
        prompt,
        max_tokens=max_tokens,
        label="rewrite_full",
        temperature=0.0,
    )
    if not text:
        return None
    cleaned = _strip_private_use(text) or (text or "")
    cleaned = _strip_foreign_schedule_noise(
        cleaned,
        event_date=getattr(event, "date", None) or candidate.date,
        end_date=getattr(event, "end_date", None) or candidate.end_date,
        event_title=getattr(event, "title", None) or candidate.title,
    )
    cleaned = _normalize_plaintext_paragraphs(cleaned)
    if not cleaned:
        return None
    cleaned = _fix_broken_initial_paragraph_splits(cleaned) or cleaned
    cleaned = (
        _sanitize_description_output(
            cleaned,
            source_text=base,
        )
        or cleaned
    )
    if _description_needs_channel_promo_strip(cleaned):
        cleaned = _strip_channel_promo_from_description(cleaned) or cleaned
    cleaned = _append_missing_fact_sentences(base=base, rewritten=cleaned, max_sentences=2)
    if _description_needs_infoblock_logistics_strip(cleaned, candidate=candidate):
        try:
            edited = await _llm_remove_infoblock_logistics(
                description=cleaned,
                candidate=candidate,
                label="rewrite_full_remove_logistics",
            )
        except Exception:  # pragma: no cover - network failures
            edited = None
        if edited:
            edited = _normalize_plaintext_paragraphs(edited) or edited
            cleaned = edited
    return _clip(cleaned, SMART_UPDATE_DESCRIPTION_MAX_CHARS)


def _min_expected_description_len_from_sources(event: Event, candidate: EventCandidate) -> int:
    richest = _pick_richest_source_text_for_description(event, candidate)
    base_len = len(richest)
    if base_len < 700:
        return 0
    return max(450, int(base_len * 0.55))


def _allowed_schedule_ddmm(event_date: str | None, end_date: str | None) -> set[str]:
    """Return allowed DD.MM anchors for the event date range (best-effort)."""
    if not event_date:
        return set()
    try:
        start = date.fromisoformat(event_date.split("..", 1)[0].strip())
    except Exception:
        return set()
    end = None
    if end_date:
        try:
            end = date.fromisoformat(end_date.strip())
        except Exception:
            end = None
    if not end and ".." in event_date:
        try:
            end = date.fromisoformat(event_date.split("..", 1)[1].strip())
        except Exception:
            end = None
    if not end:
        end = start
    # Avoid exploding on very long ranges.
    if (end - start).days > 14:
        end = start
    out: set[str] = set()
    cur = start
    while cur <= end:
        out.add(cur.strftime("%d.%m"))
        cur += timedelta(days=1)
    return out


_SCHEDULE_LINE_RE = re.compile(
    r"^\s*(?P<dd>\d{1,2})[./](?P<mm>\d{1,2})\s*(?:\\||[-–—])\s*(?P<title>.+?)\s*$"
)


def _strip_foreign_schedule_headings(
    text: str | None, *, event_date: str | None, end_date: str | None
) -> str:
    """Remove schedule-like headings for dates outside the event date range.

    This protects against Telegram "schedule" posts leaking unrelated items into
    a single-event description (e.g. "04.02 | ..." inside the 07.02 event).
    """
    if not text:
        return ""
    allowed = _allowed_schedule_ddmm(event_date, end_date)
    if not allowed:
        return (text or "").strip()
    kept: list[str] = []
    changed = False
    for line in str(text).replace("\r", "\n").split("\n"):
        m = _SCHEDULE_LINE_RE.match(line)
        if not m:
            kept.append(line)
            continue
        dd = int(m.group("dd"))
        mm = int(m.group("mm"))
        ddmm = f"{dd:02d}.{mm:02d}"
        if ddmm in allowed:
            kept.append(line)
            continue
        changed = True
        # drop the line
    out = "\n".join(kept).strip()
    if not changed:
        return (text or "").strip()
    return _dedupe_description(out) or out


def _strip_schedule_headings_all(text: str | None) -> str:
    """Remove schedule-like heading lines regardless of date range.

    Example:
      "12.02 | Фигаро"

    Even when the date/title matches the current event, this line is redundant on
    a single event page once date/time/location are present elsewhere.
    """
    if not text:
        return ""
    kept: list[str] = []
    changed = False
    for line in str(text).replace("\r", "\n").split("\n"):
        if _SCHEDULE_LINE_RE.match(line.strip()):
            changed = True
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    if not changed:
        return (text or "").strip()
    return _dedupe_description(out) or out


def _looks_like_schedule_digest(text: str | None, *, event_date: str | None, end_date: str | None) -> bool:
    """Heuristic: detect multi-event digest posts (not a single event).

    Used to avoid catastrophic merges/creations from VK/TG posts like "куда сходить" with many dated items.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    allowed = _allowed_schedule_ddmm(event_date, end_date)
    ddmm: set[str] = set()
    for dd_s, mm_s in re.findall(r"\b(\d{1,2})[./](\d{1,2})\b", raw):
        try:
            dd = int(dd_s)
            mm = int(mm_s)
        except Exception:
            continue
        if not (1 <= dd <= 31 and 1 <= mm <= 12):
            continue
        ddmm.add(f"{dd:02d}.{mm:02d}")
    foreign = [x for x in ddmm if x not in allowed]
    # If the source mentions many dates outside the target range, it's likely a schedule digest.
    if len(foreign) >= 4:
        return True
    # Extra signal: unusually long bullet-heavy text.
    lines = raw.splitlines()
    if len(lines) >= 50:
        bullets = 0
        for line in lines:
            s = line.strip()
            if s.startswith(("•", "-", "—", "*")):
                bullets += 1
        if bullets >= 10:
            return True
    return False


def _normalize_title_for_match(title: str | None) -> str:
    if not title:
        return ""
    raw = _strip_private_use(title) or (title or "")
    raw = re.sub(r"[\"«»]", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip().casefold().replace("ё", "е")
    return raw


_TITLE_MATCH_STOPWORDS = {
    "выставка",
    "концерт",
    "спектакль",
    "событие",
    "мероприятие",
    "открытие",
    "премьера",
    "встреча",
    "вечер",
    "калининград",
}


def _normalize_text_for_grounding(text: str | None) -> str:
    if not text:
        return ""
    cleaned = _strip_private_use(text) or str(text)
    cleaned = _ZERO_WIDTH_RE.sub("", cleaned)
    cleaned = cleaned.casefold().replace("ё", "е")
    cleaned = cleaned.replace("…", " ")
    cleaned = re.sub(r"[\"«»]", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]+", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _meaningful_title_tokens(title: str | None) -> set[str]:
    norm = _normalize_text_for_grounding(title)
    if not norm:
        return set()
    toks: set[str] = set()
    for tok in norm.split():
        if len(tok) < 3:
            continue
        if tok in _TITLE_MATCH_STOPWORDS:
            continue
        if tok.isdigit():
            continue
        toks.add(tok)
    return toks


def _candidate_title_grounding_corpus_norm(candidate: "EventCandidate") -> str:
    parts: list[str] = []
    if (candidate.source_text or "").strip():
        parts.append(_clip(candidate.source_text, 9000))
    if (candidate.raw_excerpt or "").strip():
        parts.append(_clip(candidate.raw_excerpt, 2200))
    for poster in list(getattr(candidate, "posters", None) or [])[:4]:
        if (getattr(poster, "ocr_title", None) or "").strip():
            parts.append(_clip(str(poster.ocr_title), 320))
        if (getattr(poster, "ocr_text", None) or "").strip():
            parts.append(_clip(str(poster.ocr_text), 1200))
    return _normalize_text_for_grounding("\n".join(parts))


def _token_is_grounded(token: str, source_tokens: set[str]) -> bool:
    if not token:
        return False
    if token in source_tokens:
        return True
    # Best-effort stemming: allow a 1-char suffix difference (e.g. "масленица" vs "масленицу").
    stems = [token]
    if len(token) >= 6:
        stems.append(token[:-1])
    for stem in stems:
        if stem in source_tokens:
            return True
        if len(stem) < 5:
            continue
        for src in source_tokens:
            if src.startswith(stem) or stem.startswith(src):
                return True
    return False


def _is_title_grounded_in_candidate_sources(
    proposed_title: str | None,
    candidate: "EventCandidate",
) -> bool:
    title_norm = _normalize_text_for_grounding(proposed_title)
    if not title_norm:
        return False
    corpus_norm = _candidate_title_grounding_corpus_norm(candidate)
    if not corpus_norm:
        return False
    if len(title_norm) >= 6 and title_norm in corpus_norm:
        return True
    source_tokens = set(corpus_norm.split())
    tokens = _meaningful_title_tokens(proposed_title)
    if not tokens:
        # Titles like "концерт" still must exist in the source corpus.
        return title_norm in corpus_norm
    for token in tokens:
        if _token_is_grounded(token, source_tokens):
            return True
    return False


def _is_generic_title_event_type_venue(
    title: str | None,
    *,
    event_type: str | None,
    location_name: str | None,
    city: str | None,
) -> bool:
    """Return True for fallback titles like "Концерт — <venue>"."""

    title_norm = _normalize_text_for_grounding(title)
    if not title_norm:
        return False
    et_norm = _normalize_text_for_grounding(event_type)
    if not et_norm:
        return False
    loc_raw = str(location_name or "").strip()
    if not loc_raw:
        return False
    venue_short = loc_raw.split(",", 1)[0].strip()
    venue_norm = _normalize_text_for_grounding(venue_short)
    if not venue_norm:
        return False

    title_toks = {t for t in title_norm.split() if len(t) >= 3}
    if not title_toks:
        return False
    et_toks = {t for t in et_norm.split() if len(t) >= 3}
    venue_toks = {t for t in venue_norm.split() if len(t) >= 3}
    if not (title_toks & et_toks):
        return False
    if not (title_toks & venue_toks):
        return False

    allowed = set(et_toks | venue_toks)
    if city:
        allowed |= {t for t in _normalize_text_for_grounding(city).split() if len(t) >= 3}
    return title_toks.issubset(allowed)


def _is_candidate_title_weak_for_llm_override(
    title: str | None,
    *,
    candidate: "EventCandidate",
    normalized_event_type: str | None,
) -> bool:
    if _is_generic_title_event_type_venue(
        title,
        event_type=normalized_event_type or candidate.event_type,
        location_name=candidate.location_name,
        city=candidate.city,
    ):
        return True
    tokens = _meaningful_title_tokens(title)
    if not tokens:
        return True
    corpus_norm = _candidate_title_grounding_corpus_norm(candidate)
    if not corpus_norm:
        return False
    source_tokens = set(corpus_norm.split())
    for token in tokens:
        if _token_is_grounded(token, source_tokens):
            return False
    return True


def _title_has_meaningful_tokens(title: str | None) -> bool:
    return bool(_meaningful_title_tokens(title))


def _is_merge_title_update_allowed(
    *,
    proposed_title: str | None,
    candidate_title: str | None,
    existing_title: str | None,
    is_canonical_site: bool,
) -> bool:
    """Guard LLM title updates against cross-event contamination.

    For non-canonical sources (telegram/vk/manual imports), accept a merged title only
    when it is semantically related to candidate title and does not conflict with an
    already meaningful existing title.

    For canonical parser sources we allow title correction by candidate title relation,
    even if existing title is already polluted by a previous bad merge.
    """
    proposed = (proposed_title or "").strip()
    if not proposed:
        return False
    if not _titles_look_related(proposed, candidate_title):
        return False
    if is_canonical_site:
        return True
    if _title_has_meaningful_tokens(existing_title) and not _titles_look_related(
        proposed, existing_title
    ):
        return False
    return True


def _titles_look_related(a: str | None, b: str | None) -> bool:
    na = _normalize_title_for_match(a)
    nb = _normalize_title_for_match(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 8 and na in nb:
        return True
    if len(nb) >= 8 and nb in na:
        return True
    toks_a = {
        t
        for t in re.findall(r"[a-zа-яё0-9]+", na)
        if len(t) >= 3 and t not in _TITLE_MATCH_STOPWORDS
    }
    toks_b = {
        t
        for t in re.findall(r"[a-zа-яё0-9]+", nb)
        if len(t) >= 3 and t not in _TITLE_MATCH_STOPWORDS
    }
    if not toks_a or not toks_b:
        return False
    overlap = toks_a & toks_b
    if not overlap:
        return False
    denom = max(1, min(len(toks_a), len(toks_b)))
    coverage = len(overlap) / denom
    return coverage >= 0.6 or (len(overlap) >= 2 and coverage >= 0.45)


def _normalize_time_for_match(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    raw = raw.replace(".", ":")
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if not m:
        return ""
    hh = int(m.group(1))
    mm = int(m.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return ""
    # "00:00" is often a placeholder from legacy imports.
    if hh == 0 and mm == 0:
        return ""
    return f"{hh:02d}:{mm:02d}"


def _candidate_anchor_time(candidate: "EventCandidate", *, is_canonical_site: bool) -> str:
    """Return time usable as a matching anchor, or empty when time is weak.

    Low-priority default times (e.g. VK source default_time) must not act as anchors:
    they should be overridden when explicit time comes from other sources.
    """
    t = _normalize_time_for_match(candidate.time)
    if not t:
        return ""
    if is_canonical_site:
        return t
    if bool(getattr(candidate, "time_is_default", False)):
        return ""
    return t


def _event_anchor_time(event: "Event") -> str:
    """Return time usable as a matching anchor, or empty when time is weak."""
    if bool(getattr(event, "time_is_default", False)):
        return ""
    return _normalize_time_for_match(getattr(event, "time", None))


def _has_explicit_time_conflict(candidate_time: str | None, event_time: str | None) -> bool:
    ct = _normalize_time_for_match(candidate_time)
    et = _normalize_time_for_match(event_time)
    return bool(ct and et and ct != et)


def _anchor_signature_for_duplicate_event(ev: Event) -> tuple[str, str, str, str]:
    """Return a compact signature to detect truly duplicated rows.

    Used by source_url-based idempotency matchers: when the same source is reprocessed,
    we want to converge on *one* event, but we must not accidentally merge distinct events
    from schedule posts. If multiple anchored candidates exist and they share the same
    signature, we treat them as duplicates and pick the best one.
    """

    return (
        str(getattr(ev, "date", "") or "").strip(),
        _event_anchor_time(ev) or "",
        _normalize_location(getattr(ev, "location_name", None)),
        _normalize_title_for_match(getattr(ev, "title", None)),
    )


def _pick_best_duplicate_event(candidates: Sequence[Event]) -> Event | None:
    if not candidates:
        return None

    def _score(ev: Event) -> tuple[int, int]:
        score = 0
        if (getattr(ev, "telegraph_url", None) or "").strip():
            score += 4
        if (getattr(ev, "description", None) or "").strip():
            score += 2
        if (getattr(ev, "search_digest", None) or "").strip():
            score += 1
        topics = getattr(ev, "topics", None) or []
        if isinstance(topics, list) and topics:
            score += 1
        if (getattr(ev, "ticket_link", None) or "").strip():
            score += 1
        if (getattr(ev, "location_address", None) or "").strip():
            score += 1
        # Prefer older IDs on ties to reduce churn in external references.
        eid = int(getattr(ev, "id", 0) or 0)
        return score, -eid

    return max(candidates, key=_score)


async def _match_existing_event_by_source_anchor(
    db: Database,
    candidate: EventCandidate,
) -> Event | None:
    """Best-effort idempotency for duplicated candidates from the same source post.

    Telegram Monitoring can emit two nearly-identical event cards for one Telegram post
    (e.g. message text + linked post enrichment). Smart Update must prefer merging into
    the already-created event instead of creating a duplicate.

    Safety: for schedule posts (multiple real events per one message) we only force-match
    when the anchor resolves to a single unambiguous event.
    """
    source_url = str(candidate.source_url or "").strip()
    if not source_url or not candidate.source_message_id:
        return None
    if str(candidate.source_type or "").strip().lower().startswith("parser:"):
        return None

    try:
        message_id = int(candidate.source_message_id)
    except Exception:
        return None

    async with db.get_session() as session:
        stmt = select(Event).where(Event.source_message_id == message_id)
        if _is_vk_wall_url(source_url):
            stmt = stmt.where(
                or_(
                    Event.source_vk_post_url == source_url,
                    Event.source_post_url == source_url,
                )
            )
        else:
            stmt = stmt.where(Event.source_post_url == source_url)
        stmt = _apply_soft_city_filter(stmt, candidate.city)
        res = await session.execute(stmt)
        anchored = list(res.scalars().all())

    if not anchored:
        return None

    date_raw = str(candidate.date or "").strip()
    title_raw = str(candidate.title or "").strip()
    is_canonical_site = str(candidate.source_type or "").startswith("parser:")
    cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
    allow_parallel = _allow_parallel_events(candidate.location_name)
    anchor_filtered: list[Event] = []
    filtered: list[Event] = []
    for ev in anchored:
        if date_raw and str(getattr(ev, "date", "") or "").strip() != date_raw:
            continue
        if _has_explicit_time_conflict(cand_time_anchor, _event_anchor_time(ev)):
            continue
        if candidate.location_name and getattr(ev, "location_name", None):
            if not _event_candidate_location_matches(ev, candidate):
                continue
        anchor_filtered.append(ev)
        ev_title = str(getattr(ev, "title", "") or "").strip()
        if title_raw and ev_title and _title_has_meaningful_tokens(title_raw) and _title_has_meaningful_tokens(ev_title):
            if not _titles_look_related(title_raw, ev_title):
                continue
        filtered.append(ev)

    if not filtered and len(anchor_filtered) == 1 and cand_time_anchor and (not allow_parallel):
        # Same source+message+anchors but title extraction may differ between retries (e.g. one post -> multiple
        # program items). If we have an explicit time anchor and the venue does not allow parallel events,
        # converge on the single anchored event to avoid duplicates.
        return anchor_filtered[0]
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]

    # Only force-match when it's clearly the same event duplicated.
    sigs = {_anchor_signature_for_duplicate_event(ev) for ev in filtered}
    if len(sigs) == 1:
        return _pick_best_duplicate_event(filtered)
    return None


async def _match_existing_event_by_event_source_url(
    db: Database,
    candidate: EventCandidate,
) -> Event | None:
    """Best-effort idempotency when check_source_url=False.

    Some flows intentionally re-run Smart Update for the same source_url (e.g. monitoring retries,
    deferred processing). We still want to converge on the same event instead of creating duplicates.

    Safety: if one source_url maps to multiple real events (schedule posts), only force-match when
    the match is unambiguous after basic anchor checks.
    """
    source_url = str(candidate.source_url or "").strip()
    if not source_url:
        return None
    if str(candidate.source_type or "").strip().lower().startswith("parser:"):
        return None

    async with db.get_session() as session:
        stmt = (
            select(Event)
            .join(EventSource, EventSource.event_id == Event.id)
            .where(EventSource.source_url == source_url)
        )
        if candidate.source_type:
            stmt = stmt.where(EventSource.source_type == candidate.source_type)
        stmt = _apply_soft_city_filter(stmt, candidate.city)
        res = await session.execute(stmt)
        anchored = list(res.scalars().all())

    if not anchored:
        return None

    date_raw = str(candidate.date or "").strip()
    title_raw = str(candidate.title or "").strip()
    is_canonical_site = str(candidate.source_type or "").startswith("parser:")
    cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
    allow_parallel = _allow_parallel_events(candidate.location_name)
    anchor_filtered: list[Event] = []
    filtered: list[Event] = []
    for ev in anchored:
        if date_raw and str(getattr(ev, "date", "") or "").strip() != date_raw:
            continue
        if _has_explicit_time_conflict(cand_time_anchor, _event_anchor_time(ev)):
            continue
        ev_title = str(getattr(ev, "title", "") or "").strip()
        if candidate.location_name and getattr(ev, "location_name", None):
            if not _event_candidate_location_matches(ev, candidate):
                continue
        anchor_filtered.append(ev)
        if title_raw and ev_title and _title_has_meaningful_tokens(title_raw) and _title_has_meaningful_tokens(ev_title):
            if not _titles_look_related(title_raw, ev_title):
                continue
        filtered.append(ev)

    if not filtered and len(anchor_filtered) == 1 and cand_time_anchor and (not allow_parallel):
        # Same source_url+anchors but title extraction may differ between retries (e.g. one post -> multiple
        # program items). If we have an explicit time anchor and the venue does not allow parallel events,
        # converge on the single anchored event to avoid duplicates.
        return anchor_filtered[0]
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]

    # Only force-match when it's clearly the same event duplicated.
    sigs = {_anchor_signature_for_duplicate_event(ev) for ev in filtered}
    if len(sigs) == 1:
        return _pick_best_duplicate_event(filtered)
    return None


def _single_candidate_auto_match_ok(
    candidate: EventCandidate,
    event_db: Event,
    *,
    is_canonical_site: bool,
) -> bool:
    # Guard against catastrophic merges when shortlist shrinks to 1 by broad anchors
    # (e.g. generic city location + long-running exhibition date range overlap).
    if is_canonical_site:
        cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=True)
        event_time_anchor = _event_anchor_time(event_db)
        # Canonical parser sources are allowed to repair a polluted title when
        # anchors are strongly aligned.
        if candidate.date and getattr(event_db, "date", None) and candidate.date != event_db.date:
            return False
        if candidate.location_name and getattr(event_db, "location_name", None):
            if not _event_candidate_location_matches(event_db, candidate):
                return False
        if _has_explicit_time_conflict(cand_time_anchor, event_time_anchor):
            return False
        ct = cand_time_anchor
        et = event_time_anchor
        if ct and et and ct == et:
            return True
        if _titles_look_related(candidate.title, getattr(event_db, "title", None)):
            return True
        # Allow parser correction when candidate has explicit time but existing event
        # has empty/placeholder/weak time.
        if ct and not et:
            return True
        return False

    if not _titles_look_related(candidate.title, getattr(event_db, "title", None)):
        # Do NOT auto-merge unrelated titles, including long-running exhibitions/fairs.
        #
        # Rationale: venues can host multiple exhibitions simultaneously, so date-range overlap
        # + location is not a safe enough signal for an automatic single-candidate match.
        # When in doubt, fall back to LLM matching / create (or create when LLM is disabled).
        return False
    cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=False)
    event_time_anchor = _event_anchor_time(event_db)
    if _has_explicit_time_conflict(cand_time_anchor, event_time_anchor):
        return False
    return True


def _deterministic_exact_title_match(
    candidate: EventCandidate,
    events: Sequence[Event],
    *,
    is_canonical_site: bool,
) -> Event | None:
    """Try to match by strict anchors without LLM.

    Goal: prevent obvious duplicates when the same event comes from multiple sources
    (TG/VK reposts), especially when LLM matching is conservative.
    """

    if not events:
        return None
    cand_date = str(candidate.date or "").strip()
    if not cand_date:
        return None
    if not _title_has_meaningful_tokens(candidate.title):
        return None

    cand_title_norm = _normalize_title_for_match(candidate.title)
    if not cand_title_norm:
        return None

    cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
    cand_loc = str(candidate.location_name or "").strip() or None

    matches: list[Event] = []
    for ev in events:
        if not getattr(ev, "id", None):
            continue
        if str(getattr(ev, "date", "") or "").strip() != cand_date:
            continue
        if cand_loc and getattr(ev, "location_name", None):
            if not _event_candidate_location_matches(ev, candidate):
                continue
        if _normalize_title_for_match(getattr(ev, "title", None)) != cand_title_norm:
            continue
        if _has_explicit_time_conflict(cand_time_anchor, _event_anchor_time(ev)):
            continue
        # Double-check semantic relation (guards against accidental collisions on short titles).
        if not _titles_look_related(candidate.title, getattr(ev, "title", None)):
            continue
        matches.append(ev)

    if len(matches) == 1:
        return matches[0]

    # If there are multiple candidates but they all share the same anchor signature,
    # it's a duplicate row situation: pick the best one.
    if matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(matches)
    return None


def _deterministic_related_title_anchor_match(
    candidate: EventCandidate,
    events: Sequence[Event],
    *,
    is_canonical_site: bool,
) -> Event | None:
    """Try to match by strong anchors plus semantically related titles.

    This is intentionally stricter than `_titles_look_related()` alone:
    we require a unique candidate on the same date, explicit start time and
    location. The helper is primarily meant to keep parser/VK/TG re-imports
    from creating duplicates when titles differ only by a descriptive suffix
    like "Гегель" vs "Гегель: философия истории".
    """

    if not events:
        return None
    cand_date = str(candidate.date or "").strip()
    cand_loc = str(candidate.location_name or "").strip()
    cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
    if not cand_date or not cand_loc or not cand_time_anchor:
        return None
    if not _title_has_meaningful_tokens(candidate.title):
        return None

    matches: list[Event] = []
    for ev in events:
        if not getattr(ev, "id", None):
            continue
        if str(getattr(ev, "date", "") or "").strip() != cand_date:
            continue
        if not _event_candidate_location_matches(ev, candidate):
            continue
        if _has_explicit_time_conflict(cand_time_anchor, _event_anchor_time(ev)):
            continue
        if not _titles_look_related(candidate.title, getattr(ev, "title", None)):
            continue
        matches.append(ev)

    if len(matches) == 1:
        return matches[0]

    if matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(matches)
    return None


def _event_has_source_url_hint(event: Event, source_url: str | None) -> bool:
    source_norm = _normalize_url(source_url)
    if not source_norm:
        return False
    return source_norm in {
        _normalize_url(getattr(event, "source_post_url", None)),
        _normalize_url(getattr(event, "source_vk_post_url", None)),
    }


def _source_texts_look_nearly_identical(a: str | None, b: str | None) -> bool:
    left = _normalize_for_similarity(a, drop_structured=False)
    right = _normalize_for_similarity(b, drop_structured=False)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 120 and left in right:
        return True
    if len(right) >= 120 and right in left:
        return True
    try:
        from difflib import SequenceMatcher
    except Exception:  # pragma: no cover
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.82


def _time_to_minutes_for_match(value: str | None) -> int | None:
    norm = _normalize_time_for_match(value)
    if not norm:
        return None
    try:
        hh, mm = norm.split(":", 1)
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def _source_text_mentions_both_times(text: str | None, *times: str | None) -> bool:
    raw = str(text or "").replace(".", ":").casefold()
    wanted = [t for t in {_normalize_time_for_match(v) for v in times} if t]
    if len(wanted) < 2:
        return False
    return all(t.casefold() in raw for t in wanted)


def _deterministic_same_post_longrun_exact_title_match(
    candidate: EventCandidate,
    events: Sequence[Event],
) -> Event | None:
    cand_title = _normalize_title_for_match(candidate.title)
    cand_start, cand_end = _candidate_date_range(candidate)
    if not cand_title or not cand_start or not cand_end or cand_start == cand_end:
        return None
    matches: list[Event] = []
    for ev in events:
        if not getattr(ev, "id", None):
            continue
        if not _event_has_source_url_hint(ev, candidate.source_url):
            continue
        if _normalize_title_for_match(getattr(ev, "title", None)) != cand_title:
            continue
        if candidate.location_name and getattr(ev, "location_name", None):
            if not _event_candidate_location_matches(ev, candidate):
                continue
        ev_start, ev_end = _event_date_range(ev)
        if not _ranges_overlap(cand_start, cand_end, ev_start, ev_end):
            continue
        ev_end_iso = str(getattr(ev, "end_date", "") or "").strip() or None
        cand_end_iso = str(candidate.end_date or "").strip() or None
        if cand_end_iso and ev_end_iso and cand_end_iso != ev_end_iso:
            continue
        matches.append(ev)
    if len(matches) == 1:
        return matches[0]
    if matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(matches)
    return None


def _deterministic_ticket_source_anchor_match(
    candidate: EventCandidate,
    events: Sequence[Event],
) -> tuple[Event | None, str]:
    cand_ticket = _normalize_url(candidate.ticket_link)
    cand_date = str(candidate.date or "").strip()
    cand_loc = str(candidate.location_name or "").strip()
    cand_time = _candidate_anchor_time(candidate, is_canonical_site=False)
    if not (cand_ticket and cand_date and cand_loc and cand_time):
        return None, ""

    slot_matches: list[Event] = []
    bridge_matches: list[Event] = []
    cand_minutes = _time_to_minutes_for_match(cand_time)
    source_text = candidate.source_text or candidate.raw_excerpt
    for ev in events:
        if not getattr(ev, "id", None):
            continue
        if str(getattr(ev, "date", "") or "").strip() != cand_date:
            continue
        if not _event_candidate_location_matches(ev, candidate):
            continue
        if _normalize_url(getattr(ev, "ticket_link", None)) != cand_ticket:
            continue
        if not _source_texts_look_nearly_identical(source_text, getattr(ev, "source_text", None)):
            continue
        ev_time = _event_anchor_time(ev)
        if ev_time == cand_time:
            slot_matches.append(ev)
            continue
        ev_minutes = _time_to_minutes_for_match(ev_time)
        if cand_minutes is None or ev_minutes is None:
            continue
        if abs(cand_minutes - ev_minutes) > 90:
            continue
        if not _source_text_mentions_both_times(source_text, cand_time, ev_time):
            continue
        if not re.search(r"(?iu)\b(сбор\s+гостей|doors|начал[оа]|start)\b", str(source_text or "")):
            continue
        bridge_matches.append(ev)

    if len(slot_matches) == 1:
        return slot_matches[0], "deterministic_specific_ticket_same_slot"
    if slot_matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in slot_matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(slot_matches), "deterministic_specific_ticket_same_slot"
    if len(bridge_matches) == 1:
        return bridge_matches[0], "deterministic_doors_start_ticket_bridge"
    if bridge_matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in bridge_matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(bridge_matches), "deterministic_doors_start_ticket_bridge"
    return None, ""


def _deterministic_copy_post_ticket_same_day_match(
    candidate: EventCandidate,
    events: Sequence[Event],
) -> Event | None:
    cand_ticket = _normalize_url(candidate.ticket_link)
    cand_date = str(candidate.date or "").strip()
    source_text = candidate.source_text or candidate.raw_excerpt
    if not (cand_ticket and cand_date and source_text):
        return None
    matches: list[Event] = []
    for ev in events:
        if not getattr(ev, "id", None):
            continue
        if str(getattr(ev, "date", "") or "").strip() != cand_date:
            continue
        if _normalize_url(getattr(ev, "ticket_link", None)) != cand_ticket:
            continue
        if not _source_texts_look_nearly_identical(source_text, getattr(ev, "source_text", None)):
            continue
        if not _titles_look_related(candidate.title, getattr(ev, "title", None)):
            continue
        matches.append(ev)
    if len(matches) == 1:
        return matches[0]
    if matches:
        sigs = {_anchor_signature_for_duplicate_event(ev) for ev in matches}
        if len(sigs) == 1:
            return _pick_best_duplicate_event(matches)
    return None


async def _match_existing_event_by_city_noise_rescue(
    db: Database,
    candidate: EventCandidate,
    *,
    is_canonical_site: bool,
) -> tuple[Event | None, str]:
    cand_start, cand_end = _candidate_date_range(candidate)
    if not cand_start or not cand_end:
        return None, ""

    async with db.get_session() as session:
        stmt = select(Event).where(
            and_(
                Event.date <= cand_end.isoformat(),
                or_(
                    and_(
                        Event.end_date.is_(None),
                        Event.date >= cand_start.isoformat(),
                    ),
                    Event.end_date >= cand_start.isoformat(),
                ),
            )
        )
        res = await session.execute(stmt)
        pool = list(res.scalars().all())

    if not pool:
        return None, ""

    exact = _deterministic_exact_title_match(
        candidate,
        pool,
        is_canonical_site=is_canonical_site,
    )
    if exact is not None:
        return exact, "city_noise_exact_title_shortlist"

    copy_post = _deterministic_copy_post_ticket_same_day_match(candidate, pool)
    if copy_post is not None:
        return copy_post, "city_noise_copy_post_shortlist"

    return None, ""


def _strip_foreign_schedule_sentences(text: str | None, *, event_title: str | None) -> str:
    """Remove sentences that look like a foreign schedule/list of other events.

    Example of unwanted leakage (from Telegram schedule posts):
    '... также пройдут спектакли \"Нюрнберг\", \"Мысли...\", ...'
    """
    if not text:
        return ""
    title_norm = _normalize_title_for_match(event_title)
    raw = str(text).strip()
    if not raw:
        return ""

    sentence_re = re.compile(r"(?<=[.!?…])\s+")
    quote_re = re.compile(r"[\"«](.+?)[\"»]")
    keywords_re = re.compile(r"\b(также|в\s+рамках|в\s+афише|указан\w*|пройдут)\b", re.IGNORECASE)
    eventish_re = re.compile(r"\b(спектакл\w*|постановк\w*|концерт\w*|мероприят\w*)\b", re.IGNORECASE)

    parts = sentence_re.split(raw)
    kept: list[str] = []
    changed = False
    for sent in parts:
        s = sent.strip()
        if not s:
            continue
        if not keywords_re.search(s) or not eventish_re.search(s):
            kept.append(s)
            continue
        quoted = [q.strip() for q in quote_re.findall(s) if q and q.strip()]
        if len(quoted) < 2:
            kept.append(s)
            continue
        # If the sentence enumerates multiple quoted titles and none of them matches
        # the current event title, it's likely a leaked schedule list.
        if title_norm:
            quoted_norm = [_normalize_title_for_match(q) for q in quoted]
            if any(title_norm and title_norm in qn for qn in quoted_norm):
                kept.append(s)
                continue
        changed = True
        # drop sentence
    out = " ".join(kept).strip()
    if not changed:
        return raw
    return _dedupe_description(out) or out


def _strip_foreign_schedule_noise(
    text: str | None,
    *,
    event_date: str | None,
    end_date: str | None,
    event_title: str | None,
) -> str:
    # Deterministic schedule stripping is not allowed (LLM handles this).
    return (text or "").strip()


def _description_has_foreign_schedule_headings(
    text: str | None, *, event_date: str | None, end_date: str | None
) -> bool:
    if not text:
        return False
    allowed = _allowed_schedule_ddmm(event_date, end_date)
    if not allowed:
        return False
    for line in str(text).replace("\r", "\n").split("\n"):
        m = _SCHEDULE_LINE_RE.match(line)
        if not m:
            continue
        try:
            dd = int(m.group("dd"))
            mm = int(m.group("mm"))
        except Exception:
            continue
        ddmm = f"{dd:02d}.{mm:02d}"
        if ddmm not in allowed:
            return True
    return False


def _description_has_foreign_schedule_noise(
    text: str | None,
    *,
    event_date: str | None,
    end_date: str | None,
    event_title: str | None,
) -> bool:
    if _description_has_foreign_schedule_headings(text, event_date=event_date, end_date=end_date):
        return True
    cleaned = _strip_foreign_schedule_sentences(text, event_title=event_title)
    return bool(text) and cleaned.strip() != (text or "").strip()


def _collect_new_candidate_sentences(
    candidate: EventCandidate,
    *,
    before_norm: str,
) -> list[str]:
    variants = []
    if candidate.source_text:
        variants.append(candidate.source_text)
    if candidate.raw_excerpt and candidate.raw_excerpt not in variants:
        variants.append(candidate.raw_excerpt)

    out: list[str] = []
    seen: set[str] = set()
    for text in variants:
        cleaned = _strip_private_use(text) or (text or "")
        cleaned = _strip_promo_lines(cleaned) or cleaned
        cleaned = _strip_giveaway_lines(cleaned) or cleaned
        for chunk in re.split(r"[.!?…]\s+|\n{2,}|\n", cleaned):
            sent = _normalize_candidate_sentence(chunk)
            if _is_low_signal_sentence(sent):
                continue
            key = sent.lower()
            if key in before_norm:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
    return out


def _sentence_quality_score(sent: str) -> int:
    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", sent)
    # Prefer richer factual sentences (more lexical content, reasonable length).
    return min(len(sent), 200) + (len(words) * 3)


def _pick_new_text_snippet(candidate: EventCandidate, before_description: str | None) -> str | None:
    """Pick a short snippet that likely contains *new* facts compared to the previous description."""
    before = _strip_private_use(before_description) or (before_description or "")
    before = re.sub(r"\s+", " ", before).strip().lower()
    new_sentences = _collect_new_candidate_sentences(candidate, before_norm=before)
    if new_sentences:
        best = max(new_sentences, key=_sentence_quality_score)
        return _normalize_fact_item(best, limit=140)
    # Fallback: best-effort excerpt
    variants = []
    if candidate.source_text:
        variants.append(candidate.source_text)
    if candidate.raw_excerpt and candidate.raw_excerpt not in variants:
        variants.append(candidate.raw_excerpt)
    best = max((v for v in variants if v), key=lambda v: len(v), default="")
    return _normalize_fact_item(best, limit=140) if best else None


def _pick_new_description_snippet(
    after_description: str | None,
    before_description: str | None,
    *,
    candidate: EventCandidate,
) -> str | None:
    """Pick a snippet that is present in the final description and likely new.

    This makes the operator-facing "Текст дополнен: ..." fact verifiable by reading
    the Telegraph page (which is rendered from `event.description`).
    """
    after = _strip_private_use(after_description) or (after_description or "")
    after = _strip_private_use(after) or after
    before = _strip_private_use(before_description) or (before_description or "")
    before = _strip_private_use(before) or before
    before_norm = re.sub(r"\s+", " ", before).strip().lower()

    candidates: list[str] = []
    for chunk in re.split(r"[.!?…]\s+|\n{2,}|\n", after):
        sent = _normalize_candidate_sentence(chunk)
        if _is_low_signal_sentence(sent):
            continue
        key = sent.lower()
        if key in before_norm:
            continue
        candidates.append(sent)

    if candidates:
        best = max(candidates, key=_sentence_quality_score)
        return _normalize_fact_item(best, limit=140)

    # Fallback to the old candidate-based heuristic.
    return _pick_new_text_snippet(candidate, before_description)


def _dedupe_source_facts(facts: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for fact in facts:
        key = re.sub(r"\s+", " ", str(fact or "")).strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(str(fact).strip())
    return out


def _drop_redundant_poster_facts(facts: Sequence[str]) -> list[str]:
    """Drop 'Афиша в источнике' when it points to the same URL as 'Добавлена афиша'."""
    url_re = re.compile(
        r"^(?P<kind>Афиша в источнике|Добавлена афиша):\s+(?P<url>https?://\S+)\s*$",
        re.IGNORECASE,
    )
    added_urls: set[str] = set()
    source_urls: set[str] = set()
    parsed: list[tuple[str, str, str]] = []
    passthrough: list[str] = []
    for fact in facts:
        m = url_re.match((fact or "").strip())
        if not m:
            passthrough.append(fact)
            continue
        kind = (m.group("kind") or "").strip().lower()
        url = (m.group("url") or "").strip()
        parsed.append((fact, kind, url))
        if "добавлена" in kind:
            added_urls.add(url)
        else:
            source_urls.add(url)
    out: list[str] = []
    for original, kind, url in parsed:
        if "афиша в источнике" in kind and url in added_urls:
            continue
        out.append(original)
    out.extend(passthrough)
    return out


def _fallback_merge_description(
    before: str | None,
    candidate: EventCandidate,
    *,
    max_sentences: int = 2,
) -> str | None:
    """Best-effort deterministic merge when LLM merge is unavailable.

    We keep the existing description as-is and append a couple of truly new sentences
    extracted from the candidate (source_text preferred, then raw_excerpt).
    """
    before_text = (before or "").strip()
    before_norm = re.sub(r"\s+", " ", before_text).strip().lower()

    new_sentences = _collect_new_candidate_sentences(candidate, before_norm=before_norm)

    if not new_sentences:
        return _dedupe_description(before_text) or before_text or None

    ranked = sorted(
        range(len(new_sentences)),
        key=lambda idx: (_sentence_quality_score(new_sentences[idx]), -idx),
        reverse=True,
    )
    picked_idx = sorted(ranked[: max(1, int(max_sentences))])
    picked = [new_sentences[idx] for idx in picked_idx]

    merged = (before_text + "\n" + " ".join(picked)).strip() if before_text else " ".join(picked)
    return _dedupe_description(merged) or merged or None


def _should_prefer_source_text_for_description(
    clean_source_text: str | None,
    clean_raw_excerpt: str | None,
) -> bool:
    """Prefer source_text as full-description seed over short excerpt."""
    source = (clean_source_text or "").strip()
    excerpt = (clean_raw_excerpt or "").strip()
    if not source:
        return False
    if not excerpt:
        return True
    source_len = len(source)
    excerpt_len = len(excerpt)
    if source_len >= excerpt_len + 120:
        return True
    if excerpt in source and source_len >= max(int(excerpt_len * 1.35), excerpt_len + 60):
        return True
    return False


async def smart_event_update(
    db: Database,
    candidate: EventCandidate,
    *,
    check_source_url: bool = True,
    schedule_tasks: bool = True,
    schedule_kwargs: dict[str, Any] | None = None,
) -> SmartUpdateResult:
    async with _SMART_UPDATE_LOCK:
        return await _smart_event_update_impl(
            db,
            candidate,
            check_source_url=check_source_url,
            schedule_tasks=schedule_tasks,
            schedule_kwargs=schedule_kwargs,
        )


async def _apply_holiday_festival_mapping(db: Database, event_id: int) -> bool:
    """Ensure pseudo-festivals from docs/reference/holidays.md are applied universally.

    This runs as part of Smart Update so that holiday grouping does not depend on
    the caller (VK auto import / Telegram Monitoring / manual add flows).
    """
    try:
        from main import ensure_festival, get_holiday_record
    except Exception:
        return False

    try:
        import vk_intake as vk_mod
    except Exception:
        return False

    async with db.get_session() as session:
        ev = await session.get(Event, int(event_id))
        if not ev:
            return False
        fest_value = (getattr(ev, "festival", None) or "").strip()
        if not fest_value:
            return False
        record = get_holiday_record(fest_value)
        if record is None:
            return False
        event_date = (getattr(ev, "date", None) or "").strip()
        event_end_date = (getattr(ev, "end_date", None) or "").strip() or None
        tolerance = getattr(record, "tolerance_days", None)
        if not vk_mod._event_date_matches_holiday(record, event_date, event_end_date, tolerance):
            return False

        photo_urls = list(getattr(ev, "photo_urls", None) or [])
        current_festival = getattr(ev, "festival", None)

    canonical_name = str(getattr(record, "canonical_name", "") or "").strip()
    if not canonical_name:
        return False

    target_year = date.today().year
    date_token = (event_date or "").split("..", 1)[0].strip()
    try:
        target_year = date.fromisoformat(date_token).year
    except Exception:
        target_year = date.today().year

    start_iso, end_iso = vk_mod._holiday_date_range(record, target_year)
    ensure_kwargs: dict[str, Any] = {}
    desc = str(getattr(record, "description", "") or "").strip()
    if desc:
        ensure_kwargs["description"] = desc
        ensure_kwargs["source_text"] = desc
    if start_iso:
        ensure_kwargs["start_date"] = start_iso
    if end_iso:
        ensure_kwargs["end_date"] = end_iso
    if photo_urls:
        ensure_kwargs["photo_url"] = photo_urls[0]
        ensure_kwargs["photo_urls"] = photo_urls
    aliases_payload = [a for a in getattr(record, "normalized_aliases", ()) if a]
    if aliases_payload:
        ensure_kwargs["aliases"] = aliases_payload

    fest_obj, fest_created, fest_updated = await ensure_festival(
        db,
        canonical_name,
        **ensure_kwargs,
    )
    _ = fest_obj

    changed = bool(fest_created or fest_updated)

    if (current_festival or "") != canonical_name:
        async with db.get_session() as session:
            ev2 = await session.get(Event, int(event_id))
            if ev2 and (getattr(ev2, "festival", None) or "") != canonical_name:
                ev2.festival = canonical_name
                session.add(ev2)
                await session.commit()
                changed = True

    return changed


_TG_DEFAULT_LOCATION_CITY_DISAMBIGUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["default", "extracted"]},
        "confidence": {"type": "number"},
        "reason_short": {"type": "string"},
    },
    "required": ["decision", "confidence", "reason_short"],
    "additionalProperties": False,
}


async def _maybe_disambiguate_telegram_default_location_city(candidate: EventCandidate) -> None:
    """Best-effort: decide whether to keep TelegramSource.default_location city.

    Telegram sources can have `default_location` (a strong prior), but sometimes posts
    genuinely describe an event in another city. At the same time, "context cities"
    like "(г. Москва)" can appear in biographies and must not flip the event city.

    This helper runs a small Gemma JSON check only when:
    - source_type is Telegram
    - default city exists
    - extracted city exists and conflicts with default
    - the current candidate city equals the default city (i.e. we overrode it earlier)
    """
    try:
        st = str(candidate.source_type or "").strip().lower()
    except Exception:
        st = ""
    if st not in {"telegram", "tg"}:
        return
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else None
    if not metrics:
        return

    default_location = str(metrics.get("tg_default_location") or "").strip()
    default_city = str(metrics.get("tg_default_city") or "").strip()
    extracted_city = str(metrics.get("tg_extracted_city") or "").strip()
    if not default_location or not default_city or not extracted_city:
        return
    if default_city.casefold() == extracted_city.casefold():
        return
    current_city = str(candidate.city or "").strip()
    if not current_city or current_city.casefold() != default_city.casefold():
        return
    if SMART_UPDATE_LLM_DISABLED:
        return

    extracted_location_name = str(metrics.get("tg_extracted_location_name") or "").strip()
    extracted_location_address = str(metrics.get("tg_extracted_location_address") or "").strip()
    source_text = _clip_to_readable_boundary(candidate.source_text, 2200)

    prompt = (
        "Ты помощник по импорту городских событий из Telegram.\n"
        "У источника есть `default_location` — это сильный prior (обычно события проходят там), "
        "но иногда пост может описывать событие в другом городе.\n"
        "Extractor мог ошибочно извлечь город из контекста (например «(г. Москва)» про участников), "
        "а не про место проведения.\n\n"
        "Задача: определить, где проходит событие — в default_location (город default_city) "
        "или в extracted_city.\n"
        "Верни строго JSON:\n"
        '{"decision":"default|extracted","confidence":0.0,"reason_short":"..."}\n\n'
        "Правила:\n"
        "- НЕ выдумывай третий город.\n"
        "- Если место проведения явно в extracted_city (адрес/площадка/метро + город) — выбери `extracted`.\n"
        "- Если extracted_city упоминается как контекст (происхождение артистов/организаторов и т.п.), "
        "а место проведения соответствует default_location — выбери `default`.\n"
        "- Если не уверен — выбери `default` и поставь confidence <= 0.6.\n\n"
        f"default_location: {default_location!r}\n"
        f"default_city: {default_city!r}\n"
        f"extracted_city: {extracted_city!r}\n"
        f"extracted_location_name: {extracted_location_name!r}\n"
        f"extracted_location_address: {extracted_location_address!r}\n"
        f"event_title: {_clip_title(candidate.title, 120)!r}\n"
        f"event_date: {str(candidate.date or '').strip()!r}\n"
        f"event_time: {str(candidate.time or '').strip()!r}\n"
        f"post_text:\n{source_text}\n"
    )

    data = await _ask_gemma_json(
        prompt,
        _TG_DEFAULT_LOCATION_CITY_DISAMBIGUATION_SCHEMA,
        max_tokens=140,
        label="tg_city_disambiguation",
    )
    if not isinstance(data, dict):
        return
    decision = str(data.get("decision") or "").strip().lower()
    try:
        confidence = float(data.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    reason_short = str(data.get("reason_short") or "").strip()

    metrics["tg_city_disambiguation_decision"] = decision or None
    metrics["tg_city_disambiguation_confidence"] = confidence
    metrics["tg_city_disambiguation_reason"] = reason_short[:160] if reason_short else None

    if decision != "extracted" or confidence < 0.7:
        logger.info(
            "smart_update.tg_city_disambiguation keep_default decision=%s conf=%.2f default=%s extracted=%s source_url=%s",
            decision,
            confidence,
            default_city,
            extracted_city,
            candidate.source_url,
        )
        return

    # Apply the override: let the region filter reject truly out-of-scope events.
    candidate.city = extracted_city
    if extracted_location_name:
        candidate.location_name = extracted_location_name
    if extracted_location_address:
        candidate.location_address = extracted_location_address
    logger.info(
        "smart_update.tg_city_disambiguation override_city default=%s extracted=%s conf=%.2f source_url=%s",
        default_city,
        extracted_city,
        confidence,
        candidate.source_url,
    )


async def _smart_event_update_impl(
    db: Database,
    candidate: EventCandidate,
    *,
    check_source_url: bool = True,
    schedule_tasks: bool = True,
    schedule_kwargs: dict[str, Any] | None = None,
) -> SmartUpdateResult:
    logger.info(
        "smart_update.start source_type=%s source_url=%s title=%s date=%s time=%s location=%s city=%s posters=%d trust=%s festival_context=%s festival=%s festival_full=%s festival_source=%s festival_series=%s",
        candidate.source_type,
        candidate.source_url,
        _clip_title(candidate.title),
        candidate.date,
        candidate.time,
        _clip_title(candidate.location_name, 60),
        candidate.city,
        len(candidate.posters),
        candidate.trust_level,
        (candidate.festival_context or "none"),
        _clip_title(candidate.festival, 80),
        _clip_title(candidate.festival_full, 120),
        int(bool(candidate.festival_source)) if candidate.festival_source is not None else None,
        _clip_title(candidate.festival_series, 80),
    )
    (
        candidate.location_name,
        candidate.location_address,
        candidate.city,
    ) = _canonicalize_location_fields(
        location_name=candidate.location_name,
        location_address=candidate.location_address,
        city=candidate.city,
        source_chat_username=candidate.source_chat_username,
        source_url=candidate.source_url,
    )
    if not candidate.date:
        logger.warning(
            "smart_update.invalid reason=missing_date source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.title),
        )
        return SmartUpdateResult(status="invalid", reason="missing_date")
    if not candidate.title:
        logger.warning(
            "smart_update.invalid reason=missing_title source_type=%s source_url=%s",
            candidate.source_type,
            candidate.source_url,
        )
        return SmartUpdateResult(status="invalid", reason="missing_title")
    if not candidate.location_name:
        logger.warning(
            "smart_update.invalid reason=missing_location source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.title),
        )
        return SmartUpdateResult(status="invalid", reason="missing_location")

    inferred_default_end_date = _maybe_apply_default_end_date_for_long_event(candidate)

    if _should_skip_past_smart_update_candidate(candidate):
        logger.info(
            "smart_update.skip reason=past_event source_type=%s source_url=%s title=%s date=%s end_date=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.title),
            candidate.date,
            candidate.end_date,
        )
        return SmartUpdateResult(status="skipped_past_event", reason="past_event")

    await _maybe_disambiguate_telegram_default_location_city(candidate)

    # Deterministic region filter (project scope: Kaliningrad Oblast).
    # If extracted city/settlement is outside the region (or cannot be reliably resolved),
    # reject early so out-of-scope events do not get created/merged.
    try:
        if (os.getenv("REGION_FILTER_ENABLED", "1") or "").strip().lower() in {"1", "true", "yes", "on"}:
            # Manual operator actions (bot commands) must not be blocked by a missing city.
            # The region filter is meant primarily for automated ingestion (VK/TG/parsers).
            if (candidate.source_type or "").strip().lower() not in {"bot"}:
                from geo_region import decide_kaliningrad_oblast

                region_decision = await decide_kaliningrad_oblast(
                    db,
                    city=candidate.city,
                    location_address=candidate.location_address,
                    gemma_client=_get_gemma_client(),
                )
                strict = (os.getenv("REGION_FILTER_STRICT", "1") or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if region_decision.allowed is False or (strict and region_decision.allowed is None):
                    logger.info(
                        "smart_update.rejected reason=%s source_type=%s source_url=%s city=%s region=%s source=%s",
                        region_decision.reason,
                        candidate.source_type,
                        candidate.source_url,
                        candidate.city,
                        region_decision.region_name or region_decision.region_code,
                        region_decision.source,
                    )
                    return SmartUpdateResult(
                        status="rejected_out_of_region",
                        reason=region_decision.reason,
                    )
    except Exception as e:  # pragma: no cover - must not crash ingestion
        logger.warning("smart_update.region_filter_failed err=%s", e)

    clean_title = _strip_private_use(candidate.title) or (candidate.title or "")
    if not clean_title:
        logger.warning(
            "smart_update.invalid reason=empty_title_after_clean source_type=%s source_url=%s",
            candidate.source_type,
            candidate.source_url,
        )
        return SmartUpdateResult(status="invalid", reason="empty_title_after_clean")
    context_value = (candidate.festival_context or "").strip().lower()
    if context_value == "festival_post":
        logger.info(
            "smart_update.skip reason=festival_post source_type=%s source_url=%s festival=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.festival or candidate.festival_series, 80),
        )
        return SmartUpdateResult(status="skipped_festival_post", reason="festival_post")
    raw_source_text = _strip_private_use(candidate.source_text) or (
        candidate.source_text or ""
    )
    raw_excerpt = _strip_private_use(candidate.raw_excerpt) or (candidate.raw_excerpt or "")

    # Avoid confusing self-references for non-parser sources (e.g. taking the post URL as "ticket_link").
    # We still allow t.me links for registration/DM, but they must not be identical to the source URL.
    if not str(candidate.source_type or "").strip().lower().startswith("parser:"):
        try:
            if _normalize_url(candidate.ticket_link) and _normalize_url(candidate.ticket_link) == _normalize_url(candidate.source_url):
                candidate.ticket_link = None
        except Exception:
            pass

    text_filter_facts: list[str] = []
    queue_notes: list[str] = []

    def _push_queue_note(note: str) -> None:
        value = str(note or "").strip()
        if not value:
            return
        if value not in queue_notes:
            queue_notes.append(value)

    async def _enqueue_ticket_sites_queue(session: Any, *, event_id: int) -> None:
        if not event_id:
            return
        try:
            from ticket_sites_queue import (
                enqueue_ticket_site_urls_in_session,
                extract_ticket_site_urls,
            )
        except Exception:
            return

        links_payload: list[Any] = []
        if candidate.links_payload is not None:
            links_payload.append(candidate.links_payload)
        if candidate.source_url:
            links_payload.append(candidate.source_url)
        if candidate.ticket_link:
            links_payload.append(candidate.ticket_link)

        urls = extract_ticket_site_urls(
            text=candidate.source_text,
            links_payload=links_payload,
            events_payload=[
                {
                    "ticket_link": candidate.ticket_link,
                    "links": candidate.links_payload,
                }
            ],
        )
        if not urls:
            return

        now = datetime.now(timezone.utc)
        enqueued = 0
        try:
            enqueued = await enqueue_ticket_site_urls_in_session(
                session,
                urls=urls,
                event_id=int(event_id),
                source_post_url=candidate.source_url,
                source_chat_username=candidate.source_chat_username,
                source_chat_id=candidate.source_chat_id,
                source_message_id=candidate.source_message_id,
                now=now,
            )
        except Exception:
            logger.warning(
                "smart_update: ticket_sites_queue enqueue failed source_type=%s source_url=%s event_id=%s",
                candidate.source_type,
                candidate.source_url,
                event_id,
                exc_info=True,
            )
            return

        total = int(enqueued or 0)
        if total <= 0:
            return

        if len(urls) == 1:
            _push_queue_note(f"🎟 {urls[0]} добавлена в очередь мониторинга билетных сайтов")
            return
        _push_queue_note(f"🎟 В очередь мониторинга билетных сайтов добавлено ссылок: {total}/{len(urls)}")
        for u in urls[:2]:
            _push_queue_note(f"🎟 {u}")
        if len(urls) > 2:
            _push_queue_note(f"🎟 … ещё {len(urls) - 2}")

    if inferred_default_end_date:
        text_filter_facts.append(f"Дата окончания по умолчанию: {inferred_default_end_date}")

    # Giveaways: keep event facts but strip giveaway mechanics when possible.
    is_giveaway = _looks_like_ticket_giveaway(clean_title, raw_source_text, raw_excerpt)
    if is_giveaway:
        before_src = raw_source_text
        before_excerpt = raw_excerpt
        raw_source_text = _strip_giveaway_lines(raw_source_text) or raw_source_text
        raw_excerpt = _strip_giveaway_lines(raw_excerpt) or raw_excerpt
        if (before_src or "") != (raw_source_text or "") or (before_excerpt or "") != (raw_excerpt or ""):
            text_filter_facts.append("Убрана механика розыгрыша")
        # If we still don't have a plausible event, treat as non-event content.
        if not (
            _giveaway_has_underlying_event_facts(raw_source_text)
            or _giveaway_has_underlying_event_facts(raw_excerpt)
        ):
            logger.info(
                "smart_update.skip reason=giveaway_no_event source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_giveaway", reason="giveaway_no_event")

    # Congratulation posts must not become events or sources.
    if _looks_like_promo_or_congrats(clean_title, raw_source_text, raw_excerpt) and not _candidate_has_event_anchors(candidate):
        logger.info(
            "smart_update.skip reason=promo_or_congrats source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(clean_title),
        )
        return SmartUpdateResult(status="skipped_promo", reason="promo_or_congrats")

    before_promo_src = raw_source_text
    before_promo_excerpt = raw_excerpt
    clean_source_text = _strip_promo_lines(raw_source_text) or raw_source_text or ""
    clean_raw_excerpt = _strip_promo_lines(raw_excerpt) or raw_excerpt
    if is_giveaway:
        clean_source_text = _strip_giveaway_lines(clean_source_text) or clean_source_text
        clean_raw_excerpt = _strip_giveaway_lines(clean_raw_excerpt) or clean_raw_excerpt
    clean_source_text = _normalize_bullet_markers(clean_source_text) or clean_source_text
    clean_raw_excerpt = _normalize_bullet_markers(clean_raw_excerpt) or clean_raw_excerpt
    if (before_promo_src or "") != (clean_source_text or "") or (before_promo_excerpt or "") != (clean_raw_excerpt or ""):
        text_filter_facts.append("Убраны промо-фрагменты")

    # Non-event notices / course ads (VK/TG): skip early to avoid creating pseudo-events.
    source_type_clean = str(candidate.source_type or "").strip().lower()
    if source_type_clean in {"vk", "telegram", "tg"}:
        combined_text = "\n".join(
            [
                clean_source_text or "",
                clean_raw_excerpt or "",
            ]
        ).strip()
        if _looks_like_open_call_not_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=open_call source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="open_call")
        if _looks_like_work_schedule_notice(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=work_schedule source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="work_schedule")
        if _looks_like_non_event_notice(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=non_event_notice source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="non_event_notice")
        if _looks_like_venue_status_update_not_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=venue_status_update source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="venue_status_update")
        if _looks_like_congrats_notice_not_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=congrats_notice source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="congrats_notice")
        if _looks_like_course_promo(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=course_promo source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="course_promo")
        if _looks_like_service_promo_not_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=service_promo source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="service_promo")
        outage_reason = _looks_like_utility_outage_or_road_closure(clean_title, combined_text)
        if outage_reason:
            logger.info(
                "smart_update.skip reason=%s source_type=%s source_url=%s title=%s",
                outage_reason,
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason=outage_reason)
        if _looks_like_too_soon_notice(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=too_soon source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="too_soon")
        # Project policy: auto-ingestion should not create online-only events.
        if _looks_like_online_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=online_event source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="online_event")
        if _looks_like_book_review_not_event(clean_title, combined_text):
            logger.info(
                "smart_update.skip reason=book_review source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="book_review")
        if _looks_like_photo_day_not_event(clean_title, combined_text, candidate=candidate):
            logger.info(
                "smart_update.skip reason=photo_day source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
            )
            return SmartUpdateResult(status="skipped_non_event", reason="photo_day")

    # Ticket price grounding: prevent hallucinated min/max prices for VK/TG sources.
    # Only accept price values when the source text/OCR contains explicit price signals.
    if source_type_clean in {"vk", "telegram", "tg"} and (
        candidate.ticket_price_min is not None or candidate.ticket_price_max is not None
    ):
        poster_texts_for_price: list[str] = []
        for p in candidate.posters or []:
            for k in ("ocr_text", "ocr_title"):
                v = getattr(p, k, None)
                if isinstance(v, str) and v.strip():
                    poster_texts_for_price.append(v.strip())
        price_probe = "\n".join(
            [
                clean_title,
                clean_source_text or "",
                clean_raw_excerpt or "",
                *poster_texts_for_price[:3],
            ]
        ).strip()
        if not _has_price_evidence(price_probe, candidate.ticket_price_min, candidate.ticket_price_max):
            before_min = candidate.ticket_price_min
            before_max = candidate.ticket_price_max
            candidate.ticket_price_min = None
            candidate.ticket_price_max = None
            note = "Цена отброшена: не найдена в источнике"
            if note not in text_filter_facts:
                text_filter_facts.append(note)
            logger.info(
                "smart_update.price_dropped source_type=%s source_url=%s title=%s before=%s..%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(clean_title),
                before_min,
                before_max,
            )

    # Blood donation actions are free-to-attend; mentions of money are typically about
    # donor compensation, not an entrance fee. If no ticket price survived grounding,
    # mark as free so Telegraph/VK summaries don't show it as "paid tickets".
    if (
        source_type_clean in {"vk", "telegram", "tg"}
        and candidate.is_free is not True
        and candidate.ticket_price_min is None
        and candidate.ticket_price_max is None
    ):
        free_probe = "\n".join(
            [clean_title, clean_source_text or "", clean_raw_excerpt or ""]
        ).strip()
        if _looks_like_blood_donation_event(clean_title, free_probe):
            candidate.is_free = True
            note = "Помечено как бесплатное: донорская акция"
            if note not in text_filter_facts:
                text_filter_facts.append(note)

    # Best-effort: if the source contains festival context (festival post OR event within a festival),
    # enqueue it into the festival queue so operators can later run `/fest_queue` and build/update
    # festival pages. This is intentionally deterministic (regex/signal-based), not another LLM call.
    try:
        from festival_queue import detect_festival_context, enqueue_festival_source
        from models import FestivalQueueItem

        def _map_source_kind(source_type: str) -> str:
            st = (source_type or "").strip().lower()
            if st in {"tg", "telegram"}:
                return "tg"
            if st in {"vk"}:
                return "vk"
            return "url"

        def _parse_vk_ids(url: str | None) -> tuple[int | None, int | None]:
            if not url:
                return None, None
            m = re.search(r"wall-?(\d+)_([0-9]+)", url)
            if not m:
                return None, None
            return int(m.group(1)), int(m.group(2))

        queue_url = (candidate.source_url or "").strip()
        if queue_url:
            poster_texts = [
                p.ocr_text
                for p in (candidate.posters or [])
                if (p.ocr_text or "").strip()
            ]
            ev_payload = {
                "title": candidate.title,
                "date": candidate.date,
                "end_date": candidate.end_date,
                "time": candidate.time,
                "location_name": candidate.location_name,
                "event_type": candidate.event_type,
                "ticket_link": candidate.ticket_link,
                "festival": candidate.festival,
                "festival_full": candidate.festival_full,
                "festival_context": candidate.festival_context,
            }
            decision = detect_festival_context(
                parsed_events=[ev_payload],
                festival_payload={
                    "festival": candidate.festival,
                    "festival_full": candidate.festival_full,
                    "festival_context": candidate.festival_context,
                },
                source_text=clean_source_text or clean_raw_excerpt,
                poster_texts=poster_texts,
                source_is_festival=bool(candidate.festival_source),
                source_series=candidate.festival_series,
            )
            if decision and decision.context != "none" and (decision.festival or decision.festival_full):
                # Backfill candidate fields for better operator logs and future merges.
                if (candidate.festival_context or "").strip().lower() in {"", "none"} and decision.context:
                    candidate.festival_context = decision.context
                if not (candidate.festival or "").strip() and decision.festival:
                    candidate.festival = decision.festival
                if not (candidate.festival_full or "").strip() and decision.festival_full:
                    candidate.festival_full = decision.festival_full
                if decision.dedup_links and not list(candidate.festival_dedup_links or []):
                    candidate.festival_dedup_links = list(decision.dedup_links)

                source_kind = _map_source_kind(candidate.source_type)
                async with db.get_session() as _fest_session:
                    done_id = (
                        await _fest_session.execute(
                            select(FestivalQueueItem.id)
                            .where(
                                FestivalQueueItem.source_kind == source_kind,
                                FestivalQueueItem.source_url == queue_url,
                                FestivalQueueItem.status == "done",
                            )
                            .limit(1)
                        )
                    ).scalar_one_or_none()

                if done_id is None:
                    gid, pid = _parse_vk_ids(queue_url)
                    item = await enqueue_festival_source(
                        db,
                        source_kind=source_kind,
                        source_url=queue_url,
                        source_text=clean_source_text or clean_raw_excerpt,
                        festival_context=decision.context,
                        festival_name=decision.festival,
                        festival_full=decision.festival_full,
                        festival_series=candidate.festival_series,
                        dedup_links=decision.dedup_links,
                        signals=decision.signals,
                        source_chat_username=candidate.source_chat_username,
                        source_chat_id=candidate.source_chat_id,
                        source_message_id=candidate.source_message_id,
                        source_group_id=gid,
                        source_post_id=pid,
                    )
                    msg = (
                        f"🎪 Добавлено в фестивальную очередь: {decision.context} "
                        f"{decision.festival or decision.festival_full} (id={getattr(item, 'id', None)})"
                    )
                    text_filter_facts.append(msg)
                    _push_queue_note(msg)
                else:
                    msg = f"🎪 Фестивальная очередь: уже done (id={done_id})"
                    text_filter_facts.append(msg)
                    _push_queue_note(msg)
    except Exception:
        logger.warning("smart_update: festival_queue enqueue failed", exc_info=True)

    # If the source is detected as a festival/program post, it must not create/update events.
    # Some upstream extractors (notably Telegram Monitoring) may not populate `festival_context`,
    # so we backfill it via deterministic detection above and enforce the policy here.
    if (candidate.festival_context or "").strip().lower() == "festival_post":
        logger.info(
            "smart_update.skip reason=festival_post_detected source_type=%s source_url=%s festival=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(candidate.festival or candidate.festival_series, 80),
        )
        return SmartUpdateResult(
            status="skipped_festival_post",
            reason="festival_post",
            queue_notes=list(queue_notes or []),
        )

    # Multi-event digests should not be imported as a single event.
    if (candidate.source_type in {"vk", "tg", "telegram"}) and _looks_like_schedule_digest(
        clean_source_text or clean_raw_excerpt,
        event_date=candidate.date,
        end_date=candidate.end_date,
    ):
        logger.info(
            "smart_update.reject reason=schedule_digest source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(clean_title),
        )
        return SmartUpdateResult(status="rejected_schedule_digest", reason="schedule_digest")

    # "Акции" must not become events. If after promo/giveaway stripping there's no real event anchor,
    # treat it as non-event content.
    if (
        not _candidate_has_event_anchors(candidate)
        and _PROMO_STRIP_RE.search((clean_title or "") + "\n" + (clean_source_text or ""))
        and len((clean_raw_excerpt or clean_source_text or "").strip()) < 140
    ):
        logger.info(
            "smart_update.skip reason=promo_only source_type=%s source_url=%s title=%s",
            candidate.source_type,
            candidate.source_url,
            _clip_title(clean_title),
        )
        return SmartUpdateResult(status="skipped_promo", reason="promo_only")

    # Posters policy:
    # Keep all posters (dedupe/order happens later). OCR is used for prioritization only.
    # This avoids events ending up without images due to overly strict filtering.
    force_silent_due_to_date_risk = False
    poster_filter_facts: list[str] = []
    if candidate.posters:
        # Best-effort: backfill missing OCR from local cache (cheap, no network).
        missing_hashes = [
            p.sha256 for p in candidate.posters if p.sha256 and not (p.ocr_text or "").strip()
        ]
        if missing_hashes:
            try:
                async with db.get_session() as session:
                    rows = (
                        await session.execute(
                            select(PosterOcrCache)
                            .where(PosterOcrCache.hash.in_(missing_hashes))
                            .order_by(PosterOcrCache.created_at.desc())
                        )
                    ).scalars().all()
                latest: dict[str, PosterOcrCache] = {}
                for row in rows:
                    if row.hash not in latest:
                        latest[row.hash] = row
                for p in candidate.posters:
                    if not p.sha256 or (p.ocr_text or "").strip():
                        continue
                    cached = latest.get(p.sha256)
                    if cached and (cached.text or "").strip():
                        p.ocr_text = cached.text
                        if cached.title:
                            p.ocr_title = cached.title
            except Exception:
                logger.warning("smart_update: poster OCR cache backfill failed", exc_info=True)
        note = _far_future_poster_date_mismatch_note(
            candidate_date=candidate.date,
            posters=candidate.posters,
            months_threshold=SMART_UPDATE_FAR_FUTURE_REVIEW_MONTHS,
        )
        if note:
            force_silent_due_to_date_risk = True
            poster_filter_facts.append(note)

    if check_source_url and candidate.source_url:
        timing = (os.getenv("SMART_UPDATE_DEBUG_TIMING") or "").strip().lower() in {"1", "true", "yes"}
        t0 = time.monotonic() if timing else 0.0
        exists = None
        # Keep this fast: avoid ORM session/engine initialization for a simple lookup.
        try:
            async with db.raw_conn() as conn:
                if candidate.source_type:
                    cur = await conn.execute(
                        "SELECT 1 FROM event_source WHERE source_type=? AND source_url=? LIMIT 1",
                        (candidate.source_type, candidate.source_url),
                    )
                else:
                    cur = await conn.execute(
                        "SELECT 1 FROM event_source WHERE source_url=? LIMIT 1",
                        (candidate.source_url,),
                    )
                exists = await cur.fetchone()
        except Exception:
            logger.warning(
                "smart_update: source_url idempotency check failed (fallback to full flow)",
                exc_info=True,
            )
            exists = None
        if timing:
            logger.info(
                "smart_update.timing idempotency_check_ms=%d source_type=%s",
                int((time.monotonic() - t0) * 1000),
                candidate.source_type,
            )
        if exists:
            logger.info(
                "smart_update.skip reason=source_url_exists source_type=%s source_url=%s title=%s",
                candidate.source_type,
                candidate.source_url,
                _clip_title(candidate.title),
            )
            return SmartUpdateResult(status="skipped_same_source_url", reason="source_url_exists")

    cand_start, cand_end = _candidate_date_range(candidate)
    if not cand_start or not cand_end:
        return SmartUpdateResult(status="invalid", reason="invalid_date")

    anchor_match = await _match_existing_event_by_source_anchor(db, candidate)
    if anchor_match is None and (not check_source_url):
        # When explicit idempotency is disabled (reprocessing allowed), still try to converge
        # on an existing event that already has this source_url attached.
        try:
            anchor_match = await _match_existing_event_by_event_source_url(db, candidate)
        except Exception:
            logger.warning("smart_update: event_source_url anchor match failed", exc_info=True)
    if anchor_match is not None:
        shortlist = [anchor_match]
        anchor_forced = True
    else:
        anchor_forced = False
        async with db.get_session() as session:
            stmt = select(Event).where(
                and_(
                    Event.date <= cand_end.isoformat(),
                    or_(
                        and_(
                            Event.end_date.is_(None),
                            Event.date >= cand_start.isoformat(),
                        ),
                        Event.end_date >= cand_start.isoformat(),
                    ),
                )
            )
            stmt = _apply_soft_city_filter(stmt, candidate.city)
            res = await session.execute(stmt)
            shortlist = list(res.scalars().all())

    is_canonical_site = str(candidate.source_type or "").startswith("parser:")
    city_noise_rescued = False
    if (not anchor_forced) and (not shortlist):
        city_noise_match, city_noise_reason = await _match_existing_event_by_city_noise_rescue(
            db,
            candidate,
            is_canonical_site=is_canonical_site,
        )
        if city_noise_match is not None:
            shortlist = [city_noise_match]
            city_noise_rescued = True
            logger.info(
                "smart_update.shortlist rescue=%s event_id=%s source_type=%s source_url=%s",
                city_noise_reason,
                getattr(city_noise_match, "id", None),
                candidate.source_type,
                candidate.source_url,
            )

    if (not anchor_forced) and (not city_noise_rescued) and candidate.location_name:
        shortlist = [
            ev for ev in shortlist if _event_candidate_location_matches(ev, candidate)
        ]

    # Time is an anchor field, but for canonical site/parser imports we allow time corrections:
    # matching must work even if a Telegram-first event had a wrong/empty time.
    cand_time_norm = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
    if (not anchor_forced) and (not city_noise_rescued) and cand_time_norm and (not is_canonical_site):
        time_filtered: list[Event] = []
        for ev in shortlist:
            ev_time_anchor = _event_anchor_time(ev)
            if (not ev_time_anchor) or (ev_time_anchor == cand_time_norm):
                time_filtered.append(ev)
        if time_filtered:
            shortlist = time_filtered

    # If the candidate has no explicit time, try to shrink the shortlist to the only
    # title-related event (helps prevent duplicates when time is missing but the match is obvious).
    if (
        (not anchor_forced)
        and (not city_noise_rescued)
        and (not cand_time_norm)
        and candidate.location_name
        and len(shortlist) > 1
    ):
        related = [
            ev
            for ev in shortlist
            if (not _has_explicit_time_conflict(cand_time_norm, _event_anchor_time(ev)))
            and _titles_look_related(candidate.title, getattr(ev, "title", None))
        ]
        if len(related) == 1:
            shortlist = related

    if (not anchor_forced) and (not shortlist) and (not city_noise_rescued):
        city_noise_match, city_noise_reason = await _match_existing_event_by_city_noise_rescue(
            db,
            candidate,
            is_canonical_site=is_canonical_site,
        )
        if city_noise_match is not None:
            shortlist = [city_noise_match]
            city_noise_rescued = True
            logger.info(
                "smart_update.shortlist rescue=%s event_id=%s source_type=%s source_url=%s",
                city_noise_reason,
                getattr(city_noise_match, "id", None),
                candidate.source_type,
                candidate.source_url,
            )

    posters_map: dict[int, list[EventPoster]] = {}
    if shortlist:
        event_ids = [ev.id for ev in shortlist if ev.id]
        posters_map = await _fetch_event_posters_map(db, event_ids)

    allow_parallel = _allow_parallel_events(candidate.location_name)
    candidate_poster_texts = [p.ocr_text for p in candidate.posters if p.ocr_text]
    candidate_hall = _extract_hall_hint(
        (candidate.source_text or "") + "\n" + "\n".join(candidate_poster_texts)
    )
    if allow_parallel and candidate_hall and shortlist:
        filtered: list[Event] = []
        for ev in shortlist:
            ev_posters = posters_map.get(ev.id or 0, [])
            ev_poster_texts = [p.ocr_text for p in ev_posters if p.ocr_text]
            hall = _extract_hall_hint(
                (ev.source_text or "")
                + "\n"
                + (ev.description or "")
                + "\n"
                + "\n".join(ev_poster_texts)
            )
            if hall and hall != candidate_hall:
                continue
            filtered.append(ev)
        shortlist = filtered

    llm_create_bundle: dict[str, Any] | None = None

    if not shortlist:
        match_event = None
        match_reason = "shortlist_empty"
    else:

        # Deterministic single-candidate match is allowed only when anchors look sane.
        # Otherwise fall back to LLM matching / create to avoid catastrophic cross-event merges.
        if anchor_forced:
            match_event = shortlist[0]
            match_reason = "anchor_forced"
        elif len(shortlist) == 1 and _single_candidate_auto_match_ok(
            candidate,
            shortlist[0],
            is_canonical_site=is_canonical_site,
        ):
            match_event = shortlist[0]
            match_reason = "single_candidate"
        else:
            match_event = None
            match_reason = ""

        candidate_hashes = _poster_hashes(candidate.posters)
        ticket_norm = _normalize_url(candidate.ticket_link)

        strong_matches: dict[int, int] = {}
        if ticket_norm:
            for ev in shortlist:
                if _normalize_url(ev.ticket_link) == ticket_norm and ev.id:
                    strong_matches[ev.id] = strong_matches.get(ev.id, 0) + 3
        if candidate_hashes:
            for ev in shortlist:
                hashes = {p.poster_hash for p in posters_map.get(ev.id or 0, [])}
                overlap = len(candidate_hashes & hashes)
                if overlap and ev.id:
                    strong_matches[ev.id] = strong_matches.get(ev.id, 0) + overlap

        logger.info(
            "smart_update.shortlist count=%d allow_parallel=%s source_type=%s source_url=%s",
            len(shortlist),
            bool(allow_parallel),
            candidate.source_type,
            candidate.source_url,
        )
        if match_event is None:
            longrun = _deterministic_same_post_longrun_exact_title_match(
                candidate,
                shortlist,
            )
            if longrun is not None:
                match_event = longrun
                match_reason = "deterministic_same_post_longrun_exact_title"
                logger.info(
                    "smart_update.match type=deterministic_same_post_longrun_exact_title event_id=%s",
                    getattr(match_event, "id", None),
                )

        if match_event is None:
            ticket_anchor_match, ticket_anchor_reason = _deterministic_ticket_source_anchor_match(
                candidate,
                shortlist,
            )
            if ticket_anchor_match is not None:
                match_event = ticket_anchor_match
                match_reason = ticket_anchor_reason
                logger.info(
                    "smart_update.match type=%s event_id=%s",
                    match_reason,
                    getattr(match_event, "id", None),
                )

        if strong_matches and match_event is None:
            best = max(strong_matches.items(), key=lambda item: item[1])
            match_event = next((ev for ev in shortlist if ev.id == best[0]), None)
            match_reason = "strong_match"
            logger.info(
                "smart_update.match type=strong event_id=%s score=%s",
                getattr(match_event, "id", None),
                best[1],
            )

        if match_event is None:
            hard = _deterministic_exact_title_match(
                candidate,
                shortlist,
                is_canonical_site=is_canonical_site,
            )
            if hard is not None:
                match_event = hard
                match_reason = "deterministic_exact_title"
                logger.info(
                    "smart_update.match type=deterministic_exact_title event_id=%s",
                    getattr(match_event, "id", None),
                )

        if match_event is None:
            related = _deterministic_related_title_anchor_match(
                candidate,
                shortlist,
                is_canonical_site=is_canonical_site,
            )
            if related is not None:
                match_event = related
                match_reason = "deterministic_related_title_anchor"
                logger.info(
                    "smart_update.match type=deterministic_related_title_anchor event_id=%s",
                    getattr(match_event, "id", None),
                )

        if match_event is None:
            threshold = 0.85 if allow_parallel and len(shortlist) > 1 else 0.6
            use_match_create_bundle = (
                (os.getenv("SMART_UPDATE_MATCH_CREATE_BUNDLE", "1") or "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            source_type_clean = str(candidate.source_type or "").strip().lower()
            use_match_create_bundle = use_match_create_bundle and source_type_clean in {"vk", "tg", "telegram"}
            if use_match_create_bundle:
                normalized_event_type_hint = _normalize_event_type_value(
                    candidate.title,
                    clean_raw_excerpt or clean_source_text or candidate.source_text,
                    candidate.event_type,
                )
                combo = await _llm_match_or_create_bundle(
                    candidate,
                    shortlist[:10],
                    posters_map=posters_map,
                    threshold=threshold,
                    clean_title=clean_title,
                    clean_source_text=clean_source_text,
                    clean_raw_excerpt=clean_raw_excerpt,
                    normalized_event_type=normalized_event_type_hint,
                )
                if combo and (combo.get("action") == "create"):
                    llm_create_bundle = combo.get("bundle") if isinstance(combo.get("bundle"), dict) else None
                    match_id = None
                    confidence = float(combo.get("confidence") or 0.0)
                    match_reason = str(combo.get("reason_short") or "llm_create_bundle")
                    match_event = None
                elif combo and (combo.get("action") == "match"):
                    match_id = combo.get("match_event_id")
                    confidence = float(combo.get("confidence") or 0.0)
                    match_reason = str(combo.get("reason_short") or "")
                    try:
                        match_id_int = int(match_id) if match_id is not None else None
                    except Exception:
                        match_id_int = None
                    if match_id_int:
                        match_event = next((ev for ev in shortlist if ev.id == match_id_int), None)
                    else:
                        match_event = None
                    if match_event is None:
                        confidence = 0.0
                        match_reason = "llm_bad_match_id"
                    elif len(shortlist) == 1 and not _single_candidate_auto_match_ok(
                        candidate,
                        match_event,
                        is_canonical_site=is_canonical_site,
                    ):
                        match_event = None
                        match_reason = "llm_single_candidate_sanity_reject"
                else:
                    match_id, confidence, reason = await _llm_match_event(
                        candidate, shortlist[:10], posters_map=posters_map
                    )
                    match_reason = reason
                    if match_id:
                        match_event = next((ev for ev in shortlist if ev.id == match_id), None)
                        if match_event is None:
                            confidence = 0.0
                        if confidence < threshold:
                            match_event = None
                            match_reason = f"llm_conf_{confidence:.2f}<={threshold:.2f}"
                        elif len(shortlist) == 1 and not _single_candidate_auto_match_ok(
                            candidate,
                            match_event,
                            is_canonical_site=is_canonical_site,
                        ):
                            match_event = None
                            match_reason = "llm_single_candidate_sanity_reject"
                    else:
                        match_event = None
            else:
                match_id, confidence, reason = await _llm_match_event(
                    candidate, shortlist[:10], posters_map=posters_map
                )
                match_reason = reason
                if match_id:
                    match_event = next((ev for ev in shortlist if ev.id == match_id), None)
                    if match_event is None:
                        confidence = 0.0
                    if confidence < threshold:
                        match_event = None
                        match_reason = f"llm_conf_{confidence:.2f}<={threshold:.2f}"
                    elif len(shortlist) == 1 and not _single_candidate_auto_match_ok(
                        candidate,
                        match_event,
                        is_canonical_site=is_canonical_site,
                    ):
                        match_event = None
                        match_reason = "llm_single_candidate_sanity_reject"
                else:
                    match_event = None
            logger.info(
                "smart_update.match type=llm match_id=%s confidence=%.2f reason=%s",
                match_id,
                float(confidence or 0.0),
                match_reason,
            )

    # Guard: if the matched existing event is semantically unrelated by title, treat it as "no match"
    # and create a new event instead of performing a catastrophic merge.
    #
    # Keep the match only when a deterministic single-candidate match would also be allowed.
    # This is intentionally conservative: long-running events (e.g. exhibitions) can overlap at the same
    # venue, so date-range overlap alone must not override an unrelated title.
    if match_event is not None and not str(candidate.source_type or "").startswith("parser:"):
        if _title_has_meaningful_tokens(candidate.title) and _title_has_meaningful_tokens(
            getattr(match_event, "title", None)
        ):
            if not _titles_look_related(candidate.title, getattr(match_event, "title", None)):
                narrow_reason = str(match_reason or "").strip().lower()
                safe_single = narrow_reason in {
                    "deterministic_specific_ticket_same_slot",
                    "deterministic_doors_start_ticket_bridge",
                }
                try:
                    if (not safe_single) and len(shortlist) == 1:
                        safe_single = _single_candidate_auto_match_ok(
                            candidate,
                            match_event,
                            is_canonical_site=False,
                        )
                except Exception:
                    safe_single = False
                if not safe_single:
                    logger.warning(
                        "smart_update.match_overruled reason=unrelated_titles source_type=%s source_url=%s candidate_title=%s existing_id=%s existing_title=%s",
                        candidate.source_type,
                        candidate.source_url,
                        _clip_title(candidate.title),
                        getattr(match_event, "id", None),
                        _clip_title(getattr(match_event, "title", None)),
                    )
                    match_event = None
                    match_reason = "unrelated_titles"

    # Rescue-match: match/create bundle can decide "create" when candidate title is weak,
    # even though the produced bundle title clearly matches an existing event in the shortlist.
    # Try a deterministic title-based match to prevent duplicates.
    if match_event is None and llm_create_bundle is not None and shortlist:
        bundle_title_raw = llm_create_bundle.get("title") if isinstance(llm_create_bundle, dict) else None
        bundle_title = ""
        if isinstance(bundle_title_raw, str) and bundle_title_raw.strip():
            # Non-semantic cleanup: some models return JSON-ish `\"...\"` fragments.
            bundle_title = (
                (bundle_title_raw or "")
                .strip()
                .replace("\\\\\"", "\"")
                .replace("\\\"", "\"")
                .strip()
            )
            bundle_title = _strip_private_use(bundle_title) or bundle_title
        if bundle_title and _title_has_meaningful_tokens(bundle_title):
            grounded = False
            try:
                grounded = _is_title_grounded_in_candidate_sources(bundle_title, candidate)
            except Exception:
                grounded = False

            if grounded:
                prev_reason = match_reason
                cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=is_canonical_site)
                rescued_matches: list[Event] = []
                for ev in shortlist:
                    if not getattr(ev, "id", None):
                        continue
                    if _has_explicit_time_conflict(cand_time_anchor, _event_anchor_time(ev)):
                        continue
                    if not _titles_look_related(bundle_title, getattr(ev, "title", None)):
                        continue
                    rescued_matches.append(ev)

                chosen: Event | None = None
                if len(rescued_matches) == 1:
                    chosen = rescued_matches[0]
                elif rescued_matches:
                    sigs = {_anchor_signature_for_duplicate_event(ev) for ev in rescued_matches}
                    if len(sigs) == 1:
                        chosen = _pick_best_duplicate_event(rescued_matches)

                if chosen is not None:
                    match_event = chosen
                    match_reason = "rescue_bundle_title"
                    note = "Матчинг: предотвращён дубль (rescue по bundle.title)"
                    if note not in text_filter_facts:
                        text_filter_facts.append(note)
                    logger.info(
                        "smart_update.match type=rescue_bundle_title event_id=%s prev_reason=%s",
                        getattr(match_event, "id", None),
                        prev_reason,
                    )

    if match_event is None:
        normalized_event_type = _normalize_event_type_value(
            candidate.title, candidate.raw_excerpt or candidate.source_text, candidate.event_type
        )
        normalized_digest = _clean_search_digest(candidate.search_digest)
        is_free_value: bool
        if candidate.is_free is True:
            is_free_value = True
        elif candidate.is_free is False:
            is_free_value = False
        else:
            is_free_value = bool(
                candidate.ticket_price_min == 0
                and (candidate.ticket_price_max in (0, None))
            )
        # Seed with excerpt/title only; never publish full source_text verbatim as a fallback.
        # Full source is preserved separately in `event.source_text`/`event_source`.
        description_value = (clean_raw_excerpt or clean_title or "").strip()

        bundled_facts: list[str] | None = None
        bundled_digest: str | None = None
        bundled_desc: str | None = None
        bundled_title: str | None = None
        bundled_short: str | None = None
        try:
            if llm_create_bundle is not None:
                bundled = llm_create_bundle
            else:
                bundled = await _llm_create_description_facts_and_digest(
                    candidate,
                    clean_title=clean_title,
                    clean_source_text=clean_source_text,
                    clean_raw_excerpt=clean_raw_excerpt,
                    normalized_event_type=normalized_event_type,
                )
        except Exception:  # pragma: no cover - provider failures
            bundled = None
        if isinstance(bundled, dict):
            bundled_title_raw = bundled.get("title")
            if isinstance(bundled_title_raw, str) and bundled_title_raw.strip():
                # Non-semantic cleanup: some models return JSON-ish `\"...\"` fragments.
                t = (bundled_title_raw or "").strip().replace("\\\\\"", "\"").replace("\\\"", "\"").strip()
                t = _strip_private_use(t) or t
                if t:
                    bundled_title = t
            bundled_desc_raw = bundled.get("description")
            if isinstance(bundled_desc_raw, str) and bundled_desc_raw.strip():
                bundled_desc = bundled_desc_raw.strip()
            bundled_digest = _clean_search_digest(bundled.get("search_digest"))
            bundled_short = _clean_short_description(bundled.get("short_description"))
            if bundled_short and not _is_short_description_acceptable(
                bundled_short, min_words=12, max_words=16
            ):
                bundled_short = None
            raw_facts_any = bundled.get("facts")
            raw_facts: list[str] = []
            if isinstance(raw_facts_any, list):
                for it in raw_facts_any:
                    raw_facts.append(str(it or ""))
            bundled_facts_out: list[str] = []
            seen_fact_keys: set[str] = set()
            for it in raw_facts:
                cleaned = _normalize_fact_item(str(it or ""), limit=180)
                if not cleaned:
                    continue
                key = cleaned.casefold()
                if key in seen_fact_keys:
                    continue
                seen_fact_keys.add(key)
                bundled_facts_out.append(cleaned)
                if len(bundled_facts_out) >= 18:
                    break
            bundled_facts = bundled_facts_out

        # Bot/manual sources should keep operator-provided titles as-is.
        if (candidate.source_type or "").strip().lower() in {"bot"}:
            bundled_title = None

        # Guard LLM title proposals on create: avoid creating an event with an unrelated title,
        # which later breaks matching and can lead to false duplicates or cross-event merges.
        if bundled_title:
            candidate_title_weak = _is_candidate_title_weak_for_llm_override(
                clean_title,
                candidate=candidate,
                normalized_event_type=normalized_event_type,
            )
            proposed_title_grounded = _is_title_grounded_in_candidate_sources(
                bundled_title,
                candidate,
            )
            if not proposed_title_grounded:
                logger.warning(
                    "smart_update.create_title_rejected reason=ungrounded_title source_type=%s source_url=%s candidate_title=%s proposed_title=%s",
                    candidate.source_type,
                    candidate.source_url,
                    _clip_title(clean_title),
                    _clip_title(bundled_title),
                )
                text_filter_facts.append(
                    f"Заголовок отклонён: {clean_title} -> {bundled_title} (причина: ungrounded_title)"
                )
                bundled_title = None
            elif (
                _title_has_meaningful_tokens(clean_title)
                and (not _titles_look_related(bundled_title, clean_title))
                and (not candidate_title_weak)
            ):
                logger.warning(
                    "smart_update.create_title_rejected source_type=%s source_url=%s candidate_title=%s proposed_title=%s",
                    candidate.source_type,
                    candidate.source_url,
                    _clip_title(clean_title),
                    _clip_title(bundled_title),
                )
                text_filter_facts.append(
                    f"Заголовок отклонён: {clean_title} -> {bundled_title} (причина: semantic_title_mismatch)"
                )
                bundled_title = None

        final_title = bundled_title or clean_title
        final_title = _strip_private_use(final_title) or final_title
        final_title = re.sub(r"\s+", " ", (final_title or "").strip())
        # Safety-net: Telegraph + Telegram UI behave poorly with extremely long titles.
        final_title = _clip_title(final_title, 160) or clean_title

        fact_first_used = False
        if SMART_UPDATE_FACT_FIRST and not SMART_UPDATE_LLM_DISABLED:
            fact_first_facts = list(bundled_facts or [])
            if not fact_first_facts:
                try:
                    fact_first_facts = await _llm_extract_candidate_facts(candidate)
                except Exception:  # pragma: no cover - provider failures
                    fact_first_facts = []
                if fact_first_facts:
                    bundled_facts = fact_first_facts
            facts_text_clean = _facts_text_clean_from_facts(
                fact_first_facts,
                anchors=[
                    candidate.date or "",
                    candidate.time or "",
                    candidate.city or "",
                    candidate.location_name or "",
                    candidate.location_address or "",
                ],
            )
            if facts_text_clean:
                try:
                    ff_desc = await _llm_fact_first_description_md(
                        title=final_title,
                        event_type=normalized_event_type or candidate.event_type,
                        facts_text_clean=facts_text_clean,
                        anchors=[
                            candidate.date or "",
                            candidate.time or "",
                            candidate.city or "",
                            candidate.location_name or "",
                            candidate.location_address or "",
                        ],
                        label="create",
                    )
                except Exception:  # pragma: no cover - provider failures
                    ff_desc = None
                if ff_desc:
                    description_value = ff_desc
                    fact_first_used = True
                    # Keep canonical facts clean: narrative facts only (anchors are logged separately).
                    bundled_facts = facts_text_clean

        if not fact_first_used:
            if bundled_desc:
                description_value = bundled_desc
            else:
                try:
                    rewritten = await _rewrite_description_journalistic(candidate)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("smart_update: description rewrite failed", exc_info=True)
                    rewritten = None
                if rewritten:
                    description_value = rewritten
            # If the model returned near-verbatim source text, force a second-pass strict rewrite.
            if _description_too_verbatim(description_value, source_text=clean_source_text):
                logger.warning(
                    "smart_update: description too verbatim; forcing strict rewrite source_type=%s source_url=%s",
                    candidate.source_type,
                    candidate.source_url,
                )
                try:
                    rewritten_strict = await _rewrite_description_journalistic(
                        candidate,
                        strict_nonverbatim=True,
                    )
                except Exception:  # pragma: no cover - provider failures
                    rewritten_strict = None
                if rewritten_strict and not _description_too_verbatim(
                    rewritten_strict, source_text=clean_source_text
                ):
                    description_value = rewritten_strict
            if _description_has_foreign_schedule_noise(
                description_value,
                event_date=candidate.date,
                end_date=candidate.end_date,
                event_title=candidate.title,
            ):
                description_value = _strip_foreign_schedule_noise(
                    description_value,
                    event_date=candidate.date,
                    end_date=candidate.end_date,
                    event_title=candidate.title,
                ) or description_value
            description_value = _dedupe_description(description_value) or description_value
            description_value = _normalize_plaintext_paragraphs(description_value) or description_value
            description_value = _promote_review_bullets_to_blockquotes(description_value) or description_value
            description_value = _promote_first_person_quotes_to_blockquotes(description_value) or description_value
            description_value = _promote_inline_quoted_direct_speech_to_blockquotes(description_value) or description_value
            description_value = _drop_reported_speech_duplicates(description_value) or description_value
            description_value = _normalize_blockquote_markers(description_value) or description_value
            description_value = _append_missing_scene_hint(
                description=description_value, source_text=clean_source_text
            ) or description_value
            description_value = (
                _sanitize_description_output(
                    description_value,
                    source_text=clean_source_text or clean_raw_excerpt or candidate.source_text,
                )
                or description_value
            )
            description_value = (
                _append_missing_small_list(
                    description=description_value,
                    source_text=clean_source_text,
                    source_type=candidate.source_type,
                )
                or description_value
            )
            description_value = _normalize_plaintext_paragraphs(description_value) or description_value
            description_value = _ensure_minimal_description_headings(description_value) or description_value
        else:
            # Fact-first output should not be "topped up" with source snippets; keep it strictly fact-driven.
            description_value = _dedupe_description(description_value) or description_value
            description_value = _normalize_plaintext_paragraphs(description_value) or description_value
            description_value = _promote_review_bullets_to_blockquotes(description_value) or description_value
            description_value = _normalize_blockquote_markers(description_value) or description_value
            description_value = (
                _sanitize_description_output(
                    description_value,
                    source_text=clean_source_text or clean_raw_excerpt or candidate.source_text,
                )
                or description_value
            )
            description_value = _ensure_minimal_description_headings(description_value) or description_value
        if _has_overlong_paragraph(description_value, limit=850):
            try:
                reflown = await _llm_reflow_description_paragraphs(description_value)
            except Exception:  # pragma: no cover - provider failures
                reflown = None
            if reflown:
                reflown = _normalize_plaintext_paragraphs(reflown) or reflown
                reflown = _normalize_blockquote_markers(reflown) or reflown
                reflown = _fix_broken_initial_paragraph_splits(reflown) or reflown
                reflown = (
                    _sanitize_description_output(
                        reflown,
                        source_text=clean_source_text or clean_raw_excerpt or candidate.source_text,
                    )
                    or reflown
                )
                description_value = reflown

        # Guard: sometimes create-time LLM output becomes an over-compressed digest even when
        # the source text (or poster OCR) is rich. This makes Telegraph pages look like they
        # have "no main text". Do a second-pass full rewrite when we can.
        try:
            event_stub = Event(
                title=clean_title,
                description=description_value or "",
                date=candidate.date or "",
                time=candidate.time or "",
                end_date=candidate.end_date,
                location_name=candidate.location_name or "",
                location_address=candidate.location_address,
                city=candidate.city,
                ticket_link=candidate.ticket_link,
                ticket_status=candidate.ticket_status,
                is_free=bool(is_free_value),
                event_type=normalized_event_type or candidate.event_type,
                festival=candidate.festival,
                source_text=clean_source_text or "",
                source_texts=[
                    t
                    for t in [clean_source_text, (clean_raw_excerpt or "")]
                    if (t or "").strip()
                ],
                photo_urls=[],
            )
            rich_base = _pick_richest_source_text_for_description(event_stub, candidate)
            base_len = len(rich_base)
            try:
                max_expected = _estimate_description_budget_chars(
                    source_type=candidate.source_type,
                    source_text=clean_source_text,
                    raw_excerpt=clean_raw_excerpt,
                    poster_texts=[p.ocr_text for p in candidate.posters if (p.ocr_text or "").strip()],
                )
            except Exception:
                max_expected = 0
            # Create-time threshold is intentionally softer than merge-time: we only need
            # to avoid "empty-looking" Telegraph pages, not match the full source length.
            # Still, avoid publishing "title-only" or ultra-short bodies when the source is meaningful.
            if base_len < 260:
                min_expected = 0
            elif base_len < 700:
                min_expected = max(220, int(base_len * 0.30))
            else:
                min_expected = max(350, int(base_len * 0.25))
        except Exception:  # pragma: no cover - defensive
            event_stub = None
            min_expected = 0
            max_expected = 0
        too_verbatim = bool(
            rich_base
            and _description_too_verbatim(description_value, source_text=rich_base)
        )
        desc_len_now = len((description_value or "").strip())
        need_full = bool(min_expected and desc_len_now < min_expected) or too_verbatim
        if need_full and not fact_first_used:
            try:
                rewritten_full = await _rewrite_description_full_from_sources(
                    event_stub, candidate  # type: ignore[arg-type]
                )
            except Exception:  # pragma: no cover - provider failures
                rewritten_full = None
            if rewritten_full and (not too_verbatim) and len(rewritten_full) >= int(min_expected * 0.85):
                description_value = rewritten_full
                text_filter_facts.append("Описание расширено: заменён слишком короткий дайджест")
                logger.info(
                    "smart_update.create_description_rewrite_full source_type=%s source_url=%s desc_len=%d min_expected=%d",
                    candidate.source_type,
                    candidate.source_url,
                    len((rewritten_full or "").strip()),
                    int(min_expected),
                )
            elif rewritten_full and too_verbatim and not _description_too_verbatim(rewritten_full, source_text=rich_base):
                description_value = rewritten_full
                text_filter_facts.append("Описание перезаписано: убрано дословное копирование источника")
                logger.info(
                    "smart_update.create_description_rewrite_full_nonverbatim source_type=%s source_url=%s desc_len=%d",
                    candidate.source_type,
                    candidate.source_url,
                    len((rewritten_full or "").strip()),
                )
            elif too_verbatim and _description_too_verbatim(description_value, source_text=rich_base):
                # Last resort: do not publish the raw source verbatim when the LLM is unavailable.
                # Keep a short neutral snippet; full source remains in `event.source_text`.
                fallback = (clean_raw_excerpt or "").strip()
                if not fallback:
                    try:
                        fallback = _fallback_digest_from_description(rich_base) or ""
                    except Exception:
                        fallback = ""
                fallback = (fallback or "").strip()
                if fallback:
                    description_value = _clip(fallback, 520)
                    text_filter_facts.append("Описание сокращено: LLM недоступна, убран дословный текст источника")
                else:
                    description_value = ""
            elif min_expected and not rewritten_full and len((description_value or "").strip()) < max(120, int(min_expected * 0.6)):
                # Another last resort: if we couldn't expand the description (LLM down),
                # avoid a "nearly empty" Telegraph body by using a short excerpt/digest.
                fallback = (clean_raw_excerpt or "").strip()
                if not fallback:
                    try:
                        fallback = _fallback_digest_from_description(rich_base) or ""
                    except Exception:
                        fallback = ""
                fallback = (fallback or "").strip()
                if fallback:
                    description_value = _clip(fallback, 520)
                    text_filter_facts.append("Описание заменено: LLM недоступна, использован краткий фрагмент")

        # Guard: avoid over-expanding short sources. If the description is much longer than
        # the total available source volume, force an LLM-only shrinking pass.
        try:
            max_expected_i = int(max_expected or 0)
        except Exception:
            max_expected_i = 0
        desc_len = len((description_value or "").strip())
        if (not fact_first_used) and max_expected_i and desc_len and desc_len > max_expected_i:
            shrunk = None
            try:
                shrunk = await _llm_shrink_description_to_budget(
                    source_type=candidate.source_type,
                    source_url=candidate.source_url,
                    description=str(description_value or ""),
                    source_text=rich_base or clean_source_text,
                    facts=bundled_facts or None,
                    max_chars=max_expected_i,
                )
            except Exception:  # pragma: no cover - provider failures
                shrunk = None
            if shrunk:
                description_value = shrunk
                text_filter_facts.append("Описание сокращено: приведено к объёму источников")
                logger.info(
                    "smart_update.create_description_shrunk source_type=%s source_url=%s desc_len=%d budget=%d",
                    candidate.source_type,
                    candidate.source_url,
                    len((description_value or "").strip()),
                    int(max_expected_i),
                )
            else:
                # If LLM isn't available, prefer a short deterministic digest from the source
                # rather than publishing a long hallucination-prone narrative.
                try:
                    fallback = _fallback_digest_from_description(rich_base) or ""
                except Exception:
                    fallback = ""
                fallback = fallback.strip()
                if fallback and len(fallback) <= int(max_expected_i * 1.15):
                    description_value = fallback
                    text_filter_facts.append("Описание сокращено: использован краткий дайджест источника")
        description_value = _clip(description_value, SMART_UPDATE_DESCRIPTION_MAX_CHARS) if description_value else ""

        # Extract atomic facts for global de-duplication + operator log.
        extracted_facts: list[str] = bundled_facts or []
        if not extracted_facts:
            try:
                # Facts must come from the SOURCE, not from the rewritten description (which is also LLM output).
                extracted_facts = await _llm_extract_candidate_facts(candidate)
            except Exception:  # pragma: no cover - defensive
                extracted_facts = []

        # Build/refresh digest from the final description (Telegram posts typically don't provide one).
        if bundled_digest:
            normalized_digest = bundled_digest
        else:
            try:
                llm_digest = await _llm_build_search_digest(
                    title=final_title,
                    description=description_value,
                    event_type=normalized_event_type or candidate.event_type,
                )
            except Exception:
                llm_digest = None
            if llm_digest:
                normalized_digest = llm_digest
        if not normalized_digest:
            normalized_digest = _fallback_digest_from_description(description_value)
        if not normalized_digest and rich_base:
            normalized_digest = _fallback_digest_from_description(rich_base)
        final_short = bundled_short
        if not final_short:
            try:
                final_short = await _llm_build_short_description(
                    title=final_title,
                    description=description_value or clean_source_text,
                    event_type=normalized_event_type or candidate.event_type,
                )
            except Exception:
                final_short = None
        if not final_short:
            candidate_short = _clean_short_description(candidate.raw_excerpt)
            if _is_short_description_acceptable(candidate_short, min_words=12, max_words=16):
                final_short = candidate_short
        if not final_short:
            final_short = _fallback_short_description_from_text(description_value or clean_source_text)
        new_event = Event(
            title=final_title,
            description=description_value,
            short_description=final_short,
            festival=candidate.festival,
            date=candidate.date or "",
            time=candidate.time or "",
            time_is_default=bool(candidate.time_is_default and (candidate.time or "").strip()),
            location_name=candidate.location_name or "",
            location_address=candidate.location_address,
            city=candidate.city or None,
            ticket_price_min=candidate.ticket_price_min,
            ticket_price_max=candidate.ticket_price_max,
            ticket_link=candidate.ticket_link,
            ticket_status=candidate.ticket_status,
            ticket_trust_level=candidate.trust_level,
            event_type=normalized_event_type or candidate.event_type,
            emoji=candidate.emoji,
            end_date=candidate.end_date,
            end_date_is_inferred=bool(candidate.end_date_is_inferred),
            is_free=is_free_value,
            pushkin_card=bool(candidate.pushkin_card),
            silent=bool(force_silent_due_to_date_risk),
            source_text=clean_source_text or "",
            source_texts=[clean_source_text] if clean_source_text else [],
            source_post_url=candidate.source_url if _is_http_url(candidate.source_url) else None,
            source_chat_id=candidate.source_chat_id,
            source_message_id=candidate.source_message_id,
            creator_id=candidate.creator_id,
            search_digest=normalized_digest,
            photo_urls=[
                (p.supabase_url or p.catbox_url)
                for p in candidate.posters
                if (p.supabase_url or p.catbox_url)
            ],
            photo_count=len(
                [p for p in candidate.posters if (p.supabase_url or p.catbox_url)]
            ),
        )
        if candidate.source_url and _is_vk_wall_url(candidate.source_url):
            new_event.source_vk_post_url = candidate.source_url

        async with db.get_session() as session:
            session.add(new_event)
            await session.commit()
            await session.refresh(new_event)

            added_posters, added_poster_urls, preview_invalidated, pruned_posters, _photo_urls_changed = await _apply_posters(
                session,
                new_event.id,
                candidate.posters,
                poster_scope_hashes=candidate.poster_scope_hashes,
                event_title=final_title,
            )
            added_sources, _same_source = await _ensure_event_source(
                session, new_event.id, candidate
            )
            await _enqueue_ticket_sites_queue(session, event_id=int(new_event.id or 0))
            if candidate.source_text:
                await _sync_source_texts(session, new_event)
            await session.flush()
            initial_records: list[tuple[str, str]] = []
            for fact in _initial_added_facts(candidate):
                initial_records.append((fact, "added"))
            for fact in (extracted_facts or [])[:18]:
                initial_records.append((fact, "added"))
            note_lines: list[str] = []
            note_lines.extend((queue_notes or [])[:6])
            note_lines.extend((text_filter_facts or [])[:2])
            note_lines.extend((poster_filter_facts or [])[:3])
            for fact in _dedupe_source_facts(note_lines):
                initial_records.append((fact, "note"))
            for url in (added_poster_urls or [])[:3]:
                initial_records.append((f"Добавлена афиша: {url}", "added"))
            if pruned_posters:
                initial_records.append((f"Удалены лишние афиши: {pruned_posters}", "note"))
            if preview_invalidated:
                initial_records.append(("3D-превью сброшено: изменились иллюстрации", "note"))
            if initial_records:
                await _record_source_facts(session, new_event.id, candidate, initial_records)
            await session.commit()

        try:
            await _apply_holiday_festival_mapping(db, new_event.id)
        except Exception:
            logger.warning(
                "smart_update: holiday mapping failed for event %s",
                new_event.id,
                exc_info=True,
            )

        await _classify_topics(db, new_event.id)

        linked_refresh_ids: list[int] = []
        try:
            from linked_events import recompute_linked_event_ids

            lr = await recompute_linked_event_ids(db, int(new_event.id or 0))
            linked_refresh_ids = [
                int(x)
                for x in (lr.changed_event_ids or [])
                if int(x) and int(x) != int(new_event.id or 0)
            ]
        except Exception:
            logger.warning(
                "smart_update: linked events recompute failed for event %s",
                new_event.id,
                exc_info=True,
            )
            linked_refresh_ids = []

        if schedule_tasks and linked_refresh_ids:
            # Only refresh Telegraph pages for linked occurrences: this is user-facing ("Другие даты"),
            # but does not require VK sync / month-page rebuilds.
            try:
                from main import JobTask, enqueue_job

                for rid in linked_refresh_ids[:80]:
                    await enqueue_job(db, int(rid), JobTask.telegraph_build, depends_on=None)
            except Exception:
                logger.warning(
                    "smart_update: failed to enqueue linked telegraph refresh for event %s",
                    new_event.id,
                    exc_info=True,
                )

        if schedule_tasks:
            try:
                from main import schedule_event_update_tasks
                async with db.get_session() as session:
                    refreshed = await session.get(Event, new_event.id)
                if refreshed:
                    await schedule_event_update_tasks(db, refreshed, **(schedule_kwargs or {}))
            except Exception:
                logger.warning("smart_update: schedule/update failed for event %s", new_event.id, exc_info=True)

        logger.info(
            "smart_update.created event_id=%s added_posters=%d added_sources=%s reason=%s",
            new_event.id,
            added_posters,
            int(bool(added_sources)),
            match_reason if "match_reason" in locals() else None,
        )
        return SmartUpdateResult(
            status="created",
            event_id=new_event.id,
            created=True,
            merged=False,
            added_posters=added_posters,
            added_sources=added_sources,
            reason=match_reason if "match_reason" in locals() else None,
            queue_notes=list(queue_notes or []),
        )

    # Merge path
    existing = match_event
    existing_start, existing_end = _event_date_range(existing)
    is_canonical_site = str(candidate.source_type or "").startswith("parser:")
    conflicting: dict[str, Any] = {}
    # By default we keep anchor fields stable; for canonical site/parser imports we allow
    # correcting anchors and therefore do not treat conflicts as "do not use".
    if not is_canonical_site:
        if existing_start and cand_start and existing_start != cand_start:
            conflicting["date"] = candidate.date
        cand_time_anchor = _candidate_anchor_time(candidate, is_canonical_site=False)
        existing_time_anchor = _event_anchor_time(existing)
        if _has_explicit_time_conflict(existing_time_anchor, cand_time_anchor):
            conflicting["time"] = candidate.time
        if existing.location_name and candidate.location_name and not _location_matches(existing.location_name, candidate.location_name):
            conflicting["location_name"] = candidate.location_name
        if existing.location_address and candidate.location_address and existing.location_address != candidate.location_address:
            conflicting["location_address"] = candidate.location_address
        if existing_end and cand_end and existing_end != cand_end:
            long_event = _is_long_event_type_value(
                getattr(existing, "event_type", None) or candidate.event_type
            )
            # For long-running events (e.g. exhibitions/fairs), later end_date is a
            # normal update, not an anchor conflict.
            if (not long_event) or (cand_end < existing_end):
                conflicting["end_date"] = candidate.end_date

    new_hashes = _poster_hashes(candidate.posters)
    existing_hashes = {p.poster_hash for p in posters_map.get(existing.id or 0, [])}
    has_new_posters = bool(new_hashes - existing_hashes)
    has_new_text = _candidate_has_new_text(candidate, existing)
    needs_schedule_cleanup = _description_has_foreign_schedule_noise(
        getattr(existing, "description", None),
        event_date=getattr(existing, "date", None),
        end_date=getattr(existing, "end_date", None),
        event_title=getattr(existing, "title", None),
    )

    ticket_changes_needed = any(
        [
            candidate.ticket_link and candidate.ticket_link != existing.ticket_link,
            candidate.ticket_price_min is not None and candidate.ticket_price_min != existing.ticket_price_min,
            candidate.ticket_price_max is not None and candidate.ticket_price_max != existing.ticket_price_max,
            candidate.ticket_status and candidate.ticket_status != getattr(existing, "ticket_status", None),
        ]
    )

    should_merge = (
        has_new_posters
        or has_new_text
        or needs_schedule_cleanup
        or ticket_changes_needed
    )

    added_facts: list[str] = []
    duplicate_facts: list[str] = []
    skipped_conflicts: list[str] = []
    conflict_facts: list[str] = []
    updated_fields = False
    updated_keys: list[str] = []
    skip_topic_reclassify = False
    merge_digest_from_llm: str | None = None
    merge_fact_first_used = False

    async with db.get_session() as session:
        event_db = await session.get(Event, existing.id)
        if not event_db:
            return SmartUpdateResult(status="error", reason="event_missing")
        before_description = event_db.description or ""
        # Self-heal legacy snapshot leaks (e.g. "Текст до Smart Update ...") that were
        # accidentally merged into the public description in older versions.
        cleaned_leak = _drop_legacy_leak_from_description(before_description)
        if cleaned_leak and cleaned_leak != before_description:
            event_db.description = cleaned_leak
            before_description = cleaned_leak
            updated_fields = True
            if "description" not in updated_keys:
                updated_keys.append("description")
            note = "Текст очищен: удалена служебная вставка legacy snapshot"
            if note not in text_filter_facts:
                text_filter_facts.append(note)
        existing_trusts = [
            str(r[0]).strip()
            for r in (
                await session.execute(
                    select(EventSource.trust_level).where(
                        EventSource.event_id == int(event_db.id or 0)
                    )
                )
            ).all()
            if (r and str(r[0] or "").strip())
        ]
        event_trust_level, event_trust_pr = _max_trust_level(existing_trusts)
        candidate_trust_pr = _trust_priority(candidate.trust_level)

        # Fill placeholder/missing time from any matched source (TG/VK/etc.), not only parser sources.
        # This prevents duplicate creation like: existing time=00:00 (legacy placeholder) + new source brings 19:00.
        if not is_canonical_site:
            ct_anchor = _candidate_anchor_time(candidate, is_canonical_site=False)
            et_anchor = _event_anchor_time(event_db)
            if (
                ct_anchor
                and bool(getattr(event_db, "time_is_default", False))
                and ct_anchor == (getattr(event_db, "time", "") or "").strip()
            ):
                event_db.time_is_default = False
                updated_fields = True
                if "time_is_default" not in updated_keys:
                    updated_keys.append("time_is_default")
            if ct_anchor and (not et_anchor) and ct_anchor != (getattr(event_db, "time", "") or "").strip():
                event_db.time = ct_anchor
                event_db.time_is_default = False
                updated_fields = True
                if "time" not in updated_keys:
                    updated_keys.append("time")
                conflicting.pop("time", None)

        if is_canonical_site:
            # Canonical site/parser source: allow correcting anchors on an existing event.
            # This makes Telegram-first -> /parse merge converge to the site truth.
            if candidate.date and candidate.date != (event_db.date or ""):
                event_db.date = candidate.date
                updated_fields = True
                updated_keys.append("date")
            if _apply_event_end_date(
                event_db,
                end_date=candidate.end_date,
                inferred=bool(candidate.end_date_is_inferred),
                updated_keys=updated_keys,
            ):
                updated_fields = True
            if candidate.time and candidate.time.strip() and candidate.time.strip() != (event_db.time or "").strip():
                event_db.time = candidate.time.strip()
                event_db.time_is_default = False
                updated_fields = True
                updated_keys.append("time")
            elif (
                candidate.time
                and candidate.time.strip()
                and bool(getattr(event_db, "time_is_default", False))
                and candidate.time.strip() == (event_db.time or "").strip()
            ):
                event_db.time_is_default = False
                updated_fields = True
                if "time_is_default" not in updated_keys:
                    updated_keys.append("time_is_default")
            if candidate.location_name and not _location_matches(event_db.location_name, candidate.location_name):
                event_db.location_name = candidate.location_name
                updated_fields = True
                updated_keys.append("location_name")
            if (
                candidate.location_address
                and candidate.location_address.strip()
                and candidate.location_address.strip() != (event_db.location_address or "").strip()
            ):
                event_db.location_address = candidate.location_address.strip()
                updated_fields = True
                updated_keys.append("location_address")

        # Operator-entered sources are allowed to корректировать title even if the
        # candidate doesn't bring enough new text/posters for LLM merge.
        cand_title = clean_title
        if candidate.source_type in ("bot", "manual") and cand_title and cand_title != event_db.title:
            event_db.title = cand_title
            updated_fields = True
            updated_keys.append("title")

        # Long-running events (e.g. exhibitions/fairs) may legitimately extend the
        # closing date across sources. Allow end_date extension by trust.
        if (
            (not is_canonical_site)
            and candidate.end_date
            and _is_long_event_type_value(
                getattr(event_db, "event_type", None) or candidate.event_type
            )
        ):
            cand_end_iso = _parse_iso_date(candidate.end_date)
            cur_end_iso = _parse_iso_date(getattr(event_db, "end_date", None))
            if cand_end_iso and (not cur_end_iso or cand_end_iso > cur_end_iso):
                if candidate_trust_pr >= event_trust_pr:
                    if _apply_event_end_date(
                        event_db,
                        end_date=cand_end_iso.isoformat(),
                        inferred=bool(candidate.end_date_is_inferred),
                        updated_keys=updated_keys,
                    ):
                        updated_fields = True
                    if "end_date" in conflicting:
                        conflicting.pop("end_date", None)
                else:
                    skipped_conflicts.append(
                        f"Дата окончания: {getattr(event_db, 'end_date', None)} -> {candidate.end_date} "
                        f"(выбран: event_before по trust {event_trust_level or 'medium'}>{candidate.trust_level or 'medium'})"
                    )

        if (
            candidate.end_date
            and candidate.end_date == getattr(event_db, "end_date", None)
            and _apply_event_end_date(
                event_db,
                end_date=candidate.end_date,
                inferred=bool(candidate.end_date_is_inferred),
                updated_keys=updated_keys,
            )
        ):
            updated_fields = True

        if should_merge:
            before_description = event_db.description or ""
            posters_texts = [p.ocr_text for p in posters_map.get(existing.id or 0, []) if p.ocr_text]
            cleanup_only = (
                needs_schedule_cleanup
                and (not has_new_posters)
                and (not has_new_text)
                and (not ticket_changes_needed)
            )
            if cleanup_only:
                cleaned = _strip_foreign_schedule_noise(
                    before_description,
                    event_date=event_db.date,
                    end_date=event_db.end_date,
                    event_title=event_db.title,
                )
                if cleaned and cleaned != before_description:
                    cleaned = _normalize_plaintext_paragraphs(cleaned) or cleaned
                    cleaned = _promote_review_bullets_to_blockquotes(cleaned) or cleaned
                    cleaned = _promote_first_person_quotes_to_blockquotes(cleaned) or cleaned
                    cleaned = _promote_inline_quoted_direct_speech_to_blockquotes(cleaned) or cleaned
                    cleaned = _drop_reported_speech_duplicates(cleaned) or cleaned
                    cleaned = _normalize_blockquote_markers(cleaned) or cleaned
                    cleaned = _append_missing_scene_hint(
                        description=cleaned, source_text=candidate.source_text
                    ) or cleaned
                    cleaned = (
                        _sanitize_description_output(
                            cleaned,
                            source_text=candidate.source_text,
                        )
                        or cleaned
                    )
                    cleaned = _ensure_minimal_description_headings(cleaned) or cleaned
                    event_db.description = _clip(cleaned, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                    updated_fields = True
                    updated_keys.append("description")
                    note = "Текст очищен: убраны строки расписания других дат"
                    if note not in text_filter_facts:
                        text_filter_facts.append(note)
                    skip_topic_reclassify = True
            else:
                quote_candidates = _extract_quote_candidates(
                    _strip_promo_lines(candidate.source_text) or candidate.source_text,
                    max_items=2,
                )
                rows = (
                    await session.execute(
                        select(EventSourceFact.fact)
                        .join(EventSource, EventSourceFact.source_id == EventSource.id)
                        .where(
                            EventSourceFact.event_id == int(event_db.id or 0),
                            # `duplicate` is still a true, already-known fact. Include it to keep
                            # canonical facts stable across re-processing the same source URL.
                            EventSourceFact.status.in_(("added", "duplicate")),
                            # Never feed legacy snapshots into the merge facts:
                            # they are not per-source "added facts" and can pollute Telegraph text.
                            EventSource.source_type != "legacy",
                        )
                    )
                ).all()
                facts_before_list = _dedupe_source_facts(
                    [str(r[0]).strip() for r in (rows or []) if (r and str(r[0] or "").strip())]
                )[:80]
                legacy_facts_seed: list[str] = []
                # Keep a compact snapshot for operator audit/debug, but do not use it as facts_before.
                # Also: if an older run created only a legacy note (without extracted facts), we still
                # backfill baseline legacy facts later from the stored legacy snapshot.
                try:
                    before_texts = list(getattr(event_db, "source_texts", None) or [])
                    legacy_facts_seed = await _ensure_legacy_description_fact(
                        session,
                        event=event_db,
                        legacy_description=before_description,
                    )
                    after_texts = list(getattr(event_db, "source_texts", None) or [])
                    if after_texts != before_texts:
                        updated_fields = True
                        if "source_texts" not in updated_keys:
                            updated_keys.append("source_texts")
                except Exception:
                    logger.warning(
                        "smart_update: legacy description snapshot store failed",
                        exc_info=True,
                    )
                    legacy_facts_seed = []
                director_name_hint = _extract_director_name_hint(
                    candidate_text=_strip_promo_lines(candidate.source_text) or candidate.source_text,
                    facts_before=facts_before_list,
                )
                merge_data = await _llm_merge_event(
                    candidate,
                    event_db,
                    conflicting_anchor_fields=conflicting,
                    poster_texts=posters_texts,
                    facts_before=facts_before_list,
                    event_trust_level=event_trust_level,
                    candidate_trust_level=candidate.trust_level,
                )
                if merge_data:
                    merge_digest_from_llm = _clean_search_digest(merge_data.get("search_digest"))
                    deterministic_skipped_conflicts = list(skipped_conflicts)
                    added_facts = list(merge_data.get("added_facts") or [])
                    duplicate_facts = list(merge_data.get("duplicate_facts") or [])
                    conflict_facts = list(merge_data.get("conflict_facts") or [])
                    llm_skipped_conflicts = list(merge_data.get("skipped_conflicts") or [])
                    skipped_conflicts = []
                    for item in deterministic_skipped_conflicts + llm_skipped_conflicts:
                        text = str(item or "").strip()
                        if not text or text in skipped_conflicts:
                            continue
                        skipped_conflicts.append(text)

                    title = merge_data.get("title")
                    description = merge_data.get("description")
                    clean_title = _strip_private_use(title) if isinstance(title, str) else None
                    if clean_title:
                        clean_title = clean_title.replace("\\\\\"", "\"").replace("\\\"", "\"").strip()
                        clean_title = re.sub(r"\s+", " ", clean_title).strip()
                        clean_title = _clip_title(clean_title, 160) or clean_title
                    clean_description = (
                        (
                            _strip_private_use(description) or description
                        )
                        if isinstance(description, str)
                        else None
                    )
                    if clean_title:
                        if clean_title.strip() == (event_db.title or "").strip():
                            # No-op title, keep as-is without recording semantic mismatch.
                            pass
                        elif _is_merge_title_update_allowed(
                            proposed_title=clean_title,
                            candidate_title=candidate.title,
                            existing_title=event_db.title,
                            is_canonical_site=is_canonical_site,
                        ):
                            event_db.title = clean_title
                            updated_fields = True
                            updated_keys.append("title")
                        else:
                            relaxed_allowed = False
                            old_title = str(getattr(event_db, "title", "") or "").strip()
                            if (
                                (not is_canonical_site)
                                and _is_generic_title_event_type_venue(
                                    old_title,
                                    event_type=getattr(event_db, "event_type", None) or candidate.event_type,
                                    location_name=getattr(event_db, "location_name", None)
                                    or candidate.location_name,
                                    city=getattr(event_db, "city", None) or candidate.city,
                                )
                                and _is_title_grounded_in_candidate_sources(clean_title, candidate)
                            ):
                                relaxed_allowed = True

                            if relaxed_allowed:
                                event_db.title = clean_title
                                updated_fields = True
                                updated_keys.append("title")
                                logger.info(
                                    "smart_update.title_updated_from_generic event_id=%s source_type=%s source_url=%s old_title=%s new_title=%s",
                                    getattr(event_db, "id", None),
                                    candidate.source_type,
                                    candidate.source_url,
                                    _clip_title(old_title),
                                    _clip_title(clean_title),
                                )
                            else:
                                # Catastrophic merge guard: if the model proposes an unrelated title,
                                # abort this merge (do not record facts/sources) to avoid polluting
                                # an existing event with content from a different one.
                                if (
                                    _title_has_meaningful_tokens(clean_title)
                                    and _title_has_meaningful_tokens(candidate.title)
                                    and (not _titles_look_related(candidate.title, event_db.title))
                                    and (not _titles_look_related(clean_title, candidate.title))
                                    and (not _titles_look_related(clean_title, event_db.title))
                                    and (not is_canonical_site)
                                ):
                                    event_id = getattr(event_db, "id", None)
                                    await session.rollback()
                                    logger.warning(
                                        "smart_update.reject reason=incoherent_merge_title event_id=%s source_type=%s source_url=%s candidate_title=%s proposed_title=%s",
                                        event_id,
                                        candidate.source_type,
                                        candidate.source_url,
                                        _clip_title(candidate.title),
                                        _clip_title(clean_title),
                                    )
                                    return SmartUpdateResult(
                                        status="rejected_incoherent_merge",
                                        reason="incoherent_merge_title",
                                    )
                                skipped_conflicts.append(
                                    f"Заголовок отклонён: {event_db.title} -> {clean_title} "
                                    "(причина: semantic_title_mismatch)"
                                )
                                logger.warning(
                                    "smart_update.title_rejected event_id=%s candidate_title=%s "
                                    "existing_title=%s proposed_title=%s source_type=%s source_url=%s",
                                    getattr(event_db, "id", None),
                                    _clip_title(candidate.title),
                                    _clip_title(getattr(event_db, "title", None)),
                                    _clip_title(clean_title),
                                    candidate.source_type,
                                    candidate.source_url,
                                )

                    # Fact-first: override public narrative with a fresh build from canonical facts,
                    # ignoring the merged description text (which is derived from raw sources).
                    if SMART_UPDATE_FACT_FIRST and not SMART_UPDATE_LLM_DISABLED:
                        legacy_facts_for_fact_first: list[str] = list(legacy_facts_seed or [])
                        if not legacy_facts_for_fact_first:
                            try:
                                legacy_rows = (
                                    await session.execute(
                                        select(EventSourceFact.fact)
                                        .join(EventSource, EventSourceFact.source_id == EventSource.id)
                                        .where(
                                            EventSourceFact.event_id == int(event_db.id or 0),
                                            EventSourceFact.status.in_(("added", "duplicate")),
                                            EventSource.source_type == "legacy",
                                        )
                                    )
                                ).all()
                                legacy_facts_for_fact_first = _dedupe_source_facts(
                                    [
                                        str(r[0]).strip()
                                        for r in (legacy_rows or [])
                                        if (r and str(r[0] or "").strip())
                                    ]
                                )[:40]
                            except Exception:
                                legacy_facts_for_fact_first = []

                        canonical_facts = [
                            *(facts_before_list or []),
                            *[
                                str(f).strip()
                                for f in (added_facts or [])
                                if isinstance(f, str) and f.strip()
                            ],
                            *(legacy_facts_for_fact_first or []),
                        ]
                        facts_text_clean = _facts_text_clean_from_facts(
                            canonical_facts,
                            max_items=36,
                            anchors=[
                                getattr(event_db, "date", None) or candidate.date or "",
                                getattr(event_db, "time", None) or candidate.time or "",
                                getattr(event_db, "city", None) or candidate.city or "",
                                getattr(event_db, "location_name", None) or candidate.location_name or "",
                                getattr(event_db, "location_address", None) or candidate.location_address or "",
                            ],
                        )
                        if facts_text_clean:
                            try:
                                ff_desc = await _llm_fact_first_description_md(
                                    title=event_db.title,
                                    event_type=getattr(event_db, "event_type", None) or candidate.event_type,
                                    facts_text_clean=facts_text_clean,
                                    anchors=[
                                        getattr(event_db, "date", None) or candidate.date or "",
                                        getattr(event_db, "time", None) or candidate.time or "",
                                        getattr(event_db, "city", None) or candidate.city or "",
                                        getattr(event_db, "location_name", None) or candidate.location_name or "",
                                        getattr(event_db, "location_address", None)
                                        or candidate.location_address
                                        or "",
                                    ],
                                    label=f"merge:{int(getattr(event_db, 'id', 0) or 0)}",
                                )
                            except Exception:  # pragma: no cover - provider failures
                                ff_desc = None
                            if ff_desc:
                                # Minimal deterministic cleanup only (no "top-up" from sources).
                                cleaned_ff = _dedupe_description(ff_desc) or ff_desc
                                cleaned_ff = _normalize_plaintext_paragraphs(cleaned_ff) or cleaned_ff
                                cleaned_ff = _normalize_blockquote_markers(cleaned_ff) or cleaned_ff
                                cleaned_ff = (
                                    _sanitize_description_output(
                                        cleaned_ff,
                                        source_text=_pick_richest_source_text_for_description(event_db, candidate),
                                    )
                                    or cleaned_ff
                                )
                                cleaned_ff = _ensure_minimal_description_headings(cleaned_ff) or cleaned_ff
                                cleaned_ff = _clip(cleaned_ff, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                                current = (event_db.description or "").strip()
                                if cleaned_ff.strip() and cleaned_ff.strip() != current:
                                    event_db.description = cleaned_ff
                                    updated_fields = True
                                    if "description" not in updated_keys:
                                        updated_keys.append("description")
                                note = "Описание перегенерировано: fact-first"
                                if note not in text_filter_facts:
                                    text_filter_facts.append(note)
                                merge_fact_first_used = True
                                # Skip merge-description-driven cleanup below.
                                clean_description = None

                    if clean_description:
                        clean_description = _dedupe_description(clean_description) or clean_description
                        clean_description = _enforce_merge_non_shrinking_description(
                            before_description=before_description,
                            merged_description=clean_description,
                            candidate=candidate,
                            has_new_text=has_new_text,
                        )
                        clean_description = _strip_foreign_schedule_noise(
                            clean_description,
                            event_date=event_db.date,
                            end_date=event_db.end_date,
                            event_title=event_db.title,
                        ) or clean_description
                        clean_description = _normalize_plaintext_paragraphs(clean_description) or clean_description
                        # If we have rich source text (usually from site import) but the merge
                        # produced an over-compressed digest, do a second-pass rewrite via LLM.
                        # We do NOT fall back to verbatim source text: Telegraph text must be LLM-produced.
                        rich_fallback_used = False
                        min_expected = _min_expected_description_len_from_sources(event_db, candidate)
                        if min_expected and len(clean_description) < min_expected:
                            try:
                                rewritten_full = await _rewrite_description_full_from_sources(event_db, candidate)
                            except Exception:  # pragma: no cover - defensive
                                rewritten_full = None
                            if rewritten_full and len(rewritten_full) >= int(min_expected * 0.85):
                                clean_description = rewritten_full
                                rich_fallback_used = True
                        clean_description = _normalize_plaintext_paragraphs(clean_description) or clean_description
                        # NOTE: We intentionally do NOT append any sentences deterministically.
                        # If the model missed important details, that is an LLM quality issue and should
                        # be fixed via prompts/models, not by verbatim injection.
                        clean_description = (
                            _preserve_blockquotes_from_previous_description(
                                before_description=before_description,
                                merged_description=clean_description,
                                event_title=event_db.title,
                            )
                            or clean_description
                        )
                        clean_description = (
                            _promote_first_person_quotes_to_blockquotes(clean_description)
                            or clean_description
                        )
                        clean_description = (
                            _promote_inline_quoted_direct_speech_to_blockquotes(clean_description)
                            or clean_description
                        )
                        clean_description = _drop_reported_speech_duplicates(clean_description) or clean_description
                        clean_description = _normalize_blockquote_markers(clean_description) or clean_description
                        clean_description = (
                            _split_overlong_first_person_blockquotes(clean_description) or clean_description
                        )
                        clean_description = (
                            _fix_broken_initial_paragraph_splits(clean_description) or clean_description
                        )
                        # When we had to fall back to a rich verbatim source text because the merge
                        # was over-compressed, avoid aggressive paragraph de-duplication: it can
                        # accidentally collapse legitimately long source material (and re-trigger
                        # the "too short" issue we are fixing).
                        if not rich_fallback_used:
                            clean_description = (
                                _dedupe_paragraphs_preserving_formatting(clean_description) or clean_description
                            )
                        clean_description = _append_missing_scene_hint(
                            description=clean_description, source_text=candidate.source_text
                        ) or clean_description
                        clean_description = (
                            _sanitize_description_output(
                                clean_description,
                                source_text=_pick_richest_source_text_for_description(event_db, candidate),
                            )
                            or clean_description
                        )
                        rich_source_text = _pick_richest_source_text_for_description(event_db, candidate)
                        if _description_too_verbatim(clean_description, source_text=rich_source_text):
                            logger.warning(
                                "smart_update: merged description too verbatim; forcing full rewrite event_id=%s source_type=%s source_url=%s",
                                getattr(event_db, "id", None),
                                candidate.source_type,
                                candidate.source_url,
                            )
                            try:
                                rewritten_full = await _rewrite_description_full_from_sources(event_db, candidate)
                            except Exception:  # pragma: no cover
                                rewritten_full = None
                            if rewritten_full and not _description_too_verbatim(
                                rewritten_full, source_text=rich_source_text
                            ):
                                clean_description = rewritten_full
                        # If we have a director name and quotes in the text, make sure at least
                        # one quote contains the attribution inside the blockquote.
                        clean_description = _ensure_blockquote_has_attribution(
                            description=clean_description,
                            attribution_name=director_name_hint,
                        )
                        if _has_overlong_paragraph(clean_description, limit=850):
                            try:
                                reflown = await _llm_reflow_description_paragraphs(clean_description)
                            except Exception:  # pragma: no cover
                                reflown = None
                            if reflown:
                                reflown = _normalize_plaintext_paragraphs(reflown) or reflown
                                reflown = _normalize_blockquote_markers(reflown) or reflown
                                reflown = _fix_broken_initial_paragraph_splits(reflown) or reflown
                                reflown = (
                                    _sanitize_description_output(
                                        reflown,
                                        source_text=_pick_richest_source_text_for_description(event_db, candidate),
                                    )
                                    or reflown
                                )
                                clean_description = reflown
                        # Ensure we keep at least one detected direct quote as a blockquote.
                        if quote_candidates and not re.search(r"(?m)^>\s+", clean_description):
                            clean_description = await _ensure_direct_quote_blockquote(
                                description=clean_description,
                                quote_candidates=quote_candidates,
                                candidate_text=(
                                    _strip_promo_lines(candidate.source_text) or candidate.source_text
                                ),
                                facts_before=facts_before_list,
                                label="merge_quote_enforce",
                            )
                            clean_description = _normalize_plaintext_paragraphs(clean_description) or clean_description
                            clean_description = _normalize_blockquote_markers(clean_description) or clean_description
                            clean_description = _drop_reported_speech_duplicates(clean_description) or clean_description
                            clean_description = _ensure_blockquote_has_attribution(
                                description=clean_description,
                                attribution_name=director_name_hint,
                            )

                        # Ensure the narrative mentions short quoted slogan-like canonical facts.
                        canonical_facts = [
                            *(facts_before_list or []),
                            *[
                                str(f).strip()
                                for f in (added_facts or [])
                                if isinstance(f, str) and f.strip()
                            ],
                        ]
                        missing = _find_missing_facts_in_description(
                            description=clean_description,
                            facts=canonical_facts,
                            max_items=5,
                        )
                        if missing:
                            try:
                                enriched = await _llm_integrate_missing_facts_into_description(
                                    description=clean_description,
                                    missing_facts=missing,
                                    source_text=_pick_richest_source_text_for_description(event_db, candidate),
                                    label="merge_fact_coverage",
                                )
                            except Exception:  # pragma: no cover
                                enriched = None
                            if enriched:
                                enriched = _strip_foreign_schedule_noise(
                                    enriched,
                                    event_date=event_db.date,
                                    end_date=event_db.end_date,
                                    event_title=event_db.title,
                                ) or enriched
                                enriched = _normalize_plaintext_paragraphs(enriched) or enriched
                                enriched = _promote_first_person_quotes_to_blockquotes(enriched) or enriched
                                enriched = _promote_inline_quoted_direct_speech_to_blockquotes(enriched) or enriched
                                enriched = _drop_reported_speech_duplicates(enriched) or enriched
                                enriched = _normalize_blockquote_markers(enriched) or enriched
                                enriched = _fix_broken_initial_paragraph_splits(enriched) or enriched
                                enriched = (
                                    _sanitize_description_output(
                                        enriched,
                                        source_text=_pick_richest_source_text_for_description(event_db, candidate),
                                    )
                                    or enriched
                                )
                                enriched = _ensure_blockquote_has_attribution(
                                    description=enriched,
                                    attribution_name=director_name_hint,
                                )
                                clean_description = enriched
                        if _description_needs_infoblock_logistics_strip(clean_description, candidate=candidate):
                            try:
                                edited = await _llm_remove_infoblock_logistics(
                                    description=clean_description,
                                    candidate=candidate,
                                    label="merge_remove_logistics",
                                )
                            except Exception:  # pragma: no cover
                                edited = None
                            if edited:
                                edited = _normalize_plaintext_paragraphs(edited) or edited
                                clean_description = edited
                        if _description_needs_channel_promo_strip(clean_description):
                            clean_description = (
                                _strip_channel_promo_from_description(clean_description) or clean_description
                            )
                        clean_description = (
                            _ensure_minimal_description_headings(clean_description) or clean_description
                        )
                        event_db.description = _clip(clean_description, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                        updated_fields = True
                        updated_keys.append("description")
                    if quote_candidates and (not merge_fact_first_used):
                        current_description = (event_db.description or "").strip()
                        if current_description and not re.search(r"(?m)^>\s+", current_description):
                            enforced_description = await _ensure_direct_quote_blockquote(
                                description=current_description,
                                quote_candidates=quote_candidates,
                                candidate_text=(
                                    _strip_promo_lines(candidate.source_text) or candidate.source_text
                                ),
                                facts_before=facts_before_list,
                                label="merge_quote_enforce_current_desc",
                            )
                            enforced_description = (
                                _normalize_plaintext_paragraphs(enforced_description)
                                or enforced_description
                            )
                            enforced_description = (
                                _normalize_blockquote_markers(enforced_description)
                                or enforced_description
                            )
                            enforced_description = (
                                _drop_reported_speech_duplicates(enforced_description)
                                or enforced_description
                            )
                            enforced_description = _ensure_blockquote_has_attribution(
                                description=enforced_description,
                                attribution_name=director_name_hint,
                            )
                            if enforced_description and enforced_description != current_description:
                                enforced_description = (
                                    _ensure_minimal_description_headings(enforced_description)
                                    or enforced_description
                                )
                                event_db.description = _clip(
                                    enforced_description, SMART_UPDATE_DESCRIPTION_MAX_CHARS
                                )
                                updated_fields = True
                                if "description" not in updated_keys:
                                    updated_keys.append("description")

                    merge_price_min = merge_data.get("ticket_price_min")
                    merge_price_max = merge_data.get("ticket_price_max")
                    if source_type_clean in {"vk", "telegram", "tg"} and (
                        merge_price_min is not None or merge_price_max is not None
                    ):
                        introducing_new_price = (
                            getattr(event_db, "ticket_price_min", None) is None
                            and getattr(event_db, "ticket_price_max", None) is None
                            and candidate.ticket_price_min is None
                            and candidate.ticket_price_max is None
                        )
                        if introducing_new_price:
                            poster_texts_for_price: list[str] = []
                            for p in candidate.posters or []:
                                for k in ("ocr_text", "ocr_title"):
                                    v = getattr(p, k, None)
                                    if isinstance(v, str) and v.strip():
                                        poster_texts_for_price.append(v.strip())
                            price_probe = "\n".join(
                                [
                                    clean_title,
                                    clean_source_text or "",
                                    clean_raw_excerpt or "",
                                    *poster_texts_for_price[:3],
                                ]
                            ).strip()
                            if not _has_price_evidence(price_probe, merge_price_min, merge_price_max):
                                merge_price_min = None
                                merge_price_max = None
                                note = "Цена отброшена: не найдена в источнике"
                                if note not in text_filter_facts:
                                    text_filter_facts.append(note)

                    ticket_updates = _apply_ticket_fields(
                        event_db,
                        ticket_link=merge_data.get("ticket_link"),
                        ticket_price_min=merge_price_min,
                        ticket_price_max=merge_price_max,
                        ticket_status=merge_data.get("ticket_status"),
                        candidate_trust=candidate.trust_level,
                    )
                    if ticket_updates:
                        updated_fields = True
                        updated_keys.extend(ticket_updates)

                elif has_new_text or needs_schedule_cleanup:
                    # LLM merge can be unavailable (offline runs, local env, transient outages).
                    # In production, avoid publishing non-LLM text to Telegraph; for offline/regression
                    # runs (schedule_tasks=False) do a deterministic merge to keep facts visible.
                    if not schedule_tasks:
                        base = before_description
                        if needs_schedule_cleanup:
                            cleaned = _strip_foreign_schedule_noise(
                                base,
                                event_date=event_db.date,
                                end_date=event_db.end_date,
                                event_title=event_db.title,
                            )
                            if cleaned:
                                base = cleaned
                                note = "Текст очищен: убраны строки расписания других дат"
                                if note not in text_filter_facts:
                                    text_filter_facts.append(note)
                        merged = base
                        if has_new_text:
                            merged = _fallback_merge_description(base, candidate, max_sentences=2) or base
                        merged = _normalize_plaintext_paragraphs(merged) or merged
                        merged = _promote_first_person_quotes_to_blockquotes(merged) or merged
                        merged = _promote_inline_quoted_direct_speech_to_blockquotes(merged) or merged
                        merged = _drop_reported_speech_duplicates(merged) or merged
                        merged = _normalize_blockquote_markers(merged) or merged
                        merged = _append_missing_scene_hint(
                            description=merged, source_text=candidate.source_text
                        ) or merged
                        merged = (
                            _sanitize_description_output(
                                merged,
                                source_text=candidate.source_text,
                            )
                            or merged
                        )
                        current = (event_db.description or "").strip()
                        merged = (merged or "").strip()
                        if merged and merged != current:
                            merged = _ensure_minimal_description_headings(merged) or merged
                            event_db.description = _clip(merged, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                            updated_fields = True
                            if "description" not in updated_keys:
                                updated_keys.append("description")
                        note = "LLM недоступна: описание обновлено детерминированно"
                        if note not in text_filter_facts:
                            text_filter_facts.append(note)
                        skip_topic_reclassify = True
                    else:
                        # Production-safe: keep description unchanged and record a service note in the source log.
                        note = "LLM недоступна: описание не обновлено"
                        if note not in text_filter_facts:
                            text_filter_facts.append(note)
                        if needs_schedule_cleanup:
                            cleaned = _strip_foreign_schedule_noise(
                                before_description,
                                event_date=event_db.date,
                                end_date=event_db.end_date,
                                event_title=event_db.title,
                            )
                            if cleaned and cleaned != before_description:
                                cleaned = _normalize_plaintext_paragraphs(cleaned) or cleaned
                                cleaned = _promote_first_person_quotes_to_blockquotes(cleaned) or cleaned
                                cleaned = _promote_inline_quoted_direct_speech_to_blockquotes(cleaned) or cleaned
                                cleaned = _drop_reported_speech_duplicates(cleaned) or cleaned
                                cleaned = _normalize_blockquote_markers(cleaned) or cleaned
                                cleaned = _append_missing_scene_hint(
                                    description=cleaned, source_text=candidate.source_text
                                ) or cleaned
                                cleaned = (
                                    _sanitize_description_output(
                                        cleaned,
                                        source_text=candidate.source_text,
                                    )
                                    or cleaned
                                )
                                cleaned = _ensure_minimal_description_headings(cleaned) or cleaned
                                event_db.description = _clip(cleaned, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                                updated_fields = True
                                updated_keys.append("description")
                                note = "Текст очищен: убраны строки расписания других дат"
                                if note not in text_filter_facts:
                                    text_filter_facts.append(note)
                                skip_topic_reclassify = True
        else:
            ticket_updates = _apply_ticket_fields(
                event_db,
                ticket_link=candidate.ticket_link,
                ticket_price_min=candidate.ticket_price_min,
                ticket_price_max=candidate.ticket_price_max,
                ticket_status=candidate.ticket_status,
                candidate_trust=candidate.trust_level,
            )
            if ticket_updates:
                updated_fields = True
                updated_keys.extend(ticket_updates)
            # Keep original description snapshot for source log snippet.
            before_description = before_description or (event_db.description or "")

        if not event_db.location_address and candidate.location_address:
            event_db.location_address = candidate.location_address
            updated_fields = True
            updated_keys.append("location_address")
        if not event_db.city and candidate.city:
            event_db.city = candidate.city
            updated_fields = True
            updated_keys.append("city")
        if (
            not event_db.end_date
            and _apply_event_end_date(
                event_db,
                end_date=candidate.end_date,
                inferred=bool(candidate.end_date_is_inferred),
                updated_keys=updated_keys,
            )
        ):
            updated_fields = True
        if not event_db.festival and candidate.festival:
            event_db.festival = candidate.festival
            updated_fields = True
            updated_keys.append("festival")
        if event_db.event_type:
            normalized_existing = _normalize_event_type_value(
                event_db.title, event_db.description, event_db.event_type
            )
            if normalized_existing and normalized_existing != event_db.event_type:
                event_db.event_type = normalized_existing
                updated_fields = True
                updated_keys.append("event_type")
        if candidate.event_type and not event_db.event_type:
            normalized = _normalize_event_type_value(
                event_db.title, event_db.description, candidate.event_type
            )
            event_db.event_type = normalized or candidate.event_type
            updated_fields = True
            updated_keys.append("event_type")
        if candidate.emoji and not event_db.emoji:
            event_db.emoji = candidate.emoji
            updated_fields = True
            updated_keys.append("emoji")
        existing_short = _clean_short_description(getattr(event_db, "short_description", None))
        short_needs_refresh = not _is_short_description_acceptable(
            existing_short, min_words=12, max_words=16
        )
        if ("description" in updated_keys) or has_new_text:
            short_needs_refresh = True
        if short_needs_refresh:
            normalized_short: str | None = None
            try:
                normalized_short = await _llm_build_short_description(
                    title=event_db.title,
                    description=event_db.description or candidate.source_text,
                    event_type=event_db.event_type or candidate.event_type,
                )
            except Exception:
                normalized_short = None
            if not normalized_short:
                candidate_short = _clean_short_description(candidate.raw_excerpt)
                if _is_short_description_acceptable(
                    candidate_short, min_words=12, max_words=16
                ):
                    normalized_short = candidate_short
            if not normalized_short:
                normalized_short = _fallback_short_description_from_text(
                    event_db.description or candidate.source_text
                )
            if normalized_short and normalized_short != (event_db.short_description or "").strip():
                event_db.short_description = normalized_short
                updated_fields = True
                updated_keys.append("short_description")
        # search_digest is a short snippet used for search/cards and also shown on Telegraph
        # before long descriptions. It should be refreshed when description meaningfully changes.
        normalized_candidate_digest = _clean_search_digest(candidate.search_digest)
        digest_should_refresh = ("description" in updated_keys) or has_new_text
        new_digest = None
        if digest_should_refresh:
            if merge_digest_from_llm and (not merge_fact_first_used):
                new_digest = merge_digest_from_llm
            else:
                try:
                    new_digest = await _llm_build_search_digest(
                        title=event_db.title,
                        description=event_db.description,
                        event_type=event_db.event_type,
                    )
                except Exception:
                    new_digest = None
        if not new_digest:
            # Fallback: accept candidate-provided digest (e.g. parsers), even if event already had one.
            new_digest = normalized_candidate_digest
        if not new_digest:
            new_digest = _fallback_digest_from_description(event_db.description)
        if new_digest and (new_digest.strip() != (event_db.search_digest or "").strip()):
            event_db.search_digest = new_digest
            updated_fields = True
            updated_keys.append("search_digest")
        if candidate.pushkin_card is True and not event_db.pushkin_card:
            event_db.pushkin_card = True
            updated_fields = True
            updated_keys.append("pushkin_card")
        if not event_db.is_free:
            if candidate.is_free is True:
                event_db.is_free = True
                updated_fields = True
                updated_keys.append("is_free")
            elif (
                event_db.ticket_price_min == 0
                and (event_db.ticket_price_max in (0, None))
            ):
                event_db.is_free = True
                updated_fields = True
                updated_keys.append("is_free")
        if not event_db.source_post_url and candidate.source_url and _is_http_url(candidate.source_url):
            event_db.source_post_url = candidate.source_url
            updated_fields = True
            updated_keys.append("source_post_url")
        if candidate.source_url and _is_vk_wall_url(candidate.source_url):
            if not event_db.source_vk_post_url:
                event_db.source_vk_post_url = candidate.source_url
                updated_fields = True
                updated_keys.append("source_vk_post_url")
        if not event_db.creator_id and candidate.creator_id:
            event_db.creator_id = candidate.creator_id
            updated_fields = True
            updated_keys.append("creator_id")

        added_posters, added_poster_urls, preview_invalidated, pruned_posters, photo_urls_changed = await _apply_posters(
            session,
            event_db.id,
            candidate.posters,
            poster_scope_hashes=candidate.poster_scope_hashes,
            event_title=candidate.title,
        )
        if added_posters or pruned_posters or photo_urls_changed:
            updated_fields = True
            updated_keys.append("posters")

        # Backfill legacy source fields into event_source for older events (e.g. /parse imports
        # created before event_source existed). This is required for deterministic merges like
        # dramteatr (site + telegram) in E2E and for operator transparency.
        await _ensure_legacy_event_sources(session, event_db)

        added_sources, same_source = await _ensure_event_source(session, event_db.id, candidate)
        if clean_source_text:
            if same_source:
                event_db.source_text = clean_source_text
                updated_fields = True
                updated_keys.append("source_text")
            if await _sync_source_texts(session, event_db):
                updated_fields = True
                updated_keys.append("source_texts")

        await _enqueue_ticket_sites_queue(session, event_id=int(event_db.id or 0))
        # If we didn't touch description in this merge, but it's clearly too short
        # compared to rich available source text (usually site import), generate a
        # full rewritten description best-effort. This is important for Telegraph
        # pages: a short "search snippet" is not acceptable as the main text.
        if "description" not in updated_keys:
            cur_desc = (event_db.description or "").strip()
            min_expected = _min_expected_description_len_from_sources(event_db, candidate)
            if min_expected and len(cur_desc) < min_expected:
                rewritten_full = None
                try:
                    rewritten_full = await _rewrite_description_full_from_sources(event_db, candidate)
                except Exception:  # pragma: no cover - defensive
                    rewritten_full = None
                if rewritten_full and len(rewritten_full) >= int(min_expected * 0.85):
                    rewritten_full = _ensure_minimal_description_headings(rewritten_full) or rewritten_full
                    event_db.description = _clip(rewritten_full, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                    updated_fields = True
                    updated_keys.append("description")
                else:
                    # Do not fall back to verbatim source text. Keep the previous description
                    # (or wait for the next LLM-backed update) to ensure Telegraph text stays LLM-produced.
                    pass

        await session.flush()
        added_log: list[str] = []
        duplicate_log: list[str] = []
        conflict_log: list[str] = []
        note_log: list[str] = []

        # 1) Added facts (LLM merge)
        added_log.extend(list(added_facts or []))
        # 1b) Duplicate facts (LLM reported as already known for this event)
        duplicate_log.extend(list(duplicate_facts or []))

        # 2) Anchor updates (deterministic, may not be present in LLM facts)
        if "date" in updated_keys and getattr(event_db, "date", None):
            added_log.append(f"Дата: {event_db.date}")
        if "end_date" in updated_keys and getattr(event_db, "end_date", None):
            added_log.append(f"Дата окончания: {event_db.end_date}")
        if (
            "end_date_is_inferred" in updated_keys
            and "end_date" not in updated_keys
            and getattr(event_db, "end_date", None)
            and not bool(getattr(event_db, "end_date_is_inferred", False))
        ):
            note_log.append(f"Дата окончания подтверждена: {event_db.end_date}")
        if "time" in updated_keys and getattr(event_db, "time", None):
            added_log.append(f"Время: {event_db.time}")
        if "location_name" in updated_keys and getattr(event_db, "location_name", None):
            loc = str(event_db.location_name or "").strip()
            if getattr(event_db, "location_address", None):
                loc = f"{loc}, {str(event_db.location_address).strip()}"
            if getattr(event_db, "city", None):
                loc = f"{loc}, {str(event_db.city).strip()}"
            if loc.strip():
                added_log.append(f"Локация: {loc.strip()}")

        # 3) Posters (added) + service notes
        for url in (added_poster_urls or [])[:3]:
            added_log.append(f"Добавлена афиша: {url}")
        if pruned_posters:
            note_log.append(f"Удалены лишние афиши: {pruned_posters}")
        if photo_urls_changed and not added_posters and not pruned_posters:
            note_log.append("Афиши переупорядочены по OCR")
        if preview_invalidated:
            note_log.append("3D-превью сброшено: изменились иллюстрации")

        # 4) Filters and text snippet are service notes
        note_log.extend((queue_notes or [])[:6])
        note_log.extend((text_filter_facts or [])[:2])
        note_log.extend((poster_filter_facts or [])[:3])
        # NOTE: We intentionally do NOT include "Текст дополнен: ..." snippets anymore.
        # Operator must see changes as explicit facts (✅/↩️) and can open Telegraph for the full text.

        # 5) Conflicts: prefer LLM-provided details, but also record deterministic anchor conflicts.
        conflict_log.extend([s for s in (conflict_facts or []) if isinstance(s, str) and s.strip()][:10])
        conflict_log.extend([s for s in (skipped_conflicts or []) if isinstance(s, str) and s.strip()][:10])
        for k, v in list((conflicting or {}).items())[:8]:
            if not v:
                continue
            conflict_log.append(f"Конфликт якоря: {k} -> {v}")

        # 6) Duplicate anchors (observed in source but already present)
        try:
            c = candidate
            blocked = set((conflicting or {}).keys())
            if c.date and "date" not in updated_keys and "date" not in blocked and (c.date == (event_db.date or "")):
                duplicate_log.append(f"Дата: {c.date}")
            if c.end_date and "end_date" not in updated_keys and "end_date" not in blocked and (c.end_date == (getattr(event_db, 'end_date', None) or "")):
                duplicate_log.append(f"Дата окончания: {c.end_date}")
            if c.time and "time" not in updated_keys and "time" not in blocked and (str(c.time).strip() == str(event_db.time or '').strip()):
                duplicate_log.append(f"Время: {str(c.time).strip()}")
            if (
                c.location_name
                and "location_name" not in updated_keys
                and "location_name" not in blocked
                and _location_matches(getattr(event_db, "location_name", None), c.location_name)
            ):
                parts = [c.location_name, c.location_address, c.city]
                loc = ", ".join(str(p).strip() for p in parts if (p or "").strip())
                if loc:
                    duplicate_log.append(f"Локация: {loc}")
            if (
                c.ticket_price_min is not None
                and c.ticket_price_max is not None
                and "ticket_price_min" not in updated_keys
                and "ticket_price_max" not in updated_keys
                and (c.ticket_price_min == getattr(event_db, "ticket_price_min", None))
                and (c.ticket_price_max == getattr(event_db, "ticket_price_max", None))
            ):
                price_text = _format_ticket_price(c.ticket_price_min, c.ticket_price_max)
                if price_text:
                    duplicate_log.append(f"Цена: {price_text}")
            if (
                c.ticket_link
                and "ticket_link" not in updated_keys
                and (c.ticket_link == (event_db.ticket_link or ""))
            ):
                label = "Регистрация" if c.is_free else "Билеты"
                duplicate_log.append(f"{label}: {c.ticket_link}")
            if c.ticket_status == "sold_out" and "ticket_status" not in updated_keys and getattr(event_db, "ticket_status", None) == "sold_out":
                duplicate_log.append("Билеты все проданы")
            if c.is_free is True and "is_free" not in updated_keys and bool(getattr(event_db, "is_free", False)) is True:
                duplicate_log.append("Бесплатно")
            if c.pushkin_card is True and "pushkin_card" not in updated_keys and bool(getattr(event_db, "pushkin_card", False)) is True:
                duplicate_log.append("Пушкинская карта")
            if c.event_type and "event_type" not in updated_keys and (c.event_type == (event_db.event_type or "")):
                duplicate_log.append(f"Тип: {c.event_type}")
            if c.festival and "festival" not in updated_keys and (c.festival == (event_db.festival or "")):
                duplicate_log.append(f"Фестиваль: {c.festival}")
        except Exception:
            # Best-effort: duplicates are for operator UX only.
            duplicate_log = duplicate_log

        # If we recorded no meaningful facts, keep the log useful and E2E-deterministic.
        if not (added_log or note_log or conflict_log or duplicate_log):
            # LLM merge can be unavailable in local/dev or for transient outages.
            # Keep source log useful and E2E-deterministic: record what we did change.
            if added_posters:
                note_log.append(f"Добавлены афиши: {added_posters}")
            if added_sources:
                note_log.append("Добавлен источник")
            if updated_keys:
                keys = [
                    k
                    for k in updated_keys
                    if k not in {"source_text", "source_texts", "end_date_is_inferred"}
                ]
                if keys:
                    note_log.append(f"Обновлено: {', '.join(keys[:6])}")

        # Demote meaning-duplicates of existing anchors (date/time) from ✅ to ↩️.
        # This solves operator confusion when LLM returns both:
        #   "Дата: 2026-02-12" and "Спектакль будет показан 12 февраля."
        try:
            added_log, duplicate_log = _demote_redundant_anchor_facts(
                added_log,
                duplicate_log,
                event_date=getattr(event_db, "date", None),
                event_time=getattr(event_db, "time", None),
                updated_keys=set(updated_keys),
            )
        except Exception:
            # Best-effort: never break the merge due to UX-only log shaping.
            pass

        # Normalize/dedupe within groups.
        added_log = _dedupe_source_facts(_drop_redundant_poster_facts(added_log))
        note_log = _dedupe_source_facts(_drop_redundant_poster_facts(note_log))
        conflict_log = _dedupe_source_facts(conflict_log)
        duplicate_log = _dedupe_source_facts(duplicate_log)

        # Remove duplicates that are actually part of the added set (by normalized key).
        def _key(v: str) -> str:
            c = _normalize_fact_item(v) or v
            return (c or "").strip().lower()

        added_keys = {_key(v) for v in added_log if _key(v)}
        duplicate_log = [v for v in duplicate_log if _key(v) and _key(v) not in added_keys]
        conflict_log = [v for v in conflict_log if _key(v) and _key(v) not in added_keys]

        fact_records: list[tuple[str, str]] = []
        for f in added_log:
            fact_records.append((f, "added"))
        for f in duplicate_log:
            fact_records.append((f, "duplicate"))
        for f in conflict_log:
            fact_records.append((f, "conflict"))
        for f in note_log:
            fact_records.append((f, "note"))

        if fact_records:
            await _record_source_facts(session, event_db.id, candidate, fact_records)

        if updated_fields:
            session.add(event_db)
        await session.commit()

    if (updated_fields or added_posters or (added_sources and not same_source)) and not skip_topic_reclassify:
        await _classify_topics(db, existing.id)

    holiday_changed = False
    try:
        holiday_changed = await _apply_holiday_festival_mapping(db, existing.id)
    except Exception:
        logger.warning(
            "smart_update: holiday mapping failed for event %s",
            existing.id,
            exc_info=True,
        )

    # Keep linked occurrences consistent when anchor-ish or grouping fields change.
    linked_refresh_ids: list[int] = []
    if any(k in {"title", "location_name", "date", "time"} for k in (updated_keys or [])):
        try:
            from linked_events import recompute_linked_event_ids

            lr = await recompute_linked_event_ids(db, int(existing.id or 0))
            linked_refresh_ids = [
                int(x)
                for x in (lr.changed_event_ids or [])
                if int(x) and int(x) != int(existing.id or 0)
            ]
        except Exception:
            logger.warning(
                "smart_update: linked events recompute failed for event %s",
                existing.id,
                exc_info=True,
            )
            linked_refresh_ids = []

    if schedule_tasks and linked_refresh_ids:
        try:
            from main import JobTask, enqueue_job

            for rid in linked_refresh_ids[:80]:
                await enqueue_job(db, int(rid), JobTask.telegraph_build, depends_on=None)
        except Exception:
            logger.warning(
                "smart_update: failed to enqueue linked telegraph refresh for event %s",
                existing.id,
                exc_info=True,
            )

    if schedule_tasks and (
        updated_fields
        or added_posters
        or (added_sources and not same_source)
        or holiday_changed
    ):
        try:
            from main import schedule_event_update_tasks
            async with db.get_session() as session:
                refreshed = await session.get(Event, existing.id)
            if refreshed:
                await schedule_event_update_tasks(db, refreshed, **(schedule_kwargs or {}))
        except Exception:
            logger.warning("smart_update: schedule/update failed for event %s", existing.id, exc_info=True)

    status = (
        "merged"
        if (updated_fields or added_posters or (added_sources and not same_source) or holiday_changed)
        else "skipped_nochange"
    )
    logger.info(
        "smart_update.merge event_id=%s status=%s updated=%s added_posters=%d added_sources=%s updated_keys=%s added_facts=%d skipped_conflicts=%d reason=%s",
        existing.id,
        status,
        int(bool(updated_fields)),
        added_posters,
        int(bool(added_sources)),
        ",".join(updated_keys[:12]) if updated_keys else "",
        len(added_facts),
        len(skipped_conflicts),
        match_reason if "match_reason" in locals() else None,
    )
    return SmartUpdateResult(
        status=status,
        event_id=existing.id,
        created=False,
        merged=bool(updated_fields or holiday_changed),
        added_posters=added_posters,
        added_sources=added_sources,
        added_facts=added_facts,
        skipped_conflicts=skipped_conflicts,
        reason=match_reason if "match_reason" in locals() else None,
        queue_notes=list(queue_notes or []),
    )


async def _apply_posters(
    session,
    event_id: int | None,
    posters: Sequence[PosterCandidate],
    poster_scope_hashes: Sequence[str] | None = None,
    event_title: str | None = None,
) -> tuple[int, list[str], bool, int, bool]:
    if not event_id:
        return 0, [], False, 0, False
    existing_rows = (
        await session.execute(select(EventPoster).where(EventPoster.event_id == event_id))
    ).scalars().all()
    existing_map = {row.poster_hash: row for row in existing_rows}
    added = 0
    now = datetime.now(timezone.utc)
    extra_urls: list[str] = []
    added_urls: list[str] = []
    preview_invalidated = False
    pruned = 0
    photo_urls_changed = False

    def _pick_display_url(p: PosterCandidate) -> str | None:
        return p.supabase_url or p.catbox_url

    def _remember_url(url: str | None) -> None:
        if url and url not in added_urls:
            added_urls.append(url)

    selected_hashes = {p.sha256 for p in posters if p.sha256}
    scope_hashes = {
        h.strip()
        for h in (poster_scope_hashes or [])
        if isinstance(h, str) and h.strip()
    }
    pruned_urls: set[str] = set()
    to_delete_by_hash: dict[str, EventPoster] = {}

    # 1) Exact prune: if the source provided the poster hash scope AND we have a non-empty
    # selected set for this event, drop any previously attached posters from that scope that
    # are not selected now.
    #
    # Important: if selection is empty, it usually means matching failed (OCR/title/time),
    # and pruning would incorrectly delete posters (regression seen in TG monitoring re-imports).
    if scope_hashes and selected_hashes:
        for h in scope_hashes:
            if h in selected_hashes:
                continue
            row = existing_map.get(h)
            if row:
                to_delete_by_hash[row.poster_hash] = row

    if to_delete_by_hash:
        for row in to_delete_by_hash.values():
            if row.catbox_url:
                pruned_urls.add(row.catbox_url)
            if getattr(row, "supabase_url", None):
                pruned_urls.add(str(getattr(row, "supabase_url")))
            await session.delete(row)
        pruned = len(to_delete_by_hash)

    for poster in posters:
        digest = poster.sha256
        if not digest:
            url = _pick_display_url(poster)
            if url:
                extra_urls.append(url)
            continue
        row = existing_map.get(digest)
        if row:
            changed = False
            if poster.catbox_url:
                if row.catbox_url != poster.catbox_url:
                    row.catbox_url = poster.catbox_url
                    changed = True
            if poster.supabase_url:
                if getattr(row, "supabase_url", None) != poster.supabase_url:
                    row.supabase_url = poster.supabase_url
                    changed = True
            if poster.supabase_path:
                if getattr(row, "supabase_path", None) != poster.supabase_path:
                    row.supabase_path = poster.supabase_path
                    changed = True
            if poster.phash:
                row.phash = poster.phash
            if poster.ocr_text is not None:
                row.ocr_text = poster.ocr_text
            if poster.ocr_title is not None:
                row.ocr_title = poster.ocr_title
            # OCR token accounting is best-effort: keep the latest non-zero values.
            if getattr(poster, "prompt_tokens", 0):
                row.prompt_tokens = int(getattr(poster, "prompt_tokens", 0) or 0)
            if getattr(poster, "completion_tokens", 0):
                row.completion_tokens = int(getattr(poster, "completion_tokens", 0) or 0)
            if getattr(poster, "total_tokens", 0):
                row.total_tokens = int(getattr(poster, "total_tokens", 0) or 0)
            row.updated_at = now
            if changed:
                _remember_url(_pick_display_url(poster))
        else:
            session.add(
                EventPoster(
                    event_id=event_id,
                    catbox_url=poster.catbox_url,
                    supabase_url=poster.supabase_url,
                    supabase_path=poster.supabase_path,
                    poster_hash=digest,
                    phash=poster.phash,
                    ocr_text=poster.ocr_text,
                    ocr_title=poster.ocr_title,
                    prompt_tokens=int(getattr(poster, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(poster, "completion_tokens", 0) or 0),
                    total_tokens=int(getattr(poster, "total_tokens", 0) or 0),
                    updated_at=now,
                )
            )
            added += 1
            _remember_url(_pick_display_url(poster))

    # Update event.photo_urls if possible
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if event:
        before_urls = list(event.photo_urls or [])
        before_count = int(getattr(event, "photo_count", 0) or len(before_urls))
        current = list(event.photo_urls or [])
        if pruned_urls:
            current = [u for u in current if u not in pruned_urls]
        for poster in posters:
            url = _pick_display_url(poster)
            if url and url not in current:
                current.append(url)
        for url in extra_urls:
            if url not in current:
                current.append(url)
        # Prefer posters that are *relevant* to this event (by OCR vs event title/date/time),
        # then fall back to OCR "quality" as a tie-breaker.
        preferred_urls: list[str] = []
        scored: list[tuple[float, int, int, str]] = []  # (relevance, quality, idx, url)
        title_for_score = (event.title or event_title or "").strip()
        date_for_score = (event.date or "").strip()
        time_for_score = (event.time or "").strip()

        def _norm(text: str | None) -> str:
            value = (text or "").strip().casefold().replace("ё", "е")
            value = unicodedata.normalize("NFKC", value)
            value = value.replace("\xa0", " ")
            value = re.sub(r"\s+", " ", value).strip()
            return value

        def _tokens(text: str | None) -> set[str]:
            raw = _norm(text)
            if not raw:
                return set()
            found = re.findall(r"[a-zа-я0-9]{3,}", raw, flags=re.IGNORECASE)
            return {t for t in found if t and not t.isdigit()}

        def _score_relevance(*, ocr_title: str | None, ocr_text: str | None) -> float:
            ocr_combined = " ".join(
                x for x in [(ocr_title or "").strip(), (ocr_text or "").strip()] if x
            ).strip()
            if not ocr_combined:
                return 0.0
            ocr_norm = _norm(ocr_combined)

            title_tokens = _tokens(title_for_score)
            ocr_tokens = _tokens(ocr_combined)
            overlap = len(title_tokens & ocr_tokens) if (title_tokens and ocr_tokens) else 0
            score = float(min(10, overlap * 2))

            title_norm = _norm(title_for_score)
            if title_norm and len(title_norm) >= 10 and title_norm in ocr_norm:
                score += 4.0

            d_raw = date_for_score
            if d_raw:
                try:
                    d_obj = date.fromisoformat(d_raw.split("..", 1)[0].strip())
                except Exception:
                    d_obj = None
                if d_obj is not None:
                    day = d_obj.day
                    month = d_obj.month
                    if re.search(rf"\b0?{day}[./-]0?{month}\b", ocr_norm):
                        score += 3.0

            t_raw = time_for_score
            if t_raw and t_raw != "00:00":
                hhmm = re.sub(r"\s+", "", t_raw)
                if re.match(r"^\d{1,2}:\d{2}$", hhmm):
                    hh, mm = hhmm.split(":", 1)
                    hh = hh.zfill(2)
                    if f"{hh}:{mm}" in ocr_norm or f"{hh}.{mm}" in ocr_norm:
                        score += 1.5

            return score

        for idx, poster in enumerate(posters):
            url = _pick_display_url(poster)
            if not url:
                continue
            quality = 0
            if poster.ocr_title:
                quality += len(poster.ocr_title)
            if poster.ocr_text:
                quality += len(poster.ocr_text)
            if quality <= 0:
                continue
            relevance = _score_relevance(ocr_title=poster.ocr_title, ocr_text=poster.ocr_text)
            scored.append((relevance, quality, idx, url))

        if scored:
            has_relevance = any(rel > 0.0 for rel, _q, _i, _u in scored)
            if has_relevance:
                scored_sorted = sorted(scored, key=lambda t: (-t[0], -t[1], t[2]))
            else:
                scored_sorted = sorted(scored, key=lambda t: (-t[1], t[2]))
            for _rel, _quality, _idx, url in scored_sorted:
                if url not in preferred_urls:
                    preferred_urls.append(url)
            current = preferred_urls + [url for url in current if url not in preferred_urls]
        photo_urls_changed = (current != before_urls) or (len(current) != before_count)
        event.photo_urls = current
        event.photo_count = len(current)
        # If the image set changed, any existing 3D preview becomes stale: force regeneration.
        if photo_urls_changed and getattr(event, "preview_3d_url", None):
            event.preview_3d_url = None
            preview_invalidated = True
        session.add(event)

    return added, added_urls, preview_invalidated, pruned, photo_urls_changed


async def _ensure_event_source(
    session,
    event_id: int | None,
    candidate: EventCandidate,
) -> tuple[bool, bool]:
    if not event_id or not candidate.source_url:
        return False, False
    raw = _strip_private_use(candidate.source_text) or (candidate.source_text or "")
    clean_source_text = _strip_promo_lines(raw) or raw
    existing = (
        await session.execute(
            select(EventSource).where(
                EventSource.event_id == event_id,
                EventSource.source_url == candidate.source_url,
            )
        )
    ).scalar_one_or_none()
    if existing:
        updated = False
        if clean_source_text and clean_source_text != existing.source_text:
            existing.source_text = clean_source_text
            existing.imported_at = datetime.now(timezone.utc)
            updated = True
            logger.info(
                "smart_update.source_text_update event_id=%s source_url=%s",
                event_id,
                candidate.source_url,
            )
        if candidate.trust_level and not existing.trust_level:
            existing.trust_level = candidate.trust_level
            updated = True
        if updated:
            session.add(existing)
        return False, True
    session.add(
        EventSource(
            event_id=event_id,
            source_type=candidate.source_type,
            source_url=candidate.source_url,
            source_chat_username=candidate.source_chat_username,
            source_chat_id=candidate.source_chat_id,
            source_message_id=candidate.source_message_id,
            source_text=clean_source_text,
            imported_at=datetime.now(timezone.utc),
            trust_level=candidate.trust_level,
        )
    )
    return True, False


async def _record_source_facts(
    session,
    event_id: int | None,
    candidate: EventCandidate,
    facts: Sequence[object],
) -> int:
    if not event_id or not candidate.source_url or not facts:
        return 0
    source = (
        await session.execute(
            select(EventSource).where(
                EventSource.event_id == event_id,
                EventSource.source_url == candidate.source_url,
            )
        )
    ).scalar_one_or_none()
    if not source:
        return 0
    # Keep source log idempotent per (event_id, source_url): repeated processing of
    # the same post must not accumulate multiple historical batches for one source.
    await session.execute(
        delete(EventSourceFact).where(
            EventSourceFact.event_id == int(event_id),
            EventSourceFact.source_id == int(source.id),
        )
    )
    now = datetime.now(timezone.utc).replace(microsecond=0)
    added = 0
    allowed_status = {"added", "duplicate", "conflict", "note"}

    def _coerce(item: object) -> tuple[str, str]:
        # Accept both legacy list[str] and new list[(fact, status)].
        if isinstance(item, tuple) and len(item) == 2:
            raw_fact = item[0]
            raw_status = item[1]
        else:
            raw_fact = item
            raw_status = "added"
        fact_s = str(raw_fact or "")
        status_s = str(raw_status or "added").strip().lower()
        if status_s not in allowed_status:
            status_s = "added"
        return fact_s, status_s

    for item in facts:
        raw_fact, status = _coerce(item)
        cleaned = _normalize_fact_item(raw_fact)
        if not cleaned:
            continue
        session.add(
            EventSourceFact(
                event_id=event_id,
                source_id=source.id,
                fact=cleaned,
                status=status,
                created_at=now,
            )
        )
        added += 1
    return added


async def _sync_source_texts(session, event: Event) -> bool:
    if not event:
        return False
    rows = (
        await session.execute(
            select(EventSource.source_text, EventSource.imported_at)
            .where(EventSource.event_id == event.id)
            .order_by(EventSource.imported_at)
        )
    ).all()
    texts: list[str] = []
    for text, _ts in rows:
        if not text:
            continue
        if text not in texts:
            texts.append(text)
    if texts != list(event.source_texts or []):
        event.source_texts = texts
        logger.info(
            "smart_update.source_texts_sync event_id=%s count=%d",
            event.id,
            len(texts),
        )
        return True
    return False


async def _classify_topics(db: Database, event_id: int | None) -> None:
    if not event_id:
        return
    try:
        from main import assign_event_topics
    except Exception:
        return
    async with db.get_session() as session:
        event = await session.get(Event, event_id)
        if not event or event.topics_manual:
            return
        try:
            await assign_event_topics(event)
        except Exception:
            logger.warning("smart_update: topic classification failed event_id=%s", event_id, exc_info=True)
            return
        session.add(event)
        await session.commit()
