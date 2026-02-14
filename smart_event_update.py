from __future__ import annotations

import asyncio
from calendar import monthrange
import json
import logging
import os
import time
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Iterable, Sequence

from sqlalchemy import and_, delete, or_, select

from db import Database
from models import Event, EventPoster, EventSource, EventSourceFact, PosterOcrCache

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
SMART_UPDATE_VISITOR_CONDITIONS_RULE = (
    "Условия участия/посещения (длительность, возраст, максимальный размер группы, формат/что взять/как одеться, "
    "что входит/не входит в оплату, нужен ли отдельный входной билет) считаются фактами о событии и должны попадать "
    "в описание и в facts/added_facts. "
    "Для description: не вставляй ссылки/телефоны и не указывай точные цены — пиши нейтрально "
    "(например «оплачивается отдельно», «входной билет нужен дополнительно»). "
    "Для facts/added_facts: точная сумма допускается только если она уточняет, что цена относится к части услуги "
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
    end_date: str | None = None
    festival: str | None = None
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
                "description": {"type": ["string", "null"]},
                "search_digest": {"type": ["string", "null"]},
                "facts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["description", "facts"],
            "additionalProperties": False,
        },
    },
}

CREATE_BUNDLE_SCHEMA = CREATE_BUNDLE_RESPONSE_FORMAT["json_schema"]["schema"]


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


_LOCATION_NOISE_PREFIXES_RE = re.compile(
    r"^(?:"
    r"кинотеатр|"
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
                re.search(r"(?:^|\\s)[А-ЯЁA-Z]\\.$", cur)
                and re.match(r"^[А-ЯЁ][а-яё]+\\b", nxt)
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
    r"\bуникальн\w+\s+возможн\w+\b"
)

_LIST_ITEM_LINE_RE = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])\s+\S")


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
    """Enforce "no hallucinated evaluations" invariants on LLM output.

    We keep the output LLM-authored, but remove a small set of high-risk patterns
    that routinely appear as generic marketing clichés or unsupported claims.
    """
    raw = (text or "").strip()
    if not raw:
        return None

    source_cf = (source_text or "").casefold()

    # Drop generic marketing clichés sentence-by-sentence.
    parts: list[str] = []
    for para in re.split(r"\n{2,}", raw):
        s = para.strip()
        if not s:
            continue
        if _looks_like_structured_block(s):
            # Do not mutate multi-line quotes: they are often exact phrases from the source.
            if re.search(r"(?m)^\s*>", s):
                parts.append(s)
                continue
            kept_lines: list[str] = []
            for line in s.splitlines():
                st = line.strip()
                if not st:
                    continue
                if _NEURAL_CLICHE_RE.search(st):
                    continue
                # Do not claim "premiere" unless the word (or close form) exists in the source.
                if re.search(r"(?i)\bпремьер\w+\b", st) and "премьер" not in source_cf:
                    continue
                kept_lines.append(st)
            if kept_lines:
                parts.append("\n".join(kept_lines))
            continue

        kept_sentences: list[str] = []
        normalized = re.sub(r"\s+", " ", s).strip()
        # Crude sentence splitter: good enough for removing obvious boilerplate.
        for sent in re.split(r"(?<=[.!?])\s+", normalized):
            st = sent.strip()
            if not st:
                continue
            if _NEURAL_CLICHE_RE.search(st):
                continue
            # Do not claim "premiere" unless the word (or close form) exists in the source.
            if re.search(r"(?i)\bпремьер\w+\b", st) and "премьер" not in source_cf:
                continue
            kept_sentences.append(st)
        if kept_sentences:
            parts.append(" ".join(kept_sentences))
    cleaned = "\n\n".join(parts).strip()

    # Avoid leading blank lines/spacers before the first heading.
    cleaned = re.sub(r"^\s*\n+", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


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
        s = re.sub(r"\s*[,.;:!?]+\s*$", "", s).strip()

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
    """Remove generic promo sentences about where to find more announcements."""
    raw = (text or "").strip()
    if not raw:
        return None
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return None
    sent_split = re.compile(r"(?<=[.!?…])\s+")
    out_paras: list[str] = []
    for para in paragraphs:
        if para.lstrip().startswith(">") or re.match(r"^\s*#{1,6}\s+\S", para) or re.match(r"^\s*[-*•]\s+\S", para):
            out_paras.append(para)
            continue
        sents = [s.strip() for s in sent_split.split(re.sub(r"\s*\n\s*", " ", para)) if s.strip()]
        kept: list[str] = []
        for sent in sents:
            if _DESCRIPTION_CHANNEL_PROMO_SENT_RE.search(sent) and (
                "t.me/" in sent.lower() or "телеграм" in sent.lower() or "telegram" in sent.lower()
            ):
                # Try to salvage the "event fact" prefix if the promo is appended to it.
                m = _DESCRIPTION_CHANNEL_PROMO_PHRASE_RE.search(sent)
                if m and m.start() > 0:
                    prefix = (sent[: m.start()] or "").rstrip()
                    prefix = re.sub(r"[\s,;:—–-]+$", "", prefix).strip()
                    if prefix and _has_datetime_signals(prefix):
                        kept.append(prefix)
                continue
            kept.append(sent)
        merged = " ".join(kept).strip()
        if merged:
            out_paras.append(merged)
    cleaned = "\n\n".join(out_paras).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned or None


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
    if re.search(r"(?i)^(дата|время|локац|адрес|город|источник)\\b", f):
        return True
    if re.search(r"(?i)^(текст\\s+очищен|llm\\s+недоступна|добавлена\\s+афиша)\\b", f):
        return True
    if "http://" in f or "https://" in f:
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
        # Only enforce coverage for short facts, unless they are explicit slogans/quotes.
        if not is_quoted and len(f) > 90:
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
    if re.search(r"(?m)^>\\s+", raw):
        return raw
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    if not paragraphs:
        return raw

    out: list[str] = []
    promoted = False

    quote_re = re.compile(r"(?s)«(?P<q>[^»]{25,900})»")
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
        if not q or not _FIRST_PERSON_QUOTE_RE.match(q.lower()):
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
    raw = re.sub(r"(?<=\S)\s+>\s+", "\n\n> ", raw)
    # Remove leading spaces before a blockquote marker.
    raw = re.sub(r"(?m)^[ \t]+(>\s+)", r"\1", raw)
    # Ensure a blank line before any blockquote paragraph.
    raw = re.sub(r"(?m)(?<!^)(?<!\n\n)^(>\s+)", r"\n\n\1", raw)
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


def _strip_promo_lines(text: str | None) -> str | None:
    if not text:
        return None
    lines: list[str] = []
    # Some offline fixtures use literal "\\n" sequences instead of real newlines.
    raw = str(text).replace("\\n", "\n")
    for line in raw.splitlines():
        # Drop generic "promo of announcement channel" lines: they are not event facts and
        # routinely bloat descriptions (e.g. "Информация о событиях ... в Telegram-канале ...").
        if _CHANNEL_PROMO_STRIP_RE.search(line):
            if not _has_datetime_signals(line):
                continue
        # Promotions often appear inside a real announcement. Drop ONLY "pure promo" lines
        # that don't carry event facts (date/time/title-like signals).
        if _PROMO_STRIP_RE.search(line):
            if not _has_datetime_signals(line) and not _EVENT_SIGNAL_RE.search(line):
                continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or None


def _strip_giveaway_lines(text: str | None) -> str | None:
    """Remove giveaway mechanics while preserving event facts when present."""
    if not text:
        return None
    kept: list[str] = []
    # Some offline fixtures use literal "\\n" sequences instead of real newlines.
    raw = str(text).replace("\\n", "\n")
    for line in raw.splitlines():
        if _GIVEAWAY_LINE_RE.search(line):
            # Keep a line if it looks like it contains event facts (date/time/title-ish).
            if _has_datetime_signals(line) or _EVENT_SIGNAL_RE.search(line):
                kept.append(line)
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    return cleaned or None


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
    # Split by sentence-ish boundaries while keeping it simple and deterministic.
    chunks = re.split(r"[.!?…]\s+|\n{2,}|\n", raw)
    candidates: list[str] = []
    seen: set[str] = set()
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
    name_re = re.compile(r"(?i)\bрежисс\w*\b[^\\n]{0,80}?([А-ЯЁ][а-яё]+\\s+[А-ЯЁ][а-яё]+)")
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
    if re.search(r"(?m)^>\\s+", desc):
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
    if re.search(r"(?m)^>\\s+", desc):
        return description

    qc = [str(q or "").strip() for q in (quote_candidates or []) if str(q or "").strip()]
    if not qc:
        return description

    quote = qc[0]
    enforced = await _llm_enforce_blockquote(description=desc, quote=quote, label=label)
    if enforced and re.search(r"(?m)^>\\s+", enforced):
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


def _initial_added_facts(candidate: EventCandidate) -> list[str]:
    facts: list[str] = []
    if candidate.date:
        facts.append(f"Дата: {candidate.date}")
    if candidate.end_date:
        facts.append(f"Дата окончания: {candidate.end_date}")
    if candidate.time:
        facts.append(f"Время: {candidate.time}")
    location_parts = [
        candidate.location_name,
        candidate.location_address,
        candidate.city,
    ]
    location = ", ".join(part.strip() for part in location_parts if part and part.strip())
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
    location_parts = [
        candidate.location_name,
        candidate.location_address,
        candidate.city,
    ]
    location = ", ".join(part.strip() for part in location_parts if part and part.strip())
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
    norm = _norm_space(value)
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
        "- Не используй хэштеги (`#...`) в формулировках фактов.\n"
        "- НЕ включай рекламные призывы, скидки/промокоды, механику розыгрыша.\n"
        "- НЕ включай промо-упоминания «где следить за анонсами» и ссылки на каналы/чаты с афишей "
        "(например «Информация о событиях ... доступна в Telegram-канале ...»).\n"
        "- Включай условия участия/посещения (длительность, возраст, максимальный размер группы, формат/что взять/как одеться, "
        "что входит/не входит в оплату, нужен ли отдельный входной билет). Не вставляй ссылки/телефоны; "
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


async def _rewrite_description_journalistic(candidate: EventCandidate) -> str | None:
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
    base = excerpt_raw or source_raw
    if _should_prefer_source_text_for_description(source_raw, excerpt_raw):
        base = source_raw
    base = _strip_promo_lines(base) or base
    base = _strip_private_use(base) or base
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
            _strip_promo_lines(candidate.source_text) or candidate.source_text,
            SMART_UPDATE_REWRITE_SOURCE_MAX_CHARS,
        ),
        "quote_candidates": _extract_quote_candidates(
            _strip_promo_lines(candidate.source_text) or candidate.source_text,
            max_items=2,
        ),
        "poster_texts": [_clip(p.ocr_text, 500) for p in candidate.posters if p.ocr_text][:3],
    }
    prompt = (
        "Ты — культурный журналист. Сделай журналистский рерайт анонса мероприятия. "
        "Передай суть и атмосферу, но НЕ копируй исходные фразы дословно. "
        "Не добавляй выдуманных фактов, используй только то, что есть в данных. "
        "Запрещено придумывать утверждения вроде 'премьера', 'впервые', 'аншлаг' и т.п., "
        "если это явно не сказано в источнике. "
        f"{SMART_UPDATE_YO_RULE} "
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE} "
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE} "
        "Без эмодзи и без хэштегов. "
        "Важно: НЕ повторяй в описании логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены) — "
        "она показывается отдельным инфоблоком сверху. "
        "Убери промо чужих/вспомогательных каналов с анонсами и призывы подписаться "
        "(например «Информация о событиях ... доступна в Telegram-канале ...»): это не факт про само событие. "
        "Можно использовать минимальную разметку для читабельности: "
        "заголовки `###`, цитаты блоком `> ...`, редкое выделение `**...**`. "
        "НЕ используй Markdown-ссылки вида `[текст](url)` и не вставляй таблицы. "
        "Убери рекламные и акционные фрагменты (скидки/промокоды/акция) и механику розыгрыша, если они не являются частью сути события. "
        "Не включай малозначимые и повторяющиеся строки (например `DD.MM | Название`, повтор даты/заголовка, «представление состоится ...» при уже указанной дате/времени). "
        "Если в источнике есть обрыв фразы/текста (в т.ч. обрезано на середине слова), не вставляй это дословно: либо перефразируй, либо опусти. "
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
    if quote_candidates and not re.search(r"(?m)^>\\s+", cleaned):
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
        stripped = _strip_infoblock_logistics_from_description(cleaned, candidate=candidate)
        if stripped:
            cleaned = stripped
    if _description_needs_channel_promo_strip(cleaned):
        cleaned = _strip_channel_promo_from_description(cleaned) or cleaned

    # For short Telegram snippets (1-2 lines), keep rewrite volume near source size.
    # This prevents aggressive expansion/hallucinated "long reads" when source is concise.
    if candidate.source_type == "telegram":
        short_base_len = len(base.strip())
        if 80 <= short_base_len <= 350:
            max_allowed = min(
                SMART_UPDATE_DESCRIPTION_MAX_CHARS,
                max(260, int(short_base_len * 1.9) + 120),
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


def _has_long_event_duration_signals(text: str | None) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    month_pat = "|".join(sorted(map(re.escape, _RU_MONTHS_GENITIVE.keys()), key=len, reverse=True))
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\s*\\.\\.\\s*20\d{2}-\d{2}-\d{2}\b", raw):
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
    if not _is_long_event_type_value(inferred_type):
        return None
    # Guardrail: event_type can be misclassified by upstream LLMs.
    # Apply a default 1-month end_date only when the source text looks like a long event
    # (exhibition/fair/exposition) or contains explicit duration signals.
    hay = "\n".join(
        [
            str(candidate.title or ""),
            str(candidate.raw_excerpt or ""),
            str(candidate.source_text or ""),
        ]
    ).strip()
    if hay and not (_LONG_EVENT_TEXT_HINT_RE.search(hay) or _has_long_event_duration_signals(hay)):
        return None
    start = _parse_iso_date(candidate.date)
    if not start:
        return None
    candidate.end_date = _add_one_calendar_month(start).isoformat()
    return candidate.end_date


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
    if start and not end:
        end = start
    return start, end


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
    if value.startswith("http://") or value.startswith("https://"):
        value = value.rstrip("/")
    return value


def _is_http_url(url: str | None) -> bool:
    if not url:
        return False
    value = url.strip().lower()
    return value.startswith("http://") or value.startswith("https://")


_VK_WALL_URL_RE = re.compile(
    r"^https?://(?:m\\.)?vk\\.com/wall-?\\d+_\\d+/?$",
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
    }
    prompt = (
        "Ты готовишь данные для создания события.\n"
        "Верни JSON строго по схеме.\n\n"
        "1) description:\n"
        "- Напиши ПОЛНОЕ развернутое описание события как культурный журналист.\n"
        "- Сохрани ВСЕ значимые факты из source_text/raw_excerpt (кроме логистики).\n"
        "- Не копируй дословно длинными кусками; перефразируй, но не сокращай смысл.\n"
        "- Структура: абзацы, разделяй пустой строкой; 1–2 предложения в абзаце (для списков правило не применимо).\n"
        "- Запрещено: хэштеги, рекламные клише/прогнозы, механика розыгрыша.\n"
        "- ВАЖНО: НЕ включай в текст логистику (дата/время/площадка/точный адрес/город/ссылки/телефон/контакты/точные цены)\n"
        "  и не дублируй строки `Дата:`, `Время:`, `Локация:`, `Билеты:`.\n"
        f"{SMART_UPDATE_YO_RULE}\n"
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE}\n\n"
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE}\n\n"
        "2) facts:\n"
        "- Верни 6–18 атомарных фактов (1 факт = 1 строка), только про ЭТО событие.\n"
        "- НЕ включай дату/время/адрес/город как отдельные факты.\n"
        "- НЕ включай скидки/промокоды/призывы подписаться/ссылки на каналы.\n"
        "- Включай условия участия/посещения (длительность, возраст, размер группы, формат/что взять/как одеться, "
        "что входит/не входит в оплату), без ссылок/телефонов; сумму указывай только если это важно, чтобы пояснить "
        "что оплачивается отдельно (не более 1 факта).\n\n"
        "- Если есть прямая речь и понятно, кто говорит, оформи как `Цитата (Имя Фамилия): ...`.\n\n"
        "3) search_digest:\n"
        "- 1 предложение, 120–220 символов (макс 260), без эмодзи/хэштегов/списков.\n"
        "- Не указывай дату/время/адрес/город/цены/ссылки.\n"
        "- Не повторяй название дословно в начале.\n\n"
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
        candidates_payload.append(
            {
                "id": ev.id,
                "title": ev.title,
                "date": ev.date,
                "time": ev.time,
                "end_date": ev.end_date,
                "location_name": ev.location_name,
                "location_address": ev.location_address,
                "city": ev.city,
                "ticket_link": ev.ticket_link,
                "description": _clip(ev.description, 600),
                "source_text": _clip(ev.source_text, 600),
                "poster_texts": poster_texts,
            }
        )

    payload = {
        "candidate": {
            "title": candidate.title,
            "date": candidate.date,
            "time": candidate.time,
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
        },
        "events": candidates_payload[:10],
    }
    prompt = (
        "Ты сопоставляешь анонс события с уже существующими событиями. "
        "Найди наиболее вероятное совпадение или верни null. "
        "Учитывай дату, время, площадку, участников, ссылки, афиши и OCR. "
        "Ответь строго JSON."
        "\n\n"
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
        f"{SMART_UPDATE_YO_RULE} "
        f"{SMART_UPDATE_PRESERVE_LISTS_RULE} "
        f"{SMART_UPDATE_VISITOR_CONDITIONS_RULE} "
        "Описание должно читаться как единый связный текст-повествование (не рваное). "
        "Разбиение на абзацы делай осмысленно. НЕ разрывай предложения пустой строкой на середине, "
        "и особенно не ставь пустую строку между инициалом и фамилией (например `Н. Любимова`). "
        "Описание должно быть журналистским рерайтом (не дословно), без эмодзи и хэштегов, без выдуманных деталей. "
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
        kept: list[str] = []
        for ln in block.splitlines():
            s = ln.strip()
            if not s:
                continue
            key = re.sub(r"\s+", " ", s).strip().lower()
            # Even short lines can be duplicated facts (e.g. awards). Dedupe more aggressively.
            if len(key) >= 15 and key in seen_line_keys:
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
    for t in [
        getattr(event, "source_text", None),
        *(getattr(event, "source_texts", None) or []),
        getattr(candidate, "source_text", None),
        getattr(candidate, "raw_excerpt", None),
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
        "Без эмодзи и без хэштегов. Убери промо/акции и механику розыгрыша (если не часть сути). "
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
        stripped = _strip_infoblock_logistics_from_description(cleaned, candidate=candidate)
        if stripped:
            cleaned = stripped
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
    for dd_s, mm_s in re.findall(r"\\b(\\d{1,2})[./](\\d{1,2})\\b", raw):
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
    raw = re.sub(r"\s+", " ", raw).strip().lower()
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


def _title_has_meaningful_tokens(title: str | None) -> bool:
    norm = _normalize_title_for_match(title)
    if not norm:
        return False
    toks = {
        t
        for t in re.findall(r"[a-zа-яё0-9]+", norm)
        if len(t) >= 3 and t not in _TITLE_MATCH_STOPWORDS and not t.isdigit()
    }
    return bool(toks)


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


def _has_explicit_time_conflict(candidate_time: str | None, event_time: str | None) -> bool:
    ct = _normalize_time_for_match(candidate_time)
    et = _normalize_time_for_match(event_time)
    return bool(ct and et and ct != et)


def _single_candidate_auto_match_ok(
    candidate: EventCandidate,
    event_db: Event,
    *,
    is_canonical_site: bool,
) -> bool:
    # Guard against catastrophic merges when shortlist shrinks to 1 by broad anchors
    # (e.g. generic city location + long-running exhibition date range overlap).
    if is_canonical_site:
        # Canonical parser sources are allowed to repair a polluted title when
        # anchors are strongly aligned.
        if candidate.date and getattr(event_db, "date", None) and candidate.date != event_db.date:
            return False
        if candidate.location_name and getattr(event_db, "location_name", None):
            if not _location_matches(candidate.location_name, event_db.location_name):
                return False
        if _has_explicit_time_conflict(candidate.time, event_db.time):
            return False
        ct = _normalize_time_for_match(candidate.time)
        et = _normalize_time_for_match(event_db.time)
        if ct and et and ct == et:
            return True
        if _titles_look_related(candidate.title, getattr(event_db, "title", None)):
            return True
        # Allow parser correction when candidate has explicit time but existing event
        # has empty/placeholder time.
        if ct and not et:
            return True
        return False

    if not _titles_look_related(candidate.title, getattr(event_db, "title", None)):
        return False
    if _has_explicit_time_conflict(candidate.time, event_db.time):
        return False
    return True


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
    cleaned = _strip_foreign_schedule_headings(text, event_date=event_date, end_date=end_date)
    cleaned = _strip_schedule_headings_all(cleaned)
    cleaned = _strip_foreign_schedule_sentences(cleaned, event_title=event_title)
    return cleaned


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


async def _smart_event_update_impl(
    db: Database,
    candidate: EventCandidate,
    *,
    check_source_url: bool = True,
    schedule_tasks: bool = True,
    schedule_kwargs: dict[str, Any] | None = None,
) -> SmartUpdateResult:
    logger.info(
        "smart_update.start source_type=%s source_url=%s title=%s date=%s time=%s location=%s city=%s posters=%d trust=%s",
        candidate.source_type,
        candidate.source_url,
        _clip_title(candidate.title),
        candidate.date,
        candidate.time,
        _clip_title(candidate.location_name, 60),
        candidate.city,
        len(candidate.posters),
        candidate.trust_level,
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

    clean_title = _strip_private_use(candidate.title) or (candidate.title or "")
    if not clean_title:
        logger.warning(
            "smart_update.invalid reason=empty_title_after_clean source_type=%s source_url=%s",
            candidate.source_type,
            candidate.source_url,
        )
        return SmartUpdateResult(status="invalid", reason="empty_title_after_clean")
    raw_source_text = _strip_private_use(candidate.source_text) or (
        candidate.source_text or ""
    )
    raw_excerpt = _strip_private_use(candidate.raw_excerpt) or (candidate.raw_excerpt or "")

    text_filter_facts: list[str] = []
    default_end_date = _maybe_apply_default_end_date_for_long_event(candidate)
    if default_end_date:
        text_filter_facts.append(f"Дата окончания по умолчанию: {default_end_date}")

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
        if not (_has_datetime_signals(raw_source_text) or _has_datetime_signals(raw_excerpt)):
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
    if (before_promo_src or "") != (clean_source_text or "") or (before_promo_excerpt or "") != (clean_raw_excerpt or ""):
        text_filter_facts.append("Убраны промо-фрагменты")

    # Multi-event digests should not be imported as a single event.
    if (candidate.source_type in {"vk", "tg"}) and _looks_like_schedule_digest(
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

    # Filter out irrelevant posters (e.g. generic discount banners) using OCR.
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

        kept: list[PosterCandidate] = []
        for p in list(candidate.posters):
            try:
                ok, reason = await _poster_is_relevant(candidate, p)
            except Exception:  # pragma: no cover - defensive
                ok, reason = True, None
            if ok:
                kept.append(p)
                continue
            url = (p.catbox_url or p.supabase_url or "").strip()
            if url:
                suffix = f" ({reason})" if reason else ""
                poster_filter_facts.append(f"Афиша пропущена: {url}{suffix}")
        candidate.posters = kept

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
        if candidate.city:
            stmt = stmt.where(Event.city == candidate.city)
        res = await session.execute(stmt)
        shortlist = list(res.scalars().all())

    if candidate.location_name:
        shortlist = [
            ev for ev in shortlist if _location_matches(ev.location_name, candidate.location_name)
        ]

    # Time is an anchor field, but for canonical site/parser imports we allow time corrections:
    # matching must work even if a Telegram-first event had a wrong/empty time.
    cand_time = (candidate.time or "").strip()
    is_canonical_site = str(candidate.source_type or "").startswith("parser:")
    if cand_time and (not is_canonical_site):
        time_filtered = [ev for ev in shortlist if (ev.time or "").strip() == cand_time]
        if time_filtered:
            shortlist = time_filtered

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

    if not shortlist:
        match_event = None
        match_reason = "shortlist_empty"
    else:

        # Deterministic single-candidate match is allowed only when anchors look sane.
        # Otherwise fall back to LLM matching / create to avoid catastrophic cross-event merges.
        if len(shortlist) == 1 and _single_candidate_auto_match_ok(
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
        if strong_matches:
            best = max(strong_matches.items(), key=lambda item: item[1])
            match_event = next((ev for ev in shortlist if ev.id == best[0]), None)
            match_reason = "strong_match"
            logger.info(
                "smart_update.match type=strong event_id=%s score=%s",
                getattr(match_event, "id", None),
                best[1],
            )

        if match_event is None:
            match_id, confidence, reason = await _llm_match_event(
                candidate, shortlist[:10], posters_map=posters_map
            )
            match_reason = reason
            if match_id:
                match_event = next((ev for ev in shortlist if ev.id == match_id), None)
                if match_event is None:
                    confidence = 0.0
                threshold = 0.85 if allow_parallel and len(shortlist) > 1 else 0.6
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
    if match_event is not None and not str(candidate.source_type or "").startswith("parser:"):
        if _title_has_meaningful_tokens(candidate.title) and _title_has_meaningful_tokens(getattr(match_event, "title", None)):
            if not _titles_look_related(candidate.title, getattr(match_event, "title", None)):
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
        description_value = (clean_raw_excerpt or clean_source_text or clean_title or "").strip()
        if _should_prefer_source_text_for_description(clean_source_text, clean_raw_excerpt):
            description_value = (clean_source_text or "").strip()
            logger.info(
                "smart_update.create_description_seed_source_text source_type=%s source_url=%s excerpt_len=%d source_len=%d",
                candidate.source_type,
                candidate.source_url,
                len((clean_raw_excerpt or "").strip()),
                len((clean_source_text or "").strip()),
            )

        bundled_facts: list[str] | None = None
        bundled_digest: str | None = None
        bundled_desc: str | None = None
        try:
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
            bundled_desc_raw = bundled.get("description")
            if isinstance(bundled_desc_raw, str) and bundled_desc_raw.strip():
                bundled_desc = bundled_desc_raw.strip()
            bundled_digest = _clean_search_digest(bundled.get("search_digest"))
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
                    title=clean_title,
                    description=description_value,
                    event_type=normalized_event_type or candidate.event_type,
                )
            except Exception:
                llm_digest = None
            if llm_digest:
                normalized_digest = llm_digest
        if not normalized_digest:
            normalized_digest = _fallback_digest_from_description(description_value)
        new_event = Event(
            title=clean_title,
            description=description_value,
            festival=candidate.festival,
            date=candidate.date or "",
            time=candidate.time or "",
            location_name=candidate.location_name or "",
            location_address=candidate.location_address,
            city=candidate.city or "Калининград",
            ticket_price_min=candidate.ticket_price_min,
            ticket_price_max=candidate.ticket_price_max,
            ticket_link=candidate.ticket_link,
            ticket_status=candidate.ticket_status,
            ticket_trust_level=candidate.trust_level,
            event_type=normalized_event_type or candidate.event_type,
            emoji=candidate.emoji,
            end_date=candidate.end_date,
            is_free=is_free_value,
            pushkin_card=bool(candidate.pushkin_card),
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

            added_posters, added_poster_urls, preview_invalidated, pruned_posters = await _apply_posters(
                session,
                new_event.id,
                candidate.posters,
                poster_scope_hashes=candidate.poster_scope_hashes,
                event_title=candidate.title,
            )
            added_sources, _same_source = await _ensure_event_source(
                session, new_event.id, candidate
            )
            if candidate.source_text:
                await _sync_source_texts(session, new_event)
            await session.flush()
            initial_records: list[tuple[str, str]] = []
            for fact in _initial_added_facts(candidate):
                initial_records.append((fact, "added"))
            for fact in (extracted_facts or [])[:18]:
                initial_records.append((fact, "added"))
            for fact in (text_filter_facts or [])[:2]:
                initial_records.append((fact, "note"))
            for fact in (poster_filter_facts or [])[:3]:
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

        await _classify_topics(db, new_event.id)

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
        if existing.time and candidate.time and existing.time != candidate.time:
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

    async with db.get_session() as session:
        event_db = await session.get(Event, existing.id)
        if not event_db:
            return SmartUpdateResult(status="error", reason="event_missing")
        before_description = event_db.description or ""
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

        if is_canonical_site:
            # Canonical site/parser source: allow correcting anchors on an existing event.
            # This makes Telegram-first -> /parse merge converge to the site truth.
            if candidate.date and candidate.date != (event_db.date or ""):
                event_db.date = candidate.date
                updated_fields = True
                updated_keys.append("date")
            if candidate.end_date and candidate.end_date != getattr(event_db, "end_date", None):
                event_db.end_date = candidate.end_date
                updated_fields = True
                updated_keys.append("end_date")
            if candidate.time and candidate.time.strip() and candidate.time.strip() != (event_db.time or "").strip():
                event_db.time = candidate.time.strip()
                updated_fields = True
                updated_keys.append("time")
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
                    event_db.end_date = cand_end_iso.isoformat()
                    updated_fields = True
                    if "end_date" not in updated_keys:
                        updated_keys.append("end_date")
                    if "end_date" in conflicting:
                        conflicting.pop("end_date", None)
                else:
                    skipped_conflicts.append(
                        f"Дата окончания: {getattr(event_db, 'end_date', None)} -> {candidate.end_date} "
                        f"(выбран: event_before по trust {event_trust_level or 'medium'}>{candidate.trust_level or 'medium'})"
                    )

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
                facts_before_list = [
                    str(r[0]).strip()
                    for r in (
                        await session.execute(
                            select(EventSourceFact.fact).where(
                                EventSourceFact.event_id == int(event_db.id or 0),
                                EventSourceFact.status == "added",
                            )
                        )
                    ).all()
                    if (r and str(r[0] or "").strip())
                ]
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
                        if quote_candidates and not re.search(r"(?m)^>\\s+", clean_description):
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
                            stripped = _strip_infoblock_logistics_from_description(
                                clean_description, candidate=candidate
                            )
                            if stripped:
                                clean_description = stripped
                        if _description_needs_channel_promo_strip(clean_description):
                            clean_description = (
                                _strip_channel_promo_from_description(clean_description) or clean_description
                            )
                        event_db.description = _clip(clean_description, SMART_UPDATE_DESCRIPTION_MAX_CHARS)
                        updated_fields = True
                        updated_keys.append("description")
                    if quote_candidates:
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
                                event_db.description = _clip(
                                    enforced_description, SMART_UPDATE_DESCRIPTION_MAX_CHARS
                                )
                                updated_fields = True
                                if "description" not in updated_keys:
                                    updated_keys.append("description")

                    ticket_updates = _apply_ticket_fields(
                        event_db,
                        ticket_link=merge_data.get("ticket_link"),
                        ticket_price_min=merge_data.get("ticket_price_min"),
                        ticket_price_max=merge_data.get("ticket_price_max"),
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
        if not event_db.end_date and candidate.end_date:
            event_db.end_date = candidate.end_date
            updated_fields = True
            updated_keys.append("end_date")
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
        # search_digest is a short snippet used for search/cards and also shown on Telegraph
        # before long descriptions. It should be refreshed when description meaningfully changes.
        normalized_candidate_digest = _clean_search_digest(candidate.search_digest)
        digest_should_refresh = ("description" in updated_keys) or has_new_text
        new_digest = None
        if digest_should_refresh:
            if merge_digest_from_llm:
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

        added_posters, added_poster_urls, preview_invalidated, pruned_posters = await _apply_posters(
            session,
            event_db.id,
            candidate.posters,
            poster_scope_hashes=candidate.poster_scope_hashes,
            event_title=candidate.title,
        )
        if added_posters or pruned_posters:
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
        if preview_invalidated:
            note_log.append("3D-превью сброшено: изменились иллюстрации")

        # 4) Filters and text snippet are service notes
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
                keys = [k for k in updated_keys if k not in {"source_text", "source_texts"}]
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
        if schedule_tasks:
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
        if (updated_fields or added_posters or (added_sources and not same_source))
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
        merged=updated_fields,
        added_posters=added_posters,
        added_sources=added_sources,
        added_facts=added_facts,
        skipped_conflicts=skipped_conflicts,
        reason=match_reason if "match_reason" in locals() else None,
    )


async def _apply_posters(
    session,
    event_id: int | None,
    posters: Sequence[PosterCandidate],
    poster_scope_hashes: Sequence[str] | None = None,
    event_title: str | None = None,
) -> tuple[int, list[str], bool, int]:
    if not event_id:
        return 0, [], False, 0
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

    # 1) Exact prune: if the source provided the poster hash scope, drop any previously
    # attached posters from that scope that are not selected for this event now.
    if scope_hashes:
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
        # Prefer posters with OCR text/title (proxy for "quality")
        preferred_urls: list[str] = []
        scored: list[tuple[int, str]] = []
        for poster in posters:
            url = _pick_display_url(poster)
            if not url:
                continue
            score = 0
            if poster.ocr_title:
                score += len(poster.ocr_title)
            if poster.ocr_text:
                score += len(poster.ocr_text)
            if score > 0:
                scored.append((score, url))
        if scored:
            for _score, url in sorted(scored, key=lambda item: item[0], reverse=True):
                if url not in preferred_urls:
                    preferred_urls.append(url)
            reordered = preferred_urls + [url for url in current if url not in preferred_urls]
            current = reordered
        event.photo_urls = current
        event.photo_count = len(current)
        # If the image set changed, any existing 3D preview becomes stale: force regeneration.
        if (current != before_urls or len(current) != before_count) and getattr(event, "preview_3d_url", None):
            event.preview_3d_url = None
            preview_invalidated = True
        session.add(event)

    return added, added_urls, preview_invalidated, pruned


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
