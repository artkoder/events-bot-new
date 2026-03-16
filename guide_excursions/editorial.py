from __future__ import annotations

import logging
import os
import re
from typing import Any, Mapping, Sequence

from .parser import (
    PHONE_RE,
    URL_RE,
    USERNAME_RE,
    audience_line as build_audience_line,
    collapse_ws,
    has_public_invite_signal,
    looks_context_only,
    looks_operational_only,
)

logger = logging.getLogger(__name__)

GUIDE_EDITORIAL_ENABLED = (
    (os.getenv("GUIDE_EDITORIAL_ENABLED") or "1").strip().lower() in {"1", "true", "yes", "on"}
)


def _normalize_phone(raw: str) -> tuple[str, str] | None:
    value = collapse_ws(raw)
    if not value:
        return None
    digits = re.sub(r"[^\d]", "", value)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    display = value
    if len(digits) == 11 and digits.startswith("7"):
        display = f"+7 {digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
    elif len(digits) == 10:
        display = f"+7 {digits[0:3]} {digits[3:6]}-{digits[6:8]}-{digits[8:10]}"
        digits = "7" + digits
    elif digits.startswith("7"):
        display = f"+{digits}"
    return display, f"tel:+{digits}" if not str(digits).startswith("+") else f"tel:{digits}"


def _phone_digits(raw: str | None) -> str:
    return re.sub(r"[^\d]", "", collapse_ws(raw))


def _is_mobile_phone(raw: str | None) -> bool:
    digits = _phone_digits(raw)
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return len(digits) == 11 and digits.startswith("79")


def _normalize_line_text(value: Any, *, prefixes: Sequence[str] = ()) -> str | None:
    text = collapse_ws(value)
    if not text:
        return None
    for prefix in prefixes:
        normalized_prefix = collapse_ws(prefix)
        if not normalized_prefix:
            continue
        low = text.lower()
        prefix_low = normalized_prefix.lower()
        if low.startswith(prefix_low):
            text = text[len(normalized_prefix) :].lstrip(" :.-")
            text = collapse_ws(text)
            break
    return text or None


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        raw = collapse_ws(value)
        items = re.split(r"\s*[;,]\s*", raw) if raw else []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return out
    for item in items:
        text = collapse_ws(item)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_booking_label(label: str | None, url: str | None) -> str | None:
    text = collapse_ws(label)
    href = collapse_ws(url)
    if href:
        tg_match = re.search(r"(?:https?://)?t\.me/([A-Za-z0-9_]{4,64})", href, re.I)
        if tg_match:
            return f"@{tg_match.group(1)}"
        if href.startswith("tel:"):
            normalized = _normalize_phone(text or "")
            if normalized:
                return normalized[0]
            digits = href.removeprefix("tel:").strip()
            if digits:
                return digits
    if not text:
        return None
    if href and text.lower() in {"запись", "подробности", "подробнее", "звоните", "пишите"}:
        if href.startswith("tel:"):
            normalized = _normalize_phone(text)
            return normalized[0] if normalized else None
        return "Сайт / запись"
    return text


def _is_generic_booking_text(value: str | None) -> bool:
    text = collapse_ws(value).lower()
    if not text:
        return True
    return text in {
        "запись",
        "подробности",
        "подробнее",
        "звоните",
        "пишите",
        "ссылка",
        "ссылка для записи",
    }


def _group_format_seed(row: Mapping[str, Any]) -> str | None:
    direct = collapse_ws(str(row.get("group_format") or ""))
    if direct:
        return direct
    audience_values = _string_list(row.get("audience_fit"), limit=8)
    low_values = [value.lower() for value in audience_values]
    if any("школь" in value for value in low_values):
        return "для школьной группы"
    if any("для группы" in value or "групп" in value for value in low_values):
        return "для группы"
    return None


def _fact_pack_value(row: Mapping[str, Any], key: str) -> Any:
    fact_pack = row.get("fact_pack")
    if isinstance(fact_pack, Mapping):
        return fact_pack.get(key)
    return None


def _profile_fact_value(row: Mapping[str, Any], key: str) -> Any:
    profile_facts = row.get("guide_profile_facts")
    if isinstance(profile_facts, Mapping):
        return profile_facts.get(key)
    return None


def _looks_raw_price_copy(value: Any) -> bool:
    text = collapse_ws(value)
    if not text:
        return False
    if re.search(r"\d+\s*/\s*\d+", text):
        return True
    return any(token in text for token in (",пенсион", ", дети", "/с ", "₽/с"))


def _guide_line_seed(row: Mapping[str, Any]) -> str | None:
    profile_line = collapse_ws(str(_profile_fact_value(row, "guide_line") or ""))
    if profile_line:
        return profile_line
    guide_names = _string_list(row.get("guide_names"), limit=3)
    guide_names = [name for name in guide_names if len(name.split()) >= 2]
    if guide_names:
        return ", ".join(guide_names)
    return None


def _organizer_line_seed(row: Mapping[str, Any]) -> str | None:
    organizer_names = _string_list(row.get("organizer_names"), limit=3)
    if organizer_names:
        return ", ".join(organizer_names)
    source_title = collapse_ws(str(row.get("source_title") or ""))
    source_kind = collapse_ws(str(row.get("source_kind") or ""))
    if source_title and source_kind in {"organization_with_tours", "excursion_operator", "aggregator"}:
        return source_title
    return None


def build_booking_candidates(row: Mapping[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(label: str | None, url: str | None, kind: str) -> None:
        text = _normalize_booking_label(label, url)
        href = collapse_ws(url)
        if not text or not href:
            return
        key = (text, href)
        if key in seen:
            return
        seen.add(key)
        out.append({"text": text, "url": href, "kind": kind})

    explicit_text = collapse_ws(str(row.get("booking_text") or ""))
    explicit_url = collapse_ws(str(row.get("booking_url") or ""))
    if explicit_url:
        add(explicit_text or "Запись", explicit_url, "explicit")

    about_text = collapse_ws(str(row.get("source_about_text") or ""))
    about_links = row.get("source_about_links") or []
    if not isinstance(about_links, list):
        about_links = []

    for link in about_links:
        href = collapse_ws(str(link))
        if not href:
            continue
        match = re.search(r"(?:https?://)?t\.me/([A-Za-z0-9_]{4,64})", href, re.I)
        if match:
            uname = match.group(1)
            add(f"@{uname}", f"https://t.me/{uname}", "source_about")
    for phone in PHONE_RE.findall(about_text):
        normalized = _normalize_phone(phone)
        if normalized:
            add(normalized[0], normalized[1], "source_about")
    for uname in USERNAME_RE.findall(about_text):
        add(f"@{uname}", f"https://t.me/{uname}", "source_about")
    for href in URL_RE.findall(about_text):
        if "t.me/" in href.lower():
            continue
        add("Сайт / запись", href, "source_about")
    kind_rank = {"explicit": 0, "source_about": 1}

    def _candidate_rank(item: Mapping[str, str]) -> tuple[int, int, int, str]:
        href = collapse_ws(item.get("url"))
        text = collapse_ws(item.get("text"))
        kind = collapse_ws(item.get("kind"))
        tel = 0 if href.startswith("tel:") and _is_mobile_phone(text) else 1
        any_tel = 0 if href.startswith("tel:") else 1
        tg = 0 if "t.me/" in href.lower() else 1
        return (
            kind_rank.get(kind, 9),
            tel,
            any_tel,
            0 if not tg else 1,
            text.lower(),
        )

    out.sort(key=_candidate_rank)
    return out


def neutralize_relative_blurb(blurb: str | None, *, date_label: str | None = None, time_text: str | None = None) -> str | None:
    text = collapse_ws(blurb)
    if not text:
        return None
    replacement = collapse_ws(date_label)
    if replacement and time_text and time_text not in replacement:
        replacement = f"{replacement}, {time_text}"
    elif time_text and not replacement:
        replacement = time_text
    if replacement:
        text = re.sub(r"(?i)\bзавтра\b", replacement, text)
        text = re.sub(r"(?i)\bсегодня\b", replacement, text)
        text = re.sub(r"(?i)\bвчера\b", replacement, text)
    return text


def repair_title_fallback(title: str | None, *, source_excerpt: str | None = None) -> str | None:
    value = collapse_ws(title)
    if not value:
        return None
    excerpt = collapse_ws(source_excerpt)
    low_excerpt = excerpt.lower()
    if value[:1].islower():
        if f"путешествие на {value.lower()}" in low_excerpt:
            return f"Путешествие на {value}"
        if f"экскурсия на {value.lower()}" in low_excerpt:
            return f"Экскурсия на {value}"
        return value[:1].upper() + value[1:]
    if value.lower().startswith("по зеленоградску") and "расширенная экскурсия по зеленоградску" in low_excerpt:
        return "Расширенная экскурсия по Зеленоградску"
    return value


def _fallback_booking_line(row: Mapping[str, Any]) -> str | None:
    candidates = build_booking_candidates(row)
    if candidates:
        return candidates[0]["text"]
    booking_text = _normalize_booking_label(row.get("booking_text"), row.get("booking_url"))
    if booking_text:
        return booking_text
    booking_url = collapse_ws(str(row.get("booking_url") or ""))
    if booking_url:
        if "t.me/" in booking_url.lower():
            match = re.search(r"t\.me/([A-Za-z0-9_]{4,64})", booking_url, flags=re.I)
            if match:
                return f"@{match.group(1)}"
        if booking_url.startswith("tel:"):
            return None
        return "Сайт / запись"
    return None


def _build_fallback_line_fields(row: Mapping[str, Any], *, date_label: str | None = None) -> dict[str, str]:
    audience_values = [
        value
        for value in _string_list(row.get("audience_fit"), limit=6)
        if "для группы" not in value.lower() and "групп" not in value.lower()
    ]
    audience = build_audience_line(audience_values)
    raw_price_line = _normalize_line_text(row.get("price_text"), prefixes=("цена", "стоимость"))
    result = {
        "guide_line": _guide_line_seed(row),
        "organizer_line": _organizer_line_seed(row),
        "schedule_line": collapse_ws(date_label) or collapse_ws(str(row.get("time") or "")) or None,
        "audience_line": audience or None,
        "group_format_line": _group_format_seed(row),
        "route_line": _normalize_line_text(
            row.get("route_summary") or _fact_pack_value(row, "route_summary"),
            prefixes=("маршрут", "в программе", "что увидим", "посетим"),
        ),
        "duration_line": _normalize_line_text(
            row.get("duration_text") or _fact_pack_value(row, "duration_text"),
            prefixes=("продолжительность", "длительность"),
        ),
        "meeting_point_line": _normalize_line_text(row.get("meeting_point"), prefixes=("встреча", "место")),
        "price_line": None if _looks_raw_price_copy(raw_price_line) else raw_price_line,
        "seats_line": _normalize_line_text(row.get("seats_text"), prefixes=("места", "статус")),
        "booking_line": _normalize_line_text(_fallback_booking_line(row), prefixes=("запись",)),
    }
    return {key: value for key, value in result.items() if value}


def apply_editorial_fallback(row: Mapping[str, Any], *, date_label: str | None = None) -> tuple[dict[str, Any] | None, str]:
    item = dict(row)
    booking_candidates = build_booking_candidates(item)
    if booking_candidates and (
        not collapse_ws(str(item.get("booking_url") or ""))
        or _is_generic_booking_text(item.get("booking_text"))
        or len(collapse_ws(str(item.get("booking_text") or ""))) > 48
    ):
        item["booking_text"] = booking_candidates[0]["text"]
        item["booking_url"] = booking_candidates[0]["url"]
    excerpt = collapse_ws(str(item.get("dedup_source_text") or ""))
    if looks_context_only(excerpt):
        return None, "context_only"
    if looks_operational_only(excerpt) and not collapse_ws(str(item.get("booking_url") or "")):
        return None, "operational_without_booking"
    if not has_public_invite_signal(excerpt) and not collapse_ws(str(item.get("booking_url") or "")):
        return None, "weak_public_signal_without_booking"
    item["canonical_title"] = repair_title_fallback(item.get("canonical_title"), source_excerpt=excerpt) or item.get("canonical_title")
    item["digest_blurb"] = neutralize_relative_blurb(
        str(item.get("digest_blurb") or item.get("summary_one_liner") or ""),
        date_label=date_label,
        time_text=collapse_ws(str(item.get("time") or "")) or None,
    )
    item.update(_build_fallback_line_fields(item, date_label=date_label))
    return item, "fallback"


async def refine_digest_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    date_formatter: Any,
) -> tuple[list[dict[str, Any]], list[int], dict[int, str]]:
    del family
    if not GUIDE_EDITORIAL_ENABLED:
        kept = [dict(row) for row in rows]
        return kept, [], {}

    kept: list[dict[str, Any]] = []
    suppressed_ids: list[int] = []
    reasons: dict[int, str] = {}
    for row in rows:
        occurrence_id = int(row.get("id") or 0)
        date_label = date_formatter(str(row.get("date") or ""), str(row.get("time") or ""))
        fallback_item, fallback_reason = apply_editorial_fallback(row, date_label=date_label)
        if fallback_item is None:
            suppressed_ids.append(occurrence_id)
            reasons[occurrence_id] = fallback_reason
            continue
        kept.append(fallback_item)
    return kept, suppressed_ids, reasons
