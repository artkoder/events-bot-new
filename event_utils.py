import re
import html
import unicodedata
from datetime import datetime, timezone, timedelta, time, date
from models import Event, Festival
from digest_helper import (
    clean_search_digest,
    clean_short_description,
    fallback_one_sentence,
    is_short_description_acceptable,
)

# Kaliningrad timezone (UTC+2)
LOCAL_TZ = timezone(timedelta(hours=2))
_TAG_RE = re.compile(r"<[^>]+>")
_MONTHS = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]

def _parse_iso_date(text: str) -> date | None:
    try:
        return date.fromisoformat(text)
    except Exception:
        return None

def _format_day_pretty(day: date) -> str:
    return f"{day.day} {_MONTHS[day.month - 1]}"

def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def recent_cutoff(tz: timezone, now: datetime | None = None) -> datetime:
    """Return UTC datetime for the start of the previous day in the given tz."""
    if now is None:
        now_local = datetime.now(tz)
    else:
        now_local = _ensure_utc(now).astimezone(tz)
    start_local = datetime.combine(
        now_local.date() - timedelta(days=1),
        time(0, 0),
        tz,
    )
    return start_local.astimezone(timezone.utc)

def is_recent(e: Event, tz: timezone | None = None, now: datetime | None = None) -> bool:
    if e.added_at is None or e.silent or getattr(e, "lifecycle_status", "active") != "active":
        return False
    if tz is None:
        tz = LOCAL_TZ
    start = recent_cutoff(tz, now)
    added_at = _ensure_utc(e.added_at)
    return added_at >= start

def _normalize_title_and_emoji(title: str, emoji: str | None) -> tuple[str, str]:
    """Ensure the emoji prefix is applied only once per rendered line."""

    if not emoji:
        return title, ""

    trimmed_title = title.lstrip()
    if trimmed_title.startswith(emoji):
        trimmed_title = trimmed_title[len(emoji) :].lstrip()

    return trimmed_title or title.strip(), f"{emoji} "

def strip_city_from_address(address: str | None, city: str | None) -> str | None:
    """Remove the city name from the end of the address if duplicated."""
    if not address or not city:
        return address
    city_clean = city.lstrip("#").strip().lower()
    addr = address.strip()
    if addr.lower().endswith(city_clean):
        addr = re.sub(r",?\s*#?%s$" % re.escape(city_clean), "", addr, flags=re.IGNORECASE)
    # Compact common Russian address noise: "ул." prefix and comma separators.
    addr = addr.rstrip(", ")
    addr = re.sub(r"\s*,\s*", " ", addr)
    addr = re.sub(r"(?i)^\s*(?:ул\.?|улица)\s+", "", addr).strip()
    addr = re.sub(r"\s{2,}", " ", addr).strip()
    return addr


def _normalize_location_fragment(part: str | None) -> str:
    if not part:
        return ""
    normalized = unicodedata.normalize("NFKC", str(part))
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.casefold()


def _contains_token_subsequence(
    haystack: list[str], needle: list[str]
) -> bool:
    if not haystack or not needle or len(needle) > len(haystack):
        return False
    for idx in range(len(haystack) - len(needle) + 1):
        if haystack[idx : idx + len(needle)] == needle:
            return True
    return False


def _location_fragment_has_number(tokens: list[str]) -> bool:
    return any(any(ch.isdigit() for ch in token) for token in tokens)


def _location_name_already_contains_address(
    location_name: str | None,
    location_address: str | None,
) -> bool:
    name_norm = _normalize_location_fragment(location_name)
    addr_norm = _normalize_location_fragment(location_address)
    if not name_norm or not addr_norm:
        return False
    if addr_norm == name_norm:
        return True
    if len(addr_norm) >= 8 and addr_norm in name_norm:
        return True

    addr_tokens = addr_norm.split()
    name_fragments = [str(location_name or "").strip()]
    name_fragments.extend(
        fragment.strip()
        for fragment in str(location_name or "").split(",")
        if fragment.strip()
    )
    for fragment in name_fragments:
        fragment_norm = _normalize_location_fragment(fragment)
        if not fragment_norm:
            continue
        if fragment_norm == addr_norm:
            return True
        if len(addr_norm) >= 8 and addr_norm in fragment_norm:
            return True
        fragment_tokens = fragment_norm.split()
        shorter_tokens = fragment_tokens
        longer_tokens = addr_tokens
        if len(fragment_tokens) > len(addr_tokens):
            shorter_tokens = addr_tokens
            longer_tokens = fragment_tokens
        if (
            len(shorter_tokens) >= 2
            and _location_fragment_has_number(shorter_tokens)
            and _contains_token_subsequence(longer_tokens, shorter_tokens)
        ):
            return True
    return False


def _compose_event_location(
    location_name: str | None,
    location_address: str | None,
    city: str | None,
) -> str:
    name = str(location_name or "").strip()
    city_value = str(city or "").lstrip("#").strip()
    address = str(location_address or "").strip()
    if address and city_value:
        address = strip_city_from_address(address, city_value) or ""

    name_norm = _normalize_location_fragment(name)
    address_norm = _normalize_location_fragment(address)
    city_norm = _normalize_location_fragment(city_value)

    parts: list[str] = []
    if name:
        parts.append(name)
    if address and not _location_name_already_contains_address(name, address):
        parts.append(address)
    if city_value and city_norm and len(city_norm) >= 4:
        if city_norm not in name_norm and city_norm not in address_norm:
            parts.append(city_value)
    elif city_value:
        parts.append(city_value)

    return ", ".join(part for part in parts if part)


def format_event_md(
    e: Event,
    festival: Festival | None = None,
    *,
    include_ics: bool = True,
    include_details: bool = True,
) -> str:
    prefix = ""
    if is_recent(e):
        prefix += "\U0001f6a9 "
    title_text, emoji_part = _normalize_title_and_emoji(e.title, e.emoji)
    lines = [f"{prefix}{emoji_part}{title_text}".strip()]
    if festival:
        link = festival.telegraph_url
        if link:
            lines.append(f"[{festival.name}]({link})")
        else:
            lines.append(festival.name)
    digest = clean_short_description(getattr(e, "short_description", None))
    if digest and not is_short_description_acceptable(digest, min_words=12, max_words=16):
        digest = fallback_one_sentence(digest, max_words=16)
    if not digest:
        digest = clean_search_digest(getattr(e, "search_digest", None))
        if digest:
            digest = fallback_one_sentence(digest, max_words=16)
    if not digest:
        digest = fallback_one_sentence(getattr(e, "description", None), max_words=16)
    if not digest:
        digest = str(getattr(e, "description", "") or "").strip()
    if digest:
        lines.append(digest)
    if e.pushkin_card:
        lines.append("\u2705 Пушкинская карта")
    if getattr(e, "ticket_status", None) == "sold_out":
        lines.append("❌ Билеты все проданы")
    elif e.is_free:
        txt = "🟡 Бесплатно"
        if e.ticket_link:
            txt += f" [по регистрации]({e.ticket_link})"
        lines.append(txt)
    elif e.ticket_link and (
        e.ticket_price_min is not None or e.ticket_price_max is not None
    ):
        status_icon = "✅ " if getattr(e, "ticket_status", None) == "available" else ""
        if e.ticket_price_max is not None and e.ticket_price_max != e.ticket_price_min:
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        else:
            price = str(e.ticket_price_min or e.ticket_price_max or "")
        lines.append(f"{status_icon}[Билеты в источнике]({e.ticket_link}) {price}".strip())
    elif e.ticket_link:
        status_icon = "✅ " if getattr(e, "ticket_status", None) == "available" else ""
        lines.append(f"{status_icon}[по регистрации]({e.ticket_link})")
    else:
        if (
            e.ticket_price_min is not None
            and e.ticket_price_max is not None
            and e.ticket_price_min != e.ticket_price_max
        ):
            price = f"от {e.ticket_price_min} до {e.ticket_price_max}"
        elif e.ticket_price_min is not None:
            price = str(e.ticket_price_min)
        elif e.ticket_price_max is not None:
            price = str(e.ticket_price_max)
        else:
            price = ""
        if price:
            status_icon = "✅ " if getattr(e, "ticket_status", None) == "available" else ""
            lines.append(f"{status_icon}Билеты {price}")
    if include_details and e.telegraph_url:
        cam = "\U0001f4f8" * min(2, max(0, e.photo_count))
        prefix = f"{cam} " if cam else ""
        more_line = f"{prefix}[подробнее]({e.telegraph_url})"
        ics = e.ics_url or e.ics_post_url
        if include_ics and ics:
            more_line += f" \U0001f4c5 [добавить в календарь]({ics})"
        lines.append(more_line)
    loc = _compose_event_location(
        e.location_name,
        e.location_address,
        e.city,
    )

    date_part = ""
    if e.date:
        date_part = e.date.split("..", 1)[0]
    d = _parse_iso_date(date_part)
    if d:
        day = _format_day_pretty(d)
    else:
        day = date_part or e.date or ""

    time_part = ""
    if e.time and e.time != "00:00":
        time_part = f" {e.time}"

    if day:
        lines.append(f"\U0001f4c5 {day}{time_part}".strip())
    if loc:
        lines.append(f"\U0001f4cd {loc}".strip())
    
    return "\n".join(lines)
