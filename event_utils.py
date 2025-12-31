import re
import html
from datetime import datetime, timezone, timedelta, time
from models import Event, Festival

# Kaliningrad timezone (UTC+2)
LOCAL_TZ = timezone(timedelta(hours=2))
_TAG_RE = re.compile(r"<[^>]+>")

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
    if e.added_at is None or e.silent:
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
    addr = addr.rstrip(", ")
    return addr

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
    lines.append(e.description.strip())
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
    loc = e.location_name
    addr = e.location_address
    if addr and e.city:
        addr = strip_city_from_address(addr, e.city)
    if addr:
        loc += f", {addr}"
    lines.append(f"📍 {loc}")
    
    return "\n".join(lines)
