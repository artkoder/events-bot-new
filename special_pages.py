"""
Special Pages module for generating holiday/event Telegraph pages.

This module provides functionality to build Telegraph pages for special periods
(holidays, festivals, etc.) with event deduplication and image support.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Iterable

from sqlalchemy import select

if TYPE_CHECKING:
    from db import Database
    from models import Event, Festival


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELEGRAPH_LIMIT = 45000
MAX_DAYS_DEFAULT = 14

# Emoji regex for normalization
_EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Data Structures for Event Deduplication
# ---------------------------------------------------------------------------

@dataclass
class SpecialEventSlot:
    """Represents a single time slot for an event."""
    time: str | None  # "18:00" or None if not specified
    ticket_line: str  # "бесплатно" / "от 500₽" / "регистрация" / ""
    source_event: "Event"


@dataclass
class SpecialEventGroup:
    """
    A group of events with the same title on the same day.
    Multiple time slots are merged into a single group.
    """
    day: date
    title: str  # normalized title
    original_title: str  # original title with emoji
    description: str  # description or search_digest
    photo_url: str | None
    link_url: str | None  # telegraph_url or source_post_url
    ics_url: str | None
    location: str  # "Place, address, city"
    slots: list[SpecialEventSlot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Title Normalization
# ---------------------------------------------------------------------------

def normalize_title(title: str) -> str:
    """
    Normalize title for grouping:
    - Remove leading emoji
    - Casefold
    - Collapse whitespace
    - Strip
    """
    # Remove emoji
    text = _EMOJI_RE.sub("", title)
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Casefold for case-insensitive comparison
    text = text.casefold()
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


# ---------------------------------------------------------------------------
# Ticket Line Formatting
# ---------------------------------------------------------------------------

def format_ticket_line(event: "Event") -> str:
    """
    Format a compact ticket line for an event slot.
    Returns: "бесплатно" / "от X₽" / "X₽" / "регистрация" / ""
    """
    if getattr(event, "ticket_status", None) == "sold_out":
        return "билеты проданы"
    
    if event.is_free:
        if event.ticket_link:
            return "бесплатно, регистрация"
        return "бесплатно"
    
    price_min = event.ticket_price_min
    price_max = event.ticket_price_max
    
    if price_min is not None and price_max is not None and price_min != price_max:
        return f"от {price_min}₽"
    elif price_min is not None:
        return f"{price_min}₽"
    elif price_max is not None:
        return f"{price_max}₽"
    elif event.ticket_link:
        return "регистрация"
    
    return ""


# ---------------------------------------------------------------------------
# Location Formatting
# ---------------------------------------------------------------------------

def format_location(event: "Event") -> str:
    """Format location string: 'Place, address, city'."""
    parts = [event.location_name]
    
    addr = event.location_address
    if addr and event.city:
        # Strip city from address if it's at the end
        city_lower = event.city.lower()
        if addr.lower().endswith(city_lower):
            addr = addr[: -len(event.city)].rstrip(", ")
        elif addr.lower().endswith(f"г. {city_lower}"):
            addr = addr[: -len(f"г. {event.city}")].rstrip(", ")
    
    if addr:
        parts.append(addr)
    if event.city:
        parts.append(event.city)
    
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Event Grouping
# ---------------------------------------------------------------------------

def group_events_for_special(
    events: list["Event"],
) -> dict[date, list[SpecialEventGroup]]:
    """
    Group events by (date, normalized_title).
    Events with same title on same day are merged into one group
    with multiple time slots.
    """
    from main import parse_iso_date
    
    # Key: (date, normalized_title) -> list of events
    groups: dict[tuple[date, str], list["Event"]] = {}
    
    for e in events:
        date_str = e.date.split("..", 1)[0] if e.date else None
        if not date_str:
            continue
        d = parse_iso_date(date_str)
        if not d:
            continue
        
        norm_title = normalize_title(e.title)
        key = (d, norm_title)
        groups.setdefault(key, []).append(e)
    
    # Build SpecialEventGroup for each group
    result: dict[date, list[SpecialEventGroup]] = {}
    
    for (day, norm_title), event_list in groups.items():
        # Sort by time
        event_list.sort(key=lambda x: x.time or "99:99")
        
        # Pick first event as "main" for description, photo, etc.
        main = event_list[0]
        
        # Description: prefer non-empty description, else search_digest
        description = ""
        for e in event_list:
            if e.description and e.description.strip():
                description = e.description.strip()
                break
        if not description:
            for e in event_list:
                if e.search_digest and e.search_digest.strip():
                    description = e.search_digest.strip()
                    break
        
        # Photo URL: first valid photo
        photo_url = None
        for e in event_list:
            if e.photo_urls:
                url = e.photo_urls[0]
                if isinstance(url, str) and url.startswith("http"):
                    photo_url = url
                    break
        
        # Link URL: telegraph_url > source_post_url
        link_url = None
        for e in event_list:
            if e.telegraph_url:
                link_url = e.telegraph_url
                break
            if e.source_post_url and not link_url:
                link_url = e.source_post_url
        
        # ICS URL
        ics_url = None
        for e in event_list:
            if e.ics_url:
                ics_url = e.ics_url
                break
            if e.ics_post_url and not ics_url:
                ics_url = e.ics_post_url
        
        # Build slots
        slots = []
        for e in event_list:
            time_val = e.time if e.time and e.time != "00:00" else None
            ticket_line = format_ticket_line(e)
            slots.append(SpecialEventSlot(
                time=time_val,
                ticket_line=ticket_line,
                source_event=e
            ))
        
        group = SpecialEventGroup(
            day=day,
            title=norm_title,
            original_title=main.title,
            description=description,
            photo_url=photo_url,
            link_url=link_url,
            ics_url=ics_url,
            location=format_location(main),
            slots=slots,
        )
        
        result.setdefault(day, []).append(group)
    
    # Sort groups within each day by earliest time
    for day in result:
        result[day].sort(key=lambda g: g.slots[0].time or "99:99" if g.slots else "99:99")
    
    return result


# ---------------------------------------------------------------------------
# Telegraph Node Rendering
# ---------------------------------------------------------------------------

def telegraph_br() -> list[dict]:
    """Return Telegraph line break nodes."""
    return [{"tag": "p", "children": [{"tag": "br", "children": []}]}]


def render_special_group(
    group: SpecialEventGroup,
    show_image: bool = True,
) -> list[dict]:
    """
    Render a SpecialEventGroup to Telegraph nodes.
    
    Format:
    - [image]
    - title (h4)
    - description
    - "Подробнее" + "Добавить в календарь" links
    - Date
    - time1 — ticket_line
    - time2 — ticket_line
    - ...
    - Location
    """
    from main import format_day_pretty
    
    nodes: list[dict] = []
    
    # Image
    if show_image and group.photo_url:
        nodes.append({
            "tag": "figure",
            "children": [
                {"tag": "img", "attrs": {"src": group.photo_url}, "children": []}
            ]
        })
    
    # Title (h4) with link if available
    title_children: list = []
    if group.link_url:
        title_children.append({
            "tag": "a",
            "attrs": {"href": group.link_url},
            "children": [group.original_title]
        })
    else:
        title_children.append(group.original_title)
    nodes.append({"tag": "h4", "children": title_children})
    
    # Description
    if group.description:
        nodes.append({"tag": "p", "children": [group.description]})
    
    # Links: "Подробнее" + "Добавить в календарь"
    link_children: list = []
    if group.link_url:
        link_children.append({
            "tag": "a",
            "attrs": {"href": group.link_url},
            "children": ["Подробнее"]
        })
    if group.ics_url:
        if link_children:
            link_children.append(" 📅 ")
        link_children.append({
            "tag": "a",
            "attrs": {"href": group.ics_url},
            "children": ["Добавить в календарь"]
        })
    if link_children:
        nodes.append({"tag": "p", "children": link_children})
    
    # Date
    date_str = format_day_pretty(group.day)
    nodes.append({"tag": "p", "children": [f"📅 {date_str}"]})
    
    # Time slots
    for slot in group.slots:
        time_str = slot.time or "время уточняется"
        line = f"🕐 {time_str}"
        if slot.ticket_line:
            # Add ticket link if available
            source_event = slot.source_event
            ticket_url = source_event.ticket_link
            if ticket_url:
                line_children: list = [f"🕐 {time_str} — "]
                line_children.append({
                    "tag": "a",
                    "attrs": {"href": ticket_url},
                    "children": [slot.ticket_line]
                })
                nodes.append({"tag": "p", "children": line_children})
            else:
                nodes.append({"tag": "p", "children": [f"{line} — {slot.ticket_line}"]})
        else:
            nodes.append({"tag": "p", "children": [line]})
    
    # Location
    if group.location:
        nodes.append({"tag": "p", "children": [f"📍 {group.location}"]})
    
    # Separator
    nodes.extend(telegraph_br())
    
    return nodes


# ---------------------------------------------------------------------------
# Content Building
# ---------------------------------------------------------------------------

def rough_size(nodes: Iterable[dict], limit: int | None = None) -> int:
    """
    Estimate the size of Telegraph content nodes.
    Import from main to use the same logic.
    """
    from main import rough_size as _rough_size
    return _rough_size(nodes, limit)


async def build_special_page_content(
    db: "Database",
    start_date: date,
    days: int,
    cover_url: str | None,
    title: str,
    size_limit: int = TELEGRAPH_LIMIT,
) -> tuple[str, list[dict], int, int]:
    """
    Build Telegraph page content for a special period.
    
    Args:
        db: Database instance
        start_date: Start date of the period
        days: Number of days to include
        cover_url: Cover image URL (optional)
        title: Page title
        size_limit: Maximum content size
    
    Returns:
        (page_title, content_nodes, content_size, used_days)
    """
    from models import Event, Festival
    from main import ensure_event_telegraph_link, format_day_pretty, LOCAL_TZ
    from datetime import datetime
    
    original_days = days
    
    while days >= 1:
        end_date = start_date + timedelta(days=days - 1)
        date_range = [
            (start_date + timedelta(days=i)).isoformat()
            for i in range(days)
        ]
        
        # Fetch events
        async with db.get_session() as session:
            result = await session.execute(
                select(Event)
                .where(Event.date.in_(date_range))
                .order_by(Event.date, Event.time)
            )
            events = list(result.scalars().all())
            
            # Fetch exhibitions that overlap with the period
            ex_result = await session.execute(
                select(Event)
                .where(
                    Event.event_type == "выставка",
                    Event.end_date.is_not(None),
                    Event.date <= end_date.isoformat(),
                    Event.end_date >= start_date.isoformat(),
                )
                .order_by(Event.date)
            )
            exhibitions = list(ex_result.scalars().all())
            
            # Festival map for links
            fest_result = await session.execute(select(Festival))
            fest_map = {f.name.casefold(): f for f in fest_result.scalars().all()}
        
        # Filter out past events
        today = datetime.now(LOCAL_TZ).date()
        events = [
            e for e in events
            if (e.end_date and e.end_date >= today.isoformat())
            or (not e.end_date and e.date >= today.isoformat())
        ]
        
        # Ensure telegraph links
        for e in events:
            fest = fest_map.get((e.festival or "").casefold())
            await ensure_event_telegraph_link(e, fest, db)
        
        # Group events
        grouped = group_events_for_special(events)
        
        # Build content
        content: list[dict] = []
        
        # Cover image
        if cover_url:
            content.append({
                "tag": "figure",
                "children": [
                    {"tag": "img", "attrs": {"src": cover_url}, "children": []}
                ]
            })
            content.extend(telegraph_br())
        
        # Title as h3
        content.append({"tag": "h3", "children": [title]})
        content.extend(telegraph_br())
        
        # Events by day
        sorted_days = sorted(grouped.keys())
        show_images = len(events) < 15  # Show images only if not too many events
        
        for day in sorted_days:
            # Day header
            day_str = format_day_pretty(day)
            weekday_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
            weekday = weekday_names[day.weekday()]
            content.append({"tag": "h3", "children": [f"{day_str}, {weekday}"]})
            
            # Events for this day
            for group in grouped[day]:
                content.extend(render_special_group(group, show_image=show_images))
        
        # Exhibitions section
        if exhibitions:
            content.append({"tag": "h3", "children": ["Постоянные выставки"]})
            content.extend(telegraph_br())
            for ex in exhibitions:
                # Simple exhibition rendering
                ex_nodes = []
                if show_images and ex.photo_urls:
                    url = ex.photo_urls[0]
                    if isinstance(url, str) and url.startswith("http"):
                        ex_nodes.append({
                            "tag": "figure",
                            "children": [{"tag": "img", "attrs": {"src": url}, "children": []}]
                        })
                
                title_children: list = []
                if ex.telegraph_url:
                    title_children.append({
                        "tag": "a",
                        "attrs": {"href": ex.telegraph_url},
                        "children": [ex.title]
                    })
                else:
                    title_children.append(ex.title)
                ex_nodes.append({"tag": "h4", "children": title_children})
                
                if ex.description:
                    ex_nodes.append({"tag": "p", "children": [ex.description.strip()]})
                
                loc = format_location(ex)
                if loc:
                    ex_nodes.append({"tag": "p", "children": [f"📍 {loc}"]})
                
                ex_nodes.extend(telegraph_br())
                content.extend(ex_nodes)
        
        # Check size
        size = rough_size(content)
        if size <= size_limit:
            page_title = f"{title} — {format_day_pretty(start_date)}"
            if days > 1:
                page_title = f"{title}"
            return page_title, content, size, days
        
        # Size exceeded, reduce days
        days -= 1
        logging.info(
            "special_page: size %d > limit %d, reducing to %d days",
            size, size_limit, days
        )
    
    # Even 1 day doesn't fit - try without images
    logging.warning("special_page: even 1 day exceeds limit, building without images")
    
    # Rebuild without images
    content = []
    if cover_url:
        content.append({
            "tag": "figure",
            "children": [{"tag": "img", "attrs": {"src": cover_url}, "children": []}]
        })
    content.append({"tag": "h3", "children": [title]})
    
    for day in sorted(grouped.keys()):
        day_str = format_day_pretty(day)
        weekday_names = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        weekday = weekday_names[day.weekday()]
        content.append({"tag": "h3", "children": [f"{day_str}, {weekday}"]})
        
        for group in grouped[day]:
            content.extend(render_special_group(group, show_image=False))
    
    size = rough_size(content)
    page_title = title
    return page_title, content, size, 1


async def create_special_telegraph_page(
    db: "Database",
    start_date: date,
    days: int,
    cover_url: str | None,
    title: str,
) -> tuple[str, int]:
    """
    Create a Telegraph page for the special period.
    
    Returns:
        (telegraph_url, used_days)
    """
    from main import telegraph_create_page, get_telegraph
    
    page_title, content, size, used_days = await build_special_page_content(
        db, start_date, days, cover_url, title
    )
    
    tg = get_telegraph()
    result = await telegraph_create_page(tg, page_title, content)
    url = result.get("url", "")
    
    logging.info(
        "special_page created: url=%s size=%d days=%d",
        url, size, used_days
    )
    
    return url, used_days
