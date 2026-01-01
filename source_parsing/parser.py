"""Parser for theatre event data from Kaggle notebook outputs."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Location name mappings from source to database
TRETYAKOV_LOCATION = "Филиал Третьяковской галереи, Парадная наб. 3, #Калининград"

LOCATION_MAPPINGS = {
    "кафедральный собор": "Кафедральный собор",
    "драматический театр": "Драматический театр",
    "музыкальный театр": "Музыкальный театр",
    "третьяков": TRETYAKOV_LOCATION,
    "третяков": TRETYAKOV_LOCATION,
    "tretyakov": TRETYAKOV_LOCATION,
    "атриум": TRETYAKOV_LOCATION,
    "кинозал": TRETYAKOV_LOCATION,
}

# Russian month names for date parsing
MONTHS_RU = {
    "января": 1, "янв": 1,
    "февраля": 2, "фев": 2,
    "марта": 3, "мар": 3,
    "апреля": 4, "апр": 4,
    "мая": 5,
    "июня": 6, "июн": 6,
    "июля": 7, "июл": 7,
    "августа": 8, "авг": 8,
    "сентября": 9, "сен": 9, "сент": 9,
    "октября": 10, "окт": 10,
    "ноября": 11, "ноя": 11,
    "декабря": 12, "дек": 12, "декабр": 12,
}

_TIME_RANGE_SPLIT = re.compile(r"\s*(?:-|–|—|\.\.\.?|…)\s*")
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
_TITLE_CLEAN_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_TITLE_SPACE_RE = re.compile(r"\s+")
_TITLE_NOISE_WORDS = {
    "спектакль",
    "мюзикл",
    "опера",
    "балет",
    "премьера",
    "оперетта",
    "музыкальная",
    "музыкальный",
}
_TITLE_MIN_TOKEN_LEN = 3


def extract_time_start(value: str | None) -> str | None:
    """Extract start time (HH:MM) from a time string or range."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parts = _TIME_RANGE_SPLIT.split(text, maxsplit=1)
    candidate = parts[0] if parts else text
    match = _TIME_RE.search(candidate)
    if not match:
        match = _TIME_RE.search(text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return f"{hour:02d}:{minute:02d}"


def _normalize_title(text: str) -> str:
    if not text:
        return ""
    normalized = (
        text.strip()
        .lower()
        .replace("ё", "е")
        .replace("\u00a0", " ")
    )
    normalized = _TITLE_CLEAN_RE.sub(" ", normalized)
    normalized = _TITLE_SPACE_RE.sub(" ", normalized).strip()
    return normalized


def _title_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [
        token
        for token in text.split()
        if len(token) >= _TITLE_MIN_TOKEN_LEN
    ]


def _strip_noise_tokens(tokens: Sequence[str]) -> list[str]:
    if not tokens:
        return []
    return [token for token in tokens if token not in _TITLE_NOISE_WORDS]


def _tokens_subset_match(tokens1: Sequence[str], tokens2: Sequence[str]) -> bool:
    if not tokens1 or not tokens2:
        return False
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 or not set2:
        return False
    return set1.issubset(set2) or set2.issubset(set1)


@dataclass
class TheatreEvent:
    """Parsed event from theatre source."""
    title: str
    date_raw: str
    ticket_status: str  # 'available' or 'sold_out'
    url: str
    photos: list[str] = field(default_factory=list)
    description: str = ""
    pushkin_card: bool = False
    location: str = ""
    age_restriction: str = ""
    scene: str = ""
    source_type: str = ""  # 'main', 'sobor', etc.
    
    # Parsed date/time
    parsed_date: Optional[str] = None  # ISO format YYYY-MM-DD
    parsed_time: Optional[str] = None  # HH:MM format


def parse_date_raw(date_raw: str) -> tuple[Optional[str], Optional[str]]:
    """Parse Russian date string into ISO date and time.
    
    Examples:
        "28 декабря 18:00" -> ("2024-12-28", "18:00")
        "02 ЯНВАРЯ 13:00" -> ("2025-01-02", "13:00")
        "28 ДЕКАБР" -> ("2024-12-28", None)
        "21.03.2026 18:00" -> ("2026-03-21", "18:00")
    """
    if not date_raw:
        return None, None
    
    text = date_raw.strip().lower()
    
    # Extract time if present (HH:MM format)
    time_match = re.search(r'(\d{1,2}):(\d{2})', text)
    parsed_time = None
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        parsed_time = f"{hour:02d}:{minute:02d}"
    
    # Try DD.MM.YYYY format first (e.g., "21.03.2026")
    numeric_date_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', text)
    if numeric_date_match:
        day = int(numeric_date_match.group(1))
        month = int(numeric_date_match.group(2))
        year = int(numeric_date_match.group(3))
        try:
            event_date = date(year, month, day)
            return event_date.isoformat(), parsed_time
        except ValueError:
            pass  # Invalid date, fall through to Russian month parsing
    
    # Extract day and month
    day_match = re.search(r'(\d{1,2})\s*', text)
    if not day_match:
        return None, parsed_time
    
    day = int(day_match.group(1))
    
    # Find month name
    month = None
    for month_name, month_num in MONTHS_RU.items():
        if month_name in text:
            month = month_num
            break
    
    if month is None:
        return None, parsed_time
    
    # Determine year (assume current or next year)
    today = date.today()
    year = today.year
    
    # If date is in the past and month is earlier, use next year
    try:
        event_date = date(year, month, day)
        if event_date < today:
            # If it's more than 2 months in the past, assume next year
            if (today - event_date).days > 60:
                year += 1
                event_date = date(year, month, day)
    except ValueError:
        return None, parsed_time
    
    return event_date.isoformat(), parsed_time


def parse_theatre_json(json_data: str | list | dict, source_name: str = "") -> list[TheatreEvent]:
    """Parse JSON data from theatre source into TheatreEvent objects.
    
    Args:
        json_data: Raw JSON string or parsed list/dict
        source_name: Source identifier (dramteatr, muzteatr, sobor)
    
    Returns:
        List of parsed TheatreEvent objects
    """
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from %s: %s", source_name, e)
            return []
    else:
        data = json_data
    
    if isinstance(data, dict):
        data = [data]
    
    events: list[TheatreEvent] = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
        
        title = item.get("title", "").strip()
        if not title:
            continue
        
        date_raw = item.get("date_raw", "")
        
        # Use pre-parsed date/time if available (e.g. from specific parsers)
        if item.get("parsed_date"):
            parsed_date = item.get("parsed_date")
            parsed_time = item.get("parsed_time")
        else:
            parsed_date, parsed_time = parse_date_raw(date_raw)
        
        # Map ticket status
        raw_status = item.get("ticket_status", "").lower()
        ticket_status = "available" if raw_status == "available" else "sold_out" if raw_status in ("sold_out", "soldout", "sold out") else "unknown"
        
        location = item.get("location", "") or ""
        if source_name.lower() == "tretyakov" and not location:
            location = TRETYAKOV_LOCATION

        event = TheatreEvent(
            title=title,
            date_raw=date_raw,
            ticket_status=ticket_status,
            url=item.get("url", ""),
            photos=item.get("photos", []) or [],
            description=item.get("description", "") or "",
            pushkin_card=bool(item.get("pushkin_card", False)),
            location=location,
            age_restriction=item.get("age_restriction", "") or "",
            scene=item.get("scene", "") or "",
            source_type=source_name,
            parsed_date=parsed_date,
            parsed_time=parsed_time or "00:00",
        )
        events.append(event)
    
    logger.info(
        "Parsed %d events from source=%s",
        len(events),
        source_name,
    )
    return events


def normalize_location_name(location: str) -> str:
    """Normalize location name to match database format.
    
    Args:
        location: Raw location name from source
    
    Returns:
        Normalized location name matching database
    """
    if not location:
        return ""
    
    normalized = location.strip().lower()
    
    # Direct mapping lookup
    for key, value in LOCATION_MAPPINGS.items():
        if key in normalized:
            return value
    
    # Return original if no mapping found
    return location.strip()


def fuzzy_title_match(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """Check if two titles are similar enough to be considered the same event.
    
    Args:
        title1: First title
        title2: Second title  
        threshold: Minimum similarity ratio (0-1)
    
    Returns:
        True if titles are similar above threshold
    """
    if not title1 or not title2:
        return False
    
    # Normalize titles (strip punctuation/emoji, normalize whitespace)
    t1 = _normalize_title(title1)
    t2 = _normalize_title(title2)
    if not t1 or not t2:
        return False
    
    # Exact match
    if t1 == t2:
        return True

    tokens1 = _strip_noise_tokens(_title_tokens(t1))
    tokens2 = _strip_noise_tokens(_title_tokens(t2))
    if tokens1 and tokens2:
        if tokens1 == tokens2:
            return True
        if _tokens_subset_match(tokens1, tokens2):
            return True
    
    # Fuzzy match using SequenceMatcher
    ratio = SequenceMatcher(None, t1, t2).ratio()
    if ratio >= threshold:
        return True

    if tokens1 and tokens2:
        core1 = " ".join(tokens1)
        core2 = " ".join(tokens2)
        ratio = SequenceMatcher(None, core1, core2).ratio()
        return ratio >= max(0.7, threshold - 0.1)

    return False


async def find_existing_event(
    db,
    location_name: str,
    event_date: str,
    event_time: str,
    title: str,
) -> tuple[int | None, bool]:
    """Find existing event in database by date, time, location and title.
    
    Uses the same matching logic as upsert_event to avoid false negatives
    that would trigger unnecessary LLM calls.
    
    Args:
        db: Database instance
        location_name: Normalized location name
        event_date: ISO format date (YYYY-MM-DD)
        event_time: Time in HH:MM format
        title: Event title for fuzzy matching
    
    Returns:
        Tuple of (event_id, needs_full_update):
        - event_id: ID of existing event or None if not found
        - needs_full_update: True if event has 00:00 time and should be fully updated
    """
    from models import Event
    from sqlalchemy import select
    from difflib import SequenceMatcher
    
    logger.debug(
        "find_existing_event: searching date=%s time=%s location=%s title=%s",
        event_date, event_time, location_name, title[:50],
    )
    
    async with db.get_session() as session:
        # Match upsert_event: search by date + time first
        stmt = select(Event).where(
            Event.date == event_date,
            Event.time == event_time,
        )
        result = await session.execute(stmt)
        candidates = result.scalars().all()
        
        logger.debug(
            "find_existing_event: found %d candidates for date=%s time=%s",
            len(candidates), event_date, event_time,
        )
        
        # Apply same matching logic as upsert_event (main.py:9823-9912)
        for ev in candidates:
            ev_loc = (ev.location_name or "").strip().lower()
            new_loc = (location_name or "").strip().lower()
            ev_addr = (ev.location_address or "").strip().lower()
            
            # Exact location match
            if ev_loc == new_loc:
                logger.info(
                    "find_existing_event: MATCHED by location event_id=%d title=%s",
                    ev.id, ev.title[:50],
                )
                return ev.id, False
            
            # Fuzzy title match (threshold 0.9 like upsert_event)
            title_ratio = SequenceMatcher(
                None, (ev.title or "").lower(), (title or "").lower()
            ).ratio()
            if title_ratio >= 0.9:
                logger.info(
                    "find_existing_event: MATCHED by title (ratio=%.2f) event_id=%d title=%s",
                    title_ratio, ev.id, ev.title[:50],
                )
                return ev.id, False
            
            # Combined fuzzy match (title >= 0.6 AND location >= 0.6)
            loc_ratio = SequenceMatcher(None, ev_loc, new_loc).ratio()
            if title_ratio >= 0.6 and loc_ratio >= 0.6:
                logger.info(
                    "find_existing_event: MATCHED by combined fuzzy (title=%.2f loc=%.2f) event_id=%d",
                    title_ratio, loc_ratio, ev.id,
                )
                return ev.id, False
        
        # Also check for placeholder events (00:00 time) that need full update
        if event_time != "00:00":
            stmt_placeholder = select(Event).where(
                Event.date == event_date,
                Event.time == "00:00",
            )
            result = await session.execute(stmt_placeholder)
            placeholders = result.scalars().all()
            
            for ev in placeholders:
                ev_loc = (ev.location_name or "").strip().lower()
                new_loc = (location_name or "").strip().lower()
                
                if ev_loc == new_loc:
                    if fuzzy_title_match(title, ev.title):
                        logger.info(
                            "find_existing_event: MATCHED placeholder event_id=%d (needs full update)",
                            ev.id,
                        )
                        return ev.id, True
        
        # Check by location + date (fallback for different time slots)
        stmt_loc = select(Event).where(
            Event.location_name == location_name,
            Event.date == event_date,
        )
        result = await session.execute(stmt_loc)
        loc_candidates = result.scalars().all()
        
        for ev in loc_candidates:
            if fuzzy_title_match(title, ev.title):
                # Different time for same event?
                if ev.time == "00:00" and event_time != "00:00":
                    logger.info(
                        "find_existing_event: MATCHED by loc+title placeholder event_id=%d",
                        ev.id,
                    )
                    return ev.id, True
                # Skip if times differ significantly (different show times)
                db_start = extract_time_start(ev.time)
                new_start = extract_time_start(event_time)
                if db_start and new_start and db_start == new_start:
                    logger.info(
                        "find_existing_event: MATCHED by loc+title+start_time event_id=%d",
                        ev.id,
                    )
                    return ev.id, False
    
    logger.debug(
        "find_existing_event: NO MATCH for title=%s", title[:50],
    )
    return None, False


def should_update_event(existing_ticket_status: str | None, new_ticket_status: str) -> bool:
    """Determine if event ticket status should be updated.
    
    Args:
        existing_ticket_status: Current status in database
        new_ticket_status: New status from source
    
    Returns:
        True if status should be updated
    """
    if existing_ticket_status == new_ticket_status:
        return False
    
    # Always update if we have new information
    if new_ticket_status in ("available", "sold_out"):
        return True
    
    return False


async def find_linked_events(
    db,
    location_name: str,
    title: str,
    exclude_event_id: int | None = None,
) -> list[int]:
    """Find events with same location and title (different dates/times).
    
    Used to link recurring shows together.
    
    Args:
        db: Database instance
        location_name: Normalized location name
        title: Event title
        exclude_event_id: Event ID to exclude from results
    
    Returns:
        List of event IDs that should be linked
    """
    from models import Event
    from sqlalchemy import select
    
    async with db.get_session() as session:
        stmt = select(Event.id).where(
            Event.location_name == location_name,
        )
        result = await session.execute(stmt)
        all_events = result.scalars().all()
        
        linked_ids: list[int] = []
        
        for event_id in all_events:
            if exclude_event_id and event_id == exclude_event_id:
                continue
            
            # Get full event for title comparison
            event = await session.get(Event, event_id)
            if event and fuzzy_title_match(title, event.title, 0.9):
                linked_ids.append(event_id)
        
        return linked_ids


def limit_photos_for_source(photos: list[str], source: str, max_photos: int = 5) -> list[str]:
    """Limit photos based on source rules.
    
    For Музтеатр: take 5 random photos instead of all.
    
    Args:
        photos: List of photo URLs
        source: Source identifier
        max_photos: Maximum number of photos
    
    Returns:
        Filtered list of photos
    """
    if not photos:
        return []
    
    if source.lower() in ("muzteatr", "музыкальный театр"):
        if len(photos) > max_photos:
            return random.sample(photos, max_photos)
    
    return photos
