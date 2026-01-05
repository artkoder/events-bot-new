"""Regional filtering for Kaliningrad oblast.

Filters out events and venues located outside the Kaliningrad region.
Uses multiple strategies: city name matching, coordinates, and LLM fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Kaliningrad oblast cities and settlements (Russian names)
KALININGRAD_CITIES = {
    # Major cities
    "калининград", "черняховск", "советск", "балтийск", "гусев",
    "светлогорск", "зеленоградск", "пионерский", "неман", "полесск",
    "гвардейск", "мамоново", "багратионовск", "правдинск", "славск",
    "краснознаменск", "ладушкин", "озёрск", "нестеров", "светлый",
    # Resorts and settlements
    "янтарный", "светлогорск", "отрадное", "куршская коса",
    "зеленоградск", "рыбачий", "морское", "лесной",
    # Common variations
    "кёнигсберг", "königsberg", "калининградская область",
}

# Known Kaliningrad venues (case-insensitive partial match)
KALININGRAD_VENUES = {
    "кафедральный собор",
    "драматический театр",
    "музей мирового океана",
    "филармония светланова",
    "историко-художественный музей",
    "замок нессельбек",
    "кинотеатр заря",
    "дом молодёжи",
    "дворец юность",
    "остров канта",
    "рыбная деревня",
    "площадь победы",
    "кинотеатр каро",
}

# Geo bounding box for Kaliningrad oblast
KALININGRAD_BOUNDS = {
    "lat_min": 54.3,
    "lat_max": 55.3,
    "lon_min": 19.6,
    "lon_max": 22.9,
}


@dataclass
class RegionalFilterResult:
    """Result of regional filtering."""
    events_total: int = 0
    events_kept: int = 0
    events_removed: int = 0
    venues_total: int = 0
    venues_kept: int = 0
    removed_events: list[dict] = field(default_factory=list)


def is_in_kaliningrad_region(venue: dict | None, event_venue_text: str | None) -> bool:
    """Check if venue/event is in Kaliningrad oblast.
    
    Uses multiple strategies:
    1. City name in venue['city']
    2. Known venue name match
    3. Geo coordinates check
    4. Address text analysis
    """
    # Strategy 1: Check city field
    if venue:
        city = (venue.get("city") or "").lower().strip()
        if city and any(kc in city for kc in KALININGRAD_CITIES):
            return True
    
    # Strategy 2: Check known venue names
    venue_title = ""
    if venue:
        venue_title = (venue.get("title") or "").lower()
    if event_venue_text:
        venue_title = event_venue_text.lower()
    
    if venue_title:
        for known in KALININGRAD_VENUES:
            if known in venue_title:
                return True
    
    # Strategy 3: Check geo coordinates
    if venue and venue.get("geo"):
        geo = venue["geo"]
        lat = geo.get("lat")
        lon = geo.get("lon")
        if lat is not None and lon is not None:
            if (KALININGRAD_BOUNDS["lat_min"] <= lat <= KALININGRAD_BOUNDS["lat_max"] and
                KALININGRAD_BOUNDS["lon_min"] <= lon <= KALININGRAD_BOUNDS["lon_max"]):
                return True
            else:
                # If geo is outside Kaliningrad, definitely out of region
                logger.debug("Venue %s has geo outside Kaliningrad: %s, %s", 
                            venue.get("title"), lat, lon)
                return False
    
    # Strategy 4: Check address for city mentions
    address = ""
    if venue:
        address = (venue.get("address") or "").lower()
    if event_venue_text:
        address = event_venue_text.lower()
    
    if address:
        # Check for Kaliningrad cities in address
        for city in KALININGRAD_CITIES:
            if city in address:
                return True
        
        # Check for explicitly non-Kaliningrad cities
        non_kaliningrad = ["москва", "санкт-петербург", "петербург", "спб",
                          "минск", "вильнюс", "варшава", "берлин", "гданьск"]
        for foreign in non_kaliningrad:
            if foreign in address:
                logger.info("Excluded venue/event with non-Kaliningrad city: %s", foreign)
                return False
    
    # Default: if city is empty but no negative signals, assume local
    # (Most festival-specific parsers are for Kaliningrad events)
    if not venue or not venue.get("city"):
        logger.debug("No city info, assuming local: %s", event_venue_text)
        return True
    
    # City is specified but not recognized - exclude
    city = (venue.get("city") or "").lower()
    if city and not any(kc in city for kc in KALININGRAD_CITIES):
        logger.info("Excluded venue with unknown city: %s", city)
        return False
    
    return True


def filter_regional(uds_data: dict) -> tuple[dict, RegionalFilterResult]:
    """Filter UDS data to keep only Kaliningrad oblast events and venues.
    
    Args:
        uds_data: Full UDS JSON structure
        
    Returns:
        Tuple of (filtered UDS, filtering stats)
    """
    result = RegionalFilterResult()
    
    # Build venue lookup
    venues = uds_data.get("venues") or []
    venue_map = {v.get("id"): v for v in venues if v.get("id")}
    
    for v in venues:
        venue_map[v.get("title", "").lower()[:30]] = v
    
    # Filter events
    program = uds_data.get("program") or []
    result.events_total = len(program)
    
    filtered_program = []
    for event in program:
        event_venue_text = event.get("venue") or ""
        
        # Try to find venue in our list
        venue = None
        if event.get("venue_ref_id"):
            venue = venue_map.get(event["venue_ref_id"])
        elif event_venue_text:
            # Try partial match
            key = event_venue_text.lower()[:30]
            venue = venue_map.get(key)
        
        if is_in_kaliningrad_region(venue, event_venue_text):
            filtered_program.append(event)
            result.events_kept += 1
        else:
            result.events_removed += 1
            result.removed_events.append({
                "title": event.get("title"),
                "venue": event_venue_text,
                "reason": "outside_kaliningrad_region"
            })
            logger.info("Removed event outside region: %s @ %s", 
                       event.get("title"), event_venue_text)
    
    # Filter venues
    result.venues_total = len(venues)
    filtered_venues = [v for v in venues if is_in_kaliningrad_region(v, None)]
    result.venues_kept = len(filtered_venues)
    
    # Build filtered UDS
    filtered_uds = uds_data.copy()
    filtered_uds["program"] = filtered_program
    filtered_uds["venues"] = filtered_venues
    
    logger.info(
        "Regional filter: kept %d/%d events, %d/%d venues",
        result.events_kept, result.events_total,
        result.venues_kept, result.venues_total
    )
    
    return filtered_uds, result
