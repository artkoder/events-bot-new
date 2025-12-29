"""Source parsing module for theatre events.

Handles parsing events from Kaggle notebook outputs (dramteatr.json, muzteatr.json, sobor.json)
and integrating them into the event management system.

Also handles Pyramida event extraction from VK posts.
"""

from source_parsing.parser import (
    TheatreEvent,
    parse_theatre_json,
    normalize_location_name,
    find_existing_event,
    should_update_event,
)
from source_parsing.handlers import run_source_parsing
from source_parsing.pyramida import (
    extract_pyramida_urls,
    run_pyramida_kaggle_kernel,
    parse_pyramida_output,
    process_pyramida_events,
)

__all__ = [
    "TheatreEvent",
    "parse_theatre_json",
    "normalize_location_name",
    "find_existing_event",
    "should_update_event",
    "run_source_parsing",
    # Pyramida
    "extract_pyramida_urls",
    "run_pyramida_kaggle_kernel",
    "parse_pyramida_output",
    "process_pyramida_events",
]
