"""Source parsing module for theatre events.

Handles parsing events from Kaggle notebook outputs (dramteatr.json, muzteatr.json, sobor.json)
and integrating them into the event management system.
"""

from source_parsing.parser import (
    TheatreEvent,
    parse_theatre_json,
    normalize_location_name,
    find_existing_event,
    should_update_event,
)
from source_parsing.handlers import run_source_parsing

__all__ = [
    "TheatreEvent",
    "parse_theatre_json",
    "normalize_location_name",
    "find_existing_event",
    "should_update_event",
    "run_source_parsing",
]
