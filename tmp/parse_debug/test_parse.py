#!/usr/bin/env python3
"""Debug script to test why Tretyakovka events aren't being added to the database."""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, "/workspaces/events-bot-new")

from db import Database
from source_parsing.parser import (
    parse_theatre_json,
    normalize_location_name,
    find_existing_event,
)


async def main():
    # Load JSON
    with open("/workspaces/events-bot-new/tmp/tretyakov.json", "r") as f:
        data = json.load(f)
    
    print(f"=== Tretyakovka JSON Analysis ===")
    print(f"Total events in JSON: {len(data)}")
    print()
    
    # Parse events
    events = parse_theatre_json(data, "tretyakov")
    print(f"Parsed events: {len(events)}")
    print()
    
    # Show all events
    print("=== Events in JSON ===")
    for i, ev in enumerate(events, 1):
        location = normalize_location_name(ev.location)
        print(f"{i}. {ev.title[:60]}")
        print(f"   Date: {ev.parsed_date} Time: {ev.parsed_time}")
        print(f"   Location (raw): {ev.location}")
        print(f"   Location (normalized): {location}")
        print()
    
    # Connect to database and check for duplicates
    db_path = os.getenv("DB_PATH", "artifacts/db/events.db")
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    db = Database(db_path)
    await db.init()
    
    print("=== Checking for existing events in database ===")
    for ev in events:
        location = normalize_location_name(ev.location)
        existing_id, needs_update = await find_existing_event(
            db,
            location,
            ev.parsed_date,
            ev.parsed_time or "00:00",
            ev.title,
        )
        
        status = "EXISTS" if existing_id else "NEW"
        update_note = " (needs update)" if needs_update else ""
        print(f"[{status}] {ev.title[:50]} - {ev.parsed_date} {ev.parsed_time}{update_note}")
        if existing_id:
            print(f"         -> event_id={existing_id}")


if __name__ == "__main__":
    asyncio.run(main())
