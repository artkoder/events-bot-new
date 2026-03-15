from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

from sqlalchemy import or_, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import Database
from location_reference import normalise_event_location_from_reference
from models import Event


TARGET_TELEGRAPH_URLS = (
    "https://telegra.ph/Master-klass-Masterstvo-hudozhnika-rospis-po-derevu-03-13",
    "https://telegra.ph/Bilet-na-2-dejstviya-03-13",
)

TARGET_CANONICAL_LOCATIONS = {
    "Клуб Спутник, Карташева 6, Калининград",
    "М/К Сфера, Энгельса 9, Калининград",
}

_CITY_RE = re.compile(r"(?iu)^молод[её]жн")
_TEXT_RE = re.compile(r"(?iu)молод[её]жн")


def _db_path() -> str:
    return (os.getenv("DB_PATH") or "").strip() or "/data/db.sqlite"


def _looks_like_target_event(event: Event) -> bool:
    telegraph_url = str(getattr(event, "telegraph_url", "") or "").strip()
    source_url = str(getattr(event, "source_post_url", "") or "").strip()
    city = str(getattr(event, "city", "") or "").strip()
    location_name = str(getattr(event, "location_name", "") or "").strip()
    location_address = str(getattr(event, "location_address", "") or "").strip()
    if telegraph_url in TARGET_TELEGRAPH_URLS or source_url in TARGET_TELEGRAPH_URLS:
        return True
    if _CITY_RE.search(city):
        return True
    if _TEXT_RE.search(location_name):
        return True
    return "карташева" in location_address.casefold() or "энгельса 9" in location_address.casefold()


async def _rebuild_telegraph_pages(db: Database, event_ids: list[int]) -> None:
    if not event_ids:
        return
    from main import update_telegraph_event_page

    for event_id in event_ids:
        url = await update_telegraph_event_page(int(event_id), db, bot=None)
        print(f"rebuild telegraph: event_id={event_id} url={url or '—'}")


async def run(*, rebuild_telegraph: bool, dry_run: bool) -> int:
    db = Database(_db_path())
    await db.init()
    changed_ids: list[int] = []
    inspected = 0
    changed = 0

    try:
        async with db.get_session() as session:
            stmt = select(Event).where(
                or_(
                    Event.telegraph_url.in_(TARGET_TELEGRAPH_URLS),
                    Event.source_post_url.in_(TARGET_TELEGRAPH_URLS),
                    Event.city.ilike("молод%"),
                    Event.location_name.ilike("%молод%"),
                    Event.location_address.ilike("%карташева%"),
                    Event.location_address.ilike("%энгельса%"),
                )
            ).order_by(Event.date, Event.id)
            result = await session.execute(stmt)
            events = list(result.scalars().all())
            for event in events:
                if not _looks_like_target_event(event):
                    continue
                inspected += 1
                before = {
                    "location_name": event.location_name,
                    "location_address": event.location_address,
                    "city": event.city,
                }
                payload = dict(before)
                matched = normalise_event_location_from_reference(payload)
                if matched is None:
                    continue
                if matched.canonical_line not in TARGET_CANONICAL_LOCATIONS:
                    continue
                if payload == before:
                    continue
                changed += 1
                print(
                    "fix",
                    f"event_id={event.id}",
                    f"title={event.title!r}",
                    f"before={before}",
                    f"after={payload}",
                )
                if dry_run:
                    continue
                event.location_name = payload["location_name"]
                event.location_address = payload.get("location_address")
                event.city = payload.get("city")
                session.add(event)
                changed_ids.append(int(event.id))
            if not dry_run and changed_ids:
                await session.commit()
    finally:
        if not dry_run and rebuild_telegraph and changed_ids:
            await _rebuild_telegraph_pages(db, changed_ids)
        await db.close()

    print(
        f"summary: inspected={inspected} changed={changed} dry_run={int(dry_run)} rebuild_telegraph={int(rebuild_telegraph)}"
    )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix youth-center events that were stored with city='МОЛОДЕЖНЫЙ'."
    )
    parser.add_argument(
        "--rebuild-telegraph",
        action="store_true",
        help="rebuild Telegraph pages for changed events after DB update",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned changes without writing to the database",
    )
    args = parser.parse_args()
    asyncio.run(run(rebuild_telegraph=bool(args.rebuild_telegraph), dry_run=bool(args.dry_run)))


if __name__ == "__main__":
    main()
