
import asyncio
import os
from sqlalchemy import select, func
from db import Database
from models import Event

async def report():
    db_path = os.getenv("EVENTS_DB_PATH", "artifacts/db/events.db")
    db = Database(db_path)
    await db.init()
    async with db.get_session() as session:
        # Check event count for February 2026
        stmt = select(func.count(Event.id)).where(Event.date >= '2026-02-01', Event.date < '2026-03-01')
        count = (await session.execute(stmt)).scalar()
        print(f"Total events in February 2026: {count}")

        # Check how many have preview_3d_url
        stmt_preview = select(Event.id, Event.title, Event.preview_3d_url).where(
            Event.date >= '2026-02-01', 
            Event.date < '2026-03-01',
            Event.preview_3d_url.is_not(None),
            Event.preview_3d_url != ""
        )
        previews = (await session.execute(stmt_preview)).all()
        print(f"Events with 3D preview: {len(previews)}")
        for eid, title, url in previews[:5]:
            print(f"  - [{eid}] {title[:30]}: {url}")

if __name__ == "__main__":
    asyncio.run(report())
