
import asyncio
import os
from sqlalchemy import select
from db import Database
from models import Event

async def report():
    db_path = os.getenv("EVENTS_DB_PATH", "artifacts/db/events.db")
    db = Database(db_path)
    await db.init()
    async with db.get_session() as session:
        events = (await session.execute(select(Event).limit(10))).scalars().all()
        print(f"Sample events: {len(events)}")
        for e in events:
            print(f"[{e.id}] {e.date} - {e.title}")

if __name__ == "__main__":
    asyncio.run(report())
