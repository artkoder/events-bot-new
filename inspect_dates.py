
import asyncio
from sqlalchemy import select
from db import Database
from models import Event

async def report():
    db = Database("events.db")
    await db.init()
    async with db.get_session() as session:
        events = (await session.execute(select(Event).limit(10))).scalars().all()
        print(f"Sample events: {len(events)}")
        for e in events:
            print(f"[{e.id}] {e.date} - {e.title}")

if __name__ == "__main__":
    asyncio.run(report())
