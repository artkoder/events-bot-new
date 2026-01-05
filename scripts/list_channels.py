import asyncio
from db import Database
from models import Channel
from sqlalchemy import select
import os

async def main():
    db = Database("db_prod_snapshot.sqlite")
    await db.init()
    async with db.get_session() as session:
        result = await session.execute(select(Channel))
        channels = result.scalars().all()
        print(f"{'ID':<5} {'Title':<30} {'Channel ID':<20} {'Daily Time':<10} {'Is Admin':<10}")
        print("-" * 80)
        for ch in channels:
            print(f"{ch.channel_id:<20} {str(ch.title)[:28]:<30} {str(ch.daily_time):<10} {ch.is_admin}")

if __name__ == "__main__":
    asyncio.run(main())
