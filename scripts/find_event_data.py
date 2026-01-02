import asyncio
import aiosqlite
import os

async def find_event():
    db_path = "db_prod_data.sqlite"
    if not os.path.exists(db_path):
        print(f"Snapshot not found at {db_path}")
        return

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        sql = "SELECT * FROM event WHERE date LIKE '2026-01-03%' AND (title LIKE '%Полицеймак%' OR description LIKE '%Полицеймак%')"
        async with db.execute(sql) as cursor:
            rows = await cursor.fetchall()
            if not rows:
                print("No event found in snapshot.")
            for row in rows:
                print(f"Found event ID: {row['id']}")
                print(f"Title: {row['title']}")
                print(f"Description: {row['description']}")
                print(f"Source Text: {row['source_text']}")
                print(f"Search Digest: {row['search_digest']}")
                print("-" * 20)

if __name__ == "__main__":
    asyncio.run(find_event())
