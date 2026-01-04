
import asyncio
import time
import os
os.environ["DB_PATH"] = "db_prod_snapshot.sqlite"
from db import Database

async def seed():
    try:
        db = Database(os.environ["DB_PATH"])
        async with db.raw_conn() as conn:
            ts = int(time.time())
            # Hint: Now + 2h + 300s (Safe margin > reject_cutoff)
            hint = ts + 2 * 3600 + 300

            # Remove existing test item if exists
            await conn.execute("DELETE FROM vk_inbox WHERE group_id=999999")

            # Push back existing items to prioritize ours (Shifting hint into future)
            await conn.execute("UPDATE vk_inbox SET event_ts_hint = event_ts_hint + 200000 WHERE group_id != 999999 AND event_ts_hint IS NOT NULL")
            
            await conn.execute(
                "INSERT INTO vk_inbox(group_id, post_id, date, text, matched_kw, status, event_ts_hint, has_date) VALUES(?,?,?,?,?,?,?,?)",
                (999999, ts, ts, "Спектакль https://домискусств.рф/skazka", "dom_iskusstv", "pending", hint, 1)
            )
            await conn.commit()
            print("Seeded.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    finally:
        if db._conn:
            await db._conn.close()

if __name__ == "__main__":
    asyncio.run(seed())
