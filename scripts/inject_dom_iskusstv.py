
import asyncio
import os
import sys

# Ensure main is importable
sys.path.append(os.getcwd())

from db import Database
from vk_review import pick_next
from db import Database
from vk_review import pick_next
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo # Fallback
import runtime
from types import SimpleNamespace

# Mock main module attributes required by vk_intake
runtime.main = SimpleNamespace(LOCAL_TZ=ZoneInfo("Europe/Kaliningrad"))

async def modify_next():
    db = Database(os.environ["DB_PATH"])
    # Mimic operator_id (any admin)
    operator_id = 12345 
    batch_id = "test_batch"
    
    post = await pick_next(db, operator_id, batch_id)
    if not post:
        print("No post found in queue!")
        return
        
    print(f"Found Post ID: {post.id}, Group: {post.group_id}")
    print(f"Current Text: {post.text[:50]}...")
    
    new_text = (post.text or "") + "\n\nhttps://домискусств.рф/test-link"
    
    async with db.raw_conn() as conn:
        await conn.execute("UPDATE vk_inbox SET text = ? WHERE id = ?", (new_text, post.id))
        await conn.commit()
        
    print(f"Updated Post {post.id} with Dom Iskusstv link.")

if __name__ == "__main__":
    # Ensure DB_PATH is set
    if "DB_PATH" not in os.environ:
        os.environ["DB_PATH"] = "db_prod_snapshot.sqlite"
    asyncio.run(modify_next())
