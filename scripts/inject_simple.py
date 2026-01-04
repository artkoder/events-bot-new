
import sqlite3
import os

db_path = os.environ.get("DB_PATH", "db_prod_snapshot.sqlite")

def modify():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Simple pick next pending
    cur.execute("SELECT id, text FROM vk_inbox WHERE status='pending' ORDER BY event_ts_hint ASC LIMIT 1")
    row = cur.fetchone()
    
    if not row:
        print("No pending post found.")
        return
        
    pid, text = row
    print(f"Modifying Post {pid}")
    
    new_text = (text or "") + "\n\nhttps://домискусств.рф/verified-link"
    cur.execute("UPDATE vk_inbox SET text = ? WHERE id = ?", (new_text, pid))
    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    modify()
