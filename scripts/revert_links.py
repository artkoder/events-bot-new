
import sqlite3
import os

db_path = os.environ.get("DB_PATH", "db_prod_snapshot.sqlite")

def revert():
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Remove link from all posts
    link = "\n\nhttps://домискусств.рф/verified-link"
    
    cur.execute("SELECT id, text FROM vk_inbox WHERE text LIKE ?", (f"%{link}%",))
    rows = cur.fetchall()
    
    for pid, text in rows:
        new_text = text.replace(link, "")
        cur.execute("UPDATE vk_inbox SET text = ? WHERE id = ?", (new_text, pid))
        print(f"Reverted Post {pid}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    revert()
