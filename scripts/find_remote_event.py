import sqlite3
import sys

def find_event():
    try:
        conn = sqlite3.connect('/data/db.sqlite')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Search by surname
        print("Searching for 'Полицеймак'...")
        cur.execute("SELECT id, date, title, description, source_text FROM event WHERE title LIKE ? OR description LIKE ?", ('%Полицеймак%', '%Полицеймак%'))
        rows = cur.fetchall()
        
        if not rows:
            print("No event found by surname.")
            # Search by date 2026-01-03
            print("Searching for date 2026-01-03...")
            cur.execute("SELECT id, date, title FROM event WHERE date = ?", ('2026-01-03',))
            rows = cur.fetchall()
            for row in rows:
                print(f"ID: {row['id']}, Date: {row['date']}, Title: {row['title']}")
        else:
            for row in rows:
                print(f"FOUND EVENT: ID={row['id']}")
                print(f"Date: {row['date']}")
                print(f"Title: {row['title']}")
                print(f"Description: {row['description']}")
                print("---")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_event()
