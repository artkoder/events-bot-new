import sqlite3
import sys

def inspect_event():
    try:
        conn = sqlite3.connect('/data/db.sqlite')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        event_id = 2031
        print(f"Inspecting Event {event_id}...")
        cur.execute("SELECT * FROM event WHERE id = ?", (event_id,))
        row = cur.fetchone()
        
        if row:
            for key in row.keys():
                val = row[key]
                if isinstance(val, str) and "Полицеймак" in val:
                    print(f"Field '{key}':")
                    print(val)
                    print("-" * 10)
        else:
            print("Event not found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_event()
