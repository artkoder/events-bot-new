import sqlite3
import re
import sys

def check_strict():
    try:
        conn = sqlite3.connect('/data/db.sqlite')
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        event_id = 2031
        # strict regex: Полицеймак but NOT followed by 'о'
        # In python re: Полицеймак(?!о)
        pattern = re.compile(r"Полицеймак(?!о)", re.IGNORECASE)

        print(f"Checking Event {event_id} strictly...")
        cur.execute("SELECT * FROM event WHERE id = ?", (event_id,))
        row = cur.fetchone()
        
        if row:
            for key in row.keys():
                val = row[key]
                if isinstance(val, str) and pattern.search(val):
                    print(f"MATCH in Event Field '{key}':")
                    # print context
                    match = pattern.search(val)
                    start = max(0, match.start() - 20)
                    end = min(len(val), match.end() + 20)
                    print(f"...{val[start:end]}...")
                    print("-" * 10)
        
        # Check Poster
        print("Checking Poster...")
        cur.execute("SELECT * FROM eventposter WHERE event_id = ?", (event_id,))
        rows = cur.fetchall()
        for p_row in rows:
            for key in p_row.keys():
                val = p_row[key]
                if isinstance(val, str) and pattern.search(val):
                    print(f"MATCH in Poster Field '{key}':")
                    match = pattern.search(val)
                    start = max(0, match.start() - 20)
                    end = min(len(val), match.end() + 20)
                    print(f"...{val[start:end]}...")
                    print("-" * 10)

        print("Strict check complete.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_strict()
