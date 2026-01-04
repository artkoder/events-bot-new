import sqlite3
import json

def compare_parsed_vs_db():
    # Load parsed events from muzteatr
    with open('muzteatr_parsed.json', 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    
    print(f"Parsed events from muzteatr.json: {len(parsed)}")
    
    # Connect to live database snapshot
    conn = sqlite3.connect('db_live_verify.sqlite')
    cursor = conn.cursor()
    
    # Get events from Музыкальный театр
    cursor.execute("""
        SELECT id, title, date, time, added_at 
        FROM event 
        WHERE location_name = 'Музыкальный театр' 
        ORDER BY date ASC
    """)
    db_events = cursor.fetchall()
    print(f"DB events for Музтеатр: {len(db_events)}")
    
    # Compare
    matched = 0
    unmatched = []
    
    for p in parsed:
        title = p.get('title', '')
        date_raw = p.get('date_raw', '')
        
        # Simple check - look for title match in DB
        found = False
        for db_ev in db_events:
            db_title = db_ev[1] if db_ev[1] else ''
            if title[:20].lower() in db_title.lower() or db_title.lower() in title.lower():
                found = True
                matched += 1
                break
        
        if not found:
            unmatched.append({'title': title, 'date': date_raw})
    
    print(f"\nMatched: {matched}")
    print(f"Unmatched: {len(unmatched)}")
    
    if unmatched:
        print("\nUnmatched events (first 10):")
        for u in unmatched[:10]:
            print(f"  - {u['title'][:50]} ({u['date']})")
    
    conn.close()

if __name__ == "__main__":
    compare_parsed_vs_db()
