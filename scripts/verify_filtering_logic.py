
from datetime import datetime, timezone, timedelta

def test_filtering():
    current_date = datetime.now(timezone.utc).date()
    print(f"Current Date: {current_date}")

    test_cases = [
        {"date": current_date - timedelta(days=1), "should_skip": True, "desc": "Yesterday"},
        {"date": current_date, "should_skip": False, "desc": "Today"},
        {"date": current_date + timedelta(days=1), "should_skip": False, "desc": "Tomorrow"},
        {"date": (current_date - timedelta(days=1)).strftime("%Y-%m-%d"), "should_skip": True, "desc": "Yesterday (String)"},
        {"date": (current_date + timedelta(days=1)).strftime("%Y-%m-%d"), "should_skip": False, "desc": "Tomorrow (String)"},
        {"date": None, "should_skip": False, "desc": "None (handled elsewhere)"}, # Logic snippet doesn't handle None, previous block does
    ]

    for case in test_cases:
        event_date_obj = case["date"]
        
        # Logic from handlers.py
        if isinstance(event_date_obj, str):
             try:
                 event_date_obj = datetime.strptime(event_date_obj, "%Y-%m-%d").date()
             except:
                 pass

        skipped = False
        if event_date_obj is not None and isinstance(event_date_obj, type(current_date)) and event_date_obj < current_date:
             skipped = True
        
        result = "PASS" if skipped == case["should_skip"] else "FAIL"
        print(f"Case {case['desc']} ({case['date']}): Expected Skip={case['should_skip']}, Got={skipped} -> {result}")

if __name__ == "__main__":
    test_filtering()
