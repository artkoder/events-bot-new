Adjusted the past-event filter to keep events whose latest known date (start or end) is today or later, so stale end dates don’t drop valid future events. This change is in `special_pages.py:524`, replacing the old conditional with `max(e.date, e.end_date or e.date) >= today.isoformat()`.

Checked `special_pages.py` for other date/end-date filtering logic; there aren’t any other similar filters in this file.

Next steps if you want:
1) Run any relevant tests or a quick manual run that generates a special page to confirm expected results.