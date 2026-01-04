Checked `main.py`; both issues are rooted in the month-nav build and scheduling flow.

**1) Missing Update (new month doesn’t refresh existing months)**
- `schedule_event_update_tasks` only enqueues `month_pages` for the event’s own month and marks that single month dirty (`main.py:10141-10146`).
- `enqueue_job` coalesces `month_pages` by that month only (`main.py:9902-9904`), so no other month page gets rebuilt.
- `build_month_nav_html` builds nav from the *set* of months with events (`main.py:5389-5423`), so when the set changes, all month pages should be regenerated.
- `refresh_month_nav` already exists but is never called (`main.py:5450-5482`).

**Recommended fix**
- Detect when the month set changes (e.g., first event in a new month or last event removed from a month) and then:
  - call `refresh_month_nav`, or
  - enqueue `month_pages` for **all** months in the nav range, or
  - store a `month_nav_hash` (like festival nav) and trigger a “month_nav_update_all” job when the hash changes.

**2) Year Suffix (all 2026 months show “2026”)**
- `build_month_nav_html` uses `month_name_nominative`, which appends the year whenever `year != current year` (`main.py:5415`, `main.py:13903-13912`).
- That makes *every* 2026 month show “2026”, but the desired behavior is “year only on January (or only when the year changes).”

**Recommended fix**
- Make nav labels year-aware inside `build_month_nav_html`: track the previous month’s year and append ` {year}` only when the year changes (or `month == 01`).
- Keep `month_name_nominative` as-is for other contexts (buttons, titles), and add a nav-specific label helper.

If you want, I can implement these changes. Suggested next steps:
1) Add a month-nav change detector + call `refresh_month_nav` (or enqueue all months).
2) Update `build_month_nav_html` to add the year suffix only at year boundaries.