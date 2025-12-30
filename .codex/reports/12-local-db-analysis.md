**DB Findings**
- April 2026 event exists: id `1940`, date `2026-04-18` (`db_prod_snapshot.sqlite` query).
- `MonthPage` exists for `2026-04` with a Telegraph URL (note the `TEST-` prefix) (`db_prod_snapshot.sqlite` query).
- `JobOutbox` for event `1940`: latest `month_pages` and `week_pages` are `done` (2025-12-30); older `expired` errors exist but are superseded (`db_prod_snapshot.sqlite` query).

**Logic Findings**
- There is no `pages.py`; rebuild logic is in `main_part2.py`. `_sync_month_page_inner` creates a `MonthPage` row if missing and proceeds to build content, so it does not only update existing months (`main_part2.py:520` and `main_part2.py:603`). This refutes the hypothesis that new months are skipped.
- `ensure_footer_nav_with_hr` just truncates after the last `<hr>` and appends the provided nav block; it doesn’t decide which months appear (`sections.py:10`).
- Month navigation is built from events in a fixed window: `[start of current month, date(today.year + 1, 4, 1))` (`main.py:5389`). The same cutoff is used for nav refresh (`main.py:5450`). If the rebuild ran on 2025-12-30, an event on 2026-04-18 is outside that window, so April won’t be included in the footer even though the April page exists.

**Hypothesis**
- The missing April link is most likely caused by the nav window cutoff (`end_nav = date(today.year + 1, 4, 1)`), not by deferred rebuilds failing to insert new `MonthPage` rows.

If you want, I can suggest a precise change to widen the nav window and a quick rebuild step to confirm April appears.