# Review Report

## Findings (ordered by severity)
- Medium: `func.lower(Festival.name)` relies on the database's Unicode case-folding. SQLite's built-in `lower()` is ASCII-only, so Cyrillic festival names can still miss and `/parse` updates can silently fail in SQLite-backed runs. Consider a normalized name column or a Unicode-aware collation. (`main_part2.py:2092`)
- Low: Wrapping `Festival.name` in `lower()` prevents use of the `idx_festival_name` index and can cause full scans on large tables. A functional index or a stored normalized field would keep lookups fast. (`main_part2.py:2092`, `models.py:606`)

## Questions / Assumptions
- Are festival names unique case-insensitively? If duplicates differ only by case, `scalar_one_or_none()` will raise and break sync. (`main_part2.py:2094`)

## Test Gaps
- No regression test exercises `sync_festival_page` with a differently cased name from `/parse`. Add a test that asserts the festival row is found and updated. (`main_part2.py:2077`)

## Change Summary
- Festival lookup now lowercases both sides to be case-insensitive.
