Log file doesn’t actually contain the “Начинаю импорт…” / “Импортировано” lines or any `update_id`, so I can’t prove concurrency from `double_import_logs.txt`. What it *does* show is the same `eid=1940` shortlink failure twice and a multi‑instance warning.

**Log Analysis**
- No “Начинаю импорт…” / “Импортировано” lines, no `update_id` fields in `double_import_logs.txt`, so we can’t verify same update or timestamp there.
- `eid=1940` appears twice in the VK shortlink block (duplicate failures + fallback): `double_import_logs.txt:22` and `double_import_logs.txt:67`, which implies the shortlink step ran twice.
- There’s a `TelegramConflictError` about another `getUpdates` request in `double_import_logs.txt:76`, indicating at least two bot instances were running.

**Handler / Logic**
- The “✅ Добавить” button is a callback (`vkrev:accept:*`) built in `main_part2.py:9989`.
- The callback handler is `handle_vk_review_cb` in `main_part2.py:10970`; it sends “Запускаю импорт…” + “⏳ Начинаю импорт события…” and then calls `_vkrev_import_flow`.
- `_vkrev_import_flow` (`main_part2.py:10151`) does **no** pre‑check for “already importing/imported”; it only calls `vk_review.mark_imported` at the end (`vk_review.py:575`).
- The persist step uses `vk_intake.persist_event_and_pages` (`vk_intake.py:1788`), which calls `upsert_event` (`main.py:9756`). If a duplicate import runs, `upsert_event` can return the same event ID, matching your observation.

**Hypothesis**
- Most likely a double callback: user double‑clicked, or the button was tapped twice while the import was running.
- The `TelegramConflictError` suggests multiple bot instances; two processes can read the same update before offsets move, leading to duplicate handling.
- Less likely: a retry due to slow callback ACK; but `callback.answer()` is called immediately before import.

**Recommended Fix**
- Add a DB‑level idempotency check/lock on `vk_inbox` before starting import (e.g., `UPDATE ... SET status='importing' WHERE id=? AND status='locked'` and only proceed if a row was updated). If not, reply “Уже импортируется/импортировано”.
- Disable the inline keyboard on accept (e.g., `edit_reply_markup(None)`) to prevent double‑clicks.
- Optionally route imports through a job queue with a coalesce key (e.g., `vkrev_import:{inbox_id}` or by `source_post_url`) so only one job runs.
- Add logging for `callback.id`, `callback.data`, and `update_id` in `handle_vk_review_cb` to confirm duplicates next time.

If you want, I can implement the DB‑level lock + button disable in:
1) `main_part2.py` (callback handler)  
2) `vk_review.py` (status transition helper)