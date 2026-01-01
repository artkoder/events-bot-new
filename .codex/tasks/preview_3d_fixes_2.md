# 3D Preview Fixes (Report & Rebuild)

## Issues
1. **Output Download Failure**: `output.json` not found even after 3 attempts. Kaggle API latency is higher.
2. **Missing Page Rebuilds**: After updating `Event.preview_3d_url`, the month pages are not regenerated, so users don't see the new images.
3. **Missing Final Report**: User wants a summary message with links to the Month Page and the top 5 updated events.

## Requirements

### 1. Robust Download (preview_3d/handlers.py)
- Modify `_download_kaggle_results` to try **10 times** with **5 second** intervals (total 50s).
- Keep the warnings log.

### 2. Trigger Page Rebuilds (preview_3d/handlers.py)
- In `_run_kaggle_render`, after `update_previews_from_results` returns:
  - Identify which months need rebuilding (based on updated events).
  - Call `schedule_event_update_tasks(db, [event_ids])` OR manually mark pages dirty and trigger rebuild.
  - Ideally, use `main_part2.py`'s existing logic if accessible, or import `schedule_event_update_tasks` from `main`.
  - **Wait** for the rebuild to complete (if possible) or just trigger it. Since user wants a link to the *updated* page, we should probably trigger it and assume it will be ready shortly, or just link to it.

### 3. Detailed Final Report (preview_3d/handlers.py)
- Modify the final success message in `_run_kaggle_render`.
- Use the `month` string (e.g., "2026-03") to generate the Month Page Telegraph URL.
  - You might need to query `MonthPage` model to get the URL.
- List up to 5 events that were successfully updated (`status="ok"`):
  - Format: `<a href="EVENT_TELEGRAPH_URL">TITLE</a>`
  - If `event.telegraph_url` is missing, just show title.
- Final format example:
  ```html
  🎨 <b>3D-превью: Март 2026</b>
  
  ✅ Успешно: 5
  ⚠️ Ошибок: 0
  
  🔗 <a href="MONTH_URL">Страница месяца</a>
  
  <b>Обновленные события:</b>
  1. <a href="...">Event 1</a>
  2. <a href="...">Event 2</a>
  ...
  ```

## Implementation Details
- Import `MonthPage` from `models`.
- Import `schedule_event_update_tasks` from `main` (if possible, watch out for circular imports). If circular import is an issue, move `schedule_event_update_tasks` to a common module or implement a simplified dirty-marker here.
  - *Better approach*: Use `mark_pages_dirty` and `enqueue_job` directly if `handlers.py` allows.
  - Actually, `preview_3d` is imported by `main`. `main` imports `handlers`. `handlers` cannot import `main`.
  - **Solution**: Pass a callback `on_complete` or similar to `_run_kaggle_render`? Or just replicate the "mark dirty + enqueue" logic.
  - Easier: `preview_3d/handlers.py` triggers a new job type or uses `session` to update `MonthPage` dirty flags directly.
  - `models.py` has `MonthPage`.
  - We need to trigger the build.
  - Let's check where `mark_pages_dirty` lives. It's likely in `main`.
  - **Refactor**: Move `mark_pages_dirty` and `schedule...` to a shared helper if needed, or just implementation the DB update (dirty=True) manually in `handlers.py` and let the background worker pick it up (if there is one).
  - *Wait*, `schedule_event_update_tasks` is what `main` uses.
  - If we can't import it, we can just update `MonthPage.dirty = True` and `MonthPage.next_run_at = now`. But we need a runner.
  - **Simplest**: Make `_run_kaggle_render` return the list of updated event_ids to the caller. But the caller `handle_3di_callback` is also in `handlers.py`...
  - **Strategy**: Creating a new task in `task_queue`?
  - Let's look at `main.py` imports later. For now, assume we can import `from main import schedule_event_update_tasks` inside the function (local import) to avoid cycle, OR user `worker.py` mechanism.

## Plan
1. Fix download loop.
2. Trigger rebuild (mark dirty + whatever mechanism).
3. Build report with links.
