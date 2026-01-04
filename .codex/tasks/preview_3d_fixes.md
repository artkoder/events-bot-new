# Bug Report: 3D Preview Issues (v1.6.0)

## 1. Kaggle Dataset Path Mismatch
**Severity**: Critical (Feature Broken)
**Symptoms**: Log shows `ERROR: /kaggle/input/preview3d-dataset/payload.json not found!`. Result is 0 events processed.
**Cause**:
- **Handler**: Creates dataset with slug `preview3d-{session_id}` (e.g., `preview3d-1735755123`).
- **Notebook**: Hardcodes path `Path("/kaggle/input/preview3d-dataset/payload.json")`.
- **Kaggle Behavior**: Mounts datasets at `/kaggle/input/{dataset-slug}`.
**Fix Required**:
- Modify `preview_3d.ipynb` to dynamically find the payload file:
  ```python
  # Search for payload.json in any subdirectory of /kaggle/input
  payloads = list(Path("/kaggle/input").rglob("payload.json"))
  if not payloads: raise FileNotFoundError(...)
  PAYLOAD_FILE = payloads[0]
  ```
- Alternatively, strictly match the slug (but dynamic slug is better for concurrency).

## 2. Silent Failure in Notebook
**Severity**: High
**Symptoms**: Notebook completes with success status even if payload is missing.
**Cause**: `load_payload()` function prints error but returns `[]`. Main loop processes 0 events.
**Fix Required**:
- `load_payload` should `raise Exception` if file is missing.
- Ensure notebook exits with non-zero code or writes "status": "error" to `output.json` if initialization fails.

## 3. Status Updates Not Working
**Severity**: Medium (UX)
**Symptoms**: Telegram message says "Статус: подготовка..." and never updates to "Rendering" or shows progress until the very end.
**Cause**:
- `_poll_kaggle_kernel` in `preview_3d/handlers.py` loops and sleeps but **does not call** `bot.edit_message_text`.
- It only updates the local `_active_sessions` dict.
- Contrast with `video_announce` which uses `wait_for_completion` with a callback or `run_kernel_poller` which actively edits the message.
**Fix Required**:
- Update `_poll_kaggle_kernel` to accept `bot`, `chat_id`, `message_id` and call `edit_message_text` on status change.
- Or implement a separate background poller task like `video_announce`.

## 4. Code Duplication (Optimization)
**Severity**: Low
**Observation**: `preview_3d` reinvents Kaggle polling logic found in `video_announce/poller.py`.
**Recommendation**: Consider extracting a generic `KagglePoller` class or simply ensuring `preview_3d` polling is as robust as `video_announce` (timeouts, logs downloading on error, etc.).

## Action Items for Codex
1. **Fix Notebook**: Update `kaggle/Preview3D/preview_3d.ipynb` to find `payload.json` dynamically and fail fast on errors.
2. **Fix Handler**: Update `preview_3d/handlers.py` to:
   - Provide real-time status updates in the Telegram message during the polling loop.
   - Handle "0 output" cases as errors if payload was non-empty.
3. **Verify**: Ensure the polling loop correctly edits the message (e.g., "Status: Running", "Status: Complete").
