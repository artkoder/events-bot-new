# 3D Preview Refinements

## Objectives
1. **Fix Old Images in Telegraph Event Pages**: Ensure the event page generator uses `event.preview_3d_url` if available.
2. **Composition for < 3 Images**: Modify Blender script to center the image(s) if fewer than 3.
3. **Transparency Handling**: Since Telegraph doesn't render transparency well, add a background (dark or solid color) during Blender rendering.

## Implementation Details

### 1. Event Page Generation
- **Target File**: `main.py` (approx line 12328, `update_telegraph_event_page`).
- **Logic**: Prepend `event.preview_3d_url` to `catbox_urls` passed to `build_source_page_content`.
  ```python
  photos = list(ev.photo_urls or [])
  if ev.preview_3d_url:
       photos.insert(0, ev.preview_3d_url)
  ```


### 2. Blender Script (limit < 3)
- In `kaggle/Preview3D/preview_3d.ipynb`:
- The script iterates over images.
- If `len(images) < 3`:
  - **Proposed Logic**:
    - If 1 image: Place it at the center (location checks).
    - If 2 images: Place them Side-by-Side or Center one and another.
    - User request: "If < 3, take from center and place simple in center".
    - Means: Just render a single centered plane? Or 2 centered planes?
    - Let's assume: If < 3, treat as "Single Hero Image" mode. Place 1 image in the exact center.
    - If 2 images, maybe pick the first one and center it? Or put 2 side by side? User said "take from center... place simply in center".
    - *Interpretation*: The user likely means "Use the middle image of the set (if 3) or just the image (if 1-2) and center it."
    - Actually, user said: "if illustrations < 3 ... take from center and place simple in center". Since we download up to 3 images (`MAX_IMAGES_PER_EVENT = 3` or similar), if we have 1 or 2, we just use them.
    - Simplest: If `len(images) < 3`, use a different layout logic -> **Single Centered Plane**.
- **Transparency**:
  - Add a background plane or set World Background to a solid color in Blender.
  - User likes "dark modes". Set background to dark gray/black hex `#1F1F1F` or similar.

### 3. Execution
- We need to modify the notebook and the bot code.
- **Priority**: Fix the telegraph page generation first (Python). Then Blender logic (Python script in Notebook).
