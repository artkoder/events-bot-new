Updated month-page retry logic to preserve `show_images`, made `main_part2.py` safer when imported standalone, and completed the conditional images tests.

- `main.py`: pass `show_images` through the CONTENT_TOO_BIG retry path in `patch_month_page_for_date`.
- `main_part2.py`: add LOCAL_TZ/format helpers fallbacks (plus month naming helpers), import `timedelta`, and only render event images when `photo_urls[0]` starts with `http`.
- `tests/test_conditional_images.py`: supply required `Event` fields, add a small helper, and implement the show_images threshold test with a patched `add_day_sections`.

Tests run:
- `pytest tests/test_conditional_images.py`