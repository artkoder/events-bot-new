# Telegraph Phone Link Fix Walkthrough

This document summarizes the debugging and resolution of the issue where phone numbers in Telegraph pages were not clickable.

## The Issue

Users reported that phone numbers (e.g., `+7921...`) in the event description were not being converted to clickable links in the generated Telegraph pages, despite local tests suggesting the regex logic was correct.

## Root Cause Analysis

1.  **Missing Function Call**: Initially, the `linkify_for_telegraph` function was not being called for plain text content in `main_part2.py`. This was fixed by ensuring the call happens after markdown-to-html conversion.
2.  **Telegraph API Limitation**: After fixing the function call, logs showed that `tel:` links were being generated (`<a href="tel:+7...">`) but were stripped by the Telegraph API upon page creation. Telegraph does not support the `tel:` URI scheme.

## The Solution

We modified `markup.py` to use the `tg://resolve?phone=...` URI scheme instead of `tel:`. This scheme is native to Telegram and is allowed by the Telegraph API.

### Changes

-   **`markup.py`**: Updated `repl_phone` to generate `tg://resolve?phone={number}` links. The phone number is stripped of the `+` prefix for this scheme.
-   **`main_part2.py`**: Ensured `linkify_for_telegraph` is imported and called for plain text content.
-   **`tests/test_markup.py`**: Updated test cases to verify the new `tg://` link format.

## Verification

-   **Local Tests**: `pytest tests/test_markup.py` passes, confirming correct link generation.
-   **Production**: User confirmed that phone numbers are now clickable links on the generated Telegraph pages.

## Example

**Input**: `Запись по тел +79216118779`
**Output**: `Запись по тел <a href="tg://resolve?phone=79216118779">+79216118779</a>`
