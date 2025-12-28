from __future__ import annotations

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "\U00002300-\U000023FF"
    "]+"
)


def _clean_text(text: str | None, *, strip_emojis: bool = True) -> str:
    """Clean text: strip emojis, normalize whitespace. NO truncation."""
    cleaned = str(text or "")
    if strip_emojis:
        cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_about_text(
    text: str | None,
    *,
    ocr_text: str | None = None,  # Kept for API compatibility, UNUSED
    word_limit: int = 12,  # Kept for API compatibility, UNUSED
    char_limit: int = 60,  # Kept for API compatibility, UNUSED
    strip_emojis: bool = True,
) -> str:
    """Clean LLM-provided about text.
    
    ONLY strips emojis and normalizes whitespace.
    NO truncation, NO deduplication - LLM is fully responsible for content.
    """
    return _clean_text(text, strip_emojis=strip_emojis)


def normalize_about_with_fallback(
    primary: str | None,
    *,
    title: str | None = None,  # Kept for API compatibility, UNUSED
    ocr_text: str | None = None,  # Kept for API compatibility, UNUSED
    word_limit: int = 12,  # Kept for API compatibility, UNUSED
    char_limit: int = 60,  # Kept for API compatibility, UNUSED
    strip_emojis: bool = True,
) -> str:
    """Clean LLM-provided about text.
    
    ONLY strips emojis and normalizes whitespace.
    NO truncation, NO deduplication - LLM is fully responsible for content.
    """
    return _clean_text(primary, strip_emojis=strip_emojis)
