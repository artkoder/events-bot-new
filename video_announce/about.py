from __future__ import annotations

import re
from typing import Iterable

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

_PUNCT_STRIP_RE = re.compile(r"[«»\"' <>.,!?:;()\[\]{}]")


def _clean_primary_about(text: str | None, *, strip_emojis: bool) -> str:
    cleaned = str(text or "")
    if strip_emojis:
        cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str | None) -> list[str]:
    cleaned = _EMOJI_RE.sub("", str(text or "")).replace("\n", " ")
    tokens: list[str] = []
    for raw in cleaned.split():
        token = _PUNCT_STRIP_RE.sub("", raw)
        if token:
            tokens.append(token)
    return tokens


def _prepare_drop_sets(title: str | None, ocr_text: str | None, anchor_limit: int) -> tuple[set[str], set[str]]:
    drop_tokens = {tok.lower() for tok in _tokenize(ocr_text)}
    title_tokens = _tokenize(title)
    anchors = set()
    for tok in title_tokens:
        if len(anchors) >= anchor_limit:
            break
        low = tok.lower()
        if low not in drop_tokens:
            anchors.add(low)
    return anchors, drop_tokens


def _shorten_about_text(
    text: str | None,
    *,
    title: str | None = None,
    ocr_text: str | None = None,
    word_limit: int = 12,
    char_limit: int = 60,
    anchor_limit: int = 2,
) -> str:
    anchors, drop_tokens = _prepare_drop_sets(title, ocr_text, anchor_limit)
    normalized: list[str] = []
    seen: set[str] = set()
    total_len = 0
    input_tokens = _tokenize(text)

    # First pass: collect valid tokens from input text
    for token in input_tokens:
        low = token.lower()
        if low in seen:
            continue
        if low in drop_tokens:
            continue
        projected = total_len + (1 if normalized else 0) + len(token)
        if len(normalized) >= word_limit or projected > char_limit:
            break
        normalized.append(token)
        seen.add(low)
        total_len = projected

    # Second pass: ensure at least some anchor tokens are present if missing
    # We only prepend anchors if NO anchor is present in the normalized list.
    has_anchor = any(tok.lower() in anchors for tok in normalized)
    if not has_anchor and anchors:
        # Try to prepend 1-2 anchors if they fit (replacing if needed or just adding)
        # Simplified strategy: take best anchor, see if it fits by removing from end
        best_anchor = next(iter(anchors)) # anchors is set but iteration order roughly insertion in python 3.7+
        # But we need original casing for best_anchor?
        # Actually _tokenize returns original casing, but anchors set is lower.
        # Let's find original casing from title if possible
        best_anchor_orig = best_anchor.upper() # Fallback
        title_tokens = _tokenize(title)
        for t in title_tokens:
            if t.lower() == best_anchor:
                best_anchor_orig = t
                break

        # If adding best_anchor exceeds limits, we might need to truncate `normalized`
        # But for simplicity, let's just prepend if it fits after clearing enough space
        # Or just Prepend and Slice.

        # New approach: rebuild
        new_normalized = [best_anchor_orig]
        current_len = len(best_anchor_orig)
        seen = {best_anchor}

        for token in normalized:
            low = token.lower()
            if low in seen: continue
            projected = current_len + 1 + len(token)
            if len(new_normalized) >= word_limit or projected > char_limit:
                break
            new_normalized.append(token)
            current_len = projected
        normalized = new_normalized

    return " ".join(normalized)


def normalize_about_text(
    text: str | None,
    *,
    title: str | None = None,
    ocr_text: str | None = None,
    word_limit: int = 12,
    char_limit: int = 60,
    anchor_limit: int = 2,
    strip_emojis: bool = True,
) -> str:
    """Clean and enforce limits on LLM-provided about text."""
    cleaned = _clean_primary_about(text, strip_emojis=strip_emojis)
    return _shorten_about_text(
        cleaned,
        title=title,
        ocr_text=ocr_text,
        word_limit=word_limit,
        char_limit=char_limit,
        anchor_limit=anchor_limit,
    )


def normalize_about_with_fallback(
    primary: str | None,
    *,
    title: str | None,
    ocr_text: str | None = None,
    fallback_parts: Iterable[str | None] = (),
    word_limit: int = 12,
    char_limit: int = 60,
    anchor_limit: int = 2,
    strip_emojis: bool = True,
) -> str:
    normalized = normalize_about_text(
        primary,
        title=title,
        ocr_text=ocr_text,
        word_limit=word_limit,
        char_limit=char_limit,
        anchor_limit=anchor_limit,
        strip_emojis=strip_emojis,
    )
    if normalized:
        return normalized

    fallback = " ".join(part for part in (title, *fallback_parts) if part)
    return _shorten_about_text(
        fallback,
        title=title,
        ocr_text=ocr_text,
        word_limit=word_limit,
        char_limit=char_limit,
        anchor_limit=anchor_limit,
        
    )
