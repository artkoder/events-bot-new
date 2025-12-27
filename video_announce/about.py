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


def _prepare_drop_sets(title: str | None, ocr_text: str | None, anchor_limit: int) -> tuple[list[str], set[str]]:
    # NOTE: ocr_text here typically contains just the ocr_title or relevant parts for dedup, passed by caller.
    drop_tokens = {tok.lower() for tok in _tokenize(ocr_text)}
    title_tokens = _tokenize(title)
    # Use list to preserve order from title (important for proper nouns like "ОДИН ДОМА")
    anchors: list[str] = []
    anchors_lower: set[str] = set()
    for tok in title_tokens:
        if len(anchors) >= anchor_limit:
            break
        low = tok.lower()
        if low not in drop_tokens and low not in anchors_lower:
            anchors.append(tok)  # Keep original casing
            anchors_lower.add(low)
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

    # Stop here if input_tokens was empty (or effectively empty)
    # to avoid inventing content via anchors if primary input was missing.
    if not input_tokens:
        return ""

    # NEW CHECK: If everything was filtered out (empty normalized but input wasn't empty),
    # do NOT inject anchors. Return empty string.
    # Requirement: "If after dedup/limits about became empty - do not invent via code, simply save empty."
    if not normalized:
        return ""

    # Second pass: ensure at least some anchor tokens are present if missing
    # We only prepend anchors if NO anchor is present in the normalized list.
    # anchors is now a list with original casing, ordered as in title
    anchors_lower = {a.lower() for a in anchors}
    has_anchor = any(tok.lower() in anchors_lower for tok in normalized)
    if not has_anchor and anchors:
        # Prepend ALL anchors (in order from title) to preserve full proper nouns
        # e.g. "ОДИН ДОМА" should be prepended as both words, not just one
        
        # Rebuild with all anchors at the start
        new_normalized = []
        current_len = 0
        seen: set[str] = set()
        
        # Add all anchors first (in title order)
        for anchor in anchors:
            low = anchor.lower()
            if low in seen:
                continue
            projected = current_len + (1 if new_normalized else 0) + len(anchor)
            if len(new_normalized) >= word_limit or projected > char_limit:
                break
            new_normalized.append(anchor)
            seen.add(low)
            current_len = projected
        
        # Then add remaining tokens from original about
        for token in normalized:
            low = token.lower()
            if low in seen:
                continue
            projected = current_len + 1 + len(token)
            if len(new_normalized) >= word_limit or projected > char_limit:
                break
            new_normalized.append(token)
            seen.add(low)
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
    # 1. Try to normalize the primary (LLM) text.
    # Note: caller should pass ocr_title as ocr_text for dedup logic.
    normalized = normalize_about_text(
        primary,
        title=title,
        ocr_text=ocr_text,
        word_limit=word_limit,
        char_limit=char_limit,
        anchor_limit=anchor_limit,
        strip_emojis=strip_emojis,
    )

    # 2. If primary was provided (even if empty string originally), we respect it.
    # BUT the requirement says: "If after dedup/limits about became empty - do not invent via code, simply save empty."
    # AND "If about was returned by LLM...".
    # So if `primary` is not None, we check `normalized`. If `normalized` is empty, we return empty.

    if primary is not None:
        return normalized

    # 3. Fallback logic only if primary was None (not generated by LLM, e.g. legacy or other flows).
    # Since we updated selection to always parse `about` (which might be None or string),
    # If LLM didn't return `about`, it is None.
    # However, prompt says "about for each selected event".
    # If LLM fails to return it, primary is None.
    # The user says: "if format not met... simply save as comes (or save empty)".
    # This implies we shouldn't fallback to constructing from title if LLM was SUPPOSED to do it.

    # Actually, the user instruction is specific:
    # "If ocr_title empty... rely on title + search_digest... and still no overlap with ocr_title" -> This is for LLM prompt.
    # "Normalization (post-LLM)... If after dedup/limits about became empty - do not invent anew via code, simply save empty."

    # So, if primary is None (LLM didn't send it), we should also probably return empty or handle it gracefully.
    # But existing code has a fallback.
    # Let's remove the fallback for the Selection LLM flow context.
    # But `normalize_about_with_fallback` might be used elsewhere?
    # It is used in `_build_about` in selection.py, and `finalize.py`.
    # In `selection.py`, `primary` comes from `RankedEvent.about`.
    # If LLM didn't return `about` (failed/old prompt), it's None.
    # If we return empty, we show empty about in UI. That's what requested: "LLM didn't offer correct option".

    return normalized
