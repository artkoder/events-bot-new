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
    title_tokens = _tokenize(title)
    anchors = {tok.lower() for tok in title_tokens[:anchor_limit] if tok}
    drop_tokens = {tok.lower() for tok in title_tokens if tok}
    drop_tokens |= {tok.lower() for tok in _tokenize(ocr_text)}
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
    for token in _tokenize(text):
        low = token.lower()
        if low in seen:
            continue
        if low in drop_tokens and low not in anchors:
            continue
        projected = total_len + (1 if normalized else 0) + len(token)
        if len(normalized) >= word_limit or projected > char_limit:
            break
        normalized.append(token)
        seen.add(low)
        total_len = projected
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
    """Lightly clean LLM-provided about text without dropping tokens."""

    return _clean_primary_about(text, strip_emojis=strip_emojis)


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
