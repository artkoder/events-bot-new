import re


_WS_RE = re.compile(r"\s+")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_LEADING_SCHEDULE_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\s*\|\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"'»”)\]]*)$")


def _collapse_ws(value: str) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _first_sentence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    parts = _SENTENCE_SPLIT_RE.split(raw, maxsplit=1)
    return (parts[0] if parts else raw).strip()

def fallback_one_sentence(text: str | None, *, max_words: int | None = None) -> str | None:
    """Best-effort one-sentence fallback from a long description/body."""
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return None
    # Drop legacy "(подробнее ...)" suffixes if they were appended into the body.
    raw = re.sub(r",?\s*подробнее\s*\([^\n]*\)\s*$", "", raw, flags=re.I).strip()
    # Skip Markdown headings and empty lines; keep the first meaningful paragraph.
    lines = [ln.strip() for ln in raw.split("\n")]
    picked: list[str] = []
    started = False
    for ln in lines:
        if not ln:
            if started:
                break
            continue
        if ln.startswith("#"):
            continue
        ln = re.sub(r"^[•*\-]\s+", "", ln).strip()
        if not ln:
            continue
        picked.append(ln)
        started = True
    collapsed = _collapse_ws(" ".join(picked) if picked else raw)
    sentence = _first_sentence(collapsed)
    sentence = _collapse_ws(sentence)
    if not sentence:
        return None
    if max_words is not None:
        # Legacy option: deterministic word cut (used only in fallbacks/UI).
        words = [w for w in sentence.split() if w.strip()]
        if max_words > 0 and len(words) > max_words:
            cut = " ".join(words[: max(0, int(max_words))]).rstrip(" ,;:-—")
            if cut and cut[-1].isalnum():
                cut += "."
            return cut
    return sentence


def clean_search_digest(digest: str | None) -> str | None:
    """Normalize `search_digest` to a single short sentence.

    Policy:
    - reject obviously truncated previews (`...` / `…`);
    - remove time tokens (`HH:MM`) and schedule-like prefixes;
    - collapse whitespace (NO word truncation).
    """
    if not digest:
        return None

    raw = str(digest or "").strip()
    if not raw:
        return None

    # Reject "truncated preview" style digests. These are usually cut mid-sentence and degrade UX.
    if "…" in raw or "..." in raw:
        return None

    cleaned = _TIME_RE.sub("", raw)
    cleaned = _LEADING_SCHEDULE_RE.sub("", cleaned)
    cleaned = _collapse_ws(cleaned)
    if not cleaned:
        return None

    return cleaned or None


def clean_short_description(text: str | None) -> str | None:
    """Normalize `short_description` (list snippet) to a single line.

    The meaning and length should come from LLM; this function does only
    non-semantic whitespace cleanup and rejects obvious truncated previews.
    """
    if not text:
        return None
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return None
    if "…" in raw or "..." in raw:
        return None
    cleaned = _collapse_ws(raw.replace("\n", " "))
    return cleaned or None


def short_description_word_count(text: str | None) -> int:
    cleaned = clean_short_description(text)
    if not cleaned:
        return 0
    return len([w for w in _WORD_RE.findall(cleaned) if w.strip()])


def is_short_description_acceptable(
    text: str | None,
    *,
    min_words: int = 12,
    max_words: int = 16,
) -> bool:
    cleaned = clean_short_description(text)
    if not cleaned:
        return False
    if not _SENTENCE_END_RE.search(cleaned):
        return False
    words_count = short_description_word_count(cleaned)
    if words_count < int(min_words) or words_count > int(max_words):
        return False
    # Exactly one sentence for festival/daily list snippets.
    sentence_marks = re.findall(r"[.!?](?:[\"'»”)\]]*)(?:\s+|$)", cleaned)
    return len(sentence_marks) == 1


def enforce_digest_word_limit(text: str | None, *, max_words: int = 16) -> str | None:
    """Trim digest-like text to at most `max_words` words for list UIs."""
    if not text:
        return None
    cleaned = _collapse_ws(text)
    if not cleaned:
        return None
    if max_words <= 0:
        return cleaned
    matches = list(_WORD_RE.finditer(cleaned))
    if len(matches) <= max_words:
        return cleaned
    cut_pos = matches[max_words - 1].end()
    clipped = cleaned[:cut_pos].rstrip(" ,;:-—")
    if clipped and clipped[-1].isalnum():
        clipped += "…"
    return clipped or None
