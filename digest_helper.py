import re

def clean_search_digest(digest: str | None) -> str | None:
    """Clean and normalize search_digest string.

    Removes HH:MM time patterns and normalizes whitespace.
    Returns None if the resulting string is empty.
    """
    if not digest:
        return None

    # Reject "truncated preview" style digests.
    # These are usually cut mid-sentence (Telegram preview, etc.) and degrade UX on Telegraph pages.
    if "…" in digest or "..." in digest:
        return None

    # Remove HH:MM pattern (time)
    cleaned = re.sub(r'\b\d{1,2}:\d{2}\b', '', digest)

    # Normalize spaces (collapse multiple spaces, strip ends)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Remove schedule-like heading fragments (low-signal on a single event).
    cleaned = re.sub(r'^\s*\d{1,2}\.\d{1,2}\s*\|\s*', '', cleaned).strip()

    if not cleaned:
        return None

    return cleaned
