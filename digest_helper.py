import re

def clean_search_digest(digest: str | None) -> str | None:
    """Clean and normalize search_digest string.

    Removes HH:MM time patterns and normalizes whitespace.
    Returns None if the resulting string is empty.
    """
    if not digest:
        return None

    # Remove HH:MM pattern (time)
    cleaned = re.sub(r'\b\d{1,2}:\d{2}\b', '', digest)

    # Normalize spaces (collapse multiple spaces, strip ends)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if not cleaned:
        return None

    return cleaned
