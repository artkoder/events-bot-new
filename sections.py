import re
import hashlib


def normalize_html(s: str) -> str:
    """Normalize HTML by collapsing whitespace."""
    return re.sub(r"\s+", " ", s).strip()


def content_hash(s: str) -> str:
    """Return sha256 hash of normalized HTML."""
    norm = normalize_html(s)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def replace_between_markers(html: str, start: str, end: str, new_block: str) -> str:
    """Replace content between markers, inserting block if markers missing."""
    start_idx = html.find(start)
    end_idx = html.find(end, start_idx + len(start)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        return html[:start_idx] + start + new_block + html[end_idx:]
    # remove stray markers if only one exists
    html = html.replace(start, "").replace(end, "")
    if html and not html.endswith("\n"):
        html += "\n"
    return html + start + new_block + end
