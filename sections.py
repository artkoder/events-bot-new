import re
import logging
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, List, Tuple


def ensure_footer_nav_with_hr(
    content: Any, nav_block: Any, *, month: str | None = None, page: int | None = None
) -> Any:
    """Ensure a single navigation block exists after the last ``<hr>``.

    The function accepts either HTML string ``content``/``nav_block`` or Telegraph
    node lists.  Any tail after the last ``<hr>`` is removed before appending the
    provided navigation block.  When no ``<hr>`` is present it is appended to the
    end of the content.  The operation is idempotent.
    """

    hr_found = False
    removed_chars = 0

    if isinstance(content, str):
        pattern = re.compile(r"<hr\s*/?>", flags=re.I)
        matches = list(pattern.finditer(content))
        if matches:
            hr_found = True
            last = matches[-1]
            removed_chars = len(content) - last.end()
            content = content[: last.end()]
        else:
            content = content.rstrip() + "<hr>"
        # nav_block is expected to be HTML string in this branch
        content = content + ("\n" if not content.endswith("\n") else "") + nav_block
    else:
        from telegraph.utils import html_to_nodes, nodes_to_html

        nodes: List[Any] = list(content)
        last_hr_idx = None
        for idx in range(len(nodes) - 1, -1, -1):
            n = nodes[idx]
            if isinstance(n, dict) and n.get("tag") == "hr":
                last_hr_idx = idx
                break
            if isinstance(n, dict) and n.get("tag") in {"p", "figure"}:
                ch = n.get("children", [])
                if (
                    len(ch) == 1
                    and isinstance(ch[0], dict)
                    and ch[0].get("tag") == "hr"
                ):
                    last_hr_idx = idx
                    break
        if last_hr_idx is not None:
            hr_found = True
            removed = nodes[last_hr_idx + 1 :]
            n = nodes[last_hr_idx]
            if isinstance(n, dict) and n.get("tag") in {"p", "figure"}:
                removed = nodes[last_hr_idx:]
                nodes = nodes[:last_hr_idx]
                nodes.append({"tag": "hr"})
                last_hr_idx = len(nodes) - 1
            else:
                nodes = nodes[: last_hr_idx + 1]
            if removed:
                removed_chars = len(nodes_to_html(removed))
        else:
            nodes.append({"tag": "hr"})
            last_hr_idx = len(nodes) - 1
        if isinstance(nav_block, str):
            nav_nodes = html_to_nodes(nav_block)
        else:
            nav_nodes = nav_block
        nodes[last_hr_idx + 1 :] = nav_nodes
        content = nodes

    logging.info(
        "month=%s, page=%s: footer_hr_found=%s, tail_removed_chars=%d, nav_inserted=%s",
        month,
        page,
        hr_found,
        removed_chars,
        True,
    )
    return content


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


MONTHS_RU = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


@dataclass
class DaySection:
    """Descriptor for a day's section on a month page."""

    date: date
    h3_idx: int
    start_idx: int
    end_idx: int


def _nodes_from_html(html_or_nodes: Any) -> List[dict | str]:
    if isinstance(html_or_nodes, str):
        from telegraph.utils import html_to_nodes

        return html_to_nodes(html_or_nodes)
    return html_or_nodes


def _header_text(node: dict) -> List[str]:
    parts: List[str] = []
    for ch in node.get("children", []):
        if isinstance(ch, str):
            parts.append(ch)
        elif isinstance(ch, dict):
            parts.extend(_header_text(ch))
    return parts


MONTH_RE = re.compile(
    r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
)


def parse_month_sections(html_or_nodes: Any) -> Tuple[List[DaySection], bool]:
    """Return sections for each day found by ``h3`` headers.

    Returns a tuple ``(sections, need_rebuild)`` where ``need_rebuild`` is
    ``True`` when no ``h3`` headers were found which signals the caller to
    perform a full rebuild.

    The function does not modify or normalise nodes; zero‑width spaces and
    other whitespace are preserved as is.  The returned ``start_idx`` points to
    the first node following the ``h3`` header.  ``end_idx`` points to the index
    of the next ``h3`` or the end of the node list.
    """

    nodes = _nodes_from_html(html_or_nodes)
    sections: List[DaySection] = []
    h3_positions = [
        i for i, n in enumerate(nodes) if isinstance(n, dict) and n.get("tag") == "h3"
    ]
    if not h3_positions:
        logging.warning("month_rebuild_markers_missing")
        return sections, True
    for idx, pos in enumerate(h3_positions):
        node = nodes[pos]
        text = "".join(_header_text(node))
        # remove emojis and extra spaces
        clean = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        clean = re.sub(r"\s+", " ", clean).strip().lower()
        m = MONTH_RE.match(clean)
        if not m:
            continue
        day = int(m.group(1))
        month_name = m.group(2)
        month = MONTHS_RU.get(month_name)
        if not month:
            continue
        d = date(2000, month, day)
        next_pos = h3_positions[idx + 1] if idx + 1 < len(h3_positions) else len(nodes)
        sections.append(
            DaySection(date=d, h3_idx=pos, start_idx=pos + 1, end_idx=next_pos)
        )
    return sections, False

