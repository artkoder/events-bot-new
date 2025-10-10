import re
import logging
import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, List, Tuple


MONTHS = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]

MONTHS_PREP = [
    "январе",
    "феврале",
    "марте",
    "апреле",
    "мае",
    "июне",
    "июле",
    "августе",
    "сентябре",
    "октябре",
    "ноябре",
    "декабре",
]

MONTHS_NOM = [
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
]

_MONTH_TZ = timezone.utc


def set_month_timezone(tz: timezone) -> None:
    """Configure timezone used for month helper functions."""

    global _MONTH_TZ
    _MONTH_TZ = tz


def month_name(month: str) -> str:
    y, m = month.split("-")
    return f"{MONTHS[int(m) - 1]} {y}"


def month_name_prepositional(month: str) -> str:
    y, m = month.split("-")
    return f"{MONTHS_PREP[int(m) - 1]} {y}"


def month_name_nominative(month: str) -> str:
    """Return month name in nominative case, add year if different from current."""

    y, m = month.split("-")
    name = MONTHS_NOM[int(m) - 1]
    if int(y) != datetime.now(_MONTH_TZ).year:
        return f"{name} {y}"
    return name


def next_month(month: str) -> str:
    d = datetime.fromisoformat(month + "-01")
    n = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return n.strftime("%Y-%m")


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


# Map various Russian month name forms to their numeric value
#
# The dictionary intentionally includes genitive, nominative and common
# abbreviations without a trailing dot.  The dot is stripped before lookup by
# the callers.
MONTHS_RU = {
    "января": 1,
    "январь": 1,
    "янв": 1,
    "февраля": 2,
    "февраль": 2,
    "фев": 2,
    "марта": 3,
    "март": 3,
    "мар": 3,
    "апреля": 4,
    "апрель": 4,
    "апр": 4,
    "мая": 5,
    "май": 5,
    "июня": 6,
    "июнь": 6,
    "июля": 7,
    "июль": 7,
    "августа": 8,
    "август": 8,
    "авг": 8,
    "сентября": 9,
    "сентябрь": 9,
    "сент": 9,
    "октября": 10,
    "октябрь": 10,
    "окт": 10,
    "ноября": 11,
    "ноябрь": 11,
    "нояб": 11,
    "декабря": 12,
    "декабрь": 12,
    "дек": 12,
}


@dataclass
class DaySection:
    """Descriptor for a day's section on a month page.

    ``start_idx`` points to the index of the ``<h3>`` header describing the
    date.  ``end_idx`` points to the index of the next date header or the first
    ``<hr>`` encountered afterwards.  The slice ``nodes[start_idx:end_idx]``
    therefore contains the entire day's section including the header.
    """

    date: date
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


def parse_month_sections(
    html_or_nodes: Any, page: int | None = None
) -> Tuple[List[DaySection], bool]:
    """Return sections for each day found by ``h3`` headers.

    The function walks through the node list, recognising ``h3`` headers that
    contain dates in the form ``DD <month_genitive>`` and ignoring other
    headers such as weekdays.  The returned ``start_idx`` points to the index of
    the date's ``<h3>``; ``end_idx`` points to the index of the next date header
    or the first ``<hr>`` encountered after the section.  When no date headers
    are found the second item of the returned tuple is ``True`` signalling the
    caller to rebuild the entire page.
    """

    nodes = _nodes_from_html(html_or_nodes)
    sections: List[DaySection] = []
    h3_total = sum(1 for n in nodes if isinstance(n, dict) and n.get("tag") == "h3")
    if h3_total == 0:
        logging.warning("month_rebuild_markers_missing")
        logging.info(
            "PARSE-MONTH page=%s h3_total=0 matched=0 dates=[]",
            page if page is not None else "?",
        )
        return sections, True

    date_h3_indices: List[int] = []
    for idx, node in enumerate(nodes):
        if not (isinstance(node, dict) and node.get("tag") == "h3"):
            continue
        text = "".join(_header_text(node))
        text = text.replace("\u00a0", " ").replace("\u200b", " ")
        text = unicodedata.normalize("NFKC", text).lower()
        text = text.replace("🟥", "").strip()
        m = MONTH_RE.fullmatch(text)
        if not m:
            continue
        day = int(m.group(1))
        month = MONTHS_RU[m.group(2)]
        date_h3_indices.append(idx)
        sections.append(
            DaySection(date=date(2000, month, day), start_idx=idx, end_idx=len(nodes))
        )

    logging.info(
        "PARSE-MONTH page=%s h3_total=%d matched=%d dates=%s",
        page if page is not None else "?",
        h3_total,
        len(sections),
        [f"{s.date.day:02d}.{s.date.month:02d}" for s in sections],
    )

    # determine end indices
    if not sections:
        return sections, False
    for i, sec in enumerate(sections):
        # search until next date_h3 or hr
        search_limit = sections[i + 1].start_idx if i + 1 < len(sections) else len(nodes)
        end = search_limit
        for j in range(sec.start_idx + 1, search_limit):
            n = nodes[j]
            if isinstance(n, dict) and n.get("tag") == "hr":
                end = j
                break
        sec.end_idx = end

    return sections, False


def dedup_same_date(nodes: List[Any], target: date) -> Tuple[List[Any], int]:
    """Remove duplicate sections for ``target`` date keeping the first one.

    The ``nodes`` list is modified in place and returned.  The function returns
    the possibly modified node list and the number of removed duplicate
    sections.
    """

    sections, _ = parse_month_sections(nodes)
    dupes = [
        s
        for s in sections
        if s.date.month == target.month and s.date.day == target.day
    ]
    removed = 0
    for sec in reversed(dupes[1:]):
        del nodes[sec.start_idx:sec.end_idx]
        removed += 1
    return nodes, removed

