import html, re
from typing import List

MD_BOLD   = re.compile(r'(?<!\w)(\*\*|__)(.+?)\1(?!\w)', re.S)
MD_ITALIC = re.compile(r'(?<!\w)(\*|_)(.+?)\1(?!\w)', re.S)

MD_LINK   = re.compile(r'\[([^\]]+?)\]\((https?://[^)]+?)\)')
MD_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.M)

_BARE_LINK_RE = re.compile(r'(?<!href=["\'])(https?://[^\s<>)]+)')
_TEXT_LINK_RE = re.compile(r'([^<\[]+?)\s*\((https?://[^)]+)\)')
_VK_LINK_RE = re.compile(r'\[([^|\]]+)\|([^\]]+)\]')

def simple_md_to_html(text: str) -> str:
    """Конвертирует небольшой подмножество markdown → HTML для Telegraph."""
    text = html.escape(text)
    text = MD_HEADER.sub(lambda m: f'<h{len(m[1])}>{m[2]}</h{len(m[1])}>', text)
    text = MD_LINK.sub(lambda m: f'<a href="{m[2]}">{m[1]}</a>', text)
    text = MD_BOLD.sub(r'<b>\2</b>', text)
    text = MD_ITALIC.sub(r'<i>\2</i>', text)

    return text.replace('\n', '<br>')



def linkify_for_telegraph(text_or_html: str) -> str:
    """Преобразует голые URL и пары «текст (url)» в кликабельные ссылки."""
    def repl_text(m: re.Match[str]) -> str:
        label, href = m.group(1).strip(), m.group(2)
        return f'<a href="{href}">{label}</a>'

    def repl_bare(m: re.Match[str]) -> str:
        url = m.group(1)
        return f'<a href="{url}">{url}</a>'

    def repl_vk(m: re.Match[str]) -> str:
        target, label = m.group(1), m.group(2)
        href = target if target.startswith(("http://", "https://")) else f"https://vk.com/{target}"
        return f'<a href="{href}">{label}</a>'

    text = _VK_LINK_RE.sub(repl_vk, text_or_html)
    text = MD_LINK.sub(lambda m: f'<a href="{m[2]}">{m[1]}</a>', text)
    text = _TEXT_LINK_RE.sub(repl_text, text)
    text = _BARE_LINK_RE.sub(repl_bare, text)
    return text


def expose_links_for_vk(text_or_html: str) -> str:
    """Преобразует HTML/markdown ссылки в формат «текст (url)» для VK."""
    def repl_html(m: re.Match[str]) -> str:
        href, label = m.group(1), m.group(2)
        return f"{label} ({href})"

    def repl_md(m: re.Match[str]) -> str:
        label, href = m.group(1), m.group(2)
        return f"{label} ({href})"

    text = re.sub(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        repl_html,
        text_or_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", repl_md, text)
    return text


def sanitize_for_vk(text_or_html: str) -> str:
    """Expose links and strip unsupported HTML for VK posts."""
    s = expose_links_for_vk(text_or_html)
    s = html.unescape(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"</?tg-(?:emoji|spoiler)[^>]*>", "", s, flags=re.I)
    s = re.sub(r"&lt;/?tg-(?:emoji|spoiler).*?&gt;", "", s, flags=re.I)
    s = re.sub(r"<\s*(?:b|strong)\s*>(.*?)<\s*/\s*(?:b|strong)\s*>", r"*\1*", s, flags=re.I | re.S)
    s = re.sub(r"<\s*(?:i|em)\s*>(.*?)<\s*/\s*(?:i|em)\s*>", r"_\1_", s, flags=re.I | re.S)
    s = re.sub(r"<\s*(?:s|del)\s*>(.*?)<\s*/\s*(?:s|del)\s*>", r"~\1~", s, flags=re.I | re.S)
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.I)
    s = re.sub(r"</\s*p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"</\s*li\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<\s*li\s*>", "• ", s, flags=re.I)
    s = re.sub(r"<\s*/?(?:p|ul|ol)\s*>", "", s, flags=re.I)
    s = re.sub(r"</?[^>]+>", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    lines = s.splitlines()
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        norm = stripped.casefold()
        has_folder_icon = "📂" in stripped
        has_addlist = "addlist" in norm
        if "полюбить 39" in norm:
            if not has_addlist:
                j = i + 1
                while j < len(lines):
                    next_stripped = lines[j].strip()
                    if not next_stripped:
                        j += 1
                        continue
                    if "t.me/addlist" in next_stripped.casefold():
                        has_addlist = True
                    break
            if has_folder_icon or has_addlist:
                i += 1
                while i < len(lines):
                    next_stripped = lines[i].strip()
                    if not next_stripped:
                        i += 1
                        continue
                    if "t.me/addlist" in next_stripped.casefold():
                        i += 1
                    break
                continue
        cleaned.append(line)
        i += 1

    s = "\n".join(cleaned)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def telegraph_br() -> list[dict]:
    """Return a safe blank line for Telegraph rendering."""
    # U+200B survives Telegraph HTML import, unlike &nbsp; in <p>
    return [{"tag": "p", "children": ["\u200B"]}]


class Marker(str):
    """Str subclass that mimics dict.get for test compatibility."""

    def get(self, _key=None, default=None):  # type: ignore[override]
        return default


DAY_START = lambda d: Marker(f"<!--DAY:{d} START-->")
DAY_END = lambda d: Marker(f"<!--DAY:{d} END-->")
WEND_START = lambda key: Marker(f"<!--WEEKEND:{key} START-->")
WEND_END = lambda key: Marker(f"<!--WEEKEND:{key} END-->")
PERM_START: Marker = Marker("<!--PERMANENT_EXHIBITIONS START-->")
PERM_END: Marker = Marker("<!--PERMANENT_EXHIBITIONS END-->")
# Canonical festival navigation markers used across the project
# ``FEST_NAV_*`` names are kept for backwards compatibility but map to the
# new ``near-festivals`` markers required by the idempotent block logic.
NEAR_FESTIVALS_START: Marker = Marker("<!-- near-festivals:start -->")
NEAR_FESTIVALS_END: Marker = Marker("<!-- near-festivals:end -->")
FEST_NAV_START: Marker = NEAR_FESTIVALS_START
FEST_NAV_END: Marker = NEAR_FESTIVALS_END

# Festivals index intro markers
FEST_INDEX_INTRO_START: Marker = Marker("<!-- festivals-index:intro:start -->")
FEST_INDEX_INTRO_END: Marker = Marker("<!-- festivals-index:intro:end -->")
