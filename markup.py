import html, re
from typing import List

MD_BOLD   = re.compile(r'(?<!\w)(\*\*|__)(.+?)\1(?!\w)', re.S)
MD_ITALIC = re.compile(r'(?<!\w)(\*|_)(.+?)\1(?!\w)', re.S)

MD_LINK   = re.compile(r'\[([^\]]+?)\]\((https?://[^)]+?)\)')
MD_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.M)

_BARE_LINK_RE = re.compile(r'(?<!href=["\'])(https?://[^\s<>)]+)')
_TEXT_LINK_RE = re.compile(r'([^<\[]+?)\s*\((https?://[^)]+)\)')

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

    text = MD_LINK.sub(lambda m: f'<a href="{m[2]}">{m[1]}</a>', text_or_html)
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
# Month navigation markers for month pages
NAV_MONTHS_START: Marker = Marker("<!-- nav-months:start -->")
NAV_MONTHS_END: Marker = Marker("<!-- nav-months:end -->")

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
