import html
import logging
import re
from functools import lru_cache

MD_BOLD   = re.compile(r'(?<!\w)(\*\*|__)(.+?)\1(?!\w)', re.S)
MD_ITALIC = re.compile(r'(?<!\w)(\*|_)(.+?)\1(?!\w)', re.S)

MD_LINK   = re.compile(r'\[([^\]]+?)\]\((https?://(?:\\\)|[^)])+?)\)')
MD_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.M)

_BARE_LINK_RE = re.compile(r'(?<!href=["\'])(https?://[^\s<>)]+)')
_TEXT_LINK_RE = re.compile(r'([^<\[]+?)\s*\((https?://(?:\\\)|[^)])+)\)')
_VK_LINK_RE = re.compile(r'\[([^|\]]+)\|([^\]]+)\]')
_TG_MENTION_RE = re.compile(r'(?<![\w/@])@([a-zA-Z0-9_]{4,32})')

_TG_TAG_RE = re.compile(r"</?tg-(?:emoji|spoiler)[^>]*?>", re.IGNORECASE)
_ESCAPED_TG_TAG_RE = re.compile(r"&lt;/?tg-(?:emoji|spoiler).*?&gt;", re.IGNORECASE)


def _unescape_md_url(url: str) -> str:
    return url.replace("\\)", ")").replace("\\\\", "\\")

def simple_md_to_html(text: str) -> str:
    """Конвертирует небольшой подмножество markdown → HTML для Telegraph."""
    text = html.escape(text)
    text = MD_HEADER.sub(lambda m: f'<h{len(m[1])}>{m[2]}</h{len(m[1])}>', text)
    text = MD_LINK.sub(lambda m: f'<a href="{_unescape_md_url(m[2])}">{m[1]}</a>', text)
    text = MD_BOLD.sub(r'<b>\2</b>', text)
    text = MD_ITALIC.sub(r'<i>\2</i>', text)

    return text.replace('\n', '<br>')



def linkify_for_telegraph(text_or_html: str) -> str:
    """Преобразует голые URL и пары «текст (url)» в кликабельные ссылки."""
    def repl_text(m: re.Match[str]) -> str:
        label, href = m.group(1).strip(), _unescape_md_url(m.group(2))
        return f'<a href="{href}">{label}</a>'

    def repl_bare(m: re.Match[str]) -> str:
        url = m.group(1)
        return f'<a href="{url}">{url}</a>'

    def repl_vk(m: re.Match[str]) -> str:
        target, label = m.group(1), m.group(2)
        href = target if target.startswith(("http://", "https://")) else f"https://vk.com/{target}"
        return f'<a href="{href}">{label}</a>'

    def repl_mention(m: re.Match[str]) -> str:
        username = m.group(1)
        return f'<a href="https://t.me/{username}">@{username}</a>'

    text = _VK_LINK_RE.sub(repl_vk, text_or_html)
    text = MD_LINK.sub(lambda m: f'<a href="{_unescape_md_url(m[2])}">{m[1]}</a>', text)
    text = _TEXT_LINK_RE.sub(repl_text, text)
    if "@" in text:
        parts = re.split(r'(<a\b[^>]*>.*?</a>)', text, flags=re.IGNORECASE | re.DOTALL)
        for idx in range(0, len(parts), 2):
            parts[idx] = _TG_MENTION_RE.sub(repl_mention, parts[idx])
        text = "".join(parts)
    text = _BARE_LINK_RE.sub(repl_bare, text)
    return text


def expose_links_for_vk(text_or_html: str) -> str:
    """Преобразует HTML/markdown ссылки в формат «текст (url)» для VK."""
    def repl_html(m: re.Match[str]) -> str:
        href, label = m.group(1), m.group(2)
        return f"{label} ({href})"

    def repl_md(m: re.Match[str]) -> str:
        label, href = m.group(1), _unescape_md_url(m.group(2))
        return f"{label} ({href})"

    text = re.sub(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        repl_html,
        text_or_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"\[([^\]]+)\]\((https?://(?:\\\)|[^)])+)\)", repl_md, text)
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


def sanitize_telegram_html(html_text: str) -> str:
    """Remove Telegram-specific HTML wrappers while keeping inner text."""

    raw = len(_TG_TAG_RE.findall(html_text))
    escaped = len(_ESCAPED_TG_TAG_RE.findall(html_text))
    if raw or escaped:
        logging.info("telegraph:sanitize tg-tags raw=%d escaped=%d", raw, escaped)
    cleaned = _TG_TAG_RE.sub("", html_text)
    cleaned = _ESCAPED_TG_TAG_RE.sub("", cleaned)
    return cleaned


@lru_cache(maxsize=8)
def md_to_html(text: str) -> str:
    html_text = simple_md_to_html(text)
    html_text = linkify_for_telegraph(html_text)
    html_text = sanitize_telegram_html(html_text)
    if not re.match(r"^<(?:h\d|p|ul|ol|blockquote|pre|table)", html_text):
        html_text = f"<p>{html_text}</p>"
    # Telegraph API does not allow h1/h2 or Telegram-specific tags
    html_text = re.sub(r"<(\/?)h[12]>", r"<\1h3>", html_text)
    html_text = sanitize_telegram_html(html_text)
    return html_text


def extract_link_from_html(html_text: str) -> str | None:
    """Return a registration or ticket link from HTML if present."""

    pattern = re.compile(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(html_text))

    # prefer anchors whose text mentions registration or tickets
    for match in matches:
        href, label = match.group(1), match.group(2)
        text = label.lower()
        if any(word in text for word in ["регистра", "ticket", "билет"]):
            return href

    # otherwise look for anchors located near the word "регистрация"
    lower_html = html_text.lower()
    for match in matches:
        href = match.group(1)
        start, end = match.span()
        context_before = lower_html[max(0, start - 60) : start]
        context_after = lower_html[end : end + 60]
        if "регистра" in context_before or "регистра" in context_after:
            return href

    if matches:
        return matches[0].group(1)
    return None


def extract_links_from_html(html_text: str) -> list[str]:
    """Return all registration or ticket links in order of appearance."""

    pattern = re.compile(
        r"<a[^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(html_text))
    lower_html = html_text.lower()
    skip_phrases = ["полюбить 39"]

    def qualifies(label: str, start: int, end: int) -> bool:
        text = label.lower()
        if any(word in text for word in ["регистра", "ticket", "билет"]):
            return True
        context_before = lower_html[max(0, start - 60) : start]
        context_after = lower_html[end : end + 60]
        return (
            "регистра" in context_before
            or "регистра" in context_after
            or "билет" in context_before
            or "билет" in context_after
        )

    prioritized: list[tuple[int, str]] = []
    others: list[tuple[int, str]] = []
    for match in matches:
        href, label = match.group(1), match.group(2)
        context_before = lower_html[max(0, match.start() - 60) : match.start()]
        if any(phrase in context_before for phrase in skip_phrases):
            continue
        if qualifies(label, *match.span()):
            prioritized.append((match.start(), href))
        else:
            others.append((match.start(), href))

    prioritized.sort(key=lambda item: item[0])
    others.sort(key=lambda item: item[0])
    links = [href for _, href in prioritized]
    links.extend(href for _, href in others)
    return links

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
