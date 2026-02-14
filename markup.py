import html, re, logging
from typing import List
from functools import lru_cache

MD_BOLD   = re.compile(r'(?<!\w)(\*\*|__)(.+?)\1(?!\w)', re.S)
MD_ITALIC = re.compile(r'(?<!\w)(\*|_)(.+?)\1(?!\w)', re.S)

MD_LINK   = re.compile(r'\[([^\]]+?)\]\((https?://(?:\\\)|[^)])+?)\)')
MD_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.M)

_BARE_LINK_RE = re.compile(r'(?<!href=["\'])(https?://[^\s<>)]+)')
_TEXT_LINK_RE = re.compile(r'([^<\[]+?)\s*\((https?://(?:\\\)|[^)])+)\)')
_VK_LINK_RE = re.compile(r'\[([^|\]]+)\|([^\]]+)\]')
_TG_MENTION_RE = re.compile(r'(?<![\w/@])@([a-zA-Z0-9_]{4,32})')

# Phone number patterns for tel: links (Telegraph).
# We intentionally match only phone-looking sequences (starting with +7 / 8 / "(...)" )
# to avoid false positives like dates ("2026-02-13").
#
# Examples we want to match:
# - +7 (495) 123-45-67
# - 8-800-555-35-35
# - +7 999 123 45 67
# - (4012) 12-34-56
# - 8 401 43 3 19 17
_PHONE_RE = re.compile(
    r"(?<![/\w])"
    r"("
    r"(?:\+7|8)\s*[\s(-]*\d{3,5}[\s)-]*[\d\s-]{5,}\d"
    r"|\(\d{3,5}\)\s*[\d\s-]{5,}\d"
    r")"
    r"(?![/\w])",
    re.IGNORECASE,
)


def _unescape_md_url(url: str) -> str:
    return url.replace("\\)", ")").replace("\\\\", "\\")

def simple_md_to_html(text: str) -> str:
    """Конвертирует небольшой подмножество markdown → HTML для Telegraph."""
    text = html.escape(text)
    text = MD_HEADER.sub(lambda m: f'<h{len(m[1])}>{m[2]}</h{len(m[1])}>', text)
    text = MD_LINK.sub(lambda m: f'<a href="{_unescape_md_url(m[2])}">{m[1]}</a>', text)
    text = MD_BOLD.sub(r'<b>\2</b>', text)
    text = MD_ITALIC.sub(r'<i>\2</i>', text)

    # Blockquotes: lines starting with "> " become <blockquote> blocks.
    # Note: at this point the input is already HTML-escaped, so ">" becomes "&gt;".
    lines = text.split("\n")
    out: list[str] = []
    quote_lines: list[str] = []

    def flush_quote() -> None:
        nonlocal quote_lines
        if not quote_lines:
            return
        out.append("<blockquote>" + "\n".join(quote_lines) + "</blockquote>")
        quote_lines = []

    for raw in lines:
        m = re.match(r"^\s*&gt;\s+(.+)$", raw)
        if m:
            quote_lines.append(m.group(1))
            continue
        flush_quote()
        out.append(raw)
    flush_quote()

    return "\n".join(out).replace("\n", "<br>")



def linkify_for_telegraph(text_or_html: str) -> str:
    """Преобразует голые URL, пары «текст (url)» и телефоны в кликабельные ссылки."""
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

    def repl_phone(m: re.Match[str]) -> str:
        original = m.group(1) or m.group(0)
        digits = re.sub(r"\D", "", original or "")
        if not digits:
            return original
        # Normalize Russian numbers:
        # - 8XXXXXXXXXX -> +7XXXXXXXXXX
        # - XXXXXXXXXX  -> +7XXXXXXXXXX
        if len(digits) == 11 and digits.startswith("8"):
            digits = "7" + digits[1:]
        if len(digits) == 10 and not digits.startswith("7"):
            digits = "7" + digits
        href = f"tel:+{digits}" if len(digits) >= 10 else f"tel:{digits}"
        return f'<a href="{href}">{original}</a>'

    text = _VK_LINK_RE.sub(repl_vk, text_or_html)
    text = MD_LINK.sub(lambda m: f'<a href="{_unescape_md_url(m[2])}">{m[1]}</a>', text)
    text = _TEXT_LINK_RE.sub(repl_text, text)
    if "@" in text:
        parts = re.split(r'(<a\b[^>]*>.*?</a>)', text, flags=re.IGNORECASE | re.DOTALL)
        for idx in range(0, len(parts), 2):
            parts[idx] = _TG_MENTION_RE.sub(repl_mention, parts[idx])
        text = "".join(parts)
    text = _BARE_LINK_RE.sub(repl_bare, text)
    # Convert phone numbers to tel: links (only outside existing links)
    parts = re.split(r'(<a\b[^>]*>.*?</a>)', text, flags=re.IGNORECASE | re.DOTALL)
    for idx in range(0, len(parts), 2):
        parts[idx] = _PHONE_RE.sub(repl_phone, parts[idx])
    text = "".join(parts)
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

_TG_TAG_RE = re.compile(r"</?tg-(?:emoji|spoiler)[^>]*?>", re.IGNORECASE)
_ESCAPED_TG_TAG_RE = re.compile(r"&lt;/?tg-(?:emoji|spoiler).*?&gt;", re.IGNORECASE)
_TG_EMOJI_BLOCK_RE = re.compile(r"<tg-emoji\b[^>]*>.*?</tg-emoji>", re.IGNORECASE | re.DOTALL)
_TG_EMOJI_SELF_RE = re.compile(r"<tg-emoji\b[^>]*/>", re.IGNORECASE)
_ESCAPED_TG_EMOJI_BLOCK_RE = re.compile(
    r"&lt;tg-emoji\b[^&]*&gt;.*?&lt;/tg-emoji&gt;",
    re.IGNORECASE | re.DOTALL,
)
_ESCAPED_TG_EMOJI_SELF_RE = re.compile(r"&lt;tg-emoji\b[^&]*/&gt;", re.IGNORECASE)

def sanitize_telegram_html(html: str) -> str:
    """Remove Telegram-specific HTML wrappers.

    For Telegraph rendering we strip Telegram-only tags:
    - ``tg-spoiler``: unwrap, keep inner text.
    - ``tg-emoji`` (custom emoji): remove entirely because Telegraph can't render it reliably.

    >>> sanitize_telegram_html("<tg-emoji e=1/>")
    ''
    >>> sanitize_telegram_html("<tg-emoji e=1></tg-emoji>")
    ''
    >>> sanitize_telegram_html("<tg-emoji e=1>➡</tg-emoji>")
    ''
    >>> sanitize_telegram_html("&lt;tg-emoji e=1/&gt;")
    ''
    >>> sanitize_telegram_html("&lt;tg-emoji e=1&gt;&lt;/tg-emoji&gt;")
    ''
    >>> sanitize_telegram_html("&lt;tg-emoji e=1&gt;➡&lt;/tg-emoji&gt;")
    ''
    """
    raw = len(_TG_TAG_RE.findall(html))
    escaped = len(_ESCAPED_TG_TAG_RE.findall(html))
    if raw or escaped:
        logging.info("telegraph:sanitize tg-tags raw=%d escaped=%d", raw, escaped)

    # Custom emoji breaks on Telegraph: remove it completely (including the inner placeholder).
    cleaned = _TG_EMOJI_BLOCK_RE.sub("", html)
    cleaned = _TG_EMOJI_SELF_RE.sub("", cleaned)
    cleaned = _ESCAPED_TG_EMOJI_BLOCK_RE.sub("", cleaned)
    cleaned = _ESCAPED_TG_EMOJI_SELF_RE.sub("", cleaned)

    # Unwrap spoiler tags (keep inner text) + remove any leftover tg-* wrappers.
    cleaned = _TG_TAG_RE.sub("", cleaned)
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
