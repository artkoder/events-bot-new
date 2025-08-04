import html, re
from typing import List

MD_BOLD   = re.compile(r'(\*\*|__)(.+?)\1', re.S)
MD_ITALIC = re.compile(r'(\*|_)([^*_].*?)\1', re.S)
MD_LINK   = re.compile(r'\[([^\]]+?)\]\((https?://[^)]+?)\)')
MD_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.M)

def simple_md_to_html(text: str) -> str:
    """Конвертирует небольшой подмножество markdown → HTML для Telegraph."""
    text = html.escape(text)
    text = MD_HEADER.sub(lambda m: f'<h{len(m[1])}>{m[2]}</h{len(m[1])}>', text)
    text = MD_BOLD.sub(r'<b>\2</b>', text)
    text = MD_ITALIC.sub(r'<i>\2</i>', text)
    text = MD_LINK.sub(r'<a href="\2">\1</a>', text)
    return text.replace('\n', '<br>')
