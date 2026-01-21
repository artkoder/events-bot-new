import pytest

from markup import simple_md_to_html, linkify_for_telegraph, expose_links_for_vk

def test_bold():
    assert simple_md_to_html('**bold** __bold__') == '<b>bold</b> <b>bold</b>'

def test_italic():
    assert simple_md_to_html('*it* _it_') == '<i>it</i> <i>it</i>'

def test_link():
    assert simple_md_to_html('see [site](https://example.com)') == 'see <a href="https://example.com">site</a>'

def test_header_and_newline():
    assert simple_md_to_html('# Title\ntext') == '<h1>Title</h1><br>text'

def test_emoji_preserved():
    assert simple_md_to_html('smile 😀') == 'smile 😀'

def test_no_italic_in_urls():
    url = 'https://example.com/image_2024_08.jpg'
    assert simple_md_to_html(url) == url

def test_link_with_underscore_url():
    assert (
        simple_md_to_html('[doc](https://site.com/a_b_c)')
        == '<a href="https://site.com/a_b_c">doc</a>'
    )

def test_plain_underscores_unchanged():
    assert simple_md_to_html('file_name') == 'file_name'

def test_linkify_plain_url():
    assert linkify_for_telegraph("https://example.com") == '<a href="https://example.com">https://example.com</a>'

def test_linkify_text_url():
    assert linkify_for_telegraph("Site (https://example.com)") == '<a href="https://example.com">Site</a>'

def test_linkify_markdown_link():
    assert linkify_for_telegraph("[site](https://example.com)") == '<a href="https://example.com">site</a>'


def test_linkify_vk_internal_link():
    assert (
        linkify_for_telegraph("[club9118984|Калининградском музее]")
        == '<a href="https://vk.com/club9118984">Калининградском музее</a>'
    )


def test_linkify_telegram_mention():
    assert (
        linkify_for_telegraph("@ruin_keepers_admin")
        == '<a href="https://t.me/ruin_keepers_admin">@ruin_keepers_admin</a>'
    )


def test_linkify_telegram_mention_inside_anchor_untouched():
    html = '<a href="https://example.com">@ruin_keepers_admin</a>'
    assert linkify_for_telegraph(html) == html


def test_linkify_telegram_mention_not_email():
    assert linkify_for_telegraph("info@example.com") == "info@example.com"

def test_expose_links_from_html():
    assert expose_links_for_vk('see <a href="https://example.com">site</a>') == 'see site (https://example.com)'

def test_expose_links_from_md():
    assert expose_links_for_vk('see [site](https://example.com)') == 'see site (https://example.com)'


def test_linkify_phone_with_country_code():
    """Phone with +7 country code becomes clickable tg://resolve?phone= link."""
    assert (
        linkify_for_telegraph("+7 (495) 123-45-67")
        == '<a href="tg://resolve?phone=74951234567">+7 (495) 123-45-67</a>'
    )


def test_linkify_phone_with_8():
    """Phone with 8 prefix converts to +7 in tg://resolve?phone= link."""
    assert (
        linkify_for_telegraph("8-800-555-35-35")
        == '<a href="tg://resolve?phone=78005553535">8-800-555-35-35</a>'
    )


def test_linkify_phone_local():
    """Local phone without country code gets +7 prefix."""
    assert (
        linkify_for_telegraph("(4012) 12-34-56")
        == '<a href="tg://resolve?phone=74012123456">(4012) 12-34-56</a>'
    )


def test_linkify_phone_inside_anchor_untouched():
    """Phone already inside a link is not modified."""
    html = '<a href="tg://resolve?phone=+74951234567">+7 (495) 123-45-67</a>'
    assert linkify_for_telegraph(html) == html

def test_linkify_phone_compact():
    """Compact phone +79216118779 becomes clickable."""
    assert (
        linkify_for_telegraph("+79216118779")
        == '<a href="tg://resolve?phone=79216118779">+79216118779</a>'
    )

def test_linkify_phone_in_text_context():
    """Phone inside text with preposition works."""
    assert (
        linkify_for_telegraph("Запись по тел +79216118779")
        == 'Запись по тел <a href="tg://resolve?phone=79216118779">+79216118779</a>'
    )

def test_linkify_phone_no_space_prefix():
    """Phone without space after word works (edge case)."""
    assert (
        linkify_for_telegraph("тел+79216118779")
        == 'тел<a href="tg://resolve?phone=79216118779">+79216118779</a>'
    )
