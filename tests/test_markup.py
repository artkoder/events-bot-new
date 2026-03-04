import pytest

from markup import simple_md_to_html, linkify_for_telegraph, expose_links_for_vk, md_to_html
from telegraph.utils import html_to_nodes

def test_bold():
    assert simple_md_to_html('**bold** __bold__') == '<p><b>bold</b> <b>bold</b></p>'

def test_italic():
    assert simple_md_to_html('*it* _it_') == '<p><i>it</i> <i>it</i></p>'

def test_link():
    assert (
        simple_md_to_html('see [site](https://example.com)')
        == '<p>see <a href="https://example.com">site</a></p>'
    )

def test_header_and_newline():
    assert simple_md_to_html('# Title\ntext') == '<h1>Title</h1>\n<p>text</p>'

def test_emoji_preserved():
    assert simple_md_to_html('smile 😀') == '<p>smile 😀</p>'

def test_no_italic_in_urls():
    url = 'https://example.com/image_2024_08.jpg'
    assert simple_md_to_html(url) == f"<p>{url}</p>"

def test_link_with_underscore_url():
    assert (
        simple_md_to_html('[doc](https://site.com/a_b_c)')
        == '<p><a href="https://site.com/a_b_c">doc</a></p>'
    )

def test_plain_underscores_unchanged():
    assert simple_md_to_html('file_name') == '<p>file_name</p>'

def test_md_to_html_balances_inline_tags_for_telegraph():
    # Regression: markdown can produce mis-nested tags like `<b>..<i>..</b>..</i>`.
    html = md_to_html("**bold _italic** text_")
    html_to_nodes(html)  # should not raise


def test_simple_md_to_html_unordered_list_dash():
    assert (
        simple_md_to_html("A\n- one\n- two")
        == "<p>A</p>\n<ul><li>one</li><li>two</li></ul>"
    )


def test_simple_md_to_html_unordered_list_bullet_no_space():
    assert simple_md_to_html("•One\n•Two") == "<ul><li>One</li><li>Two</li></ul>"


def test_simple_md_to_html_inline_bullets_split():
    assert simple_md_to_html("•One •Two •Three") == "<ul><li>One</li><li>Two</li><li>Three</li></ul>"


def test_simple_md_to_html_ordered_list():
    assert simple_md_to_html("1. One\n2) Two") == "<ol><li>One</li><li>Two</li></ol>"

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
    """Phone with +7 country code becomes clickable tel: link."""
    assert (
        linkify_for_telegraph("+7 (495) 123-45-67")
        == '<a href="tel:+74951234567">+7 (495) 123-45-67</a>'
    )


def test_linkify_phone_with_8():
    """Phone with 8 prefix converts to +7 in tel: link."""
    assert (
        linkify_for_telegraph("8-800-555-35-35")
        == '<a href="tel:+78005553535">8-800-555-35-35</a>'
    )


def test_linkify_phone_local():
    """Local phone without country code gets +7 prefix."""
    assert (
        linkify_for_telegraph("(4012) 12-34-56")
        == '<a href="tel:+74012123456">(4012) 12-34-56</a>'
    )


def test_linkify_phone_inside_anchor_untouched():
    """Phone already inside a link is not modified."""
    html = '<a href="tel:+74951234567">+7 (495) 123-45-67</a>'
    assert linkify_for_telegraph(html) == html

def test_linkify_phone_compact():
    """Compact phone +79216118779 becomes clickable."""
    assert (
        linkify_for_telegraph("+79216118779")
        == '<a href="tel:+79216118779">+79216118779</a>'
    )

def test_linkify_phone_in_text_context():
    """Phone inside text with preposition works."""
    assert (
        linkify_for_telegraph("Запись по тел +79216118779")
        == 'Запись по тел <a href="tel:+79216118779">+79216118779</a>'
    )

def test_linkify_phone_no_space_prefix():
    """Phone without separator is not linkified (avoid matching inside words)."""
    assert linkify_for_telegraph("тел+79216118779") == "тел+79216118779"
