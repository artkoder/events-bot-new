import pytest

from markup import simple_md_to_html


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
