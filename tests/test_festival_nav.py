import pytest

import main
from main import FEST_NAV_START, FEST_NAV_END, FOOTER_LINK_HTML

NAV_HTML = '<p>nav</p>'


def test_apply_festival_nav_replace_markers():
    html = f'<p>start</p>{FEST_NAV_START}<p>old</p>{FEST_NAV_END}'
    updated, strategy = main.apply_festival_nav(html, NAV_HTML)
    assert strategy == 'markers'
    assert updated.count(FEST_NAV_START) == 1
    assert NAV_HTML in updated
    assert updated.endswith(FOOTER_LINK_HTML)


def test_apply_festival_nav_fallback_heading():
    html = '<p>start</p><h3>Ближайшие фестивали</h3><p>old</p>'
    updated, strategy = main.apply_festival_nav(html, NAV_HTML)
    assert strategy == 'fallback_h3'
    assert updated.count(FEST_NAV_START) == 1
    assert '<h3>Ближайшие фестивали</h3>' not in updated
    assert updated.endswith(FOOTER_LINK_HTML)


def test_apply_festival_nav_append_when_no_markers():
    html = '<p>start</p>'
    updated, strategy = main.apply_festival_nav(html, NAV_HTML)
    assert strategy == 'markers'
    assert updated.startswith('<p>start</p>')
    assert updated.count(FEST_NAV_START) == 1
    assert updated.endswith(FOOTER_LINK_HTML)
