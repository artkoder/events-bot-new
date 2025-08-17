import pytest

import main
from markup import FESTNAV_START, FESTNAV_END
from main import FOOTER_LINK_HTML

NAV_HTML = '<p>nav</p>'


def test_apply_festival_nav_insert_when_missing():
    html = '<p>start</p>'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert updated.startswith('<p>start</p>')
    assert updated.count(FESTNAV_START) == 1
    assert '<!--NAV_HASH:' in updated
    assert updated.endswith(FOOTER_LINK_HTML)


def test_apply_festival_nav_replace_existing():
    html = f'<p>start</p>{FESTNAV_START}<p>old</p>{FESTNAV_END}'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert NAV_HTML in updated
    assert '<p>old</p>' not in updated


def test_apply_festival_nav_idempotent():
    html = '<p>start</p>'
    first, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    second, changed2 = main.apply_festival_nav(first, NAV_HTML)
    assert changed2 is False
    assert first == second


def test_apply_festival_nav_removes_legacy_heading():
    html = '<p>start</p><h3>Ближайшие фестивали</h3><p>old</p>'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert '<h3>Ближайшие фестивали</h3>' not in updated
    assert updated.count(FESTNAV_START) == 1

