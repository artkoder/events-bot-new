import pytest

import main
from markup import FEST_NAV_START, FEST_NAV_END
from main import FOOTER_LINK_HTML

NAV_HTML = '<p>nav</p>'


def test_apply_festival_nav_insert_when_missing():
    html = '<p>start</p>'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert updated.startswith('<p>start</p>')
    assert updated.count(FEST_NAV_START) == 1
    assert '<!-- FEST_NAV_START -->' not in updated
    assert '<!-- FEST_NAV_END -->' not in updated
    assert '<!--FEST_NAV_START-->' not in updated
    assert '<!--FEST_NAV_END-->' not in updated
    assert '<!--NAV_HASH:' in updated
    assert updated.endswith(FOOTER_LINK_HTML)


def test_apply_festival_nav_replace_existing():
    html = f'<p>start</p>{FEST_NAV_START}<p>old</p>{FEST_NAV_END}'
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
    assert updated.count(FEST_NAV_START) == 1


def test_apply_festival_nav_rewrites_spaced_markers():
    html = '<p>start</p><!-- FEST_NAV_START --><p>old</p><!-- FEST_NAV_END -->'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert updated.count(FEST_NAV_START) == 1
    assert '<!-- FEST_NAV_START -->' not in updated
    assert '<!-- FEST_NAV_END -->' not in updated


def test_apply_festival_nav_rewrites_uppercase_markers():
    html = '<p>start</p><!--FEST_NAV_START--><p>old</p><!--FEST_NAV_END-->'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert updated.count(FEST_NAV_START) == 1
    assert '<!--FEST_NAV_START-->' not in updated
    assert '<!--FEST_NAV_END-->' not in updated


def test_apply_festival_nav_deduplicates_multiple_blocks():
    html = (
        f"<p>start</p>{FEST_NAV_START}<p>old</p>{FEST_NAV_END}"
        f"<p>mid</p>{FEST_NAV_START}<p>old2</p>{FEST_NAV_END}"
    )
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert updated.count(FEST_NAV_START) == 1
    assert '<p>old</p>' not in updated
    assert '<p>old2</p>' not in updated


def test_apply_festival_nav_removes_heading_with_subheading():
    html = '<p>start</p><h3>Ближайшие фестивали</h3><h4>old</h4><p>end</p>'
    updated, changed = main.apply_festival_nav(html, NAV_HTML)
    assert changed is True
    assert '<h3>Ближайшие фестивали</h3>' not in updated
    assert '<h4>old</h4>' not in updated
    assert updated.count(FEST_NAV_START) == 1


def test_apply_footer_link_idempotent():
    html = '<p>start</p>'
    first = main.apply_footer_link(html)
    second = main.apply_footer_link(first)
    assert first == second
    assert second.count('https://t.me/kenigevents') == 1

