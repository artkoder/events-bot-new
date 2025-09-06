from sections import ensure_footer_nav_with_hr
from telegraph.utils import html_to_nodes, nodes_to_html


def test_html_without_hr_adds_nav():
    html = "<p>text</p>"
    nav = "<h4>nav</h4>"
    res = ensure_footer_nav_with_hr(html, nav, month="2025-01", page=1)
    assert res == "<p>text</p><hr>\n<h4>nav</h4>"


def test_html_with_duplicates():
    html = "<p>text</p><hr><h4>old</h4><p>junk</p>"
    nav = "<h4>new</h4>"
    res = ensure_footer_nav_with_hr(html, nav, month="2025-01", page=1)
    assert res == "<p>text</p><hr>\n<h4>new</h4>"


def test_nodes_without_hr():
    nodes = html_to_nodes("<p>text</p>")
    nav_nodes = html_to_nodes("<h4>nav</h4>")
    res = ensure_footer_nav_with_hr(nodes, nav_nodes, month="2025-01", page=1)
    assert nodes_to_html(res) == "<p>text</p><hr><h4>nav</h4>"


def test_nodes_with_duplicates():
    nodes = html_to_nodes("<p>text</p><hr><h4>old</h4><p>junk</p>")
    nav_nodes = html_to_nodes("<h4>nav</h4>")
    res = ensure_footer_nav_with_hr(nodes, nav_nodes, month="2025-01", page=1)
    assert nodes_to_html(res) == "<p>text</p><hr><h4>nav</h4>"


def test_idempotent_html():
    html = "<p>text</p><hr><h4>nav</h4>"
    nav = "<h4>nav</h4>"
    res1 = ensure_footer_nav_with_hr(html, nav, month="2025-01", page=1)
    res2 = ensure_footer_nav_with_hr(res1, nav, month="2025-01", page=1)
    assert res1 == res2

