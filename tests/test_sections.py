import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from sections import normalize_html, content_hash, replace_between_markers


def test_normalize_html():
    assert normalize_html(" <p> hi\n world </p> ") == "<p> hi world </p>"


def test_content_hash_changes_with_content():
    s1 = "<div>hi</div>"
    s2 = "<div>bye</div>"
    assert content_hash(s1) != content_hash(s2)


def test_replace_between_markers_existing():
    html = "before <!-- A -->old<!-- B --> after"
    res = replace_between_markers(html, "<!-- A -->", "<!-- B -->", "new")
    assert res == "before <!-- A -->new<!-- B --> after"


def test_replace_between_markers_insert():
    html = "start end"
    res = replace_between_markers(html, "<!-- A -->", "<!-- B -->", "x")
    assert res.endswith("<!-- A -->x<!-- B -->")
    assert res.startswith("start end")
