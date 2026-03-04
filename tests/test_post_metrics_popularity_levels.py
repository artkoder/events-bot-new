from __future__ import annotations

import pytest

from source_parsing.post_metrics import PopularityBaseline, popularity_marks


@pytest.mark.parametrize(
    "views,likes,expected",
    [
        # Just above median -> level 1 each
        (101, 11, "⭐👍"),
        # +0.5*median above threshold -> level 2 each
        (151, 16, "⭐⭐👍👍"),
        # Very large -> capped by POST_POPULARITY_MAX_LEVEL
        (5000, 999, "⭐⭐⭐⭐👍👍👍👍"),
    ],
)
def test_popularity_marks_support_levels(monkeypatch, views, likes, expected):
    monkeypatch.setenv("POST_POPULARITY_MIN_SAMPLE", "1")
    monkeypatch.setenv("POST_POPULARITY_VIEWS_MULT", "1.0")
    monkeypatch.setenv("POST_POPULARITY_LIKES_MULT", "1.0")
    monkeypatch.setenv("POST_POPULARITY_MAX_LEVEL", "4")
    monkeypatch.setenv("POST_POPULARITY_VIEWS_STEP", "0.5")
    monkeypatch.setenv("POST_POPULARITY_LIKES_STEP", "0.5")

    baseline = PopularityBaseline(median_views=100, median_likes=10, sample=10)
    marks = popularity_marks(views=int(views), likes=int(likes), baseline=baseline)
    assert marks.text == expected

