from __future__ import annotations


def test_drop_tiny_illustrations_when_large_present_filters_avatar_like_images() -> None:
    import main

    small = "https://example.com/small.jpg"
    large = "https://example.com/large.jpg"

    # Simulate probe metadata collected by `_image_url_is_reachable` (Range GET + Content-Range parsing).
    main.image_url_size_cache[small] = 4_000
    main.image_url_size_cache[large] = 60_000

    kept, dropped = main._drop_tiny_illustrations_when_large_present([small, large], label="test")

    assert kept == [large]
    assert dropped == [small]


def test_drop_tiny_illustrations_when_large_present_no_large_no_filter() -> None:
    import main

    only = "https://example.com/only.jpg"
    main.image_url_size_cache[only] = 8_000

    kept, dropped = main._drop_tiny_illustrations_when_large_present([only], label="test")

    assert kept == [only]
    assert dropped == []

