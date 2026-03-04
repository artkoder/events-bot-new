from __future__ import annotations

from io import BytesIO

import pytest

from media_dedup import (
    build_supabase_poster_object_path,
    prepare_image_for_supabase,
)


def _img_bytes(size: tuple[int, int]) -> bytes:
    from PIL import Image  # type: ignore

    im = Image.new("RGB", size, (255, 255, 255))
    # Deterministic pattern: black square in the center.
    for x in range(size[0] // 4, (size[0] * 3) // 4):
        for y in range(size[1] // 4, (size[1] * 3) // 4):
            im.putpixel((x, y), (0, 0, 0))
    out = BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def test_prepare_image_for_supabase_produces_webp_and_stable_dhash_across_sizes():
    raw1 = _img_bytes((64, 64))
    raw2 = _img_bytes((256, 256))

    p1 = prepare_image_for_supabase(raw1, dhash_size=16, webp_quality=82)
    p2 = prepare_image_for_supabase(raw2, dhash_size=16, webp_quality=82)

    assert p1 is not None
    assert p2 is not None
    assert p1.dhash_hex == p2.dhash_hex
    assert p1.webp_bytes.startswith(b"RIFF")
    assert b"WEBP" in p1.webp_bytes[:32]


def test_build_supabase_poster_object_path_formats_prefix_and_algo():
    dh = "0" * 64
    path = build_supabase_poster_object_path(dh, prefix="p", dhash_size=16)
    assert path == "p/dh16/00/" + dh + ".webp"


def test_prepare_image_for_supabase_returns_none_on_empty():
    assert prepare_image_for_supabase(b"") is None

