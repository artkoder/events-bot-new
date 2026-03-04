import pytest

import main


@pytest.mark.asyncio
async def test_replace_or_drop_broken_images_prefers_fallback_when_available(monkeypatch):
    async def fake_reachable(url):  # noqa: ANN001 - test helper
        return url != "https://files.catbox.moe/broken.jpg"

    monkeypatch.setattr(main, "_image_url_is_reachable", fake_reachable)

    urls = ["https://files.catbox.moe/broken.jpg", "https://files.catbox.moe/ok.jpg"]
    fallback_map = {
        "https://files.catbox.moe/broken.jpg": "https://supabase.example/ok.jpg",
    }

    resolved, dropped = await main._replace_or_drop_broken_images(  # type: ignore[attr-defined]
        urls, fallback_map=fallback_map, label="test"
    )

    assert "https://supabase.example/ok.jpg" in resolved
    assert "https://files.catbox.moe/ok.jpg" in resolved
    assert "https://files.catbox.moe/broken.jpg" in dropped


@pytest.mark.asyncio
async def test_replace_or_drop_broken_images_drops_when_no_fallback(monkeypatch):
    async def fake_reachable(url):  # noqa: ANN001 - test helper
        return False

    monkeypatch.setattr(main, "_image_url_is_reachable", fake_reachable)

    resolved, dropped = await main._replace_or_drop_broken_images(  # type: ignore[attr-defined]
        ["https://files.catbox.moe/broken.jpg"], fallback_map={}, label="test"
    )
    assert resolved == []
    assert dropped == ["https://files.catbox.moe/broken.jpg"]

