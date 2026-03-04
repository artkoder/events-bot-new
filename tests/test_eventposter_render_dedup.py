from __future__ import annotations

from datetime import datetime, timezone, timedelta


def test_select_eventposter_render_urls_dedupes_by_ocr_title() -> None:
    import main

    now = datetime.now(timezone.utc)
    vk_url = "https://sun9-88.userapi.com/s/v1/ig2/x.jpg?quality=95"
    catbox_url = "https://files.catbox.moe/x.jpg"

    rows = [
        (vk_url, None, "ОТ АВАНГАРДА ДО СОЦРЕАЛИЗМА", "text A", now - timedelta(days=1), "h1", None),
        (catbox_url, None, "ОТ АВАНГАРДА ДО СОЦРЕАЛИЗМА", "text B", now, "h2", None),
    ]
    urls, excluded = main._select_eventposter_render_urls(rows, prefer_supabase=False)
    assert catbox_url in urls
    assert vk_url not in urls
    assert vk_url in excluded


def test_select_eventposter_render_urls_dedupes_generic_titles_by_ocr_text() -> None:
    import main

    now = datetime.now(timezone.utc)
    vk_url = "https://sun9-53.userapi.com/s/v1/ig2/y.jpg?quality=95"
    catbox_url = "https://files.catbox.moe/y.jpg"
    ocr_text = "Описание: лекция представит ретроспективный обзор творчества..."

    rows = [
        (vk_url, None, "Описание", ocr_text, now - timedelta(hours=2), "h1", None),
        (catbox_url, None, "Описание", ocr_text, now, "h2", None),
    ]
    urls, excluded = main._select_eventposter_render_urls(rows, prefer_supabase=False)
    assert urls == [catbox_url]
    assert vk_url in excluded


def test_select_eventposter_render_urls_prefers_supabase_when_enabled() -> None:
    import main

    now = datetime.now(timezone.utc)
    catbox_url = "https://files.catbox.moe/z.jpg"
    supabase_url = "https://example.supabase.co/storage/v1/object/public/events-media/p/z.webp"

    rows = [
        (catbox_url, supabase_url, "ПРИРОДОВИДЕНИЕ", "text", now, "h1", None),
    ]
    urls, excluded = main._select_eventposter_render_urls(rows, prefer_supabase=True)
    assert urls == [supabase_url]
    assert catbox_url in excluded
