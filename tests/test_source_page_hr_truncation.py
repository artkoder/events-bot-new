import pytest

import main


@pytest.mark.asyncio
async def test_build_source_page_content_does_not_truncate_on_markdown_hr(monkeypatch):
    async def fake_month_nav_html(_db):
        return "<p>nav</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_month_nav_html)
    dummy_db = object()

    body = "### Заголовок\n\nАбзац 1.\n\n---\n\nАбзац 2, который нельзя терять."
    html, _, _ = await main.build_source_page_content(
        "T",
        body,
        None,
        None,
        None,
        None,
        dummy_db,
    )
    assert "Абзац 1" in html
    assert "Абзац 2, который нельзя терять" in html
    assert "<p>nav</p>" in html


@pytest.mark.asyncio
async def test_build_source_page_content_does_not_truncate_body_after_search_digest(monkeypatch):
    async def fake_month_nav_html(_db):
        return "<p>nav</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_month_nav_html)
    dummy_db = object()

    body = ("Основной текст события. " * 40).strip()
    assert len(body) > 500

    html, _, _ = await main.build_source_page_content(
        "T",
        body,
        None,
        None,
        None,
        None,
        dummy_db,
        search_digest="Короткий дайджест.",
    )
    assert "Короткий дайджест." in html
    assert "Основной текст события" in html
    assert "<p>nav</p>" in html


@pytest.mark.asyncio
async def test_build_source_page_content_renders_search_digest_even_for_short_body(monkeypatch):
    async def fake_month_nav_html(_db):
        return "<p>nav</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_month_nav_html)
    dummy_db = object()

    body = "Короткий основной текст."
    html, _, _ = await main.build_source_page_content(
        "T",
        body,
        None,
        None,
        None,
        None,
        dummy_db,
        search_digest="Дайджест для короткого тела.",
    )
    assert "Дайджест для короткого тела." in html
    assert "Короткий основной текст." in html


@pytest.mark.asyncio
async def test_build_source_page_content_does_not_add_blank_spacers_around_body_divider(monkeypatch):
    async def fake_month_nav_html(_db):
        return "<p>nav</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_month_nav_html)
    dummy_db = object()

    summary = main.SourcePageEventSummary(date="2026-02-21", time="14:00", location_name="Тест")
    body = "Абзац 1.\n\n---\n\nАбзац 2."

    html, _, _ = await main.build_source_page_content(
        "T",
        body,
        None,
        None,
        None,
        None,
        dummy_db,
        event_summary=summary,
        search_digest="Дайджест.",
    )
    # Internal divider exists.
    assert "<hr" in html
    # No ZWSP spacer paragraph directly adjacent to the internal divider marker.
    assert "<p>&#8203;</p><!--BODY_DIVIDER--><hr" not in html
    assert "<!--BODY_DIVIDER--><hr><p>&#8203;</p>" not in html
