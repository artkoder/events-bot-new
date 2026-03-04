import pytest

import main


@pytest.mark.asyncio
async def test_build_source_page_content_demotes_overlong_markdown_headings() -> None:
    md = "### " + ("Это очень длинный заголовок " * 10).strip()
    html, _, _ = await main.build_source_page_content(
        "T",
        md,
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[],
    )
    assert "<h3" not in html.lower()
    assert "Это очень длинный заголовок" in html


@pytest.mark.asyncio
async def test_build_source_page_content_demotes_overlong_html_headings() -> None:
    html_text = "<h3>" + ("Это очень длинный заголовок " * 10).strip() + "</h3>"
    html, _, _ = await main.build_source_page_content(
        "T",
        "ignored",
        None,
        html_text=html_text,
        catbox_urls=[],
    )
    assert "<h3" not in html.lower()
    assert "Это очень длинный заголовок" in html


@pytest.mark.asyncio
async def test_build_source_page_content_renders_video_links() -> None:
    html, _, _ = await main.build_source_page_content(
        "T",
        "body",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[],
        video_urls=["https://example.com/video.mp4"],
    )
    assert '<video src="https://example.com/video.mp4"' in html
