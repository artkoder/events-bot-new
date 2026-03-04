import pytest

import main
from telegraph.utils import html_to_nodes


@pytest.mark.asyncio
async def test_build_source_page_content_balances_markdown_inline_tags():
    # Regression: our lightweight markdown renderer may generate mis-nested tags:
    # `**bold _italic** text_` -> `<b>..<i>..</b>..</i>`, which breaks html_to_nodes.
    md = "**bold _italic** text_"
    html, _, _ = await main.build_source_page_content(
        "T",
        md,
        None,
        html_text=None,
        media=None,
        ics_url=None,
        db=None,
        catbox_urls=[],
    )
    # Should not raise.
    html_to_nodes(html)


@pytest.mark.asyncio
async def test_build_source_page_content_does_not_treat_br_as_b_tag():
    # Regression: our tag balancer used a prefix regex for tag names and could
    # misparse `<br/>` as an opening `<b...>` tag, leaving stray `</b>` closers.
    md = "Важные примечания:\nучастие бесплатное;\nместа ограничены;\nсообщите заранее."
    html, _, _ = await main.build_source_page_content(
        "T",
        md,
        None,
        html_text=None,
        media=None,
        ics_url=None,
        db=None,
        catbox_urls=[],
    )
    html_to_nodes(html)


@pytest.mark.asyncio
async def test_build_source_page_content_preserves_markdown_lists():
    # Regression: tag balancer must not break `<ul><li>...</li></ul>` into
    # `<ul></ul><li>...` which later becomes paragraph chunks on Telegraph pages.
    md = "### Темы\nВопросы:\n- Пункт 1\n- Пункт 2"
    html, _, _ = await main.build_source_page_content(
        "T",
        md,
        None,
        html_text=None,
        media=None,
        ics_url=None,
        db=None,
        catbox_urls=[],
    )
    assert "<ul>" in html and "<li>Пункт 1</li>" in html
    assert "<ul></ul>" not in html
    assert "<p><li>" not in html
    html_to_nodes(html)
