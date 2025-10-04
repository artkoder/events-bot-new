import pytest

import main
from telegraph.utils import InvalidHTML


@pytest.mark.asyncio
async def test_create_source_page_editor_html_fallback(monkeypatch):
    provided_catbox = ["https://cat.box/image.jpg"]
    build_calls = []
    catbox_args = []
    image_modes = []

    async def fake_build_source_page_content(
        title,
        text,
        source_url,
        html_text,
        media=None,
        ics_url=None,
        db=None,
        *,
        event_summary=None,
        display_link=True,
        catbox_urls=None,
        image_mode="tail",
        page_mode="default",
    ):
        build_calls.append(html_text)
        catbox_args.append(catbox_urls)
        image_modes.append(image_mode)
        if html_text is not None:
            return "<p>broken", "html-msg", 0
        return "<p>fixed</p>", "fallback-msg", 0

    monkeypatch.setattr(main, "build_source_page_content", fake_build_source_page_content)

    html_calls: list[str] = []

    def fake_html_to_nodes(html: str):
        html_calls.append(html)
        if "</p>" not in html:
            raise InvalidHTML("Unclosed <p>")
        return ["ok"]

    monkeypatch.setattr("telegraph.utils.html_to_nodes", fake_html_to_nodes)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    async def fake_telegraph_create_page(*args, **kwargs):
        return {"url": "https://telegra.ph/fallback", "path": "fallback"}

    monkeypatch.setattr(main, "telegraph_create_page", fake_telegraph_create_page)

    result = await main.create_source_page(
        "Title",
        "Body text",
        "https://example.com",
        html_text="<p>broken",
        catbox_urls=provided_catbox,
        image_mode="inline",
    )

    assert result == ("https://telegra.ph/fallback", "fallback", "fallback-msg", 0)
    assert build_calls == ["<p>broken", None]
    assert catbox_args == [provided_catbox, provided_catbox]
    assert catbox_args[0] is provided_catbox
    assert catbox_args[1] is provided_catbox
    assert image_modes == ["inline", "inline"]
    assert len(html_calls) == 2
    assert "</p>" in html_calls[1]


@pytest.mark.asyncio
async def test_create_source_page_editor_html_autoclose(monkeypatch):
    original_build = main.build_source_page_content
    build_calls: list = []
    html_outputs: list = []

    async def tracking_build(*args, **kwargs):
        if "html_text" in kwargs:
            html_value = kwargs["html_text"]
        elif len(args) > 3:
            html_value = args[3]
        else:
            html_value = None
        build_calls.append(html_value)
        result = await original_build(*args, **kwargs)
        html_outputs.append(result[0])
        return result

    monkeypatch.setattr(main, "build_source_page_content", tracking_build)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "token")

    async def fake_telegraph_create_page(*args, **kwargs):
        return {"url": "https://telegra.ph/autoclose", "path": "autoclose"}

    monkeypatch.setattr(main, "telegraph_create_page", fake_telegraph_create_page)

    result = await main.create_source_page(
        "Title",
        "Body text",
        "https://example.com",
        html_text="<h3>Heading</p><p>Body</p>",
        catbox_urls=["https://cat.box/image.jpg"],
    )

    assert len(build_calls) == 1
    assert build_calls[0] == "<h3>Heading</p><p>Body</p>"
    assert "<h3>Heading</h3>" in html_outputs[0]
    assert result == (
        "https://telegra.ph/autoclose",
        "autoclose",
        "",
        1,
    )
