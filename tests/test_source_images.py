import pytest
import main


@pytest.mark.asyncio
async def test_build_source_page_content_multi(monkeypatch):
    async def fake_upload(images):
        return [f"http://cat/{i}.jpg" for i in range(len(images))], "ok"

    async def fake_nav(db):
        return "<p>NAV</p>"

    monkeypatch.setattr(main, "upload_images", fake_upload)
    monkeypatch.setattr(main, "build_month_nav_html", fake_nav)

    media = [(b"a", "a.jpg"), (b"b", "b.jpg"), (b"c", "c.jpg"), (b"d", "d.jpg")]
    html, _, uploaded = await main.build_source_page_content(
        "T", "text", None, None, media, None, object()
    )
    assert uploaded == 4
    assert html.count('<figure><img src="http://cat/0.jpg"/></figure>') == 1
    assert html.count('<img src="http://cat/') == 4
    assert '<p><strong>T</strong></p>' not in html
    assert 'Добавить в календарь' not in html
    # first nav before tail, second after tail
    assert html.count("<p>NAV</p>") == 2
    first_nav = html.index("<p>NAV</p>")
    tail_start = html.index('<img src="http://cat/1.jpg"/>')
    last_nav = html.rfind("<p>NAV</p>")
    assert first_nav < tail_start < last_nav


@pytest.mark.asyncio
async def test_update_source_page_cover_tail_nav(monkeypatch):
    class DummyTG:
        def __init__(self):
            self.html = ""

        def get_page(self, path, return_html=True):
            return {"content": "<p><strong>T</strong></p><p>old</p>"}

        def edit_page(self, path, title, html_content=None, **kwargs):
            self.html = html_content

    dummy = DummyTG()
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None, domain=None: dummy)
    async def fake_nav(db):
        return "<p>NAV</p>"

    monkeypatch.setattr(main, "build_month_nav_html", fake_nav)

    urls = ["http://cat/1.jpg", "http://cat/2.jpg"]
    msg, uploaded = await main.update_source_page(
        "path", "T", "new", db=object(), catbox_urls=urls
    )
    html = dummy.html
    assert uploaded == 2
    assert html.startswith('<figure><img src="http://cat/1.jpg"/></figure>')
    assert html.count('<img src="http://cat/') == 2
    assert html.count("<p>NAV</p>") == 2
    first_nav = html.index("<p>NAV</p>")
    tail_pos = html.index('<img src="http://cat/2.jpg"/>')
    last_nav = html.rfind("<p>NAV</p>")
    assert first_nav < tail_pos < last_nav
    assert main.CONTENT_SEPARATOR in html


def test_apply_ics_link_after_figure():
    html = "<p><strong>T</strong></p><figure><img src='x'/></figure><p>body</p>"
    res = main.apply_ics_link(html, "http://i")
    assert main.ICS_LABEL in res
    assert res.index(main.ICS_LABEL) > res.index("</figure>")


@pytest.mark.asyncio
async def test_build_source_page_content_ics_with_cover():
    html, _, _ = await main.build_source_page_content(
        "T",
        "text",
        None,
        None,
        None,
        "http://ics",
        None,
        catbox_urls=["http://cat/1.jpg"],
    )
    assert html.startswith('<figure><img src="http://cat/1.jpg"/></figure>')
    assert '<p>📅 <a href="http://ics">Добавить в календарь</a></p>' in html
    assert html.index('http://ics') > html.index('</figure>')
    assert html.index('http://ics') < html.index('<p>text</p>')


@pytest.mark.asyncio
async def test_build_source_page_content_ics_no_cover():
    html, _, _ = await main.build_source_page_content(
        "T", "text", None, None, None, "http://ics", None
    )
    assert html.startswith('<p>📅 <a href="http://ics">Добавить в календарь</a></p>')
    assert html.index('http://ics') < html.index('<p>text</p>')


@pytest.mark.asyncio
async def test_build_source_page_content_limit_and_no_video():
    urls = [f"http://cat/{i}.jpg" for i in range(15)] + ["http://cat/vid.mp4"]
    html, _, uploaded = await main.build_source_page_content(
        "T", "text", None, None, None, None, None, catbox_urls=urls
    )
    assert uploaded == 12
    assert html.count('<img src="http://cat/') == 12
    assert 'vid.mp4' not in html


@pytest.mark.asyncio
async def test_build_source_page_content_cleans_tg_tags():
    html, _, _ = await main.build_source_page_content(
        "T",
        "",
        None,
        "<tg-emoji emoji-id='1'>🆓</tg-emoji><tg-emoji emoji-id='1'>🆓</tg-emoji><tg-emoji emoji-id='1'>🆓</tg-emoji><tg-emoji emoji-id='1'>🆓</tg-emoji> <tg-spoiler>secret</tg-spoiler>",
        None,
        None,
        None,
    )
    assert "tg-emoji" not in html
    assert "tg-spoiler" not in html
    assert "secret" in html
    assert "Бесплатно" in html


@pytest.mark.asyncio
async def test_build_source_page_content_inline_images():
    html, _, uploaded = await main.build_source_page_content(
        "T",
        "line1\nline2\nline3",
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[
            "http://cat/0.jpg",
            "http://cat/1.jpg",
            "http://cat/2.jpg",
        ],
        image_mode="inline",
    )
    assert uploaded == 3
    assert html.count('<img src="http://cat/') == 3
    first_paragraph = html.index("<p>line1</p>")
    second_paragraph = html.index("<p>line2</p>")
    third_paragraph = html.index("<p>line3</p>")
    tail1 = html.index('<img src="http://cat/1.jpg"/>')
    tail2 = html.index('<img src="http://cat/2.jpg"/>')
    assert first_paragraph < tail1 < second_paragraph
    assert second_paragraph < tail2 < third_paragraph


@pytest.mark.asyncio
async def test_build_source_page_content_history_footer():
    source = "https://example.com/src"
    html, _, _ = await main.build_source_page_content(
        "T",
        "line1\nline2",
        source,
        None,
        None,
        None,
        None,
        catbox_urls=["http://cat/0.jpg"],
        page_mode="history",
    )
    assert "https://t.me/kgdstories" in html
    assert "https://t.me/kenigevents" not in html
    assert f'<p><a href="{source}">Источник</a></p>' in html
    assert html.rstrip().endswith(main.HISTORY_FOOTER_HTML)


@pytest.mark.asyncio
async def test_build_source_page_content_history_strips_vk_links():
    source = "https://vk.com/source"
    text = (
        "Смотрите https://vk.com/club123\n"
        "Также https://m.vk.com/event987 и https://example.com/page\n"
        "[club999|ВК синтаксис]"
    )
    html, _, _ = await main.build_source_page_content(
        "T",
        text,
        source,
        None,
        None,
        None,
        None,
        page_mode="history",
    )
    assert html.count('href="https://vk.com') == 1
    assert f'<p><a href="{source}">Источник</a></p>' in html
    assert 'href="https://example.com/page"' in html
