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
    assert '<figure><img src="http://cat/1.jpg"/></figure>' in html
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
