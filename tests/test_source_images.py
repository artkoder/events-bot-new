from types import SimpleNamespace

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


def test_apply_ics_link_appends_to_date_paragraph():
    html = "<p>🗓 1 мая ⏰ 19:00</p><p>body</p>"
    res = main.apply_ics_link(html, "http://ics")
    assert (
        '<p>🗓 1 мая ⏰ 19:00 📅 <a href="http://ics">Добавить в календарь</a></p>'
        in res
    )


def test_apply_ics_link_inserts_before_br():
    html = "<p>🗓 1 мая ⏰ 19:00<br/>📍 Место</p>"
    res = main.apply_ics_link(html, "http://ics")
    assert (
        '🗓 1 мая ⏰ 19:00 📅 <a href="http://ics">Добавить в календарь</a><br/>📍 Место'
        in res
    )


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
async def test_build_source_page_content_summary_block_with_ics():
    summary = main.SourcePageEventSummary(
        date="2024-06-01",
        time="18:30",
        location_name="Место",
    )
    html, _, _ = await main.build_source_page_content(
        "Title",
        "Body",
        None,
        None,
        None,
        "http://ics",
        None,
        event_summary=summary,
    )
    assert (
        '🗓 1 июня ⏰ 18:30 📅 <a href="http://ics">Добавить в календарь</a>'
        in html
    )
    assert '<br/>📍 Место' in html


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
    text = "\n\n".join(f"paragraph {idx}" for idx in range(1, 7))
    html, _, uploaded = await main.build_source_page_content(
        "T",
        text,
        None,
        None,
        None,
        None,
        None,
        catbox_urls=[
            "http://cat/0.jpg",
            "http://cat/1.jpg",
            "http://cat/2.jpg",
            "http://cat/3.jpg",
            "http://cat/4.jpg",
        ],
        image_mode="inline",
    )
    assert uploaded == 5
    assert html.count('<img src="http://cat/') == 5
    spacer = main.BODY_SPACER_HTML
    assert f"<p>paragraph 1</p>{spacer}<p>paragraph 2</p>" in html
    assert f"<p>paragraph 2</p>{spacer}<img src=\"http://cat/1.jpg\"/>" in html
    assert f"<img src=\"http://cat/1.jpg\"/>{spacer}<p>paragraph 3</p>" in html
    paragraph_positions = [
        html.index(f"<p>paragraph {idx}</p>") for idx in range(1, 7)
    ]
    image_positions = [
        html.index(f'<img src="http://cat/{idx}.jpg"/>') for idx in range(1, 5)
    ]
    assert paragraph_positions[1] < image_positions[0] < paragraph_positions[2]
    assert paragraph_positions[2] < image_positions[1] < paragraph_positions[3]
    assert paragraph_positions[3] < image_positions[2] < paragraph_positions[4]
    assert paragraph_positions[4] < image_positions[3] < paragraph_positions[5]


@pytest.mark.asyncio
async def test_build_source_page_content_history_heading_without_spacer():
    html, _, _ = await main.build_source_page_content(
        "T",
        "",
        None,
        "<h2>История</h2><p>Текст истории</p>",
        None,
        None,
        None,
        page_mode="history",
    )
    heading_close = "</h2>" if "</h2>" in html else "</h3>"
    assert "<p>Текст истории</p>" in html
    close_index = html.index(heading_close) + len(heading_close)
    paragraph_index = html.index("<p>Текст истории</p>")
    assert close_index <= paragraph_index
    assert main.BODY_SPACER_HTML not in html[close_index:paragraph_index]


@pytest.mark.asyncio
async def test_build_source_page_content_history_inline_images_without_spacer():
    html, _, _ = await main.build_source_page_content(
        "T",
        "",
        None,
        "<h2>Заголовок</h2><p>Первый абзац</p><p>Второй абзац</p>",
        None,
        None,
        None,
        catbox_urls=[
            "http://cat/0.jpg",
            "http://cat/1.jpg",
            "http://cat/2.jpg",
        ],
        image_mode="inline",
        page_mode="history",
    )
    figure_end = html.index("</figure>") + len("</figure>")
    first_paragraph = html.index("<p>Первый абзац</p>")
    assert figure_end <= first_paragraph
    assert main.BODY_SPACER_HTML not in html[figure_end:first_paragraph]
    first_image_fragment = '<img src="http://cat/1.jpg"/>'
    first_image_end = html.index(first_image_fragment) + len(first_image_fragment)
    assert main.BODY_SPACER_HTML not in html[first_image_end:html.index("<p>Первый абзац</p>")]
    second_image_fragment = '<img src="http://cat/2.jpg"/>'
    second_image_end = html.index(second_image_fragment) + len(second_image_fragment)
    second_paragraph = html.index("<p>Второй абзац</p>")
    assert second_image_end <= second_paragraph
    assert main.BODY_SPACER_HTML not in html[second_image_end:second_paragraph]


@pytest.mark.asyncio
async def test_build_source_page_content_history_spacing():
    text = "Первый абзац\n\nВторой абзац\n\nТретий абзац"
    html, _, _ = await main.build_source_page_content(
        "История",
        text,
        "https://example.com/story",
        None,
        None,
        None,
        None,
        page_mode="history",
    )
    spacer = main.BODY_SPACER_HTML
    first = "<p>Первый абзац</p>"
    second = "<p>Второй абзац</p>"
    third = "<p>Третий абзац</p>"
    assert f"{first}{spacer}{second}" in html
    assert f"{second}{spacer}{third}" in html
    assert not html.rstrip().endswith(spacer)


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
async def test_build_source_page_content_summary_block(monkeypatch):
    async def fake_location(parts):
        return ", ".join(part.strip() for part in parts if part)

    monkeypatch.setattr(main, "build_short_vk_location", fake_location)

    summary = main.SourcePageEventSummary(
        date="2024-05-01",
        time="19:00",
        location_name="Дом",
        location_address="Улица",
        city="Калининград",
        ticket_price_min=500,
        ticket_price_max=1000,
        ticket_link="https://tickets.example.com/show",
    )
    html, _, _ = await main.build_source_page_content(
        "Заголовок",
        "Основной текст",
        None,
        None,
        None,
        None,
        None,
        event_summary=summary,
    )
    assert (
        '<p>🗓 1 мая ⏰ 19:00<br/>📍 Дом, Улица, Калининград<br/>🎟 '
        '<a href="https://tickets.example.com/show">Билеты</a> '
        "от 500 до 1000 руб.</p>" in html
    )


@pytest.mark.asyncio
async def test_build_source_page_content_summary_block_free(monkeypatch):
    summary = main.SourcePageEventSummary(
        date="2024-05-02",
        location_name="Локация",
        location_address="Адрес",
        city="Город",
        ticket_link="https://example.org/register",
        is_free=True,
    )
    html, _, _ = await main.build_source_page_content(
        "Title",
        "Body",
        None,
        None,
        None,
        None,
        None,
        event_summary=summary,
    )
    assert (
        '<p>🗓 2 мая<br/>📍 Локация, Адрес, Город<br/>🆓 '
        '<a href="https://example.org/register">Бесплатно, по регистрации</a></p>'
        in html
    )


@pytest.mark.asyncio
async def test_build_source_page_content_summary_block_missing_fields():
    summary = main.SourcePageEventSummary()
    html, _, _ = await main.build_source_page_content(
        "Title",
        "Body text",
        None,
        None,
        None,
        None,
        None,
        event_summary=summary,
    )
    assert html.startswith('<p>Body text</p>')
    assert '🗓' not in html
    assert '📍' not in html
    assert '🎟' not in html
    assert '🆓' not in html


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


@pytest.mark.asyncio
async def test_build_source_page_content_editor_blocks():
    html, _, _ = await main.build_source_page_content(
        "Заголовок",
        "Исходный текст",
        "https://example.com",
        "### Анонс\n\nПервый абзац.\n\nВторой с https://example.com",
        None,
        None,
        None,
    )
    assert "<h3>Анонс</h3>" in html
    assert "<p>Первый абзац.</p>" in html
    assert (
        '<p>Второй с <a href="https://example.com">https://example.com</a></p>'
        in html
    )


@pytest.mark.asyncio
async def test_build_source_page_content_consecutive_headings():
    raw_html = "<h3>First heading<h3>Second heading<p>First paragraph<p>Second paragraph"
    html, _, _ = await main.build_source_page_content(
        "Title",
        "text",
        None,
        raw_html,
    )
    assert html.count("<h3>First heading</h3>") == 1
    assert html.count("<h3>Second heading</h3>") == 1
    assert html.count("<p>First paragraph</p>") == 1
    assert html.count("<p>Second paragraph</p>") == 1


@pytest.mark.asyncio
async def test_vkrev_story_editor_fallback(monkeypatch):
    operator_id = 123
    inbox_id = 456
    text = "Исходный пост"
    main.vk_review_story_sessions[operator_id] = main.VkReviewStorySession(
        inbox_id=inbox_id,
        batch_id="batch",
    )

    class DummyCursor:
        def __init__(self, row):
            self._row = row

        async def fetchone(self):
            return self._row

    class DummyRawConn:
        def __init__(self, row):
            self._row = row

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, *_args):
            return DummyCursor(self._row)

    class DummyDB:
        def __init__(self, row):
            self._row = row

        def raw_conn(self):
            return DummyRawConn(self._row)

    dummy_db = DummyDB((-inbox_id, inbox_id, text))

    class DummyBot:
        def __init__(self):
            self.messages: list[tuple[int, str]] = []

        async def send_message(self, chat_id, message):
            self.messages.append((chat_id, message))

    class DummyMessage:
        def __init__(self, chat_id):
            self.chat = SimpleNamespace(id=chat_id)
            self.edited = False

        async def edit_reply_markup(self):
            self.edited = True

    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=operator_id),
        message=DummyMessage(999),
    )

    captured: dict[str, object] = {}

    async def fake_create_source_page(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ("https://telegraph", "/path", "", 0)

    async def fake_fetch_photos(*_args, **_kwargs):
        return []

    async def fake_editor(*_args, **_kwargs):
        raise RuntimeError("boom")

    async def fake_pitch(*_args, **_kwargs):
        return "Заманчивое предложение"

    monkeypatch.setattr(main, "create_source_page", fake_create_source_page)
    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch_photos)
    monkeypatch.setattr(main, "compose_story_editorial_via_4o", fake_editor)
    monkeypatch.setattr(main, "compose_story_pitch_via_4o", fake_pitch)

    bot = DummyBot()
    await main._vkrev_handle_story_choice(callback, "end", inbox_id, dummy_db, bot)

    assert "args" in captured
    args = captured["args"]
    assert args[1] == text
    assert args[3] == "<p><i>Заманчивое предложение</i></p>"
    assert bot.messages
    assert bot.messages[-1][1].splitlines()[-1] == "Заманчивое предложение"
    assert main.vk_review_story_sessions.get(operator_id) is None


@pytest.mark.asyncio
async def test_vkrev_story_editor_includes_pitch(monkeypatch):
    operator_id = 456
    inbox_id = 789
    text = "Первый абзац\nВторой абзац"
    main.vk_review_story_sessions[operator_id] = main.VkReviewStorySession(
        inbox_id=inbox_id,
        batch_id="batch2",
    )

    class DummyCursor:
        def __init__(self, row):
            self._row = row

        async def fetchone(self):
            return self._row

    class DummyRawConn:
        def __init__(self, row):
            self._row = row

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def execute(self, *_args):
            return DummyCursor(self._row)

    class DummyDB:
        def __init__(self, row):
            self._row = row

        def raw_conn(self):
            return DummyRawConn(self._row)

    dummy_db = DummyDB((-inbox_id, inbox_id, text))

    class DummyBot:
        def __init__(self):
            self.messages: list[tuple[int, str]] = []

        async def send_message(self, chat_id, message):
            self.messages.append((chat_id, message))

    class DummyMessage:
        def __init__(self, chat_id):
            self.chat = SimpleNamespace(id=chat_id)
            self.edited = False

        async def edit_reply_markup(self):
            self.edited = True

    callback = SimpleNamespace(
        from_user=SimpleNamespace(id=operator_id),
        message=DummyMessage(555),
    )

    captured: dict[str, object] = {}

    async def fake_create_source_page(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ("https://telegraph", "/path", "", 0)

    async def fake_fetch_photos(*_args, **_kwargs):
        return []

    async def fake_editor(*_args, **_kwargs):
        return "<p>Отредактированный текст</p>"

    async def fake_pitch(*_args, **_kwargs):
        return "Приходите на главное событие недели"

    monkeypatch.setattr(main, "create_source_page", fake_create_source_page)
    monkeypatch.setattr(main, "_vkrev_fetch_photos", fake_fetch_photos)
    monkeypatch.setattr(main, "compose_story_editorial_via_4o", fake_editor)
    monkeypatch.setattr(main, "compose_story_pitch_via_4o", fake_pitch)

    bot = DummyBot()
    await main._vkrev_handle_story_choice(callback, "middle", inbox_id, dummy_db, bot)

    args = captured["args"]
    assert args[1] == text
    assert args[3] == (
        "<p><i>Приходите на главное событие недели</i></p>\n"
        "<p>Отредактированный текст</p>"
    )
    assert bot.messages
    message_lines = bot.messages[-1][1].splitlines()
    assert message_lines[-1] == "Приходите на главное событие недели"
    assert main.vk_review_story_sessions.get(operator_id) is None
