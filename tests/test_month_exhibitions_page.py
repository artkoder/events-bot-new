import importlib
from unittest.mock import AsyncMock

import pytest
from telegraph.utils import nodes_to_html

import main as orig_main


@pytest.mark.asyncio
async def test_build_month_page_content_links_to_separate_exhibitions_page():
    m = importlib.reload(orig_main)

    month = "2026-03"
    event = m.Event(
        title="Концерт",
        description="Описание концерта.",
        source_text="Источник",
        date="2026-03-20",
        time="19:00",
        location_name="Зал",
        city="Калининград",
    )

    title, content, _ = m._build_month_page_content_sync(
        month,
        [event],
        [],
        {},
        None,
        None,
        None,
        None,
        True,
        True,
        1,
        None,
        None,
        "https://telegra.ph/exhibitions-march",
    )
    html = m.unescape_html_comments(nodes_to_html(content))

    assert "События Калининграда" in title
    assert "Постоянные выставки марта" in html
    assert "https://telegra.ph/exhibitions-march" in html


@pytest.mark.asyncio
async def test_sync_month_page_creates_separate_month_exhibitions_page(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)

    monkeypatch.setattr(m, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(m, "ensure_event_telegraph_link", AsyncMock())
    monkeypatch.setattr(m, "build_month_nav_block", AsyncMock(return_value=""))
    monkeypatch.setattr(m, "check_month_page_markers", AsyncMock())
    monkeypatch.setattr(m, "refresh_month_nav", AsyncMock())

    class FakeTelegraph:
        def __init__(self, access_token=None):
            self.access_token = access_token

    monkeypatch.setattr(m, "Telegraph", FakeTelegraph)

    created_pages: list[tuple[str, str]] = []

    async def fake_create_page(tg, *, title, html_content, caller="event_pipeline", **kwargs):
        idx = len(created_pages) + 1
        created_pages.append((title, html_content))
        return {"url": f"https://telegra.ph/page-{idx}", "path": f"page-{idx}"}

    monkeypatch.setattr(m, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(m, "telegraph_edit_page", AsyncMock())

    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    month = "2026-03"
    async with db.get_session() as session:
        session.add(
            m.Event(
                title="Главное событие",
                description="Описание события.",
                source_text="Источник",
                date="2026-03-20",
                time="18:00",
                location_name="Сцена",
                city="Калининград",
            )
        )
        for idx in range(11):
            session.add(
                m.Event(
                    title=f"Выставка {idx}",
                    description="Описание выставки.",
                    source_text="Источник выставки.",
                    date="2026-03-01",
                    time="",
                    end_date="2026-04-01",
                    location_name=f"Музей {idx}",
                    city="Калининград",
                    event_type="выставка",
                )
            )
        await session.commit()

    await m.sync_month_page(db, month, force=True)

    async with db.get_session() as session:
        month_page = await session.get(m.MonthPage, month)
        exhibitions_page = await session.get(m.MonthExhibitionsPage, month)

    assert month_page is not None
    assert exhibitions_page is not None
    assert exhibitions_page.url == "https://telegra.ph/page-1"
    assert len(created_pages) == 2

    exhibitions_title, exhibitions_html = created_pages[0]
    month_title, month_html = created_pages[1]

    assert "Постоянные выставки марта 2026" in exhibitions_title
    assert "Выставка 0" in exhibitions_html
    assert "События Калининграда" in month_title
    assert "Постоянные выставки марта" in month_html
    assert exhibitions_page.url in month_html
    assert "Выставка 0" not in month_html


def test_dedupe_exhibitions_for_display_collapses_existing_duplicate_rows():
    m = importlib.reload(orig_main)

    older = m.Event(
        id=2725,
        title="Путешествие Матрешки",
        description="Описание.",
        source_text="Источник.",
        date="2026-03-05",
        time="",
        end_date="2026-04-05",
        location_name="Музей Изобразительных искусств, Ленинский проспект 83, Калининград",
        location_address="Ленинский проспект 83",
        city="Калининград",
        event_type="выставка",
        telegraph_url="https://telegra.ph/old",
    )
    duplicate = m.Event(
        id=2961,
        title="Путешествие Матрешки",
        description="Описание.",
        source_text="Источник.",
        date="2026-03-10",
        time="",
        end_date="2026-04-05",
        location_name="Музей",
        city="Калининград",
        event_type="выставка",
        telegraph_url="https://telegra.ph/new",
    )

    deduped = m.dedupe_exhibitions_for_display([duplicate, older])

    assert len(deduped) == 1
    assert deduped[0].id == 2725


@pytest.mark.asyncio
async def test_build_month_exhibitions_page_content_dedupes_duplicate_exhibitions(tmp_path, monkeypatch):
    m = importlib.reload(orig_main)
    monkeypatch.setattr(m, "ensure_event_telegraph_link", AsyncMock())

    db = m.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    first = m.Event(
        id=2755,
        title="Женственность через века",
        description="Описание выставки.",
        source_text="Источник.",
        date="2026-03-05",
        time="12:00",
        end_date="2026-06-05",
        location_name="Информационно-туристический центр",
        location_address="Пионерская 2",
        city="Черняховск",
        event_type="выставка",
        telegraph_url="https://telegra.ph/first",
    )
    duplicate = m.Event(
        id=2756,
        title="Женственность через века",
        description="Описание выставки.",
        source_text="Источник.",
        date="2026-03-05",
        time="15:00",
        end_date="2026-06-05",
        location_name="Информационно-туристический центр",
        location_address="Пионерская 2",
        city="Черняховск",
        event_type="выставка",
        telegraph_url="https://telegra.ph/duplicate",
    )

    _, content, _ = await m.build_month_exhibitions_page_content(
        db,
        "2026-03",
        [first, duplicate],
    )
    html = m.unescape_html_comments(nodes_to_html(content))

    assert html.count("Женственность через века") == 1
