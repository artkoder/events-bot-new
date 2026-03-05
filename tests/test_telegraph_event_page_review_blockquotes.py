from __future__ import annotations

from pathlib import Path

import pytest

import main
from db import Database
from models import Event


@pytest.mark.asyncio
async def test_update_telegraph_event_page_promotes_review_bullets(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    event = Event(
        id=1,
        title="My Event",
        description=(
            "### Особенности постановки и отзывы\n\n"
            "Зрители в восторге:\n"
            "- Лариса: Превосходный спектакль!!!\n"
            "- Мария: Отличный спектакль!!!\n"
        ),
        source_text="src",
        date="2026-03-05",
        time="19:00",
        location_name="Place",
    )
    async with db.get_session() as session:
        session.add(event)
        await session.commit()

    captured: dict[str, str] = {}

    async def fake_build_source_page_content(title, text, *_args, **_kwargs):  # noqa: ANN001 - test helper
        captured["text"] = str(text or "")
        return "<p>ok</p>", None, None

    monkeypatch.setattr(main, "build_source_page_content", fake_build_source_page_content)

    from telegraph import utils as t_utils

    monkeypatch.setattr(t_utils, "html_to_nodes", lambda _html: [{"tag": "p", "children": ["ok"]}])

    async def fake_create_page(tg, title, content=None, **_kwargs):  # noqa: ANN001 - test helper
        _ = (tg, title, content)
        return {"url": "url", "path": "p"}

    monkeypatch.setattr(main, "telegraph_create_page", fake_create_page)
    monkeypatch.setattr(main, "get_telegraph_token", lambda: "t")
    monkeypatch.setattr(main, "Telegraph", lambda access_token=None: object())

    await main.update_telegraph_event_page(1, db, None)

    text = captured.get("text") or ""
    assert "> «Превосходный спектакль!!!»\n> — Лариса" in text
    assert "> «Отличный спектакль!!!»\n> — Мария" in text
    assert "- Лариса:" not in text
    assert "- Мария:" not in text
