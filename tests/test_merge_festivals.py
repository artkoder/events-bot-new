import os
import sys

import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import main
from main import Database, merge_festivals, normalize_alias
from models import Event, Festival


@pytest.mark.asyncio
async def test_merge_festivals_preserves_manual_urls(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db = Database(str(db_path))
    await db.init()

    async def fake_ask(prompt):
        return "generated"

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "sync_festival_page", noop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", noop)
    monkeypatch.setattr(main, "sync_festival_vk_post", noop)

    async with db.get_session() as session:
        dst = Festival(
            name="DestFest",
            telegraph_url="https://tele.dst",
            telegraph_path="dst-path",
            vk_post_url="https://vk.com/dst",
            vk_poll_url="https://vk.com/poll_dst",
            website_url=None,
            ticket_url="https://tickets.dst",
        )
        src = Festival(
            name="SrcFest",
            telegraph_url="https://tele.src",
            telegraph_path="src-path",
            vk_post_url="https://vk.com/src",
            vk_poll_url="https://vk.com/poll_src",
            website_url="https://src.site",
            ticket_url="https://tickets.src",
        )
        session.add(dst)
        session.add(src)
        await session.flush()
        dst_id = dst.id
        src_id = src.id

        event = Event(
            title="Event",
            description="Event description",
            festival=src.name,
            date="2025-01-01",
            time="10:00",
            location_name="Location",
            source_text="Source",
        )
        session.add(event)
        await session.flush()
        event_id = event.id

        await session.commit()

    await merge_festivals(db, src_id, dst_id, bot=None)

    async with db.get_session() as session:
        dst_fest = await session.get(Festival, dst_id)
        assert dst_fest is not None
        assert dst_fest.telegraph_url == "https://tele.dst"
        assert dst_fest.telegraph_path == "dst-path"
        assert dst_fest.vk_post_url == "https://vk.com/dst"
        assert dst_fest.vk_poll_url == "https://vk.com/poll_dst"
        assert dst_fest.website_url == "https://src.site"
        assert dst_fest.ticket_url == "https://tickets.dst"

        src_fest = await session.get(Festival, src_id)
        assert src_fest is None

        merged_event = await session.get(Event, event_id)
        assert merged_event is not None
        assert merged_event.festival == "DestFest"

    await db.engine.dispose()


@pytest.mark.asyncio
async def test_merge_festivals_limits_aliases(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db = Database(str(db_path))
    await db.init()

    async def fake_ask(prompt):
        return "generated"

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "sync_festival_page", noop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", noop)
    monkeypatch.setattr(main, "sync_festival_vk_post", noop)

    async with db.get_session() as session:
        dst = Festival(
            name="DestFest",
            aliases=["Фестиваль \"Весна\"", "Весна", "Зима"],
        )
        src = Festival(
            name="Супер фестиваль",
            full_name='Фестиваль "Супер Весна"',
            aliases=[
                "Международный Весна",
                "Лето",
                "Весна",
                "Осень",
                "Областной праздник",
                "Городской праздник",
                "Утро",
                "Вечер",
                "Ночь",
            ],
        )
        session.add(dst)
        session.add(src)
        await session.flush()
        dst_id = dst.id
        src_id = src.id
        await session.commit()

    await merge_festivals(db, src_id, dst_id, bot=None)

    async with db.get_session() as session:
        dst_fest = await session.get(Festival, dst_id)
        assert dst_fest is not None
        assert dst_fest.aliases == [
            "весна",
            "зима",
            "лето",
            "осень",
            "праздник",
            "утро",
            "вечер",
            "ночь",
        ]
        normalized_aliases = [normalize_alias(alias) for alias in dst_fest.aliases]
        assert len(normalized_aliases) == len(set(normalized_aliases)) <= 8

    await db.engine.dispose()


@pytest.mark.asyncio
async def test_merge_festivals_updates_description_on_success(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db = Database(str(db_path))
    await db.init()

    generated_text = "Короткое описание. Второе предложение."

    async def fake_ask(prompt):
        return generated_text

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "sync_festival_page", noop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", noop)
    monkeypatch.setattr(main, "sync_festival_vk_post", noop)

    async with db.get_session() as session:
        dst = Festival(
            name="MainFest",
            description="Старое описание",
            city="Казань",
        )
        src = Festival(name="SrcFest")
        session.add_all([dst, src])
        await session.flush()
        dst_id = dst.id
        src_id = src.id

        session.add_all(
            [
                Event(
                    title="Открытие",
                    description="",
                    festival=src.name,
                    date="2025-07-01",
                    time="10:00",
                    location_name="Театр",
                    city="Казань",
                    source_text="",
                ),
                Event(
                    title="Лекция",
                    description="",
                    festival=src.name,
                    date="2025-07-02",
                    time="12:00",
                    location_name="Музей",
                    city="Казань",
                    source_text="",
                ),
            ]
        )
        await session.commit()

    await merge_festivals(db, src_id, dst_id, bot=None)

    async with db.get_session() as session:
        dst_fest = await session.get(Festival, dst_id)
        assert dst_fest is not None
        assert dst_fest.description == generated_text

    await db.engine.dispose()


@pytest.mark.asyncio
async def test_merge_festivals_keeps_description_on_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "db.sqlite"
    db = Database(str(db_path))
    await db.init()

    async def fake_ask(prompt):
        return "Слишком длинный текст " * 40

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "ask_4o", fake_ask)
    monkeypatch.setattr(main, "sync_festival_page", noop)
    monkeypatch.setattr(main, "rebuild_fest_nav_if_changed", noop)
    monkeypatch.setattr(main, "sync_festival_vk_post", noop)

    async with db.get_session() as session:
        dst = Festival(
            name="MainFest",
            description="Актуальное описание",
            city="Казань",
        )
        src = Festival(name="SrcFest")
        session.add_all([dst, src])
        await session.flush()
        dst_id = dst.id
        src_id = src.id

        session.add(
            Event(
                title="Открытие",
                description="",
                festival=src.name,
                date="2025-07-01",
                time="10:00",
                location_name="Театр",
                city="Казань",
                source_text="",
            )
        )
        await session.commit()

    await merge_festivals(db, src_id, dst_id, bot=None)

    async with db.get_session() as session:
        dst_fest = await session.get(Festival, dst_id)
        assert dst_fest is not None
        assert dst_fest.description == "Актуальное описание"

    await db.engine.dispose()
