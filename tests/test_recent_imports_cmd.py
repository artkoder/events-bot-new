import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from db import Database
from handlers import recent_imports_cmd
from models import Event, EventSource


def test_parse_hours_arg_defaults_and_validates():
    assert recent_imports_cmd._parse_hours_arg("/recent_imports") == 24
    assert recent_imports_cmd._parse_hours_arg("/recent_imports 48") == 48
    with pytest.raises(ValueError):
        recent_imports_cmd._parse_hours_arg("/recent_imports many")
    with pytest.raises(ValueError):
        recent_imports_cmd._parse_hours_arg("/recent_imports 0")


@pytest.mark.asyncio
async def test_load_recent_import_items_groups_sources_and_created_flag(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc)

    created_event = Event(
        title="TG import",
        description="desc",
        date="2026-03-20",
        time="19:00",
        location_name="Club",
        source_text="src",
        telegraph_url="https://telegra.ph/tg-import",
        added_at=now - timedelta(hours=2),
    )
    updated_event = Event(
        title="Merged import",
        description="desc",
        date="2026-03-21",
        time="18:30",
        location_name="Hall",
        source_text="src",
        telegraph_path="merged-import",
        added_at=now - timedelta(days=5),
    )
    old_event = Event(
        title="Old import",
        description="desc",
        date="2026-03-22",
        time="17:00",
        location_name="Old hall",
        source_text="src",
        added_at=now - timedelta(days=6),
    )

    async with db.get_session() as session:
        session.add_all([created_event, updated_event, old_event])
        await session.commit()
        await session.refresh(created_event)
        await session.refresh(updated_event)
        await session.refresh(old_event)
        session.add_all(
            [
                EventSource(
                    event_id=created_event.id,
                    source_type="telegram",
                    source_url="https://t.me/source/1",
                    imported_at=now - timedelta(hours=2),
                ),
                EventSource(
                    event_id=updated_event.id,
                    source_type="vk",
                    source_url="https://vk.com/wall-1_2",
                    imported_at=now - timedelta(hours=1),
                ),
                EventSource(
                    event_id=updated_event.id,
                    source_type="parser:dramteatr",
                    source_url="https://dramteatr39.ru/event",
                    imported_at=now - timedelta(minutes=30),
                ),
                EventSource(
                    event_id=old_event.id,
                    source_type="telegram",
                    source_url="https://t.me/source/2",
                    imported_at=now - timedelta(hours=30),
                ),
            ]
        )
        await session.commit()

    items, start_utc, end_utc = await recent_imports_cmd._load_recent_import_items(
        db,
        hours=24,
        now_utc=now,
    )

    assert start_utc == now - timedelta(hours=24)
    assert end_utc == now
    assert [item.event_id for item in items] == [updated_event.id, created_event.id]

    assert items[0].is_created is False
    assert items[0].source_labels == {"VK", "/parse"}
    assert items[0].telegraph_url == "https://telegra.ph/merged-import"

    assert items[1].is_created is True
    assert items[1].source_labels == {"Telegram"}

    lines = recent_imports_cmd._render_recent_imports_lines(
        items,
        hours=24,
        start_utc=start_utc,
        end_utc=end_utc,
        tz=timezone.utc,
    )
    assert lines[0] == "📥 <b>Недавние импорты событий</b>"
    assert any(
        line.startswith(
            f'{updated_event.id} 🔄 <a href="https://telegra.ph/merged-import">Merged import</a> |'
        )
        and "VK, /parse" in line
        for line in lines
    )
    assert any(
        line.startswith(
            f'{created_event.id} ✅ <a href="https://telegra.ph/tg-import">TG import</a> |'
        )
        and "Telegram" in line
        for line in lines
    )
