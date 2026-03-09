from datetime import datetime, timedelta, timezone

import pytest

from db import Database
from handlers import recent_imports_cmd as recent
from models import Event, EventSource


@pytest.mark.asyncio
async def test_recent_imports_load_rows_groups_created_vs_updated(tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=24)
    try:
        async with db.get_session() as session:
            ev_tg_new = Event(
                title="TG New",
                description="desc",
                date="2026-03-09",
                time="19:00",
                location_name="Loc",
                source_text="src",
                telegraph_url="https://telegra.ph/tg-new",
                added_at=now - timedelta(hours=2),
            )
            ev_tg_updated = Event(
                title="TG Updated",
                description="desc",
                date="2026-03-10",
                time="19:00",
                location_name="Loc",
                source_text="src",
                telegraph_path="tg-updated",
                added_at=now - timedelta(days=3),
            )
            ev_vk_updated = Event(
                title="VK Updated",
                description="desc",
                date="2026-03-11",
                time="19:00",
                location_name="Loc",
                source_text="src",
                telegraph_url="https://telegra.ph/vk-updated",
                added_at=now - timedelta(days=2),
            )
            ev_parse_new = Event(
                title="Parse New",
                description="desc",
                date="2026-03-12",
                time="19:00",
                location_name="Loc",
                source_text="src",
                telegraph_path="parse-new",
                added_at=now - timedelta(hours=3),
            )
            session.add_all([ev_tg_new, ev_tg_updated, ev_vk_updated, ev_parse_new])
            await session.flush()
            session.add_all(
                [
                    EventSource(
                        event_id=int(ev_tg_new.id),
                        source_type="telegram",
                        source_url="https://t.me/source/1",
                        imported_at=now - timedelta(hours=1),
                    ),
                    EventSource(
                        event_id=int(ev_tg_updated.id),
                        source_type="tg",
                        source_url="https://t.me/source/2",
                        imported_at=now - timedelta(hours=4),
                    ),
                    EventSource(
                        event_id=int(ev_vk_updated.id),
                        source_type="vk",
                        source_url="https://vk.com/wall-1_2",
                        imported_at=now - timedelta(hours=5),
                    ),
                    EventSource(
                        event_id=int(ev_parse_new.id),
                        source_type="parser:theatres",
                        source_url="https://example.com/event",
                        imported_at=now - timedelta(hours=2),
                    ),
                    EventSource(
                        event_id=int(ev_parse_new.id),
                        source_type="parser:theatres",
                        source_url="https://example.com/event-duplicate",
                        imported_at=now - timedelta(hours=1, minutes=30),
                    ),
                ]
            )
            await session.commit()

        grouped = await recent._load_recent_import_rows(db, start_utc=start, end_utc=now)

        tg_rows = grouped["telegram"]
        vk_rows = grouped["vk"]
        parse_rows = grouped["parse"]

        assert [row.title for row in tg_rows] == ["TG New", "TG Updated"]
        assert tg_rows[0].status == "created"
        assert tg_rows[1].status == "updated"
        assert tg_rows[1].telegraph_url == "https://telegra.ph/tg-updated"

        assert len(vk_rows) == 1
        assert vk_rows[0].title == "VK Updated"
        assert vk_rows[0].status == "updated"

        assert len(parse_rows) == 1
        assert parse_rows[0].title == "Parse New"
        assert parse_rows[0].status == "created"
        assert parse_rows[0].telegraph_url == "https://telegra.ph/parse-new"
    finally:
        await db.close()


def test_recent_imports_render_pages_splits_long_output(monkeypatch):
    monkeypatch.setattr(recent, "_MAX_TG_MESSAGE_LEN", 240)
    rows = [
        recent._RecentImportRow(
            source_group="telegram",
            event_id=idx,
            title=f"Event {idx}",
            telegraph_url=f"https://telegra.ph/event-{idx}",
            status="created" if idx % 2 else "updated",
            last_imported_at=None,
        )
        for idx in range(1, 10)
    ]
    pages = recent._render_pages(
        hours=24,
        start_local=datetime(2026, 3, 8, 10, 0, tzinfo=timezone.utc),
        end_local=datetime(2026, 3, 9, 10, 0, tzinfo=timezone.utc),
        tz_label="UTC",
        grouped_rows={"telegram": rows, "vk": rows, "parse": rows},
    )

    assert len(pages) > 1
    assert pages[0].startswith("🧾 <b>События из Telegram / VK / /parse за последние 24 ч.</b>")
    assert "Страница 1/" in pages[0]
    assert any("Страница 2/" in page for page in pages[1:])
    assert any('href="https://telegra.ph/event-1"' in page for page in pages)
    assert "id=1 ✅" in pages[0]
    assert "id=2 🔄" in pages[0]


def test_recent_imports_parse_hours_arg():
    assert recent._parse_hours_arg("/recent_imports") == 24
    assert recent._parse_hours_arg("/recent_imports 48") == 48
    with pytest.raises(ValueError):
        recent._parse_hours_arg("/recent_imports abc")
