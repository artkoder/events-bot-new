from datetime import datetime as real_datetime, timezone

import pytest

import main
from main import Database, Event, VKAPIError


@pytest.mark.asyncio
async def test_collect_vk_shortlink_click_stats_filters_and_sorts(monkeypatch, tmp_path):
    db = Database(str(tmp_path / "db.sqlite"))
    await db.init()

    class FixedDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            base = real_datetime(2025, 7, 10, tzinfo=timezone.utc)
            if tz is None:
                return base
            return base.astimezone(tz)

    monkeypatch.setattr(main, "datetime", FixedDatetime)

    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    title="Future",
                    description="d",
                    source_text="s",
                    date="2025-07-18",
                    time="18:00",
                    location_name="Hall",
                    city="Town",
                    vk_ticket_short_key="future",
                ),
                Event(
                    title="Recent",
                    description="d",
                    source_text="s",
                    date="2025-07-04..2025-07-07",
                    time="12:00",
                    location_name="Hall",
                    city="Town",
                    vk_ticket_short_key="recent",
                ),
                Event(
                    title="Too Old",
                    description="d",
                    source_text="s",
                    date="2025-06-01",
                    end_date="2025-06-15",
                    time="12:00",
                    location_name="Hall",
                    city="Town",
                    vk_ticket_short_key="old",
                ),
                Event(
                    title="Broken",
                    description="d",
                    source_text="s",
                    date="2025-07-12",
                    time="12:00",
                    location_name="Hall",
                    city="Town",
                    vk_ticket_short_key="broken",
                ),
            ]
        )
        await session.commit()

    async def fake_vk_api(method, **params):
        assert method == "utils.getLinkStats"
        key = params["key"]
        if key == "broken":
            raise VKAPIError(1, "fail")
        stats_map = {
            "future": [{"clicks": 5, "views": 12}],
            "recent": [
                {"visitors": 3, "views": 10},
                {"visitors": 1, "views": 4},
            ],
        }
        return {"response": {"stats": stats_map.get(key, [])}}

    monkeypatch.setattr(main, "vk_api", fake_vk_api)

    lines = await main.collect_vk_shortlink_click_stats(db)

    assert lines == ["Future: 5", "Recent: 4"]
