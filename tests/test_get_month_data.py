import pytest

import main_part2
from models import Event


@pytest.mark.asyncio
async def test_get_month_data_returns_events_and_exhibitions(tmp_path):
    db = main_part2.Database(str(tmp_path / "db.sqlite"))
    await db.init()

    async with db.get_session() as session:
        session.add_all(
            [
                Event(
                    title="Event A",
                    description="desc",
                    source_text="src",
                    date="2099-01-10",
                    time="10:00",
                    location_name="loc",
                ),
                Event(
                    title="Exhibit",
                    description="desc",
                    source_text="src",
                    date="2098-12-20",
                    time="11:00",
                    location_name="loc",
                    event_type="\u0432\u044b\u0441\u0442\u0430\u0432\u043a\u0430",
                    end_date="2099-01-05",
                ),
                Event(
                    title="Event B",
                    description="desc",
                    source_text="src",
                    date="2099-02-02",
                    time="12:00",
                    location_name="loc",
                ),
            ]
        )
        await session.commit()

    events, exhibitions = await main_part2.get_month_data(
        db, "2099-01", fallback=False
    )

    assert {event.title for event in events} == {"Event A"}
    assert {event.title for event in exhibitions} == {"Exhibit"}
