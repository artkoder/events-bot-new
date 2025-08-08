import pytest
from datetime import date, timedelta
from pathlib import Path

import main
from main import Database, WeekPage, Event, format_week_range

@pytest.mark.asyncio
async def test_build_week_vk_message(tmp_path: Path):
    db = Database(str(tmp_path / 'db.sqlite'))
    await db.init()
    mon = date(2025, 7, 7)
    next_mon = mon + timedelta(days=7)
    async with db.get_session() as session:
        session.add(WeekPage(start=mon.isoformat()))
        session.add(WeekPage(start=next_mon.isoformat(), vk_post_url='https://vk.com/wall-1_3'))
        session.add(WeekPage(start=date(2025, 8, 4).isoformat(), vk_post_url='https://vk.com/wall-1_4'))
        session.add(Event(title='Party', description='d', source_text='s', date=mon.isoformat(), time='10:00', location_name='Club', city='Kaliningrad', source_vk_post_url='https://vk.com/wall-1_1'))
        session.add(Event(title='NoVK', description='d', source_text='s', date=mon.isoformat(), time='11:00', location_name='Hall', city='Kaliningrad'))
        await session.commit()
    msg = await main.build_week_vk_message(db, mon.isoformat())
    assert '10:00 | [https://vk.com/wall-1_1|Party]' in msg
    assert 'NoVK' not in msg
    assert f'[https://vk.com/wall-1_3|{format_week_range(next_mon)}]' in msg
    assert 'июль [https://vk.com/wall-1_4|август]' in msg
