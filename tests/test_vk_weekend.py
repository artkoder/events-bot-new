import pytest
from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from main import Database, WeekendPage, Event, format_weekend_range


@pytest.mark.asyncio
async def test_build_weekend_vk_message(tmp_path: Path):
    db = Database(str(tmp_path / 'db.sqlite'))
    await db.init()
    sat = date(2025, 7, 12)
    next_sat = sat + timedelta(days=7)
    async with db.get_session() as session:
        session.add(WeekendPage(start=sat.isoformat(), url='u1', path='p1'))
        session.add(WeekendPage(start=next_sat.isoformat(), url='u2', path='p2', vk_post_url='https://vk.com/wall-1_2'))
        session.add(Event(title='Party', description='d', source_text='s', date=sat.isoformat(), time='10:00', location_name='Club', city='Kaliningrad', source_vk_post_url='https://vk.com/wall-1_1'))
        session.add(Event(title='NoVK', description='d', source_text='s', date=sat.isoformat(), time='11:00', location_name='Hall', city='Kaliningrad'))
        await session.commit()
    msg = await main.build_weekend_vk_message(db, sat.isoformat())
    assert '[https://vk.com/wall-1_1|Party]' in msg
    assert 'NoVK' not in msg
    assert f'[https://vk.com/wall-1_2|{format_weekend_range(next_sat)}]' in msg
    assert msg.splitlines()[0] == f'{format_weekend_range(sat)} Афиша выходных'


@pytest.mark.asyncio
async def test_sync_weekend_page_posts_vk(tmp_path: Path, monkeypatch):
    db = Database(str(tmp_path / 'db.sqlite'))
    await db.init()
    sat = date(2025, 7, 12)
    await main.set_vk_group_id(db, '1')

    class DummyTG:
        def create_page(self, title, content):
            return {'url': 'u1', 'path': 'p1'}

        def edit_page(self, path, title=None, content=None):
            pass

    monkeypatch.setattr('main.get_telegraph_token', lambda: 't')
    monkeypatch.setattr('main.Telegraph', lambda access_token=None, domain=None: DummyTG())

    posted = {}

    async def fake_post_to_vk(group_id, message, db=None, bot=None, attachments=None, token=None):
        posted['message'] = message
        return 'https://vk.com/wall-1_123'

    monkeypatch.setattr(main, 'post_to_vk', fake_post_to_vk)

    async with db.get_session() as session:
        session.add(Event(title='Party', description='d', source_text='s', date=sat.isoformat(), time='10:00', location_name='Club', city='Kaliningrad', source_vk_post_url='https://vk.com/wall-1_1'))
        await session.commit()

    await main.sync_weekend_page(db, sat.isoformat())
    assert 'message' in posted
    async with db.get_session() as session:
        wp = await session.get(WeekendPage, sat.isoformat())
        assert wp and wp.vk_post_url == 'https://vk.com/wall-1_123'
