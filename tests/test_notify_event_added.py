import pytest
import main


class DummyBot:
    pass


@pytest.mark.asyncio
async def test_notify_event_added_includes_link(monkeypatch):
    captured = {}

    async def fake_notify(db, bot, text):
        captured['text'] = text

    monkeypatch.setattr(main, 'notify_superadmin', fake_notify)

    user = main.User(user_id=1, username='partner1', is_partner=True)
    ev = main.Event(
        title='Test',
        description='d',
        date='2025-01-04',
        time='12:00',
        location_name='loc',
        source_text='src',
        telegraph_url='http://t'
    )
    await main.notify_event_added(object(), DummyBot(), user, ev, True)

    assert '@partner1' in captured['text']
    assert 'Test' in captured['text']
    assert 'http://t' in captured['text']
