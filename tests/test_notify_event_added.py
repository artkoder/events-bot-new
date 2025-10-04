import pytest
import main


class DummyBot:
    def __init__(self):
        self.sent_messages = []
        self.sent_photos = []
        self.edited_captions = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent_messages.append((chat_id, text, kwargs))
        return type("Msg", (), {"message_id": len(self.sent_messages)})

    async def send_photo(self, chat_id, photo, caption=None, **kwargs):
        self.sent_photos.append((chat_id, photo, caption, kwargs))
        return type("Msg", (), {"message_id": len(self.sent_photos)})

    async def edit_message_caption(self, chat_id, message_id, caption, **kwargs):
        self.edited_captions.append((chat_id, message_id, caption, kwargs))


@pytest.mark.asyncio
async def test_notify_event_added_includes_link(monkeypatch):
    captured = {}
    main._PARTNER_ADMIN_NOTICES.clear()

    async def fake_notify(db, bot, text):
        captured['text'] = text

    monkeypatch.setattr(main, 'notify_superadmin', fake_notify)

    async def fake_admin_id(db):
        return 42

    monkeypatch.setattr(main, 'get_superadmin_id', fake_admin_id)

    user = main.User(user_id=1, username='partner1', is_partner=True)
    ev = main.Event(
        id=100,
        title='Test',
        description='d',
        date='2025-01-04',
        time='12:00',
        location_name='loc',
        source_text='src',
        telegraph_url='http://t',
        photo_urls=['http://photo1', 'http://photo2']
    )
    bot = DummyBot()
    await main.notify_event_added(object(), bot, user, ev, True)

    assert '@partner1' in captured['text']
    assert 'Test' in captured['text']
    assert 'http://t' in captured['text']
    assert bot.sent_photos
    photo_call = bot.sent_photos[0]
    assert photo_call[0] == 42
    assert photo_call[1] == 'http://photo1'
    caption = photo_call[2]
    assert 'Test' in caption
    assert 'Telegraph: http://t' in caption
    assert 'VK:' not in caption

    notice = main._PARTNER_ADMIN_NOTICES.get(ev.id)
    assert notice is not None
    assert notice.caption == caption

    ev.source_vk_post_url = 'http://vk'
    await main._send_or_update_partner_admin_notice(object(), bot, ev, user=user)
    assert bot.edited_captions
    edit_call = bot.edited_captions[0]
    assert edit_call[0] == 42
    assert edit_call[1] == notice.message_id
    assert 'Telegraph: http://t' in edit_call[2]
    assert 'VK: http://vk' in edit_call[2]
