
import pytest
from unittest.mock import AsyncMock, MagicMock
from handlers.channel_nav import handle_channel_post
from db import Database

@pytest.mark.asyncio
async def test_eve_13_button_restrictions():
    """
    Test that buttons are ONLY added if the post contains #анонс or #анонсКалининград.
    """
    # Setup Mocks
    # Setup Mocks
    db = MagicMock(spec=Database)
    bot = AsyncMock()
    
    # Mock DB session and get method
    session = AsyncMock()
    db.get_session.return_value.__aenter__.return_value = session
    
    # Mock return values for session.get
    mock_page = MagicMock()
    mock_page.url = "http://t.me/url_db"  # Must be string
    session.get.return_value = mock_page
    
    # Define test messages
    msg_plain = AsyncMock()
    msg_plain.text = "Just a random post about weather"
    msg_plain.caption = None
    msg_plain.edit_reply_markup = AsyncMock()

    msg_digest = AsyncMock()
    msg_digest.text = "Weekly digest #дайджест"
    msg_digest.caption = None
    msg_digest.edit_reply_markup = AsyncMock()
    
    # We need to mock DB dependencies for handle_channel_post not to crash/return early
    # It tries to find 'Today' url
    import handlers.channel_nav
    handlers.channel_nav.get_month_page_url = AsyncMock(return_value="http://t.me/url_month")
    handlers.channel_nav.get_next_month_url = AsyncMock(return_value="http://t.me/url_next_month")
    handlers.channel_nav.get_tomorrow_page_url = AsyncMock(return_value="http://t.me/url_tomorrow")
    handlers.channel_nav.get_weekend_page_data = AsyncMock(return_value=("http://t.me/url_weekend", None))
    
    msg_anons = AsyncMock()
    msg_anons.date = MagicMock()
    msg_anons.text = "Cool event happening #анонс"
    msg_anons.caption = None
    
    # Correct mock chain for date
    # message.date -> .astimezone(...) -> .date() -> real date object or mock with .weekday()
    from datetime import date
    mock_date_obj = MagicMock(spec=date)
    mock_date_obj.weekday.return_value = 0
    mock_date_obj.year = 2026
    mock_date_obj.month = 6
    mock_date_obj.day = 15
    
    # When astimezone is called, it returns a datetime-like object. 
    # That object's .date() method should return our mock_date_obj
    mock_dt_local = MagicMock()
    mock_dt_local.date.return_value = mock_date_obj
    
    msg_anons.date.astimezone.return_value = mock_dt_local
    msg_anons.edit_reply_markup = AsyncMock()
    
    with pytest.MonkeyPatch.context() as m:
        mock_main = MagicMock()
        mock_main.get_db.return_value = db
        mock_main.get_bot.return_value = bot
        m.setitem(sys.modules, 'main', mock_main)
        
        # 1. Plain post -> Should NOT have buttons
        await handle_channel_post(msg_plain)
        if msg_plain.edit_reply_markup.called:
            pytest.fail("Should NOT add buttons to plain post")
            
        # 2. Rubric post (e.g. #дайджест) -> Should NOT have buttons
        await handle_channel_post(msg_digest)
        if msg_digest.edit_reply_markup.called:
            pytest.fail("Should NOT add buttons to digest post")

        # 3. Target Post 1: #анонс -> Should HAVE buttons
        await handle_channel_post(msg_anons)
        assert msg_anons.edit_reply_markup.called, "#анонс post MUST get buttons"
        
        # Test #анонсКалининград
        msg_anons_kld = AsyncMock()
        msg_anons_kld.date = MagicMock()
        msg_anons_kld.text = "Event #анонсКалининград"
        msg_anons_kld.caption = None
        
        # Re-use the same date mocks logic
        msg_anons_kld.date.astimezone.return_value = mock_dt_local
        msg_anons_kld.edit_reply_markup = AsyncMock()
        
        await handle_channel_post(msg_anons_kld)
        assert msg_anons_kld.edit_reply_markup.called, "#анонсКалининград post MUST get buttons"

import sys
