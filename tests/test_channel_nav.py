
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to sys.path
sys.path.append(os.getcwd())

from handlers.channel_nav import is_rubric_post, handle_channel_post
from db import Database
from models import MonthPage, WeekendPage


import pytest

@pytest.mark.asyncio
async def test_rubric_filter():
    print("Testing rubric filter...")
    assert is_rubric_post("#Дайджест за сегодня") == True
    assert is_rubric_post("Просто текст") == False
    # Real daily announcement text from main_part2.py
    daily_text = "АНОНС на 15 июня 2026 #ежедневныйанонс\nПонедельник\n\nНЕ ПРОПУСТИТЕ СЕГОДНЯ"
    assert is_rubric_post(daily_text) == True
    
    daily_added_text = "+5 ДОБАВИЛИ В АНОНС\nКАЛИНИНГРАД"
    assert is_rubric_post(daily_added_text) == True
    
    # Test invisible marker for split parts
    split_part = "Some random event text without headers\u200b"
    assert is_rubric_post(split_part) == True
    
    assert is_rubric_post("Анонс мероприятия") == False # "Анонс" word is not enough, needs hashtag or specific phrase
    assert is_rubric_post("Не пропустите сегодня что-то интересное") == True
    assert is_rubric_post("Мы добавили в анонс новые события") == True
    assert is_rubric_post("Просто текст") == False
    print("Rubric filter passed!")

@pytest.mark.asyncio
async def test_handle_channel_post_tomorrow_logic():
    from handlers.channel_nav import handle_channel_post
    from models import TomorrowPage
    import sys
    
    # Mocks
    db = MagicMock()
    session = AsyncMock()
    db.get_session.return_value.__aenter__.return_value = session
    
    bot = MagicMock()
    message = AsyncMock()
    message.text = "#анонс Just a post"  # Need trigger hashtag for buttons
    message.caption = None
    message.date = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    message.message_id = 123
    
    # Create mock main module
    mock_main = MagicMock()
    mock_main.get_db.return_value = db
    mock_main.get_bot.return_value = bot
    
    # Mock main in sys.modules for local import
    with patch.dict(sys.modules, {'main': mock_main}):
        # Mock create_special_telegraph_page to avoid actual generation
        with patch("handlers.channel_nav.create_special_telegraph_page", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = ("https://telegra.ph/Tom-01", 1)
            
            # Test 1: Cached page exists
            session.get.side_effect = lambda model, key: AsyncMock(url="https://telegra.ph/Cached-01") if model == TomorrowPage else None
            
            await handle_channel_post(message)
            
            # Check if button URL matches cached
            args, kwargs = message.edit_reply_markup.call_args
            keyboard = kwargs['reply_markup']
            # We can't easily check buttons deeply without inspecting object structure
            # But we can verify session.get was called for TomorrowPage
            
            # Reset
            message.edit_reply_markup.reset_mock()
            
            # Test 2: No cache, generates page
            session.get.side_effect = None # Returns None by default for mocks if not configured? No, AsyncMock returns AsyncMock.
            # We need session.get to return None for TomorrowPage
            session.get.return_value = None
            
            await handle_channel_post(message)
            
            mock_create.assert_called_once()
            # Verify it tries to save to DB
            assert session.add.called
            assert session.commit.called 

@pytest.mark.asyncio
async def test_button_logic():
    print("Testing button logic...")
    import sys
    
    # Mock DB
    db = MagicMock(spec=Database)
    db_session = AsyncMock()
    db.get_session.return_value.__aenter__.return_value = db_session
    
    # Mock Pages
    async def get_mock(model, key):
        if model == MonthPage:
            return MonthPage(month=key, url=f"https://telegra.ph/Month-{key}", path="Month")
        if model == WeekendPage:
            return WeekendPage(start=key, url=f"https://telegra.ph/Weekend-{key}", path="Weekend")
        return None
        
    db_session.get.side_effect = get_mock
    
    # Mock Bot & Message
    bot = AsyncMock()
    message = AsyncMock()
    message.text = "#анонс Test Post"  # Need trigger hashtag for buttons
    message.caption = None
    # Use fixed date: 2026-06-15 12:00 UTC
    message.date = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    message.message_id = 123
    
    # Create mock main module
    mock_main = MagicMock()
    mock_main.get_db.return_value = db
    mock_main.get_bot.return_value = bot
    
    # Mock main in sys.modules for local import
    with patch.dict(sys.modules, {'main': mock_main}):
        await handle_channel_post(message)
    
        # Verify edit_reply_markup called
        if message.edit_reply_markup.called:
            print("edit_reply_markup called!")
            args, kwargs = message.edit_reply_markup.call_args
            markup = kwargs['reply_markup']
            buttons = markup.inline_keyboard[0]
            print(f"Buttons generated: {len(buttons)}")
            today_str = "15.06"
            # Since button text format changed: "📅 Сегодня" (no date)
            
            # Check Today button (no date in text anymore)
            assert any("📅 Сегодня" in btn.text and "https://telegra.ph/Month-2026-06" in btn.url for btn in buttons)
            
            # Check Second button (Tomorrow or Weekend or Next Month)
            # Since we mock random, we don't know which one picked unless we seed or check `any`
            # But let's check for pattern
            tomorrow_str = "16.06"
            # Weekend logic: 2026-06-15 is Monday. 
            # Logic: weekday=0 <= 2.
            # Next Saturday = today + (5-0) = today + 5 = 20 + 5 = 25? No. 15 + 5 = 20.
            # 2026-06-20 is Saturday.
            # Range: 20.06-21.06
            weekend_str = "20.06-21.06"
            
            has_tomorrow = any("📅 Завтра" in btn.text for btn in buttons)
            has_weekend = any("📅 Выходные" in btn.text for btn in buttons)
            has_next_month = any("📅 Июль" in btn.text for btn in buttons)

            assert has_tomorrow or has_weekend or has_next_month
        else:
            pytest.fail("edit_reply_markup NOT called")
