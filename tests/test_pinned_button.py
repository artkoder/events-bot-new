"""Tests for handlers/pinned_button.py."""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestGetButtonType:
    """Tests for _get_button_type function."""

    def test_sunday_returns_tomorrow(self):
        from handlers.pinned_button import _get_button_type
        # Sunday = weekday 6
        assert _get_button_type(6) == "tomorrow"

    def test_monday_returns_tomorrow(self):
        from handlers.pinned_button import _get_button_type
        # Monday = weekday 0
        assert _get_button_type(0) == "tomorrow"

    def test_tuesday_returns_tomorrow(self):
        from handlers.pinned_button import _get_button_type
        # Tuesday = weekday 1
        assert _get_button_type(1) == "tomorrow"

    def test_wednesday_returns_tomorrow(self):
        from handlers.pinned_button import _get_button_type
        # Wednesday = weekday 2
        assert _get_button_type(2) == "tomorrow"

    def test_thursday_returns_weekend(self):
        from handlers.pinned_button import _get_button_type
        # Thursday = weekday 3
        assert _get_button_type(3) == "weekend"

    def test_friday_returns_weekend(self):
        from handlers.pinned_button import _get_button_type
        # Friday = weekday 4
        assert _get_button_type(4) == "weekend"

    def test_saturday_returns_weekend(self):
        from handlers.pinned_button import _get_button_type
        # Saturday = weekday 5
        assert _get_button_type(5) == "weekend"


class TestFormatDayMonth:
    """Tests for format_day_month function."""

    def test_january_format(self):
        from handlers.pinned_button import format_day_month
        d = date(2026, 1, 7)
        assert format_day_month(d) == "7 Января"

    def test_february_format(self):
        from handlers.pinned_button import format_day_month
        d = date(2026, 2, 14)
        assert format_day_month(d) == "14 Февраля"

    def test_december_format(self):
        from handlers.pinned_button import format_day_month
        d = date(2026, 12, 31)
        assert format_day_month(d) == "31 Декабря"


class TestGetPinnedButtonData:
    """Tests for get_pinned_button_data function."""

    @pytest.mark.asyncio
    async def test_sunday_returns_tomorrow_type(self):
        from handlers.pinned_button import get_pinned_button_data
        
        db = MagicMock()
        
        # Patch where the function is used (inside get_pinned_button_data)
        with patch("handlers.channel_nav.get_weekend_page_data", new_callable=AsyncMock) as mock_weekend:
            mock_weekend.return_value = ("https://telegra.ph/Weekend-01", date(2026, 1, 17))
            
            # Sunday January 18, 2026 at 10:00 (before 18:00 switch)
            # Sunday is weekend, should show "Выходные"
            now = datetime(2026, 1, 18, 10, 0)  # Sunday 10:00
            label, url, button_type = await get_pinned_button_data(db, now)
            
            assert button_type == "weekend"
            assert label == "📅 Выходные"
            assert url == "https://telegra.ph/Weekend-01"
            mock_weekend.assert_called_once()

    @pytest.mark.asyncio
    async def test_saturday_returns_weekend_type(self):
        from handlers.pinned_button import get_pinned_button_data
        
        db = MagicMock()
        
        with patch("handlers.channel_nav.get_weekend_page_data", new_callable=AsyncMock) as mock_weekend:
            mock_weekend.return_value = ("https://telegra.ph/Weekend-01", date(2026, 1, 17))
            
            # Saturday January 17, 2026 at 10:00 (before 18:00 switch)
            now = datetime(2026, 1, 17, 10, 0)  # Saturday 10:00
            label, url, button_type = await get_pinned_button_data(db, now)
            
            assert button_type == "weekend"
            assert label == "📅 Выходные"
            assert url == "https://telegra.ph/Weekend-01"
            mock_weekend.assert_called_once()

    @pytest.mark.asyncio
    async def test_wednesday_returns_today_type(self):
        from handlers.pinned_button import get_pinned_button_data
        
        db = MagicMock()
        
        with patch("handlers.channel_nav.get_tomorrow_page_url", new_callable=AsyncMock) as mock_tomorrow:
            mock_tomorrow.return_value = "https://telegra.ph/Test-02"
            
            # Wednesday January 14, 2026 at 10:00 (before 18:00 switch)
            # Weekday: show today's date
            now = datetime(2026, 1, 14, 10, 0)  # Wednesday 10:00
            label, url, button_type = await get_pinned_button_data(db, now)
            
            assert button_type == "today"
            # Shows Wednesday's date (today)
            assert label == "📅 14 Января"
            assert url == "https://telegra.ph/Test-02"


class TestUpdatePinnedMessageButton:
    """Tests for update_pinned_message_button function."""

    @pytest.mark.asyncio
    async def test_updates_button_successfully(self):
        from handlers.pinned_button import update_pinned_message_button
        
        db = MagicMock()
        bot = AsyncMock()
        
        # Mock pinned message info from get_chat
        pinned_msg = MagicMock()
        pinned_msg.message_id = 4
        
        chat_info = MagicMock()
        chat_info.id = -1001234567890
        chat_info.title = "Test Channel"
        chat_info.pinned_message = pinned_msg
        bot.get_chat = AsyncMock(return_value=chat_info)
        
        # Mock forwarded message with caption
        forwarded_msg = MagicMock()
        forwarded_msg.message_id = 100
        forwarded_msg.caption = "Test caption"
        forwarded_msg.caption_entities = None
        forwarded_msg.text = None
        forwarded_msg.entities = None
        bot.forward_message = AsyncMock(return_value=forwarded_msg)
        bot.delete_message = AsyncMock()
        
        # Mock db.exec_driver_sql to return superadmin user_id
        db.exec_driver_sql = AsyncMock(return_value=[(123456,)])
        
        # Patch get_pinned_button_data to control return value
        async def mock_get_data(db, today):
            return ("📅 19 Января", "https://telegra.ph/Test-01", "tomorrow")
        
        with patch("handlers.pinned_button.get_pinned_button_data", side_effect=mock_get_data):
            result = await update_pinned_message_button(db, bot, "@kenigevents", 4)
            
            assert result is True
            bot.edit_message_caption.assert_called_once()
            call_args = bot.edit_message_caption.call_args
            assert call_args.kwargs["chat_id"] == -1001234567890
            assert call_args.kwargs["message_id"] == 4
            assert call_args.kwargs["caption"] == "Test caption"

    @pytest.mark.asyncio
    async def test_returns_false_when_no_url(self):
        from handlers.pinned_button import update_pinned_message_button
        
        db = MagicMock()
        bot = AsyncMock()
        
        # Patch get_pinned_button_data to return None url
        async def mock_get_data(db, today):
            return ("📅 19 Января", None, "tomorrow")
        
        with patch("handlers.pinned_button.get_pinned_button_data", side_effect=mock_get_data):
            result = await update_pinned_message_button(db, bot, "@kenigevents", 4)
            
            assert result is False
            bot.edit_message_caption.assert_not_called()


class TestRun3diNewOnly:
    """Tests for run_3di_new_only function."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_new_events(self):
        from handlers.pinned_button import run_3di_new_only
        
        db = MagicMock()
        bot = AsyncMock()
        
        with patch("preview_3d.handlers._get_new_events_gap", new_callable=AsyncMock) as mock_gap:
            mock_gap.return_value = []
            
            count = await run_3di_new_only(db, bot)
            
            assert count == 0

    @pytest.mark.asyncio
    async def test_returns_event_count(self):
        from handlers.pinned_button import run_3di_new_only
        
        db = MagicMock()
        bot = AsyncMock()
        
        with patch("preview_3d.handlers._get_new_events_gap", new_callable=AsyncMock) as mock_gap:
            # Simulate 5 events found
            mock_gap.return_value = [MagicMock() for _ in range(5)]
            
            count = await run_3di_new_only(db, bot)
            
            assert count == 5

