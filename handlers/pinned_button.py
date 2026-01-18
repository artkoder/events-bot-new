"""Pinned message button auto-update handler.

Updates the button on a pinned channel message daily at 18:00:
- Sunday to Wednesday: "📅 {day} {month}" → tomorrow's events page
- Thursday to Saturday: "📅 Выходные" → weekend page
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING

from aiogram import Bot
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

if TYPE_CHECKING:
    from db import Database

logger = logging.getLogger(__name__)

# Russian month names in genitive case
MONTHS_GENITIVE = [
    "", "Января", "Февраля", "Марта", "Апреля", "Мая", "Июня",
    "Июля", "Августа", "Сентября", "Октября", "Ноября", "Декабря"
]


def _get_button_type(weekday: int) -> str:
    """Determine button type based on weekday.
    
    Args:
        weekday: 0=Monday, 6=Sunday
        
    Returns:
        "tomorrow" for Sun-Wed (weekday 6, 0, 1, 2)
        "weekend" for Thu-Sat (weekday 3, 4, 5)
    """
    # Sunday=6, Monday=0, Tuesday=1, Wednesday=2 → tomorrow
    # Thursday=3, Friday=4, Saturday=5 → weekend
    if weekday in (6, 0, 1, 2):  # Sun, Mon, Tue, Wed
        return "tomorrow"
    return "weekend"


def format_day_month(d: date) -> str:
    """Format date as '7 Января' (day + month in genitive)."""
    return f"{d.day} {MONTHS_GENITIVE[d.month]}"


async def get_pinned_button_data(
    db: "Database",
    now: datetime,
) -> tuple[str, str | None, str]:
    """Determine button label and URL for pinned message.
    
    Args:
        db: Database instance
        now: Current datetime (with timezone)
        
    Returns:
        (label, url, button_type) where:
        - label: Button text like "📅 7 Января" or "📅 Выходные"
        - url: Telegraph page URL or None if not found
        - button_type: "tomorrow" or "weekend"
    
    Note:
        Before 18:00: button shows TODAY's events
        After 18:00: button shows TOMORROW's events
        Weekend (Sat/Sun) shows "📅 Выходные"
    """
    from handlers.channel_nav import get_tomorrow_page_url, get_weekend_page_data
    
    # Before 18:00: show today's events, after 18:00: show tomorrow's events
    if now.hour >= 18:
        target_date = now.date() + timedelta(days=1)
    else:
        target_date = now.date()
    
    weekday = target_date.weekday()
    
    # Weekend: Sat (5) or Sun (6)
    if weekday in (5, 6):
        label = "📅 Выходные"
        url, sat_date = await get_weekend_page_data(db, target_date)
        return label, url, "weekend"
    else:
        # Weekday: show specific date
        label = f"📅 {format_day_month(target_date)}"
        url = await get_tomorrow_page_url(db, target_date)
        return label, url, "today"


async def run_3di_new_only(db: "Database", bot: Bot) -> int:
    """Run 3D preview generation for new events only.
    
    This is equivalent to pressing "🆕 Только новые" in /3di menu.
    
    Args:
        db: Database instance
        bot: Bot instance (for notifications if needed)
        
    Returns:
        Number of events queued for generation
    """
    try:
        from preview_3d.handlers import _get_new_events_gap
        
        events = await _get_new_events_gap(db, min_images=1)
        if not events:
            logger.info("run_3di_new_only: no new events to process")
            return 0
            
        logger.info("run_3di_new_only: found %d new events", len(events))
        
        # Note: Full Kaggle rendering is async and takes time.
        # For pinned button update, we just ensure pages exist.
        # The actual 3D previews will be generated separately.
        # Here we just log the count - actual rendering is too slow for scheduled job.
        
        return len(events)
        
    except ImportError:
        logger.warning("run_3di_new_only: preview_3d module not available")
        return 0
    except Exception as e:
        logger.error("run_3di_new_only failed: %s", e, exc_info=True)
        return 0


async def update_pinned_message_button(
    db: "Database",
    bot: Bot,
    chat_id: int | str,
    message_id: int,
) -> bool:
    """Update the button on a pinned channel message.
    
    Uses the cat-weather-new approach:
    1. Forward message to get its current caption/text
    2. Use edit_message_caption with the SAME caption + new reply_markup
    
    This works for any message where bot is channel admin (not just bot's own messages).
    
    Args:
        db: Database instance
        bot: Bot instance
        chat_id: Channel ID or username
        message_id: Message ID to update
        
    Returns:
        True if button was updated, False otherwise
    """
    import os
    
    try:
        # Import LOCAL_TZ from main
        try:
            from main import LOCAL_TZ
        except ImportError:
            LOCAL_TZ = timezone(timedelta(hours=2))  # Fallback to EET
            
        now = datetime.now(LOCAL_TZ)
        
        # Get button data (function handles 18:00 switch internally)
        label, url, button_type = await get_pinned_button_data(db, now)
        
        if not url:
            logger.warning(
                "update_pinned_message_button: no URL for %s button on %s",
                button_type, now.date()
            )
            return False
        
        # Resolve chat info and get ADMIN_USER_ID
        try:
            chat_info = await bot.get_chat(chat_id)
            resolved_chat_id = chat_info.id
            
            # Determine message_id from pinned_message or use provided
            if chat_info.pinned_message:
                actual_message_id = chat_info.pinned_message.message_id
            else:
                actual_message_id = message_id
                
            logger.info(
                "update_pinned_message_button: chat=%s id=%s pinned_msg=%d",
                chat_id, resolved_chat_id, actual_message_id
            )
        except Exception as e:
            logger.error("update_pinned_message_button: get_chat failed: %s", e)
            resolved_chat_id = chat_id
            actual_message_id = message_id
        
        # Get superadmin user_id from database for forwarding
        try:
            rows = await db.exec_driver_sql(
                "SELECT user_id FROM user WHERE is_superadmin = 1 LIMIT 1"
            )
            if not rows:
                logger.error("update_pinned_message_button: no superadmin in database")
                return False
            admin_chat_id = rows[0][0]
            logger.info("update_pinned_message_button: using superadmin user_id=%d", admin_chat_id)
        except Exception as e:
            logger.error("update_pinned_message_button: failed to get superadmin: %s", e)
            return False
        
        # Forward message to admin to get its full content (like cat-weather-new)
        try:
            forwarded = await bot.forward_message(
                chat_id=admin_chat_id,
                from_chat_id=resolved_chat_id,
                message_id=actual_message_id,
            )
            
            # Extract content from forwarded message
            caption = forwarded.caption
            caption_entities = forwarded.caption_entities
            text = forwarded.text
            text_entities = forwarded.entities
            
            logger.info(
                "update_pinned_message_button: forwarded msg=%d, has_caption=%s, has_text=%s, text_len=%s",
                actual_message_id,
                caption is not None,
                text is not None,
                len(text) if text else 0
            )
            
            # Delete the forwarded message
            try:
                await bot.delete_message(chat_id=admin_chat_id, message_id=forwarded.message_id)
            except Exception as del_e:
                logger.warning("update_pinned_message_button: delete forward failed: %s", del_e)
                
        except Exception as e:
            logger.error("update_pinned_message_button: forward failed: %s", e)
            return False
        
        # Build keyboard with single button
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=label, url=url)]
            ]
        )
        
        # Step 1: Try to remove existing buttons first (like /delbutton in cat-weather-new)
        # This is needed because we can't edit buttons set by another bot
        try:
            await bot.edit_message_reply_markup(
                chat_id=resolved_chat_id,
                message_id=actual_message_id,
                reply_markup=None,
            )
            logger.info(
                "update_pinned_message_button: cleared existing buttons chat=%s msg=%d",
                resolved_chat_id, actual_message_id
            )
        except Exception as del_e:
            # Ignore errors - might not have buttons or we might not be able to delete
            logger.debug("update_pinned_message_button: clear buttons failed (ok): %s", del_e)
        
        # Step 2: Add new button
        # Update using appropriate method based on message type (like cat-weather-new)
        if caption is not None:
            # Photo/video with caption → use edit_message_caption
            await bot.edit_message_caption(
                chat_id=resolved_chat_id,
                message_id=actual_message_id,
                caption=caption,
                caption_entities=caption_entities,
                reply_markup=keyboard,
            )
            logger.info(
                "update_pinned_message_button: updated via edit_message_caption chat=%s msg=%d to %s (%s)",
                resolved_chat_id, actual_message_id, label, button_type
            )
        else:
            # Text message or any other → use edit_message_reply_markup (NOT edit_message_text!)
            await bot.edit_message_reply_markup(
                chat_id=resolved_chat_id,
                message_id=actual_message_id,
                reply_markup=keyboard,
            )
            logger.info(
                "update_pinned_message_button: updated via edit_message_reply_markup chat=%s msg=%d to %s (%s)",
                resolved_chat_id, actual_message_id, label, button_type
            )
        
        return True
        
    except Exception as e:
        logger.error(
            "update_pinned_message_button failed: chat=%s msg=%d error=%s",
            chat_id, message_id, e, exc_info=True
        )
        return False


async def pinned_button_scheduler(
    db: "Database",
    bot: Bot,
    run_id: str | None = None,
) -> None:
    """Scheduled job to update pinned message button daily at 18:00.
    
    Called by APScheduler from scheduling.py.
    """
    logger.info("pinned_button_scheduler: start run_id=%s", run_id)
    
    try:
        # Get chat_id and message_id from settings or use defaults
        # Default: @kenigevents channel, message 4
        try:
            from main import get_setting_value
            
            chat_id_setting = await get_setting_value(db, "pinned_channel_id")
            message_id_setting = await get_setting_value(db, "pinned_message_id")
            
            chat_id = chat_id_setting or "@kenigevents"
            message_id = int(message_id_setting) if message_id_setting else 4
        except Exception:
            chat_id = "@kenigevents"
            message_id = 4
        
        # Run 3di for new events (async, non-blocking count only)
        new_count = await run_3di_new_only(db, bot)
        if new_count > 0:
            logger.info(
                "pinned_button_scheduler: %d new events found for 3D preview", 
                new_count
            )
        
        # Update the button
        success = await update_pinned_message_button(db, bot, chat_id, message_id)
        
        if success:
            logger.info("pinned_button_scheduler: done run_id=%s", run_id)
        else:
            logger.warning("pinned_button_scheduler: button update failed run_id=%s", run_id)
            
    except Exception as e:
        logger.error("pinned_button_scheduler failed: %s", e, exc_info=True)
