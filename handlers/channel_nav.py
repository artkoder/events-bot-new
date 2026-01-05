
import logging
import random
import re
from datetime import datetime, timedelta, date, timezone
from aiogram import Router, types, F, Bot
from aiogram.enums import ContentType
from sqlalchemy import select

from db import Database
from models import Channel, MonthPage, WeekendPage, MonthPagePart, TomorrowPage
from special_pages import create_special_telegraph_page

# Router setup
channel_nav_router = Router()
logger = logging.getLogger(__name__)

# Constants
RUBRIC_MARKERS = [
    "#дайджест",
    "#подборка",
    "#рубрика",
    "#анонс", 
    "#итоги",
    "#конкурс",
    "#ежедневныйанонс",
    "не пропустите сегодня",
    "добавили в анонс",
    "\u200b", # Invisible marker for split daily posts
]

# Regex to find hashtags
HASHTAG_RE = re.compile(r"#\w+")

def is_rubric_post(text: str | None) -> bool:
    """Check if post is a rubric based on hashtags or keywords."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    for marker in RUBRIC_MARKERS:
        if marker in text_lower:
            return True
            
    return False

async def get_month_page_url(db: Database, target_date: date) -> str | None:
    """Get URL for the month page."""
    month_str = target_date.strftime("%Y-%m")
    
    async with db.get_session() as session:
        page = await session.get(MonthPage, month_str)
        if page and page.url:
            return page.url
    return None

async def get_tomorrow_page_url(db: Database, target_date: date) -> str | None:
    """Get or create URL for tomorrow's special page."""
    date_str = target_date.isoformat()
    
    async with db.get_session() as session:
        # Check cache
        cached = await session.get(TomorrowPage, date_str)
        if cached:
            return cached.url
            
    # Generate new page
    # Note: create_special_telegraph_page manages its own db session usually, 
    # but looking at signature it takes 'db'.
    # We need to ensure we don't have transaction conflicts if we reuse session?
    # The signature in special_pages.py is: async def create_special_telegraph_page(db: "Database", ...)
    # So we pass the db instance.
    
    try:
        url, _ = await create_special_telegraph_page(
            db=db,
            start_date=target_date,
            days=1,
            cover_url=None, # No cover for auto-generated daily tomorrow page
            title="Афиша на завтра"
        )
        
        if url:
            # Cache it
            async with db.get_session() as session:
                entry = TomorrowPage(date=date_str, url=url)
                session.add(entry)
                await session.commit()
            return url
            
    except Exception as e:
        logger.error("Failed to generate tomorrow page for %s: %s", date_str, e)
        
    return None


def format_short_date(d: date) -> str:
    """Format date as d.mm (e.g. 3.01 or 15.11)."""
    return f"{d.day}.{d.month:02}"


def format_date_range(start: date, end: date) -> str:
    """Format range as '10-11.01' if same month, else '10.01-01.02'."""
    if start.month == end.month:
        return f"{start.day}-{end.day}.{start.month:02}"
    return f"{format_short_date(start)}-{format_short_date(end)}"

async def get_weekend_page_data(db: Database, target_date: date) -> tuple[str | None, date | None]:
    """Get URL and Friday date for weekend page."""
    # Logic: Find Friday of the week
    weekday = target_date.weekday() # 0=Mon, 6=Sun
    
    if weekday <= 2:
        friday = target_date + timedelta(days=(4 - weekday))
    else:
        friday = target_date - timedelta(days=(weekday - 4))
        
    start_str = friday.isoformat()
    
    async with db.get_session() as session:
        page = await session.get(WeekendPage, start_str)
        if page and page.url:
            return page.url, friday
            
    # Fallback to MonthPage if no WeekendPage
    url = await get_month_page_url(db, friday)
    return url, friday

async def get_next_month_url(db: Database, current_date: date) -> str | None:
    """Get URL for next month."""
    # First day of next month
    if current_date.month == 12:
        next_month = date(current_date.year + 1, 1, 1)
    else:
        next_month = date(current_date.year, current_date.month + 1, 1)
        
    month_str = next_month.strftime("%Y-%m")
    async with db.get_session() as session:
        page = await session.get(MonthPage, month_str)
        if page and page.url:
            return page.url
    return None

MONTH_NAMES_GENITIVE = [
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря"
]

FULL_MONTH_NAMES = [
    "", "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
    "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
]

@channel_nav_router.channel_post()
async def handle_channel_post(message: types.Message):
    """Handle new posts in channel."""
    # Get db and bot from main module
    import main
    db = main.get_db()
    bot = main.get_bot()
    if not db or not bot:
        logger.warning("channel_nav: db or bot not initialized")
        return
    
    # 1. Filter: Check if bot is admin (implicit if we can edit, but good to check context?)
    # Actually, we just try to edit.
    
    # 2. Filter: Commands (skip /commands in channel)
    text = message.text or message.caption or ""
    if text.startswith("/"):
        logger.debug("channel_nav: skipping command post %s", message.message_id)
        return
    
    # 3. Filter: Rubrics
    if is_rubric_post(text):
        logger.info("channel_nav: skipping rubric post %s", message.message_id)
        return

    # 3. Filter: Forwarded messages? User didn't strictly specify, but usually forwards are not "admin content"
    # But admin might forward. User said "admin writes or scheduled". 
    # Valid admin post can be a forward. We keep it.
    
    # 4. Generate Buttons
    # "Today" button
    
    # Calculate dates
    # Assuming LOCAL_TZ is available in main scope or we import
    # For now assume UTC or fix imports. Importing LOCAL_TZ is tricky if circular. 
    # Use timezone.utc + 2 (EET) approximation or pass it?
    # Let's use message date
    
    # Message date is UTC timestamp
    post_date = message.date.astimezone(timezone(timedelta(hours=2))) # Kaliningrad/EET approx
    today = post_date.date()
    tomorrow = today + timedelta(days=1)
    
    buttons = []
    
    # Button 1: Сегодня
    today_url = await get_month_page_url(db, today)
    if today_url:
        buttons.append(
            types.InlineKeyboardButton(text="📅 Сегодня", url=today_url)
        )
    
    # Button 2: Random Selection using random.choice
    # Equal probability: Tomorrow, Weekend, Next Month
    
    choices = ["tomorrow", "weekend", "next_month"]
    selection = random.choice(choices)
    second_btn = None
    logger.info("channel_nav: selection=%s today=%s weekday=%d", selection, today, today.weekday())
    
    if selection == "tomorrow":
        # Tomorrow
        tmr_url = await get_tomorrow_page_url(db, tomorrow)
        if tmr_url:
            second_btn = types.InlineKeyboardButton(text="📅 Завтра", url=tmr_url)
        logger.info("channel_nav: selected Tomorrow, url=%s", tmr_url)
    elif selection == "weekend":
        # Weekend
        # WeekendPage is keyed by SATURDAY date in main_part2.py
        # Calculate upcoming Saturday
        weekday = today.weekday()
        if weekday <= 5:  # Mon-Sat
            days_to_sat = (5 - weekday) % 7
            if days_to_sat == 0 and weekday == 5:
                days_to_sat = 0  # Today is Saturday
            sat = today + timedelta(days=days_to_sat)
        else:  # Sunday
            sat = today + timedelta(days=6)  # Next Saturday
        
        start_str = sat.isoformat()
        logger.info("channel_nav: Weekend lookup for sat=%s key=%s", sat, start_str)
        
        async with db.get_session() as session:
            page = await session.get(WeekendPage, start_str)
            wk_url = page.url if page else None
        
        logger.info("channel_nav: Weekend page found=%s url=%s", page is not None, wk_url)
        
        if not wk_url:
            # Fallback to month page for that Saturday
            wk_url = await get_month_page_url(db, sat)
            logger.info("channel_nav: Weekend fallback to month url=%s", wk_url)
        
        if wk_url:
            second_btn = types.InlineKeyboardButton(text="📅 Выходные", url=wk_url)
    else:
        # Next Month (selection == "next_month")
        if today.month == 12:
            next_month_date = date(today.year + 1, 1, 1)
        else:
            next_month_date = date(today.year, today.month + 1, 1)
            
        nm_url = await get_next_month_url(db, today)
        nm_name = FULL_MONTH_NAMES[next_month_date.month]
        if nm_url:
            second_btn = types.InlineKeyboardButton(text=f"📅 {nm_name}", url=nm_url)
            
    # Fallback logic if random selection yielded nothing
    if not second_btn:
        # Try Tomorrow
        tmr_url = await get_tomorrow_page_url(db, tomorrow)
        if tmr_url:
            second_btn = types.InlineKeyboardButton(text="📅 Завтра", url=tmr_url)
    
    if second_btn:
        buttons.append(second_btn)
        
    if not buttons:
        return

    # 5. Edit Message to add markup
    try:
        # We need to construct InlineKeyboardMarkup
        keyboard = types.InlineKeyboardMarkup(inline_keyboard=[buttons])
        await message.edit_reply_markup(reply_markup=keyboard)
        logger.info("channel_nav: added buttons to %s", message.message_id)
    except Exception as e:
        logger.error("channel_nav: failed to add buttons: %s", e)
