from datetime import date, timedelta
from typing import List
from sqlalchemy import select
from db import Database
from models import MonthPage

def get_next_month_str(d: date) -> str:
    """Return YYYY-MM for the next month."""
    # Add 32 days to ensure we land in the next month, then format
    next_month_date = d + timedelta(days=32)
    return next_month_date.strftime("%Y-%m")

async def get_month_context_urls(db: Database) -> List[str]:
    """
    Get URLs of Telegraph pages for the current and next month.
    These pages serve as context for deduplicating new events.
    """
    today = date.today()
    current_month_key = today.strftime("%Y-%m")
    next_month_key = get_next_month_str(today)
    
    target_months = [current_month_key, next_month_key]
    
    async with db.get_session() as session:
        # Query MonthPage model
        result = await session.execute(
            select(MonthPage).where(MonthPage.month.in_(target_months))
        )
        pages = result.scalars().all()
        
    return [page.url for page in pages if page.url]
