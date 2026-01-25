import asyncio
import logging
import sys
import os
from sqlalchemy import select
from datetime import date

sys.path.append(os.getcwd())

from db import Database
from models import MonthPage, create_all
from source_parsing.telegram.deduplication import get_month_context_urls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier-dedup")

async def run_verification():
    db = Database("sqlite:///:memory:") 
    create_all(db.engine)
    
    # Populate DB with Mock Pages
    async with db.get_session() as session:
        today = date.today()
        current_month = today.strftime("%Y-%m")
        # next month string logic...
        # Just insert a few
        session.add(MonthPage(month=current_month, url="https://telegra.ph/Current-Month", path="path1"))
        session.add(MonthPage(month="2099-01", url="https://telegra.ph/Far-Future", path="path2"))
        await session.commit()
    
    urls = await get_month_context_urls(db)
    logger.info(f"URLs found: {urls}")
    
    assert "https://telegra.ph/Current-Month" in urls
    assert "https://telegra.ph/Far-Future" not in urls # Should filter by current/next month
    
    logger.info("Deduplication Context Verification PASSED!")

if __name__ == "__main__":
    asyncio.run(run_verification())
