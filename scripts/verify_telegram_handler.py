import asyncio
import json
import logging
import tempfile
import sys
import os
from pathlib import Path
from dataclasses import dataclass

sys.path.append(os.getcwd())

from db import Database
from models import Event, create_all
from source_parsing.telegram.handlers import process_telegram_results
from google_ai.client import UsageInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

# Mock AI Client
class MockAIClient:
    async def generate_content_async(self, model, prompt, max_output_tokens=None):
        logger.info(f"Mock AI called with prompt: {prompt[:50]}...")
        return "Rewritten Event Description by Gemma (Mock)", UsageInfo()

# Mock Bot
class MockBot:
    async def send_message(self, chat_id, text, parse_mode=None):
        logger.info(f"Mock Bot sending to {chat_id}: \n{text}")

    async def send_document(self, chat_id, document, caption=None):
        logger.info(f"Mock Bot sending document to {chat_id}: {document} (Caption: {caption})")

async def run_verification():
    # Use file-based DB to avoid "unable to open database file" in docker/concurrency edge cases
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tmp_db:
        db_path = tmp_db.name
    
    db = Database(db_path)
    import sqlmodel
    async with db.engine.begin() as conn:
        await conn.run_sync(sqlmodel.SQLModel.metadata.create_all)
    
    # Create sample results
    results = [
        {
            "status": "new",
            "data": {
                "title": "Test Event 1",
                "description": "Original description of test event.",
                "date": "2025-05-20",
                "time": "19:00",
                "location": "Test Venue",
                "price": "500 rub",
                "is_free": False,
                "link": "https://t.me/c/123/1"
            }
        },
        {
            "status": "duplicate",
            "data": {
                "title": "Duplicate Event",
                "date": "2025-05-20"
            }
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(results, tmp)
        tmp_path = tmp.name
        
    try:
        logger.info("Running process_telegram_results...")
        await process_telegram_results(
            results_path=tmp_path,
            db=db,
            bot=MockBot(),
            ai_client=MockAIClient()
        )
        
        # Verify DB
        async with db.get_session() as session:
            events = (await session.execute(select(Event))).scalars().all()
            logger.info(f"Events in DB: {len(events)}")
            for e in events:
                logger.info(f"Event: {e.title}, Desc: {e.description}, Source: {e.source_text}")
                assert e.title == "Test Event 1"
                assert "Rewritten" in e.description
                assert e.source_text == "Original description of test event."
                
        logger.info("Verification PASSED!")
        
    finally:
        Path(tmp_path).unlink()

if __name__ == "__main__":
    from sqlalchemy import select
    asyncio.run(run_verification())
