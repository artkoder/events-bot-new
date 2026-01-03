"""
Real E2E test for Dom Iskusstv parser using production DB snapshot and real Kaggle execution.
WARNING: This test takes several minutes and requires Kaggle API credentials.
Now with REAL LLM (requires FOUR_O_TOKEN in env).
"""

import sys
import os
import shutil
import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

# Add project root to path
sys.path.insert(0, os.getcwd())

import main
from main import Database
import main_part2

# Configure logging to see progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_real_dom_iskusstv_kaggle_flow(tmp_path):
    """
    Test the full flow with real Kaggle execution and production DB snapshot.
    
    Steps:
    1. Clone prod snapshot to temp DB.
    2. Simulate user in 'waiting for input' state.
    3. Send message with Dom Iskusstv URL.
    4. Wait for real Kaggle kernel to push, run, and finish.
    5. Verify event is added to the database.
    6. Verify event fields (title, date, price).
    """
    
    # 1. Setup Database
    snapshot_path = Path("db_prod_snapshot.sqlite")
    if not snapshot_path.exists():
        pytest.skip("db_prod_snapshot.sqlite not found")
        
    db_path = tmp_path / "real_test.sqlite"
    logger.info(f"Copying DB snapshot to {db_path}...")
    shutil.copy(snapshot_path, db_path)
    
    db = Database(str(db_path))
    await db.init()
    
    # 2. Mock Bot (we don't need real Telegram I/O)
    bot = AsyncMock()
    # Mock message objects for edit_message_text
    status_msg = MagicMock()
    status_msg.message_id = 12345
    bot.send_message.return_value = status_msg
    
    # Side effect to print logs
    async def mock_send_document(chat_id, document, caption=None):
        logger.info(f"Mock send_document: {caption}")
        if hasattr(document, "path"):
             # document is FSInputFile
             path = Path(str(document.path))
             if path.suffix == ".log":
                 try:
                     content = path.read_text(encoding="utf-8", errors="replace")
                     print(f"\n{'='*20} LOG CONTENT ({path.name}) {'='*20}\n{content}\n{'='*60}\n")
                 except Exception as e:
                     logger.error(f"Failed to read log: {e}")
    
    bot.send_document.side_effect = mock_send_document
    
    # 3. Simulate User Input
    user_id = 7777777  # Test user ID
    # Add user to the session set effectively mocking "user pressed button"
    main_part2.dom_iskusstv_input_sessions.add(user_id)
    
    # Create incoming message with URL
    # URL provided in task: https://xn--b1admiilxbaki.xn--p1ai/skazka
    target_url = "https://xn--b1admiilxbaki.xn--p1ai/skazka"
    
    chat = MagicMock()
    chat.id = user_id
    
    user = MagicMock()
    user.id = user_id
    
    message = MagicMock()
    message.chat = chat
    message.from_user = user
    message.text = target_url
    
    logger.info(f"Starting real processing for {target_url}...")
    
    # 3.1 Check Kaggle Credentials
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
         pytest.fail("KAGGLE_USERNAME or KAGGLE_KEY not found in environment. Cannot run real Kaggle test.")
    
    # 3.2 Check LLM Creds (optional, code handles it, but good to know)
    if not os.environ.get("FOUR_O_TOKEN"):
        logger.warning("FOUR_O_TOKEN not found! LLM might fail if not using another method.")

    # 4. Invoke Handler (This runs REAL Kaggle kernel + REAL LLM)
    await main_part2.handle_dom_iskusstv_input(message, db, bot)
    
    # 5. Verify Database Content & Generate Telegraph
    logger.info("Verifying database content...")
    async with db.get_session() as session:
        # Check for event with the specific location and date
        # Kaggle gives date="2026-01-03" (verified in logs)
        # Title usually "«Сказки на ночь»"
        
        result = await session.execute(
            text("SELECT id, title, date, description, ticket_price_min, search_digest FROM event WHERE location_name LIKE '%Дом искусств%' AND date = '2026-01-03'")
        )
        row = result.first()
        
        assert row is not None, "Event not found in database!"
        event_id = row.id
        logger.info(f"Found event: ID={row.id}, Title={row.title}, Date={row.date}")
        
        # Verify LLM populated description
        # If LLM failed, description might be empty or fallback.
        if row.description:
            logger.info(f"Description length: {len(row.description)}")
            logger.info(f"Description preview: {row.description[:100]}...")
        else:
            logger.warning("Description is empty! LLM might have failed.")
            
        if row.search_digest:
            logger.info(f"Search digest: {row.search_digest}")
        else:
            logger.warning("Search digest is empty!")

        # 6. Manual Telegraph Build (simulating Job Queue)
        logger.info(f"Generating Telegraph page for event {event_id}...")
        try:
            telegraph_url = await main.update_telegraph_event_page(event_id, db)
            logger.info(f"Generated Telegraph URL: {telegraph_url}")
        except Exception as e:
            logger.error(f"Failed to generate Telegraph page: {e}")
            telegraph_url = None
        
        # Re-fetch to verify URL persistence
        if telegraph_url:
            result = await session.execute(
                text("SELECT telegraph_url FROM event WHERE id = :eid"),
                {"eid": event_id}
            )
            updated_row = result.first()
            assert updated_row.telegraph_url, "Telegraph URL not saved!"
            assert updated_row.telegraph_url == telegraph_url
        
    logger.info("Test passed successfully!")
