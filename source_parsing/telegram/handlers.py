import json
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from db import Database
from models import Event, utc_now

logger = logging.getLogger(__name__)

# Admin Chat ID for reporting (fallback if not in env)
ADMIN_CHAT_ID = -1001606273463  # Example ID, should be loaded from config

async def process_telegram_results(
    results_path: str | Path, 
    db: Database,
    bot=None,
    ai_client=None # Kept for signature compatibility but unused
):
    """
    Process the results JSON from Kaggle Telegram Monitor.
    
    1. Parse JSON.
    2. Feed each new event candidate into the Standard Event Pipeline (`add_events_from_text`).
       This ensures the event is:
       - Parsed by GPT-4o (or configured LLM).
       - Enriched with topics.
       - Checked for duplicates (internally).
       - Saved to DB.
    3. Report to Admin.
    """
    # Import locally to avoid circular dependency with main.py
    # main.py likely imports this module (or similar)
    try:
        from main import add_events_from_text
    except ImportError:
        logger.error("Could not import add_events_from_text from main. Ensure circular imports are handled.")
        return

    path = Path(results_path)
    if not path.exists():
        logger.error(f"Results file not found: {path}")
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse results JSON: {e}")
        return

    new_events_count = 0
    duplicates_count = 0
    errors_count = 0
    added_events_titles = []

    # 'data' is a list of items from Kaggle script
    # Each item: { "status": "new"|"duplicate", "data": { "description": "...", "link": "...", ... } }
    
    for item in data:
        status = item.get("status")
        
        if status == "duplicate":
            duplicates_count += 1
            if item.get("data", {}).get("link"):
                 # Optional: Log duplicate links?
                 pass
            continue
            
        if status == "new":
            item_data = item.get("data", {})
            raw_text = item_data.get("description", "") or item_data.get("raw_text", "")
            if not raw_text:
                continue
                
            source_link = item_data.get("link")
            
            # Prepare text for the standard pipeline
            # We might want to prepend "Title: ..." if available to help the parser
            title = item_data.get("title")
            input_text = raw_text
            if title:
                input_text = f"{title}\n\n{raw_text}"
            
            try:
                # Call Standard Process
                # ensure we pass the DB and text
                # add_events_from_text returns AddEventsResult (list of tuples)
                results = await add_events_from_text(
                    db=db,
                    text=input_text,
                    source_link=source_link,
                    bot=bot,
                    display_source=True
                )
                
                # Analyze results
                for saved, added, lines, stat in results:
                    if saved and added:
                        new_events_count += 1
                        added_events_titles.append(saved.title)
                    elif saved and not added:
                        # Existing event updated?
                        duplicates_count += 1 # Or count as update
                    else:
                        # Missing fields or error
                        pass
                        
            except Exception as e:
                logger.error(f"Standard pipeline failed for item {title}: {e}")
                errors_count += 1
    
    # Report results
    report = (
        f"🕵️‍♂️ <b>Telegram Monitor Report</b>\n\n"
        f"🆕 Processed: {new_events_count}\n"
        f"♻️ Duplicates/Updated: {duplicates_count}\n"
        f"❌ Errors: {errors_count}\n"
    )
    
    if added_events_titles:
        report += "\n<b>Added:</b>\n"
        for t in added_events_titles[:10]:
             report += f"- {t}\n"
        if len(added_events_titles) > 10:
            report += f"... and {len(added_events_titles) - 10} more."

    logger.info(report)
    
    if bot:
        target_chat = ADMIN_CHAT_ID 
        try:
            await bot.send_message(target_chat, report, parse_mode="HTML")
            
            # Send raw file for debug
            from aiogram.types import FSInputFile
            input_file = FSInputFile(path)
            await bot.send_document(
                chat_id=target_chat,
                document=input_file,
                caption="📄 Raw Kaggle Results"
            )
        except Exception as e:
            logger.warning(f"Failed to send admin report: {e}")
