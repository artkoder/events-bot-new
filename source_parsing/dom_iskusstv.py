"""Dom Iskusstv special project event extraction.

This module provides functionality to:
1. Extract домискусств.рф special project URLs from text
2. Run Kaggle kernel for parsing those URLs
3. Process parsed events (without Telegraph pages rebuild)

Similar to source_parsing/pyramida.py
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

from aiogram import Bot

from db import Database
from video_announce.kaggle_client import (
    KaggleClient,
    KERNELS_ROOT_PATH,
)
from source_parsing.parser import TheatreEvent, parse_date_raw
from source_parsing.handlers import (
    SourceParsingStats,
    add_new_event_via_queue,
    update_event_ticket_status,
    update_linked_events,
    find_existing_event,
    normalize_location_name,
    EVENT_ADD_DELAY_SECONDS,
    limit_photos_for_source,
    download_images,
)
from poster_media import process_media

logger = logging.getLogger(__name__)

# Kernel folder name
DOM_ISKUSSTV_KERNEL_FOLDER = "ParseDomIskusstv"

# Regex for домискусств.рф special project URLs
# Matches: https://xn--b1admiilxbaki.xn--p1ai/skazka
#          https://домискусств.рф/aladdin
#          http://xn--b1admiilxbaki.xn--p1ai/любойпроект
DOM_ISKUSSTV_URL_PATTERN = re.compile(
    r'https?://(?:xn--b1admiilxbaki\.xn--p1ai|домискусств\.рф)/([a-zA-Zа-яА-ЯёЁ0-9_-]+)',
    re.IGNORECASE,
)

# URL patterns that are NOT special projects (skip these)
SKIP_PATHS = {
    'about-the-theater', 'news1', 'contacts', 'posters', 
    'organization-of-events', 'theater-vacancies', 'antiterrorism',
    'purchases', 'official-documents', 'ticket-refunds', 'visiting-rules',
    'available-environment', 'faq', 'pushkin-cart', 'about-the-halls',
    'theater-people',
}


def extract_dom_iskusstv_urls(text: str) -> list[str]:
    """Extract all домискусств.рф special project URLs from text.
    
    Args:
        text: Text to search for URLs
        
    Returns:
        List of unique Dom Iskusstv special project URLs
    """
    if not text:
        return []
    
    matches = DOM_ISKUSSTV_URL_PATTERN.findall(text)
    
    # Build full URLs and deduplicate
    seen: set[str] = set()
    result: list[str] = []
    
    for path in matches:
        # Skip non-project paths
        if path.lower() in SKIP_PATHS:
            continue
        
        # Clean trailing punctuation
        path = path.rstrip('.,;:!?')
        
        # Normalize to punycode domain
        url = f"https://xn--b1admiilxbaki.xn--p1ai/{path}"
        
        if url not in seen:
            seen.add(url)
            result.append(url)
    
    return result


def parse_price_string(price_str: str) -> tuple[int | None, int | None]:
    """Parse price string into min and max price.
    
    Examples:
        "500 ₽" -> (500, 500)
        "500 - 1000 ₽" -> (500, 1000)
        "от 500 ₽" -> (500, None)
        "Бесплатно" -> (0, 0)
    """
    if not price_str:
        return None, None
        
    s = price_str.lower().strip()
    if not s:
        return None, None
        
    if "бесплатно" in s or "free" in s or "свободный" in s:
        return 0, 0
        
    # Remove currency symbols and spaces
    s = s.replace("₽", "").replace("rub", "").replace("руб", "").replace(" ", "").strip()
    
    # Try range "500-1000"
    if "-" in s:
        parts = s.split("-")
        try:
            min_p = int(re.sub(r'\D', '', parts[0]))
            max_p = int(re.sub(r'\D', '', parts[1]))
            return min_p, max_p
        except (ValueError, IndexError):
            pass
            
    # Try "from 500"
    if s.startswith("от"):
        try:
            val = int(re.sub(r'\D', '', s))
            return val, None
        except ValueError:
            pass
            
    # Try simple number "500"
    try:
        val = int(re.sub(r'\D', '', s))
        if val > 0:
            return val, val
    except ValueError:
        pass
        
    return None, None


async def run_dom_iskusstv_kaggle_kernel(
    urls: list[str],
    timeout_minutes: int = 15,
    poll_interval: int = 20,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> tuple[str, list[str], float]:
    """Run Kaggle kernel for parsing Dom Iskusstv URLs.
    
    Args:
        urls: List of Dom Iskusstv URLs to parse
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    import asyncio
    import os
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / DOM_ISKUSSTV_KERNEL_FOLDER
    
    if not kernel_path.exists():
        logger.warning(
            "dom_iskusstv_kaggle: kernel not found path=%s",
            kernel_path,
        )
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("dom_iskusstv_kaggle: kernel-metadata.json not found")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", f"local/{DOM_ISKUSSTV_KERNEL_FOLDER}")
    except Exception as e:
        logger.error("dom_iskusstv_kaggle: failed to read metadata: %s", e)
        return "error", [], time.time() - start_time
    
    # Validate URLs before injection
    for url in urls:
        if not DOM_ISKUSSTV_URL_PATTERN.search(url):
            logger.warning("dom_iskusstv_kaggle: invalid url=%s", url)
            return "invalid_url", [], 0.0
    
    # Use json.dumps for safe URL injection (prevents code injection)
    urls_json = json.dumps(urls)
    logger.info(
        "dom_iskusstv_kaggle: pushing kernel folder=%s ref=%s urls=%d",
        DOM_ISKUSSTV_KERNEL_FOLDER,
        kernel_ref,
        len(urls),
    )
    
    # Create a unique temp directory for this run to avoid race conditions
    import uuid
    run_id = str(uuid.uuid4())[:8]
    temp_kernel_dir = Path(tempfile.gettempdir()) / f"dom_iskusstv_kernel_{run_id}"
    
    # Copy kernel to temp directory
    import shutil
    shutil.copytree(kernel_path, temp_kernel_dir)
    
    # Write URLs to urls.json for safe argument passing
    (temp_kernel_dir / "urls.json").write_text(json.dumps(urls), encoding="utf-8")
    
    # No need to modify script content anymore
    
    try:
        # Push kernel to Kaggle from temp directory
        try:
            client.push_kernel(kernel_path=temp_kernel_dir)
        except Exception as e:
            logger.error("dom_iskusstv_kaggle: push failed: %s", e)
            return "push_failed", [], time.time() - start_time
        
        # Wait for Kaggle to start
        await asyncio.sleep(10)
        
        # Poll for completion
        max_polls = (timeout_minutes * 60) // poll_interval
        final_status = "timeout"
        
        for poll in range(max_polls):
            await asyncio.sleep(poll_interval)
            
            try:
                status_response = client.get_kernel_status(kernel_ref)
                status = (status_response.get("status", "") or "").upper()
                
                logger.info(
                    "dom_iskusstv_kaggle: poll %d/%d status=%s",
                    poll + 1,
                    max_polls,
                    status,
                )
                
                if status_callback:
                    try:
                        await status_callback(f"Kaggle: {status} (poll {poll + 1}/{max_polls})")
                    except Exception as exc:
                        logger.warning("dom_iskusstv_kaggle: callback failed: %s", exc)
                
                if status == "COMPLETE":
                    final_status = "complete"
                    break
                elif status in ("ERROR", "FAILED", "CANCELLED"):
                    final_status = "failed"
                    failure_msg = status_response.get("failureMessage", "")
                    logger.error(
                        "dom_iskusstv_kaggle: failed status=%s message=%s",
                        status,
                        failure_msg,
                    )
                    break
                elif status in ("QUEUED", "RUNNING"):
                    continue
                    
            except Exception as e:
                logger.warning("dom_iskusstv_kaggle: status check failed: %s", e)
                continue
        
        duration = time.time() - start_time
        
        if final_status != "complete" and final_status != "failed":
            logger.error(
                "dom_iskusstv_kaggle: not complete status=%s duration=%.1fs",
                final_status,
                duration,
            )
            return final_status, [], duration
        
        # Download output files (try to get logs even if failed)
        try:
            output_files = await _download_dom_iskusstv_outputs(client, kernel_ref)
        except Exception as e:
            logger.warning("dom_iskusstv_kaggle: failed to download outputs: %s", e)
            output_files = []
            
        if final_status == "failed":
            logger.error("dom_iskusstv_kaggle: kernel checked as failed, retrieved %d files", len(output_files))
            return "failed", output_files, duration
        
        logger.info(
            "dom_iskusstv_kaggle: complete duration=%.1fs files=%d",
            duration,
            len(output_files),
        )
        
        return final_status, output_files, duration
        
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(temp_kernel_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("dom_iskusstv_kaggle: cleanup failed: %s", e)


async def _download_dom_iskusstv_outputs(client: KaggleClient, kernel_ref: str) -> list[str]:
    """Download kernel output files to unique temp directory."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    output_dir = Path(tempfile.gettempdir()) / f"dom_iskusstv_output_{run_id}"
    output_dir.mkdir(exist_ok=True)
    
    try:
        files = client.download_kernel_output(
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        result_files = [str(output_dir / f) for f in files]
        logger.info("dom_iskusstv_kaggle: downloaded %d files", len(result_files))
        return result_files
        
    except Exception as e:
        logger.error("dom_iskusstv_kaggle: download failed: %s", e)
        return []


def parse_dom_iskusstv_output(file_paths: list[str]) -> list[TheatreEvent]:
    """Parse JSON output from Dom Iskusstv Kaggle kernel.
    
    Args:
        file_paths: List of downloaded file paths
        
    Returns:
        List of TheatreEvent objects
    """
    events: list[TheatreEvent] = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        
        if "dom_iskusstv" not in path.stem.lower():
            continue
        
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.warning("dom_iskusstv_parse: expected list, got %s", type(data))
                continue
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                
                # Get parsed date/time or parse from raw
                parsed_date = item.get("parsed_date")
                parsed_time = item.get("parsed_time")
                
                if not parsed_date:
                    date_raw = (item.get("date_raw") or "").strip()
                    parsed_date, parsed_time = parse_date_raw(date_raw)
                
                # Get photos
                photos: list[str] = []
                photos_data = item.get("photos") or []
                if isinstance(photos_data, list):
                    photos = [p for p in photos_data if isinstance(p, str) and p]
                
                # Parse price
                price_min = item.get("ticket_price_min")
                price_max = item.get("ticket_price_max")
                
                # Determine ticket status
                ticket_status = (item.get("ticket_status") or "unknown").strip()
                if ticket_status == "unknown" and (price_min is not None or price_max is not None):
                    ticket_status = "available"
                
                event = TheatreEvent(
                    title=title,
                    date_raw=(item.get("date_raw") or "").strip(),
                    ticket_status=ticket_status,
                    url=(item.get("url") or "").strip(),
                    photos=photos,
                    description=(item.get("description") or "").strip(),
                    pushkin_card=False,
                    location=(item.get("location") or "Дом искусств").strip(),
                    age_restriction=(item.get("age_restriction") or "").strip(),
                    scene="",
                    source_type="dom_iskusstv",
                    parsed_date=parsed_date,
                    parsed_time=parsed_time,
                    ticket_price_min=price_min,
                    ticket_price_max=price_max,
                )
                events.append(event)
            
            logger.info(
                "dom_iskusstv_parse: parsed file=%s events=%d",
                path.name,
                len(events),
            )
            
        except Exception as e:
            logger.error(
                "dom_iskusstv_parse: failed file=%s error=%s",
                file_path,
                e,
            )
    
    return events


async def process_dom_iskusstv_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    chat_id: int | None = None,
    skip_pages_rebuild: bool = True,
) -> SourceParsingStats:
    """Process events from Dom Iskusstv (without updating month pages).
    
    Similar to process_pyramida_events.
    
    Args:
        db: Database instance
        bot: Telegram bot for notifications
        events: List of events to process
        chat_id: Chat ID for progress messages
        skip_pages_rebuild: If True, don't rebuild Telegraph pages
        
    Returns:
        Processing statistics
    """
    import asyncio
    
    stats = SourceParsingStats(source="dom_iskusstv", total_received=len(events))
    
    for i, event in enumerate(events):
        current_progress = i + 1
        
        if not event.parsed_date:
            logger.warning(
                "dom_iskusstv_process: skipping event without date title=%s",
                event.title,
            )
            stats.failed += 1
            continue
        
        location_name = normalize_location_name(event.location)
        
        # Check for existing event
        existing_id, needs_full_update = await find_existing_event(
            db,
            location_name,
            event.parsed_date,
            event.parsed_time or "00:00",
            event.title,
        )
        
        if existing_id:
            # Update ticket status
            success = await update_event_ticket_status(
                db,
                existing_id,
                event.ticket_status,
                event.url,
            )
            if success:
                stats.ticket_updated += 1
                stats.updated_event_ids.append(existing_id)
            else:
                stats.already_exists += 1
            
            # Update linked events
            await update_linked_events(db, existing_id, location_name, event.title)
        else:
            # Send progress message
            if bot and chat_id:
                try:
                    progress_text = f"🏛 Дом искусств {current_progress}/{len(events)}: {event.title[:40]}"
                    await bot.send_message(chat_id, progress_text)
                except Exception as e:
                    logger.warning("dom_iskusstv_process: failed to send progress: %s", e)
            
            # Prepare images for OCR
            poster_media_list = []
            photos = limit_photos_for_source(event.photos, event.source_type)
            
            if photos:
                try:
                    images = await download_images(photos)
                    if images:
                        poster_media_list, _ = await process_media(
                            images,
                            need_catbox=True,
                            need_ocr=True,
                        )
                except Exception as e:
                    logger.warning("dom_iskusstv_process: ocr failed: %s", e)
            
            # Add new event
            new_id, was_added = await add_new_event_via_queue(
                db,
                bot,
                event,
                current_progress,
                len(events),
                poster_media=poster_media_list,
            )
            
            if new_id:
                if was_added:
                    stats.new_added += 1
                    stats.added_event_ids.append(new_id)
                else:
                    stats.skipped += 1
                # Delay between additions
                await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
            else:
                stats.failed += 1
    
    logger.info(
        "dom_iskusstv_process: complete total=%d added=%d updated=%d failed=%d",
        len(events),
        stats.new_added,
        stats.ticket_updated,
        stats.failed,
    )
    
    return stats
