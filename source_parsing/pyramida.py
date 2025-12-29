"""Pyramida event extraction from VK posts.

This module provides functionality to:
1. Extract pyramida.info/tickets/ URLs from text
2. Run Kaggle kernel for parsing those URLs
3. Process parsed events (without Telegraph pages rebuild)
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
)

logger = logging.getLogger(__name__)

# Kernel folder name
PYRAMIDA_KERNEL_FOLDER = "ParsePyramida"

# Regex for pyramida.info ticket URLs
PYRAMIDA_URL_PATTERN = re.compile(
    r'https?://(?:www\.)?pyramida\.info/tickets/[^\s\)\]\"\'>]+',
    re.IGNORECASE,
)


def extract_pyramida_urls(text: str) -> list[str]:
    """Extract all pyramida.info/tickets/ URLs from text.
    
    Args:
        text: Text to search for URLs
        
    Returns:
        List of unique Pyramida ticket URLs
    """
    if not text:
        return []
    
    matches = PYRAMIDA_URL_PATTERN.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for url in matches:
        # Clean trailing punctuation
        url = url.rstrip('.,;:!?')
        if url not in seen:
            seen.add(url)
            result.append(url)
    
    return result


async def run_pyramida_kaggle_kernel(
    urls: list[str],
    timeout_minutes: int = 15,
    poll_interval: int = 20,
    status_callback: Optional[Callable[[str], Awaitable[None]]] = None,
) -> tuple[str, list[str], float]:
    """Run Kaggle kernel for parsing Pyramida URLs.
    
    Args:
        urls: List of Pyramida URLs to parse
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    import asyncio
    import os
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / PYRAMIDA_KERNEL_FOLDER
    
    if not kernel_path.exists():
        logger.warning(
            "pyramida_kaggle: kernel not found path=%s",
            kernel_path,
        )
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("pyramida_kaggle: kernel-metadata.json not found")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", f"local/{PYRAMIDA_KERNEL_FOLDER}")
    except Exception as e:
        logger.error("pyramida_kaggle: failed to read metadata: %s", e)
        return "error", [], time.time() - start_time
    
    # Set environment variable with URLs for the kernel
    urls_str = ",".join(urls)
    logger.info(
        "pyramida_kaggle: pushing kernel folder=%s ref=%s urls=%d",
        PYRAMIDA_KERNEL_FOLDER,
        kernel_ref,
        len(urls),
    )
    
    # Create a temporary script that sets the environment variable
    # Note: Kaggle kernels receive environment through dataset or script modification
    # For now, we'll modify the script to include URLs directly
    script_path = kernel_path / "parse_pyramida.py"
    original_content = script_path.read_text(encoding="utf-8")
    
    # Inject URLs into the script
    modified_content = original_content.replace(
        'urls_env = os.environ.get("PYRAMIDA_URLS", "")',
        f'urls_env = "{urls_str}"  # Injected by pyramida.py',
    )
    
    try:
        script_path.write_text(modified_content, encoding="utf-8")
        
        # Push kernel to Kaggle
        try:
            client.push_kernel(kernel_path=kernel_path)
        except Exception as e:
            logger.error("pyramida_kaggle: push failed: %s", e)
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
                    "pyramida_kaggle: poll %d/%d status=%s",
                    poll + 1,
                    max_polls,
                    status,
                )
                
                if status_callback:
                    try:
                        await status_callback(f"Kaggle: {status} (poll {poll + 1}/{max_polls})")
                    except Exception as exc:
                        logger.warning("pyramida_kaggle: callback failed: %s", exc)
                
                if status == "COMPLETE":
                    final_status = "complete"
                    break
                elif status in ("ERROR", "FAILED", "CANCELLED"):
                    final_status = "failed"
                    failure_msg = status_response.get("failureMessage", "")
                    logger.error(
                        "pyramida_kaggle: failed status=%s message=%s",
                        status,
                        failure_msg,
                    )
                    break
                elif status in ("QUEUED", "RUNNING"):
                    continue
                    
            except Exception as e:
                logger.warning("pyramida_kaggle: status check failed: %s", e)
                continue
        
        duration = time.time() - start_time
        
        if final_status != "complete":
            logger.error(
                "pyramida_kaggle: not complete status=%s duration=%.1fs",
                final_status,
                duration,
            )
            return final_status, [], duration
        
        # Download output files
        output_files = await _download_pyramida_outputs(client, kernel_ref)
        
        logger.info(
            "pyramida_kaggle: complete duration=%.1fs files=%d",
            duration,
            len(output_files),
        )
        
        return final_status, output_files, duration
        
    finally:
        # Restore original script
        script_path.write_text(original_content, encoding="utf-8")


async def _download_pyramida_outputs(client: KaggleClient, kernel_ref: str) -> list[str]:
    """Download kernel output files."""
    output_dir = Path(tempfile.gettempdir()) / "pyramida_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        files = client.download_kernel_output(
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        result_files = [str(output_dir / f) for f in files]
        logger.info("pyramida_kaggle: downloaded %d files", len(result_files))
        return result_files
        
    except Exception as e:
        logger.error("pyramida_kaggle: download failed: %s", e)
        return []


def parse_pyramida_output(file_paths: list[str]) -> list[TheatreEvent]:
    """Parse JSON output from Pyramida Kaggle kernel.
    
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
        
        if "pyramida" not in path.stem.lower():
            continue
        
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.warning("pyramida_parse: expected list, got %s", type(data))
                continue
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                title = (item.get("title") or "").strip()
                if not title or title == "Заголовок не найден":
                    continue
                
                date_raw = (item.get("date_raw") or "").strip()
                parsed_date, parsed_time = parse_date_raw(date_raw)
                
                # Get photos from image_url
                photos: list[str] = []
                image_url = (item.get("image_url") or "").strip()
                if image_url:
                    photos = [image_url]
                
                event = TheatreEvent(
                    title=title,
                    date_raw=date_raw,
                    ticket_status=(item.get("ticket_status") or "unknown").strip(),
                    url=(item.get("url") or "").strip(),
                    photos=photos,
                    description=(item.get("description") or "").strip(),
                    pushkin_card=False,
                    location=(item.get("location") or "").strip(),
                    age_restriction=(item.get("age_restriction") or "").strip(),
                    scene="",
                    source_type="pyramida",
                    parsed_date=parsed_date,
                    parsed_time=parsed_time,
                )
                events.append(event)
            
            logger.info(
                "pyramida_parse: parsed file=%s events=%d",
                path.name,
                len(events),
            )
            
        except Exception as e:
            logger.error(
                "pyramida_parse: failed file=%s error=%s",
                file_path,
                e,
            )
    
    return events


async def process_pyramida_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    chat_id: int | None = None,
    skip_pages_rebuild: bool = True,
) -> SourceParsingStats:
    """Process events from Pyramida (without updating month pages).
    
    Similar to source_parsing.handlers.process_source_events but:
    - Does NOT trigger pages_rebuild at the end
    - Designed for use within VK review flow
    
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
    from source_parsing.parser import find_existing_event, normalize_location_name
    
    stats = SourceParsingStats(source="pyramida", total_received=len(events))
    
    for i, event in enumerate(events):
        current_progress = i + 1
        
        if not event.parsed_date:
            logger.warning(
                "pyramida_process: skipping event without date title=%s",
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
            from source_parsing.handlers import update_event_ticket_status
            success = await update_event_ticket_status(
                db,
                existing_id,
                event.ticket_status,
                event.url,
            )
            if success:
                stats.ticket_updated += 1
            else:
                stats.already_exists += 1
            
            # Update linked events
            await update_linked_events(db, existing_id, location_name, event.title)
        else:
            # Send progress message
            if bot and chat_id:
                try:
                    progress_text = f"🔮 Pyramida {current_progress}/{len(events)}: {event.title[:40]}"
                    await bot.send_message(chat_id, progress_text)
                except Exception as e:
                    logger.warning("pyramida_process: failed to send progress: %s", e)
            
            # Add new event
            new_id, was_added = await add_new_event_via_queue(
                db,
                bot,
                event,
                current_progress,
                len(events),
            )
            
            if new_id:
                if was_added:
                    stats.new_added += 1
                else:
                    stats.skipped += 1
                # Delay between additions
                await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
            else:
                stats.failed += 1
    
    logger.info(
        "pyramida_process: complete total=%d added=%d updated=%d failed=%d",
        len(events),
        stats.new_added,
        stats.ticket_updated,
        stats.failed,
    )
    
    return stats
