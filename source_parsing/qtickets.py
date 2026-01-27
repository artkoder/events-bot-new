"""Qtickets source parsing handler.

This module handles running the Qtickets Kaggle kernel and processing
the parsed events, following the same pattern as philharmonia.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional
from datetime import datetime

from aiogram import Bot

from db import Database
from video_announce.kaggle_client import (
    KaggleClient,
    KERNELS_ROOT_PATH,
)
from kaggle_registry import register_job, remove_job
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
QTICKETS_KERNEL_FOLDER = "ParseQtickets"


async def run_qtickets_kaggle_kernel(
    timeout_minutes: int = 20,
    poll_interval: int = 30,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[str, list[str], float]:
    """Run Kaggle kernel for parsing Qtickets events.
    
    Args:
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        status_callback: Optional async callback for status updates
        
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    import asyncio
    import uuid
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / QTICKETS_KERNEL_FOLDER
    kernel_ref = "zigomaro/parse-qtickets"  # Default
    registered = False

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.warning("qtickets_kaggle: status callback failed phase=%s", phase)
    
    if not kernel_path.exists():
        logger.warning(
            "qtickets_kaggle: kernel not found path=%s",
            kernel_path,
        )
        await _notify("not_found")
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("qtickets_kaggle: kernel-metadata.json not found")
        await _notify("metadata_missing")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", kernel_ref)
    except Exception as e:
        logger.error("qtickets_kaggle: failed to read metadata: %s", e)
        await _notify("metadata_error")
        return "error", [], time.time() - start_time
    
    logger.info(
        "qtickets_kaggle: pushing kernel folder=%s ref=%s",
        QTICKETS_KERNEL_FOLDER,
        kernel_ref,
    )
    
    await _notify("prepare")

    # Create a unique temp directory for this run
    run_id = str(uuid.uuid4())[:8]
    temp_kernel_dir = Path(tempfile.gettempdir()) / f"qtickets_kernel_{run_id}"
    
    # Copy kernel to temp directory
    shutil.copytree(kernel_path, temp_kernel_dir)
    
    try:
        # Push kernel to Kaggle
        try:
            client.push_kernel(kernel_path=temp_kernel_dir)
        except Exception as e:
            logger.error("qtickets_kaggle: push failed: %s", e)
            await _notify("push_failed")
            return "push_failed", [], time.time() - start_time
        
        await _notify("pushed")
        try:
            await register_job(
                "parse_qtickets",
                kernel_ref,
                meta={"kernel_folder": QTICKETS_KERNEL_FOLDER, "pid": os.getpid()},
            )
            registered = True
        except Exception:
            logger.warning("qtickets_kaggle: failed to register recovery job", exc_info=True)

        # Wait for Kaggle to start
        await asyncio.sleep(10)
        
        # Poll for completion
        max_polls = (timeout_minutes * 60) // poll_interval
        final_status = "timeout"
        
        last_status: dict | None = None

        for poll in range(max_polls):
            await asyncio.sleep(poll_interval)
            
            try:
                status_response = client.get_kernel_status(kernel_ref)
                status = (status_response.get("status", "") or "").upper()
                last_status = status_response
                
                logger.info(
                    "qtickets_kaggle: poll %d/%d status=%s",
                    poll + 1,
                    max_polls,
                    status,
                )
                
                await _notify("poll", status_response)
                
                if status == "COMPLETE":
                    final_status = "complete"
                    await _notify("complete", status_response)
                    break
                elif status in ("ERROR", "FAILED", "CANCELLED"):
                    final_status = "failed"
                    failure_msg = status_response.get("failureMessage", "")
                    logger.error(
                        "qtickets_kaggle: failed status=%s message=%s",
                        status,
                        failure_msg,
                    )
                    await _notify("failed", status_response)
                    break
                elif status in ("QUEUED", "RUNNING"):
                    continue
                    
            except Exception as e:
                logger.warning("qtickets_kaggle: status check failed: %s", e)
                continue

        
        duration = time.time() - start_time
        
        if final_status != "complete" and final_status != "failed":
            logger.error(
                "qtickets_kaggle: not complete status=%s duration=%.1fs",
                final_status,
                duration,
            )
            if registered:
                await remove_job("parse_qtickets", kernel_ref)
            return final_status, [], duration
        
        # Download output files
        try:
            output_files = await _download_qtickets_outputs(client, kernel_ref)
        except Exception as e:
            logger.warning("qtickets_kaggle: failed to download outputs: %s", e)
            output_files = []
            
        if final_status == "failed":
            logger.error("qtickets_kaggle: kernel checked as failed, retrieved %d files", len(output_files))
            if registered:
                await remove_job("parse_qtickets", kernel_ref)
            return "failed", output_files, duration
        
        logger.info(
            "qtickets_kaggle: complete duration=%.1fs files=%d",
            duration,
            len(output_files),
        )
        if registered:
            await remove_job("parse_qtickets", kernel_ref)
        return final_status, output_files, duration
        
    finally:
        try:
            shutil.rmtree(temp_kernel_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("qtickets_kaggle: cleanup failed: %s", e)


async def _download_qtickets_outputs(client: KaggleClient, kernel_ref: str) -> list[str]:
    """Download kernel output files to unique temp directory."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    output_dir = Path(tempfile.gettempdir()) / f"qtickets_output_{run_id}"
    output_dir.mkdir(exist_ok=True)
    
    try:
        files = client.download_kernel_output(
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        result_files = [str(output_dir / f) for f in files]
        logger.info("qtickets_kaggle: downloaded %d files", len(result_files))
        return result_files
        
    except Exception as e:
        logger.error("qtickets_kaggle: download failed: %s", e)
        return []


def parse_qtickets_output(file_paths: list[str]) -> list[TheatreEvent]:
    """Parse JSON output from Qtickets Kaggle kernel."""
    events: list[TheatreEvent] = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        
        # We expect a file named 'qtickets_events.json'
        if "qtickets" not in path.stem.lower():
            continue
        
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.warning("qtickets_parse: expected list, got %s", type(data))
                continue
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                
                # Parse date - kernel outputs "date" in "YYYY-MM-DD" format
                date_str = (item.get("date") or "").strip()
                time_str = (item.get("time") or "").strip()
                
                parsed_date = None
                if date_str:
                    try:
                        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        # Try parsing via parse_date_raw
                        parsed_date, _ = parse_date_raw(date_str)
                
                parsed_time = time_str if time_str else "00:00"
                
                # Location (already normalized by parser)
                location = (item.get("location") or "").strip()
                
                # Photos
                image_url = item.get("image_url")
                photos = [image_url] if image_url else []
                
                # Ticket info
                ticket_status = "available"  # Qtickets events are available by definition
                ticket_link = (item.get("url") or "").strip()
                
                # Prices
                price_min = item.get("price_min")
                price_max = item.get("price_max")
                
                # Age restriction
                age_restriction = (item.get("age_restriction") or "").strip()
                
                event = TheatreEvent(
                    title=title,
                    date_raw=date_str,
                    ticket_status=ticket_status,
                    url=ticket_link,
                    photos=photos,
                    description=(item.get("description") or "").strip(),
                    pushkin_card=False,
                    location=location,
                    age_restriction=age_restriction,
                    scene="",  # Qtickets events may not have scene info
                    source_type="qtickets",
                    parsed_date=parsed_date,
                    parsed_time=parsed_time,
                    ticket_price_min=price_min,
                    ticket_price_max=price_max,
                )
                events.append(event)
            
            logger.info("qtickets_parse: parsed %d events", len(events))
            
        except Exception as e:
            logger.error("qtickets_parse: failed to parse %s: %s", file_path, e)
            
    return events


async def process_qtickets_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    chat_id: int | None = None,
) -> SourceParsingStats:
    """Process Qtickets events."""
    import asyncio
    
    stats = SourceParsingStats(source="qtickets", total_received=len(events))
    
    for i, event in enumerate(events):
        current_progress = i + 1
        
        if not event.parsed_date:
            logger.warning("qtickets_process: no date for %s", event.title)
            stats.failed += 1
            continue
            
        location_name = normalize_location_name(event.location)
        
        existing_id, _ = await find_existing_event(
            db, location_name, event.parsed_date, event.parsed_time or "00:00", event.title
        )
        
        if existing_id:
            success = await update_event_ticket_status(
                db, existing_id, event.ticket_status, event.url
            )
            if success:
                stats.ticket_updated += 1
                stats.updated_event_ids.append(existing_id)
            else:
                stats.already_exists += 1
            await update_linked_events(db, existing_id, location_name, event.title)
        else:
            if bot and chat_id:
                try:
                    await bot.send_message(chat_id, f"🎫 Qtickets {current_progress}/{len(events)}: {event.title[:40]}")
                except:
                    pass
            
            poster_media_list = []
            if event.photos:
                images = await download_images(limit_photos_for_source(event.photos, "qtickets"))
                if images:
                    poster_media_list, _ = await process_media(images, need_catbox=True, need_ocr=True)
            
            new_id, was_added = await add_new_event_via_queue(
                db, bot, event, current_progress, len(events), poster_media=poster_media_list
            )
            
            if new_id and was_added:
                stats.new_added += 1
                stats.added_event_ids.append(new_id)
                await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
            else:
                stats.skipped += 1
                
    logger.info("qtickets_process: done added=%d updated=%d", stats.new_added, stats.ticket_updated)
    return stats
