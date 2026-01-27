
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional

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
PHILHARMONIA_KERNEL_FOLDER = "ParsePhilharmonia"

async def run_philharmonia_kaggle_kernel(
    timeout_minutes: int = 20,
    poll_interval: int = 30,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
) -> tuple[str, list[str], float]:
    """Run Kaggle kernel for parsing Philharmonia events.
    
    Args:
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        status_callback: Optional async callback for status updates
        
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    import asyncio
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / PHILHARMONIA_KERNEL_FOLDER
    kernel_ref = f"zigomaro/parse-philharmonia"  # Default
    registered = False

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.warning("philharmonia_kaggle: status callback failed phase=%s", phase)
    
    if not kernel_path.exists():
        logger.warning(
            "philharmonia_kaggle: kernel not found path=%s",
            kernel_path,
        )
        await _notify("not_found")
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("philharmonia_kaggle: kernel-metadata.json not found")
        await _notify("metadata_missing")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", kernel_ref)
    except Exception as e:
        logger.error("philharmonia_kaggle: failed to read metadata: %s", e)
        await _notify("metadata_error")
        return "error", [], time.time() - start_time
    
    logger.info(
        "philharmonia_kaggle: pushing kernel folder=%s ref=%s",
        PHILHARMONIA_KERNEL_FOLDER,
        kernel_ref,
    )
    
    await _notify("prepare")

    # Create a unique temp directory for this run
    import uuid
    run_id = str(uuid.uuid4())[:8]
    temp_kernel_dir = Path(tempfile.gettempdir()) / f"philharmonia_kernel_{run_id}"
    
    # Copy kernel to temp directory
    shutil.copytree(kernel_path, temp_kernel_dir)
    
    try:
        # Push kernel to Kaggle
        try:
            client.push_kernel(kernel_path=temp_kernel_dir)
        except Exception as e:
            logger.error("philharmonia_kaggle: push failed: %s", e)
            await _notify("push_failed")
            return "push_failed", [], time.time() - start_time
        
        await _notify("pushed")
        try:
            await register_job(
                "parse_philharmonia",
                kernel_ref,
                meta={"kernel_folder": PHILHARMONIA_KERNEL_FOLDER, "pid": os.getpid()},
            )
            registered = True
        except Exception:
            logger.warning("philharmonia_kaggle: failed to register recovery job", exc_info=True)

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
                    "philharmonia_kaggle: poll %d/%d status=%s",
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
                        "philharmonia_kaggle: failed status=%s message=%s",
                        status,
                        failure_msg,
                    )
                    await _notify("failed", status_response)
                    break
                elif status in ("QUEUED", "RUNNING"):
                    continue
                    
            except Exception as e:
                logger.warning("philharmonia_kaggle: status check failed: %s", e)
                continue

        
        duration = time.time() - start_time
        
        if final_status != "complete" and final_status != "failed":
            logger.error(
                "philharmonia_kaggle: not complete status=%s duration=%.1fs",
                final_status,
                duration,
            )
            if registered:
                await remove_job("parse_philharmonia", kernel_ref)
            return final_status, [], duration
        
        # Download output files
        try:
            output_files = await _download_philharmonia_outputs(client, kernel_ref)
        except Exception as e:
            logger.warning("philharmonia_kaggle: failed to download outputs: %s", e)
            output_files = []
            
        if final_status == "failed":
            logger.error("philharmonia_kaggle: kernel checked as failed, retrieved %d files", len(output_files))
            if registered:
                await remove_job("parse_philharmonia", kernel_ref)
            return "failed", output_files, duration
        
        logger.info(
            "philharmonia_kaggle: complete duration=%.1fs files=%d",
            duration,
            len(output_files),
        )
        if registered:
            await remove_job("parse_philharmonia", kernel_ref)
        return final_status, output_files, duration
        
    finally:
        try:
            shutil.rmtree(temp_kernel_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("philharmonia_kaggle: cleanup failed: %s", e)

async def _download_philharmonia_outputs(client: KaggleClient, kernel_ref: str) -> list[str]:
    """Download kernel output files to unique temp directory."""
    import uuid
    run_id = str(uuid.uuid4())[:8]
    output_dir = Path(tempfile.gettempdir()) / f"philharmonia_output_{run_id}"
    output_dir.mkdir(exist_ok=True)
    
    try:
        files = client.download_kernel_output(
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        result_files = [str(output_dir / f) for f in files]
        logger.info("philharmonia_kaggle: downloaded %d files", len(result_files))
        return result_files
        
    except Exception as e:
        logger.error("philharmonia_kaggle: download failed: %s", e)
        return []

def parse_philharmonia_output(file_paths: list[str]) -> list[TheatreEvent]:
    """Parse JSON output from Philharmonia Kaggle kernel."""
    events: list[TheatreEvent] = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        
        # We expect a file named 'philharmonia_results.json'
        # Or checking content if name differs
        if "philharmonia" not in path.stem.lower():
            continue
        
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            
            if not isinstance(data, list):
                logger.warning("philharmonia_parse: expected list, got %s", type(data))
                continue
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                
                # Parse date - our scraper outputs "date_text"
                # Need to implement proper date parsing from text "25 Января 2026 ..."
                # However, parse_date_raw handles various formats.
                # Or we should implement specific parsing inside the kernel?
                # The kernel returns "date_text": "25 Января 2026 12:00 Воскресенье 12+"
                
                date_raw = (item.get("date_text") or "").strip()
                normalized_date_str = (item.get("normalized_date") or "").strip()
                time_raw = (item.get("time") or "").strip()
                age_restriction = (item.get("age_restriction") or "").strip()
                
                # Use normalized date if available, otherwise try parsing raw text
                parsed_date = None
                if normalized_date_str:
                    try:
                        from datetime import datetime
                        parsed_date = datetime.strptime(normalized_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        pass
                
                if not parsed_date:
                    parsed_date, date_parsed_time = parse_date_raw(date_raw)
                    if not time_raw and date_parsed_time:
                        time_raw = date_parsed_time
                
                parsed_time = time_raw if time_raw else "00:00"
                
                # Photos
                image_url = item.get("image_url")
                photos = [image_url] if image_url else []
                
                # Ticket Status
                ticket_status = (item.get("ticket_status") or "unknown").strip()
                
                # Prices
                price_min = item.get("price_min")
                price_max = item.get("price_max")
                
                event = TheatreEvent(
                    title=title,
                    date_raw=date_raw,
                    ticket_status=ticket_status,
                    url=(item.get("url") or "").strip(),
                    photos=photos,
                    description=(item.get("description") or "").strip(),
                    pushkin_card=False, # TODO: Detect Pushkin card
                    location="Филармония",
                    age_restriction=age_restriction,
                    scene="Концертный зал",
                    source_type="philharmonia",
                    parsed_date=parsed_date,
                    parsed_time=parsed_time,
                    ticket_price_min=price_min,
                    ticket_price_max=price_max,
                )
                events.append(event)
            
            logger.info("philharmonia_parse: parsed %d events", len(events))
            
        except Exception as e:
            logger.error("philharmonia_parse: failed to parse %s: %s", file_path, e)
            
    return events

async def process_philharmonia_events(
    db: Database,
    bot: Bot | None,
    events: list[TheatreEvent],
    chat_id: int | None = None,
) -> SourceParsingStats:
    """Process Philharmonia events."""
    import asyncio
    
    stats = SourceParsingStats(source="philharmonia", total_received=len(events))
    
    for i, event in enumerate(events):
        current_progress = i + 1
        
        if not event.parsed_date:
            logger.warning("philharmonia_process: no date for %s", event.title)
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
            else:
                stats.already_exists += 1
            await update_linked_events(db, existing_id, location_name, event.title)
        else:
            if bot and chat_id:
                try:
                    await bot.send_message(chat_id, f"🎵 Филармония {current_progress}/{len(events)}: {event.title[:40]}")
                except:
                    pass
            
            poster_media_list = []
            if event.photos:
                images = await download_images(limit_photos_for_source(event.photos, "philharmonia"))
                if images:
                    poster_media_list, _ = await process_media(images, need_catbox=True, need_ocr=True)
            
            new_id, was_added = await add_new_event_via_queue(
                db, bot, event, current_progress, len(events), poster_media=poster_media_list
            )
            
            if new_id and was_added:
                stats.new_added += 1
                await asyncio.sleep(EVENT_ADD_DELAY_SECONDS)
            else:
                stats.skipped += 1
                
    logger.info("philharmonia_process: done added=%d updated=%d", stats.new_added, stats.ticket_updated)
    return stats
