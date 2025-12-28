"""Kaggle runner for theatre afisha parsing notebook.

Reuses the existing KaggleClient from video_announce module.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from video_announce.kaggle_client import (
    KaggleClient,
    KERNELS_ROOT_PATH,
    LOCAL_KERNEL_PREFIX,
)

logger = logging.getLogger(__name__)

# Kernel folder name for theatres afisha
THEATRES_KERNEL_FOLDER = "TheatresAfisha"


async def run_kaggle_kernel(
    kernel_folder: str = THEATRES_KERNEL_FOLDER,
    timeout_minutes: int = 30,
    poll_interval: int = 30,
) -> tuple[str, list[str], float]:
    """Run the Kaggle kernel and wait for completion.
    
    Uses existing KaggleClient from video_announce module.
    
    Args:
        kernel_folder: Folder name in kaggle/ directory
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
    
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """
    import asyncio
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / kernel_folder
    
    if not kernel_path.exists():
        logger.warning(
            "theatres_kaggle: kernel not found path=%s",
            kernel_path,
        )
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("theatres_kaggle: kernel-metadata.json not found")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", f"{LOCAL_KERNEL_PREFIX}{kernel_folder}")
    except Exception as e:
        logger.error("theatres_kaggle: failed to read metadata: %s", e)
        return "error", [], time.time() - start_time
    
    logger.info(
        "theatres_kaggle: pushing kernel folder=%s ref=%s",
        kernel_folder,
        kernel_ref,
    )
    
    # Push kernel to Kaggle
    try:
        client.push_kernel(kernel_path=kernel_path)
    except Exception as e:
        logger.error("theatres_kaggle: push failed: %s", e)
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
                "theatres_kaggle: poll %d/%d status=%s",
                poll + 1,
                max_polls,
                status,
            )
            
            if status == "COMPLETE":
                final_status = "complete"
                break
            elif status in ("ERROR", "FAILED", "CANCELLED"):
                final_status = "failed"
                failure_msg = status_response.get("failureMessage", "")
                logger.error(
                    "theatres_kaggle: failed status=%s message=%s",
                    status,
                    failure_msg,
                )
                break
            elif status in ("QUEUED", "RUNNING"):
                continue
                
        except Exception as e:
            logger.warning("theatres_kaggle: status check failed: %s", e)
            continue
    
    duration = time.time() - start_time
    
    if final_status != "complete":
        logger.error(
            "theatres_kaggle: not complete status=%s duration=%.1fs",
            final_status,
            duration,
        )
        return final_status, [], duration
    
    # Download output files
    output_files = await _download_outputs(client, kernel_ref)
    
    logger.info(
        "theatres_kaggle: complete duration=%.1fs files=%d",
        duration,
        len(output_files),
    )
    
    return final_status, output_files, duration


async def _download_outputs(client: KaggleClient, kernel_ref: str) -> list[str]:
    """Download kernel output files."""
    output_dir = Path(tempfile.gettempdir()) / "theatres_afisha_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        files = client.download_kernel_output(
            kernel_ref,
            path=str(output_dir),
            force=True,
        )
        result_files = [str(output_dir / f) for f in files]
        logger.info("theatres_kaggle: downloaded %d files", len(result_files))
        return result_files
        
    except Exception as e:
        logger.error("theatres_kaggle: download failed: %s", e)
        return []


def parse_output_files(file_paths: list[str]) -> dict[str, Any]:
    """Parse the JSON output files from kernel.
    
    Args:
        file_paths: List of downloaded file paths
    
    Returns:
        Dict mapping source name to parsed events list
    """
    from source_parsing.parser import parse_theatre_json
    
    results: dict[str, Any] = {}
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        
        # Determine source from filename
        name = path.stem.lower()
        if "dramteatr" in name or "драм" in name:
            source = "dramteatr"
        elif "muzteatr" in name or "муз" in name:
            source = "muzteatr"
        elif "sobor" in name or "собор" in name:
            source = "sobor"
        else:
            source = name
        
        try:
            content = path.read_text(encoding="utf-8")
            events = parse_theatre_json(content, source)
            results[source] = events
            logger.info(
                "theatres_kaggle: parsed source=%s events=%d",
                source,
                len(events),
            )
        except Exception as e:
            logger.error(
                "theatres_kaggle: parse failed file=%s error=%s",
                file_path,
                e,
            )
    
    return results


async def run_kaggle_and_get_events() -> tuple[dict[str, list], str, float]:
    """Run Kaggle kernel and return parsed events.
    
    Returns:
        Tuple of (events_by_source, log_file_path, kernel_duration)
    """
    status, output_files, duration = await run_kaggle_kernel()
    
    if status == "not_found":
        logger.warning("theatres_kaggle: kernel not configured")
        return {}, "", 0.0
    
    if status != "complete":
        logger.error("theatres_kaggle: kernel failed status=%s", status)
        return {}, "", duration
    
    events_by_source = parse_output_files(output_files)
    
    # Create combined log file
    log_path = Path(tempfile.gettempdir()) / "theatres_afisha_log.txt"
    log_lines = [
        f"Kaggle kernel completed at {datetime.now().isoformat()}",
        f"Duration: {duration:.1f}s",
        f"Status: {status}",
        "",
        "Events by source:",
    ]
    for source, events in events_by_source.items():
        log_lines.append(f"  {source}: {len(events)} events")
    
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    
    return events_by_source, str(log_path), duration
