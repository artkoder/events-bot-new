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
from typing import Any, Awaitable, Callable

from video_announce.kaggle_client import (
    KaggleClient,
    KERNELS_ROOT_PATH,
    LOCAL_KERNEL_PREFIX,
)

logger = logging.getLogger(__name__)

# Kernel folder name for theatres afisha
THEATRES_KERNEL_FOLDER = "ParseTheatres"


async def run_kaggle_kernel(
    kernel_folder: str = THEATRES_KERNEL_FOLDER,
    timeout_minutes: int = 30,
    poll_interval: int = 30,
    status_callback: Callable[[str, str, dict | None], Awaitable[None]] | None = None,
    run_config: dict[str, Any] | None = None,
    dataset_sources: list[str] | None = None,
) -> tuple[str, list[str], float]:
    """Run the Kaggle kernel and wait for completion.
    
    Uses existing KaggleClient from video_announce module.
    
    Args:
        kernel_folder: Folder name in kaggle/ directory
        timeout_minutes: Maximum wait time
        poll_interval: Seconds between status checks
        status_callback: Optional async callback for status updates
        run_config: Optional config dict to inject into notebook
        dataset_sources: Optional list of private dataset slugs (e.g., for API keys)
    
    Returns:
        Tuple of (status, output_files, duration_seconds)
    """

    import asyncio
    
    start_time = time.time()
    client = KaggleClient()
    kernel_path = KERNELS_ROOT_PATH / kernel_folder
    kernel_ref = f"{LOCAL_KERNEL_PREFIX}{kernel_folder}"

    async def _notify(phase: str, status: dict | None = None) -> None:
        if not status_callback:
            return
        try:
            await status_callback(phase, kernel_ref, status)
        except Exception:
            logger.exception("theatres_kaggle: status callback failed phase=%s", phase)
    
    if not kernel_path.exists():
        logger.warning(
            "theatres_kaggle: kernel not found path=%s",
            kernel_path,
        )
        await _notify("not_found")
        return "not_found", [], 0.0
    
    meta_path = kernel_path / "kernel-metadata.json"
    if not meta_path.exists():
        logger.warning("theatres_kaggle: kernel-metadata.json not found")
        await _notify("metadata_missing")
        return "not_found", [], 0.0
    
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        kernel_ref = meta.get("id", kernel_ref)
    except Exception as e:
        logger.error("theatres_kaggle: failed to read metadata: %s", e)
        await _notify("metadata_error")
        return "error", [], time.time() - start_time

    await _notify("prepare")
    
    logger.info(
        "theatres_kaggle: pushing kernel folder=%s ref=%s config=%s",
        kernel_folder,
        kernel_ref,
        run_config,
    )
    
    # Push kernel to Kaggle
    try:
        if run_config:
            # Inject config by modifying the notebook code directly
            # (Kaggle push doesn't upload auxiliary files reliably for notebooks)
            import shutil
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                # Copy all files first
                for item in kernel_path.iterdir():
                    dest = tmp_path / item.name
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
                
                # Find and modify the notebook file
                notebook_files = list(tmp_path.glob("*.ipynb"))
                if notebook_files:
                    nb_path = notebook_files[0]
                    try:
                        nb_content = json.loads(nb_path.read_text(encoding="utf-8"))
                        target = run_config.get("target_source", "all")
                        
                        # Modify the code in the first code cell
                        # We look for 'target = "all"' initialization in main()
                        for cell in nb_content.get("cells", []):
                            if cell.get("cell_type") == "code":
                                source_lines = cell.get("source", [])
                                new_source = []
                                for line in source_lines:
                                    if 'target = "all"' in line and 'config.get' not in line:
                                        # Replace default init with our target
                                        new_source.append(f'    target = "{target}"\n')
                                    else:
                                        new_source.append(line)
                                cell["source"] = new_source
                                break  # Only modify the first code cell
                        
                        nb_path.write_text(json.dumps(nb_content, indent=1), encoding="utf-8")
                        logger.info("theatres_kaggle: injected target=%s into notebook", target)
                    except Exception as e:
                        logger.error("theatres_kaggle: failed to inject config: %s", e)
                
                client.push_kernel(kernel_path=tmp_path, dataset_sources=dataset_sources)
        else:
            client.push_kernel(kernel_path=kernel_path, dataset_sources=dataset_sources)
    except Exception as e:
        logger.error("theatres_kaggle: push failed: %s", e)
        await _notify("push_failed")
        return "push_failed", [], time.time() - start_time

    await _notify("pushed")
    
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
            await _notify("poll", status_response)
            
            logger.info(
                "theatres_kaggle: poll %d/%d status=%s",
                poll + 1,
                max_polls,
                status,
            )
            
            if status == "COMPLETE":
                final_status = "complete"
                await _notify("complete", status_response)
                break
            elif status in ("ERROR", "FAILED", "CANCELLED"):
                final_status = "failed"
                failure_msg = status_response.get("failureMessage", "")
                logger.error(
                    "theatres_kaggle: failed status=%s message=%s",
                    status,
                    failure_msg,
                )
                await _notify("failed", status_response)
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
        if final_status == "timeout":
            await _notify("timeout", last_status)
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


async def run_kaggle_and_get_events() -> tuple[dict[str, list], str, float, list[str]]:
    """Run Kaggle kernel and return parsed events.
    
    Returns:
        Tuple of (events_by_source, log_file_path, kernel_duration, json_file_paths)
    """
    status, output_files, duration = await run_kaggle_kernel()
    
    if status == "not_found":
        logger.warning("theatres_kaggle: kernel not configured")
        return {}, "", 0.0, []
    
    if status != "complete":
        logger.error("theatres_kaggle: kernel failed status=%s", status)
        return {}, "", duration, []
    
    # Separate JSON files from other outputs
    json_files = [f for f in output_files if f.endswith('.json')]
    
    events_by_source = parse_output_files(output_files)
    
    # Create combined log file
    log_path = Path(tempfile.gettempdir()) / "theatres_afisha_log.txt"
    log_lines = [
        f"Kaggle kernel completed at {datetime.now().isoformat()}",
        f"Duration: {duration:.1f}s",
        f"Status: {status}",
        f"Output files: {len(output_files)}",
        "",
        "Events by source:",
    ]
    for source, events in events_by_source.items():
        log_lines.append(f"  {source}: {len(events)} events")
    
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    
    return events_by_source, str(log_path), duration, json_files
