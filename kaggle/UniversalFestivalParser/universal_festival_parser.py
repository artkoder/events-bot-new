#!/usr/bin/env python3
"""Universal Festival Parser - Main Notebook Script.

RDR (Render-Distill-Reason) Architecture:
1. RENDER: Fetch and render page with Playwright
2. DISTILL: Clean HTML and extract structured content  
3. REASON: Use Gemma 3-27B to extract UDS JSON

Output artifacts:
- uds.json: Universal Data Structure with festival info
- render.html: Rendered page HTML
- screenshot.png: Full page screenshot
- distilled.json: Cleaned content for LLM
- llm_log.json: All LLM requests/responses for debugging
- metrics.json: Processing metrics
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("festival_parser")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main():
    """Main entry point for festival parser."""
    from config import ParserConfig
    from secrets import get_api_key
    from render import render_page, extract_images_from_html
    from distill import distill_html, prepare_llm_context, load_distilled
    from reason import reason_with_gemma, validate_and_enhance
    from rate_limit import GemmaRateLimiter
    from llm_logger import LLMLogger
    from uds import UDSOutput, UDSFestival, create_empty_uds
    
    start_time = time.perf_counter()
    
    # Load configuration
    try:
        config = ParserConfig.from_environment()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    
    logger.info("Starting parser: url=%s run_id=%s", config.festival_url, config.run_id)
    
    # Initialize components
    output_dir = config.output_dir
    llm_logger = LLMLogger(run_id=config.run_id)
    rate_limiter = GemmaRateLimiter()
    
    metrics = {
        "run_id": config.run_id,
        "url": config.festival_url,
        "parser_version": config.parser_version,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "phases": {},
    }
    
    # Phase 1: RENDER
    logger.info("=== Phase 1: RENDER ===")
    phase_start = time.perf_counter()
    
    render_result = await render_page(
        url=config.festival_url,
        output_dir=output_dir,
        timeout_ms=config.timeout_ms,
        capture_har=config.debug,
        headless=config.headless,
    )
    
    metrics["phases"]["render"] = {
        "success": render_result["success"],
        "duration_ms": (time.perf_counter() - phase_start) * 1000,
        "html_size": render_result.get("html_size", 0),
    }
    
    if not render_result["success"]:
        logger.error("Render failed: %s", render_result.get("error"))
        save_metrics(output_dir, metrics)
        return 1
    
    # Phase 2: DISTILL
    logger.info("=== Phase 2: DISTILL ===")
    phase_start = time.perf_counter()
    
    distill_result = distill_html(
        html_path=render_result["html_path"],
        output_dir=output_dir,
    )
    
    # Extract images from HTML
    images = extract_images_from_html(render_result["html_path"])
    
    metrics["phases"]["distill"] = {
        "success": distill_result["success"],
        "duration_ms": (time.perf_counter() - phase_start) * 1000,
        "text_length": len(distill_result.get("main_text", "")),
        "links_found": len(distill_result.get("links", [])),
        "images_found": len(images),
    }
    
    if not distill_result["success"]:
        logger.error("Distill failed: %s", distill_result.get("error"))
        save_metrics(output_dir, metrics)
        return 1
    
    # Phase 3: REASON
    logger.info("=== Phase 3: REASON ===")
    phase_start = time.perf_counter()
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("No API key available")
        metrics["phases"]["reason"] = {"error": "No API key"}
        save_metrics(output_dir, metrics)
        return 1
    
    # Prepare LLM context
    distilled = load_distilled(distill_result["distilled_path"])
    llm_context = prepare_llm_context(distilled)
    
    # Wait for rate limit
    async with rate_limiter.acquire(estimated_tokens=len(llm_context) // 4):
        extracted_data, error = await reason_with_gemma(
            distilled_content=llm_context,
            api_key=api_key,
            model=config.llm_model,
            llm_logger=llm_logger,
        )
    
    if error or not extracted_data:
        logger.error("Reason failed: %s", error)
        metrics["phases"]["reason"] = {"error": error}
        
        # Save partial results
        llm_logger.save(output_dir / "llm_log.json")
        save_metrics(output_dir, metrics)
        return 1
    
    # Validate and enhance
    async with rate_limiter.acquire(estimated_tokens=2000):
        extracted_data = await validate_and_enhance(
            extracted_data=extracted_data,
            distilled=distilled,
            api_key=api_key,
            llm_logger=llm_logger,
        )
    
    metrics["phases"]["reason"] = {
        "success": True,
        "duration_ms": (time.perf_counter() - phase_start) * 1000,
        "llm_calls": len(llm_logger.interactions),
    }
    
    # Build final UDS output
    logger.info("=== Building UDS Output ===")
    
    # Add images from HTML extraction
    if "images_festival" not in extracted_data:
        extracted_data["images_festival"] = images[:10]  # Limit to 10
    
    uds_data = {
        "uds_version": "1.0.0",
        "source_url": config.festival_url,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "parser_version": config.parser_version,
        "run_id": config.run_id,
        "llm_model": config.llm_model,
        "processing_time_ms": (time.perf_counter() - start_time) * 1000,
        **extracted_data,
    }
    
    # Validate against schema
    try:
        uds = UDSOutput(**uds_data)
        uds_json = uds.model_dump(mode="json")
    except Exception as e:
        logger.warning("UDS validation warning: %s", e)
        uds_json = uds_data  # Use raw data if validation fails
    
    # Save outputs
    uds_path = output_dir / "uds.json"
    uds_path.write_text(
        json.dumps(uds_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved UDS: %s", uds_path)
    
    # Save LLM log
    llm_logger.save(output_dir / "llm_log.json")
    
    # Save rate limiter stats
    rate_limiter.save_usage(output_dir / "rate_usage.json")
    
    # Final metrics
    metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
    metrics["total_duration_ms"] = (time.perf_counter() - start_time) * 1000
    metrics["success"] = True
    save_metrics(output_dir, metrics)
    
    logger.info(
        "Parser complete: duration=%.1fs festival=%s",
        metrics["total_duration_ms"] / 1000,
        extracted_data.get("festival", {}).get("title_short", "unknown"),
    )
    
    return 0


def save_metrics(output_dir: Path, metrics: dict) -> None:
    """Save metrics to JSON file."""
    path = output_dir / "metrics.json"
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
