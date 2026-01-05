"""Render module for Universal Festival Parser (RDR Architecture).

Uses Playwright to:
1. Fetch and render the festival page
2. Wait for JavaScript content
3. Save HTML, screenshot, and optionally HAR
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def render_page(
    url: str,
    output_dir: Path,
    timeout_ms: int = 30000,
    wait_until: str = "networkidle",
    capture_har: bool = False,
    headless: bool = True,
) -> dict:
    """Render a page using Playwright and save artifacts.
    
    Args:
        url: URL to render
        output_dir: Directory to save artifacts
        timeout_ms: Page load timeout
        wait_until: Playwright wait_until option
        capture_har: Whether to capture network HAR
        headless: Run browser in headless mode
        
    Returns:
        Dict with paths to saved artifacts and metadata
    """
    from playwright.async_api import async_playwright
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "url": url,
        "success": False,
        "html_path": None,
        "screenshot_path": None,
        "har_path": None,
        "title": None,
        "error": None,
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        
        context_kwargs = {}
        if capture_har:
            har_path = output_dir / "network.har"
            context_kwargs["record_har_path"] = str(har_path)
            result["har_path"] = str(har_path)
        
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            **context_kwargs,
        )
        
        page = await context.new_page()
        
        try:
            logger.info("Rendering page: %s", url)
            
            response = await page.goto(
                url,
                wait_until=wait_until,
                timeout=timeout_ms,
            )
            
            if response:
                result["status_code"] = response.status
                if response.status >= 400:
                    result["error"] = f"HTTP {response.status}"
            
            # Get page title
            result["title"] = await page.title()
            
            # Wait a bit more for dynamic content
            await page.wait_for_timeout(2000)
            
            # Save HTML
            html_content = await page.content()
            html_path = output_dir / "render.html"
            html_path.write_text(html_content, encoding="utf-8")
            result["html_path"] = str(html_path)
            result["html_size"] = len(html_content)
            
            # Save screenshot
            screenshot_path = output_dir / "screenshot.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            result["screenshot_path"] = str(screenshot_path)
            
            result["success"] = result["error"] is None
            if result["success"]:
                logger.info(
                    "Rendered successfully: title=%s html_size=%d",
                    result["title"],
                    result["html_size"],
                )
            else:
                logger.warning("Rendered with error: %s", result["error"])
            
        except Exception as e:
            result["error"] = str(e)
            logger.error("Render failed: %s", e)
        
        finally:
            if capture_har:
                await context.close()
            await browser.close()
    
    return result


def extract_images_from_html(html_path: str | Path) -> list[str]:
    """Extract image URLs from rendered HTML.
    
    Args:
        html_path: Path to rendered HTML file
        
    Returns:
        List of image URLs found
    """
    from bs4 import BeautifulSoup
    
    html_path = Path(html_path)
    if not html_path.exists():
        return []
    
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    
    images = []
    
    # img tags
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            images.append(src)
    
    # Background images in style attributes
    import re
    bg_pattern = r'url\(["\']?(https?://[^"\'\)]+)["\']?\)'
    for match in re.findall(bg_pattern, html):
        images.append(match)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for url in images:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    
    return unique


def extract_json_ld(html_path: str | Path) -> list[dict]:
    """Extract JSON-LD structured data from HTML.
    
    Args:
        html_path: Path to rendered HTML file
        
    Returns:
        List of parsed JSON-LD objects
    """
    from bs4 import BeautifulSoup
    
    html_path = Path(html_path)
    if not html_path.exists():
        return []
    
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    
    results = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            results.append(data)
        except json.JSONDecodeError:
            continue
    
    return results
