"""Distill module for Universal Festival Parser (RDR Architecture).

Cleans and extracts structured content from rendered HTML:
1. Remove boilerplate (nav, footer, ads)
2. Extract main content
3. Structure for LLM processing
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def distill_html(
    html_path: str | Path,
    output_dir: Path,
) -> dict:
    """Distill HTML content for LLM processing.
    
    Args:
        html_path: Path to rendered HTML
        output_dir: Directory to save distilled content
        
    Returns:
        Dict with distilled content and metadata
    """
    from bs4 import BeautifulSoup
    
    html_path = Path(html_path)
    output_dir = Path(output_dir)
    
    result = {
        "success": False,
        "main_text": "",
        "structured_data": [],
        "links": [],
        "images": [],
        "distilled_path": None,
        "error": None,
    }
    
    try:
        html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        # Remove common ad/tracking elements
        for selector in [".ads", ".advertisement", ".social-share", ".cookie-banner"]:
            for el in soup.select(selector):
                el.decompose()
        
        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        
        if main:
            # Get text content
            text = main.get_text(separator="\n", strip=True)
            # Clean up excessive whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            result["main_text"] = text
            
            # Extract links
            for a in main.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http"):
                    result["links"].append({
                        "url": href,
                        "text": a.get_text(strip=True),
                    })
            
            # Extract images
            for img in main.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src and src.startswith("http"):
                    result["images"].append({
                        "url": src,
                        "alt": img.get("alt", ""),
                    })
        
        # Extract JSON-LD
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                result["structured_data"].append(data)
            except json.JSONDecodeError:
                continue
        
        # Extract Open Graph / meta tags
        result["meta"] = extract_meta_tags(soup)
        
        # Save distilled content
        distilled = {
            "main_text": result["main_text"],
            "links": result["links"],
            "images": result["images"],
            "structured_data": result["structured_data"],
            "meta": result["meta"],
        }
        
        distilled_path = output_dir / "distilled.json"
        distilled_path.write_text(
            json.dumps(distilled, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["distilled_path"] = str(distilled_path)
        result["success"] = True
        
        logger.info(
            "Distilled: text_len=%d links=%d images=%d json_ld=%d",
            len(result["main_text"]),
            len(result["links"]),
            len(result["images"]),
            len(result["structured_data"]),
        )
        
    except Exception as e:
        result["error"] = str(e)
        logger.error("Distill failed: %s", e)
    
    return result


def extract_meta_tags(soup) -> dict:
    """Extract useful meta tags from BeautifulSoup object."""
    meta = {}
    
    # Title
    title_tag = soup.find("title")
    if title_tag:
        meta["title"] = title_tag.get_text(strip=True)
    
    # Description
    desc = soup.find("meta", attrs={"name": "description"})
    if desc:
        meta["description"] = desc.get("content", "")
    
    # Open Graph
    for og in soup.find_all("meta", property=re.compile(r"^og:")):
        prop = og.get("property", "").replace("og:", "")
        meta[f"og_{prop}"] = og.get("content", "")
    
    return meta


def prepare_llm_context(distilled: dict, max_tokens: int = 8000) -> str:
    """Prepare distilled content for LLM input.
    
    Args:
        distilled: Distilled content dict
        max_tokens: Approximate max tokens for context
        
    Returns:
        Formatted string for LLM prompt
    """
    parts = []
    
    # Add meta info
    meta = distilled.get("meta", {})
    if meta.get("title"):
        parts.append(f"Page Title: {meta['title']}")
    if meta.get("description"):
        parts.append(f"Description: {meta['description']}")
    
    # Add JSON-LD if available (usually most reliable)
    json_ld = distilled.get("structured_data", [])
    if json_ld:
        parts.append("\n--- Structured Data (JSON-LD) ---")
        for item in json_ld:
            parts.append(json.dumps(item, ensure_ascii=False, indent=2))
    
    # Add main text (truncated if needed)
    main_text = distilled.get("main_text", "")
    if main_text:
        parts.append("\n--- Page Content ---")
        # Rough token estimate: ~4 chars per token
        max_chars = max_tokens * 4
        if len(main_text) > max_chars:
            main_text = main_text[:max_chars] + "\n... [TRUNCATED]"
        parts.append(main_text)
    
    # Add key links
    links = distilled.get("links", [])
    if links:
        parts.append("\n--- Links Found ---")
        for link in links[:20]:  # Limit to 20 most important
            parts.append(f"- {link['text']}: {link['url']}")
    
    return "\n".join(parts)


def load_distilled(distilled_path: str | Path) -> dict:
    """Load distilled content from JSON file."""
    path = Path(distilled_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
