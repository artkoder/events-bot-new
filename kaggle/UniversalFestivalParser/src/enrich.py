"""Enrich module for Universal Festival Parser (RDR+E Architecture).

Phase 4: ENRICH - Parse ticket URLs to extract:
- Actual prices
- Ticket availability status
- Additional event details

UNIVERSAL APPROACH: Uses LLM to intelligently extract data from ANY ticket page,
not tied to specific sites like pyramida.info. Works with any festival website.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TicketInfo:
    """Extracted ticket information."""
    price: Optional[str] = None
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    currency: str = "RUB"
    status: Optional[str] = None  # available, sold_out, ended, registration_open
    available_count: Optional[int] = None


# LLM prompt for universal ticket extraction
TICKET_EXTRACTION_PROMPT = """Ты эксперт по извлечению информации о билетах с веб-страниц.
Проанализируй текст страницы и извлеки информацию о билетах.

Текст страницы:
{page_text}

Извлеки и верни JSON:
{{
  "price": "цена билета, например '500 руб' или 'от 500 руб' или 'бесплатно' или null",
  "price_min": число минимальной цены в рублях или null,
  "price_max": число максимальной цены в рублях или null,
  "status": "available" | "sold_out" | "ended" | "registration_open" | "registration_closed" | null,
  "available_count": число доступных билетов или null
}}

Правила:
- Если билеты распроданы/sold out/закончились → status: "sold_out"
- Если мероприятие завершено/прошло → status: "ended"  
- Если регистрация открыта (бесплатно) → status: "registration_open"
- Если можно купить → status: "available"
- Цены в рублях, ищи числа рядом со словами "руб", "₽", "стоимость", "цена"
- Верни ТОЛЬКО JSON, никаких пояснений
"""


async def fetch_and_distill_ticket_page(url: str, timeout_ms: int = 15000) -> Optional[str]:
    """Fetch ticket page and extract main text using Playwright.
    
    Args:
        url: Ticket URL
        timeout_ms: Page load timeout
        
    Returns:
        Page main text or None if failed
    """
    from playwright.async_api import async_playwright
    from bs4 import BeautifulSoup
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await asyncio.sleep(1.5)  # Wait for dynamic content
            
            html = await page.content()
            await browser.close()
            
            # Distill to main text
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            # Limit text size for LLM
            return text[:8000]
            
    except Exception as e:
        logger.warning(f"Failed to fetch ticket page {url}: {e}")
        return None


async def extract_ticket_info_with_llm(
    page_text: str,
    api_key: str,
    model: str = "gemma-3-27b-it",
) -> Optional[TicketInfo]:
    """Extract ticket info using LLM (universal approach).
    
    Args:
        page_text: Distilled page text
        api_key: Google API key
        model: LLM model to use
        
    Returns:
        TicketInfo or None
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    prompt = TICKET_EXTRACTION_PROMPT.format(page_text=page_text[:6000])
    
    try:
        model_instance = genai.GenerativeModel(model_name=f"models/{model}")
        response = await model_instance.generate_content_async(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 500},
        )
        
        response_text = response.text or ""
        
        # Strip code fences
        if "```json" in response_text:
            response_text = re.sub(r"```json\s*", "", response_text)
            response_text = re.sub(r"```\s*$", "", response_text)
        elif "```" in response_text:
            response_text = re.sub(r"```\s*", "", response_text)
        
        data = json.loads(response_text.strip())
        return TicketInfo(
            price=data.get("price"),
            price_min=data.get("price_min"),
            price_max=data.get("price_max"),
            status=data.get("status"),
            available_count=data.get("available_count"),
        )
        
    except Exception as e:
        logger.warning(f"LLM ticket extraction failed: {e}")
        return None


async def enrich_event_prices(
    events: list[dict],
    api_key: Optional[str] = None,
    max_concurrent: int = 2,
) -> list[dict]:
    """Enrich events with prices from their ticket URLs.
    
    Uses UNIVERSAL LLM-based extraction that works with any ticket site.
    
    Args:
        events: List of event dicts with ticket_url field
        api_key: Google API key (from env if not provided)
        max_concurrent: Max concurrent requests
        
    Returns:
        Events with enriched price data
    """
    import os
    
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        logger.warning("No API key for ticket enrichment")
        return events
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_event(event: dict) -> dict:
        ticket_url = event.get("ticket_url")
        
        # Skip if no URL or already has price
        if not ticket_url:
            return event
        if event.get("price") and event.get("price") != "null":
            return event
            
        async with semaphore:
            logger.info(f"Enriching: {ticket_url[:60]}...")
            
            # Fetch and distill ticket page
            page_text = await fetch_and_distill_ticket_page(ticket_url)
            if not page_text:
                return event
            
            # Use LLM for universal extraction
            info = await extract_ticket_info_with_llm(page_text, api_key)
            
            if info:
                if info.price:
                    event["price"] = info.price
                    logger.info(f"  ✓ Found price: {info.price}")
                if info.status:
                    event["ticket_status"] = info.status
                    logger.info(f"  ✓ Found status: {info.status}")
                    
        return event
    
    # Process events concurrently
    tasks = [process_event(ev.copy()) for ev in events]
    enriched = await asyncio.gather(*tasks)
    
    return list(enriched)
