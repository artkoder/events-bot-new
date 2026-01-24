
import asyncio
import re
import json
import logging
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PhilharmoniaParser")

BASE_URL = "https://filarmonia39.ru/?event"

class PhilharmoniaParser:
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.results: List[Dict[str, Any]] = []

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.page = await self.context.new_page()

    async def stop(self):
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

    async def parse(self, months_to_scan: int = 2):
        try:
            logger.info(f"Navigating to {BASE_URL}")
            # Use domcontentloaded for speed, the list is usually present or loads quickly
            await self.page.goto(BASE_URL, timeout=60000, wait_until="domcontentloaded")
            await asyncio.sleep(3) # Wait for initial JS rendering if any

            for month_idx in range(months_to_scan):
                logger.info(f"Processing month {month_idx + 1}")
                
                # Parse events on current page
                await self.parse_current_list()

                if month_idx < months_to_scan - 1:
                    # Find next month button
                    # Selector from dump: <a href="javascript:void(0)" class="nextMonth" ...>
                    next_button = self.page.locator("a.nextMonth")
                    
                    if await next_button.count() > 0 and await next_button.is_visible():
                        logger.info("Clicking next month")
                        # Capture current first event to check for changes
                        try:
                            first_event_title = await self.page.locator(".afisha_list_item h1").first.inner_text()
                        except:
                            first_event_title = ""

                        await next_button.click()
                        
                        # Wait for content update. 
                        # The site probably updates the div.schedule or div.afisha_list_item list.
                        # Simple wait for now, ideal is to wait for a specific element change.
                        await asyncio.sleep(3) 
                        
                        # Verify we moved (optional check)
                        # current_first_title = await self.page.locator(".afisha_list_item h1").first.inner_text()
                        # if current_first_title == first_event_title:
                        #     logger.warning("Month might not have changed or events are same.")
                    else:
                        logger.warning("Next month button not found or not visible")
                        break

        except Exception as e:
            logger.error(f"Error in main parse loop: {e}", exc_info=True)

    async def parse_current_list(self):
        """Parses the list of events (.afisha_list_item) currently visible."""
        # Selector for event items
        events = await self.page.locator("div.afisha_list_item").all()
        logger.info(f"Found {len(events)} events on current page.")

        for event_el in events:
            try:
                # Title & URL
                title_link = event_el.locator("h1 a.mer_item_title")
                if await title_link.count() == 0:
                    continue
                
                title = await title_link.inner_text()
                href = await title_link.get_attribute("href")
                full_url = "https://filarmonia39.ru" + href if href.startswith("/") else href
                
                # Deduplication
                if any(e['url'] == full_url for e in self.results):
                    continue

                # Image extraction
                img = event_el.locator("img")
                img_src = await img.get_attribute("src") if await img.count() > 0 else ""
                if img_src and not img_src.startswith("http"):
                    img_src = "https://filarmonia39.ru" + img_src

                # Navigate to detail page
                try:
                    new_page = await self.context.new_page()
                    # Use domcontentloaded for speed
                    await new_page.goto(full_url, timeout=45000, wait_until="domcontentloaded")
                    
                    # 1. Full Description
                    # Usually in a div with class 'text' or similar
                    desc_locator = new_page.locator("div.mer_item_info_text") 
                    if await desc_locator.count() == 0:
                         desc_locator = new_page.locator("div.text")

                    full_desc_text = ""
                    if await desc_locator.count() > 0:
                        full_desc_text = await desc_locator.first.inner_text()
                    else:
                        full_desc_text = await event_el.locator(".mer_item_list_progr").inner_text()
                    
                    # 2. Date/Time/Age Parsing
                    # Header on detail page: h1.mer_item_title
                    date_block = event_el.locator(".date_block")
                    date_text_raw = await date_block.inner_text()
                    date_text_clean = date_text_raw.replace("\n", " ").strip()
                    
                    # Extract Age: "12+" or "6+" or "0+"
                    age_restriction = ""
                    age_match = re.search(r'(\d+\+)', date_text_clean)
                    if age_match:
                        age_restriction = age_match.group(1)
                    
                    # Extract Time: HH:MM
                    time_val = "00:00"
                    time_match = re.search(r'(\d{1,2}:\d{2})', date_text_clean)
                    if time_match:
                        time_val = time_match.group(1)
                        
                    # Extract Date: clean up raw string
                    date_clean = re.sub(r'\d+\+', '', date_text_clean) # Remove age
                    date_clean = re.sub(r'Понедельник|Вторник|Среда|Четверг|Пятница|Суббота|Воскресенье', '', date_clean, flags=re.I)
                    date_clean = re.sub(r'\d{1,2}:\d{2}', '', date_clean) # Remove time
                    date_clean = re.sub(r'\s+', ' ', date_clean).strip()
                    
                    # 3. Prices
                    price_min = None
                    price_max = None
                    
                    # Check text content for "Цена: 500 - 1000 руб"
                    content_text = await new_page.locator("body").inner_text()
                    # Try explicit regex for price range first
                    price_match = re.search(r'Цена:?\s*([\d\s\,-]+)\s*(руб|₽)', content_text, re.IGNORECASE)
                    
                    if price_match:
                        price_str = price_match.group(1)
                        found_prices = [int(p) for p in re.findall(r'\d+', price_str)]
                        if found_prices:
                            price_min = min(found_prices)
                            price_max = max(found_prices)
                    
                    # Ticket Status
                    buy_btn = new_page.locator("a[href*='tickets=']")
                    has_ticket_btn = await buy_btn.count() > 0
                    ticket_status = "available" if has_ticket_btn else "unavailable"
                    
                    logger.info(f"Parsed: {title} | Date: {date_clean} | Tickets: {ticket_status}")
                    
                    self.results.append({
                        "title": title.strip(),
                        "url": full_url,
                        "date_text": date_clean,
                        "time": time_val,
                        "age_restriction": age_restriction,
                        "image_url": img_src,
                        "description": full_desc_text.strip(),
                        "price_min": price_min,
                        "price_max": price_max,
                        "ticket_status": ticket_status
                    })
                    
                    await new_page.close()
                    
                except Exception as e:
                    logger.error(f"Error parsing event detail {full_url}: {e}")
                    if 'new_page' in locals():
                        await new_page.close()
                    continue

            except Exception as e:
                 logger.error(f"Error processing event item: {e}")
                 continue

async def main():
    parser = PhilharmoniaParser()
    await parser.start()
    try:
        await parser.parse()
        print(json.dumps(parser.results, ensure_ascii=False, indent=2))
        
        with open("philharmonia_results.json", "w", encoding="utf-8") as f:
            json.dump(parser.results, f, ensure_ascii=False, indent=2)
            
    finally:
        await parser.stop()

if __name__ == "__main__":
    asyncio.run(main())
