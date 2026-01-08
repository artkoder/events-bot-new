"""
Focused test for Tian Lan date extraction after fix.
"""
import asyncio
import re
from datetime import date
from playwright.async_api import async_playwright

BASE_URL = "https://kaliningrad.tretyakovgallery.ru"

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
}


async def test_tian_lan_fix():
    """Test that Tian Lan now gets correct Feb 6 date from detail page."""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Test the detail page for Tian Lan
        url = f"{BASE_URL}/events/400/"
        print(f"📄 Testing Tian Lan detail page: {url}")
        
        await page.goto(url, timeout=30000, wait_until='domcontentloaded')
        await page.wait_for_timeout(2000)
        
        # Get title
        h1 = await page.query_selector('h1')
        title = (await h1.inner_text()).strip() if h1 else "NO TITLE"
        print(f"   Title: {title}")
        
        # Extract date from body
        body_text = await page.inner_text("body")
        
        match = re.search(r'(\d{1,2})\s+(\w+)\s+(?:в|В)\s+(\d{1,2}:\d{2})', body_text, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month_name = match.group(2).lower()
            time_str = match.group(3)
            
            month_num = MONTHS_RU.get(month_name)
            if month_num:
                today = date.today()
                year = today.year
                if today.month >= 10 and month_num < 3:
                    year += 1
                
                parsed_date = f"{year}-{month_num:02d}-{day:02d}"
                parsed_time = time_str
                
                print(f"\n   📅 Extracted date: {parsed_date}")
                print(f"   ⏰ Extracted time: {parsed_time}")
                
                # Validate
                expected_date = "2026-02-06"
                expected_time = "20:00"
                
                print(f"\n   ✅ Date is {expected_date}: {parsed_date == expected_date}")
                print(f"   ✅ Time is {expected_time}: {parsed_time == expected_time}")
                
                if parsed_date == expected_date and parsed_time == expected_time:
                    print("\n   🎉 FIX VERIFIED: Tian Lan will get correct date (Feb 6)!")
                else:
                    print("\n   ❌ FIX FAILED: Wrong date extracted")
        else:
            print("   ❌ No date found in page text")
        
        await context.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(test_tian_lan_fix())
