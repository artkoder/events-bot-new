
import asyncio
from playwright.async_api import async_playwright
import re
import os

URLS = [
    ("Fairy Tale", "https://kaliningrad.tretyakovgallery.ru/tickets/#/buy/event/43348"),
    ("Excursion", "https://kaliningrad.tretyakovgallery.ru/tickets/#/buy/excursion/combined/4493/73/2/0/106"),
]

ARTIFACTS_DIR = "/home/codespace/.gemini/antigravity/brain/c9aab43e-aa74-4a77-85a0-0b792ad7e910"

async def analyze_url(page, label, url):
    print(f"\n--- Analyzing: {label} ---")
    print(f"URL: {url}")
    
    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        
        # Wait for widget
        try:
            await page.wait_for_selector('.skin-inner, app-root, .wrapper, .page-buy-container', timeout=20000)
            print("✅ Widget container loaded.")
        except:
            print("❌ Widget container NOT found (timeout).")
            return

        await page.wait_for_timeout(3000)
        
        # DUMP HTML of the main container
        # We saw 'page_buy_event-date' in previous logs
        container_sel = '.page_buy_event-date'
        if await page.locator(container_sel).count() > 0:
            print(f"✅ Found container '{container_sel}'")
            html = await page.inner_html(container_sel)
            print(f"--- HTML START ({container_sel}) ---")
            print(html[:2000]) # Print first 2000 chars
            print("--- HTML END ---")
        else:
            print(f"⚠️ Container '{container_sel}' NOT found. Dumping body start...")
            body = await page.inner_html("body")
            print(body[:1000])

        # Excursions might use different classes
        if "Excursion" in label:
             print("Checking excursion specifics...")
             # Check for session items
             sessions_sel = '.session-item, .event-item'
             count = await page.locator(sessions_sel).count()
             print(f"Found {count} elements matching '{sessions_sel}'")
             if count > 0:
                 html = await page.inner_html(sessions_sel + ":first-child")
                 print(f"--- HTML SESSION ITEM 1 ---")
                 print(html)
                 print("--- END ---")

    except Exception as e:
        print(f"❌ Error analysis: {e}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
        page = await context.new_page()
        for label, url in URLS:
            await analyze_url(page, label, url)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
