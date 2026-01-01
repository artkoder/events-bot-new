
import asyncio
from playwright.async_api import async_playwright
import re
import os

URLS = [
    ("Fairy Tale", "https://kaliningrad.tretyakovgallery.ru/tickets/#/buy/event/43348", 25, "11:00"),
    ("Pianissimo", "https://kaliningrad.tretyakovgallery.ru/tickets/#/buy/event/42168/2026-01-16/20:00:00", 16, "20:00"),
]

ARTIFACTS_DIR = "/home/codespace/.gemini/antigravity/brain/c9aab43e-aa74-4a77-85a0-0b792ad7e910"

async def interact_and_debug(page, label, url, target_day, target_time):
    print(f"\n--- Analyzing: {label} ---")
    print(f"URL: {url}")
    
    try:
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        try:
            await page.wait_for_selector('div[class*="calendar"], .skin-inner, app-root, .wrapper', timeout=10000)
        except:
             print("Widget load timeout")
             return

        await page.wait_for_timeout(2000)

        # 1. DATE CLICK (Hybrid)
        click_success = False
        print(f"   Searching Day: {target_day}")
        
        # Standard
        candidates = await page.locator('.cell:not(.day-header), .day, .date-item').all()
        for cand in candidates:
            if not await cand.is_visible(): continue
            txt = (await cand.inner_text()).strip()
            if txt == str(target_day):
                await cand.click(force=True)
                click_success = True
                print(f"   >>> [Standard] Clicked date {target_day}")
                break
        
        # Fuzzy
        if not click_success:
             print("   ...Standard failed, trying Fuzzy JS")
             candidates = await page.evaluate_handle(r"""
                (day) => {
                    const els = Array.from(document.querySelectorAll('div, span, button, td'));
                    const matches = els.filter(el => {
                        if (el.offsetParent === null) return false;
                        const txt = el.innerText.trim();
                        const dayRx = new RegExp("(^|\\D)" + day + "(\\D|$)", "i");
                        if (!dayRx.test(txt)) return false;
                        if (txt.length > 50) return false;
                        return true;
                    });
                    matches.sort((a,b) => a.innerText.length - b.innerText.length);
                    return matches.length > 0 ? matches[0] : null;
                }
            """, target_day)
             if candidates.as_element():
                await candidates.as_element().click()
                print(f"   >>> [Fuzzy] Clicked date {target_day}")

        # 2. TIME CLICK
        await page.wait_for_timeout(1500)
        print(f"   Searching Time: {target_time}")
        time_clicked = False
        
        time_el = await page.evaluate_handle(r"""
            (tgt) => {
                const els = Array.from(document.querySelectorAll('div, span, button, td'));
                const matches = els.filter(el => {
                    if (el.offsetParent === null) return false;
                    const txt = el.innerText.trim();
                    const timeMatch = txt.match(/\d{1,2}:\d{2}/);
                    if (!timeMatch) return false;
                    if (tgt && !txt.includes(tgt)) return false;
                    return true;
                });
                matches.sort((a,b) => a.innerText.length - b.innerText.length);
                return matches.length > 0 ? matches[0] : null;
            }
        """, target_time)
        
        if time_el.as_element():
             await time_el.as_element().click()
             time_clicked = True
             print(f"   >>> Clicked time: {target_time}")
        
        # 3. ANALYZE POST-TIME STATE
        print("   >>> WAITING FOR NEXT STEP (SECTOR/PRICE)...")
        await page.wait_for_timeout(4000)
        
        # Screenshot
        safe_label = label.replace(" ", "_").lower()
        path = f"{ARTIFACTS_DIR}/debug_{safe_label}_post_time.png"
        await page.screenshot(path=path)
        print(f"📸 Screenshot: {path}")
        
        # Dump visible text
        body_text = await page.inner_text("body")
        print(f"   VISIBLE TEXT:\n{body_text[:500]}...")
        
        # Check for Canvas/SVG (Map?)
        svgs = await page.locator('svg').count()
        canvas = await page.locator('canvas').count()
        print(f"   Map Elements? SVGs: {svgs}, Canvases: {canvas}")
        
        # Check for "Select" buttons
        buttons = await page.locator('button:visible').all_inner_texts()
        print(f"   Visible Buttons: {buttons}")

    except Exception as e:
        print(f"❌ Error analysis: {e}")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
        page = await context.new_page()
        for label, url, day, time in URLS:
            await interact_and_debug(page, label, url, day, time)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
