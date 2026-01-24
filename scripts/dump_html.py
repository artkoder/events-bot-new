
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating...")
        try:
            await page.goto("https://filarmonia39.ru/?event", timeout=60000, wait_until="domcontentloaded")
            # Wait a bit for JS to render calendar
            await asyncio.sleep(5)
            content = await page.content()
            with open("philharmonia_dump.html", "w", encoding="utf-8") as f:
                f.write(content)
            print("Dumped HTML to philharmonia_dump.html")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
