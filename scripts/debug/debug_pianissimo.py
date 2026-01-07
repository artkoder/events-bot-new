"""
Debug script to test Tretyakov date extraction with cross-validation.
Tests that dates come from detail page, not from shared ticket widget.
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

def parse_russian_date(text: str) -> tuple[str | None, str | None]:
    """Parse Russian date like '6 февраля в 20:00' -> ('2026-02-06', '20:00')"""
    if not text:
        return None, None
    
    # Pattern: "6 февраля в 20:00" or "6 ФЕВРАЛЯ В 20:00"
    match = re.search(r'(\d{1,2})\s+(\w+)\s+(?:в|В)\s+(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        time_str = match.group(3)
        
        month_num = MONTHS_RU.get(month_name)
        if month_num:
            today = date.today()
            year = today.year
            # Handle year rollover
            if today.month >= 10 and month_num < 3:
                year += 1
            elif today.month <= 2 and month_num >= 10:
                year -= 1
            
            date_iso = f"{year}-{month_num:02d}-{day:02d}"
            return date_iso, time_str
    
    return None, None


async def test_pianissimo_dates():
    """Test that Pianissimo performer pages have correct dates from detail page."""
    
    # Known test cases from user's screenshots
    test_cases = [
        {
            "detail_url": "/events/400/",
            "expected_title": "ТЯНЬ ЛАН",
            "expected_date": "2026-02-06",
            "expected_time": "20:00",
        },
    ]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        print("=" * 60)
        print("PIANISSIMO DATE EXTRACTION TEST")
        print("=" * 60)
        
        for tc in test_cases:
            url = f"{BASE_URL}{tc['detail_url']}"
            print(f"\n📄 Testing: {url}")
            
            await page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await page.wait_for_timeout(2000)
            
            # Extract title
            h1 = await page.query_selector('h1')
            title = (await h1.inner_text()).strip() if h1 else "NO TITLE"
            print(f"   Title: {title}")
            
            # Extract date from visible text on page
            body_text = await page.inner_text("body")
            
            # Look for date pattern in body
            parsed_date, parsed_time = parse_russian_date(body_text)
            print(f"   Parsed date: {parsed_date}")
            print(f"   Parsed time: {parsed_time}")
            
            # Validate
            title_ok = tc["expected_title"] in title.upper()
            date_ok = parsed_date == tc["expected_date"]
            time_ok = parsed_time == tc["expected_time"]
            
            print(f"\n   ✅ Title contains '{tc['expected_title']}': {title_ok}")
            print(f"   ✅ Date is {tc['expected_date']}: {date_ok} (got {parsed_date})")
            print(f"   ✅ Time is {tc['expected_time']}: {time_ok} (got {parsed_time})")
            
            if title_ok and date_ok and time_ok:
                print(f"\n   🎉 PASSED")
            else:
                print(f"\n   ❌ FAILED")
        
        await context.close()
        await browser.close()


async def scan_all_events_list():
    """Scan /events/ page and extract detail_url for each event."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        url = f"{BASE_URL}/events/"
        print(f"\n🖼️ Scanning: {url}")
        
        await page.goto(url, timeout=60000, wait_until='domcontentloaded')
        await page.wait_for_timeout(3000)
        
        # Scroll to load all
        for _ in range(3):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1000)
        
        # Extract cards
        events = await page.evaluate("""
            () => {
                const events = [];
                document.querySelectorAll('.card').forEach(card => {
                    const titleEl = card.querySelector('.card_title');
                    if (!titleEl) return;
                    
                    const title = titleEl.innerText.trim();
                    if (title.toUpperCase().includes('ЭКСКУРСИЯ')) return;
                    
                    // Get detail_url from onclick
                    let detailUrl = null;
                    const onclick = card.getAttribute('onclick');
                    if (onclick) {
                        const match = onclick.match(/window\\.open\\(['\"]([^'\"]+)['\"]\\)/);
                        if (match) detailUrl = match[1];
                    }
                    
                    // Get ticket_url
                    let ticketUrl = null;
                    const ticketLink = card.querySelector('a[href*="tickets"]');
                    if (ticketLink) {
                        ticketUrl = ticketLink.getAttribute('href');
                    }
                    
                    events.push({
                        title: title,
                        detail_url: detailUrl,
                        ticket_url: ticketUrl,
                    });
                });
                return events;
            }
        """)
        
        print(f"\n📋 Found {len(events)} events:")
        
        # Filter Pianissimo events
        pianissimo_events = [e for e in events if 'PIANISSIMO' in e['title'].upper()]
        print(f"\n🎹 Pianissimo events: {len(pianissimo_events)}")
        
        for e in pianissimo_events:
            print(f"   - {e['title'][:50]}")
            print(f"     detail: {e['detail_url']}")
            print(f"     ticket: {e['ticket_url'][:50] if e['ticket_url'] else 'None'}...")
        
        await context.close()
        await browser.close()
        
        return pianissimo_events


async def test_all_pianissimo_dates():
    """Extract dates from detail pages for all Pianissimo events and validate."""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        list_page = await context.new_page()
        detail_page = await context.new_page()
        
        # Get events list
        url = f"{BASE_URL}/events/"
        await list_page.goto(url, timeout=60000, wait_until='domcontentloaded')
        await list_page.wait_for_timeout(3000)
        
        for _ in range(3):
            await list_page.mouse.wheel(0, 3000)
            await list_page.wait_for_timeout(1000)
        
        events = await list_page.evaluate("""
            () => {
                const events = [];
                document.querySelectorAll('.card').forEach(card => {
                    const titleEl = card.querySelector('.card_title');
                    if (!titleEl) return;
                    
                    const title = titleEl.innerText.trim();
                    
                    let detailUrl = null;
                    const onclick = card.getAttribute('onclick');
                    if (onclick) {
                        const match = onclick.match(/window\\.open\\(['\"]([^'\"]+)['\"]\\)/);
                        if (match) detailUrl = match[1];
                    }
                    
                    let ticketUrl = null;
                    const ticketLink = card.querySelector('a[href*="tickets"]');
                    if (ticketLink) {
                        ticketUrl = ticketLink.getAttribute('href');
                    }
                    
                    if (title.toUpperCase().includes('PIANISSIMO')) {
                        events.push({
                            title: title,
                            detail_url: detailUrl,
                            ticket_url: ticketUrl,
                        });
                    }
                });
                return events;
            }
        """)
        
        print("=" * 80)
        print("PIANISSIMO EVENTS - DETAIL PAGE DATE EXTRACTION")
        print("=" * 80)
        
        results = []
        
        for e in events:
            if not e['detail_url']:
                print(f"\n⚠️ {e['title'][:40]} - NO DETAIL URL")
                continue
            
            detail_url = f"{BASE_URL}{e['detail_url']}"
            print(f"\n📄 {e['title'][:50]}")
            print(f"   Detail: {detail_url}")
            
            await detail_page.goto(detail_url, timeout=30000, wait_until='domcontentloaded')
            await detail_page.wait_for_timeout(2000)
            
            # Get page text
            body_text = await detail_page.inner_text("body")
            
            # Parse date from body
            parsed_date, parsed_time = parse_russian_date(body_text)
            
            print(f"   📅 Date: {parsed_date} | Time: {parsed_time}")
            
            # Also check ticket URL
            if e['ticket_url']:
                is_shared = '/event/42168' in e['ticket_url']
                print(f"   🎫 Ticket URL: {'SHARED FESTIVAL' if is_shared else 'UNIQUE'}")
            
            results.append({
                'title': e['title'],
                'detail_url': e['detail_url'],
                'parsed_date': parsed_date,
                'parsed_time': parsed_time,
            })
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        for r in results:
            status = "✅" if r['parsed_date'] else "❌"
            print(f"{status} {r['title'][:40]}: {r['parsed_date']} {r['parsed_time']}")
        
        await context.close()
        await browser.close()
        
        return results


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STEP 1: Test known Tian Lan case")
    print("=" * 80)
    asyncio.run(test_pianissimo_dates())
    
    print("\n" + "=" * 80)
    print("STEP 2: Test ALL Pianissimo events")
    print("=" * 80)
    asyncio.run(test_all_pianissimo_dates())
