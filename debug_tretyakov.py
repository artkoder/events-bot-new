"""
COMPLETE End-to-End Local Test for Tretyakov Parser.
Gets title+description from detail page, dates/times/prices from ticket page.
"""
import asyncio
import json
import re
from datetime import date
from playwright.async_api import async_playwright

BASE_URL = "https://kaliningrad.tretyakovgallery.ru"

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
}


async def scrape_events_list(page):
    """Scrape events from /events/ page, extracting both detail_url and ticket_url."""
    url = f"{BASE_URL}/events/"
    print(f"\n📋 STEP 1: Scraping events from {url}")
    
    await page.goto(url, timeout=60000, wait_until='domcontentloaded')
    await page.wait_for_timeout(3000)
    
    for _ in range(3):
        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(1000)
    
    # Extract events with BOTH detail_url (from onclick) and ticket_url
    events = await page.evaluate("""
        () => {
            const events = [];
            const seen = new Set();
            
            document.querySelectorAll('.card').forEach(card => {
                const titleEl = card.querySelector('.card_title');
                if (!titleEl) return;
                
                const title = titleEl.innerText.trim();
                if (title.toUpperCase().includes('ЭКСКУРСИЯ')) return;
                
                // Get detail_url from onclick attribute
                let detailUrl = null;
                const onclick = card.getAttribute('onclick');
                if (onclick) {
                    const match = onclick.match(/window\\.open\\(['"]([^'"]+)['"]/);
                    if (match) {
                        detailUrl = match[1];
                    }
                }
                
                // Get ticket_url
                let ticketUrl = null;
                const ticketLink = card.querySelector('a[href*="tickets"]');
                if (ticketLink) {
                    let href = ticketLink.getAttribute('href');
                    if (href.startsWith('//')) ticketUrl = 'https:' + href;
                    else ticketUrl = href;
                }
                
                if (ticketUrl && ticketUrl.includes('timepad')) return;
                
                // Get photo
                let photo = null;
                const img = card.querySelector('img.card_img');
                if (img && img.src) {
                    photo = img.src;
                }
                
                // Get location
                const text = card.innerText.toUpperCase();
                let location = 'Третьяковка';
                if (text.includes('АТРИУМ')) location = 'Атриум';
                else if (text.includes('КИНОЗАЛ')) location = 'Кинозал';
                
                const key = title + ticketUrl;
                if (seen.has(key)) return;
                seen.add(key);
                
                if (ticketUrl) {
                    events.push({
                        title_raw: title,
                        detail_url: detailUrl,
                        ticket_url: ticketUrl,
                        photo: photo,
                        location: location
                    });
                }
            });
            return events;
        }
    """)
    
    print(f"   Found {len(events)} total events")
    
    # Select 2 events: PIANISSIMO and СКАЗКА
    selected = []
    for e in events:
        t = e['title_raw'].upper()
        if len(selected) < 2:
            if "ЗИМНИЙ ФЕСТИВАЛЬ PIANISSIMO" in t or "СКАЗКА ПРО ПРИТАИВШЕГОСЯ" in t:
                selected.append(e)
                print(f"   ✓ {e['title_raw'][:50]}...")
                print(f"      detail: {e['detail_url']}")
    
    return selected if selected else events[:2]


async def scrape_detail_page(page, detail_url):
    """Visit event detail page to get title and description."""
    if not detail_url:
        return {"title": None, "description": None}
    
    full_url = f"{BASE_URL}{detail_url}" if detail_url.startswith('/') else detail_url
    print(f"\n   📄 Detail page: {full_url}")
    
    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await page.wait_for_timeout(2000)
        
        # Title from h1
        title = None
        h1 = await page.query_selector('h1')
        if h1:
            title = (await h1.inner_text()).strip()
        
        # Description from paragraphs
        description = None
        paragraphs = await page.query_selector_all('p')
        for p in paragraphs:
            text = (await p.inner_text()).strip()
            if len(text) > 80 and not any(skip in text.lower() for skip in ['cookie', 'политик', 'hours']):
                description = text[:500]
                break
        
        print(f"      Title: {title[:50] if title else 'N/A'}...")
        print(f"      Description: {description[:50] if description else 'N/A'}...")
        
        return {"title": title, "description": description}
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        return {"title": None, "description": None}


async def parse_ticket_page(page, ticket_url, max_combos=4):
    """Parse ticket page for dates, times, prices."""
    full_url = f"{BASE_URL}{ticket_url}" if ticket_url.startswith('/') else ticket_url
    print(f"\n   🎫 Ticket page: {full_url[:60]}...")
    today = date.today()
    
    try:
        await page.goto(full_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(3000)
        
        # ACTIVE dates
        active_dates = await page.evaluate("""
            () => {
                const items = [];
                document.querySelectorAll('div.item.active').forEach(item => {
                    const dayEl = item.querySelector('.calendarDay');
                    const monthEl = item.querySelector('.calendarMonth');
                    if (dayEl) {
                        items.push({ day: dayEl.innerText.trim(), month: monthEl ? monthEl.innerText.trim().toLowerCase() : '' });
                    }
                });
                return items;
            }
        """)
        
        print(f"      📅 ACTIVE dates: {len(active_dates)}")
        entries = []
        
        for d in active_dates[:2]:
            if len(entries) >= max_combos: break
            
            month_num = MONTHS_RU.get(d['month'], 1)
            year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
            
            try:
                date_obj = date(year, month_num, int(d['day']))
                if date_obj < today: continue
            except: continue
            
            # Click date
            day_val = d['day']
            await page.evaluate(f"() => {{ document.querySelectorAll('div.item.active').forEach(i => {{ const dayEl = i.querySelector('.calendarDay'); if (dayEl && dayEl.innerText.trim() === '{day_val}') i.click(); }}); }}")
            await page.wait_for_timeout(1500)
            
            # Get times
            times = await page.evaluate("""() => [...document.querySelectorAll('label.select-time-button:not(.disabled)')].map(b => b.innerText.trim().match(/^\\d{1,2}:\\d{2}$/)?.[0]).filter(Boolean)""")
            print(f"         {d['day']} {d['month']}: {len(times)} times ({', '.join(times) or 'none'})")
            
            for t in times[:2]:
                if len(entries) >= max_combos: break
                
                # Click time
                await page.evaluate(f"() => {{ document.querySelectorAll('label.select-time-button').forEach(b => {{ if (b.innerText.includes('{t}')) b.click(); }}); }}")
                await page.wait_for_timeout(1500)
                
                # Sector click
                sectors = await page.locator('text=/[Сс]ектор/').all()
                if sectors:
                    try: await sectors[0].click(); await page.wait_for_timeout(1000)
                    except: pass
                
                # Price
                prices = await page.evaluate("""() => [...new Set([...document.querySelectorAll('*')].map(e => e.innerText?.match(/(\\d+)\\s*₽/)?.[1]).filter(Boolean).map(Number).filter(p => p > 100))]""")
                price = min(prices) if prices else None
                status = "available" if prices else "unknown"
                
                entries.append({
                    "parsed_date": date_obj.isoformat(),
                    "parsed_time": t,
                    "date_raw": f"{d['day']} {d['month']} в {t}",
                    "price": price,
                    "status": status,
                })
        
        return entries
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        return []


async def main():
    print("=" * 70)
    print("🖼️ TRETYAKOV COMPLETE E2E TEST (with description)")
    print(f"📅 Today: {date.today()}")
    print("=" * 70)
    
    all_events = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
        
        list_page = await context.new_page()
        detail_page = await context.new_page()
        ticket_page = await context.new_page()
        
        events_raw = await scrape_events_list(list_page)
        
        for idx, event in enumerate(events_raw):
            print(f"\n{'='*70}")
            print(f"📌 [{idx+1}/{len(events_raw)}] {event['title_raw'][:50]}...")
            
            # Get title and description from detail page
            detail = await scrape_detail_page(detail_page, event.get('detail_url'))
            title = detail['title'] or event['title_raw']
            description = detail['description']
            
            # Get dates/times/prices from ticket page
            entries = await parse_ticket_page(ticket_page, event['ticket_url'])
            
            photo = event['photo']
            if photo and photo.startswith('/'):
                photo = f"{BASE_URL}{photo}"
            
            for e in (entries or [{"parsed_date": None, "parsed_time": None, "date_raw": "", "price": None, "status": "unknown"}]):
                all_events.append({
                    "title": title,
                    "description": description,
                    "date_raw": e['date_raw'],
                    "parsed_date": e['parsed_date'],
                    "parsed_time": e['parsed_time'],
                    "ticket_status": e['status'],
                    "ticket_price_min": e['price'],
                    "ticket_price_max": e['price'],
                    "url": f"{BASE_URL}{event['ticket_url']}/{e['parsed_date']}/{e['parsed_time']}:00" if e['parsed_date'] else f"{BASE_URL}{event['ticket_url']}",
                    "photos": [photo] if photo else [],
                    "location": event['location'],
                })
        
        await browser.close()
    
    # Save
    with open("/tmp/tretyakov_final.json", "w", encoding="utf-8") as f:
        json.dump(all_events, f, ensure_ascii=False, indent=2)
    
    # JSON Output
    print("\n" + "=" * 70)
    print("📋 JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(all_events, ensure_ascii=False, indent=2))
    
    # Summary with checkmarks
    print("\n" + "=" * 70)
    print("📊 FIELD SUMMARY (✅ = OK, ❌ = missing)")
    print("=" * 70)
    
    for e in all_events:
        print(f"\n🎭 {e['title'][:55]}...")
        print(f"   📅 {e['date_raw'] or 'N/A'}")
        print(f"   ├─ title:       {'✅' if e['title'] else '❌'}")
        print(f"   ├─ description: {'✅' if e['description'] else '❌'}")
        print(f"   ├─ parsed_date: {'✅' if e['parsed_date'] else '❌'} {e['parsed_date'] or ''}")
        print(f"   ├─ parsed_time: {'✅' if e['parsed_time'] else '❌'} {e['parsed_time'] or ''}")
        print(f"   ├─ price:       {'✅' if e['ticket_price_min'] else '❌'} {e['ticket_price_min'] or 'N/A'} ₽")
        print(f"   ├─ status:      {'✅' if e['ticket_status'] == 'available' else '⚠️'} {e['ticket_status']}")
        print(f"   ├─ photo:       {'✅' if e['photos'] else '❌'}")
        print(f"   └─ location:    {'✅' if e['location'] else '❌'} {e['location']}")
    
    # Stats
    print(f"\n{'='*70}")
    total = len(all_events)
    print(f"📈 STATS: {total} entries")
    print(f"   title:       {sum(1 for e in all_events if e['title'])}/{total}")
    print(f"   description: {sum(1 for e in all_events if e['description'])}/{total}")
    print(f"   date:        {sum(1 for e in all_events if e['parsed_date'])}/{total}")
    print(f"   time:        {sum(1 for e in all_events if e['parsed_time'])}/{total}")
    print(f"   price:       {sum(1 for e in all_events if e['ticket_price_min'])}/{total}")
    print(f"   photo:       {sum(1 for e in all_events if e['photos'])}/{total}")
    print(f"\n✅ Saved to /tmp/tretyakov_final.json")


if __name__ == "__main__":
    asyncio.run(main())
