"""
EXAM: Test Tretyakov Parser on 5 random events.
Uses the SAME algorithm as in Kaggle notebook.
"""
import asyncio
import json
import re
import random
from datetime import date
from playwright.async_api import async_playwright

BASE_URL = "https://kaliningrad.tretyakovgallery.ru"

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
}


async def scrape_events_list(page):
    """Scrape all events from /events/ page."""
    url = f"{BASE_URL}/events/"
    print(f"\n📋 STEP 1: Scraping events from {url}")
    
    await page.goto(url, timeout=60000, wait_until='domcontentloaded')
    await page.wait_for_timeout(3000)
    
    for _ in range(3):
        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(1000)
    
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
                    const match = onclick.match(/window\\.open\\(['\"]([^'\"]+)['\"]/) ;
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
                let location = 'Третьяковка Калининград';
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
    return events


async def scrape_detail_page(page, detail_url):
    """Visit event detail page to get title, description, AND date/time."""
    if not detail_url:
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None}
    
    full_url = f"{BASE_URL}{detail_url}" if detail_url.startswith('/') else detail_url
    print(f"\n   📄 Detail page: {full_url}")
    
    today = date.today()
    
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
        
        # Extract date and time from page text
        body_text = await page.inner_text("body")
        parsed_date = None
        parsed_time = None
        fallback = None
        
        for match in re.finditer(r'(\d{1,2})\s+([а-яё]+)\s*,?\s*(?:в|В)\s*(\d{1,2}:\d{2})', body_text, re.IGNORECASE):
            day = int(match.group(1))
            month_name = match.group(2).lower().strip('.,')
            time_str = match.group(3)
            
            month_num = MONTHS_RU.get(month_name)
            if not month_num:
                continue
            
            year = today.year
            if today.month >= 10 and month_num < 3:
                year += 1
            
            try:
                date_obj = date(year, month_num, day)
            except ValueError:
                continue
            
            if date_obj >= today:
                parsed_date = date_obj.isoformat()
                parsed_time = time_str
                break
            
            if fallback is None:
                fallback = (date_obj, time_str)
        
        if not parsed_date and fallback:
            parsed_date = fallback[0].isoformat()
            parsed_time = fallback[1]
        
        print(f"      Title: {title[:50] if title else 'N/A'}...")
        print(f"      Description: {description[:50] if description else 'N/A'}...")
        if parsed_date:
            print(f"      📅 Detail page date: {parsed_date} {parsed_time}")
        
        return {"title": title, "description": description, "parsed_date": parsed_date, "parsed_time": parsed_time}
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None}


async def parse_ticket_page(page, ticket_url, max_combos=3):
    """Parse ticket page for dates, times, prices."""
    # Clean URL
    raw_url = ticket_url
    if raw_url.startswith(BASE_URL):
        raw_url = raw_url[len(BASE_URL):]
    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', raw_url)
    
    full_url = f"{BASE_URL}{clean_url}" if clean_url.startswith('/') else clean_url
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
        
        for d in active_dates[:3]:
            if len(entries) >= max_combos:
                break
            
            month_num = MONTHS_RU.get(d['month'], 1)
            year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
            
            try:
                date_obj = date(year, month_num, int(d['day']))
                if date_obj < today:
                    continue
            except:
                continue
            
            day_val = d['day']
            await page.evaluate(f"() => {{ document.querySelectorAll('div.item.active').forEach(i => {{ const dayEl = i.querySelector('.calendarDay'); if (dayEl && dayEl.innerText.trim() === '{day_val}') i.click(); }}); }}")
            await page.wait_for_timeout(1500)
            
            times = await page.evaluate("""() => [...document.querySelectorAll('label.select-time-button:not(.disabled)')].map(b => b.innerText.trim().match(/^\\d{1,2}:\\d{2}$/)?.[0]).filter(Boolean)""")
            print(f"         {d['day']} {d['month']}: {len(times)} times ({', '.join(times) or 'none'})")
            
            for t in times[:2]:
                if len(entries) >= max_combos:
                    break
                
                await page.evaluate(f"() => {{ document.querySelectorAll('label.select-time-button').forEach(b => {{ if (b.innerText.includes('{t}')) b.click(); }}); }}")
                await page.wait_for_timeout(1500)
                
                sectors = await page.locator('text=/[Сс]ектор/').all()
                if sectors:
                    try:
                        await sectors[0].click()
                        await page.wait_for_timeout(1000)
                    except:
                        pass
                
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
    print("🖼️ TRETYAKOV PARSER EXAM - 5 RANDOM EVENTS")
    print(f"📅 Today: {date.today()}")
    print("=" * 70)
    
    all_results = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
        
        list_page = await context.new_page()
        detail_page = await context.new_page()
        ticket_page = await context.new_page()
        
        events_raw = await scrape_events_list(list_page)
        
        # Select 5 random events for exam
        if len(events_raw) > 5:
            random.seed(42)  # Reproducible for verification
            selected = random.sample(events_raw, 5)
        else:
            selected = events_raw
        
        print(f"\n🎯 SELECTED 5 EVENTS FOR EXAM:")
        for i, e in enumerate(selected):
            print(f"   {i+1}. {e['title_raw'][:50]}...")
        
        for idx, event in enumerate(selected):
            print(f"\n{'='*70}")
            print(f"📌 EVENT {idx+1}/5: {event['title_raw'][:50]}...")
            
            # Step 1: Get title, description, AND date from detail page
            detail = await scrape_detail_page(detail_page, event.get('detail_url'))
            title = detail['title'] or event['title_raw']
            description = detail['description']
            detail_date = detail.get('parsed_date')
            detail_time = detail.get('parsed_time')
            
            photo = event['photo']
            if photo and photo.startswith('/'):
                photo = f"{BASE_URL}{photo}"
            
            # Step 2: If detail page has date, use it as authoritative
            if detail_date and detail_time:
                print(f"\n   ✅ Using DETAIL PAGE date: {detail_date} {detail_time}")
                
                # Get price from ticket widget for this specific date
                entries = await parse_ticket_page(ticket_page, event['ticket_url'])
                
                def normalize_time(value):
                    if not value:
                        return value
                    parts = value.split(':')
                    if len(parts) != 2:
                        return value
                    try:
                        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
                    except ValueError:
                        return value
                
                price = None
                status = "unknown"
                target_time = normalize_time(detail_time)
                for e in entries:
                    if e['parsed_date'] == detail_date and normalize_time(e['parsed_time']) == target_time:
                        price = e['price']
                        status = e['status']
                        break
                
                # Format date_raw
                day = int(detail_date.split('-')[2])
                month_num = int(detail_date.split('-')[1])
                month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                              7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
                date_raw = f"{day} {month_names.get(month_num, '')} в {detail_time}"
                
                clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', event['ticket_url'])
                base = clean_url if not clean_url.startswith('/') else f"{BASE_URL}{clean_url}"
                direct_url = f"{base}/{detail_date}/{detail_time}:00"
                
                all_results.append({
                    "title": title,
                    "description": description,
                    "date_raw": date_raw,
                    "parsed_date": detail_date,
                    "parsed_time": detail_time,
                    "ticket_status": status,
                    "ticket_price_min": price,
                    "url": direct_url,
                    "photos": [photo] if photo else [],
                    "location": event['location'],
                    "source": "detail_page"
                })
            else:
                # No detail page date - use ticket widget dates
                print(f"\n   ℹ️ No date in detail page, using TICKET WIDGET dates")
                entries = await parse_ticket_page(ticket_page, event['ticket_url'])
                
                if not entries:
                    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', event['ticket_url'])
                    target_url = clean_url if not clean_url.startswith('/') else f"{BASE_URL}{clean_url}"
                    
                    all_results.append({
                        "title": title,
                        "description": description,
                        "date_raw": "",
                        "parsed_date": None,
                        "parsed_time": None,
                        "ticket_status": "unknown",
                        "ticket_price_min": None,
                        "url": target_url,
                        "photos": [photo] if photo else [],
                        "location": event['location'],
                        "source": "no_dates"
                    })
                else:
                    for e in entries:
                        clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', event['ticket_url'])
                        base = clean_url if not clean_url.startswith('/') else f"{BASE_URL}{clean_url}"
                        direct_url = f"{base}/{e['parsed_date']}/{e['parsed_time']}:00"
                        
                        all_results.append({
                            "title": title,
                            "description": description,
                            "date_raw": e['date_raw'],
                            "parsed_date": e['parsed_date'],
                            "parsed_time": e['parsed_time'],
                            "ticket_status": e['status'],
                            "ticket_price_min": e['price'],
                            "url": direct_url,
                            "photos": [photo] if photo else [],
                            "location": event['location'],
                            "source": "ticket_widget"
                        })
        
        await browser.close()
    
    # Save full results
    with open("/tmp/tretyakov_exam.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Print exam report
    print("\n" + "=" * 70)
    print("📋 EXAM RESULTS")
    print("=" * 70)
    
    for i, r in enumerate(all_results):
        print(f"\n{'─'*70}")
        print(f"📌 RESULT {i+1}: {r['title'][:55]}...")
        print(f"   📅 Date: {r['parsed_date'] or 'N/A'}")
        print(f"   ⏰ Time: {r['parsed_time'] or 'N/A'}")
        print(f"   📝 date_raw: {r['date_raw'] or 'N/A'}")
        print(f"   💰 Price: {r['ticket_price_min'] or 'N/A'} ₽")
        print(f"   🎫 Status: {r['ticket_status']}")
        print(f"   📍 Location: {r['location']}")
        print(f"   📸 Photo: {'✅' if r['photos'] else '❌'}")
        print(f"   📝 Description: {'✅' if r['description'] else '❌'} ({len(r['description']) if r['description'] else 0} chars)")
        print(f"   🔗 URL: {r['url'][:70]}...")
        print(f"   📊 Source: {r['source']}")
    
    print(f"\n{'='*70}")
    print(f"📈 SUMMARY:")
    print(f"   Total entries: {len(all_results)}")
    print(f"   With date: {sum(1 for r in all_results if r['parsed_date'])}")
    print(f"   With price: {sum(1 for r in all_results if r['ticket_price_min'])}")
    print(f"   With description: {sum(1 for r in all_results if r['description'])}")
    print(f"\n✅ Full results saved to /tmp/tretyakov_exam.json")


if __name__ == "__main__":
    asyncio.run(main())
