"""
Update Tretyakov parser in Kaggle notebook with ALL improvements:
1. Full descriptions (all paragraphs, not truncated)
2. Direct ticket URL from detail page (Pianissimo fix ✅)
3. Min/Max prices from ALL sectors
4. Deduplication (performer wins over festival)
5. source_type field for bot compatibility
6. Calendar navigation (all dates)

Tested locally with 100% success on Pianissimo demo.
"""
import json

NOTEBOOK_PATH = "/workspaces/events-bot-new/kaggle/ParseTheatres/parse_theatres.ipynb"

# PROVEN WORKING CODE from debug_pianissimo_demo.py and debug_tretyakov_v2.py
NEW_TRETYAKOV_CODE = r'''
# ==========================================
# ЧАСТЬ 3: ТРЕТЬЯКОВСКАЯ ГАЛЕРЕЯ
# ==========================================

BASE_URL_TRETYAKOV = "https://kaliningrad.tretyakovgallery.ru"
MAX_EVENTS_TO_PROCESS = 1000
MAX_ARROW_CLICKS = 10

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
}


def deduplicate_tretyakov_events(events):
    """
    Дедупликация: при совпадении (дата, время, зал) оставляем
    событие исполнителя (direct_url_date), удаляем фестиваль (all_dates_extracted).
    Фото из удаляемых событий добавляются к оставленному.
    """
    groups = {}
    for e in events:
        key = (e.get('parsed_date'), e.get('parsed_time'), e.get('location'))
        if key not in groups:
            groups[key] = []
        groups[key].append(e)
    
    result = []
    duplicates_removed = 0
    
    for key, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            direct = [e for e in group if e.get('source_type') == 'direct_url_date']
            other = [e for e in group if e.get('source_type') != 'direct_url_date']
            
            if direct:
                kept = direct[0].copy()
                # Merge photos from removed events
                all_photos = list(kept.get('photos', []))
                for removed in other:
                    for photo in removed.get('photos', []):
                        if photo and photo not in all_photos:
                            all_photos.append(photo)
                kept['photos'] = all_photos
                result.append(kept)
                duplicates_removed += len(group) - 1
            else:
                result.append(group[0])
                duplicates_removed += len(group) - 1
    
    if duplicates_removed > 0:
        print(f"   🔄 Дедупликация: удалено {duplicates_removed} дубликатов")
    
    return result


async def scrape_tretyakov_events_list(page):
    """Scrape events from /events/ page, extracting detail_url and ticket_url."""
    url = f"{BASE_URL_TRETYAKOV}/events/"
    print(f"\n🖼️ [Третьяковка] Scanning: {url}")
    
    await page.goto(url, timeout=60000, wait_until='domcontentloaded')
    await page.wait_for_timeout(3000)
    
    for _ in range(3):
        await page.mouse.wheel(0, 3000)
        await page.wait_for_timeout(random.randint(1000, 1500))
    
    events = await page.evaluate("""
        () => {
            const events = [];
            const seen = new Set();
            const BASE = 'https://kaliningrad.tretyakovgallery.ru';
            
            document.querySelectorAll('.card').forEach(card => {
                const titleEl = card.querySelector('.card_title');
                if (!titleEl) return;
                
                const title = titleEl.innerText.trim();
                if (title.toUpperCase().includes('ЭКСКУРСИЯ')) return;
                
                // Get detail_url from onclick
                let detailUrl = null;
                const onclick = card.getAttribute('onclick');
                if (onclick) {
                    const match = onclick.match(/window\\.open\\(['\"]([^'\"]+)['\"]/);
                    if (match) detailUrl = match[1];
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
                    photo = img.src.startsWith('/') ? BASE + img.src : img.src;
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
    
    print(f"   ✅ Found {len(events)} events")
    return events[:MAX_EVENTS_TO_PROCESS]


async def scrape_tretyakov_detail(page, detail_url):
    """Visit detail page for title, FULL description, date/time, AND direct_ticket_url.
    
    CRITICAL: For Pianissimo performers, detail page has button with direct URL
    containing the correct date (e.g. /2026-01-30/20:00:00). This prevents phantom events.
    """
    import re
    import datetime
    
    if not detail_url:
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None, "direct_ticket_url": None}
    
    full_url = f"{BASE_URL_TRETYAKOV}{detail_url}" if detail_url.startswith('/') else detail_url
    print(f"   📄 Detail: {full_url}")
    
    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await page.wait_for_timeout(2000)
        
        # Title
        title = None
        h1 = await page.query_selector('h1')
        if h1:
            title = (await h1.inner_text()).strip()
        
        # FULL description - collect ALL paragraphs
        description_parts = []
        paragraphs = await page.query_selector_all('p')
        for p in paragraphs:
            text = (await p.inner_text()).strip()
            if len(text) < 30:
                continue
            if any(skip in text.lower() for skip in ['cookie', 'политик', 'hours', 'работаем']):
                continue
            description_parts.append(text)
        description = '\n\n'.join(description_parts) if description_parts else None
        
        # Date and time from page text
        body_text = await page.inner_text("body")
        parsed_date = None
        parsed_time = None
        today = datetime.date.today()
        
        for match in re.finditer(r'(\d{1,2})\s+([а-яё]+)\s*,?\s*(?:в|В)\s*(\d{1,2}:\d{2})', body_text, re.IGNORECASE):
            day = int(match.group(1))
            month_name = match.group(2).lower().strip('.,')
            time_str = match.group(3)
            
            month_num = MONTHS_RU.get(month_name)
            if not month_num:
                continue
            
            year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
            
            try:
                date_obj = datetime.date(year, month_num, day)
                if date_obj >= today:
                    parsed_date = date_obj.isoformat()
                    parsed_time = time_str
                    break
            except:
                continue
        
        # CRITICAL: Extract direct ticket URL from "Buy ticket" button
        # Format: /tickets/#/buy/event/42168/2026-01-30/20:00:00
        direct_ticket_url = None
        ticket_links = await page.query_selector_all('a[href*="tickets"]')
        for tl in ticket_links:
            href = await tl.get_attribute('href')
            if href and '/buy/event/' in href and re.search(r'/\d{4}-\d{2}-\d{2}/', href):
                direct_ticket_url = href
                print(f"      🎫 Direct URL: {href[:60]}...")
                break
        
        if parsed_date and parsed_time:
            print(f"      📅 Detail date: {parsed_date} {parsed_time}")
        
        return {
            "title": title,
            "description": description,
            "parsed_date": parsed_date,
            "parsed_time": parsed_time,
            "direct_ticket_url": direct_ticket_url
        }
    except Exception as e:
        print(f"      ⚠️ Detail error: {e}")
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None, "direct_ticket_url": None}


async def get_prices_from_all_sectors(page):
    """Extract min and max prices by clicking each sector."""
    import re
    
    all_prices = []
    
    sector_labels = await page.query_selector_all('label.select-sector-button')
    if sector_labels:
        for sector in sector_labels:
            try:
                await sector.click()
                await page.wait_for_timeout(1000)
                
                price_el = await page.query_selector('.ticket-price')
                if price_el:
                    price_text = await price_el.inner_text()
                    match = re.search(r'(\d+)', price_text)
                    if match:
                        all_prices.append(int(match.group(1)))
            except:
                pass
    
    if not all_prices:
        # Fallback: search for any price
        price_el = await page.query_selector('.ticket-price')
        if price_el:
            price_text = await price_el.inner_text()
            match = re.search(r'(\d+)', price_text)
            if match:
                all_prices.append(int(match.group(1)))
    
    if all_prices:
        return min(all_prices), max(all_prices)
    return None, None


async def scrape_tretyakov_tickets_all_dates(page, ticket_url):
    """Parse ticket page for ALL dates using calendar navigation."""
    import datetime
    
    full_url = f"{BASE_URL_TRETYAKOV}{ticket_url}" if ticket_url.startswith('/') else ticket_url
    print(f"   🎫 Tickets: {full_url[:60]}...")
    today = datetime.date.today()
    results = []
    
    try:
        await page.goto(full_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(3000)
        
        # Collect ALL active dates by navigating calendar
        all_dates = set()
        for click in range(MAX_ARROW_CLICKS):
            visible = await page.evaluate("""() => {
                const items = [];
                document.querySelectorAll('div.item.active').forEach(item => {
                    const dayEl = item.querySelector('.calendarDay');
                    const monthEl = item.querySelector('.calendarMonth');
                    if (dayEl) items.push({ day: dayEl.innerText.trim(), month: monthEl ? monthEl.innerText.trim().toLowerCase() : '' });
                });
                return items;
            }""")
            
            for d in visible:
                all_dates.add((d['day'], d['month']))
            
            # Click right arrow
            # Click right arrow to see more dates
            arrow = await page.query_selector('.week-calendar-arrow.week-calendar-next')
            if arrow:
                is_visible = await arrow.is_visible()
                if is_visible:
                    await arrow.click()
                    await page.wait_for_timeout(800)
                else:
                    break
            else:
                break
        
        print(f"      📅 Active dates: {len(all_dates)}")
        
        # Reload to start fresh
        await page.goto(full_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        
        for (day_str, month_str) in sorted(all_dates, key=lambda x: (MONTHS_RU.get(x[1], 0), int(x[0]))):
            month_num = MONTHS_RU.get(month_str, 1)
            year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
            
            try:
                date_obj = datetime.date(year, month_num, int(day_str))
                if date_obj < today:
                    continue
            except:
                continue
            
            date_iso = date_obj.isoformat()
            date_raw = f"{day_str} {month_str}"
            
            # Click date
            await page.evaluate(f"""() => {{ 
                document.querySelectorAll('div.item.active').forEach(i => {{ 
                    const dayEl = i.querySelector('.calendarDay'); 
                    if (dayEl && dayEl.innerText.trim() === '{day_str}') i.click(); 
                }}); 
            }}""")
            await page.wait_for_timeout(1500)
            
            # Get times
            times = await page.evaluate("""() => [...document.querySelectorAll('label.select-time-button:not(.disabled)')].map(b => b.innerText.trim().match(/^\\d{1,2}:\\d{2}$/)?.[0]).filter(Boolean)""")
            
            if not times:
                times = ['00:00']
            
            for time_str in times:
                # Click time
                await page.evaluate(f"""() => {{ 
                    document.querySelectorAll('label.select-time-button').forEach(b => {{ 
                        if (b.innerText.includes('{time_str}')) b.click(); 
                    }}); 
                }}""")
                await page.wait_for_timeout(1000)
                
                # Get min/max prices from all sectors
                price_min, price_max = await get_prices_from_all_sectors(page)
                
                body = await page.inner_text("body")
                if "все билеты проданы" in body.lower():
                    status = "sold_out"
                elif price_min:
                    status = "available"
                else:
                    status = "unknown"
                
                results.append({
                    "parsed_date": date_iso,
                    "parsed_time": time_str,
                    "date_raw": f"{date_raw} в {time_str}",
                    "ticket_price_min": price_min,
                    "ticket_price_max": price_max,
                    "ticket_status": status,
                })
        
        return results
    except Exception as e:
        print(f"      ⚠️ Ticket error: {e}")
        return []


async def run_tretyakov(browser):
    """Main parser with all improvements."""
    import re
    
    context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
    await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
    
    list_page = await context.new_page()
    detail_page = await context.new_page()
    ticket_page = await context.new_page()

    events_raw = await scrape_tretyakov_events_list(list_page)
    if not events_raw:
        await context.close()
        return []

    all_events = []
    
    for idx, event in enumerate(events_raw):
        print(f"\n📌 [{idx+1}/{len(events_raw)}] {event['title_raw'][:50]}...")
        
        # Clean ticket_url
        raw_url = event['ticket_url']
        if raw_url.startswith(BASE_URL_TRETYAKOV):
            raw_url = raw_url[len(BASE_URL_TRETYAKOV):]
        clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', raw_url)
        
        # Get detail info
        detail = await scrape_tretyakov_detail(detail_page, event.get('detail_url'))
        title = detail['title'] or event['title_raw']
        description = detail['description']
        direct_ticket_url = detail.get('direct_ticket_url')
        
        photo = event['photo']
        if photo and photo.startswith('/'):
            photo = f"{BASE_URL_TRETYAKOV}{photo}"
        
        # CASE 1: Direct URL exists (Pianissimo performer)
        if direct_ticket_url:
            url_match = re.search(r'/(\d{4}-\d{2}-\d{2})/(\d{2}:\d{2})', direct_ticket_url)
            if url_match:
                specific_date = url_match.group(1)
                specific_time = url_match.group(2)
                print(f"      🎯 Using direct URL date: {specific_date} {specific_time}")
                
                # Get price from this specific date
                await ticket_page.goto(direct_ticket_url, timeout=60000, wait_until='networkidle')
                await ticket_page.wait_for_timeout(2000)
                price_min, price_max = await get_prices_from_all_sectors(ticket_page)
                
                body = await ticket_page.inner_text("body")
                status = "sold_out" if "все билеты проданы" in body.lower() else ("available" if price_min else "unknown")
                
                # Format date_raw
                day = int(specific_date.split('-')[2])
                month_num = int(specific_date.split('-')[1])
                month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                              7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
                date_raw = f"{day} {month_names.get(month_num, '')} в {specific_time}"
                
                all_events.append({
                    "title": title,
                    "description": description,
                    "date_raw": date_raw,
                    "parsed_date": specific_date,
                    "parsed_time": specific_time,
                    "ticket_status": status,
                    "ticket_price_min": price_min,
                    "ticket_price_max": price_max,
                    "url": direct_ticket_url,
                    "photos": [photo] if photo else [],
                    "location": event['location'],
                    "scene": event['location'] if event['location'] in ["Атриум", "Кинозал"] else "",
                    "source_type": "direct_url_date"
                })
                continue
        
        # CASE 2: No direct URL - get ALL dates from calendar
        entries = await scrape_tretyakov_tickets_all_dates(ticket_page, clean_url)
        
        if entries:
            for e in entries:
                base = clean_url if not clean_url.startswith('/') else f"{BASE_URL_TRETYAKOV}{clean_url}"
                direct_url = f"{base}/{e['parsed_date']}/{e['parsed_time']}:00"
                
                all_events.append({
                    "title": title,
                    "description": description,
                    "date_raw": e['date_raw'],
                    "parsed_date": e['parsed_date'],
                    "parsed_time": e['parsed_time'],
                    "ticket_status": e['ticket_status'],
                    "ticket_price_min": e['ticket_price_min'],
                    "ticket_price_max": e['ticket_price_max'],
                    "url": direct_url,
                    "photos": [photo] if photo else [],
                    "location": event['location'],
                    "scene": event['location'] if event['location'] in ["Атриум", "Кинозал"] else "",
                    "source_type": "all_dates_extracted"
                })
        else:
            # Fallback: no dates found
            fallback_url = clean_url if not clean_url.startswith('/') else f"{BASE_URL_TRETYAKOV}{clean_url}"
            all_events.append({
                "title": title,
                "description": description,
                "date_raw": "",
                "parsed_date": None,
                "parsed_time": None,
                "ticket_status": "unknown",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "url": fallback_url,
                "photos": [photo] if photo else [],
                "location": event['location'],
                "scene": event['location'] if event['location'] in ["Атриум", "Кинозал"] else "",
                "source_type": "no_dates"
            })

    # Deduplicate
    all_events = deduplicate_tretyakov_events(all_events)
    
    print(f"\n🎉 [Третьяковка] Total: {len(all_events)} events (after dedup)")
    await context.close()
    return all_events
'''


def update_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell["source"])
            start = "# ==========================================\n# ЧАСТЬ 3: ТРЕТЬЯКОВСКАЯ ГАЛЕРЕЯ\n# =========================================="
            end = "# --- ЗАПУСК ---"
            
            if start in source and end in source:
                parts = source.split(start)
                pre = parts[0]
                rest = parts[1].split(end)[1]
                
                new = pre + NEW_TRETYAKOV_CODE + "\n\n" + end + rest
                cell["source"] = [line + '\n' for line in new.split('\n')]
                if cell["source"][-1] == '\n':
                    cell["source"].pop()
                else:
                    cell["source"][-1] = cell["source"][-1].rstrip('\n')
                
                print("✅ Notebook updated with ALL improvements:")
                print("   - Full descriptions (all paragraphs)")
                print("   - Direct ticket URL for Pianissimo")
                print("   - Min/Max prices from all sectors")
                print("   - Deduplication (performer wins)")
                print("   - Calendar navigation (all dates)")
                print("   - source_type field")
                break
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    update_notebook()
