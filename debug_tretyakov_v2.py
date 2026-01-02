"""
IMPROVED Tretyakov Parser v2.
Fixes:
1. Collect ALL paragraphs for description (not just first)
2. Click calendar arrows to navigate and find ALL dates
3. Process each date systematically
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

MAX_ARROW_CLICKS = 10  # Maximum calendar navigation clicks


def deduplicate_events(events):
    """
    Дедупликация событий фестиваля.
    
    Если два события имеют одинаковые (дата, время, зал), оставляем только
    событие с source='direct_url_date' (конкретный исполнитель).
    События с 'all_dates_extracted' (общий фестиваль) удаляются.
    """
    # Группируем по (дата, время, зал)
    groups = {}
    for e in events:
        key = (e['parsed_date'], e['parsed_time'], e['location'])
        if key not in groups:
            groups[key] = []
        groups[key].append(e)
    
    # Для каждой группы выбираем приоритетное событие
    result = []
    duplicates_removed = 0
    
    for key, group in groups.items():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Предпочитаем direct_url_date (конкретный исполнитель)
            direct = [e for e in group if e['source'] == 'direct_url_date']
            if direct:
                result.append(direct[0])
                duplicates_removed += len(group) - 1
                print(f"   🔄 Дедупликация: {key[0]} {key[1]} {key[2]} - оставлен '{direct[0]['title'][:40]}...'")
            else:
                # Если нет direct_url_date, берём первое
                result.append(group[0])
                duplicates_removed += len(group) - 1
    
    if duplicates_removed > 0:
        print(f"\n📊 Дедупликация: удалено {duplicates_removed} дубликатов")
    
    return result


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
    """Visit event detail page to get title, FULL description, and date/time."""
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
        
        # IMPROVED: Collect ALL description paragraphs
        description_parts = []
        paragraphs = await page.query_selector_all('p')
        for p in paragraphs:
            text = (await p.inner_text()).strip()
            # Skip short or boilerplate text
            if len(text) < 30:
                continue
            if any(skip in text.lower() for skip in ['cookie', 'политик', 'hours', 'работаем', 'режим работы']):
                continue
            description_parts.append(text)
        
        # Join all paragraphs into full description
        description = '\n\n'.join(description_parts) if description_parts else None
        
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
        
        # CRITICAL: Extract direct ticket URL from detail page
        # This contains the correct date/time for specific performers (e.g. Pianissimo)
        direct_ticket_url = None
        ticket_links = await page.query_selector_all('a[href*="tickets"]')
        for tl in ticket_links:
            href = await tl.get_attribute('href')
            if href and '/buy/event/' in href and re.search(r'/\d{4}-\d{2}-\d{2}/', href):
                direct_ticket_url = href
                print(f"      🎫 Direct ticket URL: {href}")
                break
        
        print(f"      Title: {title[:50] if title else 'N/A'}...")
        print(f"      Description: {len(description) if description else 0} chars")
        if parsed_date:
            print(f"      📅 Detail page date: {parsed_date} {parsed_time}")
        
        return {
            "title": title, 
            "description": description, 
            "parsed_date": parsed_date, 
            "parsed_time": parsed_time,
            "direct_ticket_url": direct_ticket_url
        }
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None, "direct_ticket_url": None}


async def get_visible_active_dates(page):
    """Get all visible ACTIVE dates from current calendar view."""
    return await page.evaluate("""
        () => {
            const items = [];
            document.querySelectorAll('div.item.active').forEach(item => {
                const dayEl = item.querySelector('.calendarDay');
                const monthEl = item.querySelector('.calendarMonth');
                if (dayEl) {
                    items.push({ 
                        day: dayEl.innerText.trim(), 
                        month: monthEl ? monthEl.innerText.trim().toLowerCase() : '' 
                    });
                }
            });
            return items;
        }
    """)


async def click_calendar_arrow_right(page):
    """Click the right arrow to navigate calendar forward. Returns True if clicked."""
    try:
        # Look for right arrow button (→)
        arrows = await page.locator('button, div, span, a').filter(has_text='→').all()
        if arrows:
            await arrows[0].click()
            await page.wait_for_timeout(1000)
            return True
        
        # Also try common arrow class patterns
        arrow = await page.query_selector('.arrow-right, .next-arrow, [class*="next"], [class*="right-arrow"]')
        if arrow:
            await arrow.click()
            await page.wait_for_timeout(1000)
            return True
            
        return False
    except:
        return False


async def parse_ticket_page_with_navigation(page, ticket_url, target_date=None, target_time=None, max_combos=30):
    """
    Parse ticket page for dates, times, prices.
    IMPROVED: Clicks calendar arrows to find ALL dates.
    If target_date is provided, tries to find that specific date.
    """
    # Clean URL
    raw_url = ticket_url
    if raw_url.startswith(BASE_URL):
        raw_url = raw_url[len(BASE_URL):]
    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', raw_url)
    
    full_url = f"{BASE_URL}{clean_url}" if clean_url.startswith('/') else clean_url
    print(f"\n   🎫 Ticket page: {full_url[:70]}...")
    today = date.today()
    
    try:
        await page.goto(full_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(3000)
        
        # Collect ALL dates by navigating calendar
        all_dates = set()
        seen_dates_snapshot = ""
        arrows_clicked = 0
        
        while arrows_clicked < MAX_ARROW_CLICKS:
            # Get visible active dates
            visible = await get_visible_active_dates(page)
            
            # Create snapshot to detect if new dates appeared
            current_snapshot = str(visible)
            if current_snapshot == seen_dates_snapshot:
                # No new dates, stop navigating
                break
            seen_dates_snapshot = current_snapshot
            
            for d in visible:
                key = f"{d['day']}_{d['month']}"
                all_dates.add((d['day'], d['month']))
            
            # Try to click right arrow for more dates
            clicked = await click_calendar_arrow_right(page)
            if clicked:
                arrows_clicked += 1
                print(f"         → Navigated calendar (click {arrows_clicked})")
            else:
                break
        
        print(f"      📅 Total ACTIVE dates found: {len(all_dates)}")
        
        # If looking for specific date, filter
        if target_date and target_time:
            target_day = int(target_date.split('-')[2])
            target_month_num = int(target_date.split('-')[1])
            month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                          7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
            target_month_name = month_names.get(target_month_num, '')
            
            # Check if target date exists
            found_target = False
            for (day, month) in all_dates:
                if str(day) == str(target_day) and month == target_month_name:
                    found_target = True
                    break
            
            if found_target:
                print(f"      ✅ Found target date {target_date} in calendar!")
                
                # Reload and navigate to click the date and extract price
                await page.goto(full_url, timeout=60000, wait_until='networkidle')
                await page.wait_for_timeout(2000)
                
                # Navigate to the target date
                for _ in range(MAX_ARROW_CLICKS):
                    visible = await get_visible_active_dates(page)
                    date_visible = any(str(d['day']) == str(target_day) and d['month'] == target_month_name for d in visible)
                    if date_visible:
                        break
                    clicked = await click_calendar_arrow_right(page)
                    if not clicked:
                        break
                    await page.wait_for_timeout(500)
                
                # Click the target date
                await page.evaluate(f"""() => {{ 
                    document.querySelectorAll('div.item.active').forEach(i => {{ 
                        const dayEl = i.querySelector('.calendarDay'); 
                        if (dayEl && dayEl.innerText.trim() === '{target_day}') i.click(); 
                    }}); 
                }}""")
                await page.wait_for_timeout(1500)
                
                # Click the target time
                await page.evaluate(f"""() => {{ 
                    document.querySelectorAll('label.select-time-button').forEach(b => {{ 
                        if (b.innerText.includes('{target_time}')) b.click(); 
                    }}); 
                }}""")
                await page.wait_for_timeout(1000)
                
                # Click sector if present - improved selector for checkboxes
                # Look for sector checkboxes like "Сектор 1 (лестница атриума)"
                sector_clicked = False
                try:
                    # Try clicking on sector label/checkbox
                    sector_labels = await page.query_selector_all('[class*="sector"], label:has-text("Сектор"), div:has-text("Сектор 1"), div:has-text("Сектор 2")')
                    for sl in sector_labels[:1]:
                        await sl.click()
                        sector_clicked = True
                        await page.wait_for_timeout(500)
                        break
                    
                    if not sector_clicked:
                        # Try locator method
                        sectors = await page.locator('text=/[Сс]ектор\\s*\\d/').all()
                        if sectors:
                            await sectors[0].click()
                            sector_clicked = True
                            await page.wait_for_timeout(500)
                except:
                    pass
                
                if sector_clicked:
                    await page.wait_for_timeout(800)
                
                # Extract price
                prices = await page.evaluate("""() => [...new Set([...document.querySelectorAll('*')].map(e => e.innerText?.match(/(\\d+)\\s*₽/)?.[1]).filter(Boolean).map(Number).filter(p => p > 100))]""")
                price = min(prices) if prices else None
                status = "available" if prices else "unknown"
                print(f"         💰 Price found: {price} ₽")
                
                return [{
                    "parsed_date": target_date,
                    "parsed_time": target_time,
                    "date_raw": f"{target_day} {target_month_name} в {target_time}",
                    "price": price,
                    "status": status,
                }]
            else:
                print(f"      ⚠️ Target date {target_date} NOT in calendar, using all dates")
        
        # Process all dates
        entries = []
        
        # Reload page to reset calendar
        await page.goto(full_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        
        processed_dates = set()
        arrows_clicked = 0
        
        while arrows_clicked < MAX_ARROW_CLICKS and len(entries) < max_combos:
            visible = await get_visible_active_dates(page)
            
            for d in visible:
                if len(entries) >= max_combos:
                    break
                    
                date_key = f"{d['day']}_{d['month']}"
                if date_key in processed_dates:
                    continue
                processed_dates.add(date_key)
                
                month_num = MONTHS_RU.get(d['month'], 1)
                year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
                
                try:
                    date_obj = date(year, month_num, int(d['day']))
                    if date_obj < today:
                        continue
                except:
                    continue
                
                # Click this date
                day_val = d['day']
                await page.evaluate(f"""() => {{ 
                    document.querySelectorAll('div.item.active').forEach(i => {{ 
                        const dayEl = i.querySelector('.calendarDay'); 
                        if (dayEl && dayEl.innerText.trim() === '{day_val}') i.click(); 
                    }}); 
                }}""")
                await page.wait_for_timeout(1500)
                
                # Get times
                times = await page.evaluate("""() => [...document.querySelectorAll('label.select-time-button:not(.disabled)')].map(b => b.innerText.trim().match(/^\\d{1,2}:\\d{2}$/)?.[0]).filter(Boolean)""")
                
                if not times:
                    times = ['00:00']
                    
                print(f"         {d['day']} {d['month']}: {len(times)} times ({', '.join(times[:3])}{'...' if len(times) > 3 else ''})")
                
                for t in times[:5]:  # Limit times per date
                    if len(entries) >= max_combos:
                        break
                    
                    # Click time
                    await page.evaluate(f"""() => {{ 
                        document.querySelectorAll('label.select-time-button').forEach(b => {{ 
                            if (b.innerText.includes('{t}')) b.click(); 
                        }}); 
                    }}""")
                    await page.wait_for_timeout(1000)
                    
                    # Click sector if present
                    sectors = await page.locator('text=/[Сс]ектор/').all()
                    if sectors:
                        try:
                            await sectors[0].click()
                            await page.wait_for_timeout(800)
                        except:
                            pass
                    
                    # Extract price
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
            
            # Try to navigate to next week
            if len(entries) < max_combos:
                clicked = await click_calendar_arrow_right(page)
                if clicked:
                    arrows_clicked += 1
                else:
                    break
            else:
                break
        
        return entries
        
    except Exception as e:
        print(f"      ⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    print("=" * 70)
    print("🖼️ TRETYAKOV PARSER v2 - IMPROVED")
    print("   - Full description collection")
    print("   - Calendar navigation with arrows")
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
        
        # Select 5 random events including specific test cases
        selected = []
        for e in events_raw:
            t = e['title_raw'].upper()
            if "PIANISSIMO" in t and "АЙЗЕК" in t:
                selected.append(e)
            elif "ДВЕНАДЦАТАЯ НОЧЬ" in t:
                selected.append(e)
            elif "ПРИТАИВШЕГОСЯ ЛЮТА" in t:
                selected.append(e)
        
        # Add more random if needed
        random.seed(42)
        for e in events_raw:
            if len(selected) >= 5:
                break
            if e not in selected:
                selected.append(e)
        
        print(f"\n🎯 TESTING {len(selected)} EVENTS:")
        for i, e in enumerate(selected[:5]):
            print(f"   {i+1}. {e['title_raw'][:50]}...")
        
        for idx, event in enumerate(selected[:5]):
            print(f"\n{'='*70}")
            print(f"📌 EVENT {idx+1}/5: {event['title_raw'][:50]}...")
            
            # Step 1: Get title, FULL description, AND date from detail page
            detail = await scrape_detail_page(detail_page, event.get('detail_url'))
            title = detail['title'] or event['title_raw']
            description = detail['description']
            detail_date = detail.get('parsed_date')
            detail_time = detail.get('parsed_time')
            
            photo = event['photo']
            if photo and photo.startswith('/'):
                photo = f"{BASE_URL}{photo}"
            
            # Get direct ticket URL from detail page (contains correct date for specific performers like Pianissimo)
            direct_ticket_url = detail.get('direct_ticket_url')
            
            # Determine which ticket URL to use
            # For Pianissimo performers, use direct URL (ensures correct event/date)
            # For others, use ticket URL from card
            ticket_url_to_use = direct_ticket_url if direct_ticket_url else event['ticket_url']
            
            # SPECIAL CASE: If we have direct_ticket_url with embedded date (Pianissimo performers)
            # Use ONLY that specific date - don't fetch all dates from shared calendar
            if direct_ticket_url:
                url_match = re.search(r'/(\d{4}-\d{2}-\d{2})/(\d{2}:\d{2})', direct_ticket_url)
                if url_match:
                    specific_date = url_match.group(1)
                    specific_time = url_match.group(2)
                    print(f"\n   🎯 Using SPECIFIC date from direct URL: {specific_date} {specific_time}")
                    
                    # Get price from ticket page with this specific date
                    entries = await parse_ticket_page_with_navigation(
                        ticket_page, direct_ticket_url,
                        target_date=specific_date, target_time=specific_time
                    )
                    
                    day = int(specific_date.split('-')[2])
                    month_num = int(specific_date.split('-')[1])
                    month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                                  7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
                    date_raw = f"{day} {month_names.get(month_num, '')} в {specific_time}"
                    
                    price = entries[0]['price'] if entries else None
                    status = entries[0]['status'] if entries else "unknown"
                    
                    all_results.append({
                        "title": title,
                        "description": description,
                        "date_raw": date_raw,
                        "parsed_date": specific_date,
                        "parsed_time": specific_time,
                        "ticket_status": status,
                        "ticket_price_min": price,
                        "url": direct_ticket_url,
                        "photos": [photo] if photo else [],
                        "location": event['location'],
                        "source": "direct_url_date"
                    })
                    continue  # Skip to next event
            
            # Step 2: Get ALL active dates from ticket widget
            # Each date+time combination becomes a separate event entry
            print(f"\n   🎫 Getting ALL dates from ticket page")
            entries = await parse_ticket_page_with_navigation(
                ticket_page, ticket_url_to_use,
                target_date=None, target_time=None  # Get ALL dates
            )
            
            if entries:
                print(f"      📅 Found {len(entries)} date+time entries")
                for e in entries:
                    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', ticket_url_to_use)
                    if clean_url.startswith('/'):
                        clean_url = f"{BASE_URL}{clean_url}"
                    direct_url = f"{clean_url}/{e['parsed_date']}/{e['parsed_time']}:00"
                    
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
                        "source": "all_dates_extracted"
                    })
            else:
                # No entries found - create one with detail page date if available
                if detail_date and detail_time:
                    day = int(detail_date.split('-')[2])
                    month_num = int(detail_date.split('-')[1])
                    month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                                  7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
                    date_raw = f"{day} {month_names.get(month_num, '')} в {detail_time}"
                    
                    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', ticket_url_to_use)
                    if clean_url.startswith('/'):
                        clean_url = f"{BASE_URL}{clean_url}"
                    direct_url = f"{clean_url}/{detail_date}/{detail_time}:00"
                    
                    all_results.append({
                        "title": title,
                        "description": description,
                        "date_raw": date_raw,
                        "parsed_date": detail_date,
                        "parsed_time": detail_time,
                        "ticket_status": "unknown",
                        "ticket_price_min": None,
                        "url": direct_url,
                        "photos": [photo] if photo else [],
                        "location": event['location'],
                        "source": "detail_page_fallback"
                    })
                else:
                    # No date info at all
                    clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', ticket_url_to_use)
                    if clean_url.startswith('/'):
                        clean_url = f"{BASE_URL}{clean_url}"
                    
                    all_results.append({
                        "title": title,
                        "description": description,
                        "date_raw": "",
                        "parsed_date": None,
                        "parsed_time": None,
                        "ticket_status": "unknown",
                        "ticket_price_min": None,
                        "url": clean_url,
                        "photos": [photo] if photo else [],
                        "location": event['location'],
                        "source": "no_dates"
                    })
        
        await browser.close()
    
    # Deduplicate: remove festival duplicates when specific performer exists
    print("\n" + "=" * 70)
    print("🔄 STEP 3: Deduplication")
    all_results = deduplicate_events(all_results)
    
    # Save full results
    with open("/tmp/tretyakov_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # Print exam report
    print("\n" + "=" * 70)
    print("📋 EXAM RESULTS v2")
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
    
    # Description lengths
    print(f"\n   📝 Description lengths:")
    for r in all_results:
        desc_len = len(r['description']) if r['description'] else 0
        print(f"      {r['title'][:40]}... : {desc_len} chars")
    
    print(f"\n✅ Full results saved to /tmp/tretyakov_v2.json")


if __name__ == "__main__":
    asyncio.run(main())
