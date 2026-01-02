"""
ДЕМОНСТРАЦИЯ ДЕДУПЛИКАЦИИ - Фестиваль Pianissimo

Сканируем:
1. Общий фестиваль "ЗИМНИЙ ФЕСТИВАЛЬ PIANISSIMO"
2. Несколько конкретных пианистов

Цель: показать что события пианистов дедуплицируют общий фестиваль
когда даты совпадают.
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


def log(msg, indent=0):
    """Подробный лог с отступами."""
    prefix = "   " * indent
    print(f"{prefix}{msg}")


def deduplicate_events(events):
    """
    Дедупликация: при совпадении (дата, время, зал) оставляем
    событие исполнителя (direct_url_date), удаляем фестиваль (all_dates_extracted).
    
    ВАЖНО: Фото удаляемых событий добавляются к оставленному событию.
    """
    log("\n" + "=" * 70)
    log("🔄 ДЕДУПЛИКАЦИЯ")
    log("=" * 70)
    
    groups = {}
    for e in events:
        key = (e['parsed_date'], e['parsed_time'], e['location'])
        if key not in groups:
            groups[key] = []
        groups[key].append(e)
    
    result = []
    duplicates_removed = 0
    
    for key, group in groups.items():
        date_str, time_str, location = key
        
        if len(group) == 1:
            result.append(group[0])
            log(f"✓ {date_str} {time_str} {location}: единственное событие", 1)
        else:
            log(f"\n⚠️ ДУБЛИКАТ НАЙДЕН: {date_str} {time_str} {location}", 1)
            for e in group:
                photos_count = len(e.get('photos', []))
                log(f"   - '{e['title'][:40]}' (source={e['source']}, photos={photos_count})", 1)
            
            # Предпочитаем direct_url_date (конкретный исполнитель)
            direct = [e for e in group if e['source'] == 'direct_url_date']
            other = [e for e in group if e['source'] != 'direct_url_date']
            
            if direct:
                kept = direct[0].copy()  # Копируем чтобы не мутировать оригинал
                
                # Собираем ВСЕ фото из удаляемых событий
                all_photos = list(kept.get('photos', []))
                for removed in other:
                    for photo in removed.get('photos', []):
                        if photo and photo not in all_photos:
                            all_photos.append(photo)
                            log(f"   📸 Добавлено фото от '{removed['title'][:30]}...'", 1)
                
                kept['photos'] = all_photos
                result.append(kept)
                duplicates_removed += len(group) - 1
                log(f"   → ОСТАВЛЕН: '{kept['title'][:40]}' ({len(all_photos)} фото)", 1)
                log(f"   → УДАЛЁН: остальные {len(group)-1} событий", 1)
            else:
                result.append(group[0])
                duplicates_removed += len(group) - 1
                log(f"   → ОСТАВЛЕН: '{group[0]['title'][:45]}' (первое)", 1)
    
    log(f"\n📊 Итого: удалено {duplicates_removed} дубликатов")
    return result


async def get_price_and_status(page, ticket_url, target_date=None, target_time=None):
    """
    Получить цену и статус с виджета билетов.
    Подробный лог каждого действия.
    """
    log(f"🎫 Открываю виджет билетов: {ticket_url[:60]}...", 2)
    
    try:
        await page.goto(ticket_url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(2000)
        log(f"   Страница загружена", 2)
        
        # Навигация по календарю для поиска даты
        if target_date:
            target_day = str(int(target_date.split('-')[2]))
            target_month_num = int(target_date.split('-')[1])
            month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                          7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
            target_month = month_names.get(target_month_num, '')
            
            log(f"   Ищу дату {target_day} {target_month}...", 2)
            
            for click in range(10):
                # Проверяем видимые даты
                visible = await page.evaluate("""() => {
                    const items = [];
                    document.querySelectorAll('div.item.active').forEach(item => {
                        const dayEl = item.querySelector('.calendarDay');
                        const monthEl = item.querySelector('.calendarMonth');
                        if (dayEl) items.push({ day: dayEl.innerText.trim(), month: monthEl ? monthEl.innerText.trim().toLowerCase() : '' });
                    });
                    return items;
                }""")
                
                found = any(d['day'] == target_day and d['month'] == target_month for d in visible)
                if found:
                    log(f"   ✓ Дата найдена в календаре", 2)
                    break
                
                # Кликаем стрелку
                arrows = await page.locator('button, div, span, a').filter(has_text='→').all()
                if arrows:
                    await arrows[0].click()
                    await page.wait_for_timeout(800)
                    log(f"   → Клик стрелки ({click+1})", 2)
                else:
                    break
            
            # Кликаем на дату
            await page.evaluate(f"""() => {{ 
                document.querySelectorAll('div.item.active').forEach(i => {{ 
                    const dayEl = i.querySelector('.calendarDay'); 
                    if (dayEl && dayEl.innerText.trim() === '{target_day}') i.click(); 
                }}); 
            }}""")
            await page.wait_for_timeout(1500)
            log(f"   ✓ Кликнул на дату {target_day}", 2)
        
        # Кликаем на время
        if target_time:
            await page.evaluate(f"""() => {{ 
                document.querySelectorAll('label.select-time-button').forEach(b => {{ 
                    if (b.innerText.includes('{target_time}')) b.click(); 
                }}); 
            }}""")
            await page.wait_for_timeout(1000)
            log(f"   ✓ Кликнул на время {target_time}", 2)
        
        # Извлекаем цены со ВСЕХ секторов (min и max)
        all_prices = []
        
        sector_labels = await page.query_selector_all('label.select-sector-button')
        if sector_labels:
            log(f"   📍 Найдено секторов: {len(sector_labels)}", 2)
            for i, sector in enumerate(sector_labels):
                try:
                    await sector.click()
                    await page.wait_for_timeout(1000)
                    
                    # Получаем цену из .ticket-price
                    price_el = await page.query_selector('.ticket-price')
                    if price_el:
                        price_text = await price_el.inner_text()
                        match = re.search(r'(\d+)', price_text)
                        if match:
                            price = int(match.group(1))
                            all_prices.append(price)
                            sector_text = await sector.inner_text()
                            log(f"      Сектор {i+1}: {price} ₽ ({sector_text.strip()[:30]})", 2)
                except:
                    pass
        
        # Если секторов нет, пробуем получить цену напрямую
        if not all_prices:
            price_el = await page.query_selector('.ticket-price')
            if price_el:
                price_text = await price_el.inner_text()
                match = re.search(r'(\d+)', price_text)
                if match:
                    all_prices.append(int(match.group(1)))
        
        # Запасной вариант
        if not all_prices:
            prices = await page.evaluate("""() => [...new Set([...document.querySelectorAll('*')].map(e => e.innerText?.match(/(\\d+)\\s*₽/)?.[1]).filter(Boolean).map(Number).filter(p => p > 100))]""")
            all_prices = prices
        
        if all_prices:
            price_min = min(all_prices)
            price_max = max(all_prices)
            log(f"   💰 Цены: {price_min}–{price_max} ₽", 2)
            return (price_min, price_max), "available"
        
        log(f"   💰 Цена: не найдена", 2)
        return (None, None), "unknown"
        
    except Exception as e:
        log(f"   ⚠️ Ошибка: {e}", 2)
        return (None, None), "error"


async def scrape_detail_page(page, detail_url):
    """Получить описание и direct URL с детальной страницы."""
    full_url = f"{BASE_URL}{detail_url}" if detail_url.startswith('/') else detail_url
    log(f"📄 Открываю детальную страницу: {full_url}", 1)
    
    try:
        await page.goto(full_url, timeout=30000, wait_until='domcontentloaded')
        await page.wait_for_timeout(2000)
        
        # Название
        title = None
        h1 = await page.query_selector('h1')
        if h1:
            title = (await h1.inner_text()).strip()
        log(f"   Название: {title[:50] if title else 'N/A'}...", 1)
        
        # Описание
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
        log(f"   Описание: {len(description) if description else 0} символов", 1)
        
        # Дата из текста страницы
        body_text = await page.inner_text("body")
        parsed_date = None
        parsed_time = None
        today = date.today()
        
        for match in re.finditer(r'(\d{1,2})\s+([а-яё]+)\s*,?\s*(?:в|В)\s*(\d{1,2}:\d{2})', body_text, re.IGNORECASE):
            day = int(match.group(1))
            month_name = match.group(2).lower().strip('.,')
            time_str = match.group(3)
            
            month_num = MONTHS_RU.get(month_name)
            if not month_num:
                continue
            
            year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
            try:
                date_obj = date(year, month_num, day)
                if date_obj >= today:
                    parsed_date = date_obj.isoformat()
                    parsed_time = time_str
                    break
            except:
                continue
        
        if parsed_date:
            log(f"   📅 Дата из текста: {parsed_date} {parsed_time}", 1)
        
        # Direct URL кнопки "Купить билет"
        direct_ticket_url = None
        ticket_links = await page.query_selector_all('a[href*="tickets"]')
        for tl in ticket_links:
            href = await tl.get_attribute('href')
            if href and '/buy/event/' in href and re.search(r'/\d{4}-\d{2}-\d{2}/', href):
                direct_ticket_url = href
                log(f"   🎫 Direct URL: {href[:70]}...", 1)
                break
        
        return {
            "title": title,
            "description": description,
            "parsed_date": parsed_date,
            "parsed_time": parsed_time,
            "direct_ticket_url": direct_ticket_url
        }
        
    except Exception as e:
        log(f"   ⚠️ Ошибка: {e}", 1)
        return {"title": None, "description": None, "parsed_date": None, "parsed_time": None, "direct_ticket_url": None}


async def main():
    log("=" * 70)
    log("🎹 ДЕМО ДЕДУПЛИКАЦИИ - Фестиваль PIANISSIMO")
    log(f"📅 Сегодня: {date.today()}")
    log("=" * 70)
    
    all_results = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        await context.route("**/*{google,yandex,metrika,analytics}*", lambda route: route.abort())
        
        list_page = await context.new_page()
        detail_page = await context.new_page()
        ticket_page = await context.new_page()
        
        # Шаг 1: Получить список событий
        log("\n📋 ШАГ 1: Получение списка событий с /events/")
        await list_page.goto(f"{BASE_URL}/events/", timeout=60000, wait_until='domcontentloaded')
        await list_page.wait_for_timeout(3000)
        
        for _ in range(3):
            await list_page.mouse.wheel(0, 3000)
            await list_page.wait_for_timeout(1000)
        
        events = await list_page.evaluate("""() => {
            const events = [];
            const seen = new Set();
            
            document.querySelectorAll('.card').forEach(card => {
                const titleEl = card.querySelector('.card_title');
                if (!titleEl) return;
                
                const title = titleEl.innerText.trim();
                if (title.toUpperCase().includes('ЭКСКУРСИЯ')) return;
                
                let detailUrl = null;
                const onclick = card.getAttribute('onclick');
                if (onclick) {
                    const match = onclick.match(/window\\.open\\(['\"]([^'\"]+)['\"]/);
                    if (match) detailUrl = match[1];
                }
                
                let ticketUrl = null;
                const ticketLink = card.querySelector('a[href*="tickets"]');
                if (ticketLink) {
                    let href = ticketLink.getAttribute('href');
                    if (href.startsWith('//')) ticketUrl = 'https:' + href;
                    else ticketUrl = href;
                }
                
                if (ticketUrl && ticketUrl.includes('timepad')) return;
                
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
                        location: location
                    });
                }
            });
            return events;
        }""")
        
        log(f"   Найдено событий: {len(events)}")
        
        # Фильтруем только PIANISSIMO события
        pianissimo_events = [e for e in events if 'PIANISSIMO' in e['title_raw'].upper()]
        log(f"   Из них PIANISSIMO: {len(pianissimo_events)}")
        
        for e in pianissimo_events:
            log(f"      - {e['title_raw'][:60]}", 1)
        
        # Шаг 2: Обработать каждое событие
        log("\n" + "=" * 70)
        log("📋 ШАГ 2: Обработка каждого события PIANISSIMO")
        log("=" * 70)
        
        for idx, event in enumerate(pianissimo_events):
            log(f"\n{'─'*60}")
            log(f"📌 СОБЫТИЕ {idx+1}/{len(pianissimo_events)}: {event['title_raw'][:55]}...")
            
            # Детальная страница
            detail = await scrape_detail_page(detail_page, event.get('detail_url'))
            title = detail['title'] or event['title_raw']
            description = detail['description']
            
            # Определяем источник даты
            direct_ticket_url = detail.get('direct_ticket_url')
            
            if direct_ticket_url:
                # Событие конкретного исполнителя - берём дату из URL
                url_match = re.search(r'/(\d{4}-\d{2}-\d{2})/(\d{2}:\d{2})', direct_ticket_url)
                if url_match:
                    specific_date = url_match.group(1)
                    specific_time = url_match.group(2)
                    log(f"\n   🎯 КОНКРЕТНЫЙ ИСПОЛНИТЕЛЬ - используем дату из URL: {specific_date} {specific_time}", 0)
                    
                    prices, status = await get_price_and_status(
                        ticket_page, direct_ticket_url,
                        target_date=specific_date, target_time=specific_time
                    )
                    
                    day = int(specific_date.split('-')[2])
                    month_num = int(specific_date.split('-')[1])
                    month_names = {1: 'января', 2: 'февраля', 3: 'марта', 4: 'апреля', 5: 'мая', 6: 'июня',
                                  7: 'июля', 8: 'августа', 9: 'сентября', 10: 'октября', 11: 'ноября', 12: 'декабря'}
                    date_raw = f"{day} {month_names.get(month_num, '')} в {specific_time}"
                    
                    price_min, price_max = prices if isinstance(prices, tuple) else (prices, prices)
                    
                    all_results.append({
                        "title": title,
                        "description": description,
                        "date_raw": date_raw,
                        "parsed_date": specific_date,
                        "parsed_time": specific_time,
                        "ticket_status": status,
                        "ticket_price_min": price_min,
                        "ticket_price_max": price_max,
                        "location": event['location'],
                        "source": "direct_url_date"
                    })
            else:
                # Общий фестиваль - получаем ВСЕ даты
                log(f"\n   📅 ОБЩИЙ ФЕСТИВАЛЬ - получаем все даты из календаря", 0)
                
                ticket_url = event['ticket_url']
                clean_url = re.sub(r'/\d{4}-\d{2}-\d{2}/\d{2}:\d{2}(:\d{2})?$', '', ticket_url)
                full_url = clean_url if not clean_url.startswith('/') else f"{BASE_URL}{clean_url}"
                
                log(f"🎫 Открываю виджет билетов: {full_url[:60]}...", 2)
                await ticket_page.goto(full_url, timeout=60000, wait_until='networkidle')
                await ticket_page.wait_for_timeout(3000)
                
                # Навигация и сбор всех дат
                all_dates = set()
                for click in range(10):
                    visible = await ticket_page.evaluate("""() => {
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
                    
                    arrows = await ticket_page.locator('button, div, span, a').filter(has_text='→').all()
                    if arrows:
                        await arrows[0].click()
                        await ticket_page.wait_for_timeout(800)
                    else:
                        break
                
                log(f"   📅 Найдено активных дат: {len(all_dates)}", 2)
                for d in sorted(all_dates, key=lambda x: (MONTHS_RU.get(x[1], 0), int(x[0]))):
                    log(f"      - {d[0]} {d[1]}", 2)
                
                # Обрабатываем первые 5 дат для демонстрации
                today = date.today()
                dates_processed = 0
                
                await ticket_page.goto(full_url, timeout=60000, wait_until='networkidle')
                await ticket_page.wait_for_timeout(2000)
                
                for (day_str, month_str) in sorted(all_dates, key=lambda x: (MONTHS_RU.get(x[1], 0), int(x[0]))):
                    if dates_processed >= 5:
                        break
                    
                    month_num = MONTHS_RU.get(month_str, 1)
                    year = today.year + (1 if today.month >= 10 and month_num < 3 else 0)
                    try:
                        date_obj = date(year, month_num, int(day_str))
                        if date_obj < today:
                            continue
                    except:
                        continue
                    
                    parsed_date = date_obj.isoformat()
                    
                    log(f"\n   Обработка даты: {day_str} {month_str}", 2)
                    
                    # Кликаем на дату
                    await ticket_page.evaluate(f"""() => {{ 
                        document.querySelectorAll('div.item.active').forEach(i => {{ 
                            const dayEl = i.querySelector('.calendarDay'); 
                            if (dayEl && dayEl.innerText.trim() === '{day_str}') i.click(); 
                        }}); 
                    }}""")
                    await ticket_page.wait_for_timeout(1500)
                    
                    # Получаем времена
                    times = await ticket_page.evaluate("""() => [...document.querySelectorAll('label.select-time-button:not(.disabled)')].map(b => b.innerText.trim().match(/^\\d{1,2}:\\d{2}$/)?.[0]).filter(Boolean)""")
                    
                    if not times:
                        times = ['00:00']
                    
                    log(f"      Времена: {times}", 2)
                    
                    for t in times[:2]:  # Первые 2 времени
                        # Кликаем время
                        await ticket_page.evaluate(f"""() => {{ 
                            document.querySelectorAll('label.select-time-button').forEach(b => {{ 
                                if (b.innerText.includes('{t}')) b.click(); 
                            }}); 
                        }}""")
                        await ticket_page.wait_for_timeout(1000)
                        
                        # Сектор
                        sector_labels = await ticket_page.query_selector_all('label:has-text("Сектор")')
                        if sector_labels:
                            try:
                                await sector_labels[0].click()
                                await ticket_page.wait_for_timeout(800)
                            except:
                                pass
                        
                        # Цена
                        prices = await ticket_page.evaluate("""() => [...new Set([...document.querySelectorAll('*')].map(e => e.innerText?.match(/(\\d+)\\s*₽/)?.[1]).filter(Boolean).map(Number).filter(p => p > 100))]""")
                        price = min(prices) if prices else None
                        status = "available" if prices else "unknown"
                        
                        log(f"      {day_str} {month_str} {t}: цена={price}₽, статус={status}", 2)
                        
                        all_results.append({
                            "title": title,
                            "description": description,
                            "date_raw": f"{day_str} {month_str} в {t}",
                            "parsed_date": parsed_date,
                            "parsed_time": t,
                            "ticket_status": status,
                            "ticket_price_min": price,
                            "location": event['location'],
                            "source": "all_dates_extracted"
                        })
                    
                    dates_processed += 1
        
        await browser.close()
    
    # Шаг 3: Дедупликация
    log("\n" + "=" * 70)
    log(f"ДО ДЕДУПЛИКАЦИИ: {len(all_results)} записей")
    
    final_results = deduplicate_events(all_results)
    
    log(f"\nПОСЛЕ ДЕДУПЛИКАЦИИ: {len(final_results)} записей")
    
    # Итоговый отчёт
    log("\n" + "=" * 70)
    log("📋 ИТОГОВЫЙ ОТЧЁТ")
    log("=" * 70)
    
    for i, r in enumerate(final_results):
        log(f"\n{i+1}. {r['title'][:50]}...")
        log(f"   📅 {r['parsed_date']} {r['parsed_time']}", 1)
        log(f"   💰 Цена: {r['ticket_price_min']} ₽", 1)
        log(f"   🎫 Статус: {r['ticket_status']}", 1)
        log(f"   📍 Зал: {r['location']}", 1)
        log(f"   📊 Источник: {r['source']}", 1)
    
    # Статистика
    log("\n" + "=" * 70)
    log("📈 СТАТИСТИКА")
    with_price = sum(1 for r in final_results if r['ticket_price_min'])
    with_status = sum(1 for r in final_results if r['ticket_status'] == 'available')
    log(f"   Всего: {len(final_results)}")
    log(f"   С ценой: {with_price}")
    log(f"   Доступно: {with_status}")
    
    # Сохраняем
    with open("/tmp/pianissimo_demo.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    log(f"\n✅ Сохранено в /tmp/pianissimo_demo.json")


if __name__ == "__main__":
    asyncio.run(main())
