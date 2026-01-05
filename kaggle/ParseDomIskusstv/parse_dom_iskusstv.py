"""
Kaggle notebook для парсинга событий со спецпроектов сайта Дом искусств.
С защитой от изменения селекторов (fallback strategies).

Входные данные: URL спецпроекта через переменную DOM_ISKUSSTV_URLS
Выходные данные: dom_iskusstv_events.json
"""

import asyncio
import os
import subprocess
import sys
import json
import re
from datetime import date
from typing import Optional


def install_libs():
    try:
        import playwright
        import bs4
    except ImportError:
        print("⏳ Устанавливаем библиотеки...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "playwright", "beautifulsoup4", "pandas"])
        os.system("playwright install chromium")
        os.system("playwright install-deps")
        print("✅ Библиотеки готовы.")

install_libs()

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd


BASE_DOMAIN = "https://xn--b1admiilxbaki.xn--p1ai"

MONTHS_RU = {
    "января": 1, "янв": 1, "ЯНВАРЯ": 1,
    "февраля": 2, "фев": 2, "ФЕВРАЛЯ": 2,
    "марта": 3, "мар": 3, "МАРТА": 3,
    "апреля": 4, "апр": 4, "АПРЕЛЯ": 4,
    "мая": 5, "МАЯ": 5,
    "июня": 6, "июн": 6, "ИЮНЯ": 6,
    "июля": 7, "июл": 7, "ИЮЛЯ": 7,
    "августа": 8, "авг": 8, "АВГУСТА": 8,
    "сентября": 9, "сен": 9, "СЕНТЯБРЯ": 9,
    "октября": 10, "окт": 10, "ОКТЯБРЯ": 10,
    "ноября": 11, "ноя": 11, "НОЯБРЯ": 11,
    "декабря": 12, "дек": 12, "ДЕКАБРЯ": 12,
}

# === RESILIENT SELECTORS CONFIG ===
IMAGE_SELECTORS = [
    'img[data-original]',
    '.t-cover__carrier[data-bgimage]',
    '.t-bgimg[data-original]',
    '.t-img img',
    '.t396__elem img',
    '.tn-atom__img',
    'img[src*="tildacdn"]',
    'img[src*="upload"]',
    '[style*="background-image"]',
    'img.lazyload',
    'img[data-src]',
    'img',
]

DESCRIPTION_SELECTORS = [
    '.t-text',
    '.t-descr',
    '.t396__elem p',
    '.t456__text',
    '.tn-atom p',
    '[data-elem-type="text"] p',
    '.t-card__descr',
    'article p',
    '.content p',
    '.description',
    '[class*="descr"] p',
]

TITLE_SELECTORS = [
    '.t-cover__title',
    'h1.t-title',
    '.t-title',
    '.tn-atom h1',
    '[data-elem-type="title"]',
    '.t396__elem h1',
    'h1',
    '.title',
    'header h1',
]

PRICE_PATTERNS = [
    r'(\d[\d\s]*)\s*(?:₽|руб\.?|rub)',
    r'цена[:\s]*(\d[\d\s]*)',
    r'от\s*(\d[\d\s]*)\s*(?:₽|руб)',
    r'(\d{3,5})\s*₽',
]

SKIP_IMAGE_PATTERNS = ['logo', 'icon', 'favicon', '.svg', '-s.jpg', '_s.jpg', 'pixel', 'tracking']
SKIP_DESCRIPTION_PATTERNS = ['cookie', 'политик', '@', '+7', 'http', 'vk.com', 'instagram']


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def parse_date_time(date_str: str) -> tuple[Optional[str], Optional[str]]:
    if not date_str:
        return None, None
    text = date_str.strip()
    match = re.search(r'(\d{1,2})\s+([А-ЯЁа-яё]+)\s+(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        time_str = match.group(3)
        month_num = MONTHS_RU.get(month_str) or MONTHS_RU.get(month_str.upper()) or MONTHS_RU.get(month_str.lower())
        if month_num:
            today = date.today()
            year = today.year
            if today.month >= 10 and month_num < 3:
                year += 1
            try:
                event_date = date(year, month_num, day)
                return event_date.isoformat(), time_str
            except ValueError:
                pass
    return None, None


def is_valid_image_url(url: str) -> bool:
    if not url:
        return False
    url_lower = url.lower()
    return not any(skip in url_lower for skip in SKIP_IMAGE_PATTERNS)


def is_valid_description(text: str) -> bool:
    if not text or len(text) < 50:
        return False
    text_lower = text.lower()
    return not any(skip in text_lower for skip in SKIP_DESCRIPTION_PATTERNS)


async def extract_images_resilient(page, content: str) -> list[str]:
    photos = []
    for selector in IMAGE_SELECTORS:
        try:
            elements = await page.query_selector_all(selector)
            for elem in elements:
                url = None
                for attr in ['data-original', 'data-bgimage', 'data-src', 'src']:
                    url = await elem.get_attribute(attr)
                    if url and is_valid_image_url(url):
                        break
                if not url:
                    style = await elem.get_attribute('style') or ''
                    bg_match = re.search(r'url\(["\']?([^"\']+)["\']?\)', style)
                    if bg_match:
                        url = bg_match.group(1)
                if url and is_valid_image_url(url) and url not in photos:
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        url = BASE_DOMAIN + url
                    photos.append(url)
            if len(photos) >= 5:
                break
        except Exception:
            continue
    return photos[:5]


async def extract_description_resilient(page) -> str:
    description_parts = []
    for selector in DESCRIPTION_SELECTORS:
        try:
            elements = await page.query_selector_all(selector)
            for elem in elements:
                text = await elem.inner_text()
                text = clean_text(text)
                if is_valid_description(text) and text not in description_parts:
                    description_parts.append(text)
            if len(description_parts) >= 3:
                break
        except Exception:
            continue
    return '\n\n'.join(description_parts[:3])[:1500]


async def extract_title_resilient(page, soup) -> str:
    for selector in TITLE_SELECTORS:
        try:
            elem = await page.query_selector(selector)
            if elem:
                text = clean_text(await elem.inner_text())
                # Filter out date patterns from title
                if text and len(text) > 3:
                    # Remove leading date if present
                    text = re.sub(r'^\d{1,2}\s+[А-ЯЁа-яё]+\s+\d{1,2}:\d{2}\s*', '', text)
                    if text and len(text) > 3:
                        return text
        except Exception:
            continue
    title_tag = soup.find('title')
    if title_tag:
        return clean_text(title_tag.get_text().split('|')[0].split('—')[0])
    return ""


def extract_prices_resilient(content: str, body_text: str) -> tuple[Optional[int], Optional[int]]:
    prices = []
    for pattern in PRICE_PATTERNS:
        try:
            matches = re.findall(pattern, content + ' ' + body_text, re.IGNORECASE)
            for m in matches:
                try:
                    val = int(re.sub(r'\s', '', m))
                    if 100 <= val <= 50000:
                        prices.append(val)
                except ValueError:
                    pass
        except Exception:
            continue
    if prices:
        return min(prices), max(prices)
    return None, None


async def parse_special_project_page(page, url: str) -> list[dict]:
    print(f"\n🏛 Загрузка: {url}")
    
    try:
        await page.goto(url, timeout=60000, wait_until='networkidle')
        await page.wait_for_timeout(3000)
        
        for _ in range(5):
            await page.mouse.wheel(0, 500)
            await page.wait_for_timeout(500)
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        title = await extract_title_resilient(page, soup)
        print(f"   📌 Название: {title}")
        
        photos = await extract_images_resilient(page, content)
        print(f"   🖼 Фото: {len(photos)}")
        
        description = await extract_description_resilient(page)
        print(f"   📝 Описание: {len(description)} символов")
        
        age_restriction = ""
        age_match = re.search(r'(\d+)\s*\+', content)
        if age_match:
            age_restriction = f"{age_match.group(1)}+"
        
        events: list[dict] = []
        date_pattern = re.compile(
            r'(\d{1,2})\s+(ЯНВАРЯ|ФЕВРАЛЯ|МАРТА|АПРЕЛЯ|МАЯ|ИЮНЯ|ИЮЛЯ|АВГУСТА|СЕНТЯБРЯ|ОКТЯБРЯ|НОЯБРЯ|ДЕКАБРЯ)\s+(\d{1,2}:\d{2})',
            re.IGNORECASE
        )
        
        ticket_links = soup.find_all('a', href=re.compile(r'unifd-performance-id=\d+'))
        print(f"   🎫 Найдено ссылок на билеты: {len(ticket_links)}")
        
        seen_dates = set()
        
        for link in ticket_links:
            href = link.get('href', '')
            perf_match = re.search(r'unifd-performance-id=(\d+)', href)
            if not perf_match:
                continue
            
            perf_id = perf_match.group(1)
            ticket_url = f"{BASE_DOMAIN}/?unifd-performance-id={perf_id}"
            
            date_raw = ""
            parent = link.parent
            for _ in range(7):
                if parent is None:
                    break
                parent_text = parent.get_text(" ", strip=True)
                date_match = date_pattern.search(parent_text)
                if date_match:
                    date_raw = f"{date_match.group(1)} {date_match.group(2)} {date_match.group(3)}"
                    break
                parent = parent.parent
            
            dedup_key = (date_raw, perf_id)
            if dedup_key in seen_dates:
                continue
            seen_dates.add(dedup_key)
            
            parsed_date, parsed_time = parse_date_time(date_raw)
            print(f"   📅 {date_raw} -> {parsed_date} {parsed_time} (ID: {perf_id})")
            
            events.append({
                "title": title,
                "date_raw": date_raw,
                "parsed_date": parsed_date,
                "parsed_time": parsed_time,
                "location": "Дом искусств",
                "url": ticket_url,
                "ticket_status": "unknown",
                "ticket_price_min": None,
                "ticket_price_max": None,
                "photos": photos.copy(),
                "description": description,
                "age_restriction": age_restriction,
                "source_type": "dom_iskusstv"
            })
        
        print(f"\n✅ Найдено событий: {len(events)}")
        return events
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return []


async def check_ticket_availability(page, ticket_url: str) -> tuple[str, Optional[int], Optional[int]]:
    print(f"   🔍 Проверка: {ticket_url[:60]}...")
    
    try:
        await page.goto(ticket_url, timeout=45000, wait_until='networkidle')
        await page.wait_for_timeout(5000)
        
        try:
            await page.wait_for_selector('iframe, [class*="widget"], [class*="ticket"]', timeout=5000)
        except Exception:
            pass
        
        frames = page.frames
        widget_content = ""
        for frame in frames:
            try:
                frame_content = await frame.content()
                widget_content += frame_content
            except Exception:
                pass
        
        content = await page.content()
        all_content = content + widget_content
        body_text = await page.inner_text("body")
        body_lower = body_text.lower()
        
        status = "unknown"
        if any(sold in body_lower for sold in ["билетов нет", "sold out", "распродано"]):
            status = "sold_out"
        elif any(avail in body_lower for avail in ["купить", "в корзину", "выбрать"]):
            status = "available"
        
        price_min, price_max = extract_prices_resilient(all_content, body_text)
        
        if price_min and status == "unknown":
            status = "available"
        
        print(f"      Status: {status}, Price: {price_min}-{price_max}")
        return status, price_min, price_max
        
    except Exception as e:
        print(f"      ⚠️ Ошибка: {e}")
        return "unknown", None, None


async def main():
    urls = []
    # Priority 1: urls.json (injected by bot)
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urls.json")
    if os.path.exists("urls.json"):
        json_path = "urls.json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                urls = json.load(f)
            print(f"📂 Загружено {len(urls)} URL из {json_path}")
        except Exception as e:
            print(f"⚠️ Ошибка чтения urls.json: {e}")
            urls = []
            
    # Priority 2: ENV variable
    if not urls:
        urls_env = os.environ.get("DOM_ISKUSSTV_URLS", "")
        if urls_env:
            urls = [u.strip() for u in urls_env.split(",") if u.strip()]
            
    # Fallback: Default test URL
    if not urls:
        urls = ["https://xn--b1admiilxbaki.xn--p1ai/skazka"]
    
    print(f"📋 Парсинг {len(urls)} URL(s)...")
    print(f"🛡 Режим: с защитой от изменения селекторов")
    all_events = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        ticket_page = await context.new_page()
        
        for url in urls:
            events = await parse_special_project_page(page, url)
            
            for event in events:
                if "unifd-performance-id" in event["url"]:
                    status, price_min, price_max = await check_ticket_availability(ticket_page, event["url"])
                    event["ticket_status"] = status
                    event["ticket_price_min"] = price_min
                    event["ticket_price_max"] = price_max
            
            all_events.extend(events)
        
        await browser.close()
    
    print("\n" + "=" * 60)
    print(f"🎉 Результат: {len(all_events)} событий")
    
    for i, event in enumerate(all_events, 1):
        print(f"\n{i}. {event['title']}")
        print(f"   📅 {event['date_raw']} -> {event['parsed_date']} {event['parsed_time']}")
        print(f"   🎫 {event['ticket_status']} | {event['ticket_price_min']}-{event['ticket_price_max']} ₽")
    
    if all_events:
        df = pd.DataFrame(all_events)
        df.to_json('dom_iskusstv_events.json', orient='records', force_ascii=False, indent=2)
        print("\n💾 Сохранено в dom_iskusstv_events.json")
    else:
        with open('dom_iskusstv_events.json', 'w', encoding='utf-8') as f:
            json.dump([], f)
        print("⚠️ Создан пустой файл.")


asyncio.get_event_loop().run_until_complete(main())
