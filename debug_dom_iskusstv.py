"""
Debug script для тестирования парсинга событий с сайта Дом искусств.
С защитой от изменения селекторов (fallback strategies).

Использование:
    python debug_dom_iskusstv.py [URL]

По умолчанию парсит: https://xn--b1admiilxbaki.xn--p1ai/skazka
"""

import asyncio
import json
import re
import sys
from datetime import date, datetime
from dataclasses import dataclass, asdict
from typing import Optional

# --- 1. УСТАНОВКА ---
def install_libs():
    try:
        import playwright
        import bs4
    except ImportError:
        import subprocess
        print("⏳ Устанавливаем библиотеки...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "playwright", "beautifulsoup4"])
        import os
        os.system("playwright install chromium")
        print("✅ Библиотеки готовы.")

install_libs()

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


BASE_DOMAIN = "https://xn--b1admiilxbaki.xn--p1ai"

# Russian months mapping
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
# Multiple selectors per category - if one breaks, others may work
# Ordered by priority (most specific first)

IMAGE_SELECTORS = [
    # Tilda-specific (current)
    'img[data-original]',
    '.t-cover__carrier[data-bgimage]',
    '.t-bgimg[data-original]',
    '.t-img img',
    '.t396__elem img',
    '.tn-atom__img',
    # Generic fallbacks
    'img[src*="tildacdn"]',
    'img[src*="upload"]',
    '[style*="background-image"]',
    'img.lazyload',
    'img[data-src]',
    'picture source',
    'img',  # Last resort
]

DESCRIPTION_SELECTORS = [
    # Tilda-specific (current)
    '.t-text',
    '.t-descr',
    '.t396__elem p',
    '.t456__text',
    '.tn-atom p',
    '[data-elem-type="text"] p',
    '.t-card__descr',
    # Generic fallbacks
    'article p',
    '.content p',
    '.description',
    '.text-block p',
    'main p',
    '[class*="descr"] p',
    '[class*="text"] p',
]

TITLE_SELECTORS = [
    # Most specific first (Tilda)
    '.t-cover__title',
    'h1.t-title',
    '.t-title',
    '.tn-atom h1',
    '[data-elem-type="title"]',
    '.t396__elem h1',
    # Generic h1 (reliable)
    'h1',
    # Fallbacks (may include unwanted text)
    '.title',
    'header h1',
]

PRICE_PATTERNS = [
    r'(\d[\d\s]*)\s*(?:₽|руб\.?|rub)',
    r'цена[:\s]*(\d[\d\s]*)',
    r'стоимость[:\s]*(\d[\d\s]*)',
    r'от\s*(\d[\d\s]*)\s*(?:₽|руб)',
    r'билет[ы]?[:\s]*(\d[\d\s]*)',
    r'(\d{3,5})\s*₽',  # Simple price pattern
    r'(\d{3,5})\s*руб',
]

# Skip patterns for filtering
SKIP_IMAGE_PATTERNS = ['logo', 'icon', 'favicon', '.svg', '-s.jpg', '_s.jpg', 'pixel', 'tracking', 'blank']
SKIP_DESCRIPTION_PATTERNS = ['cookie', 'политик', '@', '+7', 'телефон', 'http', 'vk.com', 'instagram', 'telegram', 'facebook', 'whatsapp']


@dataclass
class ParsedEvent:
    """Parsed event from Dom Iskusstv."""
    title: str
    date_raw: str
    parsed_date: Optional[str]
    parsed_time: Optional[str]
    location: str
    url: str
    ticket_status: str
    ticket_price_min: Optional[int]
    ticket_price_max: Optional[int]
    photos: list[str]
    description: str
    age_restriction: str
    source_type: str = "dom_iskusstv"
    # Metadata for debugging
    selectors_used: dict = None


def parse_date_time(date_str: str) -> tuple[Optional[str], Optional[str]]:
    """Parse Russian date string like '3 ЯНВАРЯ 14:00' into ISO date and time."""
    if not date_str:
        return None, None
    
    text = date_str.strip()
    
    # Pattern: "3 ЯНВАРЯ 14:00"
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


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def is_valid_image_url(url: str) -> bool:
    """Check if URL is a valid image (not icon/logo/tracking)."""
    if not url:
        return False
    url_lower = url.lower()
    return not any(skip in url_lower for skip in SKIP_IMAGE_PATTERNS)


def is_valid_description(text: str) -> bool:
    """Check if text is valid description (not navigation/contact)."""
    if not text or len(text) < 50:
        return False
    text_lower = text.lower()
    return not any(skip in text_lower for skip in SKIP_DESCRIPTION_PATTERNS)


async def extract_images_resilient(page, content: str) -> tuple[list[str], list[str]]:
    """
    Extract images using multiple fallback selectors.
    Returns (photos, selectors_that_worked).
    """
    photos = []
    working_selectors = []
    
    for selector in IMAGE_SELECTORS:
        try:
            elements = await page.query_selector_all(selector)
            if not elements:
                continue
                
            found_new = False
            for elem in elements:
                # Try multiple attributes
                url = None
                for attr in ['data-original', 'data-bgimage', 'data-src', 'src']:
                    url = await elem.get_attribute(attr)
                    if url and is_valid_image_url(url):
                        break
                
                # Try style background-image
                if not url:
                    style = await elem.get_attribute('style') or ''
                    bg_match = re.search(r'url\(["\']?([^"\']+)["\']?\)', style)
                    if bg_match:
                        url = bg_match.group(1)
                
                if url and is_valid_image_url(url) and url not in photos:
                    # Normalize URL
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        url = BASE_DOMAIN + url
                    photos.append(url)
                    found_new = True
            
            if found_new:
                working_selectors.append(selector)
                
            # Stop after finding enough images
            if len(photos) >= 5:
                break
                
        except Exception as e:
            # Selector failed, try next
            continue
    
    # Fallback: extract from HTML with regex if no images found
    if not photos:
        img_urls = re.findall(r'(https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp))', content, re.IGNORECASE)
        for url in img_urls[:5]:
            if is_valid_image_url(url) and url not in photos:
                photos.append(url)
        if photos:
            working_selectors.append('regex_fallback')
    
    return photos[:5], working_selectors


async def extract_description_resilient(page) -> tuple[str, list[str]]:
    """
    Extract description using multiple fallback selectors.
    Returns (description, selectors_that_worked).
    """
    description_parts = []
    working_selectors = []
    
    for selector in DESCRIPTION_SELECTORS:
        try:
            elements = await page.query_selector_all(selector)
            if not elements:
                continue
                
            found_new = False
            for elem in elements:
                text = await elem.inner_text()
                text = clean_text(text)
                
                if is_valid_description(text) and text not in description_parts:
                    description_parts.append(text)
                    found_new = True
            
            if found_new:
                working_selectors.append(selector)
                
            # Stop after finding enough paragraphs
            if len(description_parts) >= 3:
                break
                
        except Exception:
            continue
    
    full_desc = '\n\n'.join(description_parts[:3])
    return full_desc[:1500], working_selectors


async def extract_title_resilient(page, soup) -> tuple[str, str]:
    """
    Extract title using multiple fallback selectors.
    Returns (title, selector_that_worked).
    """
    for selector in TITLE_SELECTORS:
        try:
            elem = await page.query_selector(selector)
            if elem:
                text = clean_text(await elem.inner_text())
                if text and len(text) > 3:
                    # Remove leading date pattern if present (e.g. "3 ЯНВАРЯ 14:00 ")
                    text = re.sub(r'^\d{1,2}\s+[А-ЯЁа-яё]+\s+\d{1,2}:\d{2}\s*', '', text)
                    if text and len(text) > 3:
                        return text, selector
        except Exception:
            continue
    
    # Fallback to <title> tag
    title_tag = soup.find('title')
    if title_tag:
        text = clean_text(title_tag.get_text().split('|')[0].split('—')[0])
        return text, 'title_tag'
    
    return "", "not_found"


def extract_prices_resilient(content: str, body_text: str) -> tuple[Optional[int], Optional[int], str]:
    """
    Extract prices using multiple fallback patterns.
    Returns (min_price, max_price, pattern_that_worked).
    """
    prices = []
    working_pattern = ""
    
    for pattern in PRICE_PATTERNS:
        try:
            matches = re.findall(pattern, content + ' ' + body_text, re.IGNORECASE)
            for m in matches:
                try:
                    val = int(re.sub(r'\s', '', m))
                    if 100 <= val <= 50000:  # Reasonable price range
                        prices.append(val)
                        if not working_pattern:
                            working_pattern = pattern
                except ValueError:
                    pass
        except Exception:
            continue
    
    if prices:
        return min(prices), max(prices), working_pattern
    return None, None, "not_found"


async def parse_special_project_page(page, url: str) -> list[ParsedEvent]:
    """Parse a special project page and extract all events with resilient selectors."""
    print(f"\n🏛 Загрузка: {url}")
    
    selectors_metadata = {
        "images": [],
        "description": [],
        "title": "",
        "price": "",
    }
    
    try:
        # Navigate with extended timeout
        await page.goto(url, timeout=60000, wait_until='networkidle')
        
        # Wait for content to load
        await page.wait_for_timeout(3000)
        
        # Scroll to load lazy images
        for i in range(5):
            await page.mouse.wheel(0, 500)
            await page.wait_for_timeout(500)
        
        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")
        await page.wait_for_timeout(1000)
        
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # --- НАЗВАНИЕ (с fallback) ---
        title, title_selector = await extract_title_resilient(page, soup)
        selectors_metadata["title"] = title_selector
        print(f"   📌 Название: {title} (selector: {title_selector})")
        
        # --- ИЗОБРАЖЕНИЯ (с fallback) ---
        photos, img_selectors = await extract_images_resilient(page, content)
        selectors_metadata["images"] = img_selectors
        print(f"   🖼 Фото: {len(photos)} (selectors: {', '.join(img_selectors[:3])})")
        
        # --- ОПИСАНИЕ (с fallback) ---
        description, desc_selectors = await extract_description_resilient(page)
        selectors_metadata["description"] = desc_selectors
        print(f"   📝 Описание: {len(description)} символов (selectors: {', '.join(desc_selectors[:2])})")
        
        # --- ВОЗРАСТНОЕ ОГРАНИЧЕНИЕ ---
        age_restriction = ""
        age_match = re.search(r'(\d+)\s*\+', content)
        if age_match:
            age_restriction = f"{age_match.group(1)}+"
        print(f"   👶 Возраст: {age_restriction or 'не указан'}")
        
        # --- КАРТОЧКИ СОБЫТИЙ ---
        events: list[ParsedEvent] = []
        
        # Multiple date patterns for resilience
        date_patterns = [
            re.compile(r'(\d{1,2})\s+(ЯНВАРЯ|ФЕВРАЛЯ|МАРТА|АПРЕЛЯ|МАЯ|ИЮНЯ|ИЮЛЯ|АВГУСТА|СЕНТЯБРЯ|ОКТЯБРЯ|НОЯБРЯ|ДЕКАБРЯ)\s+(\d{1,2}:\d{2})', re.IGNORECASE),
            re.compile(r'(\d{1,2})\s+(янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек)[а-я]*\s+(\d{1,2}:\d{2})', re.IGNORECASE),
            re.compile(r'(\d{2})\.(\d{2})\s+в?\s*(\d{1,2}:\d{2})', re.IGNORECASE),  # DD.MM HH:MM
        ]
        
        # Multiple link patterns for resilience
        link_patterns = [
            re.compile(r'unifd-performance-id=(\d+)'),
            re.compile(r'event[_-]?id=(\d+)'),
            re.compile(r'/tickets?/(\d+)'),
        ]
        
        # Find ticket links
        ticket_links = []
        for pattern in link_patterns:
            links = soup.find_all('a', href=pattern)
            ticket_links.extend(links)
        
        print(f"   🎫 Найдено ссылок на билеты: {len(ticket_links)}")
        
        seen_dates = set()
        
        for link in ticket_links:
            href = link.get('href', '')
            
            # Try to extract performance ID
            perf_id = None
            for pattern in link_patterns:
                perf_match = pattern.search(href)
                if perf_match:
                    perf_id = perf_match.group(1)
                    break
            
            if not perf_id:
                continue
            
            ticket_url = f"{BASE_DOMAIN}/?unifd-performance-id={perf_id}"
            
            # Find date using multiple patterns
            date_raw = ""
            parent = link.parent
            for _ in range(7):
                if parent is None:
                    break
                parent_text = parent.get_text(" ", strip=True)
                
                for date_pattern in date_patterns:
                    date_match = date_pattern.search(parent_text)
                    if date_match:
                        groups = date_match.groups()
                        if len(groups) == 3:
                            date_raw = f"{groups[0]} {groups[1]} {groups[2]}"
                        break
                
                if date_raw:
                    break
                parent = parent.parent
            
            # Deduplicate
            dedup_key = (date_raw, perf_id)
            if dedup_key in seen_dates:
                continue
            seen_dates.add(dedup_key)
            
            parsed_date, parsed_time = parse_date_time(date_raw)
            
            print(f"   📅 {date_raw} -> {parsed_date} {parsed_time} (ID: {perf_id})")
            
            event = ParsedEvent(
                title=title,
                date_raw=date_raw,
                parsed_date=parsed_date,
                parsed_time=parsed_time,
                location="Дом искусств",
                url=ticket_url,
                ticket_status="unknown",
                ticket_price_min=None,
                ticket_price_max=None,
                photos=photos.copy(),
                description=description,
                age_restriction=age_restriction,
                source_type="dom_iskusstv",
                selectors_used=selectors_metadata.copy()
            )
            events.append(event)
        
        print(f"\n✅ Найдено событий: {len(events)}")
        return events
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return []


async def check_ticket_availability(page, ticket_url: str) -> tuple[str, Optional[int], Optional[int]]:
    """Check ticket availability and price with resilient patterns."""
    print(f"   🔍 Проверка: {ticket_url[:60]}...")
    
    try:
        await page.goto(ticket_url, timeout=45000, wait_until='networkidle')
        await page.wait_for_timeout(5000)
        
        # Wait for any widget
        try:
            await page.wait_for_selector('iframe, [class*="widget"], [class*="ticket"], [id*="ticket"]', timeout=5000)
        except Exception:
            pass
        
        # Collect all content including iframes
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
        
        # Detect status with multiple patterns
        status = "unknown"
        sold_out_patterns = ["билетов нет", "sold out", "распродано", "нет в продаже", "закончились", "недоступно"]
        available_patterns = ["купить", "в корзину", "выбрать", "добавить", "оформить", "заказать", "приобрести"]
        
        if any(sold in body_lower for sold in sold_out_patterns):
            status = "sold_out"
        elif any(avail in body_lower for avail in available_patterns):
            status = "available"
        
        # Extract prices with resilient patterns
        price_min, price_max, _ = extract_prices_resilient(all_content, body_text)
        
        if price_min and status == "unknown":
            status = "available"
        
        print(f"      Status: {status}, Price: {price_min}-{price_max} ₽")
        return status, price_min, price_max
        
    except Exception as e:
        print(f"      ⚠️ Ошибка: {e}")
        return "unknown", None, None


async def main():
    # Parse command line
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://xn--b1admiilxbaki.xn--p1ai/skazka"
    
    print(f"🏛 Парсинг Дом искусств: {url}")
    print("🛡 Режим: с защитой от изменения селекторов")
    print("=" * 60)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        ticket_page = await context.new_page()
        
        # Parse page
        events = await parse_special_project_page(page, url)
        
        # Check tickets for each event
        for event in events:
            if "unifd-performance-id" in event.url:
                status, price_min, price_max = await check_ticket_availability(ticket_page, event.url)
                event.ticket_status = status
                event.ticket_price_min = price_min
                event.ticket_price_max = price_max
        
        await browser.close()
    
    # Output
    results = [asdict(e) for e in events]
    
    print("\n" + "=" * 60)
    print(f"🎉 Результат: {len(results)} событий")
    print("=" * 60)
    
    for i, event in enumerate(results, 1):
        print(f"\n{i}. {event['title']}")
        print(f"   📅 {event['date_raw']} -> {event['parsed_date']} {event['parsed_time']}")
        print(f"   📍 {event['location']}")
        print(f"   🎫 {event['ticket_status']} | {event['ticket_price_min']}-{event['ticket_price_max']} ₽")
        print(f"   👶 Возраст: {event['age_restriction']}")
        print(f"   🖼 Фото: {len(event['photos'])} шт.")
        print(f"   📝 Описание: {len(event['description'])} символов")
        if event.get('selectors_used'):
            sel = event['selectors_used']
            print(f"   🛡 Селекторы: title={sel.get('title')}, images={len(sel.get('images', []))}, desc={len(sel.get('description', []))}")
    
    # Save JSON
    output_file = "dom_iskusstv_events.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Сохранено в {output_file}")
    
    # Save selector diagnostics
    if results and results[0].get('selectors_used'):
        diag_file = "dom_iskusstv_selectors.json"
        with open(diag_file, 'w', encoding='utf-8') as f:
            json.dump(results[0]['selectors_used'], f, ensure_ascii=False, indent=2)
        print(f"🛡 Диагностика селекторов: {diag_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
